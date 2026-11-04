# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequence parallelism on real process groups, composed under FSDP2.

Tensor parallelism shards weights and leaves the norm, residual and modulation
region replicated; on these architectures that region holds most of the
activation memory. Sequence parallelism shards it along the token axis and
gathers back only around attention, which costs nothing extra because an
all-reduce *is* a reduce-scatter followed by an all-gather.

Two of its failure modes produce a **bit-identical loss** and are therefore
invisible to any check that does not compare gradients parameter by parameter:

* a bare parameter broadcast over tokens -- a modulation table -- accumulates
  only this rank's share of its gradient, measured at exactly 1/2 on two ranks;
* so does everything upstream of a *conditioning activation* that broadcasts
  over tokens, because the value itself was never sharded and no placement
  records the debt. Wan's time embedder came out at 27% of its true gradient.

Both have a style, and the tests below exist to notice if either stops working.
The model here is shaped like Wan's block for that reason: a bare
``scale_shift_table``, a conditioning stack feeding it, and the full-width fp32
modulation arithmetic that sequence parallelism is for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from conftest import requires_gloo, run_gloo_ranks
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

HEADS, HEAD_DIM = 4, 16
DIM = HEADS * HEAD_DIM
FFN = 2 * DIM
BLOCKS = 2
SEQ = 32
BATCH = 2


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = HEADS
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_k = nn.Linear(DIM, DIM, bias=False)
        self.to_v = nn.Linear(DIM, DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        shape = (batch, length, self.heads, -1)
        query = self.to_q(hidden_states).view(shape).transpose(1, 2)
        key = self.to_k(hidden_states).view(shape).transpose(1, 2)
        value = self.to_v(hidden_states).view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.to_out(attended.transpose(1, 2).reshape(batch, length, -1))


class _FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(DIM, FFN, bias=False)
        self.down = nn.Linear(FFN, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(hidden_states)))


class _Block(nn.Module):
    """Wan's shape: a bare modulation table around each sublayer."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 4, DIM) / DIM**0.5)
        self.norm1 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.attn = _Attention()
        self.norm2 = nn.LayerNorm(DIM)
        self.ffn = _FeedForward()

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale, ff_shift, ff_scale = (self.scale_shift_table + temb.float()).chunk(4, dim=1)
        normed = (self.norm1(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = (hidden_states.float() + self.attn(normed).float()).type_as(hidden_states)
        normed = (self.norm2(hidden_states.float()) * (1 + ff_scale) + ff_shift).type_as(
            hidden_states
        )
        return (hidden_states.float() + self.ffn(normed).float()).type_as(hidden_states)


class _Conditioning(nn.Module):
    """Produces one vector per batch element, broadcast over every token."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, 4 * DIM)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.proj(timestep).unflatten(-1, (4, DIM))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition = _Conditioning()
        self.blocks = nn.ModuleList(_Block() for _ in range(BLOCKS))
        self.proj_out = nn.Linear(DIM, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        temb = self.condition(timestep)
        for block in self.blocks:
            hidden_states = block(hidden_states, temb)
        return self.proj_out(hidden_states)


def _tensor_parallel_plan() -> Dict[str, Any]:
    return {
        "blocks.*.attn.to_q": ColwiseParallel(),
        "blocks.*.attn.to_k": ColwiseParallel(),
        "blocks.*.attn.to_v": ColwiseParallel(),
        "blocks.*.attn.to_out": RowwiseParallel(),
        "blocks.*.ffn.up": ColwiseParallel(),
        "blocks.*.ffn.down": RowwiseParallel(),
    }


def _sequence_parallel_plan(*, omit: str = "", passthrough: bool = False) -> Dict[str, Any]:
    """The overlay, with one entry optionally dropped to prove it is load-bearing.

    ``passthrough`` adds the shape H3 writes: a wildcard ``PrepareModuleInput``
    declaring the layout its hidden state *already* arrives in, so the rest of
    the positional signature can be described, with only ``blocks.0`` doing the
    real ``Replicate() -> Shard(1)``.
    """
    from flow_factory.models.tensor_parallel import (
        ComposedParallelStyle,
        SequenceBroadcastOutput,
        SequenceBroadcastParameter,
    )

    gather = PrepareModuleInput(
        input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),), use_local_output=True
    )
    plan: Dict[str, Any] = {
        "blocks.*": SequenceBroadcastParameter(),
        "blocks.0": ComposedParallelStyle(
            SequenceBroadcastParameter(),
            PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Shard(1), None),
                use_local_output=True,
            ),
        ),
        # `norm1` has no affine parameter, so a token slice needs no entry at
        # all; normalizing over the last axis is already correct.
        "blocks.*.norm2": SequenceBroadcastParameter(),
        "blocks.*.attn": gather,
        "blocks.*.ffn": gather,
        "blocks.*.attn.to_out": RowwiseParallel(output_layouts=Shard(1)),
        "blocks.*.ffn.down": RowwiseParallel(output_layouts=Shard(1)),
        "condition": SequenceBroadcastOutput(),
        "proj_out": ComposedParallelStyle(
            SequenceBroadcastParameter(),
            PrepareModuleOutput(
                output_layouts=Shard(1),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            ),
        ),
    }
    if passthrough:
        plan["blocks.*"] = ComposedParallelStyle(
            SequenceBroadcastParameter(),
            PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Shard(1), None),
                use_local_output=True,
            ),
        )
    if omit == "broadcast_parameter":
        # The block keeps a parameter style so the plan stays total, but one
        # that does not know its gradient is a partial sum.
        from flow_factory.models.tensor_parallel import ReplicateParallel

        plan["blocks.*"] = ReplicateParallel()
        plan["blocks.0"] = ComposedParallelStyle(ReplicateParallel(), plan["blocks.0"].styles[1])
    elif omit == "broadcast_output":
        del plan["condition"]
    return plan


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    from flow_factory.models.tensor_parallel import apply_tensor_parallel

    dp_size, tp_size = options["dp_size"], options["tp_size"]
    world_size = dp_size * tp_size
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    result: Dict[str, Any] = {"rank": rank}
    try:
        mesh = init_device_mesh("cpu", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp"))
        torch.manual_seed(1234)
        model = _Model()
        if tp_size > 1:
            plan = _tensor_parallel_plan()
            if options["sequence_parallel"]:
                plan = {
                    **plan,
                    **_sequence_parallel_plan(
                        omit=options["omit"], passthrough=options["passthrough"]
                    ),
                }
            apply_tensor_parallel(model, mesh["tp"], plan)
        if dp_size > 1:
            fully_shard(model, mesh=mesh["dp_shard"])

        if options.get("probe_mesh"):
            # Read the same way a style reading the live parameter would, and
            # measure which group a one-placement Partial actually reduces over.
            table = dict(model.named_parameters())["blocks.0.scale_shift_table"]
            contribution = torch.full((2, 2), float(rank))
            result["mesh_probe"] = {
                "mesh_after_fsdp": list(table.device_mesh.mesh_dim_names or ()),
                "one_placement_sum": float(
                    DTensor.from_local(
                        contribution, table.device_mesh, [Partial()], run_check=False
                    ).full_tensor()[0, 0]
                ),
                "two_placement_sum": float(
                    DTensor.from_local(
                        contribution,
                        table.device_mesh,
                        [Replicate(), Partial()],
                        run_check=False,
                    ).full_tensor()[0, 0]
                ),
            }

        # Keyed by the data-parallel coordinate, not by a constant: distinct
        # replicas must see distinct samples, as a real run gives them. With one
        # shared seed the partial-gradient reduction can run over the *wrong*
        # mesh dimension and still agree, because summing two identical token
        # slices from data-parallel peers is indistinguishable from summing the
        # two different slices the tensor-parallel peers hold.
        dp_rank = mesh["dp_shard"].get_local_rank() if dp_size > 1 else 0
        generator = torch.Generator().manual_seed(7 + 100 * dp_rank)
        length = options.get("sequence_length") or SEQ
        hidden = torch.randn(BATCH, length, DIM, generator=generator)
        timestep = torch.randn(BATCH, DIM, generator=generator)
        output = model(hidden, timestep)
        result["output_shape"] = list(output.shape)
        loss = output.float().pow(2).mean()
        result["loss"] = round(float(loss.detach()), 8)
        loss.backward()
        result["grad_names"] = sorted(n for n, p in model.named_parameters() if p.grad is not None)
        result["grad_placements"] = {
            name: (
                f"{tuple(p.grad.device_mesh.mesh_dim_names or ())}:"
                f"{[str(x) for x in p.grad.placements]}"
            )
            for name, p in model.named_parameters()
            if isinstance(p.grad, DTensor)
        }
        result["local_grads"] = {
            name: round(float(p.grad.to_local().double().sum()), 7)
            for name, p in model.named_parameters()
            if isinstance(p.grad, DTensor)
        }
        result["grads"] = {
            name: round(
                float(
                    (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).double().sum()
                ),
                7,
            )
            for name, p in model.named_parameters()
            if p.grad is not None
        }
    except Exception as error:
        import traceback

        # Wide enough to keep a guard message whole; the assertions read it.
        result["error"] = f"{type(error).__name__}: {str(error)[:600]}"
        result["traceback"] = traceback.format_exc()[-1500:]
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run(tmp_path: Path, **options: Any) -> List[Dict[str, Any]]:
    """Spawn the ranks this file's worker needs; the harness is shared."""
    options.setdefault("dp_size", 1)
    options.setdefault("sequence_parallel", False)
    options.setdefault("omit", "")
    options.setdefault("sequence_length", None)
    options.setdefault("probe_mesh", False)
    options.setdefault("passthrough", False)
    return run_gloo_ranks(
        _worker, tmp_path, options["dp_size"] * options["tp_size"], options, timeout=300
    )


pytestmark = requires_gloo


@pytest.mark.parametrize("tp_size", [2, 4], ids=["tp2", "tp4"])
def test_sharding_the_token_axis_changes_no_gradient(tmp_path: Path, tp_size: int) -> None:
    """Sequence parallelism is numerically transparent, like tensor parallelism.

    Compared against the unsharded model rather than against tensor parallelism
    alone, so a style that happens to agree with the other sharded run but not
    with the truth is still caught.
    """
    reference = _run(tmp_path / "ref", tp_size=1)
    sharded = _run(tmp_path / f"sp{tp_size}", tp_size=tp_size, sequence_parallel=True)

    for result in reference + sharded:
        assert "error" not in result, result["error"]
    assert sharded[0]["output_shape"] == reference[0]["output_shape"], "sequence not restored"
    assert sharded[0]["grad_names"] == reference[0]["grad_names"]
    assert sharded[0]["loss"] == pytest.approx(reference[0]["loss"], rel=1e-6, abs=1e-8)
    for name, expected in reference[0]["grads"].items():
        assert sharded[0]["grads"][name] == pytest.approx(
            expected, rel=1e-4, abs=1e-6
        ), f"{name}: sequence parallelism changed this gradient"


def test_sequence_parallelism_composes_with_fsdp(tmp_path: Path) -> None:
    """The token axis and the data-parallel axis are independent.

    ``fully_shard`` runs over a model whose activations are already token-sharded
    and whose parameters already carry tensor-parallel placements, so this is the
    configuration a real run uses and the one neither axis alone exercises.

    It is also the configuration that catches a partial gradient declared on the
    wrong mesh dimension. ``fully_shard`` moves every parameter onto the
    concatenated ``(dp_shard, tp)`` mesh, and ``DTensor.from_local`` accepts a
    one-element placement list there without complaint -- putting the ``Partial``
    on ``dp_shard``, so the missing token-slice sum is reduced over replicas
    whose gradients FSDP has already reduced, and never over the tensor-parallel
    peers holding the missing terms. Distinct data per replica is what makes the
    two distinguishable; see ``_worker``.
    """
    reference = _run(tmp_path / "ref", dp_size=2, tp_size=1)
    composed = _run(tmp_path / "composed", dp_size=2, tp_size=2, sequence_parallel=True)

    for result in reference + composed:
        assert "error" not in result, result["error"]
    assert composed[0]["grad_names"] == reference[0]["grad_names"]
    for name, expected in reference[0]["grads"].items():
        assert composed[0]["grads"][name] == pytest.approx(expected, rel=1e-4, abs=1e-6), name


@pytest.mark.parametrize(
    "omit,victim",
    [
        ("broadcast_parameter", "blocks.0.scale_shift_table"),
        ("broadcast_output", "condition.proj.weight"),
    ],
)
def test_a_missing_broadcast_style_is_wrong_and_the_loss_does_not_say_so(
    tmp_path: Path, omit: str, victim: str
) -> None:
    """Pin both silent failures, and that they are silent.

    Dropping either style leaves a forward that is correct on every rank -- so
    the loss matches to the last bit -- while the affected gradient comes back
    as this rank's share of the tokens. A test that compared losses, or even
    the gradient norm, would pass.
    """
    reference = _run(tmp_path / "ref", tp_size=1)
    broken = _run(tmp_path / "broken", tp_size=2, sequence_parallel=True, omit=omit)

    for result in reference + broken:
        assert "error" not in result, result["error"]
    assert broken[0]["loss"] == pytest.approx(
        reference[0]["loss"], rel=1e-6, abs=1e-8
    ), "the loss is supposed to stay correct -- that is what makes this silent"
    assert broken[0]["grads"][victim] != pytest.approx(
        reference[0]["grads"][victim], rel=1e-3, abs=1e-7
    ), f"{victim} was expected to lose its reduction without the style"


def _trainer_without_a_sequence_parallel_plan(monkeypatch: pytest.MonkeyPatch, requested):
    """A trainer whose adapter shards weights but declares no token-axis plan."""
    from types import SimpleNamespace

    from flow_factory.hparams.parallelism_args import ParallelismArguments
    from flow_factory.trainers import abc as trainer_abc
    from flow_factory.trainers.abc import BaseTrainer

    trainer = object.__new__(BaseTrainer)
    trainer.accelerator = SimpleNamespace(torch_device_mesh={"tp": object()})
    trainer.config = SimpleNamespace(
        parallelism_args=ParallelismArguments(tp_size=2, sequence_parallel=requested)
    )
    from flow_factory.models.variants import DEFAULT_BASE_VARIANT

    module = nn.Linear(4, 4)
    trainer.adapter = SimpleNamespace(
        tensor_parallel_plan=lambda: {"weight": object()},
        sequence_parallel_plan=lambda: {},
        trainable_component_names=["transformer"],
        component_variant_registry=SimpleNamespace(
            variant_names=(DEFAULT_BASE_VARIANT,),
            bundle_members=lambda: {"transformer": module},
            resolve_route=lambda _variant, _component: "transformer",
            rebind_parameters=lambda _members: None,
        ),
    )
    # The plan is applied for real here; only the fallback decision is under test.
    monkeypatch.setattr(trainer_abc, "apply_tensor_parallel", lambda *_args: {})
    # `abc` imported the name, so the module attribute is where it is read.
    monkeypatch.setattr(trainer_abc, "tensor_parallel_size", lambda _accelerator: 2)
    return trainer


def test_requesting_sequence_parallelism_by_name_refuses_an_adapter_without_a_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sequence_parallel: true`` asked for something this architecture cannot give.

    The plan has to say which norms live in the sharded region, where attention
    gathers back and which parameters broadcast over tokens, so there is no safe
    default -- and falling back silently would hand back the memory the flag was
    set to save, with no indication why.
    """
    trainer = _trainer_without_a_sequence_parallel_plan(monkeypatch, True)

    with pytest.raises(ValueError, match="declares no sequence-parallel plan"):
        trainer._apply_tensor_parallel()


def test_the_default_falls_back_to_tensor_parallelism_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Unset means "use it where there is a plan", not "require one".

    Sequence parallelism is on by default because the two together are the
    configuration worth running, but several architectures have no derivation
    yet and that must not block a tensor-parallel run. It warns, because this is
    where most of the memory saving would have come from.
    """
    import logging

    trainer = _trainer_without_a_sequence_parallel_plan(monkeypatch, None)

    with caplog.at_level(logging.WARNING, logger="flow_factory.trainers.abc"):
        trainer._apply_tensor_parallel()

    assert any("no sequence-parallel plan" in record.message for record in caplog.records)


def test_turning_it_off_explicitly_warns_about_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """``sequence_parallel: false`` is a decision, so it needs no commentary."""
    import logging

    trainer = _trainer_without_a_sequence_parallel_plan(monkeypatch, False)

    with caplog.at_level(logging.WARNING, logger="flow_factory.trainers.abc"):
        trainer._apply_tensor_parallel()

    assert not [r for r in caplog.records if "sequence-parallel" in r.message]


def test_an_indivisible_token_count_is_refused_at_the_scatter(tmp_path: Path) -> None:
    """Turn a silent deadlock into a message naming the cause.

    `PrepareModuleInput` lifts a plain tensor with `DTensor.from_local`, which
    has no global length to read and infers `local * tp_size`. An odd sequence
    makes the ranks infer *different* global lengths, so the next gather posts
    all-gathers of mismatched sizes -- which does not raise. Measured on LTX-2:
    both GPUs at 100%, no output, four minutes until killed.

    The scatter is the one place that still sees the whole sequence, so the
    check costs no collective.
    """
    results = _run(tmp_path, tp_size=2, sequence_parallel=True, sequence_length=SEQ + 1)

    for result in results:
        assert "error" in result, "an indivisible token count must not be accepted"
        assert "divisible by tp_size=2" in result["error"], result["error"]
        assert "deadlocks" in result["error"], "the message must say what it prevents"
        assert "sequence_parallel: false" in result["error"], "and the configuration that works"


def test_an_even_token_count_still_passes_the_guard(tmp_path: Path) -> None:
    """The guard must not reject the ordinary case it exists to protect."""
    results = _run(tmp_path, tp_size=2, sequence_parallel=True, sequence_length=SEQ)

    for result in results:
        assert "error" not in result, result["error"]


def test_a_pass_through_layout_is_not_measured_as_a_scatter(tmp_path: Path) -> None:
    """A style that declares the layout it already receives moves nothing.

    H3 writes ``transformer_blocks.*`` with ``Shard(1)`` on *both* sides so the
    rest of its positional signature can be described, and does the real
    ``Replicate() -> Shard(1)`` on ``transformer_blocks.0`` alone. Counting that
    pass-through puts the guard's hook on blocks 1..N, where the hidden state
    has already been sliced, so it measures ``S / tp`` and demands
    ``(S / tp) % tp == 0`` -- effectively ``tp**2 | S``.

    34 tokens on two ranks is the discriminating case: 34 divides, 17 does not.
    The gradients are compared against the unsharded model as well, because a
    guard that stops rejecting is only right if what it let through is correct.
    """
    length = SEQ + 2
    assert length % 2 == 0 and (length // 2) % 2 == 1, "the case must be divisible but not twice"
    reference = _run(tmp_path / "ref", tp_size=1, sequence_length=length)
    sharded = _run(
        tmp_path / "passthrough",
        tp_size=2,
        sequence_parallel=True,
        sequence_length=length,
        passthrough=True,
    )

    for result in reference + sharded:
        assert "error" not in result, result["error"]
    for name, expected in reference[0]["grads"].items():
        assert sharded[0]["grads"][name] == pytest.approx(
            expected, rel=1e-4, abs=1e-6
        ), f"{name}: the pass-through changed this gradient"


def test_the_guard_still_names_the_real_scatter_behind_a_pass_through(tmp_path: Path) -> None:
    """Narrowing the guard must not blind it: an odd count is still refused.

    And refused at ``blocks.0``, reporting the *global* length -- the earlier
    behaviour reported whichever slice the later hook happened to see.
    """
    results = _run(
        tmp_path, tp_size=2, sequence_parallel=True, sequence_length=SEQ + 1, passthrough=True
    )

    for result in results:
        assert "error" in result, "an indivisible token count must not be accepted"
        assert "'blocks.0'" in result["error"], result["error"]
        assert f"received {SEQ + 1}" in result["error"], result["error"]


def test_a_one_placement_partial_lands_on_the_wrong_mesh_dimension(tmp_path: Path) -> None:
    """Pin why the tensor-parallel submesh is captured when the style is applied.

    ``fully_shard`` replaces every parameter with one on the concatenated
    ``(dp_shard, tp)`` mesh, so a style that reads ``parameter.device_mesh``
    inside its forward hook sees a 2-D mesh. ``DTensor.from_local`` accepts a
    one-element placement list there **without complaint**, and puts the
    ``Partial`` on dimension 0 -- which is ``dp_shard``, not the tensor-parallel
    peers that hold the missing token slices.

    Asserted at the mesh level rather than on gradients, and that is the honest
    scope: measured end to end on this model, FSDP2's own gradient reduction
    consumes the ``Partial`` either way and the gradients come out identical, so
    the wrong declaration is a latent hazard rather than an active error on
    torch 2.10. The style captures the submesh regardless, because correctness
    should not rest on another layer incidentally covering for it.
    """
    results = _run(tmp_path, dp_size=2, tp_size=2, sequence_parallel=True, probe_mesh=True)

    for result in results:
        assert "error" not in result, result["error"]
    probe = results[0]["mesh_probe"]
    assert probe["mesh_after_fsdp"] == [
        "dp_shard",
        "tp",
    ], "a style reading the live parameter's mesh would see this, not the tp submesh"
    # Rank 0 is (dp=0, tp=0). One placement sums its dp peer, rank 2; the
    # correct two-placement form sums its tensor-parallel peer, rank 1.
    assert probe["one_placement_sum"] == 2.0, probe
    assert probe["two_placement_sum"] == 1.0, probe


def test_every_variant_route_of_a_multi_role_component_is_sharded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-role full finetune has one module per role, and all must be sharded.

    ``_materialize_full_variants`` deep-copies the base component once per
    non-base role and registers each under its own bundle route. Sharding only
    the active route -- what ``adapter.get_component(name)`` resolves to --
    leaves the replicas holding plain tensors, and ``ModelBundle`` then hands
    ``prepare`` a root mixing meshes: measured at tp=2, the sharded role lands on
    ``('dp_shard', 'tp')`` while the replica lands on ``('dp_shard',)`` and the
    first ``clip_grad_norm_`` raises "DTensor does not support cross-mesh
    operation" -- after the weights are already loaded.

    LoRA roles are the other case and need the identity de-duplication rather
    than a second pass: they are extra PEFT adapters on *one* module, which
    ``derive_lora_plan`` already reaches, and sharding it twice would halve the
    attention head count twice.
    """
    from types import SimpleNamespace

    from flow_factory.models.variants import DEFAULT_BASE_VARIANT, ComponentVariantSpec
    from flow_factory.trainers import abc as trainer_abc
    from flow_factory.trainers.abc import BaseTrainer

    def spec(name: str, route: str) -> ComponentVariantSpec:
        return ComponentVariantSpec(
            name=name,
            storage_mode="full",
            component_routes={"transformer": route},
            adapter_name=name,
        )

    shared = nn.Linear(4, 4)
    replica = nn.Linear(4, 4)
    registry = SimpleNamespace(
        variant_names=(DEFAULT_BASE_VARIANT, "fake"),
        bundle_members=lambda: {"transformer": shared, "transformer_fake": replica},
        resolve_route=lambda variant, component: (
            "transformer" if variant == DEFAULT_BASE_VARIANT else "transformer_fake"
        ),
        rebind_parameters=lambda _members: None,
    )

    sharded: list[nn.Module] = []
    monkeypatch.setattr(trainer_abc, "tensor_parallel_size", lambda _accelerator: 2)
    monkeypatch.setattr(
        trainer_abc,
        "apply_tensor_parallel",
        lambda module, _mesh, _plan: sharded.append(module) or {},
    )

    trainer = object.__new__(BaseTrainer)
    trainer.accelerator = SimpleNamespace(torch_device_mesh={"tp": object()})
    trainer.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            sequence_parallel_is_wanted=False,
            sequence_parallel_is_required=False,
            verify_replicated_inputs=False,
        )
    )
    trainer.adapter = SimpleNamespace(
        tensor_parallel_plan=lambda: {"weight": object()},
        sequence_parallel_plan=lambda: {},
        trainable_component_names=["transformer"],
        component_variant_registry=registry,
    )

    trainer._apply_tensor_parallel()

    assert [id(m) for m in sharded] == [
        id(shared),
        id(replica),
    ], "both the active route and the deep-copied replica must be sharded"
    # And the same module reached twice is sharded once: localizing the head
    # count a second time would halve it again.
    assert len(sharded) == len(set(id(m) for m in sharded))
