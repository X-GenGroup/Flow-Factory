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

"""One input-gradient collective per stream, not one per projection.

torch's ``ColwiseParallel`` all-reduces its input gradient itself, so q, k and v
reading one activation cost three all-reduces where Megatron's region entry
costs one -- and under sequence parallelism three full-sequence all-reduces
where one reduce-scatter suffices. These tests pin both halves of the change:
the collective count, and that every parameter's gradient still matches the
unsharded model, with and without LoRA. The model reads its attention by
keyword and its feed-forward positionally, the two conventions diffusers uses.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from conftest import requires_gloo, run_gloo_ranks
from torch import nn

WORLD_SIZE = 2
DIM = 16
HEAD_DIM = 4
SEQ = 8
TEXT = 4
BATCH = 2


class _JointAttention(nn.Module):
    """Two streams sharing one softmax, as SD3.5 and Qwen-Image attend."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = DIM // HEAD_DIM
        self.to_q = nn.Linear(DIM, DIM)
        self.to_k = nn.Linear(DIM, DIM)
        self.to_v = nn.Linear(DIM, DIM)
        self.add_q_proj = nn.Linear(DIM, DIM)
        self.add_k_proj = nn.Linear(DIM, DIM)
        self.add_v_proj = nn.Linear(DIM, DIM)
        self.to_out = nn.Linear(DIM, DIM)
        self.to_add_out = nn.Linear(DIM, DIM)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2)

        query = torch.cat(
            [heads(self.add_q_proj(encoder_hidden_states)), heads(self.to_q(hidden_states))], 2
        )
        key = torch.cat(
            [heads(self.add_k_proj(encoder_hidden_states)), heads(self.to_k(hidden_states))], 2
        )
        value = torch.cat(
            [heads(self.add_v_proj(encoder_hidden_states)), heads(self.to_v(hidden_states))], 2
        )
        attended = F.scaled_dot_product_attention(query, key, value).transpose(1, 2).flatten(-2)
        text = encoder_hidden_states.shape[1]
        return self.to_out(attended[:, text:]), self.to_add_out(attended[:, :text])


class _FeedForward(nn.Module):
    """SwiGLU, whose gate and up projections read one input."""

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(DIM, 2 * DIM)
        self.w3 = nn.Linear(DIM, 2 * DIM)
        self.w2 = nn.Linear(2 * DIM, DIM)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(DIM, DIM)
        self.text_embed = nn.Linear(DIM, DIM)
        self.attn = _JointAttention()
        self.ff = _FeedForward()
        self.head = nn.Linear(DIM, DIM)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed(hidden_states)
        text = self.text_embed(encoder_hidden_states)
        attended, text_attended = self.attn(hidden_states=hidden, encoder_hidden_states=text)
        hidden = hidden + attended
        hidden = hidden + self.ff(hidden)
        return self.head(hidden), text + text_attended


_PROJECTIONS = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")
_LORA_TARGETS = [*_PROJECTIONS, "to_out", "to_add_out", "w1", "w3", "w2"]


def _build(lora: bool) -> nn.Module:
    torch.manual_seed(0)
    model = _Model().double()
    if not lora:
        return model
    from peft import LoraConfig, get_peft_model

    # Random `lora_B` as well as `lora_A`: PEFT's zero `lora_B` would leave
    # every `lora_A` gradient zero and hide a wrong reduction there.
    config = LoraConfig(r=4, lora_alpha=4, target_modules=_LORA_TARGETS, init_lora_weights=False)
    return get_peft_model(model, config).double()


def _plan(layout: str) -> Dict[str, Any]:
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        PrepareModuleInput,
        PrepareModuleOutput,
        RowwiseParallel,
    )

    from flow_factory.models.tensor_parallel import (
        ColumnParallelRegion,
        SequenceParallelRegion,
        SharedInputColwiseParallel,
    )

    shared = layout.endswith("shared")
    colwise = SharedInputColwiseParallel if shared else ColwiseParallel
    plan: Dict[str, Any] = {f"attn.{name}": colwise() for name in _PROJECTIONS}
    plan.update(
        {
            "attn.to_out": RowwiseParallel(),
            "attn.to_add_out": RowwiseParallel(),
            "ff.w1": colwise(),
            "ff.w3": colwise(),
            "ff.w2": RowwiseParallel(),
        }
    )
    if layout == "tp_shared":
        plan["attn"] = ColumnParallelRegion(inputs=("hidden_states", "encoder_hidden_states"))
        plan["ff"] = ColumnParallelRegion(inputs=("hidden_states",))
    if layout.startswith("sp"):
        plan.update(
            {
                "embed": PrepareModuleOutput(
                    output_layouts=Replicate(), desired_output_layouts=Shard(1)
                ),
                "attn.to_out": RowwiseParallel(output_layouts=Shard(1)),
                "ff.w2": RowwiseParallel(output_layouts=Shard(1)),
                "head": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_output=True,
                ),
            }
        )
        if shared:
            plan["attn"] = SequenceParallelRegion(
                gather=("hidden_states",), replicated=("encoder_hidden_states",)
            )
            plan["ff"] = SequenceParallelRegion(gather=("hidden_states",))
        else:
            plan["attn"] = PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
                use_local_output=True,
            )
            plan["ff"] = PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_output=True,
            )
    return plan


def _inputs() -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(BATCH, SEQ, DIM, generator=generator, dtype=torch.float64)
    text = torch.randn(BATCH, TEXT, DIM, generator=generator, dtype=torch.float64)
    return hidden, text


def _run(model: nn.Module, count: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.debug import CommDebugMode

    hidden, text = _inputs()
    output, text_output = model(hidden, text)
    loss = output.pow(2).sum() + text_output.pow(2).sum()
    counts: Dict[str, int] = {}
    if count:
        with CommDebugMode() as mode:
            loss.backward()
        counts = dict(
            Counter({str(op).split(".")[-1]: n for op, n in mode.get_comm_counts().items() if n})
        )
    else:
        loss.backward()
    values = {"output": output.detach(), "text_output": text_output.detach()}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad
        values[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach()
    return values, counts


def _worst_relative(expected: Dict[str, torch.Tensor], received: Dict[str, torch.Tensor]):
    assert set(expected) == set(received), sorted(set(expected) ^ set(received))
    relative = {
        name: ((received[name] - expected[name]).abs().max() / expected[name].abs().max()).item()
        for name in expected
        if expected[name].abs().max() > 0
    }
    worst = max(relative, key=relative.get)
    return relative[worst], worst


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.tensor_parallel import apply_tensor_parallel, peel_peft_wrapper

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        for lora in (False, True):
            reference = _build(lora)
            expected, _ = _run(copy.deepcopy(reference), count=False)
            for layout in ("tp_torch", "tp_shared", "sp_torch", "sp_shared"):
                sharded = copy.deepcopy(reference)
                apply_tensor_parallel(peel_peft_wrapper(sharded), mesh, _plan(layout))
                received, counts = _run(sharded, count=True)
                worst, name = _worst_relative(expected, received)
                result[f"{layout}/lora={lora}"] = {"worst": worst, "name": name, "counts": counts}
                if layout == "tp_shared" and not lora:
                    # The first-call input check removes itself; the local-view
                    # shadow pair is all that stays on the projection.
                    projection = peel_peft_wrapper(sharded).attn.to_q
                    result["hooks_left"] = len(projection._forward_pre_hooks)
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_one_collective_per_stream_and_every_gradient_matches(tmp_path: Path) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})
    for result in results:
        assert "error" not in result, result["error"]
        for key, entry in result.items():
            if "/" in key:
                assert entry["worst"] < 1e-12, (key, entry)
        # Backward collectives. The input-gradient ones are the point: one per
        # stream that needs a gradient -- the attention's two and the
        # feed-forward's one -- where torch's style paid one per projection.
        assert result["tp_torch/lora=False"]["counts"] == {"all_reduce": 8}
        assert result["tp_shared/lora=False"]["counts"] == {"all_reduce": 3}
        assert result["sp_torch/lora=False"]["counts"] == {
            "all_reduce": 8,
            "all_gather_into_tensor": 3,
        }
        # Gathered streams reduce-scatter, the replicated text stream
        # all-reduces; the output projections' all-gathers are unchanged.
        assert result["sp_shared/lora=False"]["counts"] == {
            "reduce_scatter_tensor": 2,
            "all_reduce": 1,
            "all_gather_into_tensor": 3,
        }
        # Under LoRA the embeddings are frozen, so only the feed-forward's
        # input needs a gradient, and `lora_B` no longer reduces its own input:
        # its `lora_A` weight gradient is reduced with the rest under FSDP2.
        assert result["tp_torch/lora=True"]["counts"] == {"all_reduce": 10}
        assert result["tp_shared/lora=True"]["counts"] == {"all_reduce": 1}
        assert result["sp_shared/lora=True"]["counts"] == {
            "reduce_scatter_tensor": 1,
            "all_gather_into_tensor": 4,
        }
        assert result["hooks_left"] == 1, result["hooks_left"]


def _misdeclared_plan() -> Dict[str, Any]:
    """A region that lifts the image stream but not the text stream ``add_*`` read."""
    from flow_factory.models.tensor_parallel import ColumnParallelRegion

    plan = _plan("tp_shared")
    plan["attn"] = ColumnParallelRegion(inputs=("hidden_states",))
    return plan


def _misdeclared_worker(
    rank: int, store_path: str, result_dir: str, options: Dict[str, Any]
) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models import tensor_parallel
        from flow_factory.models.tensor_parallel import apply_tensor_parallel

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        reference = _build(lora=False)
        expected, _ = _run(copy.deepcopy(reference), count=False)

        sharded = copy.deepcopy(reference)
        apply_tensor_parallel(sharded, mesh, _misdeclared_plan())
        try:
            _run(sharded, count=False)
            result["guard"] = "did not raise"
        except RuntimeError as error:
            result["guard"] = str(error)

        # The control: with the check bypassed the same plan trains, and the
        # text stream's upstream gradient comes out at this rank's share.
        tensor_parallel._passed_region_entry = lambda value, depth=4: True
        sharded = copy.deepcopy(reference)
        apply_tensor_parallel(sharded, mesh, _misdeclared_plan())
        received, _ = _run(sharded, count=False)
        result["bypassed"] = _worst_relative(expected, received)
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_a_projection_reading_an_input_its_entry_does_not_lift_is_refused(
    tmp_path: Path,
) -> None:
    results = run_gloo_ranks(_misdeclared_worker, tmp_path, WORLD_SIZE, {})
    for result in results:
        assert "error" not in result, result["error"]
        assert "attn.add_q_proj" in result["guard"], result["guard"]
        assert "does not lift the argument" in result["guard"], result["guard"]
        # Orders of magnitude above the parity tolerance, and in the text
        # stream's upstream weight; the magnitude itself depends on how much of
        # that gradient flows through attention rather than the residual.
        worst, name = result["bypassed"]
        assert worst > 1e-3 and name.startswith("text_embed"), result["bypassed"]


def test_a_shared_input_projection_needs_an_enclosing_region() -> None:
    from flow_factory.models.tensor_parallel import _require_region_entries

    plan = _plan("tp_shared")
    _require_region_entries(plan)
    del plan["ff"]
    try:
        _require_region_entries(plan)
    except ValueError as error:
        assert "['ff.w1', 'ff.w3']" in str(error), error
    else:
        raise AssertionError("a projection without an enclosing region was accepted")


def test_region_entries_name_arguments_the_module_takes() -> None:
    import pytest

    from flow_factory.models.tensor_parallel import (
        ColumnParallelRegion,
        SequenceParallelRegion,
        _argument_positions,
    )

    attention = _JointAttention()
    assert _argument_positions(attention, ("hidden_states", "encoder_hidden_states")) == {
        "hidden_states": 0,
        "encoder_hidden_states": 1,
    }
    with pytest.raises(ValueError, match="takes no argument named 'image_rotary_emb'"):
        _argument_positions(attention, ("image_rotary_emb",))
    with pytest.raises(TypeError, match="tuple of argument names"):
        ColumnParallelRegion(inputs=["hidden_states"])
    with pytest.raises(ValueError, match="both gathered and replicated"):
        SequenceParallelRegion(gather=("hidden_states",), replicated=("hidden_states",))
    with pytest.raises(ValueError, match="at least one input to gather"):
        SequenceParallelRegion(gather=())
