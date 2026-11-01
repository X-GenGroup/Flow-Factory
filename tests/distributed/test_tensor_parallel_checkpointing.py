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

"""Checkpoints written from a tensor-parallel run, on real process groups.

Flow-Factory has two serialization paths and they do not share code:

* **model-only** -- ``BaseAdapter.get_state_dict`` gathers an FSDP2 model with
  ``get_model_state_dict(full_state_dict=True, broadcast_from_rank0=True)``;
* **exact training state** -- ``accelerator.save_state``/``load_state``, which
  writes a *sharded* DCP checkpoint and reads it back into the same layout.

Tensor parallelism changes what both are handed. FSDP2 composes over an
already-sharded model by concatenating the two meshes, so a column-parallel
weight -- sharded on dim 0 by tensor parallelism -- cannot also be plainly
``Shard(0)`` for data parallelism; FSDP2 describes it with
``_StridedShard(0, split_factor=tp_size)`` instead. A full gather then has to
reassemble across both axes.

Getting that wrong does not raise. It produces a checkpoint that loads cleanly
and holds a *permutation* of the weights, which is the failure mode diffusers
avoids by refusing to save under tensor parallelism at all. So the assertions
here are bit-for-bit against the never-sharded reference, and one of them
separates "scrambled" from "numerically off" by comparing sorted values.
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
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

DIM = 32
HEADS = 4
HEAD_DIM = DIM // HEADS
SEQ = 8
BATCH = 2


class _Block(nn.Module):
    """One attention + FFN block, plus a projection the plan leaves replicated."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = HEADS
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_k = nn.Linear(DIM, DIM, bias=False)
        self.to_v = nn.Linear(DIM, DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=True)
        self.ffn_up = nn.Linear(DIM, 2 * DIM, bias=True)
        self.ffn_down = nn.Linear(2 * DIM, DIM, bias=True)
        self.proj_out = nn.Linear(DIM, DIM, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = (-1, HEAD_DIM)
        query = self.to_q(hidden_states).unflatten(-1, shape).transpose(1, 2)
        key = self.to_k(hidden_states).unflatten(-1, shape).transpose(1, 2)
        value = self.to_v(hidden_states).unflatten(-1, shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states + self.to_out(attended.transpose(1, 2).flatten(-2))
        hidden_states = hidden_states + self.ffn_down(F.gelu(self.ffn_up(hidden_states)))
        return self.proj_out(hidden_states)


def _shard_plan() -> Dict[str, Any]:
    return {
        "to_q": ColwiseParallel(),
        "to_k": ColwiseParallel(),
        "to_v": ColwiseParallel(),
        "to_out": RowwiseParallel(),
        "ffn_up": ColwiseParallel(),
        "ffn_down": RowwiseParallel(),
    }


def _build(lora: bool) -> nn.Module:
    torch.manual_seed(1234)
    model: nn.Module = _Block()
    if lora:
        peft = pytest.importorskip("peft")
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=4,
                init_lora_weights="gaussian",
                target_modules=["to_q", "to_k", "to_v", "to_out", "ffn_up", "ffn_down"],
            ),
        )
    return model


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    from flow_factory.models.tensor_parallel import apply_tensor_parallel, peel_peft_wrapper

    dp_size, tp_size, lora = options["dp_size"], options["tp_size"], options["lora"]
    world_size = dp_size * tp_size
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    result: Dict[str, Any] = {"rank": rank}
    try:
        mesh = init_device_mesh("cpu", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp"))

        # Built from the same seed on every rank and never sharded: the standard
        # the gather has to reproduce.
        reference = {name: p.detach().clone() for name, p in _build(lora).named_parameters()}

        model = _build(lora)
        if tp_size > 1:
            apply_tensor_parallel(peel_peft_wrapper(model), mesh["tp"], _shard_plan())
        if dp_size > 1:
            fully_shard(model, mesh=mesh["dp_shard"])
        result["placements"] = sorted(
            {
                f"{tuple(p.device_mesh.mesh_dim_names)}:{[str(x) for x in p.placements]}"
                for p in model.parameters()
                if hasattr(p, "device_mesh")
            }
        )

        # The options `BaseAdapter.get_state_dict` uses. `broadcast_from_rank0`
        # names the load direction; on a save it means only rank 0 materializes
        # the full dict, which is also the only rank that writes it.
        full = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True)
        gathered = get_model_state_dict(model, options=full)
        scrambled, missing = [], []
        for name, expected in reference.items() if gathered else ():
            # A PEFT wrapper prefixes paths; match on the suffix.
            keys = [key for key in gathered if key == name or key.endswith("." + name)]
            if not keys:
                missing.append(name)
                continue
            actual = gathered[keys[0]].to(dtype=expected.dtype)
            if torch.equal(actual, expected):
                continue
            scrambled.append(
                {
                    "name": name,
                    "max_abs_delta": float((actual.double() - expected.double()).abs().max()),
                    # A permuted gather holds the right multiset of values, so
                    # this separates a wrong layout from a wrong computation.
                    "same_values_sorted": bool(
                        torch.equal(
                            actual.flatten().sort().values, expected.flatten().sort().values
                        )
                    ),
                }
            )
        result["gathered_key_count"] = len(gathered)
        result["reference_key_count"] = len(reference)
        result["missing"] = missing
        result["scrambled"] = scrambled

        # Take one real optimizer step so the saved state is not the initial one,
        # then perturb and reload: a no-op restore would pass trivially.
        batch = torch.randn(BATCH, SEQ, DIM, generator=torch.Generator().manual_seed(7))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-2)
        model(batch).float().pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        stepped = round(float(model(batch).detach().double().sum()), 8)

        sharded = StateDictOptions(full_state_dict=False)
        checkpoint = str(Path(result_dir, "dcp"))
        dist.checkpoint.save(
            {
                "model": get_model_state_dict(model, options=sharded),
                "optim": get_optimizer_state_dict(model, optimizer, options=sharded),
            },
            checkpoint_id=checkpoint,
        )
        with torch.no_grad():
            for parameter in trainable:
                parameter.add_(1.0)
        perturbed = round(float(model(batch).detach().double().sum()), 8)

        model_state = get_model_state_dict(model, options=sharded)
        optimizer_state = get_optimizer_state_dict(model, optimizer, options=sharded)
        dist.checkpoint.load(
            {"model": model_state, "optim": optimizer_state}, checkpoint_id=checkpoint
        )
        set_model_state_dict(model, model_state, options=sharded)
        set_optimizer_state_dict(model, optimizer, optimizer_state, options=sharded)
        result["stepped"] = stepped
        result["perturbed"] = perturbed
        result["restored"] = round(float(model(batch).detach().double().sum()), 8)
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run(tmp_path: Path, **options: Any) -> List[Dict[str, Any]]:
    options.setdefault("lora", False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    store_path = str(tmp_path / f"store-{options['dp_size']}-{options['tp_size']}")
    world_size = options["dp_size"] * options["tp_size"]
    processes = [
        context.Process(target=_worker, args=(rank, store_path, str(tmp_path), options))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(timeout=5)
    assert not hanging, "tensor-parallel checkpoint run exceeded 300 seconds"
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
@pytest.mark.parametrize(
    "dp_size,tp_size",
    [(1, 2), (2, 1), (2, 2)],
    ids=["tp_only", "dp_only", "composed"],
)
def test_the_gathered_checkpoint_matches_the_unsharded_weights(
    tmp_path: Path, lora: bool, dp_size: int, tp_size: int
) -> None:
    """The model-only save path reconstructs every weight exactly.

    ``(2, 2)`` is the case tensor parallelism introduces: the parameter carries
    both meshes, and a column-parallel weight is ``_StridedShard`` on the
    data-parallel axis. The other two are the one-dimensional baselines, so a
    failure localizes to the composition rather than to either axis alone.
    """
    results = _run(tmp_path, dp_size=dp_size, tp_size=tp_size, lora=lora)

    for result in results:
        assert "error" not in result, result["error"]
        # Report the full record: `same_values_sorted` says whether the layout
        # was reassembled wrongly or the values themselves are off.
        assert result["scrambled"] == [], f"rank {result['rank']}: {result['scrambled']}"
        assert result["missing"] == [], f"rank {result['rank']} lost parameters"

    main = next(result for result in results if result["rank"] == 0)
    assert main["gathered_key_count"] == main["reference_key_count"]
    # The other ranks hold nothing, which is what makes rank 0 authoritative.
    assert all(result["gathered_key_count"] == 0 for result in results if result["rank"] != 0), [
        result["gathered_key_count"] for result in results
    ]


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
def test_the_sharded_checkpoint_round_trips_over_the_composed_mesh(
    tmp_path: Path, lora: bool
) -> None:
    """Exact-state save and load restore the model over the concatenated mesh.

    This is the path that carries optimizer state, whose moments inherit the
    parameter's placements -- including ``_StridedShard`` -- so it exercises
    writing and reading a layout that neither axis produces on its own.

    The perturbation is asserted to have moved the forward, because a restore
    that silently loaded nothing would otherwise pass.
    """
    results = _run(tmp_path, dp_size=2, tp_size=2, lora=lora)

    for result in results:
        assert "error" not in result, result["error"]
        assert result["perturbed"] != result["stepped"], "the perturbation did not take effect"
        assert result["restored"] == result["stepped"], (
            f"rank {result['rank']} restored {result['restored']} " f"but saved {result['stepped']}"
        )


def test_composition_strides_the_data_parallel_shard_of_a_column_parallel_weight(
    tmp_path: Path,
) -> None:
    """Record the layout the save path has to understand.

    Tensor parallelism shards a column-parallel weight on dim 0. Data
    parallelism cannot then also take a contiguous ``Shard(0)`` of it, because
    each rank's local tensor is already a slice of dim 0 -- FSDP2 expresses the
    interleave as ``_StridedShard(0, split_factor=tp_size)``. If a torch release
    ever changes that representation this test says so directly, instead of the
    round-trip tests failing for a reason that looks like data corruption.
    """
    results = _run(tmp_path, dp_size=2, tp_size=2)

    placements = results[0]["placements"]
    assert any("_S(0, 2)" in entry for entry in placements), placements
    assert all(entry.startswith("('dp_shard', 'tp')") for entry in placements), placements
