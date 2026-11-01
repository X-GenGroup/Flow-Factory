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

"""Tensor parallelism on real process groups, composed under FSDP2.

The invariant worth protecting is that tensor parallelism is *numerically
transparent*: the same batch must produce the same gradients whatever
``tp_size`` is. Every bug this module guards against was found by comparing
gradients, and two of them are invisible in the total gradient norm:

* per-head qk-norm accumulating a partial sum, whose gradients are ~1e-5
  against a norm of ~1;
* torch's own parallel styles unfreezing ``base_layer`` under LoRA, which only
  shows up as a *change in which parameters have gradients at all*.

So the comparisons below check the gradient of every parameter and the set of
parameters that have one, not a summary statistic.
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
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

DIM = 32
HEADS = 4
HEAD_DIM = DIM // HEADS
SEQ = 8
BATCH = 2


class _Block(nn.Module):
    """One attention + FFN block, reshaping the way a TP-agnostic processor does."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = HEADS
        self.norm = nn.LayerNorm(DIM)
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_k = nn.Linear(DIM, DIM, bias=False)
        self.to_v = nn.Linear(DIM, DIM, bias=False)
        self.norm_q = nn.RMSNorm(HEAD_DIM)
        self.to_out = nn.Linear(DIM, DIM, bias=False)
        self.ffn_up = nn.Linear(DIM, 2 * DIM, bias=False)
        self.ffn_down = nn.Linear(2 * DIM, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.norm(hidden_states)
        # `-1` for the head count: under column-parallel q/k/v this rank holds
        # only heads/tp heads and the global count would mis-split the row.
        query = self.norm_q(self.to_q(normed).unflatten(-1, (-1, HEAD_DIM))).transpose(1, 2)
        key = self.to_k(normed).unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2)
        value = self.to_v(normed).unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states + self.to_out(attended.transpose(1, 2).flatten(-2))
        return hidden_states + self.ffn_down(F.gelu(self.ffn_up(hidden_states)))


def _shard_plan() -> Dict[str, Any]:
    from flow_factory.models.tensor_parallel import ShardedInputReplicatedWeight

    return {
        "to_q": ColwiseParallel(),
        "to_k": ColwiseParallel(),
        "to_v": ColwiseParallel(),
        "to_out": RowwiseParallel(),
        "ffn_up": ColwiseParallel(),
        "ffn_down": RowwiseParallel(),
        "norm_q": ShardedInputReplicatedWeight(),
    }


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    from flow_factory.models.tensor_parallel import apply_tensor_parallel

    dp_size, tp_size = options["dp_size"], options["tp_size"]
    world_size = dp_size * tp_size
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    result: Dict[str, Any] = {"rank": rank}
    try:
        mesh = init_device_mesh("cpu", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp"))
        dp_rank = mesh["dp_shard"].get_local_rank()

        torch.manual_seed(1234)
        model: nn.Module = _Block()
        if options["lora"]:
            peft = pytest.importorskip("peft")
            model = peft.get_peft_model(
                model,
                peft.LoraConfig(r=4, target_modules=["to_q", "to_k", "to_v", "to_out"]),
            )
        target = model.base_model.model if options["lora"] else model

        if tp_size > 1:
            plan = _shard_plan() if options["total_plan"] else _partial_plan()
            apply_tensor_parallel(target, mesh["tp"], plan)
        if dp_size > 1:
            fully_shard(model, mesh=mesh["dp_shard"])

        # Keyed by dp_rank: tensor-parallel peers must consume the same sample.
        generator = torch.Generator().manual_seed(9000 + dp_rank)
        batch = torch.randn(BATCH, SEQ, DIM, generator=generator)
        loss = model(batch).float().pow(2).mean()
        # Recorded before backward: FSDP reduces gradients across the
        # data-parallel mesh, so only the local loss still shows which
        # sample this replica was handed.
        result["loss"] = round(float(loss.detach()), 6)
        loss.backward()

        gradients = {
            name: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).detach().float()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        result["grad_names"] = sorted(gradients)
        result["grad_checksums"] = {
            name: round(float(tensor.double().sum()), 6)
            for name, tensor in sorted(gradients.items())
        }
        try:
            clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
            result["clip_norm"] = round(
                float(clipped.full_tensor() if isinstance(clipped, DTensor) else clipped), 6
            )
        except Exception as error:
            result["clip_error"] = f"{type(error).__name__}: {str(error)[:120]}"
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _partial_plan() -> Dict[str, Any]:
    """A plan that forgets the norms, to prove the applier refuses it."""
    plan = _shard_plan()
    del plan["norm_q"]
    return plan


def _run(tmp_path: Path, **options: Any) -> List[Dict[str, Any]]:
    options.setdefault("lora", False)
    options.setdefault("total_plan", True)
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
    assert not hanging, "tensor-parallel sharding run exceeded 300 seconds"
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
def test_gradients_are_unchanged_by_the_tensor_parallel_degree(tmp_path: Path, lora: bool) -> None:
    """Two data-parallel replicas either way, each split two ways or not.

    The replica count is held constant on purpose: it fixes the global batch,
    so any difference left is the sharding itself rather than the data.
    """
    baseline = _run(tmp_path / "tp1", dp_size=2, tp_size=1, lora=lora)
    sharded = _run(tmp_path / "tp2", dp_size=2, tp_size=2, lora=lora)

    for result in baseline + sharded:
        assert "error" not in result, result["error"]

    # Same parameters must carry gradients. torch's own styles rebuild each
    # parameter without forwarding `requires_grad`, which under LoRA unfreezes
    # every sharded `base_layer` -- visible here and nowhere in the norm.
    assert sharded[0]["grad_names"] == baseline[0]["grad_names"]

    for name, expected in baseline[0]["grad_checksums"].items():
        actual = sharded[0]["grad_checksums"][name]
        assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6), name

    assert sharded[0]["clip_norm"] == pytest.approx(baseline[0]["clip_norm"], rel=1e-4, abs=1e-6)


def test_tensor_parallel_peers_consume_one_sample(tmp_path: Path) -> None:
    """Ranks of one group must see the same sample; distinct replicas must not.

    Compared on the local loss rather than the gradients: FSDP reduces
    gradients over the data-parallel mesh, so every rank ends up holding the
    same averaged gradient and that comparison cannot tell the two cases apart.
    A replica pair that agreed here would mean the sampler handed both the same
    rows, and the run would silently train on half its data.
    """
    results = _run(tmp_path, dp_size=2, tp_size=2)
    for result in results:
        assert "error" not in result, result["error"]
    by_rank = {result["rank"]: result["loss"] for result in results}

    assert by_rank[0] == by_rank[1], "tensor-parallel peers saw different samples"
    assert by_rank[2] == by_rank[3], "tensor-parallel peers saw different samples"
    assert by_rank[0] != by_rank[2], "distinct replicas saw the same batch"


def test_a_per_head_norm_treated_as_replicated_gets_a_wrong_gradient(
    tmp_path: Path,
) -> None:
    """Lock the reason :class:`ShardedInputReplicatedWeight` exists.

    Dropping ``norm_q`` from the shard plan does not leave it undistributed --
    completion fills it in as plainly replicated. That is the subtle case: its
    weight is shared across heads but its input is head-sharded, so each rank
    accumulates a partial sum over only its own heads and nothing reconciles
    them. The forward pass and every other gradient stay correct, which is why
    this is caught by comparing one parameter rather than a summary statistic.
    """
    baseline = _run(tmp_path / "ref", dp_size=2, tp_size=1)
    replicated = _run(tmp_path / "wrong", dp_size=2, tp_size=2, total_plan=False)
    correct = _run(tmp_path / "right", dp_size=2, tp_size=2)

    for result in baseline + replicated + correct:
        assert "error" not in result, result["error"]

    expected = baseline[0]["grad_checksums"]["norm_q.weight"]

    assert correct[0]["grad_checksums"]["norm_q.weight"] == pytest.approx(
        expected, rel=1e-4, abs=1e-6
    )
    assert replicated[0]["grad_checksums"]["norm_q.weight"] != pytest.approx(
        expected, rel=1e-4, abs=1e-6
    )
    # Everything the styles do cover still agrees, so the damage is confined to
    # the one parameter and would not show up in the gradient norm.
    assert replicated[0]["grad_checksums"]["to_q.weight"] == pytest.approx(
        baseline[0]["grad_checksums"]["to_q.weight"], rel=1e-4, abs=1e-6
    )
