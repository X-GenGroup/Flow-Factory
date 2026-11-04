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

"""H3's chunked qk-norm under tensor parallelism matches the unsharded model.

Every H3 workflow installs a token-chunked RMSNorm on the per-head qk-norms,
which the plan gives a head-sharded ``DTensor`` input. Assembling the chunks
with ``new_empty`` allocated a replicated tensor, so the copy all-gathered every
head and any sequence longer than one chunk failed downstream.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from torch import nn

WORLD_SIZE = 2
HEADS, HEAD_DIM, DIM = 4, 8, 16


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = HEADS
        self.fused_projections = False
        self.to_q = nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.to_k = nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.norm_q = nn.RMSNorm(HEAD_DIM)
        self.norm_k = nn.RMSNorm(HEAD_DIM)
        self.to_out = nn.ModuleList([nn.Linear(HEADS * HEAD_DIM, DIM, bias=False), nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, -1)))
        return self.to_out[0]((query * key).flatten(-2))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.attn(hidden_states)


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList([_Block()])
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in (*self.token_refiner.refiner_blocks, *self.transformer_blocks):
            hidden_states = block(hidden_states)
        return hidden_states


def _plan() -> Dict[str, Any]:
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    from flow_factory.models.tensor_parallel import ShardedInputReplicatedWeight

    plan: Dict[str, Any] = {}
    for stack in ("token_refiner.refiner_blocks", "transformer_blocks"):
        plan[f"{stack}.*.attn.to_q"] = ColwiseParallel()
        plan[f"{stack}.*.attn.to_k"] = ColwiseParallel()
        plan[f"{stack}.*.attn.to_out.0"] = RowwiseParallel()
        plan[f"{stack}.*.attn.norm_q"] = ShardedInputReplicatedWeight()
        plan[f"{stack}.*.attn.norm_k"] = ShardedInputReplicatedWeight()
    return plan


def _forward_backward(model: nn.Module) -> Dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    hidden = torch.randn(2, 9, DIM, generator=torch.Generator().manual_seed(3)).double()
    output = model(hidden)
    output.pow(2).sum().backward()
    values = {"output": output.detach()}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        values[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach()
    return values


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.minimax_h3._chunking import install_h3_attention_norm_chunking
        from flow_factory.models.tensor_parallel import apply_tensor_parallel

        torch.manual_seed(0)
        reference = _Transformer().double()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(0, 0.3)
        install_h3_attention_norm_chunking(reference, max_tokens=4)
        sharded = copy.deepcopy(reference)
        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        apply_tensor_parallel(sharded, mesh, _plan())

        expected = _forward_backward(reference)
        received = _forward_backward(sharded)
        result["max_relative_diff"] = max(
            ((received[name] - expected[name]).abs().max() / expected[name].abs().max()).item()
            for name in expected
        )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_chunked_qk_norm_across_more_than_one_chunk_matches_the_unsharded_model(
    tmp_path: Path,
) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        assert result["max_relative_diff"] < 1e-9, result
