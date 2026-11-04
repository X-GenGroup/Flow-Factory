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

"""``parallelism.verify_replicated_inputs`` names the input the peers disagree on.

Peers that feed a sharded component different inputs get an output matching
neither one's unsharded result, and nothing raises; the check turns that into
an error on every peer at once, so none is left waiting in a collective.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from torch import nn

WORLD_SIZE = 2


class _Conditioned(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, x: torch.Tensor, *, cond: Mapping[str, Any]) -> torch.Tensor:
        return self.mlp(x + cond["shift"].to(x.dtype))


def _outcome(module: nn.Module, x: torch.Tensor, cond: Mapping[str, Any]) -> str:
    try:
        with torch.no_grad():
            module(x, cond=cond)
    except RuntimeError as error:
        return str(error)
    return "ok"


def _check_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            install_replicated_input_check,
        )

        torch.manual_seed(0)
        module = _Conditioned()
        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        apply_tensor_parallel(module, mesh, {"mlp.0": ColwiseParallel(), "mlp.2": RowwiseParallel()})
        install_replicated_input_check(module, mesh, "generator/transformer")

        shared = torch.randn(2, 8, generator=torch.Generator().manual_seed(7))
        shift = torch.zeros(2, 8, dtype=torch.bfloat16)
        result["identical"] = _outcome(module, shared, {"shift": shift, "step": torch.tensor(3)})
        result["positional"] = _outcome(module, shared + rank, {"shift": shift})
        result["nested"] = _outcome(module, shared, {"shift": shift + rank})
        result["shapes"] = _outcome(module, torch.zeros(2 + rank, 8), {"shift": shift[:1]})
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_the_check_passes_agreeing_peers_and_names_what_differs(tmp_path: Path) -> None:
    results = run_gloo_ranks(_check_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        assert result["identical"] == "ok"
        assert "with different args[0]: 16 of 16 element(s) differ" in result["positional"]
        assert "generator/transformer" in result["positional"]
        assert "with different kwargs['cond']['shift']" in result["nested"]
        assert "torch.bfloat16" in result["nested"]
        assert "differently shaped inputs" in result["shapes"]
