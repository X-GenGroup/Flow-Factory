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

"""Prepared dataloaders must hand one batch per replica, not per rank.

Evaluation loaders are built unsharded and sharded by ``Accelerator.prepare``.
If that dispatch counted launched ranks, tensor-parallel peers would evaluate
different prompts -- the model would then combine partial attention results
belonging to different samples, with matching shapes and no error.

Accelerate's ``prepare_data_loader`` is mesh-aware and divides the dispatch by
the tensor-parallel degree itself. This pins that behaviour rather than trusting
it: the same reasoning said ``"dp" in mesh.mesh_dim_names`` would work, and it
does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

DP_SIZE = 2
TP_SIZE = 2
WORLD_SIZE = DP_SIZE * TP_SIZE
ROWS = 8


class _Prompts(Dataset):
    """One distinguishable row per index, as an evaluation set would be."""

    def __len__(self) -> int:
        return ROWS

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"row": index}


def _worker(rank: int, store_path: str, result_dir: str, use_mesh: bool) -> None:
    # `prepare_data_loader` is exercised directly rather than through
    # `Accelerator`: a ParallelismConfig refuses a CPU backend, while the
    # dispatch arithmetic under test lives in this function and takes the mesh
    # as an argument. `Accelerator.prepare_data_loader` passes exactly the
    # launched-world `num_processes`/`process_index` plus the mesh, as below.
    from accelerate.data_loader import prepare_data_loader
    from torch.distributed.device_mesh import init_device_mesh

    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        mesh = (
            init_device_mesh("cpu", (DP_SIZE, TP_SIZE), mesh_dim_names=("dp_shard", "tp"))
            if use_mesh
            else None
        )
        loader = prepare_data_loader(
            DataLoader(_Prompts(), batch_size=1, shuffle=False),
            device=torch.device("cpu"),
            num_processes=WORLD_SIZE,
            process_index=rank,
            put_on_device=False,
            torch_device_mesh=mesh,
        )
        result["rows"] = [int(batch["row"][0]) for batch in loader]
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()


def _run(tmp_path: Path, *, use_mesh: bool) -> List[Dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    store_path = str(tmp_path / "store")
    processes = [
        context.Process(target=_worker, args=(rank, store_path, str(tmp_path), use_mesh))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=240)
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(timeout=5)
    assert not hanging, "prepared dataloader scope run exceeded 240 seconds"
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(WORLD_SIZE)
    ]


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


def test_tensor_parallel_peers_receive_the_same_rows(tmp_path: Path) -> None:
    """Two replicas split two ways: peers share rows, replicas do not."""
    results = _run(tmp_path / "mesh", use_mesh=True)

    for result in results:
        assert "error" not in result, result["error"]
    rows = {result["rank"]: result["rows"] for result in results}

    assert rows[0] == rows[1], f"tensor-parallel peers diverged: {rows}"
    assert rows[2] == rows[3], f"tensor-parallel peers diverged: {rows}"
    assert rows[0] != rows[2], f"distinct replicas saw the same rows: {rows}"

    # Every row is still covered exactly once across the replicas.
    assert sorted(rows[0] + rows[2]) == list(range(ROWS)), rows


def test_without_a_mesh_every_rank_is_a_replica(tmp_path: Path) -> None:
    """The unchanged path: four ranks, four disjoint shards."""
    results = _run(tmp_path / "plain", use_mesh=False)

    for result in results:
        assert "error" not in result, result["error"]
    rows = {result["rank"]: result["rows"] for result in results}

    assert len({tuple(value) for value in rows.values()}) == WORLD_SIZE, rows
    assert sorted(row for value in rows.values() for row in value) == list(range(ROWS))
