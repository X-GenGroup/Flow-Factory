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

"""Data-parallel geometry read from a real Accelerate device mesh.

These run on real process groups rather than a stub mesh on purpose. The
geometry is read out of a ``DeviceMesh`` whose joint ``dp`` dimension Accelerate
creates with ``DeviceMesh._flatten``, and a flattened dimension is recorded in
the mesh's flatten mapping while staying *out* of ``mesh_dim_names``. A stub
that sets ``mesh_dim_names = ("dp", "tp")`` therefore passes while the same code
reads the wrong geometry from every real mesh, so the stub cannot be the test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.parallelism_config import ParallelismConfig

from flow_factory.utils.dist import data_parallel_geometry

WORLD_SIZE = 4


def _geometry_worker(rank: int, store_path: str, result_dir: str, sizes: Dict[str, int]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        parallelism_config = ParallelismConfig(**sizes)
        mesh = parallelism_config.build_device_mesh("cpu")
        accelerator = SimpleNamespace(
            parallelism_config=parallelism_config,
            torch_device_mesh=mesh,
            num_processes=WORLD_SIZE,
            process_index=rank,
        )
        world_size, dp_rank = data_parallel_geometry(accelerator)
        result.update(
            dp_world_size=world_size,
            dp_rank=dp_rank,
            root_mesh_dim_names=list(mesh.mesh_dim_names or ()),
            dp_in_root_names="dp" in (mesh.mesh_dim_names or ()),
        )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run_four_ranks(tmp_path: Path, sizes: Dict[str, int]) -> List[Dict[str, Any]]:
    context = mp.get_context("spawn")
    store_path = str(tmp_path / "gloo-store")
    processes = [
        context.Process(target=_geometry_worker, args=(rank, store_path, str(tmp_path), sizes))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(timeout=5)
    assert not hanging, "four-rank data-parallel geometry run exceeded 120 seconds"
    assert [process.exitcode for process in processes] == [0] * WORLD_SIZE
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(WORLD_SIZE)
    ]


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


def test_without_a_parallelism_plan_the_launched_world_is_the_replica_set() -> None:
    """No plan means one replica per process, which is every current run."""
    accelerator = SimpleNamespace(num_processes=8, process_index=3)

    assert data_parallel_geometry(accelerator) == (8, 3)


def test_tensor_parallel_ranks_share_a_data_parallel_coordinate(tmp_path: Path) -> None:
    """Two-way tensor parallelism over four ranks leaves two replicas, not four."""
    results = _run_four_ranks(tmp_path, {"dp_shard_size": 2, "tp_size": 2})

    for result in results:
        assert "error" not in result, result["error"]
        assert result["dp_world_size"] == 2, result

    # Ranks 0/1 form one tensor-parallel group and 2/3 the other, so each pair
    # must report the same replica index -- that index is what selects samples.
    by_rank = {result["rank"]: result["dp_rank"] for result in results}
    assert by_rank == {0: 0, 1: 0, 2: 1, 3: 1}


def test_the_joint_dp_dimension_is_absent_from_the_root_mesh_names(
    tmp_path: Path,
) -> None:
    """Guard the reason this is not read from ``mesh_dim_names``.

    Accelerate flattens ``dp_shard`` into a joint ``dp`` dimension, which does
    not appear in the root mesh's names. Code that gates on
    ``"dp" in mesh.mesh_dim_names`` silently falls back to the launched world
    and hands every tensor-parallel rank its own sample.
    """
    results = _run_four_ranks(tmp_path, {"dp_shard_size": 2, "tp_size": 2})

    for result in results:
        assert result["root_mesh_dim_names"] == ["dp_shard", "tp"], result
        assert result["dp_in_root_names"] is False, result


def test_a_world_fully_reserved_for_tensor_parallelism_has_one_replica(
    tmp_path: Path,
) -> None:
    """With every rank in one tensor-parallel group there is a single replica.

    The parallelism config carries no data-parallel dimension at all here, so
    the size cannot come from the mesh and has to be read from the plan.
    """
    results = _run_four_ranks(tmp_path, {"tp_size": WORLD_SIZE})

    for result in results:
        assert "error" not in result, result["error"]
        assert (result["dp_world_size"], result["dp_rank"]) == (1, 0), result
