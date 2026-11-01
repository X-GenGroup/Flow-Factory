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

"""Sample-level collectives must span data-parallel replicas, not launched ranks.

A tensor-parallel group holds one sample, so a gather over the launched world
returns every sample ``tp_size`` times. Downstream that is not a shape error: a
group-relative advantage would be computed over duplicated rows, and a groupwise
reward model would be handed ``K * tp_size`` members for a group of ``K``.

The scope is owned by ``GroupCoordinator``. Its ``uses_global_collective``
originally meant "delegate to Accelerate's world collectives", which conflates
"all replicas" with "all ranks" -- the same thing only when no rank is reserved.
These tests pin the distinction.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.parallelism_config import ParallelismConfig

DP_SIZE = 2
TP_SIZE = 2
WORLD_SIZE = DP_SIZE * TP_SIZE


def _accelerator(parallelism_config, mesh, rank: int) -> SimpleNamespace:
    """The surface GroupCoordinator and the gather helpers actually read."""
    return SimpleNamespace(
        parallelism_config=parallelism_config,
        torch_device_mesh=mesh,
        num_processes=WORLD_SIZE,
        process_index=rank,
        device=torch.device("cpu"),
        gather=lambda tensor: _world_gather(tensor),
        reduce=lambda tensor, reduction="sum": _world_reduce(tensor, reduction),
    )


def _world_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Stand in for `accelerator.gather`, which spans every launched rank."""
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def _world_reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if reduction == "mean":
        reduced /= dist.get_world_size()
    return reduced


def _worker(rank: int, store_path: str, result_dir: str, sampler_type: str) -> None:
    from flow_factory.utils.group_coordinator import GroupCoordinator

    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        parallelism_config = ParallelismConfig(dp_shard_size=DP_SIZE, tp_size=TP_SIZE)
        mesh = parallelism_config.build_device_mesh("cpu")
        accelerator = _accelerator(parallelism_config, mesh, rank)
        dp_rank = mesh["dp_shard"].get_local_rank()

        coordinator = GroupCoordinator(accelerator, sampler_type=sampler_type)
        result["dp_world_size"] = coordinator.dp_world_size
        result["group_world_size"] = coordinator.group_world_size
        result["dp_rank"] = dp_rank

        # One value per replica, duplicated across each tensor-parallel group,
        # which is what a real rollout produces. Each probe is recorded
        # independently so one failure does not mask the others' results.
        payload = torch.tensor([float(dp_rank)])
        for key, probe in (
            ("gathered", lambda: coordinator.gather(payload).tolist()),
            ("reduced", lambda: coordinator.reduce_sum(payload.clone()).tolist()),
            ("gathered_objects", lambda: coordinator.gather_object([dp_rank])),
        ):
            try:
                result[key] = probe()
            except Exception as error:
                result[f"{key}_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run(tmp_path: Path, sampler_type: str) -> List[Dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    store_path = str(tmp_path / "store")
    processes = [
        context.Process(target=_worker, args=(rank, store_path, str(tmp_path), sampler_type))
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
    assert not hanging, "sample collective scope run exceeded 240 seconds"
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(WORLD_SIZE)
    ]


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


@pytest.mark.parametrize("sampler_type", ["global_batch", "global_tile"])
def test_a_cross_rank_gather_returns_one_row_per_replica(tmp_path: Path, sampler_type: str) -> None:
    """Two replicas split two ways must gather two rows, not four."""
    results = _run(tmp_path / sampler_type, sampler_type)

    for result in results:
        assert "error" not in result, result["error"]
        assert result["dp_world_size"] == DP_SIZE
        assert result["group_world_size"] == DP_SIZE
        # Each replica contributes its own index exactly once.
        assert "gathered_error" not in result, result["gathered_error"]
        assert result["gathered"] == [0.0, 1.0], result


@pytest.mark.parametrize("sampler_type", ["global_batch", "global_tile"])
def test_a_cross_rank_reduction_sums_over_replicas_only(tmp_path: Path, sampler_type: str) -> None:
    """Summing over launched ranks would double every group statistic."""
    results = _run(tmp_path / sampler_type, sampler_type)

    for result in results:
        # dp_rank values 0 and 1 sum to 1; over four ranks they would sum to 2.
        assert "reduced_error" not in result, result["reduced_error"]
        assert result["reduced"] == [1.0], result


@pytest.mark.parametrize("sampler_type", ["global_batch", "global_tile"])
def test_object_gathering_is_available_at_replica_scope(tmp_path: Path, sampler_type: str) -> None:
    """The object path must cover the all-replicas case, not refuse it.

    It used to delegate the global case to Accelerate and raise if asked, which
    is only correct while the launched world is the replica set.
    """
    results = _run(tmp_path / sampler_type, sampler_type)

    for result in results:
        assert "error" not in result, result["error"]
        assert "gathered_objects_error" not in result, result["gathered_objects_error"]
        assert result["gathered_objects"] == [0, 1], result


def test_a_rank_local_layout_issues_no_collective(tmp_path: Path) -> None:
    """Whole groups live on one replica, so there is nothing to span."""
    results = _run(tmp_path / "rank_local", "group_contiguous")

    for result in results:
        assert "error" not in result, result["error"]
        assert result["group_world_size"] == 1
        # The payload comes back untouched rather than concatenated.
        assert result["gathered"] == [float(result["dp_rank"])], result
        assert result["reduced"] == [float(result["dp_rank"])], result


def test_a_subgroup_tile_layout_is_still_refused(tmp_path: Path) -> None:
    """Its subgroups are built from contiguous launched ranks.

    That stops matching the replica set once ranks are reserved, and silently
    averaging duplicated rows is worse than refusing to start.
    """
    results = _run(tmp_path / "subgroup", "subgroup_tile")

    for result in results:
        assert "reserved for tensor parallelism" in result.get("error", ""), result
