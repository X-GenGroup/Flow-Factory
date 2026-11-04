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

"""TP/CP geometry and logical sample scopes on actual meshes, without model CP."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from accelerate.state import PartialState
from accelerate.utils import DistributedType
from conftest import requires_gloo, run_gloo_ranks
from torch.utils.data import DistributedSampler

from flow_factory.contracts.topology import ParallelTopology
from flow_factory.data_utils.sampling_plan import build_sampling_plan
from flow_factory.utils.dist import (
    _gather_field_values,
    all_reduce_across_replicas_,
    context_parallel_geometry,
    context_parallel_process_group,
    data_parallel_geometry,
    data_parallel_process_group,
    parallel_topology,
    reduce_across_replicas,
    seed_data_parallel_replica,
    tensor_parallel_geometry,
    tensor_parallel_process_group,
)
from flow_factory.utils.group_coordinator import GroupCoordinator


def _world_gather(tensor: torch.Tensor) -> torch.Tensor:
    values = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(values, tensor)
    return torch.cat(values)


def _world_reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    result = tensor.clone()
    dist.all_reduce(result)
    return result / dist.get_world_size() if reduction == "mean" else result


def _scope_worker(rank: int, store_path: str, result_dir: str, options: dict[str, Any]) -> None:
    topology = ParallelTopology(**options["dimensions"])
    device_type = options.get("device_type", "cpu")
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        store=dist.FileStore(store_path, topology.world_size),
        rank=rank,
        world_size=topology.world_size,
    )
    try:
        # Accelerate's world-scope object gather consults PartialState, not the
        # mere existence of a torch process group. Mirror launcher identity so
        # the unsharded compatibility path exercises the real world collective.
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(topology.world_size),
            LOCAL_WORLD_SIZE=str(topology.world_size),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )
        PartialState(cpu=device_type == "cpu")
        config = ParallelismConfig(**topology.dimensions, sp_size=1)
        mesh = config.build_device_mesh(device_type)
        accelerator = SimpleNamespace(
            parallelism_config=config,
            torch_device_mesh=mesh,
            num_processes=topology.world_size,
            process_index=rank,
            device=device,
            gather=_world_gather,
            reduce=_world_reduce,
            distributed_type=DistributedType.FSDP,
        )
        dp_size, dp_rank = data_parallel_geometry(accelerator)
        assert parallel_topology(accelerator) == topology
        result = {
            "rank": rank,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "runtime": {
                "world_size": dist.get_world_size(),
                "rank": dist.get_rank(),
                "device": device.type,
                "process_group_backend": dist.get_backend(),
            },
        }
        for scope, getter in (
            ("data", data_parallel_process_group),
            ("tensor", tensor_parallel_process_group),
            ("context", context_parallel_process_group),
        ):
            group = getter(accelerator)
            members = topology.group_ranks(rank, scope)
            if len(members) > 1:
                actual = dist.get_process_group_ranks(group)
                assert actual == list(members), (scope, actual, members)
        assert tensor_parallel_geometry(accelerator) == (topology.tp_size, rank % topology.tp_size)
        assert context_parallel_geometry(accelerator) == (
            topology.cp_size,
            (rank // topology.tp_size) % topology.cp_size,
        )
        plan = build_sampling_plan(
            layout_name="rank_local",
            dataset_size=32,
            per_device_batch_size=2,
            group_size=2,
            unique_group_count=2 * dp_size,
            num_replicas=dp_size,
            seed=17,
            epoch=0,
        )
        result["sample_ids"] = [
            item.sample_id for batch in plan.batches_for_rank(dp_rank) for item in batch
        ]
        result["offline_indices"] = list(
            DistributedSampler(
                range(8 * dp_size),
                num_replicas=dp_size,
                rank=dp_rank,
                shuffle=False,
            )
        )
        seed_data_parallel_replica(accelerator, 123)
        result["rng"] = torch.rand(3, device=device).cpu().tolist()
        payload = torch.tensor([1.0, float(dp_rank + 1)], device=device)
        result["replica_sum"] = reduce_across_replicas(accelerator, payload).cpu().tolist()
        copied = payload.clone()
        all_reduce_across_replicas_(copied, dist.ReduceOp.SUM, accelerator)
        assert torch.equal(copied, torch.tensor(result["replica_sum"], device=device))
        coordinator = GroupCoordinator(accelerator, sampler_type="global_tile")
        result["gather"] = coordinator.gather(payload[1:]).cpu().tolist()
        result["group_sum"] = coordinator.reduce_sum(payload).cpu().tolist()
        result["objects"] = coordinator.gather_object([dp_rank])
        if dp_size == 1:
            heterogeneous = [torch.arange(2, device=device), torch.arange(3, device=device)]
            gathered = _gather_field_values(accelerator, heterogeneous, "cpu", coordinator)
            assert [item.tolist() for item in gathered] == [[0, 1], [0, 1, 2]]
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def _assert_scopes(rows: list[dict[str, Any]], topology: ParallelTopology) -> None:
    expected_sum = [
        float(topology.data_parallel_size),
        sum(range(1, topology.data_parallel_size + 1)),
    ]
    for row in rows:
        rank = row["rank"]
        assert row["runtime"]["rank"] == rank
        assert row["runtime"]["world_size"] == topology.world_size
        assert row["dp_size"] == topology.data_parallel_size
        assert row["replica_sum"] == expected_sum
        assert row["group_sum"] == expected_sum
        assert row["gather"] == list(range(1, topology.data_parallel_size + 1))
        assert row["objects"] == list(range(topology.data_parallel_size))
        for peer in topology.group_ranks(rank, "sample"):
            assert rows[peer]["sample_ids"] == row["sample_ids"]
            assert rows[peer]["offline_indices"] == row["offline_indices"]
            assert rows[peer]["rng"] == row["rng"]
        for peer in topology.group_ranks(rank, "data"):
            if peer != rank:
                assert set(rows[peer]["sample_ids"]).isdisjoint(row["sample_ids"])
                assert set(rows[peer]["offline_indices"]).isdisjoint(row["offline_indices"])
                assert rows[peer]["rng"] != row["rng"]
    print(
        json.dumps(
            {
                "probe": "topology_only",
                "dimensions": topology.dimensions,
                "ranks_completed": [row["rank"] for row in rows],
                "runtime": rows[0]["runtime"],
                "replica_sum": expected_sum,
                "sampling_rng_and_feedback_assertions": "passed",
            }
        )
    )


@requires_gloo
@pytest.mark.parametrize(
    "dimensions",
    [
        {"dp_shard_size": 2, "cp_size": 2, "tp_size": 2},
        {"cp_size": 2, "tp_size": 2},
        {"dp_replicate_size": 2, "dp_shard_size": 2, "tp_size": 2},
    ],
)
def test_combined_meshes_preserve_sampling_and_feedback_scopes(
    tmp_path: Path, dimensions: dict[str, int]
) -> None:
    topology = ParallelTopology(**dimensions)
    rows = run_gloo_ranks(_scope_worker, tmp_path, topology.world_size, {"dimensions": dimensions})
    _assert_scopes(rows, topology)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
@pytest.mark.parametrize("dimensions", [{"dp_shard_size": 2}, {"tp_size": 2}, {"cp_size": 2}])
def test_two_rank_cuda_scopes(tmp_path: Path, dimensions: dict[str, int]) -> None:
    """NCCL smoke covers replica sums and singleton-replica bypass, not CP models."""
    topology = ParallelTopology(**dimensions)
    rows = run_gloo_ranks(
        _scope_worker, tmp_path, 2, {"dimensions": dimensions, "device_type": "cuda"}
    )
    _assert_scopes(rows, topology)


@pytest.mark.parametrize(
    "backend,expected",
    [
        (DistributedType.MULTI_GPU, (8, 1)),
        (DistributedType.DEEPSPEED, (8, 1)),
        (DistributedType.FSDP, (1, 8)),
    ],
)
def test_unconfigured_backend_keeps_its_physical_dp_ownership(backend, expected) -> None:
    accelerator = SimpleNamespace(num_processes=8, process_index=3, distributed_type=backend)
    topology = parallel_topology(accelerator)
    assert (topology.dp_replicate_size, topology.dp_shard_size) == expected
    assert data_parallel_geometry(accelerator) == (8, 3)
    accelerator.parallelism_config = ParallelismConfig()
    assert parallel_topology(accelerator) == topology


def test_independent_sp_cannot_silently_change_logical_geometry() -> None:
    accelerator = SimpleNamespace(num_processes=8, parallelism_config=SimpleNamespace(sp_size=2))
    with pytest.raises(ValueError, match="independent sequence parallelism sp_size=2"):
        parallel_topology(accelerator)


@pytest.mark.parametrize("permuted_ranks", [False, True])
def test_noncanonical_backend_placement_is_not_silently_described_as_canonical(
    permuted_ranks,
) -> None:
    class Placement:
        mesh_dim_names = ("dp_shard", "tp") if permuted_ranks else ("tp", "dp_shard")
        mesh = torch.tensor([[1, 0], [3, 2]]) if permuted_ranks else torch.arange(4).reshape(2, 2)

    accelerator = SimpleNamespace(
        num_processes=4,
        parallelism_config=ParallelismConfig(dp_shard_size=2, tp_size=2),
        torch_device_mesh=Placement(),
    )
    with pytest.raises(ValueError, match="does not match canonical topology"):
        parallel_topology(accelerator)
