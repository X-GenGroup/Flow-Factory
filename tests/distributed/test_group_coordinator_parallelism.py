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

from types import SimpleNamespace

import pytest

from flow_factory.utils.group_coordinator import GroupCoordinator


class _SubMesh:
    """One sliced DeviceMesh dimension."""

    def __init__(self, size: int, rank: int) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


class _DeviceMesh:
    """Stand in for a torch DeviceMesh with a flattened ``dp`` dimension."""

    def __init__(self, dp_size: int, dp_rank: int) -> None:
        self.mesh_dim_names = ("dp", "tp")
        self._dp = _SubMesh(dp_size, dp_rank)

    def __getitem__(self, key: str) -> _SubMesh:
        if key != "dp":
            raise KeyError(key)
        return self._dp


def _accelerator(torch_device_mesh: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        num_processes=8,
        process_index=3,
        torch_device_mesh=torch_device_mesh,
    )


def test_rank_local_needs_no_collective_scope() -> None:
    """A rank-local layout owns its groups outright, with or without a mesh."""
    coordinator = GroupCoordinator(_accelerator(), sampler_type="rank_local")

    assert (coordinator.group_world_size, coordinator.group_rank) == (1, 0)


def test_rank_local_stays_rank_local_under_tensor_parallelism() -> None:
    """Rank-local groups never cross a replica, so tensor parallelism is safe there."""
    coordinator = GroupCoordinator(
        _accelerator(_DeviceMesh(dp_size=4, dp_rank=1)), sampler_type="rank_local"
    )

    assert (coordinator.group_world_size, coordinator.group_rank) == (1, 0)


@pytest.mark.parametrize("sampler_type", ["global_batch", "global_tile"])
def test_cross_rank_layouts_span_every_replica_without_tensor_parallelism(
    sampler_type: str,
) -> None:
    """With no ranks reserved, the launched world is the replica set."""
    coordinator = GroupCoordinator(_accelerator(), sampler_type=sampler_type)

    assert coordinator.group_world_size == 8
    assert coordinator.group_rank == 3
    assert coordinator.uses_global_collective is True


@pytest.mark.parametrize(
    "sampler_type",
    ["global_batch", "global_tile", "subgroup_tile", "contiguous_shard", "global_random"],
)
def test_cross_rank_layouts_are_rejected_under_tensor_parallelism(sampler_type: str) -> None:
    """Their collectives are not scoped to the data-parallel group yet.

    Gathering across the whole world would return each sample once per tensor-parallel
    rank, so the reduction would silently average duplicated rows.
    """
    accelerator = _accelerator(_DeviceMesh(dp_size=4, dp_rank=1))

    with pytest.raises(ValueError, match="not yet scoped to the data-parallel group"):
        GroupCoordinator(accelerator, sampler_type=sampler_type)


def test_the_rejection_names_both_scopes() -> None:
    """The failure has to say what was found, not just that something was unsupported."""
    accelerator = _accelerator(_DeviceMesh(dp_size=4, dp_rank=1))

    with pytest.raises(ValueError, match=r"data_parallel_size=4, num_processes=8"):
        GroupCoordinator(accelerator, sampler_type="global_batch")
