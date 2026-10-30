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
from typing import Optional, Tuple

from flow_factory.utils.dist import data_parallel_geometry


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
    """Stand in for a torch DeviceMesh, exposing only slicing and dimension names."""

    def __init__(self, mesh_dim_names: Tuple[str, ...], dp: Optional[_SubMesh]) -> None:
        self.mesh_dim_names = mesh_dim_names
        self._dp = dp

    def __getitem__(self, key: str) -> _SubMesh:
        if key != "dp" or self._dp is None:
            raise KeyError(key)
        return self._dp


def _accelerator(torch_device_mesh: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        num_processes=8,
        process_index=3,
        torch_device_mesh=torch_device_mesh,
    )


def test_without_a_mesh_the_launched_world_is_the_data_parallel_group() -> None:
    """A run with no parallelism plan keeps one replica per process."""
    assert data_parallel_geometry(_accelerator()) == (8, 3)


def test_the_flattened_dp_dimension_sizes_the_data_parallel_group() -> None:
    """Four data-parallel replicas of a two-way tensor-parallel group, not eight."""
    mesh = _DeviceMesh(("dp", "tp"), _SubMesh(size=4, rank=1))

    assert data_parallel_geometry(_accelerator(mesh)) == (4, 1)


def test_a_mesh_without_a_dp_dimension_falls_back_to_the_launched_world() -> None:
    """A mesh that reserves every rank for another dimension has no dp group to read."""
    mesh = _DeviceMesh(("tp",), None)

    assert data_parallel_geometry(_accelerator(mesh)) == (8, 3)
