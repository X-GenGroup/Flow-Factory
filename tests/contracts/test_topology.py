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

"""Independent rank-membership and validation oracles for logical topology."""

from dataclasses import FrozenInstanceError

import pytest

from flow_factory.contracts.topology import ParallelTopology


def test_combined_dimensions_keep_sample_and_statistics_scopes_distinct() -> None:
    topology = ParallelTopology(dp_replicate_size=2, dp_shard_size=2, cp_size=2, tp_size=2)

    assert topology.world_size == 16
    assert topology.data_parallel_size == 4
    assert topology.model_parallel_size == 4
    assert topology.coordinates(11) == (1, 0, 1, 1)
    assert topology.data_parallel_rank(11) == 2
    assert topology.group_ranks(11, "sample") == (8, 9, 10, 11)
    assert topology.group_ranks(11, "tensor") == (10, 11)
    assert topology.group_ranks(11, "context") == (9, 11)
    assert topology.group_ranks(11, "data") == (3, 7, 11, 15)
    # A sample count/statistic is contributed once per distinct replica.
    values = [10, 20, 30, 40]
    rank_values = [value for value in values for _ in range(4)]
    assert sum(rank_values[rank] for rank in topology.group_ranks(11, "data")) == 100
    assert sum(rank_values) == 400


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_disabled_parallelism_assigns_every_rank_its_own_samples(world_size: int) -> None:
    topology = ParallelTopology.resolve(world_size)
    for rank in range(world_size):
        assert topology.data_parallel_rank(rank) == rank
        assert topology.group_ranks(rank, "data") == tuple(range(world_size))
        for scope in ("sample", "tensor", "context"):
            assert topology.group_ranks(rank, scope) == (rank,)


def test_omitted_shard_degree_is_the_only_inferred_dimension() -> None:
    topology = ParallelTopology.resolve(32, dp_replicate_size=2, tp_size=2, cp_size=2)
    assert topology.dimensions == {
        "dp_replicate_size": 2,
        "dp_shard_size": 4,
        "tp_size": 2,
        "cp_size": 2,
    }
    assert topology == ParallelTopology.resolve(32, **topology.dimensions)
    with pytest.raises(FrozenInstanceError):
        topology.tp_size = 4


@pytest.mark.parametrize("value", [0, -2, True, "2", 2.0])
@pytest.mark.parametrize("dimension", ["dp_replicate_size", "dp_shard_size", "tp_size", "cp_size"])
def test_invalid_degree_is_rejected_before_any_backend(value: object, dimension: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ParallelTopology.resolve(8, **{dimension: value})


@pytest.mark.parametrize("world_size", [0, -1, True, 4.0])
def test_invalid_world_size_is_rejected(world_size: object) -> None:
    with pytest.raises(ValueError, match="world_size must be a positive integer"):
        ParallelTopology.resolve(world_size)


def test_explicit_product_and_implicit_remainder_are_both_validated() -> None:
    with pytest.raises(ValueError, match="product 4, but the launched world size is 8"):
        ParallelTopology.resolve(8, dp_shard_size=2, tp_size=2)
    with pytest.raises(ValueError, match="must divide the launched world size 6"):
        ParallelTopology.resolve(6, cp_size=2, tp_size=2)


@pytest.mark.parametrize("rank", [-1, 4, True, 1.0])
def test_invalid_rank_cannot_select_another_samples_group(rank: object) -> None:
    with pytest.raises(ValueError, match="expected rank"):
        ParallelTopology(tp_size=2, cp_size=2).group_ranks(rank, "sample")


def test_unknown_scope_does_not_fall_back_to_the_world() -> None:
    with pytest.raises(ValueError, match="unknown collective scope"):
        ParallelTopology().group_ranks(0, "optimizer")
