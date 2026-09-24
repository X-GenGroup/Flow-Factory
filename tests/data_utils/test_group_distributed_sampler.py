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

"""Global-microbatch invariants for group-distributed rollout sampling."""

from collections import Counter

import pytest

from flow_factory.contracts import SAMPLER_LAYOUT_CONTRACTS, get_sampler_layout_contract
from flow_factory.data_utils.sampler import GroupDistributedSampler, GroupTiledSampler
from flow_factory.data_utils.sampler_loader import SAMPLER_REGISTRY


def test_sampler_registry_and_layout_contracts_have_the_same_keys() -> None:
    assert set(SAMPLER_REGISTRY) == set(SAMPLER_LAYOUT_CONTRACTS)


@pytest.mark.parametrize(
    ("sampler_name", "world_size", "batch_size", "group_size", "expected_batches"),
    [
        ("group_contiguous", 6, 2, 3, 3),
        ("group_distributed", 6, 1, 3, 1),
        ("group_tiled", 6, 1, 4, 2),
    ],
)
def test_sampler_contract_reports_smallest_group_complete_window(
    sampler_name: str,
    world_size: int,
    batch_size: int,
    group_size: int,
    expected_batches: int,
) -> None:
    contract = get_sampler_layout_contract(sampler_name)

    assert (
        contract.global_group_window_batches(
            num_replicas=world_size,
            per_device_batch_size=batch_size,
            group_size=group_size,
        )
        == expected_batches
    )


def test_arbitrary_sampler_has_no_bounded_group_complete_window() -> None:
    contract = get_sampler_layout_contract("distributed_k_repeat")

    with pytest.raises(ValueError, match="no bounded group-complete window"):
        contract.global_group_window_batches(
            num_replicas=4,
            per_device_batch_size=1,
            group_size=8,
        )


def test_group_distributed_packs_k16_groups_across_32_single_sample_ranks() -> None:
    world_size = 32
    group_size = 16
    unique_sample_num = 48
    samplers = [
        GroupDistributedSampler(
            dataset=list(range(128)),
            batch_size=1,
            group_size=group_size,
            unique_sample_num=unique_sample_num,
            num_replicas=world_size,
            rank=rank,
            seed=7,
        )
        for rank in range(world_size)
    ]
    iterators = [iter(sampler) for sampler in samplers]

    assert {sampler.num_batches_per_epoch for sampler in samplers} == {24}
    observed = Counter()
    for _ in range(24):
        global_microbatch = [next(iterator)[0] for iterator in iterators]
        counts = Counter(global_microbatch)
        assert sorted(counts.values()) == [group_size, group_size]
        observed.update(counts)

    assert len(observed) == unique_sample_num
    assert set(observed.values()) == {group_size}


def test_group_distributed_preserves_equal_share_layout_when_available() -> None:
    samplers = [
        GroupDistributedSampler(
            dataset=list(range(32)),
            batch_size=2,
            group_size=4,
            unique_sample_num=8,
            num_replicas=2,
            rank=rank,
            seed=11,
        )
        for rank in range(2)
    ]

    first_batches = [next(iter(sampler)) for sampler in samplers]
    assert first_batches[0] == first_batches[1]


def test_group_tiled_closes_groups_across_multiple_global_batches() -> None:
    world_size = 6
    group_size = 4
    samplers = [
        GroupTiledSampler(
            dataset=list(range(32)),
            batch_size=1,
            group_size=group_size,
            unique_sample_num=6,
            num_replicas=world_size,
            rank=rank,
            seed=13,
        )
        for rank in range(world_size)
    ]
    iterators = [iter(sampler) for sampler in samplers]

    assert {sampler.global_batches_per_tile for sampler in samplers} == {2}
    assert {sampler.groups_per_tile for sampler in samplers} == {3}
    observed = Counter()
    for _tile in range(2):
        tile_samples = []
        for _batch in range(2):
            tile_samples.extend(next(iterator)[0] for iterator in iterators)
        counts = Counter(tile_samples)
        assert sorted(counts.values()) == [group_size] * 3
        observed.update(counts)

    assert len(observed) == 6
    assert set(observed.values()) == {group_size}


def test_group_tiled_supports_groups_larger_than_a_global_batch() -> None:
    samplers = [
        GroupTiledSampler(
            dataset=list(range(8)),
            batch_size=1,
            group_size=8,
            unique_sample_num=2,
            num_replicas=4,
            rank=rank,
            seed=3,
        )
        for rank in range(4)
    ]
    iterators = [iter(sampler) for sampler in samplers]

    for _tile in range(2):
        tile_samples = [next(iterator)[0] for _batch in range(2) for iterator in iterators]
        assert list(Counter(tile_samples).values()) == [8]
