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

import pickle
from collections import Counter

import pytest
from torch.utils.data import DataLoader

from flow_factory.contracts import SAMPLER_LAYOUT_CONTRACTS, get_sampler_layout_contract
from flow_factory.data_utils.sampler import (
    ContiguousShardSampler,
    GroupDistributedSampler,
    GroupTiledSampler,
    SubgroupTiledSampler,
)
from flow_factory.data_utils.sampler_loader import SAMPLER_REGISTRY
from flow_factory.data_utils.sampling_plan import (
    SAMPLING_GROUP_ID_COLUMN,
    SAMPLING_GROUP_MEMBER_ID_COLUMN,
    SAMPLING_SAMPLE_ID_COLUMN,
    SamplingDatasetView,
    build_sampling_plan,
)


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


def test_semantic_sampler_names_resolve_to_the_legacy_contracts() -> None:
    aliases = {
        "distributed_k_repeat": "global_random",
        "group_contiguous": "rank_local",
        "group_distributed": "global_batch",
        "group_tiled": "global_tile",
    }

    for alias, canonical in aliases.items():
        assert get_sampler_layout_contract(alias) is get_sampler_layout_contract(canonical)


def test_contiguous_shard_flattens_groups_before_rank_partitioning() -> None:
    samplers = [
        ContiguousShardSampler(
            dataset=list(range(16)),
            batch_size=2,
            group_size=3,
            unique_sample_num=4,
            num_replicas=3,
            rank=rank,
            seed=19,
        )
        for rank in range(3)
    ]

    rank_group_ids = [
        [
            assignment.group_id
            for batch in sampler._initial_plan.batches_for_rank(rank)
            for assignment in batch
        ]
        for rank, sampler in enumerate(samplers)
    ]
    flattened = [group_id for shard in rank_group_ids for group_id in shard]
    assert sorted(Counter(flattened).values()) == [3, 3, 3, 3]
    assert sum(flattened[index] != flattened[index - 1] for index in range(1, len(flattened))) == 3


def test_subgroup_tile_closes_groups_inside_each_rank_subgroup() -> None:
    world_size = 8
    subgroup_size = 2
    samplers = [
        SubgroupTiledSampler(
            dataset=list(range(32)),
            batch_size=1,
            group_size=4,
            unique_sample_num=8,
            num_replicas=world_size,
            subgroup_size=subgroup_size,
            rank=rank,
            seed=23,
        )
        for rank in range(world_size)
    ]
    iterators = [iter(sampler) for sampler in samplers]

    assert {sampler.global_batches_per_tile for sampler in samplers} == {2}
    assert {sampler.groups_per_subgroup_tile for sampler in samplers} == {1}
    for _window in range(2):
        subgroup_members = [[] for _ in range(world_size // subgroup_size)]
        for _batch in range(2):
            for rank, iterator in enumerate(iterators):
                subgroup_members[rank // subgroup_size].extend(next(iterator))
        for members in subgroup_members:
            assert list(Counter(member.group_id for member in members).values()) == [4]


def test_sampling_plan_assigns_content_independent_exact_identities() -> None:
    plan = build_sampling_plan(
        layout_name="global_random",
        dataset_size=32,
        per_device_batch_size=2,
        group_size=3,
        unique_group_count=8,
        num_replicas=4,
        seed=29,
        epoch=2,
    )
    assignments = [
        assignment
        for rank_batches in plan.rank_batches
        for batch in rank_batches
        for assignment in batch
    ]

    assert set(Counter(a.group_id for a in assignments).values()) == {3}
    assert len({a.sample_id for a in assignments}) == len(assignments)
    assert {
        tuple(sorted(a.group_member_id for a in assignments if a.group_id == group_id))
        for group_id in {a.group_id for a in assignments}
    } == {(0, 1, 2)}


def test_sampling_dataset_view_transports_assignment_metadata() -> None:
    view = SamplingDatasetView([{"prompt": "same"}, {"prompt": "same"}])
    plan = build_sampling_plan(
        layout_name="rank_local",
        dataset_size=2,
        per_device_batch_size=2,
        group_size=2,
        unique_group_count=2,
        num_replicas=1,
        seed=31,
        epoch=0,
    )
    first, second = plan.rank_batches[0][0]

    row = view[first]
    assert row[SAMPLING_GROUP_ID_COLUMN] == first.group_id
    assert row[SAMPLING_GROUP_MEMBER_ID_COLUMN] == first.group_member_id
    assert row[SAMPLING_SAMPLE_ID_COLUMN] == first.sample_id
    assert int(first) == int(second)


def test_sampling_identity_survives_pickle_and_dataloader_workers() -> None:
    dataset = [{"prompt": f"prompt-{index}"} for index in range(8)]
    sampler = GroupTiledSampler(
        dataset=dataset,
        batch_size=2,
        group_size=2,
        unique_sample_num=4,
        num_replicas=1,
        rank=0,
        seed=37,
    )
    assignment = sampler._initial_plan.rank_batches[0][0][0]
    restored = pickle.loads(pickle.dumps(assignment))

    assert int(restored) == int(assignment)
    assert restored.group_id == assignment.group_id
    assert restored.group_member_id == assignment.group_member_id
    assert restored.sample_id == assignment.sample_id

    expected_group_ids = [member.group_id for member in sampler._initial_plan.rank_batches[0][0]]
    loader = DataLoader(
        SamplingDatasetView(dataset),
        batch_sampler=sampler,
        num_workers=1,
        multiprocessing_context="spawn",
    )
    batch = next(iter(loader))
    assert batch[SAMPLING_GROUP_ID_COLUMN].tolist() == expected_group_ids


def test_sampler_set_epoch_selects_a_disjoint_group_identity_namespace() -> None:
    sampler = GroupTiledSampler(
        dataset=list(range(16)),
        batch_size=2,
        group_size=2,
        unique_sample_num=4,
        num_replicas=1,
        rank=0,
        seed=41,
    )
    sampler.set_epoch(3)

    assignments = next(iter(sampler))

    assert {assignment.group_id // 4 for assignment in assignments} == {3}
