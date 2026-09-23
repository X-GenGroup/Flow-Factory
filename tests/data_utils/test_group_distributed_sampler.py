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

from flow_factory.data_utils.sampler import GroupDistributedSampler


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
