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

"""Batch samplers backed by one topology-neutral :mod:`sampling_plan`."""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Sized, cast

from torch.utils.data import Dataset, Sampler

from .sampling_plan import SampleAssignment, SamplingPlan, build_sampling_plan


def _dataset_size(dataset: Dataset) -> int:
    if not hasattr(dataset, "__len__"):
        raise TypeError(f"Sampler requires dataset with __len__, got {type(dataset).__name__}.")
    return len(cast(Sized, dataset))


class PlannedGroupSampler(Sampler):
    """Common infinite batch-sampler facade over deterministic epoch plans."""

    layout_name: str

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int,
        unique_sample_num: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        subgroup_size: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = group_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.m = unique_sample_num
        self.subgroup_size = subgroup_size
        self.epoch = 0

        if type(rank) is not int or not 0 <= rank < num_replicas:
            raise ValueError(f"expected rank in [0, {num_replicas}), got {rank!r}")
        # Construct once so every geometry error is reported at initialization,
        # before DataLoader workers or model allocation begin.
        self._initial_plan: Optional[SamplingPlan] = self._build_plan(epoch=0)
        self.num_batches_per_epoch = self._initial_plan.num_batches_per_epoch

    def _build_plan(self, *, epoch: int) -> SamplingPlan:
        return build_sampling_plan(
            layout_name=self.layout_name,
            dataset_size=_dataset_size(self.dataset),
            per_device_batch_size=self.batch_size,
            group_size=self.k,
            unique_group_count=self.m,
            num_replicas=self.num_replicas,
            seed=self.seed,
            epoch=epoch,
            subgroup_size=self.subgroup_size,
        )

    def __iter__(self) -> Iterator[List[SampleAssignment]]:
        while True:
            if self._initial_plan is not None and self._initial_plan.epoch == self.epoch:
                plan = self._initial_plan
                self._initial_plan = None
            else:
                plan = self._build_plan(epoch=self.epoch)
            for batch in plan.batches_for_rank(self.rank):
                yield list(batch)
            self.epoch += 1

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError(f"expected epoch to be a non-negative integer, got {epoch!r}")
        self.epoch = epoch


class DistributedKRepeatSampler(PlannedGroupSampler):
    """Globally shuffle all ``U * K`` members before rank sharding."""

    layout_name = "global_random"


class ContiguousShardSampler(PlannedGroupSampler):
    """Flatten group-major samples, then give every rank one contiguous shard."""

    layout_name = "contiguous_shard"


class GroupContiguousSampler(PlannedGroupSampler):
    """Keep every complete group on exactly one rank (``rank_local``)."""

    layout_name = "rank_local"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.groups_per_rank = self.m // self.num_replicas


class GroupDistributedSampler(PlannedGroupSampler):
    """Close complete groups in every synchronized global microbatch."""

    layout_name = "global_batch"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._equal_share_layout = self.k % self.num_replicas == 0
        self.copies_per_rank = self.k // self.num_replicas if self._equal_share_layout else None


class GroupTiledSampler(PlannedGroupSampler):
    """Close groups in the smallest synchronized all-rank batch window."""

    layout_name = "global_tile"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        global_batch_size = self.num_replicas * self.batch_size
        divisor = math.gcd(global_batch_size, self.k)
        self.global_batches_per_tile = self.k // divisor
        self.groups_per_tile = global_batch_size // divisor


class SubgroupTiledSampler(PlannedGroupSampler):
    """Close groups independently in fixed contiguous rank subgroups."""

    layout_name = "subgroup_tile"

    def __init__(self, *args, subgroup_size: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, subgroup_size=subgroup_size, **kwargs)
        if self.subgroup_size is None:  # pragma: no cover - planner fails first
            raise RuntimeError("subgroup_tile did not resolve subgroup_size")
        subgroup_batch_size = self.subgroup_size * self.batch_size
        divisor = math.gcd(subgroup_batch_size, self.k)
        self.global_batches_per_tile = self.k // divisor
        self.groups_per_subgroup_tile = subgroup_batch_size // divisor
        self.num_subgroups = self.num_replicas // self.subgroup_size
        self.groups_per_tile = self.groups_per_subgroup_tile * self.num_subgroups


# Semantic public names. The legacy names above stay import-compatible.
GlobalRandomSampler = DistributedKRepeatSampler
RankLocalSampler = GroupContiguousSampler
GlobalBatchSampler = GroupDistributedSampler
GlobalTiledSampler = GroupTiledSampler


__all__ = [
    "ContiguousShardSampler",
    "DistributedKRepeatSampler",
    "GlobalBatchSampler",
    "GlobalRandomSampler",
    "GlobalTiledSampler",
    "GroupContiguousSampler",
    "GroupDistributedSampler",
    "GroupTiledSampler",
    "PlannedGroupSampler",
    "RankLocalSampler",
    "SubgroupTiledSampler",
]
