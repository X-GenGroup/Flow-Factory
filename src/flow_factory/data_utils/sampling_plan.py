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

"""Deterministic, topology-aware plans for grouped rollout sampling."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..contracts.sampler import CanonicalSamplerLayout, get_sampler_layout_contract

SAMPLING_GROUP_ID_COLUMN = "__sampling_group_id__"
SAMPLING_GROUP_MEMBER_ID_COLUMN = "__sampling_group_member_id__"
SAMPLING_SAMPLE_ID_COLUMN = "__sampling_sample_id__"
SAMPLING_IDENTITY_COLUMNS = frozenset(
    {
        SAMPLING_GROUP_ID_COLUMN,
        SAMPLING_GROUP_MEMBER_ID_COLUMN,
        SAMPLING_SAMPLE_ID_COLUMN,
    }
)


class SampleAssignment(int):
    """Dataset index carrying its sampling-time identity.

    Subclassing :class:`int` preserves the public batch-sampler API: ordinary
    datasets and tests can still use an assignment as an integer index. A
    :class:`SamplingDatasetView` unwraps the attached identity for production
    dataloaders.
    """

    def __new__(
        cls,
        dataset_index: int,
        group_id: int,
        group_member_id: int,
        sample_id: int,
    ) -> "SampleAssignment":
        obj = int.__new__(cls, dataset_index)
        obj.group_id = group_id
        obj.group_member_id = group_member_id
        obj.sample_id = sample_id
        return obj

    def __reduce__(self):
        """Preserve identity when DataLoader workers pickle indices."""

        return (
            type(self),
            (int(self), self.group_id, self.group_member_id, self.sample_id),
        )


class SamplingDatasetView(Dataset):
    """Inject a :class:`SampleAssignment` into one dataset row.

    The wrapper keeps sampling identity out of persistent preprocessing caches
    and transports it through ordinary DataLoader workers as reserved columns.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        assignment = index if isinstance(index, SampleAssignment) else None
        row = self.dataset[int(index)]
        if assignment is None:
            return row
        if not isinstance(row, Mapping):
            raise TypeError(
                "sampling identity requires mapping dataset rows, " f"got {type(row).__name__}"
            )
        collisions = SAMPLING_IDENTITY_COLUMNS.intersection(row)
        if collisions:
            raise ValueError(
                "dataset rows cannot define reserved sampling identity columns: "
                f"{sorted(collisions)!r}"
            )
        return {
            **row,
            SAMPLING_GROUP_ID_COLUMN: assignment.group_id,
            SAMPLING_GROUP_MEMBER_ID_COLUMN: assignment.group_member_id,
            SAMPLING_SAMPLE_ID_COLUMN: assignment.sample_id,
        }


@dataclass(frozen=True)
class SamplingPlan:
    """One complete, immutable acquisition plan for every data-parallel rank."""

    layout: CanonicalSamplerLayout
    epoch: int
    unique_group_count: int
    group_size: int
    num_replicas: int
    per_device_batch_size: int
    subgroup_size: Optional[int]
    group_window_batches: Optional[int]
    groups_per_window: int
    rank_batches: Tuple[Tuple[Tuple[SampleAssignment, ...], ...], ...]

    @property
    def num_batches_per_epoch(self) -> int:
        """Return the synchronized rank-local batch count."""

        return len(self.rank_batches[0])

    def batches_for_rank(self, rank: int) -> Tuple[Tuple[SampleAssignment, ...], ...]:
        """Return one rank's batches after validating its index."""

        if type(rank) is not int or not 0 <= rank < self.num_replicas:
            raise ValueError(f"expected rank in [0, {self.num_replicas}), got {rank!r}")
        return self.rank_batches[rank]


def build_sampling_plan(
    *,
    layout_name: str,
    dataset_size: int,
    per_device_batch_size: int,
    group_size: int,
    unique_group_count: int,
    num_replicas: int,
    seed: int,
    epoch: int,
    subgroup_size: Optional[int] = None,
) -> SamplingPlan:
    """Build a deterministic grouped sampling plan.

    Group and member identities are assigned before topology placement. They
    therefore remain exact even for random sharding, repeated prompt text, or
    two dataset rows with identical content.
    """

    for value, label in (
        (dataset_size, "dataset_size"),
        (per_device_batch_size, "per_device_batch_size"),
        (group_size, "group_size"),
        (unique_group_count, "unique_group_count"),
        (num_replicas, "num_replicas"),
    ):
        _require_positive_int(value, label)
    if type(seed) is not int:
        raise ValueError(f"expected seed to be an integer, got {seed!r}")
    if type(epoch) is not int or epoch < 0:
        raise ValueError(f"expected epoch to be a non-negative integer, got {epoch!r}")
    if unique_group_count > dataset_size:
        raise ValueError(
            f"unique_group_count ({unique_group_count}) must be <= dataset size "
            f"({dataset_size})"
        )

    contract = get_sampler_layout_contract(layout_name)
    canonical_layout = contract.name
    resolved_subgroup_size = None
    if canonical_layout == "subgroup_tile":
        resolved_subgroup_size = contract.resolve_subgroup_size(
            num_replicas=num_replicas,
            subgroup_size=subgroup_size,
        )
    elif subgroup_size is not None:
        raise ValueError(
            f"sampler layout {canonical_layout!r} does not accept sampler_subgroup_size"
        )

    groups_per_window = contract.unique_groups_per_window(
        num_replicas=num_replicas,
        per_device_batch_size=per_device_batch_size,
        group_size=group_size,
        subgroup_size=resolved_subgroup_size,
    )
    if unique_group_count % groups_per_window:
        raise ValueError(
            f"sampler layout {canonical_layout!r} requires unique_group_count to be a "
            f"multiple of {groups_per_window}; got {unique_group_count}"
        )

    global_batch_size = num_replicas * per_device_batch_size
    if canonical_layout == "global_batch" and global_batch_size % group_size:
        raise ValueError(
            "global_batch requires group_size to divide the global microbatch: "
            f"global_batch_size={global_batch_size}, group_size={group_size}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    selected_indices = torch.randperm(dataset_size, generator=generator)[
        :unique_group_count
    ].tolist()
    groups = tuple(
        _make_group(
            dataset_index=dataset_index,
            group_id=epoch * unique_group_count + group_slot,
            group_size=group_size,
        )
        for group_slot, dataset_index in enumerate(selected_indices)
    )

    if canonical_layout == "global_random":
        assignments = [assignment for group in groups for assignment in group]
        order = torch.randperm(len(assignments), generator=generator).tolist()
        rank_batches = _deal_global_batches(
            tuple(assignments[index] for index in order),
            num_replicas=num_replicas,
            per_device_batch_size=per_device_batch_size,
        )
    else:
        group_order = torch.randperm(unique_group_count, generator=generator).tolist()
        ordered_groups = tuple(groups[index] for index in group_order)
        if canonical_layout == "contiguous_shard":
            rank_batches = _build_contiguous_shards(
                ordered_groups,
                num_replicas=num_replicas,
                per_device_batch_size=per_device_batch_size,
            )
        elif canonical_layout == "rank_local":
            rank_batches = _build_rank_local(
                ordered_groups,
                num_replicas=num_replicas,
                per_device_batch_size=per_device_batch_size,
            )
        elif canonical_layout == "global_batch":
            rank_batches = _build_global_batch(
                ordered_groups,
                group_size=group_size,
                num_replicas=num_replicas,
                per_device_batch_size=per_device_batch_size,
            )
        elif canonical_layout == "global_tile":
            rank_batches = _deal_global_batches(
                tuple(assignment for group in ordered_groups for assignment in group),
                num_replicas=num_replicas,
                per_device_batch_size=per_device_batch_size,
            )
        elif canonical_layout == "subgroup_tile":
            if resolved_subgroup_size is None:  # pragma: no cover - validated above
                raise RuntimeError("subgroup_tile did not resolve its subgroup size")
            rank_batches = _build_subgroup_tiles(
                ordered_groups,
                group_size=group_size,
                num_replicas=num_replicas,
                subgroup_size=resolved_subgroup_size,
                per_device_batch_size=per_device_batch_size,
            )
        else:  # pragma: no cover - exhaustive over CanonicalSamplerLayout
            raise RuntimeError(f"unhandled sampler layout {canonical_layout!r}")

    group_window_batches = None
    if contract.supports_incremental_group_feedback:
        group_window_batches = contract.global_group_window_batches(
            num_replicas=num_replicas,
            per_device_batch_size=per_device_batch_size,
            group_size=group_size,
            subgroup_size=resolved_subgroup_size,
        )
    plan = SamplingPlan(
        layout=canonical_layout,
        epoch=epoch,
        unique_group_count=unique_group_count,
        group_size=group_size,
        num_replicas=num_replicas,
        per_device_batch_size=per_device_batch_size,
        subgroup_size=resolved_subgroup_size,
        group_window_batches=group_window_batches,
        groups_per_window=groups_per_window,
        rank_batches=rank_batches,
    )
    _validate_plan(plan)
    return plan


def _make_group(
    *,
    dataset_index: int,
    group_id: int,
    group_size: int,
) -> Tuple[SampleAssignment, ...]:
    return tuple(
        SampleAssignment(
            dataset_index,
            group_id=group_id,
            group_member_id=member_id,
            sample_id=group_id * group_size + member_id,
        )
        for member_id in range(group_size)
    )


def _chunk_batches(
    assignments: Tuple[SampleAssignment, ...],
    batch_size: int,
) -> Tuple[Tuple[SampleAssignment, ...], ...]:
    if len(assignments) % batch_size:
        raise ValueError(
            f"rank-local samples ({len(assignments)}) must divide batch_size ({batch_size})"
        )
    return tuple(
        assignments[start : start + batch_size] for start in range(0, len(assignments), batch_size)
    )


def _deal_global_batches(
    assignments: Tuple[SampleAssignment, ...],
    *,
    num_replicas: int,
    per_device_batch_size: int,
) -> Tuple[Tuple[Tuple[SampleAssignment, ...], ...], ...]:
    global_batch_size = num_replicas * per_device_batch_size
    if len(assignments) % global_batch_size:
        raise ValueError(
            f"global samples ({len(assignments)}) must divide global_batch_size "
            f"({global_batch_size})"
        )
    rank_batches = [[] for _ in range(num_replicas)]
    for global_start in range(0, len(assignments), global_batch_size):
        for rank in range(num_replicas):
            start = global_start + rank * per_device_batch_size
            rank_batches[rank].append(assignments[start : start + per_device_batch_size])
    return tuple(tuple(batches) for batches in rank_batches)


def _build_contiguous_shards(
    groups: Tuple[Tuple[SampleAssignment, ...], ...],
    *,
    num_replicas: int,
    per_device_batch_size: int,
) -> Tuple[Tuple[Tuple[SampleAssignment, ...], ...], ...]:
    assignments = tuple(assignment for group in groups for assignment in group)
    samples_per_rank = len(assignments) // num_replicas
    return tuple(
        _chunk_batches(
            assignments[rank * samples_per_rank : (rank + 1) * samples_per_rank],
            per_device_batch_size,
        )
        for rank in range(num_replicas)
    )


def _build_rank_local(
    groups: Tuple[Tuple[SampleAssignment, ...], ...],
    *,
    num_replicas: int,
    per_device_batch_size: int,
) -> Tuple[Tuple[Tuple[SampleAssignment, ...], ...], ...]:
    groups_per_rank = len(groups) // num_replicas
    return tuple(
        _chunk_batches(
            tuple(
                assignment
                for group in groups[rank * groups_per_rank : (rank + 1) * groups_per_rank]
                for assignment in group
            ),
            per_device_batch_size,
        )
        for rank in range(num_replicas)
    )


def _build_global_batch(
    groups: Tuple[Tuple[SampleAssignment, ...], ...],
    *,
    group_size: int,
    num_replicas: int,
    per_device_batch_size: int,
) -> Tuple[Tuple[Tuple[SampleAssignment, ...], ...], ...]:
    if group_size % num_replicas:
        return _deal_global_batches(
            tuple(assignment for group in groups for assignment in group),
            num_replicas=num_replicas,
            per_device_batch_size=per_device_batch_size,
        )

    # Preserve the historic striped/equal-share policy when every group can
    # contribute the same number of members to every rank.
    copies_per_rank = group_size // num_replicas
    return tuple(
        _chunk_batches(
            tuple(
                assignment
                for group in groups
                for assignment in group[rank * copies_per_rank : (rank + 1) * copies_per_rank]
            ),
            per_device_batch_size,
        )
        for rank in range(num_replicas)
    )


def _build_subgroup_tiles(
    groups: Tuple[Tuple[SampleAssignment, ...], ...],
    *,
    group_size: int,
    num_replicas: int,
    subgroup_size: int,
    per_device_batch_size: int,
) -> Tuple[Tuple[Tuple[SampleAssignment, ...], ...], ...]:
    subgroup_batch_size = subgroup_size * per_device_batch_size
    divisor = math.gcd(subgroup_batch_size, group_size)
    batches_per_window = group_size // divisor
    groups_per_subgroup_window = subgroup_batch_size // divisor
    subgroup_count = num_replicas // subgroup_size
    groups_per_window = subgroup_count * groups_per_subgroup_window
    rank_batches = [[] for _ in range(num_replicas)]

    for window_start in range(0, len(groups), groups_per_window):
        window_groups = groups[window_start : window_start + groups_per_window]
        subgroup_samples = []
        for subgroup_index in range(subgroup_count):
            start = subgroup_index * groups_per_subgroup_window
            subgroup_samples.append(
                tuple(
                    assignment
                    for group in window_groups[start : start + groups_per_subgroup_window]
                    for assignment in group
                )
            )
        for batch_index in range(batches_per_window):
            for subgroup_index, assignments in enumerate(subgroup_samples):
                batch_start = batch_index * subgroup_batch_size
                for subgroup_rank in range(subgroup_size):
                    rank = subgroup_index * subgroup_size + subgroup_rank
                    start = batch_start + subgroup_rank * per_device_batch_size
                    rank_batches[rank].append(assignments[start : start + per_device_batch_size])
    return tuple(tuple(batches) for batches in rank_batches)


def _validate_plan(plan: SamplingPlan) -> None:
    expected_batches = (
        plan.unique_group_count
        * plan.group_size
        // (plan.num_replicas * plan.per_device_batch_size)
    )
    if any(len(batches) != expected_batches for batches in plan.rank_batches):
        raise RuntimeError("sampling plan produced different batch counts across ranks")
    assignments = [
        assignment for batches in plan.rank_batches for batch in batches for assignment in batch
    ]
    if any(
        len(batch) != plan.per_device_batch_size
        for batches in plan.rank_batches
        for batch in batches
    ):
        raise RuntimeError("sampling plan produced an incomplete rank-local microbatch")
    group_counts = Counter(assignment.group_id for assignment in assignments)
    if len(group_counts) != plan.unique_group_count or set(group_counts.values()) != {
        plan.group_size
    }:
        raise RuntimeError(
            "sampling plan did not place every group exactly K times: "
            f"counts={dict(group_counts)!r}"
        )
    if len({assignment.sample_id for assignment in assignments}) != len(assignments):
        raise RuntimeError("sampling plan produced duplicate sample identities")


def _require_positive_int(value: object, label: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"expected {label} to be a positive integer, got {value!r}")


__all__ = [
    "SAMPLING_GROUP_ID_COLUMN",
    "SAMPLING_GROUP_MEMBER_ID_COLUMN",
    "SAMPLING_IDENTITY_COLUMNS",
    "SAMPLING_SAMPLE_ID_COLUMN",
    "SampleAssignment",
    "SamplingDatasetView",
    "SamplingPlan",
    "build_sampling_plan",
]
