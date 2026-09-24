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

"""Stable identities shared by sampling, reward, and optimization stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence, Tuple

if TYPE_CHECKING:
    from .samples import BaseSample


LEGACY_SOURCE_ID = -1


@dataclass(frozen=True, order=True)
class GroupKey:
    """Identity of one comparison group across datasets and ranks.

    ``unique_id`` identifies the conditioning input while ``source_id`` keeps
    equal prompts from independently configured datasets in separate groups.
    Legacy single-source samples use :data:`LEGACY_SOURCE_ID`.
    """

    source_id: int
    unique_id: int

    @classmethod
    def from_sample(cls, sample: "BaseSample") -> "GroupKey":
        """Build the canonical group key for a sample.

        Args:
            sample: Generated sample carrying prompt and source identity.

        Returns:
            Source-aware comparison-group key.
        """
        raw_source_id = getattr(sample, "source_id", None)
        source_id = LEGACY_SOURCE_ID if raw_source_id is None else int(raw_source_id)
        return cls(source_id=source_id, unique_id=int(sample.unique_id))


def sample_group_key(sample: "BaseSample") -> GroupKey:
    """Return the canonical comparison-group identity for a sample.

    Args:
        sample: Generated sample to identify.

    Returns:
        Source-aware comparison-group key.
    """

    return GroupKey.from_sample(sample)


def group_identity_rows(samples: Iterable["BaseSample"]) -> Tuple[Tuple[int, int], ...]:
    """Return canonical identity rows for tensor packing.

    Args:
        samples: Generated samples to identify.

    Returns:
        Exact integer ``(source_id, unique_id)`` rows in input order.
    """

    return tuple(
        (group_key.source_id, group_key.unique_id)
        for group_key in (sample_group_key(sample) for sample in samples)
    )


@dataclass(frozen=True)
class AcquisitionManifest:
    """Immutable rank-local identity and rollout-batch manifest.

    Scheduling may reorder complete optimization work units, but it must not
    mutate sample identity or split a recorded rollout batch for adapters whose
    forward pass depends on packed batch composition.
    """

    sample_object_ids: Tuple[int, ...]
    group_keys: Tuple[GroupKey, ...]
    rollout_batches: Tuple[Tuple[int, ...], ...] = ()

    @classmethod
    def from_samples(
        cls,
        samples: Sequence["BaseSample"],
        *,
        rollout_batch_object_ids: Iterable[Sequence[int]] = (),
    ) -> "AcquisitionManifest":
        """Capture immutable identity and rollout-batch boundaries.

        Args:
            samples: Rank-local acquisition in rollout order.
            rollout_batch_object_ids: Object IDs for every original rollout batch.

        Returns:
            Validated immutable acquisition manifest.

        Raises:
            ValueError: If object identity is duplicated or batches do not partition samples.
        """
        object_ids = tuple(id(sample) for sample in samples)
        object_id_to_index = {object_id: index for index, object_id in enumerate(object_ids)}
        if len(object_id_to_index) != len(object_ids):
            raise ValueError("an acquisition cannot contain the same sample object more than once")

        rollout_batches = []
        seen_indices = []
        for batch in rollout_batch_object_ids:
            batch_indices = []
            for object_id in batch:
                if object_id not in object_id_to_index:
                    raise ValueError(
                        "rollout batch references a sample outside the acquisition: "
                        f"object_id={object_id}"
                    )
                batch_indices.append(object_id_to_index[object_id])
            if not batch_indices:
                raise ValueError("rollout batches cannot be empty")
            rollout_batches.append(tuple(batch_indices))
            seen_indices.extend(batch_indices)
        if rollout_batches and tuple(seen_indices) != tuple(range(len(samples))):
            raise ValueError(
                "rollout batches must partition the acquisition in sample order: "
                f"received={tuple(seen_indices)!r}"
            )

        return cls(
            sample_object_ids=object_ids,
            group_keys=tuple(sample_group_key(sample) for sample in samples),
            rollout_batches=tuple(rollout_batches),
        )

    def validate_samples(self, samples: Sequence["BaseSample"]) -> None:
        """Require an unchanged sample sequence and group identity.

        Args:
            samples: Current rank-local acquisition sequence.

        Raises:
            ValueError: If sample objects, order, or group identities changed.
        """

        object_ids = tuple(id(sample) for sample in samples)
        if object_ids != self.sample_object_ids:
            raise ValueError("acquisition samples changed after its manifest was created")
        group_keys = tuple(sample_group_key(sample) for sample in samples)
        if group_keys != self.group_keys:
            raise ValueError("acquisition group identities changed after manifest creation")

    @property
    def rollout_boundaries(self) -> frozenset[int]:
        """Return every boundary between recorded rollout batches."""

        boundaries = {0}
        for batch in self.rollout_batches:
            boundaries.add(batch[-1] + 1)
        return frozenset(boundaries)


__all__ = [
    "AcquisitionManifest",
    "GroupKey",
    "LEGACY_SOURCE_ID",
    "group_identity_rows",
    "sample_group_key",
]
