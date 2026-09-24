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

"""Pure optimizer work-unit geometry for streamed reward-driven training."""

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

from ..samples import AcquisitionManifest, BaseSample, GroupKey, sample_group_key

RewardGroupLayout = Literal["rank_local", "cross_rank_sharded", "cross_rank_tiled"]
RewardAccumulationScope = Literal["tile", "acquisition"]


@dataclass(frozen=True)
class RewardTileGeometry:
    """Describe how reward groups become optimizer work.

    ``optimizer_examples_per_group`` is rank-local.  Rank-local layouts can
    transform one K-sample reward group into a different number of optimizer
    examples (online DPO turns it into one preference pair).  Cross-rank
    layouts consume rank-local samples directly and recover complete groups
    through collectives, so the value must remain ``None`` there.
    """

    group_layout: RewardGroupLayout = "rank_local"
    optimizer_examples_per_group: Optional[int] = None
    optimizer_terms_per_batch: int = 1
    accumulation_scope: RewardAccumulationScope = "tile"

    def __post_init__(self) -> None:
        if self.group_layout not in {
            "rank_local",
            "cross_rank_sharded",
            "cross_rank_tiled",
        }:
            raise ValueError(f"unknown reward group layout: {self.group_layout!r}")
        if self.accumulation_scope not in {"tile", "acquisition"}:
            raise ValueError(f"unknown reward accumulation scope: {self.accumulation_scope!r}")
        _require_positive_int(self.optimizer_terms_per_batch, "optimizer_terms_per_batch")
        if self.group_layout == "rank_local":
            if self.optimizer_examples_per_group is not None:
                _require_positive_int(
                    self.optimizer_examples_per_group,
                    "optimizer_examples_per_group",
                )
        elif self.optimizer_examples_per_group is not None:
            raise ValueError(
                "cross-rank reward geometry consumes local samples directly and cannot "
                "declare optimizer_examples_per_group"
            )


@dataclass(frozen=True)
class RewardTile:
    """One rank-local, group-complete optimizer work unit.

    A work unit controls when advantages and optimizer steps become runnable;
    reward-model request batches remain independently sized and may span its
    boundaries.
    """

    tile_id: int
    start: int
    stop: int
    group_ids: Tuple[GroupKey, ...]

    @property
    def sample_indices(self) -> Tuple[int, ...]:
        """Return rank-local sample indices owned by this tile."""
        return tuple(range(self.start, self.stop))

    @property
    def sample_count(self) -> int:
        """Return the number of rank-local samples in this tile."""
        return self.stop - self.start


@dataclass(frozen=True)
class RewardTilePlan:
    """Immutable tiling of one rank-local generated acquisition."""

    sample_count: int
    samples_per_tile: int
    batches_per_tile: int
    geometry: RewardTileGeometry
    tiles: Tuple[RewardTile, ...]
    manifest: Optional[AcquisitionManifest] = None

    def samples_for(self, tile: RewardTile, samples: Sequence[BaseSample]) -> List[BaseSample]:
        """Slice samples after validating that they still match this plan.

        Args:
            tile: Tile owned by this plan.
            samples: Rank-local acquisition in original rollout order.

        Returns:
            Samples belonging to ``tile``.

        Raises:
            ValueError: If the acquisition changed or the tile belongs to another plan.
        """
        if len(samples) != self.sample_count:
            raise ValueError(
                "reward tile plan sample count changed after construction: "
                f"expected {self.sample_count}, received {len(samples)}"
            )
        if tile not in self.tiles:
            raise ValueError(f"reward tile {tile!r} does not belong to this plan")
        if self.manifest is not None:
            self.manifest.validate_samples(samples)
        return list(samples[tile.start : tile.stop])


def build_reward_tile_plan(
    samples: Sequence[BaseSample],
    *,
    group_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    optimizer_terms_per_batch: int,
    group_layout: RewardGroupLayout = "rank_local",
    optimizer_examples_per_group: Optional[int] = None,
    accumulation_scope: RewardAccumulationScope = "tile",
    manifest: Optional[AcquisitionManifest] = None,
    num_replicas: int = 1,
) -> RewardTilePlan:
    """Build tiles that close both reward groups and optimizer accumulation.

    A coupled replay batch enters gradient accumulation once per selected training
    timestep. The smallest accumulation-closing replay span is therefore
    ``GAS / gcd(GAS, optimizer_terms_per_batch)`` batches. Combining that span
    with ``group_size`` yields the smallest legal tile.

    Args:
        samples: Rank-local samples in rollout order.
        group_size: Number of repeated samples per prompt.
        per_device_batch_size: Rank-local replay microbatch size.
        gradient_accumulation_steps: Number of accumulation contexts per update.
        optimizer_terms_per_batch: Number of accumulation contexts entered for
            each replay microbatch, normally the number of train timesteps.
        group_layout: Whether complete reward groups are rank-local or sharded.
        optimizer_examples_per_group: Rank-local optimizer examples produced by
            one reward group, or ``None`` when samples are consumed directly.
        accumulation_scope: Whether each tile or the full acquisition closes
            gradient accumulation.
        manifest: Original acquisition identity and rollout-batch partition.
        num_replicas: Number of data-parallel ranks participating in the layout.

    Returns:
        A validated immutable tile plan.

    Raises:
        ValueError: If the acquisition cannot form complete groups and
            accumulation windows.
    """
    if not samples:
        raise ValueError("cannot build a reward tile plan for an empty acquisition")
    manifest = manifest or AcquisitionManifest.from_samples(samples)
    manifest.validate_samples(samples)

    samples_per_tile, batches_per_tile = resolve_reward_tile_size(
        group_size=group_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_terms_per_batch=optimizer_terms_per_batch,
        group_layout=group_layout,
        optimizer_examples_per_group=optimizer_examples_per_group,
        accumulation_scope=accumulation_scope,
        num_replicas=num_replicas,
    )
    geometry = RewardTileGeometry(
        group_layout=group_layout,
        optimizer_examples_per_group=optimizer_examples_per_group,
        optimizer_terms_per_batch=optimizer_terms_per_batch,
        accumulation_scope=accumulation_scope,
    )
    sample_count = len(samples)
    if sample_count % samples_per_tile != 0:
        raise ValueError(
            "rank-local acquisition cannot be tiled into complete reward groups and "
            "optimizer accumulation windows: "
            f"samples={sample_count}, group_size={group_size}, "
            f"per_device_batch_size={per_device_batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"optimizer_terms_per_batch={optimizer_terms_per_batch}, "
            f"required_tile_multiple={samples_per_tile}"
        )

    tiles = []
    if group_layout == "rank_local":
        group_ids = []
        for start in range(0, sample_count, group_size):
            group = samples[start : start + group_size]
            ids = tuple(sample_group_key(sample) for sample in group)
            if len(group) != group_size or len(set(ids)) != 1:
                raise ValueError(
                    "rank-local reward optimization overlap requires group_contiguous "
                    f"rollout order; sample range [{start}, {start + group_size}) has "
                    f"group ids={ids!r}"
                )
            group_ids.append(ids[0])
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(
                "reward optimization overlap requires one contiguous span per group identity; "
                f"received duplicate group ids={group_ids!r}"
            )

        groups_per_tile = samples_per_tile // group_size
        for tile_id, start in enumerate(range(0, sample_count, samples_per_tile)):
            group_start = start // group_size
            tiles.append(
                RewardTile(
                    tile_id=tile_id,
                    start=start,
                    stop=start + samples_per_tile,
                    group_ids=tuple(group_ids[group_start : group_start + groups_per_tile]),
                )
            )
    else:
        for tile_id, start in enumerate(range(0, sample_count, samples_per_tile)):
            stop = start + samples_per_tile
            tiles.append(
                RewardTile(
                    tile_id=tile_id,
                    start=start,
                    stop=stop,
                    group_ids=tuple(
                        sorted({sample_group_key(sample) for sample in samples[start:stop]})
                    ),
                )
            )
    return RewardTilePlan(
        sample_count=sample_count,
        samples_per_tile=samples_per_tile,
        batches_per_tile=batches_per_tile,
        geometry=geometry,
        tiles=tuple(tiles),
        manifest=manifest,
    )


def resolve_reward_tile_size(
    *,
    group_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    optimizer_terms_per_batch: int,
    group_layout: RewardGroupLayout = "rank_local",
    optimizer_examples_per_group: Optional[int] = None,
    accumulation_scope: RewardAccumulationScope = "tile",
    num_replicas: int = 1,
) -> Tuple[int, int]:
    """Resolve the smallest sample span that closes groups and accumulation.

    Args:
        group_size: Number of repeated samples per prompt.
        per_device_batch_size: Rank-local optimizer microbatch size.
        gradient_accumulation_steps: Number of accumulation contexts per update.
        optimizer_terms_per_batch: Accumulation contexts entered per optimizer batch.
        group_layout: Whether reward groups are rank-local or cross-rank sharded.
        optimizer_examples_per_group: Rank-local optimizer examples produced by
            one reward group.
        accumulation_scope: Whether accumulation closes per tile or acquisition.
        num_replicas: Number of data-parallel ranks participating in the layout.

    Returns:
        ``(samples_per_tile, batches_per_tile)`` for rank-local rollout order.

    Raises:
        ValueError: If any geometry value is invalid.
    """
    _require_positive_int(group_size, "group_size")
    _require_positive_int(per_device_batch_size, "per_device_batch_size")
    _require_positive_int(gradient_accumulation_steps, "gradient_accumulation_steps")
    _require_positive_int(optimizer_terms_per_batch, "optimizer_terms_per_batch")
    _require_positive_int(num_replicas, "num_replicas")
    geometry = RewardTileGeometry(
        group_layout=group_layout,
        optimizer_examples_per_group=optimizer_examples_per_group,
        optimizer_terms_per_batch=optimizer_terms_per_batch,
        accumulation_scope=accumulation_scope,
    )
    batches_per_window = 1
    if geometry.accumulation_scope == "tile":
        batches_per_window = gradient_accumulation_steps // math.gcd(
            gradient_accumulation_steps,
            optimizer_terms_per_batch,
        )

    if geometry.group_layout in {"cross_rank_sharded", "cross_rank_tiled"}:
        group_window_batches = 1
        if geometry.group_layout == "cross_rank_tiled":
            global_batch_size = num_replicas * per_device_batch_size
            group_window_batches = group_size // math.gcd(global_batch_size, group_size)
        batches_per_tile = math.lcm(group_window_batches, batches_per_window)
        return per_device_batch_size * batches_per_tile, batches_per_tile

    examples_per_group = geometry.optimizer_examples_per_group or group_size
    examples_per_window = per_device_batch_size * batches_per_window
    groups_per_tile = examples_per_window // math.gcd(
        examples_per_group,
        examples_per_window,
    )
    samples_per_tile = groups_per_tile * group_size
    optimizer_examples = groups_per_tile * examples_per_group
    if optimizer_examples % per_device_batch_size:
        raise RuntimeError(
            "reward tile geometry failed to form complete optimizer batches: "
            f"optimizer_examples={optimizer_examples}, "
            f"per_device_batch_size={per_device_batch_size}"
        )
    return samples_per_tile, optimizer_examples // per_device_batch_size


def _require_positive_int(value: object, name: str) -> None:
    """Require a positive non-boolean integer geometry value."""
    if type(value) is not int or value < 1:
        raise ValueError(f"expected {name} to be a positive integer, received {value!r}")


__all__ = [
    "RewardAccumulationScope",
    "RewardGroupLayout",
    "RewardTile",
    "RewardTileGeometry",
    "RewardTilePlan",
    "build_reward_tile_plan",
    "resolve_reward_tile_size",
]
