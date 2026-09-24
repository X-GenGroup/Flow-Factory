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

"""Dependency-neutral sampler layout contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

GroupPlacement = Literal["arbitrary", "rank_local", "global_batch", "global_tile"]


@dataclass(frozen=True)
class SamplerLayoutContract:
    """Describe where a sampler guarantees complete comparison groups."""

    name: str
    group_placement: GroupPlacement

    @property
    def groups_are_rank_local(self) -> bool:
        """Return whether one rank owns every member of each group."""

        return self.group_placement == "rank_local"

    @property
    def supports_incremental_group_feedback(self) -> bool:
        """Return whether bounded work units can close complete groups."""

        return self.group_placement != "arbitrary"

    def global_group_window_batches(
        self,
        *,
        num_replicas: int,
        per_device_batch_size: int,
        group_size: int,
    ) -> int:
        """Return training iterations in the smallest group-complete window.

        Args:
            num_replicas: Number of data-parallel ranks.
            per_device_batch_size: Rank-local microbatch size.
            group_size: Number of samples in one comparison group.

        Returns:
            Number of synchronized rank-local microbatches needed to close every group.

        Raises:
            ValueError: If geometry is invalid or the layout has no bounded window.
        """

        for value, label in (
            (num_replicas, "num_replicas"),
            (per_device_batch_size, "per_device_batch_size"),
            (group_size, "group_size"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"expected {label} to be a positive integer, got {value!r}")
        global_batch_size = num_replicas * per_device_batch_size
        if self.group_placement == "global_batch":
            if global_batch_size % group_size:
                raise ValueError(
                    f"sampler {self.name!r} requires group_size({group_size}) to divide "
                    f"the global batch size({global_batch_size})"
                )
            return 1
        if self.group_placement == "global_tile":
            return group_size // math.gcd(global_batch_size, group_size)
        if self.group_placement == "rank_local":
            return group_size // math.gcd(per_device_batch_size, group_size)
        raise ValueError(f"sampler {self.name!r} has no bounded group-complete window")


SAMPLER_LAYOUT_CONTRACTS: Dict[str, SamplerLayoutContract] = {
    "distributed_k_repeat": SamplerLayoutContract(
        name="distributed_k_repeat",
        group_placement="arbitrary",
    ),
    "group_contiguous": SamplerLayoutContract(
        name="group_contiguous",
        group_placement="rank_local",
    ),
    "group_distributed": SamplerLayoutContract(
        name="group_distributed",
        group_placement="global_batch",
    ),
    "group_tiled": SamplerLayoutContract(
        name="group_tiled",
        group_placement="global_tile",
    ),
}


def get_sampler_layout_contract(name: str) -> SamplerLayoutContract:
    """Resolve one registered sampler layout contract.

    Args:
        name: Registered sampler name.

    Returns:
        Immutable layout contract for the sampler.

    Raises:
        ValueError: If ``name`` is not registered.
    """

    try:
        return SAMPLER_LAYOUT_CONTRACTS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown sampler layout {name!r}; expected one of "
            f"{tuple(sorted(SAMPLER_LAYOUT_CONTRACTS))!r}"
        ) from error


@dataclass(frozen=True)
class SamplerSelectionContract:
    """Declare the sampler layouts an algorithm can consume."""

    allowed_group_placements: Tuple[GroupPlacement, ...]
    auto_preference: Tuple[GroupPlacement, ...]
    requires_group_complete_microbatch: bool = False

    def __post_init__(self) -> None:
        known = {"arbitrary", "rank_local", "global_batch", "global_tile"}
        unknown = set(self.allowed_group_placements).difference(known)
        if unknown:
            raise ValueError(f"unknown allowed sampler placements: {sorted(unknown)!r}")
        if not self.allowed_group_placements:
            raise ValueError("a sampler selection contract needs at least one allowed layout")
        if not self.auto_preference:
            raise ValueError("a sampler selection contract needs an auto preference")
        outside = set(self.auto_preference).difference(self.allowed_group_placements)
        if outside:
            raise ValueError(
                "sampler auto preferences must be allowed placements: "
                f"outside={sorted(outside)!r}"
            )

    def supports(self, layout: SamplerLayoutContract) -> bool:
        """Return whether a layout satisfies the algorithm contract.

        Args:
            layout: Sampler layout to validate.

        Returns:
            Whether the layout's group placement is allowed.
        """

        return layout.group_placement in self.allowed_group_placements


FLEXIBLE_SAMPLER_SELECTION = SamplerSelectionContract(
    allowed_group_placements=("arbitrary", "rank_local", "global_batch", "global_tile"),
    auto_preference=("rank_local", "arbitrary"),
)
RANK_LOCAL_SAMPLER_SELECTION = SamplerSelectionContract(
    allowed_group_placements=("rank_local",),
    auto_preference=("rank_local",),
)
GLOBAL_BATCH_SAMPLER_SELECTION = SamplerSelectionContract(
    allowed_group_placements=("global_batch",),
    auto_preference=("global_batch",),
    requires_group_complete_microbatch=True,
)
WHOLE_GROUP_BATCH_SAMPLER_SELECTION = SamplerSelectionContract(
    allowed_group_placements=("rank_local", "global_batch"),
    auto_preference=("rank_local", "global_batch"),
    requires_group_complete_microbatch=True,
)


__all__ = [
    "FLEXIBLE_SAMPLER_SELECTION",
    "GLOBAL_BATCH_SAMPLER_SELECTION",
    "GroupPlacement",
    "RANK_LOCAL_SAMPLER_SELECTION",
    "SAMPLER_LAYOUT_CONTRACTS",
    "SamplerLayoutContract",
    "SamplerSelectionContract",
    "WHOLE_GROUP_BATCH_SAMPLER_SELECTION",
    "get_sampler_layout_contract",
]
