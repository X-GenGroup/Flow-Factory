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
from typing import Dict, Literal, Optional, Tuple

GroupPlacement = Literal[
    "arbitrary",
    "rank_local",
    "global_batch",
    "global_tile",
    "subgroup_tile",
]
CanonicalSamplerLayout = Literal[
    "global_random",
    "contiguous_shard",
    "global_batch",
    "rank_local",
    "global_tile",
    "subgroup_tile",
]


@dataclass(frozen=True)
class SamplerLayoutContract:
    """Describe where a sampler guarantees complete comparison groups."""

    name: CanonicalSamplerLayout
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
        subgroup_size: Optional[int] = None,
    ) -> int:
        """Return training iterations in the smallest group-complete window.

        Args:
            num_replicas: Number of data-parallel ranks.
            per_device_batch_size: Rank-local microbatch size.
            group_size: Number of samples in one comparison group.

        Returns:
            Number of synchronized rank-local microbatches needed to close every group
            in the layout's synchronization scope.

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
        if self.group_placement == "subgroup_tile":
            resolved_subgroup_size = self.resolve_subgroup_size(
                num_replicas=num_replicas,
                subgroup_size=subgroup_size,
            )
            subgroup_batch_size = resolved_subgroup_size * per_device_batch_size
            return group_size // math.gcd(subgroup_batch_size, group_size)
        if self.group_placement == "rank_local":
            return group_size // math.gcd(per_device_batch_size, group_size)
        raise ValueError(f"sampler {self.name!r} has no bounded group-complete window")

    def unique_groups_per_window(
        self,
        *,
        num_replicas: int,
        per_device_batch_size: int,
        group_size: int,
        subgroup_size: Optional[int] = None,
    ) -> int:
        """Return the minimum group-count multiple closed by this layout.

        This is the single source of truth used by configuration alignment and
        the runtime sampling planner. Arbitrary layouts only need the complete
        acquisition to divide evenly over all rank-local microbatches.
        """

        for value, label in (
            (num_replicas, "num_replicas"),
            (per_device_batch_size, "per_device_batch_size"),
            (group_size, "group_size"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"expected {label} to be a positive integer, got {value!r}")
        global_batch_size = num_replicas * per_device_batch_size
        if self.group_placement == "rank_local":
            groups_per_rank_window = per_device_batch_size // math.gcd(
                per_device_batch_size,
                group_size,
            )
            return num_replicas * groups_per_rank_window
        if self.group_placement == "subgroup_tile":
            resolved_subgroup_size = self.resolve_subgroup_size(
                num_replicas=num_replicas,
                subgroup_size=subgroup_size,
            )
            subgroup_batch_size = resolved_subgroup_size * per_device_batch_size
            groups_per_subgroup_window = subgroup_batch_size // math.gcd(
                subgroup_batch_size,
                group_size,
            )
            return (num_replicas // resolved_subgroup_size) * groups_per_subgroup_window
        return global_batch_size // math.gcd(global_batch_size, group_size)

    def resolve_subgroup_size(
        self,
        *,
        num_replicas: int,
        subgroup_size: Optional[int],
    ) -> int:
        """Validate and return the rank count in one contiguous subgroup."""

        if self.group_placement != "subgroup_tile":
            if subgroup_size is not None:
                raise ValueError(f"sampler {self.name!r} does not accept sampler_subgroup_size")
            return num_replicas
        if type(subgroup_size) is not int or subgroup_size < 1:
            raise ValueError(
                "subgroup_tile requires sampler_subgroup_size to be a positive integer, "
                f"got {subgroup_size!r}"
            )
        if subgroup_size > num_replicas or num_replicas % subgroup_size:
            raise ValueError(
                "subgroup_tile requires sampler_subgroup_size to divide num_replicas: "
                f"num_replicas={num_replicas}, sampler_subgroup_size={subgroup_size}"
            )
        return subgroup_size


_CANONICAL_SAMPLER_LAYOUT_CONTRACTS: Dict[str, SamplerLayoutContract] = {
    "global_random": SamplerLayoutContract(
        name="global_random",
        group_placement="arbitrary",
    ),
    "contiguous_shard": SamplerLayoutContract(
        name="contiguous_shard",
        group_placement="arbitrary",
    ),
    "rank_local": SamplerLayoutContract(
        name="rank_local",
        group_placement="rank_local",
    ),
    "global_batch": SamplerLayoutContract(
        name="global_batch",
        group_placement="global_batch",
    ),
    "global_tile": SamplerLayoutContract(
        name="global_tile",
        group_placement="global_tile",
    ),
    "subgroup_tile": SamplerLayoutContract(
        name="subgroup_tile",
        group_placement="subgroup_tile",
    ),
}

SAMPLER_NAME_ALIASES: Dict[str, str] = {
    "distributed_k_repeat": "global_random",
    "group_contiguous": "rank_local",
    "group_distributed": "global_batch",
    "group_tiled": "global_tile",
}

SAMPLER_LAYOUT_CONTRACTS: Dict[str, SamplerLayoutContract] = {
    **_CANONICAL_SAMPLER_LAYOUT_CONTRACTS,
    **{
        alias: _CANONICAL_SAMPLER_LAYOUT_CONTRACTS[canonical]
        for alias, canonical in SAMPLER_NAME_ALIASES.items()
    },
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
        known = {
            "arbitrary",
            "rank_local",
            "global_batch",
            "global_tile",
            "subgroup_tile",
        }
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
    allowed_group_placements=(
        "arbitrary",
        "rank_local",
        "global_batch",
        "global_tile",
        "subgroup_tile",
    ),
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
    "CanonicalSamplerLayout",
    "GroupPlacement",
    "RANK_LOCAL_SAMPLER_SELECTION",
    "SAMPLER_LAYOUT_CONTRACTS",
    "SAMPLER_NAME_ALIASES",
    "SamplerLayoutContract",
    "SamplerSelectionContract",
    "WHOLE_GROUP_BATCH_SAMPLER_SELECTION",
    "get_sampler_layout_contract",
]
