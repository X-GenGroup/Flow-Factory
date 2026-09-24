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

"""Typed trainer capability for overlapping runtime rewards with optimization."""

from dataclasses import dataclass
from typing import Literal, Tuple

from .sampler import GroupPlacement

RewardTileScheduling = Literal["ordered", "ready"]


@dataclass(frozen=True)
class RewardOptimizationOverlapContract:
    """Declare whether a trainer can consume complete reward tiles incrementally.

    This contract is intentionally independent from :class:`ExecutionContract`.
    The execution contract says which semantic stages exist; this capability says
    whether a concrete objective may pipeline its reward and optimization stages.
    """

    supported: bool = False
    scheduling_modes: Tuple[RewardTileScheduling, ...] = ()
    sampler_group_placements: Tuple[GroupPlacement, ...] = ()

    def __post_init__(self) -> None:
        """Reject contradictory or unknown capability declarations."""
        valid_modes = {"ordered", "ready"}
        unknown = tuple(mode for mode in self.scheduling_modes if mode not in valid_modes)
        if unknown:
            raise ValueError(f"unknown reward tile scheduling modes: {unknown!r}")
        if not self.supported and self.scheduling_modes:
            raise ValueError(
                "an unsupported reward-overlap contract cannot declare scheduling modes"
            )
        if self.supported and not self.scheduling_modes:
            raise ValueError("a supported reward-overlap contract must declare a scheduling mode")
        valid_placements = {"rank_local", "global_batch", "global_tile"}
        unknown_placements = tuple(
            placement
            for placement in self.sampler_group_placements
            if placement not in valid_placements
        )
        if unknown_placements:
            raise ValueError(
                f"unknown reward-overlap sampler group placements: {unknown_placements!r}"
            )
        if not self.supported and self.sampler_group_placements:
            raise ValueError(
                "an unsupported reward-overlap contract cannot declare sampler layouts"
            )
        if self.supported and not self.sampler_group_placements:
            raise ValueError("a supported reward-overlap contract must declare sampler layouts")

    def supports(self, mode: str) -> bool:
        """Return whether a scheduling mode is declared.

        Args:
            mode: Requested tile scheduling mode.

        Returns:
            Whether this supported contract declares ``mode``.
        """
        return self.supported and mode in self.scheduling_modes

    def supports_sampler_group_placement(self, placement: str) -> bool:
        """Return whether the objective can consume this sampler layout."""

        return self.supported and placement in self.sampler_group_placements


NO_REWARD_OPTIMIZATION_OVERLAP = RewardOptimizationOverlapContract()
COUPLED_REWARD_OPTIMIZATION_OVERLAP = RewardOptimizationOverlapContract(
    supported=True,
    scheduling_modes=("ordered", "ready"),
    sampler_group_placements=("rank_local", "global_batch", "global_tile"),
)
# The original name described the first supported family (GRPO/DPPO), not the
# capability itself.  Keep it as a compatibility alias while new decoupled and
# multi-role trainers use the topology-neutral spelling.
GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP = COUPLED_REWARD_OPTIMIZATION_OVERLAP
RANK_LOCAL_REWARD_OPTIMIZATION_OVERLAP = RewardOptimizationOverlapContract(
    supported=True,
    scheduling_modes=("ordered", "ready"),
    sampler_group_placements=("rank_local",),
)
GLOBAL_BATCH_REWARD_OPTIMIZATION_OVERLAP = RewardOptimizationOverlapContract(
    supported=True,
    scheduling_modes=("ordered", "ready"),
    sampler_group_placements=("global_batch",),
)
WHOLE_GROUP_BATCH_REWARD_OPTIMIZATION_OVERLAP = RewardOptimizationOverlapContract(
    supported=True,
    scheduling_modes=("ordered", "ready"),
    sampler_group_placements=("rank_local", "global_batch"),
)


__all__ = [
    "COUPLED_REWARD_OPTIMIZATION_OVERLAP",
    "GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP",
    "GLOBAL_BATCH_REWARD_OPTIMIZATION_OVERLAP",
    "NO_REWARD_OPTIMIZATION_OVERLAP",
    "RANK_LOCAL_REWARD_OPTIMIZATION_OVERLAP",
    "RewardOptimizationOverlapContract",
    "RewardTileScheduling",
    "WHOLE_GROUP_BATCH_REWARD_OPTIMIZATION_OVERLAP",
]
