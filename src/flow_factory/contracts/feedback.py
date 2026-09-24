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

"""Contracts for reducing one or more rewards into training feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RewardCombinationOrder = Literal[
    "aggregate_then_group_normalize",
    "group_normalize_then_aggregate",
    "custom",
]


@dataclass(frozen=True)
class FeedbackReducerContract:
    """Describe the scope and ordering of a reward-to-advantage reducer."""

    name: str
    combination_order: RewardCombinationOrder
    group_relative: bool
    requires_acquisition_statistics: bool

    @property
    def supports_group_complete_streaming(self) -> bool:
        """Return whether independent complete groups can be reduced early."""

        return self.group_relative and not self.requires_acquisition_statistics


def resolve_feedback_reducer_contract(
    aggregation: Any,
    *,
    global_std: bool,
) -> FeedbackReducerContract:
    """Resolve multi-reward semantics without executing the reducer.

    Args:
        aggregation: Built-in reducer name or a custom callable.
        global_std: Whether reduction needs acquisition-wide standardization.

    Returns:
        Static reducer capability contract.

    Raises:
        ValueError: If ``aggregation`` is unsupported.
    """

    if aggregation == "sum":
        return FeedbackReducerContract(
            name="sum",
            combination_order="aggregate_then_group_normalize",
            group_relative=True,
            requires_acquisition_statistics=global_std,
        )
    if aggregation == "gdpo":
        return FeedbackReducerContract(
            name="gdpo",
            combination_order="group_normalize_then_aggregate",
            group_relative=True,
            requires_acquisition_statistics=global_std,
        )
    if callable(aggregation):
        return FeedbackReducerContract(
            name=getattr(aggregation, "__name__", type(aggregation).__name__),
            combination_order="custom",
            group_relative=False,
            requires_acquisition_statistics=True,
        )
    raise ValueError(
        f"unsupported advantage aggregation {aggregation!r}; expected 'sum', 'gdpo', "
        "or a callable"
    )


__all__ = [
    "FeedbackReducerContract",
    "RewardCombinationOrder",
    "resolve_feedback_reducer_contract",
]
