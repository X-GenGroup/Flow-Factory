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

"""Dependency-neutral execution semantics for training algorithms."""

from dataclasses import dataclass
from enum import Enum


class AcquisitionMode(str, Enum):
    """Describe where optimization examples come from."""

    GENERATION = "generation"
    DATASET = "dataset"


class FeedbackMode(str, Enum):
    """Describe whether an acquisition requires runtime reward feedback."""

    RUNTIME_REWARD = "runtime_reward"
    DIFFERENTIABLE_REWARD = "differentiable_reward"
    NONE = "none"


@dataclass(frozen=True)
class ExecutionContract:
    """Declare orthogonal acquisition and feedback semantics for an algorithm.

    Acquisition determines only how examples enter the training kernel. Feedback
    independently determines whether the acquired examples pass through the runtime
    reward/advantage stage before optimization. Cycle and loader details are derived
    runtime policy, not additional user-configurable axes.
    """

    acquisition: AcquisitionMode
    feedback: FeedbackMode

    def __post_init__(self) -> None:
        """Require typed enum members without coercing ambiguous strings."""
        _require_enum(self.acquisition, AcquisitionMode, "acquisition")
        _require_enum(self.feedback, FeedbackMode, "feedback")


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    """Require one typed execution enum member."""
    if not isinstance(value, enum_type):
        raise TypeError(
            f"expected {field_name} to be {enum_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


ONLINE_EXECUTION_CONTRACT = ExecutionContract(
    acquisition=AcquisitionMode.GENERATION,
    feedback=FeedbackMode.RUNTIME_REWARD,
)

DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT = ExecutionContract(
    acquisition=AcquisitionMode.GENERATION,
    feedback=FeedbackMode.DIFFERENTIABLE_REWARD,
)


ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT = ExecutionContract(
    acquisition=AcquisitionMode.GENERATION,
    feedback=FeedbackMode.NONE,
)

OFFLINE_EXECUTION_CONTRACT = ExecutionContract(
    acquisition=AcquisitionMode.DATASET,
    feedback=FeedbackMode.NONE,
)


__all__ = [
    "DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT",
    "AcquisitionMode",
    "ExecutionContract",
    "FeedbackMode",
    "OFFLINE_EXECUTION_CONTRACT",
    "ONLINE_EXECUTION_CONTRACT",
    "ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT",
]
