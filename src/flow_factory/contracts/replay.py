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

"""Selection of actual rollout transitions, independent of noise injection."""

from collections.abc import Sequence
from typing import Literal

ReplaySteps = Literal["auto", "all"] | list[int] | None


def validate_replay_steps(selection: ReplaySteps, num_steps: int | None = None) -> None:
    """Validate a selection without reordering it or consuming randomness."""
    if selection is None or isinstance(selection, str) and selection in ("auto", "all"):
        return
    if not isinstance(selection, list) or any(type(i) is not int for i in selection):
        raise ValueError(
            "scheduler.replay_steps must be 'auto', 'all', null, or a list of integer "
            f"transition indices; received {selection!r}"
        )
    if len(set(selection)) != len(selection) or any(
        i < 0 or (num_steps is not None and i >= num_steps) for i in selection
    ):
        raise ValueError(
            "scheduler.replay_steps must contain unique, non-negative rollout transition "
            f"indices below num_inference_steps={num_steps}; received {selection!r}"
        )


def resolve_replay_steps(
    selection: ReplaySteps, num_steps: int, default_steps: Sequence[int]
) -> list[int]:
    """Resolve explicit transitions or preserve the caller's automatic selection."""
    validate_replay_steps(selection, num_steps)
    if selection == "auto":
        return list(default_steps)
    if selection == "all":
        return list(range(num_steps))
    return list(selection or [])
