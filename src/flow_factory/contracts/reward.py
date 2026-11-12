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

"""Model-owned input derivative capabilities for reward scoring."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardGradientCapability:
    """Name media inputs whose complete scoring path preserves derivatives."""

    input_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_fields, tuple)
            or not self.input_fields
            or any(name not in ("image", "video", "audio") for name in self.input_fields)
            or len(set(self.input_fields)) != len(self.input_fields)
        ):
            raise ValueError(f"invalid reward gradient input_fields={self.input_fields!r}")
