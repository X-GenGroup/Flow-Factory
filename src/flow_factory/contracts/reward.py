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
    """Declare complete differentiable input sets, in preference order.

    Each set names inputs consumed together. Alternatives let an image scorer
    also implement a video path without requiring both media in one request.
    """

    input_fields: tuple[str, ...]
    alternative_input_fields: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.alternative_input_fields, tuple):
            raise ValueError("reward gradient alternative_input_fields must be a tuple")
        for fields in (self.input_fields, *self.alternative_input_fields):
            if (
                not isinstance(fields, tuple)
                or not fields
                or any(name not in ("image", "video", "audio") for name in fields)
                or len(set(fields)) != len(fields)
            ):
                raise ValueError(f"invalid reward gradient input_fields={fields!r}")

    def select_input_fields(self, available: tuple[str, ...]) -> tuple[str, ...]:
        """Choose one complete input set supported by the model output."""
        for fields in (self.input_fields, *self.alternative_input_fields):
            if set(fields).issubset(available):
                return fields
        return ()
