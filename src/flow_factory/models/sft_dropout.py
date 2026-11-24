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

"""Adapter-owned SFT policies and record-addressed condition sampling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..data_utils.schema import NormalizedModelInput


class SFTConditionDropoutPolicy(Protocol):
    revision: str
    supports_native: bool
    cached_empty_caption: bool

    def sample(
        self, mode: float | str, model_input: NormalizedModelInput, draw: Callable[[str], float]
    ) -> frozenset[str]: ...

    def validate_native(self, checkpoint: str) -> None: ...


@dataclass(frozen=True)
class CaptionDropoutPolicy:
    """Omit the instruction while preserving all task-defining media."""

    empty_prompt: str = ""
    revision: str = "empty-caption-v1"
    share_across_references: bool = False
    supports_native = False
    cached_empty_caption = True

    def sample(
        self, mode: float | str, model_input: NormalizedModelInput, draw: Callable[[str], float]
    ) -> frozenset[str]:
        if mode == "native":
            raise ValueError("No published native training recipe for caption dropout")
        if model_input.prompt and draw("caption") < mode:
            return frozenset({"text"})
        return frozenset()

    def validate_native(self, checkpoint: str) -> None:
        raise ValueError(f"checkpoint {checkpoint!r} supports numeric SFT condition_dropout only")


CAPTION_DROPOUT = CaptionDropoutPolicy()
# Request this policy only when the same preprocessor can encode text on its own.
SEPARABLE_CAPTION_DROPOUT = CaptionDropoutPolicy(share_across_references=True)


def record_draw(seed: int, epoch: int, record_id: str) -> Callable[[str], float]:
    """Return a private uniform draw for each stable event, independent of rank and call order."""
    prefix = json.dumps(["sft-condition-v1", seed, epoch, record_id]).encode()

    def draw(event: str) -> float:
        digest = hashlib.blake2b(prefix + b"\0" + event.encode(), digest_size=8).digest()
        return (int.from_bytes(digest, "big") >> 11) * (2.0**-53)

    return draw


def resolve_sft_dropout(
    adapter_cls: type, checkpoint: str, mode: float | str
) -> SFTConditionDropoutPolicy | None:
    """Admit an enabled policy before constructing the accelerator or loading any weights."""
    if mode == 0:
        return None
    policy = getattr(adapter_cls, "sft_condition_dropout_policy", None)
    if policy is None:
        raise ValueError(
            f"{adapter_cls.__name__} at {checkpoint!r} has no SFT condition_dropout "
            f"policy for {mode!r}; use train.condition_dropout=0"
        )
    if mode == "native":
        policy.validate_native(checkpoint)
    return policy
