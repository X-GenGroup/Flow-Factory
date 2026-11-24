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

"""Bind an SFT record's sampled conditioning to a cache row or an on-demand encoder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from datasets import Dataset

from ..models.sft_dropout import CaptionDropoutPolicy, SFTConditionDropoutPolicy, record_draw
from .offline_conditions import RawOfflineConditionSource
from .offline_dataset import OFFLINE_CONDITION_ID_COLUMN
from .schema import NormalizedDatasetRecord, NormalizedModelInput


def absent_caption_records(
    records: Sequence[NormalizedDatasetRecord],
    empty_prompt: str,
    *,
    share_across_references: bool = False,
) -> tuple[list[NormalizedDatasetRecord], list[int]]:
    """Index deterministic alternatives once per distinct retained input, in source order."""
    alternatives, positions, indices = [], {}, []
    for record in records:
        absent = replace(
            record.model_input,
            prompt=empty_prompt,
            media=() if share_across_references else record.model_input.media,
        )
        if absent not in positions:
            positions[absent] = len(alternatives)
            alternatives.append(replace(record, model_input=absent))
        indices.append(positions[absent])
    return alternatives, indices


@dataclass
class SFTConditionSource(RawOfflineConditionSource):
    """Worker-side state. All model components stay on the training rank."""

    policy: SFTConditionDropoutPolicy
    mode: float | str
    seed: int
    raw: bool
    alternatives: Dataset | None = None
    indices: tuple[int, ...] = ()
    epoch: int = 0

    def select(
        self,
        index: int,
        model_input: NormalizedModelInput,
        record_id: str,
        condition: dict[str, Any],
    ) -> dict[str, Any]:
        """Realize a sampled input while preserving the positive cache and supervision."""
        policy = self.policy
        mask = policy.sample(self.mode, model_input, record_draw(self.seed, self.epoch, record_id))
        if self.raw:
            if mask and isinstance(policy, CaptionDropoutPolicy):
                condition = {**condition, "prompt": policy.empty_prompt}
            condition = super().select(index, model_input, record_id, condition)
        elif mask and isinstance(policy, CaptionDropoutPolicy):
            if self.alternatives is None:
                raise RuntimeError(
                    "SFT caption dropout selected a record without an alternate cache"
                )
            alternative = self.alternatives[self.indices[index]]
            absent = {k: v for k, v in alternative.items() if k != OFFLINE_CONDITION_ID_COLUMN}
            condition = {**condition, **absent} if policy.share_across_references else absent
        condition["_sft_mask"] = mask
        return condition
