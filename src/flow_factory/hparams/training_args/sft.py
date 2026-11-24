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

"""Training arguments for supervised flow-matching fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ...contracts.execution import OFFLINE_EXECUTION_CONTRACT, ExecutionContract
from ._offline import OfflineFlowMatchingTrainingArguments


@dataclass
class SFTTrainingArguments(OfflineFlowMatchingTrainingArguments):
    """Configure finite-dataset supervised flow-matching training."""

    execution_contract: ClassVar[ExecutionContract] = OFFLINE_EXECUTION_CONTRACT

    trainer_type: Literal["sft"] = field(
        default="sft",
        metadata={"help": "Select the offline supervised fine-tuning trainer."},
    )

    condition_dropout: float | Literal["native"] = field(
        default=0.1,
        metadata={
            "help": "Chance to omit eligible conditioning per example, or 'native' for a published recipe."
        },
    )

    def __post_init__(self) -> None:
        """Validate the fixed SFT trainer identity."""
        super().__post_init__()
        mode = self.condition_dropout
        if mode != "native":
            if type(mode) not in (int, float) or not math.isfinite(mode) or not 0 <= mode <= 1:
                raise ValueError(
                    f"train.condition_dropout must be in [0, 1] or 'native', got {mode!r}"
                )
            self.condition_dropout = float(mode)
        if self.trainer_type != "sft":
            raise ValueError(
                "SFTTrainingArguments requires train.trainer_type='sft', "
                f"received {self.trainer_type!r}"
            )

    @property
    def requires_ref_model(self) -> bool:
        """Return false because supervised flow matching has no reference branch."""
        return False


__all__ = ["SFTTrainingArguments"]
