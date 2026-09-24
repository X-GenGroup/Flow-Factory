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

"""Training arguments for GRPO / GRPO-Guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ._base import TrainingArguments, _standardize_clip_range


@dataclass
class GRPOTrainingArguments(TrainingArguments):
    r"""Training arguments for GRPO / GRPO-Guard."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to apply acquisition-wide std normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."
        },
    )
    # Clipping / KL
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for PPO/GRPO ratio."},
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={
            "help": "Type of KL divergence. 'v-based': velocity space, 'x-based': latent space."
        },
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )

    def __post_init__(self):
        super().__post_init__()
        # Guard kl_beta against scientific-notation strings (e.g. "1e-3" from CLI overrides).
        self.kl_beta = float(self.kl_beta)
        self.clip_range = _standardize_clip_range(self.clip_range, "clip_range")
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.kl_type not in ["v-based", "x-based"]:
            raise ValueError(
                f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based', 'x-based']."
            )

    def get_num_train_timesteps(self, args: Any) -> int:
        return args.scheduler_args.num_sde_steps
