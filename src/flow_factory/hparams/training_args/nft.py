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

"""Training arguments for DiffusionNFT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple, Union

from ._base import TrainingArguments, _standardize_clip_range, _standardize_timestep_range


@dataclass
class NFTTrainingArguments(TrainingArguments):
    r"""Training arguments for DiffusionNFT."""

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
    # NFT core
    nft_beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for NFT trainer."},
    )
    off_policy: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA parameters for sampling off-policy data."},
    )

    # Clipping / KL
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based"] = field(
        default="v-based",
        metadata={"help": "Type of KL divergence. NFT defaults to 'v-based'."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": "Total number of training timesteps. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."
        },
    )
    time_sampling_strategy: Literal[
        "uniform", "logit_normal", "discrete", "discrete_with_init", "discrete_wo_init"
    ] = field(
        default="discrete",
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.9,
        metadata={
            "help": "Fraction range along denoise axis 1000->0; maps to scheduler times "
            "[1000*(1-end), 1000*(1-start)]. Float means [0, value]."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Guard kl_beta against scientific-notation strings (e.g. "1e-3" from CLI overrides).
        self.kl_beta = float(self.kl_beta)

        self.timestep_range = _standardize_timestep_range(self.timestep_range)

        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )

        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.kl_type not in ["v-based"]:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps
