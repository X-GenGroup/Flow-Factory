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

"""Configuration for rank-local DiffusionART."""

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ...contracts.sampler import RANK_LOCAL_SAMPLER_SELECTION, SamplerSelectionContract
from ._base import TrainingArguments


@dataclass
class ARTTrainingArguments(TrainingArguments):
    """Scalar-reward ART; step indices refer to the actual ODE rollout."""

    sampler_selection_contract: ClassVar[SamplerSelectionContract] = RANK_LOCAL_SAMPLER_SELECTION
    trainer_type: Literal["art"] = field(default="art")
    group_size: int = field(default=4)
    ema_decay: float = field(default=0.0)
    latent_storage_dtype: str = field(default="float32")
    gradient_step_per_epoch: int = field(default=2)
    num_inner_epochs: int = field(default=1)
    global_std: bool = field(default=False)
    advantage_aggregation: Literal["sum", "gdpo"] = field(default="gdpo")
    art_guidance_scale: float = field(
        default=1.0, metadata={"help": "Magnitude of normalized endpoint covariance guidance."}
    )
    art_kl_threshold: float = field(
        default=1e-5,
        metadata={"help": "Reference-transition KL threshold for the directional trust mask."},
    )
    art_reference_noise_level: float = field(
        default=0.8,
        metadata={
            "help": "Virtual Flow-SDE noise for the ODE trust metric; does not change sampling."
        },
    )
    art_epsilon: float = field(
        default=1e-5,
        metadata={"help": "Numerical floor for reward direction and transition variance."},
    )

    def __post_init__(self) -> None:
        if self.trainer_type != "art":
            raise ValueError("ARTTrainingArguments requires trainer_type='art'")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError(f"ART requires group_size >= 2, received {self.group_size!r}")
        for name in (
            "art_guidance_scale",
            "art_kl_threshold",
            "art_reference_noise_level",
            "art_epsilon",
        ):
            value = float(getattr(self, name))
            positive = name in ("art_reference_noise_level", "art_epsilon")
            if not math.isfinite(value) or value < 0 or (positive and value == 0):
                domain = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {domain}, received {value!r}")
            setattr(self, name, value)
        super().__post_init__()

    def get_num_train_timesteps(self, args: Any) -> int:
        """Count selected replay positions for shared accumulation geometry."""
        if args.scheduler_args.dynamics_type != "ODE":
            raise ValueError("ART requires scheduler.dynamics_type='ODE'")
        if args.acceleration_args.rollout:
            raise ValueError(
                "ART requires matching rollout/replay forwards; rollout-only accelerators are unsupported"
            )
        return args.scheduler_args.get_num_replay_steps(self.num_inference_steps)
