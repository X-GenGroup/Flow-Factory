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

"""Training arguments for Centered Reward Distillation (CRD)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple, Union

from ._base import TrainingArguments, _standardize_clip_range, _standardize_timestep_range


@dataclass
class CRDTrainingArguments(TrainingArguments):
    r"""Training arguments for Centered Reward Distillation (CRD).

    Reference:
        Diffusion Reinforcement Learning via Centered Reward Distillation
        https://arxiv.org/abs/2603.14128
    """

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

    # CRD core
    crd_beta: float = field(
        default=1.0,
        metadata={
            "help": "Beta scaling for CRD reward matching loss. Controls implicit vs external reward balance."
        },
    )
    crd_loss_type: Literal["mse", "bce"] = field(
        default="mse",
        metadata={
            "help": "Loss type for CRD reward distillation. 'mse': squared error, 'bce': binary cross-entropy."
        },
    )
    use_old_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Use 'old' model snapshot (instead of ref) for implicit reward estimation."
        },
    )
    adaptive_logp: bool = field(
        default=True,
        metadata={"help": "Adaptively weight implicit reward terms by prediction error magnitude."},
    )
    weight_temp: float = field(
        default=-1.0,
        metadata={
            "help": "Temperature for softmax weighting of advantages in CRD. Negative means uniform (inf temp)."
        },
    )
    # Decay schedules for model snapshots
    old_model_decay: str = field(
        default="0-0.25-0.005-0.999",
        metadata={
            "help": "Decay schedule for old model blending: 'start_step-start_value-slope-end_value' or preset name."
        },
    )
    sampling_model_decay: Union[str, int] = field(
        default="75-0.0-0.0075-0.999",
        metadata={
            "help": "Decay schedule for sampling model blending. Same format as old_model_decay, or int preset."
        },
    )

    # Clipping / KL
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based"] = field(
        default="v-based",
        metadata={"help": "Type of KL divergence. CRD uses 'v-based' (velocity space)."},
    )
    kl_beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty beta for regularization against the reference model."},
    )
    kl_cfg: float = field(
        default=4.5,
        metadata={
            "help": (
                "CFG scale for the teacher (reference) model during KL computation. "
                "If > 1.0, the reference forward pass uses classifier-free guidance: "
                "``velocity = velocity_uncond + kl_cfg * (velocity_cond - velocity_uncond)``. "
                "Set to 1.0 (default) to disable CFG on the teacher."
            )
        },
    )
    reward_adaptive_kl: bool = field(
        default=True,
        metadata={"help": "Dynamically adjust KL strength based on reward signal."},
    )
    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": "Number of training timesteps. 0 = auto from num_inference_steps * timestep_range."
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
        default=0.99,
        metadata={
            "help": "Fraction range along denoise axis 1000->0. Default 0.99 matches original CRD's timestep_fraction."
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

    @property
    def requires_ref_model(self) -> bool:
        """CRD always needs a reference model for KL and implicit reward."""
        return True

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps

    def get_preprocess_guidance_scale(self) -> float:
        """Account for kl_cfg: ref model may need CFG even when sampling does not."""
        return max(self.guidance_scale, self.kl_cfg)
