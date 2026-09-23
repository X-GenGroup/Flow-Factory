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

"""Training arguments for DGPO (Direct Group Preference Optimization)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple, Union

from ._base import TrainingArguments, _standardize_clip_range, _standardize_timestep_range


@dataclass
class DGPOTrainingArguments(TrainingArguments):
    r"""Training arguments for DGPO (Direct Group Preference Optimization).

    Combines group-level DPO loss with PPO-style clipping, shared noise,
    and per-timestep training controls.
    """

    # --- Group-wise advantage & clipping (same semantics as GRPO) ---
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."
        },
    )
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for PPO/DSM ratio."},
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based", "x-based"] = field(
        default="v-based",
        metadata={"help": "Type of KL divergence. DGPO defaults to 'v-based'."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )

    # DGPO core
    dpo_beta: float = field(
        default=100.0,
        metadata={"help": "DPO beta for group preference scaling."},
    )
    use_shared_noise: bool = field(
        default=True,
        metadata={"help": "Whether to share noise across samples within the same group."},
    )
    clip_dsm: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply PPO-style DSM clipping using EMA old-policy predictions."
        },
    )
    clip_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply PPO-style clipping to the KL loss using the same ratio-based mask."
        },
    )
    switch_ema_ref: int = field(
        default=200,
        metadata={
            "help": "After this many optimizer steps, use EMA parameters for sampling instead of current params."
        },
    )
    off_policy: bool = field(
        default=False,
        metadata={
            "help": "Whether to use EMA parameters for sampling from the start (off-policy)."
        },
    )
    kl_cfg: float = field(
        default=1.0,
        metadata={
            "help": "CFG scale for reference model predictions. >1.0 enables CFG on the frozen ref model."
        },
    )
    use_ema_ref: bool = field(
        default=False,
        metadata={
            "help": "Use EMA (old policy) as DGPO loss reference instead of frozen pretrained. Dynamic ref from TDM-R1."
        },
    )

    # Old-policy EMA ref (ema_ref) — a fast-tracking EMA separate from the sampling EMA
    ema_ref_max_decay: float = field(
        default=0.3,
        metadata={
            "help": "Maximum decay for old-policy EMA ref. Actual decay is min(ema_ref_max_decay, ema_ref_ramp_rate * step)."
        },
    )
    ema_ref_ramp_rate: float = field(
        default=0.001,
        metadata={
            "help": "Linear ramp rate for old-policy EMA ref decay. decay(step) = min(max_decay, ramp_rate * step)."
        },
    )
    ema_ref_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device for old-policy EMA ref parameters ('cuda' or 'cpu')."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": "Number of training timesteps per sample. 0 defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."
        },
    )
    time_sampling_strategy: Literal[
        "uniform", "logit_normal", "discrete", "discrete_with_init", "discrete_wo_init"
    ] = field(
        default="discrete",
        metadata={"help": "Strategy for sampling training timesteps."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Shift parameter for logit-normal timestep sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.6,
        metadata={
            "help": "Timestep range for discrete sampling. Float for [0, value], tuple for [start, end]."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Guard kl_beta against scientific-notation strings (e.g. "1e-3").
        self.kl_beta = float(self.kl_beta)
        self.clip_range = _standardize_clip_range(self.clip_range, "clip_range")
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")

        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps

    @property
    def requires_ref_model(self) -> bool:
        """DGPO always requires a reference model for the group DPO loss."""
        return True

    def get_preprocess_guidance_scale(self) -> float:
        """Account for kl_cfg: ref model may need CFG even when sampling does not."""
        return max(self.guidance_scale, self.kl_cfg)
