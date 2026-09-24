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

"""Training arguments for Flow-DPPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ._base import TrainingArguments, _standardize_clip_range


@dataclass
class DPPOTrainingArguments(TrainingArguments):
    r"""Training arguments for Flow-DPPO.

    DPPO keeps GRPO's group advantages and the optional KL-vs-reference penalty,
    but replaces the PPO ratio-clip with a KL trust-region mask. It therefore does
    NOT inherit ``GRPOTrainingArguments`` (no ``clip_range``); the field set is
    kept minimal. The two KL computations are decoupled:

    - ``kl_type`` selects the space of the KL-vs-reference penalty (``kl_beta``).
    - ``kl_mask_type`` selects the space of the per-step KL(current || old) used by
      the trust-region mask (``kl_mask_threshold``).
    """

    # Group-wise advantage normalization / aggregation (same semantics as GRPO).
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
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )

    # KL-vs-reference penalty (optional).
    kl_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={
            "help": "Space of the KL-vs-reference penalty. 'v-based': velocity, 'x-based': latent."
        },
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL(current || reference) penalty beta. 0 to disable the ref term."},
    )
    kl_guidance_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "CFG scale for the KL-vs-reference forward. None uses the training "
            "guidance_scale; >1.0 enables CFG on the frozen reference model."
        },
    )
    # DPPO trust-region mask.
    kl_mask_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={"help": "Space of the KL(current || old) used by the trust-region mask."},
    )
    kl_mask_threshold: float = field(
        default=1.0e-6,
        metadata={
            "help": "Mask (zero-gradient) samples whose per-step KL(current || old) "
            "exceeds this threshold and push the wrong way."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Guard against scientific-notation strings from CLI/YAML overrides.
        self.kl_beta = float(self.kl_beta)
        self.kl_mask_threshold = float(self.kl_mask_threshold)
        if self.kl_guidance_scale is not None:
            self.kl_guidance_scale = float(self.kl_guidance_scale)
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.kl_type not in ("v-based", "x-based"):
            raise ValueError(f"expected kl_type in ('v-based', 'x-based'), got {self.kl_type!r}")
        if self.kl_mask_type not in ("v-based", "x-based"):
            raise ValueError(
                f"expected kl_mask_type in ('v-based', 'x-based'), got {self.kl_mask_type!r}"
            )

    def get_num_train_timesteps(self, args: Any) -> int:
        return args.scheduler_args.num_sde_steps

    def get_preprocess_guidance_scale(self) -> float:
        """Ensure negative prompts are encoded when the KL-ref branch needs CFG."""
        return max(self.guidance_scale, self.kl_guidance_scale or 0.0)
