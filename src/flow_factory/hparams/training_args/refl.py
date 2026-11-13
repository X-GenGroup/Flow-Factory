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

"""ReFL over selected rollout states with input-differentiable feedback."""

import math
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ...contracts.execution import DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT, ExecutionContract
from ._base import TrainingArguments


@dataclass
class ReFLTrainingArguments(TrainingArguments):
    """Select one saved transition per microbatch and optimize its clean prediction."""

    execution_contract: ClassVar[ExecutionContract] = DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    trainer_type: Literal["refl"] = field(default="refl")
    group_size: int = field(default=1)
    ema_decay: float = field(default=0.0)
    latent_storage_dtype: str = field(default="float32")
    refl_reward_scale: float = field(
        default=1.0, metadata={"help": "Multiplier of the differentiated reward objective."}
    )
    refl_reward_target: float | None = field(
        default=None,
        metadata={
            "help": "Optional hinge target: relu(target - score); null maximizes score directly."
        },
    )

    def __post_init__(self) -> None:
        if self.trainer_type != "refl":
            raise ValueError("ReFLTrainingArguments requires trainer_type='refl'")
        if not math.isfinite(self.refl_reward_scale) or self.refl_reward_scale < 0:
            raise ValueError(
                f"refl_reward_scale must be finite and non-negative, received {self.refl_reward_scale!r}"
            )
        if self.refl_reward_target is not None and not math.isfinite(self.refl_reward_target):
            raise ValueError(
                f"refl_reward_target must be finite or null, received {self.refl_reward_target!r}"
            )
        super().__post_init__()

    def validate_configuration(self, args) -> None:
        """Require saved Euler ODE states and matching generation/replay semantics."""
        if args.scheduler_args.dynamics_type != "ODE":
            raise ValueError("ReFL currently requires scheduler.dynamics_type='ODE'")
        args.scheduler_args.validate_replay_selection(self.num_inference_steps)
        if args.scheduler_args.get_num_replay_steps(self.num_inference_steps) == 0:
            raise ValueError("ReFL requires at least one retained replay transition")
        if args.acceleration_args.rollout:
            raise ValueError("ReFL requires matching rollout and replay forwards")
        from ...models.registry import get_model_adapter_class

        adapter_cls = get_model_adapter_class(args.model_args.model_type)
        targeted_decoders = set(args.model_args.target_components).intersection(
            adapter_cls.differentiable_decode_modules
        )
        if targeted_decoders:
            raise ValueError(f"ReFL requires frozen decoders, targeted={sorted(targeted_decoders)}")

    def get_reward_gradient_fields(self, args) -> tuple[str, ...]:
        """Use the selected adapter's qualified media without a modality knob."""
        from ...models.registry import get_model_adapter_class

        return get_model_adapter_class(args.model_args.model_type).differentiable_decode_fields
