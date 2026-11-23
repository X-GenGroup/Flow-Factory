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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from ...contracts.execution import DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT, ExecutionContract
from ._base import TrainingArguments

if TYPE_CHECKING:
    from ..args import Arguments


@dataclass
class DRaFTTrainingArguments(TrainingArguments):
    """Differentiate the last K contiguous Euler flow steps."""

    execution_contract: ClassVar[ExecutionContract] = DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    reward_gradient_fields: ClassVar[tuple[str, ...]] = ("image",)
    trainer_type: Literal["draft"] = "draft"
    group_size: int = 1
    ema_decay: float = 0.0
    latent_storage_dtype: str = "float32"
    draft_backprop_steps: int = field(
        default=1, metadata={"help": "Number of contiguous terminal ODE steps differentiated."}
    )

    def validate_configuration(self, args: "Arguments") -> None:
        """Validate the first SD3.5 Euler ODE composition before loading."""
        steps = self.draft_backprop_steps
        if type(steps) is not int or not 1 <= steps <= self.num_inference_steps:
            raise ValueError(
                f"draft_backprop_steps must be in [1, {self.num_inference_steps}], received {steps!r}"
            )
        if args.scheduler_args.dynamics_type != "ODE":
            raise ValueError("DRaFT-K requires scheduler.dynamics_type='ODE'")
        if args.model_args.model_type.lower() != "sd3-5":
            raise ValueError("DRaFT-K currently supports the SD3.5 adapter")
        if "vae" in args.model_args.target_components:
            raise ValueError("DRaFT-K requires a frozen VAE")
        if args.acceleration_args.rollout:
            raise ValueError("DRaFT-K requires matching generation and differentiable sampling")
        if (
            args.parallelism_args.effective_tensor_parallel_size > 1
            or args.parallelism_args.effective_context_parallel_size > 1
        ):
            raise ValueError("DRaFT-K currently requires a data-only topology")
