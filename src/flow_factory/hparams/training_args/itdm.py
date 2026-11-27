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

"""SD3.5 instantaneous trajectory distribution matching configuration."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from ..optimizer_args import AdamWOptimizerArguments
from ._base import TrainingArguments
from .dmd2 import _finite_float


@dataclass
class ITDMTrainingArguments(TrainingArguments):
    """Train a reward-revised generator and its online denoising score."""

    required_trainable_roles: ClassVar = ("generator", "fake")
    global_std: bool = True
    advantage_aggregation: Literal["sum", "gdpo"] = "gdpo"
    shuffle_samples: bool = False
    group_size: int = 24
    unique_sample_num_per_epoch: int = 24
    gradient_step_per_epoch: int = 1
    num_train_timesteps: int = 4
    max_timestep_index: int = 8
    ttur: int = 1
    kl_beta: float = 0.001
    advantage_clip_range: float = 5.0

    def validate_configuration(self, args):
        if args.model_args.model_type != "sd3-5" or args.model_args.finetune_type != "lora":
            raise ValueError("iTDM requires an sd3-5 LoRA model")
        if args.scheduler_args.dynamics_type != "ODE":
            raise ValueError("iTDM requires ODE dynamics")
        if not args.reward_args or self.group_size < 2:
            raise ValueError("iTDM requires a training reward and group_size >= 2")
        for name in ("num_train_timesteps", "max_timestep_index", "ttur"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"train.{name} must be an integer")
        if (
            not 1
            <= self.num_train_timesteps
            <= self.max_timestep_index + 1
            <= self.num_inference_steps
        ):
            raise ValueError("iTDM timestep count and index must fit the rollout schedule")
        if self.ttur < 1:
            raise ValueError("iTDM ttur must be positive")
        self.kl_beta = _finite_float(self.kl_beta, "kl_beta", allow_zero=True)
        self.advantage_clip_range = _finite_float(
            self.advantage_clip_range, "advantage_clip_range", allow_zero=False
        )

    def get_num_train_timesteps(self, args):
        return self.num_train_timesteps

    def default_role_optimizers(self):
        return tuple(
            AdamWOptimizerArguments(
                name=name,
                learning_rate=3e-4,
                betas=(beta1, 0.999),
                weight_decay=1e-4,
                update_frequency=1,
            )
            for name, beta1 in (("generator", 0.9), ("fake", 0.0))
        )

    def role_update_plan(self, update_frequencies):
        from ...trainers.role_optimization import RolePhase, RoleUpdatePlan

        if any(value != 1 for value in update_frequencies.values()):
            raise ValueError("iTDM uses train.ttur for policy cadence")
        return RoleUpdatePlan((RolePhase("fake"), RolePhase("generator")))
