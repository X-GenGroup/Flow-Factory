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

"""DRaFT-K on an Euler flow sampling chain."""

from collections.abc import Callable

import torch
from accelerate.utils import DistributedType

from ...contracts.execution import DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
from ...samples import BaseSample, LatentState, StackedSampleBatch
from ...scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from ..abc import BaseTrainer
from ..common.euler_reward import acquire_euler_inputs, optimize_euler_reward, sample_euler_endpoint
from ..common.reward_gradient import scorer_identity


def unroll_sampling_suffix(
    initial: torch.Tensor,
    timesteps: torch.Tensor,
    steps: int,
    advance: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply one policy across the schedule, preserving only suffix derivatives."""
    total = timesteps.shape[-1]
    if not 1 <= steps <= total:
        raise ValueError(f"expected DRaFT-K in [1, {total}], received {steps}")
    latent = initial.detach()
    cut = total - steps
    for index in range(total):
        current = timesteps[..., index]
        following = timesteps[..., index + 1] if index + 1 < total else torch.zeros_like(current)
        if index < cut:
            with torch.no_grad():
                latent = advance(latent, current, following)
        else:
            latent = advance(latent, current, following)
    return latent


class DRaFTTrainer(BaseTrainer):
    """Score the differentiably sampled endpoint through frozen auxiliaries."""

    execution_contract = DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    paradigm = "decoupled"

    def _validate_paradigm_dynamics(self) -> None:
        scheduler = self.adapter.scheduler
        if (
            not isinstance(scheduler, FlowMatchEulerDiscreteSDEScheduler)
            or scheduler.dynamics_type != "ODE"
        ):
            raise ValueError("DRaFT-K requires the FlowMatch Euler ODE scheduler")
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            raise ValueError("DRaFT-K currently supports local, DDP and FSDP2 policy execution")
        if (
            self.accelerator.distributed_type == DistributedType.FSDP
            and not self.accelerator.is_fsdp2
        ):
            raise ValueError("DRaFT-K requires FSDP2 when the FSDP policy backend is selected")

    def runtime_execution_identity_payload(self) -> dict:
        """Bind the realized differentiable score and initial-state acquisition."""
        identity = super().runtime_execution_identity_payload()
        identity["differentiable_scorers"] = scorer_identity(self.reward_models)
        identity["draft_acquisition"] = "initial_latent_and_euler_schedule"
        return identity

    def _initialize_snapshots(self) -> None:
        super()._initialize_snapshots()
        self.adapter._require_component("vae").eval().requires_grad_(False)

    def sample(self) -> list[BaseSample]:
        """Acquire only initial noise, conditioning and the native schedule."""
        return acquire_euler_inputs(self)

    def _sample_endpoint(self, batch: StackedSampleBatch) -> LatentState:
        def unroll(initial, timesteps, advance):
            return unroll_sampling_suffix(
                initial, timesteps, self.training_args.draft_backprop_steps, advance
            )

        return sample_euler_endpoint(self, batch, unroll)

    def optimize(self, samples: list[BaseSample]) -> None:
        """Differentiate one complete suffix and reward per accumulated microstep."""
        optimize_euler_reward(
            self, samples, lambda batch, _inner_epoch, _batch_index: self._sample_endpoint(batch)
        )
