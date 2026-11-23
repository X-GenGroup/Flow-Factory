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

"""DRTune with a live Euler state and sparse denoiser parameter gradients."""

from collections.abc import Callable

import torch
from accelerate.utils import DistributedType

from ...contracts.execution import DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
from ...samples import BaseSample, LatentState, StackedSampleBatch
from ...scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from ..abc import BaseTrainer
from ..common.euler_reward import acquire_euler_inputs, optimize_euler_reward, sample_euler_endpoint
from ..common.group_noise import identity_timestep_generator
from ..common.reward_gradient import scorer_identity


def spaced_training_steps(total: int, count: int, generator: torch.Generator) -> range:
    """Draw an equal-stride set of K positions from the native schedule."""
    if not 1 <= count <= total:
        raise ValueError(f"expected {count} DRTune steps to fit a schedule of length {total}")
    stride = total // count
    valid_starts = total - (count - 1) * stride
    offset = int(torch.randint(valid_starts, (), generator=generator, device="cpu"))
    return range(offset, offset + count * stride, stride)


def unroll_sparse_sampling(
    initial: torch.Tensor,
    timesteps: torch.Tensor,
    selected: range,
    advance: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Carry the live Euler state through every selected and unselected step."""
    latent = initial.detach()
    total = timesteps.shape[-1]
    for index in range(total):
        current = timesteps[..., index]
        following = timesteps[..., index + 1] if index + 1 < total else torch.zeros_like(current)
        latent = advance(
            latent,
            current,
            following,
            prediction_gradients="parameters" if index in selected else "none",
        )
    return latent


class DRTuneTrainer(BaseTrainer):
    """Score the Euler endpoint with sparse stop-gradient supervision."""

    execution_contract = DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    paradigm = "decoupled"

    def _validate_paradigm_dynamics(self) -> None:
        scheduler = self.adapter.scheduler
        if (
            not isinstance(scheduler, FlowMatchEulerDiscreteSDEScheduler)
            or scheduler.dynamics_type != "ODE"
        ):
            raise ValueError("DRTune requires the FlowMatch Euler ODE scheduler")
        if self.accelerator.distributed_type not in (
            DistributedType.NO,
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
        ):
            raise ValueError("DRTune currently supports local and data-only DDP policy execution")

    def runtime_execution_identity_payload(self) -> dict:
        """Bind reward revisions and the rank-consistent mask to exact resume."""
        identity = super().runtime_execution_identity_payload()
        identity["differentiable_scorers"] = scorer_identity(self.reward_models)
        identity["drtune_mask"] = "rank_shared_euler_stride_uniform_offset_v1"
        return identity

    def _initialize_snapshots(self) -> None:
        super()._initialize_snapshots()
        self.adapter._require_component("vae").eval().requires_grad_(False)

    def sample(self) -> list[BaseSample]:
        """Acquire the same native Euler inputs as DRaFT-K."""
        return acquire_euler_inputs(self)

    def _sample_endpoint(
        self, batch: StackedSampleBatch, inner_epoch: int = 0, batch_index: int = 0
    ) -> LatentState:
        def unroll(initial, timesteps, advance):
            # All replicas use the same mask, even when their prompts differ.
            generator = identity_timestep_generator(self, (batch_index,), inner_epoch)
            selected = spaced_training_steps(
                timesteps.shape[-1], self.training_args.drtune_backprop_steps, generator
            )
            return unroll_sparse_sampling(initial, timesteps, selected, advance)

        return sample_euler_endpoint(self, batch, unroll)

    def optimize(self, samples: list[BaseSample]) -> None:
        """Differentiate the endpoint reward using the sparse Euler graph."""
        optimize_euler_reward(self, samples, self._sample_endpoint)
