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


def sparse_training_schedule(
    total: int,
    count: int,
    stop_range: tuple[int, int] | None,
    generator: torch.Generator,
) -> tuple[int | None, range]:
    """Draw the stop before picking derivative positions inside that prefix."""
    stop = None
    if stop_range is not None:
        first, last = stop_range
        stop = int(torch.randint(first, last + 1, (), generator=generator, device="cpu"))
    return stop, spaced_training_steps(total if stop is None else stop, count, generator)


def unroll_sparse_sampling(
    initial: torch.Tensor,
    timesteps: torch.Tensor,
    selected: range,
    advance: Callable[..., torch.Tensor],
    *,
    stop_after: int | None = None,
) -> torch.Tensor:
    """Carry the live Euler state through the sampling endpoint or clean projection."""
    latent = initial.detach()
    visited = timesteps.shape[-1] if stop_after is None else stop_after
    for index in range(visited):
        current = timesteps[..., index]
        terminal = index + 1 == visited
        following = torch.zeros_like(current) if terminal else timesteps[..., index + 1]
        latent = advance(
            latent,
            current,
            following,
            prediction_gradients="parameters" if index in selected else "none",
            project_clean=terminal and stop_after is not None,
        )
    return latent


class DRTuneTrainer(BaseTrainer):
    """Score the Euler endpoint or early clean estimate with sparse supervision."""

    execution_contract = DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    paradigm = "decoupled"

    def _validate_paradigm_dynamics(self) -> None:
        scheduler = self.adapter.scheduler
        if (
            not isinstance(scheduler, FlowMatchEulerDiscreteSDEScheduler)
            or scheduler.dynamics_type != "ODE"
        ):
            raise ValueError("DRTune requires the FlowMatch Euler ODE scheduler")
        distributed_type = self.accelerator.distributed_type
        if distributed_type not in (
            DistributedType.NO,
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
            DistributedType.FSDP,
        ):
            raise ValueError("DRTune requires local, data-only DDP or FSDP2 policy execution")
        if distributed_type == DistributedType.FSDP and not self.accelerator.is_fsdp2:
            raise ValueError("DRTune requires FSDP2 when the FSDP policy backend is selected")

    def runtime_execution_identity_payload(self) -> dict:
        """Bind reward revisions and the rank-consistent mask to exact resume."""
        identity = super().runtime_execution_identity_payload()
        identity["differentiable_scorers"] = scorer_identity(self.reward_models)
        identity["drtune_mask"] = "rank_shared_euler_stride_uniform_offset_v1"
        if self.training_args.drtune_stop_step_range is None:
            identity["training"].pop("drtune_stop_step_range", None)
        else:
            identity["drtune_stop_rule"] = "rank_shared_uniform_prefix_clean_projection_v1"
        return identity

    def _initialize_snapshots(self) -> None:
        super()._initialize_snapshots()
        self.adapter._require_component("vae").eval().requires_grad_(False)

    def sample(self) -> list[BaseSample]:
        """Acquire the same native Euler inputs as DRaFT-K."""
        return acquire_euler_inputs(self)

    def _sample_endpoint(
        self, batch: StackedSampleBatch, inner_epoch: int = 0, batch_index: int = 0
    ) -> tuple[LatentState, dict[str, int]]:
        total = batch["timesteps"].shape[-1]
        # All prepared ranks use the same call order, even with different prompts.
        generator = identity_timestep_generator(self, (batch_index,), inner_epoch)
        stop, selected = sparse_training_schedule(
            total,
            self.training_args.drtune_backprop_steps,
            self.training_args.drtune_stop_step_range,
            generator,
        )

        def unroll(initial, timesteps, advance):
            return unroll_sparse_sampling(initial, timesteps, selected, advance, stop_after=stop)

        endpoint = sample_euler_endpoint(self, batch, unroll)
        return endpoint, {
            "drtune/visited_steps": total if stop is None else stop,
            "drtune/backprop_steps": len(selected),
        }

    def optimize(self, samples: list[BaseSample]) -> None:
        """Differentiate the endpoint reward using the sparse Euler graph."""
        optimize_euler_reward(self, samples, self._sample_endpoint)
