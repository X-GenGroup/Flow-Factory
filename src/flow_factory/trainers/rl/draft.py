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

from collections import defaultdict
from collections.abc import Callable

import torch
from accelerate.utils import DistributedType
from torch.nn.parallel import DistributedDataParallel

from ...contracts.execution import DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
from ...models.model_bundle import composed_forward
from ...samples import BaseSample, ComponentTimes, LatentState, StackedSampleBatch
from ...scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from ..abc import BaseTrainer
from ..common.forward_kwargs import replay_forward_kwargs
from ..common.reward_gradient import reward_source_names, scorer_identity


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
        return self.generate_samples(
            reward_buffer=None,
            compute_log_prob=False,
            trajectory_indices=[0],
            initial_state_only=True,
        )

    def _sample_endpoint(self, batch: StackedSampleBatch) -> LatentState:
        initial = batch["all_latents"][:, 0]
        kwargs = replay_forward_kwargs(self, batch)

        def advance(state, current, following):
            result = self.adapter.forward_state(
                batch=batch,
                state=LatentState({"latent": state}),
                times=ComponentTimes(
                    timestep={"latent": current}, next_timestep={"latent": following}
                ),
                compute_log_prob=False,
                return_fields=("next_latents",),
                **kwargs,
            )
            return self.adapter.cast_latents(
                result.next_state.components["latent"], default_dtype=initial.dtype
            )

        def sample():
            return unroll_sampling_suffix(
                initial, batch["timesteps"], self.training_args.draft_backprop_steps, advance
            )

        endpoint = (
            composed_forward(self.model_bundle, sample)
            if isinstance(self.model_bundle, DistributedDataParallel)
            else sample()
        )
        return LatentState({"latent": endpoint})

    def optimize(self, samples: list[BaseSample]) -> None:
        """Differentiate one complete suffix and reward per accumulated microstep."""
        loss_info = defaultdict(list)
        self.adapter.train()
        for inner_epoch in range(self.training_args.num_inner_epochs):
            ordered = self._order_samples_for_optimize(samples, inner_epoch)
            for batch in self._iter_prefetched_batches(
                ordered, self.training_args.per_device_batch_size
            ):
                with self.accumulate_gradients():
                    with self.autocast():
                        endpoint = self._sample_endpoint(batch)
                        media = self.adapter.decode_differentiable_state(endpoint, batch=batch)
                    feedback = self.differentiable_feedback(
                        prompt=list(batch["prompt"]),
                        source_names=reward_source_names(self.config.data_args, batch.samples),
                        **media,
                    )
                    terms = []
                    for name, scores in feedback.scores.items():
                        terms.append(-(scores * feedback.weights[name]).mean())
                        loss_info[f"reward/{name}"].append(scores.detach().mean())
                    loss = torch.stack(terms).sum()
                    loss_info["loss"].append(loss.detach())
                    self._backward_full_output_loss(loss)
                    self.log_train_samples(batch.samples, media=media, rewards=feedback.scores)
                    if self.accelerator.sync_gradients:
                        loss_info = self._apply_optimizer_step(loss_info)
