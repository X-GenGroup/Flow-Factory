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

"""Native Euler acquisition and terminal feedback for reward-gradient trainers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch.nn.parallel import DistributedDataParallel

from ...models.model_bundle import composed_forward
from ...samples import BaseSample, ComponentTimes, LatentState, StackedSampleBatch
from .forward_kwargs import replay_forward_kwargs
from .reward_gradient import reward_source_names

if TYPE_CHECKING:
    from ..abc import BaseTrainer


def acquire_euler_inputs(trainer: BaseTrainer) -> list[BaseSample]:
    """Acquire native initial noise, conditioning and timesteps."""
    return trainer.generate_samples(
        reward_buffer=None,
        compute_log_prob=False,
        trajectory_indices=[0],
        initial_state_only=True,
    )


def sample_euler_endpoint(
    trainer: BaseTrainer,
    batch: StackedSampleBatch,
    unroll: Callable[[torch.Tensor, torch.Tensor, Callable[..., torch.Tensor]], torch.Tensor],
) -> LatentState:
    """Run the trainer's derivative policy through the native Euler forward."""
    initial = batch["all_latents"][:, 0]
    kwargs = replay_forward_kwargs(trainer, batch)

    def advance(state, current, following, *, prediction_gradients="full"):
        result = trainer.adapter.forward_state(
            batch=batch,
            state=LatentState({"latent": state}),
            times=ComponentTimes(timestep={"latent": current}, next_timestep={"latent": following}),
            compute_log_prob=False,
            return_fields=("next_latents",),
            **{**kwargs, "prediction_gradients": prediction_gradients},
        )
        return trainer.adapter.cast_latents(
            result.next_state.components["latent"], default_dtype=initial.dtype
        )

    def sample():
        return unroll(initial, batch["timesteps"], advance)

    endpoint = (
        composed_forward(trainer.model_bundle, sample)
        if isinstance(trainer.model_bundle, DistributedDataParallel)
        else sample()
    )
    return LatentState({"latent": endpoint})


def optimize_euler_reward(
    trainer: BaseTrainer,
    samples: list[BaseSample],
    sample_endpoint: Callable[[StackedSampleBatch, int, int], LatentState],
) -> None:
    """Optimize one terminal reward graph per accumulated microbatch."""
    loss_info = defaultdict(list)
    trainer.adapter.train()
    for inner_epoch in range(trainer.training_args.num_inner_epochs):
        ordered = trainer._order_samples_for_optimize(samples, inner_epoch)
        for batch_index, batch in enumerate(
            trainer._iter_prefetched_batches(ordered, trainer.training_args.per_device_batch_size)
        ):
            with trainer.accumulate_gradients():
                with trainer.autocast():
                    endpoint = sample_endpoint(batch, inner_epoch, batch_index)
                    media = trainer.adapter.decode_differentiable_state(endpoint, batch=batch)
                feedback = trainer.differentiable_feedback(
                    prompt=list(batch["prompt"]),
                    source_names=reward_source_names(trainer.config.data_args, batch.samples),
                    **media,
                )
                terms = []
                for name, scores in feedback.scores.items():
                    terms.append(-(scores * feedback.weights[name]).mean())
                    loss_info[f"reward/{name}"].append(scores.detach().mean())
                loss = torch.stack(terms).sum()
                loss_info["loss"].append(loss.detach())
                trainer._backward_full_output_loss(loss)
                trainer.log_train_samples(batch.samples, media=media, rewards=feedback.scores)
                if trainer.accelerator.sync_gradients:
                    loss_info = trainer._apply_optimizer_step(loss_info)
