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

# src/flow_factory/trainers/rl/nft.py
"""
DiffusionNFT Trainer.
Reference:
[1] DiffusionNFT: Online Diffusion Reinforcement with Forward Process
    - https://arxiv.org/abs/2509.16117
"""

import os
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from ...contracts.reward_overlap import GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP
from ...hparams import NFTTrainingArguments
from ...rewards import RewardBuffer
from ...samples import BaseSample, ComponentTimes, LatentState, NoisedState, StackedSampleBatch
from ...utils.base import create_generator_by_prompt
from ...utils.logger_utils import setup_logger
from ..abc import BaseTrainer
from ..decoupled import iter_decoupled_replay_batches, iter_decoupled_steps
from ..forward_process import forward_velocity_state

logger = setup_logger(__name__)


class NFTPrecomputedStep(NamedTuple):
    """Forward-process state for one training time plus the sampling-policy velocity."""

    times: ComponentTimes
    noised: NoisedState
    velocity: LatentState


class DiffusionNFTTrainer(BaseTrainer):
    """
    DiffusionNFT Trainer with off-policy and continuous timestep support.
    Reference: https://arxiv.org/abs/2509.16117
    """

    # Decoupled paradigm: rollout trajectory log-probs do not enter the loss,
    # so lossy rollout acceleration is permitted (constraints.md #7).
    paradigm = "decoupled"
    reward_optimization_overlap_contract = GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # NFT-specific config (from NFTTrainingArguments)
        self.training_args: NFTTrainingArguments
        self.nft_beta = self.training_args.nft_beta
        self.off_policy = self.training_args.off_policy

        # Timestep sampling config
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.num_train_timesteps = self.training_args.num_train_timesteps
        self.timestep_range = self.training_args.timestep_range

        self.kl_type = self.training_args.kl_type

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    @contextmanager
    def sampling_context(self):
        """Context manager for sampling with or without EMA parameters."""
        if self.off_policy:
            with self.adapter.use_ema_parameters():
                yield
        else:
            yield

    # =========================== Advantage Computation ============================
    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DiffusionNFT (final latents only)."""
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=[-1],
        )

    # =========================== Optimization Loop ============================
    def _precompute_sampling_policy_steps(
        self,
        batch: StackedSampleBatch,
        clean_state: LatentState,
        batch_size: int,
    ) -> List[NFTPrecomputedStep]:
        """Sample training times, noise the terminal state once per time, record old velocities.

        The forward-process states are kept so the current-policy pass reuses the
        exact same noise instead of drawing again, which keeps the RNG stream and
        the loss inputs identical across the two passes.

        Args:
            batch: Collated sample batch supplying conditioning arguments.
            clean_state: Terminal clean state to noise.
            batch_size: Number of samples in the batch.

        Returns:
            One precomputed step per training timestep, in sampling order.
        """
        self.adapter.rollout()
        steps: List[NFTPrecomputedStep] = []
        with torch.no_grad(), self.autocast(), self.sampling_context():
            all_timesteps = self._sample_timesteps(batch_size)  # (T, B)
            for t_idx in range(self.num_train_timesteps):
                times = self.adapter.build_training_component_times(
                    all_timesteps[t_idx], batch=batch
                )
                noised = self.adapter.add_forward_process_noise(clean_state, times)
                velocity = forward_velocity_state(
                    self, batch, noised.state, times, source="sampling policy"
                )
                steps.append(
                    NFTPrecomputedStep(
                        times=times,
                        noised=noised,
                        velocity=LatentState(
                            {name: value.detach() for name, value in velocity.components.items()}
                        ),
                    )
                )
        return steps

    def _normalized_squared_errors(
        self,
        x0_prediction: Dict[str, torch.Tensor],
        clean_state: LatentState,
        noised: NoisedState,
    ) -> Dict[str, torch.Tensor]:
        """Divide each component's squared x0 error by its own deviation scale.

        The scale is the detached mean absolute x0 deviation over the component's
        active elements, clipped at ``1e-5``.

        Args:
            x0_prediction: Predicted clean latents per component.
            clean_state: Terminal clean state per component.
            noised: Forward-noised state supplying per-sample reduction context.

        Returns:
            Normalized squared errors per component, elementwise.
        """
        with torch.no_grad():
            deviations = {
                name: torch.abs(value.double() - clean_state.components[name].double())
                for name, value in x0_prediction.items()
            }
            scales = self.adapter.reduce_component_latent_values(deviations, state=noised.state)
        errors: Dict[str, torch.Tensor] = {}
        for name, value in x0_prediction.items():
            clean = clean_state.components[name]
            scale = scales[name].clip(min=1e-5).reshape((-1,) + (1,) * (clean.ndim - 1))
            errors[name] = (value - clean) ** 2 / scale
        return errors

    def _matching_losses(
        self,
        clean_state: LatentState,
        noised: NoisedState,
        times: ComponentTimes,
        new_velocity: LatentState,
        old_velocity: LatentState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample positive and negative x0 matching losses.

        Args:
            clean_state: Terminal clean state the forward process noised.
            noised: Forward-noised state shared by both policy passes.
            times: Component times supplying each component's sigma.
            new_velocity: Current-policy velocity per component.
            old_velocity: Sampling-policy velocity per component.

        Returns:
            Positive and negative per-sample losses, both of shape ``(B,)``.
        """
        beta = self.nft_beta
        positive_velocity: Dict[str, torch.Tensor] = {}
        negative_velocity: Dict[str, torch.Tensor] = {}
        for name in self.adapter.trajectory_component_order:
            new_v = new_velocity.components[name]
            old_v = old_velocity.components[name]
            positive_velocity[name] = beta * new_v + (1 - beta) * old_v
            negative_velocity[name] = (1.0 + beta) * old_v - beta * new_v
        positive_x0 = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            LatentState(positive_velocity, active_masks=noised.state.active_masks),
        ).components
        negative_x0 = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            LatentState(negative_velocity, active_masks=noised.state.active_masks),
        ).components
        positive_errors = self._normalized_squared_errors(positive_x0, clean_state, noised)
        negative_errors = self._normalized_squared_errors(negative_x0, clean_state, noised)
        return (
            self.adapter.reduce_latent_values(positive_errors, state=noised.state),
            self.adapter.reduce_latent_values(negative_errors, state=noised.state),
        )

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): NFT matching loss with optional KL.

        Per-batch interleave (matches the official DiffusionNFT impl):
        for each micro-batch -> lazy reload to GPU -> precompute old v
        predictions under the sampling policy (rollout + sampling_context)
        -> train per timestep under the current policy (train + forward /
        backward / optimizer step).

        See ``.agents/knowledge/topics/sample_lifecycle.md`` for the memory,
        train-inference consistency, and RNG-order trade-offs.
        """
        for inner_epoch in range(self.training_args.num_inner_epochs):
            loss_info = defaultdict(list)

            for replay_batch in iter_decoupled_replay_batches(
                self,
                samples,
                inner_epoch,
                self._precompute_sampling_policy_steps,
            ):
                batch = replay_batch.batch
                clean_state = replay_batch.clean_state

                # ---------- Train this batch under current policy ----------
                for _, step in iter_decoupled_steps(self, replay_batch.steps):
                    with self.accumulate_gradients():
                        # 1. Reuse the forward-process state drawn for this timestep
                        # 2. Forward pass for current policy
                        with self.autocast():
                            new_velocity = forward_velocity_state(
                                self, batch, step.noised.state, step.times, source="policy"
                            )

                        # 3. Compute NFT loss
                        adv = batch["advantage"]
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                        # Normalize advantage to [0, 1]
                        normalized_adv = (adv / max(adv_clip_range)) / 2.0 + 0.5
                        r = torch.clamp(normalized_adv, 0, 1)

                        positive_loss, negative_loss = self._matching_losses(
                            clean_state, step.noised, step.times, new_velocity, step.velocity
                        )

                        # Combined loss
                        ori_policy_loss = (
                            r * positive_loss + (1.0 - r) * negative_loss
                        ) / self.nft_beta
                        policy_loss = (ori_policy_loss * adv_clip_range[1]).mean()
                        loss = policy_loss

                        # 4. KL penalty
                        if self.enable_kl_loss:
                            with self.autocast():
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_velocity = forward_velocity_state(
                                        self,
                                        batch,
                                        step.noised.state,
                                        step.times,
                                        source="reference",
                                    )
                                # KL-loss in v-space
                                kl_div = self._velocity_kl(new_velocity, ref_velocity, step.noised)
                                kl_loss = self.training_args.kl_beta * kl_div.mean()
                                loss = loss + kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        loss_info["policy_loss"].append(policy_loss.detach())
                        loss_info["unweighted_policy_loss"].append(ori_policy_loss.mean().detach())
                        loss_info["loss"].append(loss.detach())

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)
