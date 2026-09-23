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

# src/flow_factory/trainers/rl/awm.py
"""
Advantage Weighted Matching (AWM) Trainer.
References:
[1] Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models
    - https://arxiv.org/pdf/2509.25050
"""

import math
import os
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ...contracts.reward_overlap import GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP
from ...hparams import AWMTrainingArguments
from ...rewards import BaseRewardModel, RewardBuffer
from ...samples import BaseSample, ComponentTimes, LatentState, NoisedState, StackedSampleBatch
from ...utils.base import create_generator_by_prompt
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import TimeSampler, flow_match_sigma
from ..abc import BaseTrainer
from ..common.state_validation import require_component_sigmas
from ..decoupled import iter_decoupled_replay_batches, iter_decoupled_steps
from ..forward_process import forward_velocity_state

logger = setup_logger(__name__)

WEIGHTING_SCHEMES = ("Uniform", "t", "t**2", "huber", "ghuber")


class AWMPrecomputedStep(NamedTuple):
    """Forward-process state for one training time plus the sampling-policy log-prob."""

    times: ComponentTimes
    noised: NoisedState
    log_prob: torch.Tensor


# ============================ AWM Trainer ============================
class AWMTrainer(BaseTrainer):
    """
    Advantage Weighted Matching (AWM) Trainer.
    References:
    [1] Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models
        - https://arxiv.org/pdf/2509.25050
    """

    # Decoupled paradigm: lossy rollout acceleration is permitted (constraints.md #7).
    paradigm = "decoupled"
    reward_optimization_overlap_contract = GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # AWM-specific config (from AWMTrainingArguments)
        self.training_args: AWMTrainingArguments
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.weighting = self.training_args.awm_weighting
        self.ghuber_power = self.training_args.ghuber_power
        self.off_policy = self.training_args.off_policy
        self.num_train_timesteps = self.training_args.num_train_timesteps
        self.timestep_range = self.training_args.timestep_range

        # KL regularization
        self.kl_beta = self.training_args.kl_beta
        self.ema_kl_beta = self.training_args.ema_kl_beta
        self.kl_type = self.training_args.kl_type

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.kl_beta > 0.0

    @property
    def enable_ema_kl_loss(self) -> bool:
        """Check if EMA-based KL penalty is enabled."""
        return self.ema_kl_beta > 0.0

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
        """Generate rollouts for AWM (final latents only)."""
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=[-1],
        )

    # =========================== Optimization Loop ============================
    @staticmethod
    def apply_matching_weighting(
        log_prob: torch.Tensor,
        sigma: torch.Tensor,
        weighting: Literal["Uniform", "t", "t**2", "huber", "ghuber"] = "Uniform",
        ghuber_power: float = 0.25,
    ) -> torch.Tensor:
        """
        Reweight one per-sample matching value with its own noise level.

        Args:
            log_prob: Per-sample matching value (B,), negative mean squared error.
            sigma: Per-sample noise level (B,) of the component being matched.
            weighting: Weighting scheme for the loss.
            ghuber_power: Power parameter for generalized huber loss.

        Returns:
            Reweighted per-sample value of shape (B,).
        """
        if weighting == "Uniform":
            return log_prob
        if weighting == "t":
            return log_prob * sigma
        if weighting == "t**2":
            return log_prob * sigma**2
        if weighting == "huber":
            return -(torch.sqrt(-log_prob + 1e-10) - 1e-5) * sigma
        if weighting == "ghuber":
            eps = torch.tensor(1e-10, device=log_prob.device, dtype=log_prob.dtype)
            return (
                -(torch.pow(-log_prob + eps, ghuber_power) - torch.pow(eps, ghuber_power))
                * sigma
                / ghuber_power
            )
        raise ValueError(
            f"expected awm_weighting one of {WEIGHTING_SCHEMES}, received {weighting!r}"
        )

    @staticmethod
    def compute_weighted_log_prob(
        model_output: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        weighting: Literal["Uniform", "t", "t**2", "huber", "ghuber"] = "Uniform",
        ghuber_power: float = 0.25,
    ) -> torch.Tensor:
        """
        Compute weighted log probability (matching loss) for one tensor.

        Args:
            model_output: Model's velocity prediction, shape varies by model.
            target: Target velocity = noise - clean_latents, same shape as model_output.
            timestep: Scheduler-scale timesteps (B,) in ``[0, 1000]``; weighting uses ``σ = t/1000``.
            weighting: Weighting scheme for the loss.
            ghuber_power: Power parameter for generalized huber loss.

        Returns:
            Weighted log probability tensor of shape (B,).
        """
        # Matching loss (negative MSE as log prob)
        # Mean over all dimensions except batch (dim 0)
        log_prob = -((model_output.double() - target.double()) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))  # Dynamic: works for any shape

        return AWMTrainer.apply_matching_weighting(
            log_prob,
            flow_match_sigma(timestep.view(-1)),
            weighting,
            ghuber_power,
        ).float()

    def _matching_log_prob(
        self,
        velocity: LatentState,
        noised: NoisedState,
        times: ComponentTimes,
    ) -> torch.Tensor:
        """Compute the weighted matching log probability of a velocity prediction.

        Each component is matched against its own target velocity in double
        precision and reweighted by its own sigma before the components are
        combined by their active degrees of freedom, so a component's noise level
        never leaks into another's weighting.

        Args:
            velocity: Predicted velocity per component.
            noised: Forward-noised state carrying the per-component target velocity.
            times: Component times supplying each component's sigma.

        Returns:
            Joint per-sample log probability of shape ``(B,)`` in float32.
        """
        sigmas = require_component_sigmas(self, times)
        errors = {
            name: -(
                (
                    velocity.components[name].double()
                    - noised.target_velocity.components[name].double()
                )
                ** 2
            )
            for name in self.adapter.trajectory_component_order
        }
        component_values = self.adapter.reduce_component_latent_values(errors, state=noised.state)
        weighted = {
            name: self.apply_matching_weighting(
                value, sigmas[name], self.weighting, self.ghuber_power
            )
            for name, value in component_values.items()
        }
        return self.adapter.reduce_latent_values(
            weighted,
            active_numel=self.adapter.get_state_active_numel(noised.state),
            state=noised.state,
        ).float()

    def _precompute_old_log_probs(
        self,
        batch: StackedSampleBatch,
        clean_state: LatentState,
        batch_size: int,
    ) -> List[AWMPrecomputedStep]:
        """Sample training times, noise the terminal state once per time, record old log-probs.

        The forward-process states are kept so the current-policy and reference
        passes reuse the exact same noise instead of drawing again, which keeps the
        PPO ratio comparable and the RNG stream identical to the legacy loop.

        Args:
            batch: Collated sample batch supplying conditioning arguments.
            clean_state: Terminal clean state to noise.
            batch_size: Number of samples in the batch.

        Returns:
            One precomputed step per training timestep, in sampling order.
        """
        self.adapter.rollout()
        steps: List[AWMPrecomputedStep] = []
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
                    AWMPrecomputedStep(
                        times=times,
                        noised=noised,
                        log_prob=self._matching_log_prob(velocity, noised, times).detach(),
                    )
                )
        return steps

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): AWM weighted matching with optional KL.

        Per-batch interleave (matches the official AWM paper):
        for each micro-batch -> lazy reload to GPU -> precompute old log-probs
        under the sampling policy (rollout + sampling_context) -> train per
        timestep under the current policy (train + forward / backward /
        optimizer step).

        Unlike GRPO which iterates over trajectory timesteps, AWM decouples
        sampling / training timesteps and passes over all sampled timesteps
        per batch.

        See ``.agents/knowledge/topics/sample_lifecycle.md`` for the memory,
        train-inference consistency, and RNG-order trade-offs.
        """
        for inner_epoch in range(self.training_args.num_inner_epochs):
            loss_info = defaultdict(list)

            for replay_batch in iter_decoupled_replay_batches(
                self,
                samples,
                inner_epoch,
                self._precompute_old_log_probs,
            ):
                batch = replay_batch.batch

                # ---------- Train this batch under current policy ----------
                # Get advantages and clip (batch-scoped, shared across timesteps)
                adv = batch["advantage"]
                adv_clip_range = self.training_args.adv_clip_range
                adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                ratio_clip_range = self.training_args.clip_range

                for _, step in iter_decoupled_steps(self, replay_batch.steps):
                    with self.accumulate_gradients():
                        # 1. Reuse the forward-process state drawn for this timestep
                        old_log_prob = step.log_prob  # (B,)

                        # 2. Forward pass for current policy
                        with self.autocast():
                            velocity = forward_velocity_state(
                                self, batch, step.noised.state, step.times, source="policy"
                            )
                            log_prob = self._matching_log_prob(
                                velocity, step.noised, step.times
                            )  # (B,)

                        # 3. Compute PPO-style clipped loss
                        ratio = torch.exp(log_prob - old_log_prob)
                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(
                            ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1]
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. KL regularization with reference model
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
                                # KL-div in velocity space
                                kl_div = self._velocity_kl(velocity, ref_velocity, step.noised)
                                kl_loss = self.kl_beta * kl_div.mean()
                                loss = loss + kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 5. EMA-based KL regularization
                        if self.enable_ema_kl_loss:
                            with self.autocast():
                                with torch.no_grad(), self.adapter.use_ema_parameters():
                                    ema_velocity = forward_velocity_state(
                                        self, batch, step.noised.state, step.times, source="EMA"
                                    )
                                # KL-div in velocity space
                                ema_kl = self._velocity_kl(velocity, ema_velocity, step.noised)
                                ema_kl_loss = self.ema_kl_beta * ema_kl.mean()
                                loss = loss + ema_kl_loss
                                loss_info["ema_kl_div"].append(ema_kl.detach())
                                loss_info["ema_kl_loss"].append(ema_kl_loss.detach())

                        # 6. Log per-timestep info
                        loss_info["ratio"].append(ratio.detach())
                        loss_info["unclipped_loss"].append(unclipped_loss.detach())
                        loss_info["clipped_loss"].append(clipped_loss.detach())
                        loss_info["policy_loss"].append(policy_loss.detach())
                        loss_info["loss"].append(loss.detach())
                        clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                        clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                        loss_info["clip_frac_high"].append(clip_frac_high.detach())
                        loss_info["clip_frac_low"].append(clip_frac_low.detach())
                        loss_info["clip_frac_total"].append(
                            (clip_frac_high + clip_frac_low).detach()
                        )

                        # 6. Backward pass and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)
