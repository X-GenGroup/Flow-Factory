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

# src/flow_factory/trainers/rl/grpo.py
"""
Group Relative Policy Optimization (GRPO) Trainer.
Implements GRPO algorithm for flow matching models.
"""

import os
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Literal, Mapping, Union

import numpy as np
import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...contracts.reward_overlap import COUPLED_REWARD_OPTIMIZATION_OVERLAP
from ...hparams import GRPOTrainingArguments
from ...samples import (
    BaseSample,
    LatentState,
    MultiModalStepOutput,
    ReplayStep,
)
from ...utils.base import create_generator_by_prompt
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import TrajectoryCollector, compute_trajectory_indices
from ..abc import BaseTrainer
from ..coupled import CoupledReplayRuntimeMixin

logger = setup_logger(__name__)


# ============================ GRPO Trainer ============================
class GRPOTrainer(CoupledReplayRuntimeMixin, BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    References:
    [1] Flow-GRPO: Training Flow Matching Models via Online RL
        - https://arxiv.org/abs/2505.05470
    """

    # Coupled paradigm: rollout log-probs feed the PPO ratio, so lossy rollout
    # acceleration is disallowed (constraints.md #7). Inherited by GRPOGuard/DPPO.
    paradigm = "coupled"
    reward_optimization_overlap_contract = COUPLED_REWARD_OPTIMIZATION_OVERLAP

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: GRPOTrainingArguments
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO (stores full trajectory + log-probs)."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
        )

    # =========================== Reward / advantage (Stages 4--5) ============================
    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): PPO-style clipped loss and optional KL."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        train_step_indices = self.adapter.get_train_step_indices()
        # Forward fields: the ratio needs `log_prob`, and `dt` keeps the legacy request
        # shape; the reference KL space adds its own comparison field.
        if self.enable_kl_loss:
            _, ref_return_field = self._kl_space_fields(self.training_args.kl_type, "kl_type")
            requested_fields = (
                {"log_prob", "velocity", "dt"}
                if self.training_args.kl_type == "v-based"
                else {"log_prob", "next_latents", "next_latents_mean", "dt"}
            )
        else:
            ref_return_field = None
            requested_fields = {"log_prob", "dt"}
        return_fields = self._canonical_return_fields(requested_fields)
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle unless disabled for pack-composition-dependent adapters.
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            # Reload each micro-batch onto the device (H2D when samples are
            # CPU-offloaded; a no-op when GPU-resident). _iter_prefetched_batches
            # overlaps the next batch's H2D with compute when offload is enabled.
            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Iterate through timesteps
                for timestep_index in tqdm(
                    train_step_indices,
                    desc=f"Epoch {self.epoch} Timestep",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    step_index = int(timestep_index)
                    with self.accumulate_gradients():
                        # 1. Prepare inputs
                        replay = self.adapter.get_replay_step(batch, step_index)
                        old_log_prob = self._require_replay_log_prob(replay, step_index)
                        # 2. Forward pass
                        with self.autocast():
                            output = self._replay_forward(batch, replay, return_fields)

                        # 3. Compute loss
                        # Clip advantages
                        adv = batch["advantage"]
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # PPO-style clipped loss
                        new_log_prob = self._require_policy_log_prob(
                            output, step_index, self._replay_batch_size(replay)
                        )
                        ratio = torch.exp(new_log_prob - old_log_prob)
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(
                            ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1]
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. Compute KL-div
                        if self.enable_kl_loss:
                            with self.autocast():
                                ref_output = self._reference_forward(
                                    batch, replay, (ref_return_field,)
                                )
                                # kl_div must be computed outside `torch.no_grad()` for correct gradient behavior.
                                # See: issue #122, PR #123 (https://github.com/X-GenGroup/Flow-Factory/pull/123)
                                kl_div = self._reference_kl_divergence(output, ref_output, replay)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 5. Log per-timestep info
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

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)

    # =========================== Advantage Computation ============================


# ============================ GRPO-Guard Trainer ============================
class GRPOGuardTrainer(GRPOTrainer):
    """
    GRPOGuard Trainer with reweighted loss.
    References:
    [1] GRPO-Guard: https://arxiv.org/abs/2510.22319
    [2] Temp-FlowGRPO: https://arxiv.org/abs/2508.04324
    """

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO-Guard with transition means."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
            extra_call_back_kwargs=["next_latents_mean"],
        )

    def _guard_ratio(
        self,
        output: MultiModalStepOutput,
        replay: ReplayStep,
        old_state_mean: LatentState,
        step_index: int = 0,
    ) -> torch.Tensor:
        """Variance-reweighted importance ratio, per component then DOF-reduced.

        Each component contributes ``(delta_log_prob * scale + mse / (2 * scale))``
        with ``scale = sqrt(-dt) * std_dev_t``; the terms are combined by active
        stochastic degrees of freedom before a single exponentiation.

        Args:
            output: Current-policy step output.
            replay: Stored rollout transition being replayed.
            old_state_mean: Stored rollout transition mean.
            step_index: Rollout position, reported by validation errors.
        """
        expected_names = self.adapter.trajectory_component_order
        new_state_mean = self._require_output_state(output, "next_state_mean", "policy")
        std_dev_t = self._require_component_mapping(output.std_dev_t, "std_dev_t", "policy output")
        dt = self._require_component_mapping(output.dt, "dt", "policy output")
        new_log_probs = self._require_component_mapping(
            output.component_log_probs, "component_log_probs", "policy output"
        )
        old_log_probs = self._require_component_mapping(
            replay.component_log_probs, "component_log_probs", "stored rollout"
        )
        if old_state_mean.component_names != expected_names:
            raise ValueError(
                f"expected stored rollout next_latents_mean in component order "
                f"{expected_names}, received {old_state_mean.component_names}"
            )
        batch_size = self._replay_batch_size(replay)
        # Guard reweights per component, but the joint log probability must still be a
        # well-formed (B,) tensor before any PPO arithmetic consumes this ratio.
        self._require_policy_log_prob(output, step_index, batch_size)
        component_mse = self.adapter.reduce_component_latent_values(
            self._component_squared_error_elements(
                new_state_mean, old_state_mean, "stored rollout"
            ),
            state=replay.state,
        )
        guard_terms = {}
        for name in expected_names:
            scale_factor = self._effective_transition_std(
                name,
                self._scalar_component_statistic(std_dev_t, "std_dev_t", name, batch_size),
                self._scalar_component_statistic(dt, "dt", name, batch_size),
                context="GRPO-Guard ratio",
            )
            guard_terms[name] = (
                new_log_probs[name] - old_log_probs[name]
            ) * scale_factor + component_mse[name] / (2 * scale_factor)
        return torch.exp(
            self.adapter.reduce_latent_values(
                guard_terms,
                active_numel=self.adapter.get_state_active_numel(replay.state),
                state=replay.state,
            )
        )

    def _scalar_component_statistic(
        self,
        values: Mapping[str, torch.Tensor],
        field: str,
        component: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Flatten a broadcast ``(B, 1, ...)`` scheduler statistic to ``(B,)``.

        The Guard ratio combines statistics with ``(B,)`` log probabilities and
        ``(B,)`` component errors, so a leftover broadcast axis would silently
        produce a cross-product instead of a per-sample scale.
        """
        statistic = values[component]
        if statistic.ndim < 1 or statistic.shape[0] != batch_size:
            raise ValueError(
                f"expected policy output {field} for component {component!r} to use batch "
                f"size {batch_size}, received shape {tuple(statistic.shape)}"
            )
        if statistic.numel() != batch_size:
            raise ValueError(
                f"expected policy output {field} for component {component!r} to hold exactly "
                f"one value per sample, received shape {tuple(statistic.shape)}"
            )
        return statistic.reshape(batch_size)

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): GRPO-Guard reweighted loss and optional KL."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        train_step_indices = self.adapter.get_train_step_indices()
        # The reweighted ratio always needs the transition mean, its std, and dt.
        requested_fields = {"log_prob", "next_latents_mean", "std_dev_t", "dt"}
        if self.enable_kl_loss:
            _, ref_return_field = self._kl_space_fields(self.training_args.kl_type, "kl_type")
            requested_fields.add(ref_return_field)
        else:
            ref_return_field = None
        return_fields = self._canonical_return_fields(requested_fields)
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle unless disabled for pack-composition-dependent adapters.
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            # Reload each micro-batch onto the device (H2D when offloaded);
            # _iter_prefetched_batches overlaps the next batch's H2D with compute.
            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Iterate through timesteps
                for timestep_index in tqdm(
                    train_step_indices,
                    desc=f"Epoch {self.epoch} Timestep",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    step_index = int(timestep_index)
                    with self.accumulate_gradients():
                        # 1. Prepare inputs
                        replay = self.adapter.get_replay_step(batch, step_index)
                        self._require_replay_log_prob(replay, step_index)
                        old_state_mean = self.adapter.get_replay_callback(
                            batch, step_index, "next_latents_mean"
                        )
                        # 2. Forward pass
                        with self.autocast():
                            output = self._replay_forward(batch, replay, return_fields)

                        # 3. Compute loss
                        # Clip advantages
                        adv = batch["advantage"]
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # Reweighted ratio
                        ratio = self._guard_ratio(output, replay, old_state_mean, step_index)
                        # PPO-style clipped loss
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(
                            ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1]
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. Compute KL-div
                        if self.enable_kl_loss:
                            with self.autocast():
                                ref_output = self._reference_forward(
                                    batch, replay, (ref_return_field,)
                                )
                                # kl_div must be computed outside `torch.no_grad()` for correct gradient behavior.
                                # See: issue #122, PR #123 (https://github.com/X-GenGroup/Flow-Factory/pull/123)
                                kl_div = self._reference_kl_divergence(output, ref_output, replay)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 5. Log per-timestep info
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

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)
