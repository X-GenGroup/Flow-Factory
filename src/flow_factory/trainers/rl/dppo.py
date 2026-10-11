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

# src/flow_factory/trainers/rl/dppo.py
"""Flow-DPPO trainer.

DPPO is a strict Flow-GRPO variant: it keeps GRPO's group advantages and the
optional KL-vs-reference penalty, but replaces the PPO ratio-clip with a KL
trust-region mask. A sample's gradient is zeroed when its per-step
KL(current || rollout-old) exceeds ``kl_mask_threshold`` and the update would
push the action further in the wrong direction.
"""

from collections import defaultdict
from functools import partial
from typing import Dict, List

import torch
import tqdm as tqdm_

from ...hparams import DPPOTrainingArguments
from ...samples import BaseSample, LatentState, MultiModalStepOutput, ReplayStep
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import compute_trajectory_indices
from ..registry import register_trainer
from .grpo import GRPOTrainer

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)


def gaussian_kl_div(p: torch.Tensor, q: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL-style squared error between Gaussian means scaled by variance (x-space)."""
    return (p - q) ** 2 / (2 * sigma**2)


# ============================ Flow-DPPO Trainer ============================
@register_trainer("dppo")
class DPPOTrainer(GRPOTrainer):
    """Flow-DPPO Trainer: GRPO with a KL trust-region mask instead of PPO clipping.

    References:
        [1] Flow-GRPO: Training Flow Matching Models via Online RL
            (https://arxiv.org/abs/2505.05470)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: DPPOTrainingArguments

    def _effective_sigma(
        self, component: str, std_dev_t: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Per-step Gaussian std for the x-space KL, by component sampling dynamics.

        Args:
            component: Trajectory component name owning the scheduler.
            std_dev_t: Per-step diffusion std from the adapter forward.
            dt: Per-step time delta from the adapter forward (negative).

        Returns:
            Effective sigma tensor broadcastable to the component latent shape.
        """
        return self._effective_transition_std(
            component,
            std_dev_t,
            dt,
            context="DPPO x-based KL",
        )

    def _trust_region_kl(
        self,
        output: MultiModalStepOutput,
        replay: ReplayStep,
        old_state: LatentState,
    ) -> torch.Tensor:
        """Per-sample KL(current || rollout-old) driving the trust-region mask.

        The v-space mask uses unscaled squared error (GRPO convention); the x-space
        mask uses each component scheduler's variance-scaled Gaussian KL. Component
        results are combined by active stochastic degrees of freedom.
        """
        kl_mask_type = self.training_args.kl_mask_type
        output_field, _ = self._kl_space_fields(kl_mask_type, "kl_mask_type")
        new_state = self._require_output_state(output, output_field, "policy")
        component_kl: Dict[str, torch.Tensor]
        if kl_mask_type == "v-based":
            component_kl = self._component_squared_error_elements(
                new_state, old_state, "stored rollout"
            )
        else:
            expected_names = self.adapter.trajectory_component_order
            if old_state.component_names != expected_names:
                raise ValueError(
                    f"expected stored rollout next_latents_mean in component order "
                    f"{expected_names}, received {old_state.component_names}"
                )
            std_dev_t = self._require_component_mapping(
                output.std_dev_t, "std_dev_t", "policy output"
            )
            dt = self._require_component_mapping(output.dt, "dt", "policy output")
            component_kl = {}
            for name in expected_names:
                sigma_t = self._effective_sigma(name, std_dev_t[name], dt[name])
                component_kl[name] = gaussian_kl_div(
                    new_state.components[name], old_state.components[name], sigma_t
                )
        # Raw per-element values, so one global reduction weights every element once.
        return self.adapter.reduce_latent_values(component_kl, state=replay.state)

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts and store the rollout-old quantity the KL mask needs."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
        )
        # The trust-region mask compares the current policy against the rollout-old
        # policy in `kl_mask_type` space, so only that per-step quantity is stored
        # (the ref-KL penalty compares current vs reference, never the old policy).
        mask_field = (
            "velocity" if self.training_args.kl_mask_type == "v-based" else "next_latents_mean"
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
            extra_call_back_kwargs=[mask_field],
        )

    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): KL trust-region masked loss and optional KL-vs-ref."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        kl_type = self.training_args.kl_type
        kl_mask_type = self.training_args.kl_mask_type
        kl_guidance_scale = self.training_args.kl_guidance_scale
        kl_mask_threshold = self.training_args.kl_mask_threshold
        train_step_indices = self.adapter.get_train_step_indices()
        # Mask space picks the single rollout-old tensor stored by sample().
        _, mask_field = self._kl_space_fields(kl_mask_type, "kl_mask_type")
        # Forward fields: the mask uses kl_mask_type; the ref penalty uses kl_type.
        # std_dev_t/dt feed the x-based mask's variance scaling only.
        requested_fields = {"log_prob", mask_field}
        if kl_mask_type == "x-based":
            requested_fields.update(("std_dev_t", "dt"))
        if self.enable_kl_loss:
            _, ref_return_field = self._kl_space_fields(kl_type, "kl_type")
            requested_fields.add(ref_return_field)
        else:
            ref_return_field = None
        return_fields = self._canonical_return_fields(requested_fields)
        for inner_epoch in range(self.training_args.num_inner_epochs):
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
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
                        # Rollout-old policy in mask space (the only stored callback tensor).
                        old_mask_state = self.adapter.get_replay_callback(
                            batch, step_index, mask_field
                        )
                        # 2. Forward pass — request only what the mask (and optional ref KL) need.
                        with self.autocast():
                            output = self._replay_forward(batch, replay, return_fields)

                        # 3. Compute loss
                        adv = batch["advantage"]
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        new_log_prob = self._require_policy_log_prob(
                            output, step_index, self._replay_batch_size(replay)
                        )
                        ratio = torch.exp(new_log_prob - old_log_prob)

                        # Per-step KL(current || old) for the trust-region mask.
                        kl_new_old = self._trust_region_kl(output, replay, old_mask_state)

                        # DPPO mask: zero gradient for trust-region violators that
                        # push the wrong way (ratio>1 & adv>0, or ratio<1 & adv<0).
                        unclipped_loss = -adv * ratio
                        violate = kl_new_old >= kl_mask_threshold
                        pos_rm = violate & (ratio > 1.0) & (adv > 0)
                        neg_rm = violate & (ratio < 1.0) & (adv < 0)
                        keep_mask = (
                            torch.logical_not(pos_rm | neg_rm)
                            .to(dtype=unclipped_loss.dtype)
                            .detach()
                        )
                        policy_loss = torch.mean(unclipped_loss * keep_mask)

                        loss = policy_loss

                        # 4. Optional KL-vs-reference penalty (run at kl_guidance_scale CFG).
                        # negative_* embeds already rode `sample.to(device)` into `batch`, so the
                        # ref forward reuses them on-device without an extra move.
                        if self.enable_kl_loss:
                            ref_overrides = (
                                {}
                                if kl_guidance_scale is None
                                else {"guidance_scale": kl_guidance_scale}
                            )
                            with self.autocast():
                                ref_output = self._reference_forward(
                                    batch, replay, (ref_return_field,), **ref_overrides
                                )
                                # kl_div must be computed outside `torch.no_grad()` for correct
                                # gradients.
                                kl_div = self._reference_kl_divergence(output, ref_output, replay)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        keep_frac = keep_mask.mean().detach()
                        loss_info["ratio"].append(ratio.detach())
                        loss_info["kl_new_old"].append(kl_new_old.detach())
                        loss_info["unclipped_loss"].append(unclipped_loss.detach())
                        loss_info["policy_loss"].append(policy_loss.detach())
                        loss_info["loss"].append(loss.detach())
                        loss_info["keep_ratio"].append(keep_frac)
                        loss_info["masked_ratio"].append(1.0 - keep_frac)

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)
