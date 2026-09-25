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

# src/flow_factory/trainers/rl/dpo.py
"""
Diffusion-DPO (Direct Preference Optimization) Trainer.
Implements online DPO for flow matching models using velocity MSE (target = noise - x_0).

References:
[1] Diffusion Model Alignment Using Direct Preference Optimization
    - https://arxiv.org/abs/2311.12908
[2] flow_grpo reference implementation
    - https://github.com/yifan123/flow_grpo
"""

import os
from collections import defaultdict
from dataclasses import fields as dc_fields
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...contracts.reward_overlap import RANK_LOCAL_REWARD_OPTIMIZATION_OVERLAP
from ...hparams import DPOTrainingArguments
from ...rewards import RewardTile, RewardTileGeometry
from ...samples import BaseSample, LatentState, NoisedState, group_identity_rows
from ...utils.base import create_generator, create_generator_by_prompt
from ...utils.dist import gather_samples
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import TimeSampler
from ..abc import BaseTrainer
from ..common import dpo_objective, pairwise_policy_activation_context
from ..common.state_validation import require_latent_state, state_batch_size
from ..forward_process import forward_velocity_state

logger = setup_logger(__name__)


class DPOTrainer(BaseTrainer):
    """
    Diffusion-DPO Trainer for Flow Matching models.

    Implements online DPO: generates multiple samples per prompt via K-repeat
    sampling, scores them with reward models, forms chosen/rejected pairs from
    the best/worst within each group, then optimises a velocity MSE DPO loss against a frozen reference model.

    Loss:
        L = -log sigma(-beta/2 * ((theta_w_err - ref_w_err) - (theta_l_err - ref_l_err)))
    where err = MSE(velocity, noise - x_0) averaged over spatial dims (same as flow_grpo train_sd3_dpo).

    References:
    [1] Diffusion Model Alignment Using Direct Preference Optimization
        - https://arxiv.org/abs/2311.12908
    """

    # Decoupled paradigm: lossy rollout acceleration is permitted (constraints.md #7).
    paradigm = "decoupled"
    reward_optimization_overlap_contract = RANK_LOCAL_REWARD_OPTIMIZATION_OVERLAP

    @classmethod
    def reward_optimization_overlap_geometry(cls, config):
        """Map each complete reward group to one chosen/rejected pair."""
        return RewardTileGeometry(
            optimizer_examples_per_group=1,
            optimizer_terms_per_batch=config.training_args.get_num_train_timesteps(config),
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: DPOTrainingArguments
        self.num_train_timesteps = self.training_args.num_train_timesteps

    # ====================== Main Loop ======================
    # ====================== Sampling ======================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DPO (final latents only, no log-probs)."""
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=[-1],
        )

    # ====================== Advantage Computation ======================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        The computed advantages respect the user's ``advantage_aggregation``
        setting (``'sum'`` or ``'gdpo'``).  Call ``self.advantage_processor.pop_advantage_metrics``
        after this when logging training statistics.
        """
        aggregation_func = self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    # ====================== Pair Formation ======================

    @staticmethod
    def _get_advantage(sample: BaseSample) -> float:
        """Extract scalar advantage from a sample."""
        adv = sample.extra_kwargs["advantage"]
        return adv.item() if hasattr(adv, "item") else float(adv)

    def _form_pairs(
        self,
        samples: List[BaseSample],
    ) -> Tuple[List[Tuple[BaseSample, BaseSample]], Dict[str, Any]]:
        """Form (chosen, rejected) pairs from pre-computed advantages.

        Called from :meth:`optimize` after :meth:`prepare_feedback` has run in the same epoch.
        Advantages must already be stored in each sample's
        ``extra_kwargs['advantage']`` (via ``compute_advantages`` with
        ``store_to_samples=True``).

        With a rank-local layout, all K copies of a group reside on this rank
        and pairs are formed locally. With a cross-rank layout, samples are
        gathered via ``gather_samples()`` so every group's K copies are
        available. Pairs are formed on the global data, assigned round-robin,
        and padded to the same rank-local length so optimization stays in
        lockstep.

        Returns:
            pairs: list of (chosen_sample, rejected_sample) tuples
            log_data: dict of DPO-specific statistics (logged from :meth:`optimize`)
        """
        if self.advantage_processor.group_on_same_rank:
            # Rank-local layout: all K copies are present here.
            pairs = self._form_pairs_from_advantages(samples)
            stat_pairs = pairs
        else:
            # Cross-rank layout: gather full samples for pair formation.
            gather_field_names = [
                f.name
                for f in dc_fields(samples[0])
                if f.name not in {"_unique_id", "extra_kwargs", "applicable_rewards"}
            ]
            global_samples = gather_samples(
                accelerator=self.accelerator,
                samples=samples,
                field_names=gather_field_names,
                device=self.accelerator.device,
                group_coordinator=self.group_coordinator,
                extra_field_names=["advantage"],
            )

            # Form pairs over the sampler's smallest complete-group scope.
            all_pairs = self._form_pairs_from_advantages(global_samples)

            # Distribute pairs evenly inside that scope. Global layouts use the
            # full world; subgroup_tile stops here instead of replicating every
            # trajectory across unrelated subgroups.
            n_pairs = len(all_pairs)
            world_size = self.group_coordinator.group_world_size
            rank = self.group_coordinator.group_rank
            if world_size > 1 and n_pairs < world_size:
                raise RuntimeError(
                    "DPOTrainer (cross-rank sampler): need at least "
                    f"{world_size} chosen/rejected pairs for one pair per rank in the sampler "
                    f"communication scope; got {n_pairs}. "
                    "Increase unique prompts/groups or use sampler_type group_contiguous."
                )

            pairs_sharded = all_pairs[rank::world_size]
            stat_pairs = pairs_sharded
            target = (n_pairs + world_size - 1) // world_size if n_pairs else 0
            if pairs_sharded:
                m = len(pairs_sharded)
                pairs = (pairs_sharded * ((target + m - 1) // m))[:target]
                if m < target:
                    logger.warning(
                        "DPOTrainer: cycled local DPO pair shard to equalize per-rank optimize steps "
                        "(cross-rank sampler; local_pairs(%d), padded_to(%d), "
                        "num_processes(%d), process_index(%d), epoch(%d)). "
                        "Some preference pairs are trained more than once on this rank.",
                        m,
                        target,
                        world_size,
                        rank,
                        self.epoch,
                    )
            else:
                pairs = []

        if getattr(self, "_dpo_reward_overlap_tile_active", False):
            # Overlap validation has already proved that every rank owns the same
            # number of complete rank-local groups in this tile. Partial pair
            # metrics are intentionally discarded by the tile hook, so neither a
            # metric reduction nor a pair-count collective carries useful data.
            return pairs, {"train/dpo_num_pairs": len(stat_pairs) * self.accelerator.num_processes}

        # DPO-specific keys — globally reduced across all ranks (unpadded pairs only)
        _log_data: Dict[str, Any] = {}
        n = len(stat_pairs)
        if n > 0:
            chosen_advs = np.array([self._get_advantage(p[0]) for p in stat_pairs])
            rejected_advs = np.array([self._get_advantage(p[1]) for p in stat_pairs])
            margins = chosen_advs - rejected_advs
            local_stats = torch.tensor(
                [
                    float(n),
                    float(chosen_advs.sum()),
                    float(rejected_advs.sum()),
                    float(margins.sum()),
                ],
                device=self.accelerator.device,
                dtype=torch.float64,
            )
        else:
            local_stats = torch.zeros(4, device=self.accelerator.device, dtype=torch.float64)

        global_stats = self.accelerator.reduce(local_stats, reduction="sum")
        total_n = global_stats[0].item()
        _log_data["train/dpo_num_pairs"] = int(total_n)
        if total_n > 0:
            _log_data["train/dpo_chosen_adv_mean"] = global_stats[1].item() / total_n
            _log_data["train/dpo_rejected_adv_mean"] = global_stats[2].item() / total_n
            _log_data["train/dpo_adv_margin_mean"] = global_stats[3].item() / total_n

        return pairs, _log_data

    @staticmethod
    def _form_pairs_from_advantages(
        samples: List[BaseSample],
    ) -> List[Tuple[BaseSample, BaseSample]]:
        """Form (chosen, rejected) pairs based on per-sample advantages.

        Groups samples by ``(source_id, unique_id)``. For each group with >= 2 samples,
        the highest-advantage sample is chosen and the lowest-advantage sample
        is rejected.

        Args:
            samples: sample list with ``extra_kwargs['advantage']`` populated.

        Returns:
            List of ``(chosen, rejected)`` sample pairs.
        """
        group_identities = np.asarray(group_identity_rows(samples), dtype=np.int64)
        _, group_indices = np.unique(group_identities, axis=0, return_inverse=True)

        # Extract advantage values
        advantages = np.array(
            [DPOTrainer._get_advantage(s) for s in samples],
            dtype=np.float64,
        )

        pairs: List[Tuple[BaseSample, BaseSample]] = []
        for gid in np.unique(group_indices):
            mask = np.where(group_indices == gid)[0]
            if len(mask) < 2:
                logger.warning(f"Group {gid} has less than 2 samples, skipping pair formation.")
                continue
            group_adv = advantages[mask]
            best = mask[np.argmax(group_adv)]
            worst = mask[np.argmin(group_adv)]
            pairs.append((samples[best], samples[worst]))
        return pairs

    # ====================== Timestep Sampling ======================
    def _sample_timesteps(
        self, batch_size: int, num_timesteps: int, timestep_range: Tuple[float, float]
    ) -> torch.Tensor:
        """Sample T×B timesteps for DPO training.

        Reuses ``TimeSampler`` from ``utils.noise_schedule``.
        Rescales output to ``timestep_range`` configured on the training args.

        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with values
            in [t_lo, t_hi].
        """
        device = self.accelerator.device
        if self.training_args.weighting_scheme == "logit_normal":
            t = TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=num_timesteps,
                timestep_range=timestep_range,
                logit_mean=self.training_args.logit_mean,
                logit_std=self.training_args.logit_std,
                time_shift=self.training_args.time_shift,
                device=device,
                stratified=False,
            )  # (T, B)
        else:  # uniform
            t = TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=num_timesteps,
                timestep_range=timestep_range,
                time_shift=self.training_args.time_shift,
                device=device,
            )
        return t

    # ====================== Forward Helpers ======================
    def _require_paired_terminal_states(
        self,
        chosen_state: LatentState,
        rejected_state: LatentState,
    ) -> int:
        """Validate that both preference arms describe the same latent geometry.

        The arms are noised with one shared noise tensor per component, so any
        divergence in component order, shape, dtype or device silently changes the
        comparison the preference loss makes.

        Args:
            chosen_state: Terminal clean state of the winner arm.
            rejected_state: Terminal clean state of the loser arm.

        Returns:
            Batch size shared by both arms.
        """
        expected_names = self.adapter.trajectory_component_order
        batch_size = state_batch_size(self, chosen_state, "chosen terminal state")
        require_latent_state(self, rejected_state, "rejected terminal state")
        for name in expected_names:
            chosen = chosen_state.components[name]
            rejected = rejected_state.components[name]
            if chosen.ndim < 2 or chosen.shape[0] != batch_size:
                raise ValueError(
                    f"expected chosen component {name!r} for {type(self).__name__} with shape "
                    f"(B, ...) and batch size {batch_size}, received {tuple(chosen.shape)}"
                )
            if rejected.shape != chosen.shape:
                raise ValueError(
                    f"expected paired component {name!r} for {type(self).__name__} to share the "
                    f"chosen shape {tuple(chosen.shape)}, received {tuple(rejected.shape)}"
                )
            if rejected.dtype != chosen.dtype or rejected.device != chosen.device:
                raise ValueError(
                    f"expected paired component {name!r} for {type(self).__name__} to share the "
                    f"chosen dtype/device ({chosen.dtype}, {chosen.device}), received "
                    f"({rejected.dtype}, {rejected.device})"
                )
        return batch_size

    def _arm_velocity_error(
        self,
        velocity: LatentState,
        noised: NoisedState,
    ) -> torch.Tensor:
        """Compute one arm's per-sample squared error against its target velocity.

        Args:
            velocity: Predicted velocity per component for this arm.
            noised: This arm's noised state carrying its target velocity.

        Returns:
            Per-sample error of shape ``(B,)``.
        """
        errors = {
            name: (
                velocity.components[name].float() - noised.target_velocity.components[name].float()
            )
            ** 2
            for name in self.adapter.trajectory_component_order
        }
        return self.adapter.reduce_latent_values(errors, state=noised.state)

    def _preference_loss(
        self,
        theta_w_err: torch.Tensor,
        theta_l_err: torch.Tensor,
        ref_w_err: torch.Tensor,
        ref_l_err: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the DPO preference loss and its implicit-reward metrics.

        Args:
            theta_w_err: Policy error on the winner arm, shape ``(B,)``.
            theta_l_err: Policy error on the loser arm, shape ``(B,)``.
            ref_w_err: Reference error on the winner arm, shape ``(B,)``.
            ref_l_err: Reference error on the loser arm, shape ``(B,)``.

        Returns:
            Scalar loss and the per-sample implicit reward / accuracy metrics.
        """
        return dpo_objective(
            policy_chosen_loss=theta_w_err,
            policy_rejected_loss=theta_l_err,
            reference_chosen_loss=ref_w_err,
            reference_rejected_loss=ref_l_err,
            beta=self.training_args.beta,
        )

    # ====================== Reward / advantage (Stages 4--5) ======================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards, compute advantages, and log advantage-processor metrics.

        Does not form chosen/rejected pairs; :meth:`optimize` calls :meth:`_form_pairs` after
        advantages are stored on each sample.
        """
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    # ====================== Optimization ======================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Run online DPO and publish pair statistics for a full acquisition."""
        self._optimize_dpo_samples(samples, log_pair_metrics=True)

    def _optimize_dpo_samples(
        self,
        samples: List[BaseSample],
        *,
        log_pair_metrics: bool,
    ) -> None:
        """Policy optimization (Stage 6): build chosen/rejected pairs, then DPO preference loss.

        Requires :meth:`prepare_feedback` in the same epoch so ``extra_kwargs['advantage']`` is set.
        """
        pairs, pair_log_data = self._form_pairs(samples)
        if log_pair_metrics:
            self.log_data(pair_log_data, step=self.step)

        global_pair_count = int(pair_log_data.get("train/dpo_num_pairs", 0))
        if global_pair_count == 0:
            raise RuntimeError(
                f"DPOTrainer: no valid chosen/rejected pairs at epoch {self.epoch}. "
                "Each prompt group needs at least two samples with comparable advantages to form "
                "a winner and a loser. Check group_size, reward models, and advantage_aggregation."
            )

        # Optimize
        for inner_epoch in range(self.training_args.num_inner_epochs):
            if self.training_args.shuffle_samples:
                perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
                perm = torch.randperm(len(pairs), generator=perm_gen)
                shuffled_pairs = [pairs[i] for i in perm]
            else:
                shuffled_pairs = pairs

            # Batch pairs. Prefetch chosen and rejected micro-batches in lockstep
            # via two copy-stream iterators so their H2D overlaps compute under
            # offload (a plain blocking stack when offload is off).
            batch_size = self.training_args.per_device_batch_size
            chosen_list = [p[0] for p in shuffled_pairs]
            rejected_list = [p[1] for p in shuffled_pairs]
            num_pair_batches = (len(shuffled_pairs) + batch_size - 1) // batch_size

            self.adapter.train()
            loss_info = defaultdict(list)

            for chosen_batch, rejected_batch in tqdm(
                zip(
                    self._iter_prefetched_batches(chosen_list, batch_size),
                    self._iter_prefetched_batches(rejected_list, batch_size),
                ),
                total=num_pair_batches,
                desc=f"Epoch {self.epoch} DPO Training",
                position=0,
                disable=not self.show_progress_bar,
            ):

                # Get clean terminal states of both arms
                chosen_state = self.adapter.get_terminal_state(chosen_batch)
                rejected_state = self.adapter.get_terminal_state(rejected_batch)

                current_batch_size = self._require_paired_terminal_states(
                    chosen_state, rejected_state
                )

                # Pre-sample T×B timesteps for this pair batch
                all_timesteps = self._sample_timesteps(
                    batch_size=current_batch_size,
                    num_timesteps=self.num_train_timesteps,
                    timestep_range=self.training_args.timestep_range,
                )  # (T, B)

                for t_idx in range(self.num_train_timesteps):
                    with self.accumulate_gradients():
                        times = self.adapter.build_training_component_times(
                            all_timesteps[t_idx], batch=chosen_batch
                        )

                        # Noise both arms at the same σ with the same noise tensor
                        chosen_noised = self.adapter.add_forward_process_noise(chosen_state, times)
                        rejected_noised = self.adapter.apply_forward_process_noise(
                            rejected_state, times, chosen_noised.noise
                        )

                        # Policy forward
                        with self.autocast(), pairwise_policy_activation_context(self):
                            theta_w_pred = forward_velocity_state(
                                self,
                                chosen_batch,
                                chosen_noised.state,
                                times,
                                source="policy chosen",
                            )
                        with self.autocast(), pairwise_policy_activation_context(self):
                            theta_l_pred = forward_velocity_state(
                                self,
                                rejected_batch,
                                rejected_noised.state,
                                times,
                                source="policy rejected",
                            )

                        # Reference forward (frozen)
                        with torch.no_grad(), self.adapter.use_ref_parameters(), self.autocast():
                            ref_w_pred = forward_velocity_state(
                                self,
                                chosen_batch,
                                chosen_noised.state,
                                times,
                                source="reference chosen",
                            )
                            ref_l_pred = forward_velocity_state(
                                self,
                                rejected_batch,
                                rejected_noised.state,
                                times,
                                source="reference rejected",
                            )

                        # MSE errors per sample — target is flow-matching velocity (noise - x_0), same as
                        # flow_grpo train_sd3_dpo.py: target = noise - model_input
                        theta_w_err = self._arm_velocity_error(theta_w_pred, chosen_noised)
                        theta_l_err = self._arm_velocity_error(theta_l_pred, rejected_noised)
                        ref_w_err = self._arm_velocity_error(ref_w_pred, chosen_noised)
                        ref_l_err = self._arm_velocity_error(ref_l_pred, rejected_noised)

                        # DPO loss and logging metrics
                        loss, metrics = self._preference_loss(
                            theta_w_err, theta_l_err, ref_w_err, ref_l_err
                        )

                        loss_info["loss"].append(loss.detach())
                        loss_info["theta_w_err"].append(theta_w_err.mean().detach())
                        loss_info["theta_l_err"].append(theta_l_err.mean().detach())
                        loss_info["ref_w_err"].append(ref_w_err.mean().detach())
                        loss_info["ref_l_err"].append(ref_l_err.mean().detach())
                        loss_info["implicit_accuracy"].append(metrics["implicit_accuracy"].detach())
                        loss_info["implicit_reward_chosen"].append(
                            metrics["implicit_reward_chosen"].mean().detach()
                        )
                        loss_info["implicit_reward_rejected"].append(
                            metrics["implicit_reward_rejected"].mean().detach()
                        )

                        # Backward + optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)

    def _optimize_reward_overlap_tile(
        self,
        tile: RewardTile,
        samples: List[BaseSample],
        context: Any,
    ) -> None:
        """Optimize one pair-complete tile without emitting partial pair metrics."""
        del tile, context
        self._dpo_reward_overlap_tile_active = True
        try:
            self._optimize_dpo_samples(samples, log_pair_metrics=False)
        finally:
            del self._dpo_reward_overlap_tile_active

    def _finalize_reward_optimization_overlap(
        self,
        context: Any,
        samples: List[BaseSample],
    ) -> Dict[str, Any]:
        """Build acquisition-wide pair metrics after full advantages are restored."""
        del context
        _pairs, pair_metrics = self._form_pairs(samples)
        return pair_metrics
