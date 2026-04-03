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

# src/flow_factory/trainers/dpo.py
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
from typing import List, Dict, Any, Tuple
from dataclasses import fields as dc_fields
from functools import partial
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import DPOTrainingArguments
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt, to_broadcast_tensor
from ..utils.dist import gather_samples
from ..utils.noise_schedule import TimeSampler
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class DPOTrainer(BaseTrainer):
    """
    Diffusion-DPO Trainer for Flow Matching models.

    Implements online DPO: generates multiple samples per prompt via K-repeat
    sampling, scores them with reward models, forms chosen/rejected pairs from
    the best/worst within each group, then optimises a velocity MSE DPO loss against a frozen reference model.

    Loss:
        L = -log sigma(-beta/2 * ((theta_w_err - ref_w_err) - (theta_l_err - ref_l_err)))
    where err = MSE(noise_pred, noise - x_0) averaged over spatial dims (same as flow_grpo train_sd3_dpo).

    References:
    [1] Diffusion Model Alignment Using Direct Preference Optimization
        - https://arxiv.org/abs/2311.12908
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: DPOTrainingArguments
        self.num_train_timesteps = self.training_args.num_train_timesteps

    # ====================== Main Loop ======================
    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            # Save checkpoint
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0
                and self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            samples = self.sample()
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # ====================== Evaluation ======================
    def evaluate(self) -> None:
        """Evaluation loop — same pattern as GRPO."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples: List[BaseSample] = []

            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    'trajectory_indices': None,
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
                self.eval_reward_buffer.add_samples(samples)

            rewards = self.eval_reward_buffer.finalize(store_to_samples=False, split='pointwise')

            # Gather and log rewards
            rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
            gathered_rewards = {
                key: self.accelerator.gather(value).cpu().numpy()
                for key, value in rewards.items()
            }

            if self.accelerator.is_main_process:
                _log_data = {f'eval/reward_{key}_mean': np.mean(value) for key, value in gathered_rewards.items()}
                _log_data.update({f'eval/reward_{key}_std': np.std(value) for key, value in gathered_rewards.items()})
                _log_data['eval_samples'] = all_samples
                self.log_data(_log_data, step=self.step)
            self.accelerator.wait_for_everyone()

    # ====================== Sampling ======================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts — DPO does NOT need log-probs or full trajectories."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,  # DPO doesn't need log-probs
                    'trajectory_indices': [-1],  # Only keep final latents (clean image)
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    # ====================== Advantage Computation ======================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        The computed advantages respect the user's ``advantage_aggregation``
        setting (``'sum'`` or ``'gdpo'``).  Reward/advantage statistics are
        logged internally by the processor.
        """
        aggregation_func = self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
            step=self.step,
        )

    # ====================== Pair Formation ======================

    @staticmethod
    def _get_advantage(sample: BaseSample) -> float:
        """Extract scalar advantage from a sample."""
        adv = sample.extra_kwargs['advantage']
        return adv.item() if hasattr(adv, 'item') else float(adv)

    def _form_pairs(
        self,
        samples: List[BaseSample],
    ) -> Tuple[List[Tuple[BaseSample, BaseSample]], Dict[str, Any]]:
        """Form (chosen, rejected) pairs from pre-computed advantages.

        Advantages must already be stored in each sample's
        ``extra_kwargs['advantage']`` (via ``compute_advantages`` with
        ``store_to_samples=True``).

        When ``group_on_same_rank`` (group_contiguous), all K copies of a
        group reside on this rank — pairs are formed locally.
        When not ``group_on_same_rank`` (distributed_k_repeat), samples are
        gathered across all ranks via ``gather_samples()`` so that every
        group's K copies are available. Pairs are formed on the global data
        then distributed evenly across ranks.

        Returns:
            pairs: list of (chosen_sample, rejected_sample) tuples
            log_data: dict of DPO-specific statistics for logging
        """
        if self.advantage_processor.group_on_same_rank:
            # group_contiguous: all K copies on this rank — form pairs locally
            pairs = self._form_pairs_from_advantages(samples)
        else:
            # distributed_k_repeat: gather full samples across ranks so that
            # every group's K copies are available for pairing.
            gather_field_names = [
                f.name for f in dc_fields(samples[0])
                if f.name != '_unique_id'
                and getattr(samples[0], f.name) is not None
            ]
            global_samples = gather_samples(
                accelerator=self.accelerator,
                samples=samples,
                field_names=gather_field_names,
                device=self.accelerator.device,
            )

            # Form pairs on global data (every group has all K copies)
            all_pairs = self._form_pairs_from_advantages(global_samples)

            # Distribute pairs evenly across ranks
            world_size = self.accelerator.num_processes
            rank = self.accelerator.process_index
            pairs_per_rank = len(all_pairs) // max(1, world_size)
            start = rank * pairs_per_rank
            end = start + pairs_per_rank if rank < world_size - 1 else len(all_pairs)
            pairs = all_pairs[start:end]

        # DPO-specific log data (reward stats already logged by compute_advantages)
        _log_data: Dict[str, Any] = {}
        _log_data['train/dpo_num_pairs'] = len(pairs)
        if pairs:
            chosen_advs = np.array([self._get_advantage(p[0]) for p in pairs])
            rejected_advs = np.array([self._get_advantage(p[1]) for p in pairs])
            _log_data['train/dpo_chosen_adv_mean'] = float(np.mean(chosen_advs))
            _log_data['train/dpo_rejected_adv_mean'] = float(np.mean(rejected_advs))
            _log_data['train/dpo_adv_margin_mean'] = float(np.mean(chosen_advs - rejected_advs))

        return pairs, _log_data

    @staticmethod
    def _form_pairs_from_advantages(
        samples: List[BaseSample],
    ) -> List[Tuple[BaseSample, BaseSample]]:
        """Form (chosen, rejected) pairs based on per-sample advantages.

        Groups samples by ``unique_id``.  For each group with >= 2 samples,
        the highest-advantage sample is chosen and the lowest-advantage sample
        is rejected.

        Args:
            samples: sample list with ``extra_kwargs['advantage']`` populated.

        Returns:
            List of ``(chosen, rejected)`` sample pairs.
        """
        # Build group mapping from unique_id
        unique_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
        _, group_indices = np.unique(unique_ids, return_inverse=True)

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
    def _sample_timesteps(self, batch_size: int, num_timesteps: int, timestep_range: Tuple[float, float]) -> torch.Tensor:
        """Sample T×B timesteps for DPO training.

        Reuses ``TimeSampler`` from ``utils.noise_schedule``.
        Rescales output to ``timestep_range`` configured on the training args.

        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with values
            in [t_lo, t_hi].
        """
        device = self.accelerator.device
        if self.training_args.weighting_scheme == 'logit_normal':
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
    def _forward_noise_pred(self, latents: torch.Tensor, base_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Run a single forward pass and return the noise prediction."""
        fwd_kwargs = {**base_kwargs, 'latents': latents}
        fwd_kwargs = filter_kwargs(self.adapter.forward, **fwd_kwargs)
        return self.adapter.forward(**fwd_kwargs).noise_pred

    # ====================== Optimization ======================
    def optimize(self, samples: List[BaseSample]) -> None:
        """DPO optimisation loop.

        1. Finalise rewards, compute advantages, and form chosen/rejected pairs.
        2. For each pair:
           - sample a random timestep, add shared noise
           - compute policy & reference noise predictions
           - compute MSE DPO loss
        3. Backward + optimizer step.
        """
        # Finalize reward computation
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')

        # Compute advantages (handles communication, aggregation, logging)
        self.compute_advantages(samples, rewards, store_to_samples=True)

        # Form pairs from pre-computed advantages
        pairs, pair_log_data = self._form_pairs(samples)

        # Log pair statistics
        self.log_data(pair_log_data, step=self.step)

        if not pairs:
            logger.warning(f"Epoch {self.epoch}: no valid pairs formed, skipping optimization.")
            return

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle pairs
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(pairs), generator=perm_gen)
            shuffled_pairs = [pairs[i] for i in perm]

            # Batch pairs
            batch_size = self.training_args.per_device_batch_size
            pair_batches = [
                shuffled_pairs[i:i + batch_size]
                for i in range(0, len(shuffled_pairs), batch_size)
            ]

            self.adapter.train()
            loss_info = defaultdict(list)

            with self.autocast():
                for pair_batch in tqdm(
                    pair_batches,
                    total=len(pair_batches),
                    desc=f'Epoch {self.epoch} DPO Training',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    # Stack chosen and rejected latents (shared across timesteps)
                    chosen_samples = [p[0] for p in pair_batch]
                    rejected_samples = [p[1] for p in pair_batch]

                    chosen_batch = BaseSample.stack(chosen_samples)
                    rejected_batch = BaseSample.stack(rejected_samples)

                    # Get clean latents (final step from trajectory, index -1)
                    chosen_latents = chosen_batch['all_latents'][:, -1]
                    rejected_latents = rejected_batch['all_latents'][:, -1]

                    current_batch_size = chosen_latents.shape[0]

                    # Pre-sample T×B timesteps for this pair batch
                    all_timesteps = self._sample_timesteps(
                        batch_size=current_batch_size,
                        num_timesteps=self.num_train_timesteps,
                        timestep_range=self.training_args.timestep_range,
                    )  # (T, B)

                    # Build static forward kwargs (shared across timesteps)
                    _excluded_batch_keys = {'all_latents', 'timesteps', 'advantage'}
                    static_kwargs = {
                        **self.training_args,
                        'compute_log_prob': False,
                        'return_kwargs': ['noise_pred'],
                        'noise_level': 0.0,
                        **{k: v for k, v in chosen_batch.items()
                           if k not in _excluded_batch_keys},
                    }

                    for t_idx in range(self.num_train_timesteps):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            t = all_timesteps[t_idx]  # (B,), scale in [0, 1000]
                            noise = randn_tensor(
                                chosen_latents.shape,
                                device=chosen_latents.device,
                                dtype=chosen_latents.dtype,
                            )

                            t_broadcast_chosen = to_broadcast_tensor(t, chosen_latents)
                            t_broadcast_rejected = to_broadcast_tensor(t, rejected_latents)

                            # Noise both at same timestep: x_t = (1 - t) * x_0 + t * noise
                            noised_chosen = (1 - t_broadcast_chosen) * chosen_latents + t_broadcast_chosen * noise
                            noised_rejected = (1 - t_broadcast_rejected) * rejected_latents + t_broadcast_rejected * noise

                            # Per-timestep forward kwargs
                            base_kwargs = {
                                **static_kwargs,
                                't': t,
                                't_next': torch.zeros_like(t),
                            }

                            # Policy forward
                            theta_w_pred = self._forward_noise_pred(noised_chosen, base_kwargs)
                            theta_l_pred = self._forward_noise_pred(noised_rejected, base_kwargs)

                            # Reference forward (frozen)
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                ref_w_pred = self._forward_noise_pred(noised_chosen, base_kwargs)
                                ref_l_pred = self._forward_noise_pred(noised_rejected, base_kwargs)

                            # MSE errors per sample — target is flow-matching velocity (noise - x_0), same as
                            # flow_grpo train_sd3_dpo.py: target = noise - model_input
                            target_w = noise - chosen_latents
                            target_l = noise - rejected_latents
                            spatial_dims = tuple(range(1, theta_w_pred.ndim))
                            theta_w_err = ((theta_w_pred - target_w) ** 2).mean(dim=spatial_dims)
                            theta_l_err = ((theta_l_pred - target_l) ** 2).mean(dim=spatial_dims)
                            ref_w_err = ((ref_w_pred - target_w) ** 2).mean(dim=spatial_dims)
                            ref_l_err = ((ref_l_pred - target_l) ** 2).mean(dim=spatial_dims)

                            # DPO loss
                            beta = self.training_args.beta
                            w_diff = theta_w_err - ref_w_err
                            l_diff = theta_l_err - ref_l_err
                            w_l_diff = w_diff - l_diff
                            inside_term = -0.5 * beta * w_l_diff
                            loss = -F.logsigmoid(inside_term).mean()

                            # Logging metrics
                            with torch.no_grad():
                                implicit_reward_chosen = -0.5 * beta * w_diff
                                implicit_reward_rejected = -0.5 * beta * l_diff
                                implicit_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()

                            loss_info['loss'].append(loss.detach())
                            loss_info['theta_w_err'].append(theta_w_err.mean().detach())
                            loss_info['theta_l_err'].append(theta_l_err.mean().detach())
                            loss_info['ref_w_err'].append(ref_w_err.mean().detach())
                            loss_info['ref_l_err'].append(ref_l_err.mean().detach())
                            loss_info['implicit_accuracy'].append(implicit_accuracy.detach())
                            loss_info['implicit_reward_chosen'].append(implicit_reward_chosen.mean().detach())
                            loss_info['implicit_reward_rejected'].append(implicit_reward_rejected.mean().detach())

                            # Backward + optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                loss_info = {
                                    k: torch.stack(v).mean()
                                    for k, v in loss_info.items()
                                }
                                loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
