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
Implements online DPO for flow matching models using noise-prediction MSE.

References:
[1] Diffusion Model Alignment Using Direct Preference Optimization
    - https://arxiv.org/abs/2311.12908
[2] flow_grpo reference implementation
    - https://github.com/yifan123/flow_grpo
"""
import os
from typing import List, Dict, Any, Union, Optional, Tuple
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
from ..rewards import RewardProcessor, RewardBuffer
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt, to_broadcast_tensor
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def compute_density_for_timestep_sampling(
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
) -> torch.Tensor:
    """Sample timesteps from a logit-normal distribution.

    From the Diffusion-DPO / flow_grpo reference implementation.
    Returns values in (0, 1).
    """
    u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,))
    u = torch.nn.functional.sigmoid(u)
    return u


class DPOTrainer(BaseTrainer):
    """
    Diffusion-DPO Trainer for Flow Matching models.

    Implements online DPO: generates multiple samples per prompt via K-repeat
    sampling, scores them with reward models, forms chosen/rejected pairs from
    the best/worst within each group, then optimises a noise-prediction MSE
    DPO loss against a frozen reference model.

    Loss:
        L = -log sigma(-beta/2 * ((theta_w_err - ref_w_err) - (theta_l_err - ref_l_err)))
    where err = MSE(noise_pred, target) averaged over spatial dims.

    References:
    [1] Diffusion Model Alignment Using Direct Preference Optimization
        - https://arxiv.org/abs/2311.12908
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: DPOTrainingArguments

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

    # ====================== Pair Formation ======================
    def _form_pairs(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> Tuple[List[Tuple[BaseSample, BaseSample]], Dict[str, Any]]:
        """Form (chosen, rejected) pairs from grouped samples.

        Strategy:
        1. **Global reward gathering** — gather rewards and unique_ids across all
           ranks so every rank sees the full picture for logging and global
           chosen/rejected reward statistics.
        2. **Local pair formation** — each rank forms pairs only from its own
           local samples.  This is the correct default because the standard
           ``DistributedKRepeatSampler`` shuffles and spreads a group's K
           samples across different ranks.  When ``GroupContiguousSampler``
           is used (async rewards), all K samples of a group are on one rank,
           which is a strict subset of this logic.

        For each group of samples sharing a ``unique_id`` on this rank, the
        sample with the **globally highest** aggregated reward is chosen and
        the one with the **globally lowest** is rejected.  If fewer than 2
        samples from a group land on this rank, that group is skipped on this
        rank (another rank will form the pair for it).

        Returns:
            pairs: list of (chosen_sample, rejected_sample) tuples
            log_data: dict of statistics for logging
        """
        # 1. Gather rewards across all processes (for global statistics)
        rewards_tensors = {
            key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()
        }
        gathered_rewards = {
            key: self.accelerator.gather(value).cpu().numpy()
            for key, value in rewards_tensors.items()
        }

        # 2. Aggregate rewards (weighted sum) — global view for logging
        aggregated_rewards_global = np.zeros_like(next(iter(gathered_rewards.values())), dtype=np.float64)
        for key, reward_array in gathered_rewards.items():
            aggregated_rewards_global += reward_array * self.reward_models[key].config.weight

        # Global group indices for logging
        unique_ids_global = torch.tensor(
            [s.unique_id for s in samples], dtype=torch.int64, device=self.accelerator.device
        )
        gathered_ids = self.accelerator.gather(unique_ids_global).cpu().numpy()
        _unique_ids_global, group_indices_global = np.unique(gathered_ids, return_inverse=True)

        # Global chosen/rejected stats (for logging only)
        chosen_rewards_list = []
        rejected_rewards_list = []
        for group_id in np.unique(group_indices_global):
            mask = np.where(group_indices_global == group_id)[0]
            group_rewards = aggregated_rewards_global[mask]
            chosen_rewards_list.append(group_rewards.max())
            rejected_rewards_list.append(group_rewards.min())

        # 3. Build local aggregated rewards for this rank's samples
        local_rewards = np.zeros(len(samples), dtype=np.float64)
        for key, value in rewards.items():
            local_rewards += np.asarray(value, dtype=np.float64) * self.reward_models[key].config.weight

        # 4. Form pairs locally: group by unique_id among this rank's samples
        local_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
        local_unique_ids, local_inv = np.unique(local_ids, return_inverse=True)

        pairs: List[Tuple[BaseSample, BaseSample]] = []
        for gid in range(len(local_unique_ids)):
            mask = np.where(local_inv == gid)[0]
            if len(mask) < 2:
                # Only one sample from this group on this rank — skip.
                # Another rank has enough samples to form a pair.
                continue
            group_r = local_rewards[mask]
            best_local = mask[np.argmax(group_r)]
            worst_local = mask[np.argmin(group_r)]
            pairs.append((samples[best_local], samples[worst_local]))

        # 5. Prepare log data
        _log_data: Dict[str, Any] = {}
        for key, value in gathered_rewards.items():
            _log_data[f'train/reward_{key}_mean'] = np.mean(value)
            _log_data[f'train/reward_{key}_std'] = np.std(value)
        _log_data['train/reward_mean'] = np.mean(aggregated_rewards_global)
        _log_data['train/reward_std'] = np.std(aggregated_rewards_global)
        _log_data['train/dpo_num_pairs'] = len(pairs)
        if chosen_rewards_list:
            _log_data['train/dpo_chosen_reward_mean'] = np.mean(chosen_rewards_list)
            _log_data['train/dpo_rejected_reward_mean'] = np.mean(rejected_rewards_list)
            _log_data['train/dpo_reward_margin_mean'] = np.mean(
                np.array(chosen_rewards_list) - np.array(rejected_rewards_list)
            )
        # Log per-reward group stats
        for key, reward_array in gathered_rewards.items():
            g_means, g_stds = RewardProcessor.compute_group_reward_stats(reward_array, group_indices_global)
            _log_data.update({
                f'train/reward_{key}_group_std_mean': float(np.mean(g_stds)),
                f'train/reward_{key}_group_std_max': float(np.max(g_stds)),
                f'train/reward_{key}_group_mean_std': float(np.std(g_means)),
            })
        _log_data['train_samples'] = samples[:30]

        return pairs, _log_data

    # ====================== Timestep Sampling ======================
    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps for DPO training.

        Returns:
            Tensor of shape (batch_size,) with values in (0, 1).
        """
        if self.training_args.weighting_scheme == 'logit_normal':
            u = compute_density_for_timestep_sampling(
                batch_size,
                logit_mean=self.training_args.logit_mean,
                logit_std=self.training_args.logit_std,
                mode_scale=self.training_args.mode_scale,
            )
        else:  # uniform
            u = torch.rand(batch_size)
        return u.to(self.accelerator.device)

    # ====================== Optimization ======================
    def optimize(self, samples: List[BaseSample]) -> None:
        """DPO optimisation loop.

        1. Finalise rewards and form chosen/rejected pairs.
        2. For each pair:
           - sample a random timestep, add shared noise
           - compute policy & reference noise predictions
           - compute MSE DPO loss
        3. Backward + optimizer step.
        """
        # Finalize reward computation
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        pairs, reward_log_data = self._form_pairs(samples, rewards)

        # Log reward statistics
        self.log_data(reward_log_data, step=self.step)

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
                for batch_idx, pair_batch in enumerate(tqdm(
                    pair_batches,
                    total=len(pair_batches),
                    desc=f'Epoch {self.epoch} DPO Training',
                    position=0,
                    disable=not self.show_progress_bar,
                )):
                    with self.accelerator.accumulate(*self.adapter.trainable_components):
                        # Stack chosen and rejected latents
                        chosen_samples = [p[0] for p in pair_batch]
                        rejected_samples = [p[1] for p in pair_batch]

                        chosen_batch = BaseSample.stack(chosen_samples)
                        rejected_batch = BaseSample.stack(rejected_samples)

                        # Get clean latents (final step from trajectory, index -1)
                        chosen_latents = chosen_batch['all_latents'][:, -1]
                        rejected_latents = rejected_batch['all_latents'][:, -1]

                        current_batch_size = chosen_latents.shape[0]

                        # Sample shared timestep
                        t = self._sample_timesteps(current_batch_size)  # (B,) in (0, 1)
                        t_broadcast_chosen = to_broadcast_tensor(t, chosen_latents)
                        t_broadcast_rejected = to_broadcast_tensor(t, rejected_latents)

                        # Sample shared noise
                        noise = randn_tensor(
                            chosen_latents.shape,
                            device=chosen_latents.device,
                            dtype=chosen_latents.dtype,
                        )

                        # Noise both at same timestep: x_t = (1 - t) * x_0 + t * noise
                        noised_chosen = (1 - t_broadcast_chosen) * chosen_latents + t_broadcast_chosen * noise
                        noised_rejected = (1 - t_broadcast_rejected) * rejected_latents + t_broadcast_rejected * noise

                        # Scale timestep for model input
                        t_scaled = (t * 1000).view(-1)
                        t_next = torch.zeros_like(t_scaled)

                        # Prepare forward kwargs (use chosen_batch for prompt embeddings etc.)
                        base_forward_kwargs = {
                            **self.training_args,
                            't': t_scaled,
                            't_next': t_next,
                            'compute_log_prob': False,
                            'return_kwargs': ['noise_pred'],
                            'noise_level': 0.0,
                            **{k: v for k, v in chosen_batch.items()
                               if k not in ['all_latents', 'timesteps', 'advantage']},
                        }

                        # ---- Policy forward ----
                        chosen_fwd = {**base_forward_kwargs, 'latents': noised_chosen}
                        chosen_fwd = filter_kwargs(self.adapter.forward, **chosen_fwd)
                        chosen_output = self.adapter.forward(**chosen_fwd)

                        rejected_fwd = {**base_forward_kwargs, 'latents': noised_rejected}
                        rejected_fwd = filter_kwargs(self.adapter.forward, **rejected_fwd)
                        rejected_output = self.adapter.forward(**rejected_fwd)

                        # ---- Reference forward (no grad) ----
                        with torch.no_grad(), self.adapter.use_ref_parameters():
                            ref_chosen_fwd = {**base_forward_kwargs, 'latents': noised_chosen}
                            ref_chosen_fwd = filter_kwargs(self.adapter.forward, **ref_chosen_fwd)
                            ref_chosen_output = self.adapter.forward(**ref_chosen_fwd)

                            ref_rejected_fwd = {**base_forward_kwargs, 'latents': noised_rejected}
                            ref_rejected_fwd = filter_kwargs(self.adapter.forward, **ref_rejected_fwd)
                            ref_rejected_output = self.adapter.forward(**ref_rejected_fwd)

                        # ---- Compute MSE errors ----
                        # Target is the noise (flow matching: v = noise - x_0, pred should match noise direction)
                        target = noise
                        spatial_dims = tuple(range(1, chosen_output.noise_pred.ndim))

                        theta_w_err = ((chosen_output.noise_pred - target) ** 2).mean(dim=spatial_dims)
                        theta_l_err = ((rejected_output.noise_pred - target) ** 2).mean(dim=spatial_dims)
                        ref_w_err = ((ref_chosen_output.noise_pred - target) ** 2).mean(dim=spatial_dims)
                        ref_l_err = ((ref_rejected_output.noise_pred - target) ** 2).mean(dim=spatial_dims)

                        # ---- DPO loss ----
                        beta = self.training_args.beta
                        inside_term = -0.5 * beta * (
                            (theta_w_err - ref_w_err) - (theta_l_err - ref_l_err)
                        )
                        loss = -F.logsigmoid(inside_term).mean()

                        # ---- Logging ----
                        with torch.no_grad():
                            implicit_reward_chosen = -0.5 * beta * (theta_w_err - ref_w_err)
                            implicit_reward_rejected = -0.5 * beta * (theta_l_err - ref_l_err)
                            implicit_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()

                        loss_info['loss'].append(loss.detach())
                        loss_info['theta_w_err'].append(theta_w_err.mean().detach())
                        loss_info['theta_l_err'].append(theta_l_err.mean().detach())
                        loss_info['ref_w_err'].append(ref_w_err.mean().detach())
                        loss_info['ref_l_err'].append(ref_l_err.mean().detach())
                        loss_info['implicit_accuracy'].append(implicit_accuracy.detach())
                        loss_info['implicit_reward_chosen'].append(implicit_reward_chosen.mean().detach())
                        loss_info['implicit_reward_rejected'].append(implicit_reward_rejected.mean().detach())

                        # ---- Backward + optimizer step ----
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # Log
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
