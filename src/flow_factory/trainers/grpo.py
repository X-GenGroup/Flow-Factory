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

# src/flow_factory/trainers/grpo.py
"""
Group Relative Policy Optimization (GRPO) Trainer.
Implements GRPO algorithm for flow matching models.
"""
import os
from typing import List, Dict, Optional, Any, Union, Literal, Callable
from functools import partial
from collections import defaultdict
import inspect
import logging
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..rewards import BaseRewardModel
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator
from ..utils.logger_utils import setup_logger
from ..rewards import (
    BaseRewardModel,
    RewardProcessor,
)
from ..utils.trajectory_collector import TrajectoryCollector, compute_trajectory_indices

logger = setup_logger(__name__)


# ============================ GRPO Trainer ============================
class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    References:
    [1] Flow-GRPO: Training Flow Matching Models via Online RL
        - https://arxiv.org/abs/2505.05470
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    def start(self):
        """Main training loop."""
        while True:
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.log_args.save_freq > 0 and 
                self.epoch % self.log_args.save_freq == 0 and 
                self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.config.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0 and
                self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            samples = self.sample()
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)

            self.epoch += 1

    # =========================== Evaluation Loop ============================
    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return
        
        self.adapter.eval()
        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples : List[BaseSample] = []
            
            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.show_progress_bar,
            ):
                generator = create_generator(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    'trajectory_indices': None, # No need to store all trajectories during evaluation
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
            
            # Compute rewards with eval reward models
            rewards = self.eval_reward_processor.compute_rewards(
                samples=all_samples,
                store_to_samples=False,
                epoch=self.epoch,
                split='pointwise',  # Only `pointwise` reward can be compute when evaluation, since there is no `group` here.
            )
            # Gather and log rewards
            rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
            gathered_rewards = {
                key: self.accelerator.gather(value).cpu().numpy()
                for key, value in rewards.items()
            }
            
            # Log statistics
            if self.accelerator.is_main_process:
                _log_data = {f'eval/reward_{key}_mean': np.mean(value) for key, value in gathered_rewards.items()}
                _log_data.update({f'eval/reward_{key}_std': np.std(value) for key, value in gathered_rewards.items()})
                _log_data['eval_samples'] = all_samples
                self.log_data(_log_data, step=self.step)
            self.accelerator.wait_for_everyone()

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO."""
        self.adapter.rollout()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices, # Selectively store required trajectory positions for memory efficiency
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)        
                samples.extend(sample_batch)

        return samples
    
    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        # Compute rewards and advantages for samples
        rewards = self.reward_processor.compute_rewards(samples, store_to_samples=True, epoch=self.epoch)
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)

        # Create batches for optimization
        # `BaseSample.stack` will try to stack all tensor fields,
        # stack non-tensor fields as a list, keep shared fields as single value
        sample_batches : List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
            BaseSample.stack(samples[i:i + self.training_args.per_device_batch_size])
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]

        self.adapter.train()
        loss_info = defaultdict(list)

        with self.autocast():
            for batch_idx, batch in enumerate(tqdm(
                sample_batches,
                total=len(sample_batches),
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            )):
                latents_index_map = batch['latent_index_map']  # (T+1,) LongTensor
                log_probs_index_map = batch['log_prob_index_map']  # (T,) LongTensor
                # Iterate through timesteps
                for idx, timestep_index in enumerate(tqdm(
                    self.adapter.scheduler.train_timesteps,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                )):
                    with self.accelerator.accumulate(*self.adapter.trainable_components):
                        # 1. Prepare inputs
                        # Get old log prob
                        old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
                        # Get current timestep data
                        num_timesteps = batch['timesteps'].shape[1]
                        t = batch['timesteps'][:, timestep_index]
                        t_next = (
                            batch['timesteps'][:, timestep_index + 1]
                            if timestep_index + 1 < num_timesteps
                            else torch.tensor(0, device=self.accelerator.device)
                        )
                        # Get latents
                        latents = batch['all_latents'][:, latents_index_map[timestep_index]]
                        next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]
                        # Prepare forward input
                        forward_inputs = {
                            **self.training_args, # Pass kwargs like `guidance_scale` and `do_classifier_free_guidance`
                            't': t,
                            't_next': t_next,
                            'latents': latents,
                            'next_latents': next_latents,
                            'compute_log_prob': True,
                            'noise_level': self.adapter.scheduler.noise_level,
                            **batch
                        }
                        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                        # 2. Forward pass
                        if self.enable_kl_loss:
                            if self.training_args.kl_type == 'v-based':
                                return_kwargs = ['log_prob', 'noise_pred', 'dt']
                            elif self.training_args.kl_type == 'x-based':
                                return_kwargs = ['log_prob', 'next_latents', 'next_latents_mean', 'dt']
                        else:
                            return_kwargs = ['log_prob', 'dt']
                        
                        forward_inputs['return_kwargs'] = return_kwargs
                        output = self.adapter.forward(**forward_inputs)

                        # 3. Compute loss
                        # Clip advantages
                        adv = batch['advantage']
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # PPO-style clipped loss
                        ratio = torch.exp(output.log_prob - old_log_prob)
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. Compute KL-div
                        if self.enable_kl_loss:
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                ref_forward_inputs = forward_inputs.copy()
                                ref_forward_inputs['compute_log_prob'] = False
                                if self.training_args.kl_type == 'v-based':
                                    # KL in velocity space
                                    ref_forward_inputs['return_kwargs'] = ['noise_pred']
                                    ref_output = self.adapter.forward(**ref_forward_inputs)
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True
                                    )
                                elif self.training_args.kl_type == 'x-based':
                                    # KL in latent space
                                    ref_forward_inputs['return_kwargs'] = ['next_latents_mean']
                                    ref_output = self.adapter.forward(**ref_forward_inputs)
                                    kl_div = torch.mean(
                                        ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True
                                    )
                            
                            kl_div = torch.mean(kl_div)
                            kl_loss = self.training_args.kl_beta * kl_div
                            loss += kl_loss
                            loss_info['kl_div'].append(kl_div.detach())
                            loss_info['kl_loss'].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        loss_info["clip_frac_high"].append(torch.mean((ratio > 1.0 + ratio_clip_range[1]).float()))
                        loss_info["clip_frac_low"].append(torch.mean((ratio < 1.0 + ratio_clip_range[0]).float()))

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # Communicate and log losses
                            loss_info = {
                                k: torch.stack(v).mean() 
                                for k, v in loss_info.items()
                            }
                            loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                            self.log_data(
                                {f'train/{k}': v for k, v in loss_info.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)

    # =========================== Advantage Computation ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal['sum', 'gdpo'], Callable]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages for GRPO.
        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors - these should be aligned with samples
            store_to_samples: Whether to store computed advantages back to samples' extra_kwargs
            aggregation_func: Method to aggregate advantages within each group.
                Options: 'sum' (default GRPO), 'gdpo' (GDPO-style), or a custom callable.
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        if aggregation_func == 'sum':
            return self.compute_advantage_weighted_sum(samples, rewards, store_to_samples)
        elif aggregation_func == 'gdpo':
            return self.compute_advantages_gdpo(samples, rewards, store_to_samples)
        elif callable(aggregation_func):
            return aggregation_func(self, samples, rewards, store_to_samples)
        else:
            raise ValueError(
                f"Unsupported advantage aggregation method: {aggregation_func}. "
                " Supported: ['sum', 'gdpo'] "
                "or a callable function that takes (trainer, samples, rewards, store_to_samples) as inputs."
            )
        

    def compute_advantage_weighted_sum(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True
    ) -> torch.Tensor:
        """
        Compute advantages for GRPO using weighted sum aggregation.
        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors - these should be aligned with samples
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages

        Notes:
            - If you want to customize advantage computation (e.g., different normalization),
            you can override this method in a subclass, e.g., for GDPO.
        """

        # 1. Get rewards
        rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
        gathered_rewards = {
            key: self.accelerator.gather(value).cpu().numpy()
            for key, value in rewards.items()
        }

        # 2. Aggregate rewards if multiple reward models
        aggregated_rewards = np.zeros_like(next(iter(gathered_rewards.values())), dtype=np.float64)
        for key, reward_array in gathered_rewards.items():
            # Simple weighted sum
            aggregated_rewards += reward_array * self.reward_models[key].config.weight

        # 3. Group rewards by unique_ids - each sample has its `unique_id` hashed from its prompt, conditioning, etc.
        unique_ids = torch.tensor([s.unique_id for s in samples], dtype=torch.int64, device=self.accelerator.device)
        gathered_ids = self.accelerator.gather(unique_ids).cpu().numpy()
        _unique_ids, group_indices, _counts = np.unique(gathered_ids, return_inverse=True, return_counts=True)
        
        # 4. Compute advantages within each group
        advantages = np.zeros_like(aggregated_rewards, dtype=np.float64)

        if self.training_args.global_std:
            std = max(np.std(aggregated_rewards, axis=0, keepdims=True), 1e-6)

        for group_id in np.unique(group_indices):
            mask = (group_indices == group_id)
            group_rewards = aggregated_rewards[mask]
            assert len(group_rewards) == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {len(group_rewards)}"

            mean = np.mean(group_rewards, axis=0, keepdims=True)
            if not self.training_args.global_std:
                std = max(np.std(group_rewards, axis=0, keepdims=True), 1e-6)
            
            advantages[mask] = (group_rewards - mean) / std

        # 5. Log statistics
        # Log per-reward mean
        _log_data = {
            f'train/reward_{key}_mean': np.mean(value)
            for key, value in gathered_rewards.items()
        }
        # Log per-reward std
        _log_data.update({
            f'train/reward_{key}_std': np.std(value)
            for key, value in gathered_rewards.items()
        })
        # Log aggregated reward zero std ratio
        zero_std_ratio = RewardProcessor.compute_group_zero_std_ratio(aggregated_rewards, group_indices)
        _log_data['train/reward_zero_std_ratio'] = zero_std_ratio
        # Log other stats
        _log_data.update({
            'train/reward_mean': np.mean(aggregated_rewards),
            'train/reward_std': np.std(aggregated_rewards),
            'train/adv_max': np.max(advantages),
            'train/adv_min': np.min(advantages),
            'train/adv_abs_mean': np.mean(np.abs(advantages)),
        })
        _log_data['train_samples'] = samples[:30]

        self.log_data(_log_data, step=self.step)

        # 6. Scatter advantages back to align with samples
        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        # Store advantages to samples' extra_kwargs
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs['advantage'] = adv

        return advantages


    def compute_advantages_gdpo(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True
    ) -> torch.Tensor:
        """
        Compute advantages using GDPO: normalize each reward group-wise first,
        then combine with weights and apply batch normalization.
        References:
        [1] GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization
            - https://arxiv.org/abs/2601.05242
        """
        # 1. Gather rewards across processes
        rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
        gathered_rewards = {
            key: self.accelerator.gather(value).cpu().numpy()
            for key, value in rewards.items()
        }

        # 2. Get group indices
        unique_ids = torch.tensor([s.unique_id for s in samples], dtype=torch.int64, device=self.accelerator.device)
        gathered_ids = self.accelerator.gather(unique_ids).cpu().numpy()
        _unique_ids, group_indices = np.unique(gathered_ids, return_inverse=True)

        # 3. Compute per-reward group-wise advantages
        all_reward_advantages = []
        for key, reward_array in gathered_rewards.items():
            reward_adv = np.zeros_like(reward_array, dtype=np.float64)
            
            for group_id in np.unique(group_indices):
                mask = (group_indices == group_id)
                group_rewards = reward_array[mask]
                
                mean = np.mean(group_rewards)
                std = max(np.std(group_rewards), 1e-6)
                reward_adv[mask] = (group_rewards - mean) / std
            
            all_reward_advantages.append(reward_adv * self.reward_models[key].config.weight)

        # 4. Combine and batch normalize
        combined_advantages = np.sum(all_reward_advantages, axis=0)
        bn_mean = np.mean(combined_advantages)
        bn_std = max(np.std(combined_advantages), 1e-6)
        advantages = (combined_advantages - bn_mean) / bn_std

        # 5. Log statistics
        # Log per-reward mean
        _log_data = {
            f'train/reward_{key}_mean': np.mean(value)
            for key, value in gathered_rewards.items()
        }
        # Log per-reward std
        _log_data.update({
            f'train/reward_{key}_std': np.std(value)
            for key, value in gathered_rewards.items()
        })
        # Log per-reward zero std ratio
        _log_data.update({
            f'train/reward_{key}_zero_std_ratio': RewardProcessor.compute_group_zero_std_ratio(arr, group_indices)
            for key, arr in gathered_rewards.items()
        })
        # Log combined stats
        _log_data.update({
            'train/batch_norm_mean': bn_mean,
            'train/batch_norm_std': bn_std,
            'train/adv_max': np.max(advantages),
            'train/adv_min': np.min(advantages),
            'train/adv_abs_mean': np.mean(np.abs(advantages)),
            'train_samples': samples[:30],
        })
        self.log_data(_log_data, step=self.step)

        # 6. Scatter back to local process
        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        # Store advantages to samples' extra_kwargs
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs['advantage'] = adv

        return advantages


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
        """Generate rollouts for GRPO."""
        self.adapter.rollout()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices, # Selectively store required trajectory positions for memory efficiency
                    'extra_call_back_kwargs': ['next_latents_mean'], # For GRPO-Guard, we need to store `next_latents_mean` for ratio normalization
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)        
                samples.extend(sample_batch)

        return samples
        
    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        self.adapter.train()
        # Compute rewards and advantages for samples
        rewards = self.reward_processor.compute_rewards(samples, store_to_samples=True, epoch=self.epoch)
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)
        
        # Create batches for optimization
        sample_batches : List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
            BaseSample.stack(samples[i:i + self.training_args.per_device_batch_size])
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]

        loss_info = defaultdict(list)

        with self.autocast():
            for batch_idx, batch in enumerate(tqdm(
                sample_batches,
                total=len(sample_batches),
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            )):
                latents_index_map = batch['latent_index_map']  # (T+1,) LongTensor
                log_probs_index_map = batch['log_prob_index_map']  # (T,) LongTensor
                callback_index_map = batch['callback_index_map'][0]  # (T,) LongTensor, shared across batch.
                # Iterate through timesteps
                for idx, timestep_index in enumerate(tqdm(
                    self.adapter.scheduler.train_timesteps,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                )):
                    with self.accelerator.accumulate(*self.adapter.trainable_components):
                        # 1. Prepare inputs
                        # Get old log prob
                        old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
                        # Get current timestep data
                        num_timesteps = batch['timesteps'].shape[1]
                        t = batch['timesteps'][:, timestep_index]
                        t_next = (
                            batch['timesteps'][:, timestep_index + 1]
                            if timestep_index + 1 < num_timesteps
                            else torch.tensor(0, device=self.accelerator.device)
                        )
                        # Get latents
                        latents = batch['all_latents'][:, latents_index_map[timestep_index]]
                        next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]
                        # Prepare forward input
                        forward_inputs = {
                            **self.training_args, # Pass kwargs like `guidance_scale` and `do_classifier_free_guidance`
                            't': t,
                            't_next': t_next,
                            'latents': latents,
                            'next_latents': next_latents,
                            'compute_log_prob': True,
                            'noise_level': self.adapter.scheduler.noise_level,
                            **batch
                        }
                        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                        # 2. Forward pass
                        return_kwargs = set(['log_prob', 'next_latents_mean', 'std_dev_t', 'dt'])
                        if self.enable_kl_loss:
                            if self.training_args.kl_type == 'v-based':
                                return_kwargs.add('noise_pred')
                            elif self.training_args.kl_type == 'x-based':
                                return_kwargs.add('next_latents_mean')
                        
                        forward_inputs['return_kwargs'] = list(return_kwargs)
                        output = self.adapter.forward(**forward_inputs)

                        # 3. Compute loss
                        # Clip advantages
                        adv = batch['advantage']
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # Reweighted ratio
                        scale_factor = torch.sqrt(-output.dt) * output.std_dev_t
                        old_next_latents_mean = batch['next_latents_mean']
                        mse = (output.next_latents_mean - old_next_latents_mean).flatten(1).pow(2).mean(dim=1)
                        ratio = torch.exp((output.log_prob - old_log_prob) * scale_factor + mse / (2 * scale_factor))
                        # PPO-style clipped loss
                        ratio = torch.exp(output.log_prob - old_log_prob)
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. Compute KL-div
                        if self.enable_kl_loss:
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                ref_forward_inputs = forward_inputs.copy()
                                ref_forward_inputs['compute_log_prob'] = False
                                if self.training_args.kl_type == 'v-based':
                                    # KL in velocity space
                                    ref_forward_inputs['return_kwargs'] = ['noise_pred']
                                    ref_output = self.adapter.forward(**ref_forward_inputs)
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True
                                    )
                                elif self.training_args.kl_type == 'x-based':
                                    # KL in latent space
                                    ref_forward_inputs['return_kwargs'] = ['next_latents_mean']
                                    ref_output = self.adapter.forward(**ref_forward_inputs)
                                    kl_div = torch.mean(
                                        ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True
                                    )
                            
                            kl_div = torch.mean(kl_div)
                            kl_loss = self.training_args.kl_beta * kl_div
                            loss += kl_loss
                            loss_info['kl_div'].append(kl_div.detach())
                            loss_info['kl_loss'].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        loss_info["clip_frac_high"].append(torch.mean((ratio > 1.0 + ratio_clip_range[1]).float()))
                        loss_info["clip_frac_low"].append(torch.mean((ratio < 1.0 + ratio_clip_range[0]).float()))

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # Communicate and log losses
                            loss_info = {
                                k: torch.stack(v).mean() 
                                for k, v in loss_info.items()
                            }
                            loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                            self.log_data(
                                {f'train/{k}': v for k, v in loss_info.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)