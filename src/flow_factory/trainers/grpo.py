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
import torch
import numpy as np
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import GRPOTrainingArguments
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt
from ..utils.logger_utils import setup_logger
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
        self.training_args : GRPOTrainingArguments
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.log_args.save_freq > 0 and 
                self.epoch % self.log_args.save_freq == 0 and 
                self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
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
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples : List[BaseSample] = []
            
            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    'trajectory_indices': None, # No need to store trajectories during evaluation
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
        self.reward_buffer.clear() # Clear reward buffer
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
                self.reward_buffer.add_samples(sample_batch)

        return samples
    
    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        # Finalize reward computation and store to samples' extra_kwargs
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            # Create batches for optimization
            # `BaseSample.stack` will try to stack all tensor fields,
            # stack non-tensor fields as a list, keep shared fields as single value
            sample_batches : List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
                BaseSample.stack(shuffled_samples[i:i + self.training_args.per_device_batch_size])
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
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
                            loss_info['ratio_min'].append(ratio.min().detach())
                            loss_info['ratio_max'].append(ratio.max().detach())
                            loss_info['ratio_std'].append(ratio.std().detach())
                            loss_info['unclipped_loss'].append(unclipped_loss.detach())
                            loss_info['clipped_loss'].append(clipped_loss.detach())
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                            # 6. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
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
                                loss_info['grad_norm'] = grad_norm
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
        """Compute advantages — delegates to AdvantageProcessor.

        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors aligned with samples
            store_to_samples: Whether to store computed advantages back to samples' extra_kwargs
            aggregation_func: Method to aggregate advantages within each group.
                Options: 'sum' (default GRPO), 'gdpo' (GDPO-style), or a custom callable.
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
            step=self.step,
        )


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
        self.reward_buffer.clear()
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
                self.reward_buffer.add_samples(sample_batch)

        return samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        # Finalize reward computation and store to samples' extra_kwargs
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            # Re-group samples into batches
            sample_batches : List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
                BaseSample.stack(shuffled_samples[i:i + self.training_args.per_device_batch_size])
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
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
                            old_next_latents_mean = batch['next_latents_mean'][:, callback_index_map[timestep_index]]
                            mse = (output.next_latents_mean - old_next_latents_mean).flatten(1).pow(2).mean(dim=1)
                            ratio = torch.exp((output.log_prob - old_log_prob) * scale_factor + mse / (2 * scale_factor))
                            # PPO-style clipped loss
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
                            loss_info['ratio_min'].append(ratio.min().detach())
                            loss_info['ratio_max'].append(ratio.max().detach())
                            loss_info['ratio_std'].append(ratio.std().detach())
                            loss_info['unclipped_loss'].append(unclipped_loss.detach())
                            loss_info['clipped_loss'].append(clipped_loss.detach())
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                            # 6. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
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
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)