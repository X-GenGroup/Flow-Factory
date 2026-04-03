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

# src/flow_factory/trainers/nft.py
"""
DiffusionNFT Trainer.
Reference: 
[1] DiffusionNFT: Online Diffusion Reinforcement with Forward Process
    - https://arxiv.org/abs/2509.16117
"""
import os
from typing import List, Dict, Any, Union, Optional
from functools import partial
from collections import defaultdict
from contextlib import nullcontext, contextmanager
import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import NFTTrainingArguments
from ..samples import BaseSample
from ..rewards import RewardBuffer
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma

logger = setup_logger(__name__)



class DiffusionNFTTrainer(BaseTrainer):
    """
    DiffusionNFT Trainer with off-policy and continuous timestep support.
    Reference: https://arxiv.org/abs/2509.16117
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # NFT-specific config (from NFTTrainingArguments)
        self.training_args : NFTTrainingArguments
        self.nft_beta = self.training_args.nft_beta
        self.off_policy = self.training_args.off_policy

        # Timestep sampling config
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.num_train_timesteps = self.training_args.num_train_timesteps
        self.timestep_range = self.training_args.timestep_range

        self.kl_type = self.training_args.kl_type
        if self.kl_type != 'v-based':
            logger.warning(f"DiffusionNFT-Trainer only supports 'v-based' KL loss, got {self.kl_type}, switching to 'v-based'.")
            self.kl_type = 'v-based'

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

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample continuous or discrete timesteps based on configured `time_sampling_strategy`.

        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with scheduler-scale ``t`` in ``[0, 1000]``.
        """
        device = self.accelerator.device
        time_sampling_strategy = self.time_sampling_strategy.lower()
        available = ['logit_normal', 'uniform', 'discrete', 'discrete_with_init', 'discrete_wo_init']

        if time_sampling_strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif time_sampling_strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                shift=self.time_shift,
                device=device,
            )
        elif time_sampling_strategy.startswith('discrete'):
            discrete_config = {
                'discrete': (True, False),
                'discrete_with_init': (True, True),
                'discrete_wo_init': (False, False),
            }
            if time_sampling_strategy not in discrete_config:
                raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")

            include_init, force_init = discrete_config[time_sampling_strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")

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

    # =========================== Advantage Computation ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
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

            # Sampling: use EMA if off_policy
            with self.sampling_context():
                samples = self.sample()

            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DiffusionNFT."""
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
                    'compute_log_prob': False,
                    'trajectory_indices': [-1], # For NFT, only keep the final latents
                    **batch
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)


        return samples

    # =========================== Optimization Loop ============================
    def _compute_nft_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NFT forward pass for a single timestep.
        
        Args:
            batch: Batch containing prompt embeddings and other inputs.
            timestep: Timestep tensor of shape (B,) in scheduler scale ``[0, 1000]``.
            noised_latents: Interpolated latents ``x_t = (1-σ) x_1 + σ noise`` with ``σ = t/1000``.
        
        Returns:
            Dict with noise_pred.
        """
        t_b = timestep.view(-1) # Scale [0, 1000]

        forward_kwargs = {
            **self.training_args,
            't': t_b,
            't_next': torch.zeros_like(t_b),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            'noise_level': 0.0,
            **{k: v for k, v in batch.items() if k not in ['all_latents', 'timesteps', 'advantage']},
        }
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        
        output = self.adapter.forward(**forward_kwargs)
        
        return {
            'noise_pred': output.noise_pred,
        }

    def optimize(self, samples: List[BaseSample]) -> None:
        """
        Main optimization loop for DiffusionNFT.
        """
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]
            
            # Re-group samples into batches
            sample_batches: List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
                BaseSample.stack(shuffled_samples[i:i + self.training_args.per_device_batch_size])
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
            ]

            # ==================== Pre-compute: Timesteps, Noise, and Old V Predictions ====================
            self.adapter.rollout()
            with torch.no_grad(), self.autocast(), self.sampling_context():
                for batch in tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Pre-computing Old V Predictions',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    batch_size = batch['all_latents'].shape[0]
                    clean_latents = batch['all_latents'][:, -1]
                    
                    # Sample timesteps: (T, B)
                    all_timesteps = self._sample_timesteps(batch_size)
                    batch['_all_timesteps'] = all_timesteps
                    batch['_all_random_noise'] = [] # List[torch.Tensor]
                    
                    # Compute old v predictions with `sampling` policy
                    old_v_pred_list = []
                    for t_idx in range(self.num_train_timesteps):
                        t_flat = all_timesteps[t_idx]  # (B,) scheduler scale [0, 1000]
                        sigma_broadcast = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                        noise = randn_tensor(
                            clean_latents.shape,
                            device=clean_latents.device,
                            dtype=clean_latents.dtype,
                        )
                        batch['_all_random_noise'].append(noise)
                        noised_latents = (1 - sigma_broadcast) * clean_latents + sigma_broadcast * noise
                        old_output = self._compute_nft_output(batch, t_flat, noised_latents)
                        old_v_pred_list.append(old_output['noise_pred'].detach())
                    
                    batch['_old_v_pred_list'] = old_v_pred_list

            # ==================== Training Loop ====================
            self.adapter.train()
            loss_info = defaultdict(list)

            with self.autocast():
                for batch in tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Training',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    # Retrieve pre-computed data
                    batch_size = batch['all_latents'].shape[0]
                    clean_latents = batch['all_latents'][:, -1]
                    all_timesteps = batch['_all_timesteps']
                    all_random_noise = batch['_all_random_noise']
                    old_v_pred_list = batch['_old_v_pred_list']
                    # Iterate through timesteps
                    for t_idx in tqdm(
                        range(self.num_train_timesteps),
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # 1. Prepare inputs
                            t_flat = all_timesteps[t_idx]  # (B,) [0, 1000]
                            sigma_broadcast = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                            noise = all_random_noise[t_idx]
                            noised_latents = (1 - sigma_broadcast) * clean_latents + sigma_broadcast * noise
                            old_v_pred = old_v_pred_list[t_idx]
                            
                            # 2. Forward pass for current policy
                            output = self._compute_nft_output(batch, t_flat, noised_latents)
                            new_v_pred = output['noise_pred']
                            
                            # 3. Compute NFT loss
                            adv = batch['advantage']
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                            
                            # Normalize advantage to [0, 1]
                            normalized_adv = (adv / max(adv_clip_range)) / 2.0 + 0.5
                            r = torch.clamp(normalized_adv, 0, 1).view(-1, *([1] * (new_v_pred.dim() - 1)))
                            
                            # Positive/negative predictions
                            positive_pred = self.nft_beta * new_v_pred + (1 - self.nft_beta) * old_v_pred
                            negative_pred = (1.0 + self.nft_beta) * old_v_pred - self.nft_beta * new_v_pred
                            
                            # Positive loss
                            x0_pred = noised_latents - sigma_broadcast * positive_pred
                            with torch.no_grad():
                                weight = torch.abs(x0_pred.double() - clean_latents.double()).mean(
                                    dim=tuple(range(1, clean_latents.ndim)), keepdim=True
                                ).clip(min=1e-5)
                            positive_loss = ((x0_pred - clean_latents) ** 2 / weight).mean(dim=tuple(range(1, clean_latents.ndim)))
                            
                            # Negative loss
                            neg_x0_pred = noised_latents - sigma_broadcast * negative_pred
                            with torch.no_grad():
                                neg_weight = torch.abs(neg_x0_pred.double() - clean_latents.double()).mean(
                                    dim=tuple(range(1, clean_latents.ndim)), keepdim=True
                                ).clip(min=1e-5)
                            negative_loss = ((neg_x0_pred - clean_latents) ** 2 / neg_weight).mean(dim=tuple(range(1, clean_latents.ndim)))
                            
                            # Combined loss
                            ori_policy_loss = (r.squeeze() * positive_loss + (1.0 - r.squeeze()) * negative_loss) / self.nft_beta
                            policy_loss = (ori_policy_loss * adv_clip_range[1]).mean()
                            loss = policy_loss
                            
                            # 4. KL penalty
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_output = self._compute_nft_output(batch, t_flat, noised_latents)
                                # KL-loss in v-space
                                kl_div = torch.mean(
                                    (new_v_pred - ref_output['noise_pred']) ** 2,
                                    dim=tuple(range(1, new_v_pred.ndim))
                                )
                                kl_loss = self.training_args.kl_beta * kl_div.mean()
                                loss = loss + kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            # 5. Log per-timestep info
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['unweighted_policy_loss'].append(ori_policy_loss.mean().detach())
                            loss_info['loss'].append(loss.detach())
                                
                            # 6. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                # Log loss info
                                loss_info = {k: torch.stack(v).mean() for k, v in loss_info.items()}
                                loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                                loss_info['grad_norm'] = grad_norm
                                self.log_data({f'train/{k}': v for k, v in loss_info.items()}, step=self.step)
                                self.step += 1
                                loss_info = defaultdict(list)