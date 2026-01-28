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

# src/flow_factory/trainers/awm.py
"""
Advantage Weighted Matching (AWM) Trainer.
References:
[1] Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models
    - https://arxiv.org/pdf/2509.25050
"""
import os
from typing import List, Dict, Optional, Any, Union, Literal
from functools import partial
from collections import defaultdict
import math
from contextlib import nullcontext, contextmanager
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..models.abc import BaseSample
from .grpo import GRPOTrainer
from ..rewards import BaseRewardModel
from ..utils.base import filter_kwargs, create_generator, to_broadcast_tensor
from ..utils.noise_schedule import TimeSampler
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# ============================ AWM Trainer ============================
class AWMTrainer(GRPOTrainer):
    """
    Advantage Weighted Matching (AWM) Trainer.
    References:
    [1] Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models
        - https://arxiv.org/pdf/2509.25050
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # AWM-specific config (from training_args)
        self.time_sampling_strategy = getattr(self.training_args, 'time_sampling_strategy', 'logit_normal')
        self.time_shift = getattr(self.training_args, 'time_shift', 3.0)
        self.weighting = getattr(self.training_args, 'awm_weighting', 'Uniform')
        self.ghuber_power = getattr(self.training_args, 'ghuber_power', 0.25)
        self.off_policy = getattr(self.training_args, 'off_policy', False)
        self.num_train_timesteps = getattr(self.training_args, 'num_train_timesteps', self.training_args.num_inference_steps)
        self.timestep_fraction = getattr(self.training_args, 'timestep_fraction', 0.9) 

        # KL regularization
        self.kl_beta = getattr(self.training_args, 'kl_beta', 0.0)
        self.ema_kl_beta = getattr(self.training_args, 'ema_kl_beta', 0.0)
        self.kl_type = getattr(self.training_args, 'kl_type', 'v-based')
        if self.kl_type != 'v-based':
            logger.warning(f"DiffusionNFT-Trainer only supports 'v-based' KL loss, got {self.kl_type}, switching to 'v-based'.")
            self.kl_type = 'v-based'
    
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

            # Sample with EMA model if off-policy
            with self.sampling_context():
                samples = self.sample()
            
            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample continuous or discrete timesteps based on configured `time_sampling_strategy`.
        
        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with t in (0, 1).
        """
        device = self.accelerator.device
        time_sampling_strategy = self.time_sampling_strategy.lower()
        available = ['logit_normal', 'uniform', 'discrete', 'discrete_with_init', 'discrete_wo_init']
        
        if time_sampling_strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif time_sampling_strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                shift=self.time_shift,
                device=device,
            )
        elif time_sampling_strategy.startswith('discrete'):
            # Map time_sampling_strategy to (include_init, force_init)
            discrete_config = {
                'discrete':           (True,  False),
                'discrete_with_init': (True,  True),
                'discrete_wo_init':   (False, False),
            }
            if time_sampling_strategy not in discrete_config:
                raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")
            
            include_init, force_init = discrete_config[time_sampling_strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_fraction=self.timestep_fraction,
                normalize=True,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")
    
    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for AWM training."""
        self.adapter.rollout()
        samples = []
        data_iter = iter(self.dataloader)
        
        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.accelerator.is_local_main_process,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,  # Skip log prob computation during sampling
                    'trajectory_indices': [-1], # For AWM, only keep the final latents
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)        
                samples.extend(sample_batch)

        return samples

    # =========================== Optimization Loop ============================
    @staticmethod
    def compute_weighted_log_prob(
        model_output: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        weighting: Literal['Uniform', 't', 't**2', 'huber', 'ghuber'] = 'Uniform',
        ghuber_power: float = 0.25,
    ) -> torch.Tensor:
        """
        Compute weighted log probability (matching loss) for AWM.
        
        Args:
            model_output: Model's velocity prediction, shape varies by model.
            target: Target velocity = noise - clean_latents, same shape as model_output.
            timestep: Current timestep values (B,).
            weighting: Weighting scheme for the loss.
            ghuber_power: Power parameter for generalized huber loss.
        
        Returns:
            Weighted log probability tensor of shape (B,).
        """
        model_output = model_output.double()
        target = target.double()
        
        # Matching loss (negative MSE as log prob)
        # Mean over all dimensions except batch (dim 0)
        log_prob = -(model_output - target) ** 2
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))  # Dynamic: works for any shape
        
        t = timestep.view(-1)
        
        if weighting == 'Uniform':
            pass  # No reweighting
        elif weighting == 't':
            log_prob = log_prob * t
        elif weighting == 't**2':
            log_prob = log_prob * t ** 2
        elif weighting == 'huber':
            log_prob = -(torch.sqrt(-log_prob + 1e-10) - 1e-5) * t
        elif weighting == 'ghuber':
            eps = torch.tensor(1e-10, device=log_prob.device, dtype=log_prob.dtype)
            log_prob = -(
                torch.pow(-log_prob + eps, ghuber_power) - torch.pow(eps, ghuber_power)
            ) * t / ghuber_power
        else:
            raise ValueError(f"Unknown weighting method: {weighting}")
        
        return log_prob.float()

    def _compute_awm_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        random_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute AWM forward pass for a single timestep.
        
        Args:
            batch: Batch containing prompt embeddings and other inputs.
            timestep: Timestep tensor of shape (B,), with values in (0, 1).
            noised_latents: Interpolated latents x_t = (1-t)*x_1 + t*noise.
            clean_latents: Clean latents x_1 (final denoised).
            random_noise: Sampled noise.
        
        Returns:
            Dictionary with:
                - log_prob: (B,)
                - noise_pred: same shape as latents
        """
        t_scaled = (timestep * 1000).view(-1)  # Scale to [0, 1000], ensure (B,)
        
        # Prepare forward inputs
        forward_kwargs = {
            **self.training_args,
            't': t_scaled,
            't_next': torch.zeros_like(t_scaled),  # Use zeros as t_next
            'latents': noised_latents,
            'compute_log_prob': False,  # Compute log prob based on matching loss
            'return_kwargs': ['noise_pred'],
            **{k: v for k, v in batch.items() if k not in ['all_latents', 'timesteps', 'advantage']},
        }
        
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        
        output = self.adapter.forward(**forward_kwargs)
        
        # Target for matching loss: v_target = noise - x_1
        target = random_noise - clean_latents
        
        # Compute weighted log probability
        log_prob = self.compute_weighted_log_prob(
            model_output=output.noise_pred,
            target=target,
            timestep=timestep, # Input timestep (B,) in (0, 1)
            weighting=self.weighting,
            ghuber_power=self.ghuber_power,
        )
                
        return {
            'log_prob': log_prob,             # (B,)
            'noise_pred': output.noise_pred,  # Same shape as latents
        }

    def optimize(self, samples: List[BaseSample]) -> None:
        """
        Main optimization loop for AWM.
        
        Unlike GRPO which iterates over discrete timesteps from the trajectory,
        AWM decouples sampling/training timesteps and performs multiple passes
        over all sampled timesteps for each batch.
        """
        # Compute rewards and advantages for samples
        rewards = self.reward_processor.compute_rewards(samples, store_to_samples=True, epoch=self.epoch)
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)
        
        # Create batches
        sample_batches: List[Dict[str, Any]] = [
            BaseSample.stack(samples[i:i + self.training_args.per_device_batch_size])
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]

        # ==================== Pre-compute: Timesteps, Noise, and Old Log Probs ====================
        self.adapter.rollout()
        with torch.no_grad(), self.autocast(), self.sampling_context():
            for batch in tqdm(
                sample_batches,
                total=len(sample_batches),
                desc=f'Epoch {self.epoch} Pre-computing Old Log Probs',
                position=0,
                disable=not self.accelerator.is_local_main_process,
            ):
                batch_size = batch['all_latents'].shape[0]
                clean_latents = batch['all_latents'][:, -1]
                
                # Sample timesteps: (T, B)
                all_timesteps = self._sample_timesteps(batch_size)
                batch['_all_timesteps'] = all_timesteps
                batch['_all_random_noise'] = [] # List[torch.Tensor]
                
                # Compute old log probs with `sampling` policy
                old_log_probs_list = []
                for t_idx in range(self.num_train_timesteps):
                    # Prepare timesteps
                    t_flat = all_timesteps[t_idx]  # (B,)
                    t_broadcast = to_broadcast_tensor(t_flat, clean_latents)
                    # Prepare initial noise
                    noise = randn_tensor(
                        clean_latents.shape,
                        device=clean_latents.device,
                        dtype=clean_latents.dtype,
                    )
                    batch['_all_random_noise'].append(noise)
                    # Interpolate noised latents
                    noised_latents = (1 - t_broadcast) * clean_latents + t_broadcast * noise
                    
                    old_output = self._compute_awm_output(
                        batch, t_flat, noised_latents, clean_latents, noise
                    )
                    old_log_probs_list.append(old_output['log_prob'].detach())
                
                batch['_old_log_probs'] = old_log_probs_list

        # ==================== Training Loop ====================
        self.adapter.train()
        loss_info = defaultdict(list)
        
        with self.autocast():
            for batch in tqdm(
                sample_batches,
                total=len(sample_batches),
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.accelerator.is_local_main_process,
            ):
                # Retrieve pre-computed data
                batch_size = batch['all_latents'].shape[0]
                clean_latents = batch['all_latents'][:, -1]                
                all_timesteps = batch['_all_timesteps']  # (T, B)
                all_random_noise = batch['_all_random_noise']
                old_log_probs_list = batch['_old_log_probs']
                # Get advantages and clip
                adv = batch['advantage']
                adv_clip_range = self.training_args.adv_clip_range
                adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                ratio_clip_range = self.training_args.clip_range
                # Iterate over all timesteps for current batch
                for t_idx in tqdm(
                    range(self.num_train_timesteps),
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                        leave=False,
                        disable=not self.accelerator.is_local_main_process,
                    ):
                    with self.accelerator.accumulate(*self.adapter.trainable_components):
                        # 1. Prepare inputs for current timestep
                        t_flat = all_timesteps[t_idx]  # (B,)
                        t_broadcast = to_broadcast_tensor(t_flat, clean_latents)  # (B, 1, ..., 1)
                        
                        noise = all_random_noise[t_idx]
                        noised_latents = (1 - t_broadcast) * clean_latents + t_broadcast * noise
                        old_log_prob = old_log_probs_list[t_idx]  # (B,)
                        
                        # 2. Forward pass for current policy
                        current_output = self._compute_awm_output(
                            batch, t_flat, noised_latents, clean_latents, noise
                        )
                        
                        log_prob = current_output['log_prob']  # (B,)
                        
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
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                ref_output = self._compute_awm_output(
                                    batch, t_flat, noised_latents, clean_latents, noise
                                )
                            # KL-div in velocity space
                            noise_pred = current_output['noise_pred']
                            ref_noise_pred = ref_output['noise_pred']

                            # Uniform across all dimensions except batch
                            kl_div = ((noise_pred - ref_noise_pred) ** 2).mean(dim=tuple(range(1, noise_pred.ndim)))
                            kl_loss = self.kl_beta * kl_div.mean()
                            loss = loss + kl_loss
                            loss_info['kl_div'].append(kl_div.detach())
                            loss_info['kl_loss'].append(kl_loss.detach())
                        
                        # 5. EMA-based KL regularization
                        if self.enable_ema_kl_loss:
                            with torch.no_grad(), self.adapter.use_ema_parameters():
                                ema_output = self._compute_awm_output(
                                    batch, t_flat, noised_latents, clean_latents, noise
                                )
                            # KL-div in velocity space
                            noise_pred = current_output['noise_pred']
                            ema_noise_pred = ema_output['noise_pred']
                            
                            ema_kl = ((noise_pred - ema_noise_pred) ** 2).mean(dim=tuple(range(1, noise_pred.ndim)))
                            ema_kl_loss = self.ema_kl_beta * ema_kl.mean()
                            loss = loss + ema_kl_loss
                            loss_info['ema_kl_div'].append(ema_kl.detach())
                            loss_info['ema_kl_loss'].append(ema_kl_loss.detach())

                        # 6. Log per-timestep info
                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        loss_info['clip_frac_high'].append(torch.mean((ratio > 1.0 + ratio_clip_range[1]).float()))
                        loss_info['clip_frac_low'].append(torch.mean((ratio < 1.0 + ratio_clip_range[0]).float()))

                        # 6. Backward pass and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            loss_info = {k: torch.stack(v).mean() for k, v in loss_info.items()}
                            loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                            self.log_data({f'train/{k}': v for k, v in loss_info.items()}, step=self.step)
                            self.step += 1
                            loss_info = defaultdict(list)
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()