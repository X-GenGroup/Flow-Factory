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
Advantage Weighted Matching (AWM) Trainer
"""
import os
from typing import List, Dict, Optional, Literal
from functools import partial
from collections import defaultdict
import numpy as np
import torch
from torch.distributions import Normal
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .grpo import GRPOTrainer
from ..models.abc import BaseSample
from ..scheduler import SDESchedulerOutput
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class AWMTrainer(GRPOTrainer):
    """
    AWM (Advantage Weighted Matching) Trainer.
    
    Key features:
    - Independent timestep sampling (discrete, logit_normal, uniform)
    - Weighted log probability computation (uniform, t, huber, ghuber)
    - Dual KL penalties (ref model + EMA model)
    - Compatible with both full fine-tuning and LoRA
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_config()
        self._init_ema_named_parameters()
    
    def _validate_config(self):
        """Validate AWM-specific hyperparameters."""
        valid_time_types = ['discrete', 'discrete_wo_init', 'discrete_with_init', 'logit_normal', 'uniform']
        assert self.training_args.time_type in valid_time_types, \
            f"time_type must be in {valid_time_types}, got {self.training_args.time_type}"
        
        valid_weightings = ['Uniform', 't', 't**2', 'huber', 'ghuber']
        assert self.training_args.weighting in valid_weightings, \
            f"weighting must be in {valid_weightings}, got {self.training_args.weighting}"

    def _init_ema_named_parameters(self):
        """Initialize EMA named parameters for KL penalty."""
        if self.enable_ema_kl_penalty:
            # Determine device based on config
            ema_device = (
                self.accelerator.device 
                if self.training_args.ema_kl_device == "cuda" 
                else "cpu"
            )
            
            # Add EMA snapshot for KL computation
            self.adapter.add_named_parameters(
                name='ema_kl',
                target_components=self.model_args.target_components,
                device=ema_device,
                overwrite=True,
            )
            logger.info(f"Initialized EMA named parameters for KL penalty on {ema_device}")

    @property
    def enable_ema_kl_penalty(self) -> bool:
        """Check if EMA KL penalty is enabled."""
        return hasattr(self.training_args, 'ema_kl_beta') and self.training_args.ema_kl_beta > 0.0

    # ======================== Timestep Sampling ========================
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        reference_sample: Optional[BaseSample] = None,
    ) -> torch.Tensor:
        """
        Sample timesteps using configured strategy.
        
        Args:
            batch_size: Number of samples per timestep
            device: Target device
            reference_sample: Reference sample for discrete sampling
            
        Returns:
            timesteps: Shape (num_train_timesteps * batch_size, 1, 1, 1)
        """
        num_timesteps = self.training_args.num_train_timesteps
        time_type = self.training_args.time_type
        shift = self.training_args.time_shift
        
        if time_type == 'logit_normal':
            timesteps = self._sample_logit_normal(num_timesteps, batch_size, shift, device)
            
        elif time_type == 'uniform':
            timesteps = self._sample_uniform(num_timesteps, batch_size, shift, device)
            
        elif time_type in ['discrete', 'discrete_wo_init', 'discrete_with_init']:
            timesteps = self._sample_discrete(
                num_timesteps, batch_size, time_type, reference_sample, device
            )
        else:
            raise ValueError(f"Unknown time_type: {time_type}")
        
        return timesteps.view(-1, 1, 1, 1)

    def _sample_logit_normal(
        self, num_timesteps: int, batch_size: int, shift: float, device: torch.device
    ) -> torch.Tensor:
        """Logit-Normal stratified sampling with shift."""
        # Stratified uniform base
        base_uniform = (
            torch.arange(num_timesteps, device=device) + 
            torch.rand(num_timesteps, device=device)
        ) / num_timesteps
        
        # Transform to standard normal
        normal_dist = Normal(loc=0.0, scale=1.0)
        u_standard = normal_dist.icdf(torch.clamp(base_uniform, 1e-7, 1-1e-7))
        u_standard_shuffled = u_standard[torch.randperm(num_timesteps, device=device)]
        
        # Apply logit-normal transform with shift
        t_vector = torch.sigmoid(u_standard_shuffled)
        t_shifted = shift * t_vector / (1 + (shift - 1) * t_vector)
        
        timesteps = torch.repeat_interleave(t_shifted, repeats=batch_size)
        return torch.clamp(timesteps, min=0.01)

    def _sample_uniform(
        self, num_timesteps: int, batch_size: int, shift: float, device: torch.device
    ) -> torch.Tensor:
        """Uniform stratified sampling with shift."""
        lower, upper = 0.20, 1.0
        rand_u = torch.rand(num_timesteps, batch_size, device=device)
        normalized = (
            torch.arange(num_timesteps, device=device).unsqueeze(1) + rand_u
        ) / num_timesteps
        
        matrix = lower + normalized * (upper - lower)
        timesteps = torch.gather(matrix, 0, torch.rand_like(matrix).argsort(dim=0)).flatten()
        
        # Apply shift
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        return timesteps

    def _sample_discrete(
        self,
        num_timesteps: int,
        batch_size: int,
        time_type: str,
        reference_sample: BaseSample,
        device: torch.device,
    ) -> torch.Tensor:
        """Discrete timestep sampling from denoising trajectory."""
        assert reference_sample is not None, "reference_sample required for discrete sampling"
        
        num_inference_steps = len(reference_sample.timesteps)
        max_step_idx = int(num_inference_steps * self.training_args.timestep_fraction)
        
        # Determine step indices
        if time_type == 'discrete_with_init':
            init_index = torch.tensor([0], device=device, dtype=torch.long)
            remaining_indices = self._stratified_indices(
                num_timesteps - 1, 1, max_step_idx, device
            )
            t_indices = torch.cat([init_index, remaining_indices])
        else:
            start_idx = 0 if time_type == 'discrete' else 1
            t_indices = self._stratified_indices(
                num_timesteps, start_idx, max_step_idx, device
            )
        
        # Map to actual timestep values
        train_totalindex = torch.repeat_interleave(t_indices, repeats=batch_size)
        timesteps = reference_sample.timesteps[train_totalindex] / 1000  # Normalize to [0, 1]
        return timesteps

    def _stratified_indices(
        self, n: int, start: int, end: int, device: torch.device
    ) -> torch.Tensor:
        """Generate stratified random indices."""
        boundaries = torch.linspace(start, end, steps=n + 1, device=device)
        lower_bounds = boundaries[:-1].long()
        upper_bounds = boundaries[1:].long()
        rand_u = torch.rand(n, device=device)
        return lower_bounds + (rand_u * (upper_bounds - lower_bounds)).long()

    # ======================== Weighted Log Probability ========================
    
    def compute_log_prob_with_weighting(
        self,
        noise_pred: torch.Tensor,
        clean_latents: torch.Tensor,
        random_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted log probability (AWM-style).
        
        Args:
            noise_pred: Model prediction
            clean_latents: Target clean latents (x0)
            random_noise: Sampled noise
            timesteps: Timestep values (sigma)
            
        Returns:
            log_prob: Weighted log probabilities per sample
        """
        # Base log prob (negative MSE in double precision)
        target = random_noise.double() - clean_latents.double()
        log_prob = -(noise_pred.double() - target) ** 2
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
        # Apply weighting scheme
        weighting = self.training_args.weighting
        t = timesteps.view(-1)
        
        if weighting == 'Uniform':
            pass
        elif weighting == 't':
            log_prob = log_prob * t
        elif weighting == 't**2':
            log_prob = log_prob * (t ** 2)
        elif weighting == 'huber':
            log_prob = -(torch.sqrt(-log_prob + 1e-10) - 1e-5) * t
        elif weighting == 'ghuber':
            power = self.training_args.ghuber_power
            log_prob = -(
                torch.pow(-log_prob + 1e-10, power) 
                - torch.pow(torch.tensor(1e-10, device=log_prob.device, dtype=log_prob.dtype), power)
            ) * t / power
        
        return log_prob

    # ======================== Sampling with Off-Policy Support ========================
    
    def sample(self) -> List[BaseSample]:
        """Generate rollouts, optionally precomputing old_log_probs for off-policy."""
        samples = super().sample()  # Reuse parent's sampling logic
        
        # Off-policy: precompute old_log_probs with EMA parameters
        if self.training_args.off_policy:
            self._precompute_old_log_probs(samples)
        
        return samples

    def _precompute_old_log_probs(self, samples: List[BaseSample]):
        """Precompute old log probs using EMA parameters (off-policy only)."""
        logger.info("Precomputing old_log_probs with EMA parameters (off-policy mode)")
        
        # Use EMA named parameters
        with self.adapter.use_named_parameters('ema_kl'):
            sample_batches = [
                samples[i:i + self.training_args.per_device_batch_size]
                for i in range(0, len(samples), self.training_args.per_device_batch_size)
            ]
            
            for batch in tqdm(
                sample_batches,
                desc='Computing old_log_probs',
                disable=not self.accelerator.is_local_main_process,
            ):
                # Sample timesteps
                timesteps = self.sample_timesteps(
                    batch_size=len(batch),
                    device=self.accelerator.device,
                    reference_sample=batch[0],
                )
                num_train_timesteps = self.training_args.num_train_timesteps
                
                # Prepare inputs
                clean_latents = torch.cat([
                    torch.stack([s.all_latents[-1] for s in batch], dim=0)
                ] * num_train_timesteps, dim=0)
                
                random_noise = torch.cat([
                    torch.randn_like(clean_latents[:len(batch)])
                    for _ in range(num_train_timesteps)
                ], dim=0)
                
                noised_latents = (1 - timesteps) * clean_latents + timesteps * random_noise
                
                # Forward pass
                with torch.no_grad(), self.autocast():
                    output = self._forward_with_custom_timesteps(
                        batch * num_train_timesteps,
                        noised_latents,
                        timesteps,
                    )
                    
                    # Compute weighted log probs
                    log_prob = self.compute_log_prob_with_weighting(
                        noise_pred=output.noise_pred,
                        clean_latents=clean_latents,
                        random_noise=random_noise,
                        timesteps=timesteps,
                    )
                    log_prob = log_prob.reshape(num_train_timesteps, len(batch))
                
                # Store old_log_probs in samples
                for i, sample in enumerate(batch):
                    sample.extra_kwargs['old_log_probs'] = log_prob[:, i].cpu()

    # ======================== Training Optimization ========================
    
    def optimize(self, samples: List[BaseSample]) -> None:
        """
        Main training loop with AWM-specific timestep sampling.
        """
        self.adapter.train()
        
        # Compute rewards and advantages (reuse parent's logic)
        rewards = self.reward_processor.compute_rewards(
            samples, store_to_samples=True, epoch=self.epoch
        )
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)
        
        # Create batches
        sample_batches = [
            samples[i:i + self.training_args.per_device_batch_size]
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]
        
        loss_info = defaultdict(list)
        
        for batch_idx, batch in enumerate(tqdm(
            sample_batches,
            desc=f'Epoch {self.epoch} Training',
            disable=not self.accelerator.is_local_main_process,
        )):
            with self.accelerator.accumulate(self.adapter.transformer):
                # Sample timesteps for this batch (KEY DIFFERENCE from GRPO)
                timesteps = self.sample_timesteps(
                    batch_size=len(batch),
                    device=self.accelerator.device,
                    reference_sample=batch[0],
                )
                num_train_timesteps = self.training_args.num_train_timesteps
                
                # Prepare inputs: replicate for each timestep
                clean_latents = torch.cat([
                    torch.stack([s.all_latents[-1] for s in batch], dim=0)
                ] * num_train_timesteps, dim=0)
                
                random_noise = torch.cat([
                    torch.randn_like(clean_latents[:len(batch)])
                    for _ in range(num_train_timesteps)
                ], dim=0)
                
                noised_latents = (1 - timesteps) * clean_latents + timesteps * random_noise
                
                # Forward pass
                with self.autocast():
                    output = self._forward_with_custom_timesteps(
                        batch * num_train_timesteps,
                        noised_latents,
                        timesteps,
                    )
                    
                    # Compute weighted log probs
                    log_prob = self.compute_log_prob_with_weighting(
                        noise_pred=output.noise_pred,
                        clean_latents=clean_latents,
                        random_noise=random_noise,
                        timesteps=timesteps,
                    )
                    log_prob = log_prob.reshape(num_train_timesteps, len(batch))
                    
                    # Compute KL penalties
                    kl_div, ema_kl_div = self._compute_kl_penalties(
                        batch, output, noised_latents, timesteps, num_train_timesteps
                    )
                
                # Get old log probs
                old_log_prob = self._get_old_log_probs(batch, batch_idx, log_prob)
                
                # Compute PPO loss
                loss, loss_dict = self._compute_ppo_loss(
                    log_prob, old_log_prob, batch, kl_div, ema_kl_div
                )
                
                # Backward
                self.accelerator.backward(loss)
                
                # Accumulate loss info
                for k, v in loss_dict.items():
                    loss_info[k].append(v)
                
                # Gradient step
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        self.training_args.max_grad_norm,
                    )
                    
                    # Reduce and log
                    reduced_loss = {k: torch.stack(v).mean() for k, v in loss_info.items()}
                    reduced_loss = self.accelerator.reduce(reduced_loss, reduction="mean")
                    self.log_data(
                        {f'train/{k}': v for k, v in reduced_loss.items()},
                        step=self.step,
                    )
                    self.step += 1
                    loss_info = defaultdict(list)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Update EMA named parameters if enabled
        if self.enable_ema_kl_penalty:
            self._update_ema_named_parameters()

    # ======================== Helper Methods ========================
    
    def _forward_with_custom_timesteps(
        self,
        batch: List[BaseSample],
        noised_latents: torch.Tensor,
        timesteps: torch.Tensor,
        clean_latents: torch.Tensor,
    ) -> SDESchedulerOutput:
        """
        Forward pass with custom noised latents and timesteps.
        
        Args:
            batch: Original samples (for prompt_embeds, etc.)
            noised_latents: Shape (num_train_timesteps * batch_size, ...)
            timesteps: Shape (num_train_timesteps * batch_size, 1, 1, 1)
            clean_latents: Shape (num_train_timesteps * batch_size, ...) for next_latents
        
        Returns:
            SDESchedulerOutput with noise_pred, std_dev_t, etc.
        """
        num_train_timesteps = len(timesteps) // len(batch)
        
        # Store original state
        original_state = [
            {
                'all_latents': s.all_latents,
                'timesteps': s.timesteps,
            }
            for s in batch
        ]
        
        try:
            # Reshape inputs: (T*B, ...) -> (B, T, ...)
            noised_latents_reshaped = noised_latents.view(
                num_train_timesteps, len(batch), *noised_latents.shape[1:]
            ).transpose(0, 1).contiguous()  # (B, T, ...)
            
            timesteps_reshaped = timesteps.view(
                num_train_timesteps, len(batch), *timesteps.shape[1:]
            ).transpose(0, 1).contiguous()  # (B, T, 1, 1, 1)
            
            clean_latents_reshaped = clean_latents.view(
                num_train_timesteps, len(batch), *clean_latents.shape[1:]
            ).transpose(0, 1).contiguous()  # (B, T, ...)
            
            # Create fake trajectory for each sample
            for i, sample in enumerate(batch):
                # all_latents: [noised_0, noised_1, ..., clean]
                sample.all_latents = list(noised_latents_reshaped[i]) + [clean_latents_reshaped[i, -1]]
                # timesteps: [t_0, t_1, ..., t_T-1]
                sample.timesteps = (timesteps_reshaped[i].squeeze() * 1000).squeeze()  # Denormalize
            
            # Collect outputs for each timestep
            noise_preds = []
            std_dev_ts = []
            
            for t_idx in range(num_train_timesteps):
                forward_kwargs = {
                    'samples': batch,
                    'timestep_index': t_idx,
                    'compute_log_prob': False,
                    'return_kwargs': ['noise_pred', 'std_dev_t'],
                }
                forward_kwargs.update(filter_kwargs(self.adapter.forward, **self.training_args))
                
                output = self.adapter.forward(**forward_kwargs)
                noise_preds.append(output.noise_pred)
                std_dev_ts.append(output.std_dev_t)
            
            # Concatenate outputs: List[(B, ...)] -> (T*B, ...)
            output = SDESchedulerOutput(
                noise_pred=torch.cat(noise_preds, dim=0),
                std_dev_t=torch.cat(std_dev_ts, dim=0) if std_dev_ts[0] is not None else None,
            )
            
            return output
            
        finally:
            # Restore original state
            for i, sample in enumerate(batch):
                sample.all_latents = original_state[i]['all_latents']
                sample.timesteps = original_state[i]['timesteps']

    def _compute_kl_penalties(
        self, 
        batch, 
        output, 
        noised_latents,
        timesteps, 
        num_train_timesteps
    ):
        """Compute KL penalties with ref and EMA models."""
        kl_div = ema_kl_div = None
        
        if self.enable_kl_penalty:
            with torch.no_grad(), self.adapter.use_ref_parameters():
                ref_output = self._forward_with_custom_timesteps(
                    batch * num_train_timesteps,
                    noised_latents,
                    timesteps,
                )
            
            kl_div = torch.mean(
                ((output.noise_pred - ref_output.noise_pred) ** 2),
                dim=tuple(range(1, output.noise_pred.ndim)),
            ).mean()
        
        if self.enable_ema_kl_penalty:
            # Use EMA named parameters for KL
            with self.adapter.use_named_parameters('ema_kl'):
                with torch.no_grad():
                    ema_output = self._forward_with_custom_timesteps(
                        batch * num_train_timesteps,
                        noised_latents,
                        timesteps,
                    )
            
            ema_kl_div = torch.mean(
                ((output.noise_pred - ema_output.noise_pred) ** 2),
                dim=tuple(range(1, output.noise_pred.ndim)),
            ).mean()
        
        return kl_div, ema_kl_div

    def _get_old_log_probs(self, batch, batch_idx, current_log_prob):
        """Get old log probs (on-policy or off-policy)."""
        if not self.training_args.off_policy or batch_idx < self.training_args.gradient_accumulation_steps:
            return current_log_prob.detach()
        
        # Off-policy: use precomputed
        return torch.stack([
            s.extra_kwargs.get('old_log_probs', current_log_prob[:, i].detach())
            for i, s in enumerate(batch)
        ], dim=1).to(self.accelerator.device)

    def _compute_ppo_loss(self, log_prob, old_log_prob, batch, kl_div, ema_kl_div):
        """Compute PPO loss with clipping."""
        # Get advantages
        adv = torch.stack([s.extra_kwargs['advantage'] for s in batch], dim=0)
        adv_clip_range = self.training_args.adv_clip_range
        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
        
        # Compute ratio
        if self.training_args.loss_type == 'sum_first':
            ratio = torch.exp(log_prob.mean(dim=0) - old_log_prob.mean(dim=0))
        elif self.training_args.loss_type == 'exp_first':
            ratio = torch.exp(log_prob.view(-1) - old_log_prob.view(-1))
            adv = adv.unsqueeze(0).repeat(log_prob.size(0), 1).view(-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.training_args.loss_type}")
        
        # Clipped loss
        ratio_clip_range = self.training_args.clip_range
        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(
            ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1]
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        # Total loss
        loss = policy_loss
        loss_dict = {
            'policy_loss': policy_loss.detach(),
            'ratio': ratio.detach(),
            'clip_frac_high': torch.mean((ratio > 1.0 + ratio_clip_range[1]).float()),
            'clip_frac_low': torch.mean((ratio < 1.0 + ratio_clip_range[0]).float()),
        }
        
        if kl_div is not None:
            kl_penalty = self.training_args.kl_beta * kl_div
            loss += kl_penalty
            loss_dict['kl_div'] = kl_div.detach()
            loss_dict['kl_penalty'] = kl_penalty.detach()
        
        if ema_kl_div is not None:
            ema_kl_penalty = self.training_args.ema_kl_beta * ema_kl_div
            loss += ema_kl_penalty
            loss_dict['ema_kl_div'] = ema_kl_div.detach()
            loss_dict['ema_kl_penalty'] = ema_kl_penalty.detach()
        
        loss_dict['loss'] = loss.detach()
        return loss, loss_dict

    def _update_ema_named_parameters(self):
        """Update EMA named parameters with exponential moving average."""
        decay_type = self.training_args.kl_ema_decay_type
        
        if decay_type == 'constant':
            decay = self.training_args.kl_ema_decay
        elif decay_type == 'linear':
            decay = min(self.training_args.kl_ema_decay, 0.001 * self.step)
        else:
            raise ValueError(f"Unknown kl_ema_decay_type: {decay_type}")
        
        # Get current parameters
        current_params = self.adapter.get_trainable_parameters()
        
        # Manual EMA update
        info = self.adapter._named_parameters['ema_kl']
        with torch.no_grad():
            for ema_param, current_param in zip(info.ema_wrapper.ema_parameters, current_params):
                ema_param.data.copy_(
                    ema_param.data * decay + current_param.detach().to(ema_param.device) * (1.0 - decay)
                )
        
        logger.debug(f"Updated EMA named parameters with decay={decay:.6f}")