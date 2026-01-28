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
from .grpo import GRPOTrainer
from ..models.abc import BaseSample
from ..utils.base import filter_kwargs, create_generator, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler

logger = setup_logger(__name__)



class DiffusionNFTTrainer(GRPOTrainer):
    """
    DiffusionNFT Trainer with off-policy and continuous timestep support.
    Reference: https://arxiv.org/abs/2509.16117
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # NFT-specific config
        self.nft_beta = getattr(self.training_args, 'nft_beta', 1.0)
        self.off_policy = getattr(self.training_args, 'off_policy', False)
        
        # Timestep sampling config
        self.time_sampling_strategy = getattr(self.training_args, 'time_sampling_strategy', 'logit_normal')
        self.time_shift = getattr(self.training_args, 'time_shift', 3.0)
        self.num_train_timesteps = getattr(self.training_args, 'num_train_timesteps', self.training_args.num_inference_steps)
        self.timestep_fraction = getattr(self.training_args, 'timestep_fraction', 0.9)
    
        # Check args
        self.kl_type = getattr(self.training_args, 'kl_type', 'v-based')
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
                    'compute_log_prob': False,
                    'trajectory_indices': [-1], # For NFT, only keep the final latents
                    **batch
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                samples.extend(sample_batch)

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
            timestep: Timestep tensor of shape (B,) in [0, 1].
            noised_latents: Interpolated latents x_t = (1-t)*x_1 + t*noise.
        
        Returns:
            Dict with noise_pred.
        """
        t_scaled = (timestep * 1000).view(-1)  # Scale to [0, 1000], ensure (B,)
        
        forward_kwargs = {
            **self.training_args,
            't': t_scaled,
            't_next': torch.zeros_like(t_scaled),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
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
        # Compute rewards and advantages for samples
        rewards = self.reward_processor.compute_rewards(samples, store_to_samples=True, epoch=self.epoch)
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)
    
        sample_batches: List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
            BaseSample.stack(samples[i:i + self.training_args.per_device_batch_size])
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]

        # ==================== Pre-compute: Timesteps, Noise, and Old V Predictions ====================
        self.adapter.rollout()
        with torch.no_grad(), self.autocast(), self.sampling_context():
            for batch in tqdm(
                sample_batches,
                total=len(sample_batches),
                desc=f'Epoch {self.epoch} Pre-computing Old V Predictions',
                position=0,
                disable=not self.accelerator.is_local_main_process,
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
                    # Compute old v prediction
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
                disable=not self.accelerator.is_local_main_process,
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
                    disable=not self.accelerator.is_local_main_process,
                ):
                    with self.accelerator.accumulate(*self.adapter.trainable_components):
                        # 1. Prepare inputs
                        t_flat = all_timesteps[t_idx]  # (B,)
                        t_broadcast = to_broadcast_tensor(t_flat, clean_latents)
                        noise = all_random_noise[t_idx]
                        noised_latents = (1 - t_broadcast) * clean_latents + t_broadcast * noise
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
                        x0_pred = noised_latents - t_broadcast * positive_pred
                        with torch.no_grad():
                            weight = torch.abs(x0_pred.double() - clean_latents.double()).mean(
                                dim=tuple(range(1, clean_latents.ndim)), keepdim=True
                            ).clip(min=1e-5)
                        positive_loss = ((x0_pred - clean_latents) ** 2 / weight).mean(dim=tuple(range(1, clean_latents.ndim)))
                        
                        # Negative loss
                        neg_x0_pred = noised_latents - t_broadcast * negative_pred
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