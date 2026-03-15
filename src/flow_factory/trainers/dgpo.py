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

# src/flow_factory/trainers/dgpo.py
"""
DGPO (Direct Group Preference Optimization) Trainer.
Reference:
[1] DGPO: Discovering Diverse Generations via Direct Group Preference Optimization
    - ICLR 2026
"""
import os
from typing import List, Dict, Any, Union, Optional
from functools import partial
from collections import defaultdict
from contextlib import nullcontext, contextmanager
import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from .grpo import GRPOTrainer
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler

logger = setup_logger(__name__)


class DGPOTrainer(GRPOTrainer):
    """
    DGPO Trainer: Direct Group Preference Optimization for diffusion models.

    Uses a group-level DPO loss instead of per-sample PPO ratio loss.
    Partitions samples into groups by prompt, computes DSM losses vs a frozen
    reference model, aggregates group-level preference signals via sigmoid,
    and applies PPO-style DSM clipping using an EMA "old policy".

    Reference: DGPO (ICLR 2026)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # DGPO-specific config
        self.beta_dpo = getattr(self.training_args, 'beta_dpo', 100.0)
        self.use_shared_noise = getattr(self.training_args, 'use_shared_noise', True)
        self.clip_dsm = getattr(self.training_args, 'clip_dsm', True)
        self.switch_ema_ref = getattr(self.training_args, 'switch_ema_ref', 200)

        # Timestep sampling config (shared with NFT/AWM)
        self.off_policy = getattr(self.training_args, 'off_policy', False)
        self.time_sampling_strategy = getattr(self.training_args, 'time_sampling_strategy', 'discrete')
        self.time_shift = getattr(self.training_args, 'time_shift', 3.0)
        self.num_train_timesteps = getattr(self.training_args, 'num_train_timesteps', self.training_args.num_inference_steps)
        self.timestep_range = getattr(self.training_args, 'timestep_range', 0.6)

        # KL regularization
        self.kl_beta = getattr(self.training_args, 'kl_beta', 0.0)
        self.kl_type = getattr(self.training_args, 'kl_type', 'v-based')
        if self.kl_type != 'v-based':
            logger.warning(f"DGPOTrainer only supports 'v-based' KL loss, got {self.kl_type}, switching to 'v-based'.")
            self.kl_type = 'v-based'

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.kl_beta > 0.0

    @contextmanager
    def sampling_context(self):
        """
        Context manager for sampling.
        After `switch_ema_ref` steps, use EMA parameters for sampling;
        otherwise use current parameters (or EMA if off_policy is set).
        """
        if self.step >= self.switch_ema_ref or self.off_policy:
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

            # Sampling: use EMA after switch_ema_ref steps
            with self.sampling_context():
                samples = self.sample()

            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DGPO (ODE sampling, final latents only)."""
        self.adapter.rollout()
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
                    'trajectory_indices': [-1],  # Only keep final latents
                    **batch
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                samples.extend(sample_batch)

        return samples

    # =========================== Timestep Sampling ============================
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
                timestep_range=self.timestep_range,
                normalize=True,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")

    def _sample_shared_timesteps(self) -> torch.Tensor:
        """
        Sample timesteps once on rank 0 and broadcast to all GPUs.
        Mirrors DGPO's generate_shared_sampled_timesteps().

        Returns:
            Tensor of shape (num_train_timesteps,) with t in (0, 1),
            identical across all processes.
        """
        device = self.accelerator.device
        # Sample on rank 0 using _sample_timesteps with batch_size=1, then squeeze
        if self.accelerator.is_main_process:
            t = self._sample_timesteps(batch_size=1).squeeze(-1)  # (T,)
        else:
            t = torch.empty(self.num_train_timesteps, device=device)
        dist.broadcast(t, src=0)
        return t

    # =========================== Forward Pass ============================
    def _compute_dgpo_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DGPO forward pass for a single timestep.

        Args:
            batch: Batch containing prompt embeddings and other inputs.
            timestep: Timestep tensor of shape (B,) in [0, 1].
            noised_latents: Interpolated latents x_t = (1-t)*x_1 + t*noise.

        Returns:
            Dict with 'noise_pred' (velocity prediction).
        """
        t_scaled = (timestep * 1000).view(-1)  # Scale to [0, 1000], ensure (B,)

        forward_kwargs = {
            **self.training_args,
            't': t_scaled,
            't_next': torch.zeros_like(t_scaled),
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

    # =========================== Group Info ============================
    def _precompute_group_info(
        self,
        samples: List[BaseSample],
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Precompute group information from sample unique_ids.
        Adapted from DGPO's precompute_group_info().

        Groups samples by their unique_id (prompt hash) across all GPUs.

        Args:
            samples: List of samples in current batch.
            batch_size: Local batch size on this GPU.

        Returns:
            Dict with inverse_indices, local_group_indices, num_groups,
            local_start, local_end.
        """
        device = self.accelerator.device

        # Build unique_id tensor for local batch
        unique_ids = torch.tensor(
            [s.unique_id for s in samples],
            dtype=torch.int64,
            device=device,
        ).view(batch_size, -1)

        # Gather across all GPUs
        all_group_ids = self.accelerator.gather(unique_ids)

        _, inverse_indices = torch.unique(all_group_ids, dim=0, return_inverse=True)
        num_groups = inverse_indices.max().item() + 1

        rank = self.accelerator.process_index
        start_idx = rank * batch_size
        end_idx = start_idx + batch_size

        local_group_indices = inverse_indices[start_idx:end_idx]

        return {
            'inverse_indices': inverse_indices,
            'local_group_indices': local_group_indices,
            'num_groups': num_groups,
            'local_start': start_idx,
            'local_end': end_idx,
            'batch_size': batch_size,
        }

    def _generate_shared_noise_for_groups(
        self,
        x0: torch.Tensor,
        group_info: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Generate shared noise for each group, synchronized across GPUs.
        Adapted from DGPO's generate_shared_noise_for_groups().

        Args:
            x0: [batch_size, C, H, W] tensor to get shape from.
            group_info: Precomputed group info dict.

        Returns:
            noise_diffuse: [batch_size, C, H, W] with shared noise per group.
        """
        device = x0.device
        num_groups = group_info['num_groups']
        inverse_indices = group_info['inverse_indices']

        # Only rank 0 generates noise, then broadcast
        if self.accelerator.is_main_process:
            group_noises = torch.randn(num_groups, *x0.shape[1:], device=device)
        else:
            group_noises = torch.empty(num_groups, *x0.shape[1:], device=device)

        dist.broadcast(group_noises, src=0)

        # Index noise by group assignment for all samples
        all_noises = group_noises[inverse_indices]

        # Extract local GPU's portion
        noise_diffuse = all_noises[group_info['local_start']:group_info['local_end']]

        return noise_diffuse

    # =========================== Group DGPO Loss ============================
    def _compute_group_dgpo_loss(
        self,
        model_v: torch.Tensor,
        ref_v: torch.Tensor,
        target_v: torch.Tensor,
        advantages: torch.Tensor,
        group_info: Dict[str, Any],
        group_size: int,
        dsm_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute group-level DGPO loss with AllReduce.
        Adapted from DGPO's compute_group_dgpo_loss_allreduce().

        Args:
            model_v: Current model velocity prediction (B, C, H, W).
            ref_v: Reference model velocity prediction (B, C, H, W).
            target_v: Target velocity = noise - x0 (B, C, H, W).
            advantages: Per-sample advantages (B,).
            group_info: Precomputed group info dict.
            group_size: Expected group size.
            dsm_loss: Optional pre-computed DSM loss (possibly clipped).

        Returns:
            Scalar group DGPO loss.
        """
        batch_size = model_v.shape[0]
        device = model_v.device

        # Compute DSM losses
        if dsm_loss is None:
            dsm_loss = (target_v - model_v).square().reshape(batch_size, -1).mean(dim=1)
        with torch.no_grad():
            ref_dsm_loss = (target_v - ref_v).square().reshape(batch_size, -1).mean(dim=1)

        # Compute delta and per-sample terms
        delta_diff = dsm_loss.detach() - ref_dsm_loss.detach()
        per_sample_term = advantages * self.beta_dpo * delta_diff / group_size

        local_group_indices = group_info['local_group_indices']
        num_groups = group_info['num_groups']

        # Scatter-add into group sums
        local_group_sums = torch.zeros(num_groups, device=device, dtype=per_sample_term.dtype)
        local_group_sums.scatter_add_(0, local_group_indices, per_sample_term)

        # All-reduce across GPUs
        global_group_sums = local_group_sums.clone().detach()
        dist.all_reduce(global_group_sums, op=dist.ReduceOp.SUM)

        # Sigmoid group weights
        group_weights = torch.sigmoid(global_group_sums)
        local_weights = group_weights[local_group_indices]

        # Final loss
        loss = (local_weights.detach() * advantages * dsm_loss).mean()

        return loss

    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """
        Main optimization loop for DGPO.

        1. Compute rewards & advantages (inherited from GRPO)
        2. For each inner epoch:
           a. Shuffle and batch samples
           b. Pre-compute: timesteps, noise, group info, old_v (EMA) for clipping
           c. Training: for each batch x timestep, compute DGPO group loss
        """
        # Compute rewards and advantages
        rewards = self.reward_processor.compute_rewards(samples, store_to_samples=True, epoch=self.epoch)
        advantages = self.compute_advantages(samples, rewards, store_to_samples=True)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            # Re-group into batches
            sample_batches: List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
                BaseSample.stack(shuffled_samples[i:i + self.training_args.per_device_batch_size])
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
            ]
            # Keep original sample lists for group info computation
            sample_lists = [
                shuffled_samples[i:i + self.training_args.per_device_batch_size]
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
            ]

            # ==================== Pre-compute Phase ====================
            # Sample shared timesteps once per inner_epoch, broadcast across GPUs
            # (mirrors DGPO's generate_shared_sampled_timesteps)
            shared_timesteps = self._sample_shared_timesteps()  # (T,)

            self.adapter.rollout()
            with torch.no_grad(), self.autocast(), self.sampling_context():
                for batch_idx, batch in enumerate(tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Pre-computing',
                    position=0,
                    disable=not self.show_progress_bar,
                )):
                    batch_size = batch['all_latents'].shape[0]
                    clean_latents = batch['all_latents'][:, -1]

                    # Precompute group info
                    group_info = self._precompute_group_info(
                        sample_lists[batch_idx], batch_size
                    )
                    batch['_group_info'] = group_info

                    # Expand shared timesteps to (T, B)
                    all_timesteps = shared_timesteps.unsqueeze(1).expand(-1, batch_size)
                    batch['_all_timesteps'] = all_timesteps
                    batch['_all_random_noise'] = []

                    # Compute old v predictions with EMA policy (for DSM clipping)
                    old_v_pred_list = []
                    for t_idx in range(self.num_train_timesteps):
                        t_flat = all_timesteps[t_idx]  # (B,)
                        t_broadcast = to_broadcast_tensor(t_flat, clean_latents)

                        # Generate noise (shared per group or random)
                        if self.use_shared_noise:
                            noise = self._generate_shared_noise_for_groups(
                                clean_latents, group_info
                            )
                        else:
                            noise = randn_tensor(
                                clean_latents.shape,
                                device=clean_latents.device,
                                dtype=clean_latents.dtype,
                            )
                        batch['_all_random_noise'].append(noise)

                        # Interpolate noised latents: x_t = (1-t)*x_0 + t*noise
                        noised_latents = (1 - t_broadcast) * clean_latents + t_broadcast * noise

                        # Compute old v prediction (EMA/rollout policy) for clipping
                        if self.clip_dsm:
                            old_output = self._compute_dgpo_output(batch, t_flat, noised_latents)
                            old_v_pred_list.append(old_output['noise_pred'].detach())
                        else:
                            old_v_pred_list.append(None)

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
                    group_info = batch['_group_info']

                    # Get advantages and clip
                    adv = batch['advantage']
                    adv_clip_range = self.training_args.adv_clip_range
                    adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

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
                            t_flat = all_timesteps[t_idx]  # (B,)
                            t_broadcast = to_broadcast_tensor(t_flat, clean_latents)
                            noise = all_random_noise[t_idx]
                            noised_latents = (1 - t_broadcast) * clean_latents + t_broadcast * noise
                            target_v = noise - clean_latents

                            # 2. Forward pass: current model (with grad)
                            output = self._compute_dgpo_output(batch, t_flat, noised_latents)
                            model_v = output['noise_pred']

                            # 3. Reference model prediction (no grad)
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                ref_output = self._compute_dgpo_output(batch, t_flat, noised_latents)
                                ref_v = ref_output['noise_pred']

                            # 4. Compute DSM loss
                            dsm_loss = (target_v - model_v).square().reshape(batch_size, -1).mean(dim=1)

                            # 5. PPO-style DSM clipping (using pre-computed old_v from EMA)
                            if self.clip_dsm and old_v_pred_list[t_idx] is not None:
                                old_v = old_v_pred_list[t_idx]
                                old_dsm_loss = (target_v - old_v).square().reshape(batch_size, -1).mean(dim=1)
                                ratio = torch.exp(-dsm_loss.detach() + old_dsm_loss.detach())
                                clip_range = self.training_args.clip_range
                                # Asymmetric clipping based on advantage sign
                                should_clip = torch.where(
                                    adv > 0,
                                    ratio > 1.0 + clip_range[1],
                                    ratio < 1.0 + clip_range[0],
                                )
                                # Detach loss where clipped to stop gradients
                                dsm_loss = torch.where(should_clip, dsm_loss.detach(), dsm_loss)
                                loss_info['clip_ratio'].append(should_clip.float().mean().detach())

                            # 6. Compute group DGPO loss
                            dgpo_loss = self._compute_group_dgpo_loss(
                                model_v=model_v,
                                ref_v=ref_v,
                                target_v=target_v,
                                advantages=adv,
                                group_info=group_info,
                                group_size=self.training_args.group_size,
                                dsm_loss=dsm_loss,
                            )
                            loss = dgpo_loss

                            # 7. KL penalty (v-based)
                            if self.enable_kl_loss:
                                kl_div = (model_v - ref_v).square().reshape(batch_size, -1).mean(dim=1)
                                kl_loss = self.kl_beta * kl_div.mean()
                                loss = loss + kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            # 8. Log
                            loss_info['dgpo_loss'].append(dgpo_loss.detach())
                            loss_info['dsm_loss'].append(dsm_loss.mean().detach())
                            loss_info['loss'].append(loss.detach())

                            # 9. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                # Log loss info
                                loss_info = {k: torch.stack(v).mean() for k, v in loss_info.items()}
                                loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                                self.log_data({f'train/{k}': v for k, v in loss_info.items()}, step=self.step)
                                self.step += 1
                                loss_info = defaultdict(list)
