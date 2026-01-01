# src/flow_factory/trainers/grpo.py
"""
Group Relative Policy Optimization (GRPO) Trainer.
Implements GRPO algorithm for flow matching models.
"""
import os
from typing import List
from functools import partial
from collections import defaultdict
import inspect
import logging
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .trainer import BaseTrainer
from ..models.adapter import BaseSample
from ..utils.base import filter_kwargs, create_generator
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)



class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    @property
    def enable_kl_penalty(self) -> bool:
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

    def sample(self, **kwargs) -> List[BaseSample]:
        """Generate rollouts for GRPO."""
        self.adapter.rollout()
        samples = []
        data_iter = iter(self.dataloader)
        
        for batch_index in tqdm(
            range(self.training_args.num_batches_per_epoch),
            desc=f'Epoch {self.epoch} Sampling',
            disable=not self.accelerator.is_local_main_process,
        ):
            batch = next(data_iter)
            
            with torch.no_grad(), self.autocast():
                    sample_kwargs = {
                        'compute_log_prob': True,
                        **self.training_args,
                    }
                    sample_kwargs.update(**batch)
                    sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                    sample_batch = self.adapter.inference(**sample_kwargs)
            
            samples.extend(sample_batch)

        return samples

    def compute_rewards(self, samples: List[BaseSample]) -> torch.Tensor:
        """Compute rewards using the reward model."""
        rewards = []
        
        filtered_key_fields = filter_kwargs(self.reward_model.__call__, **samples[0])
        
        for i in tqdm(
            range(0, len(samples), self.reward_args.batch_size),
            desc=f'Epoch {self.epoch} Computing Rewards',
            disable=not self.accelerator.is_local_main_process,
        ):
            batch_samples = [
                {key: getattr(sample, key) for key in filtered_key_fields}
                for sample in samples[i:i + self.reward_args.batch_size]
            ]
            
            batch_samples = {
                key: (torch.stack([sample[key] for sample in batch_samples], dim=0)
                      if isinstance(batch_samples[0][key], torch.Tensor)
                      else [sample[key] for sample in batch_samples])
                for key in filtered_key_fields
            }
            
            reward_output = self.reward_model(**batch_samples)
            reward_tensor = torch.as_tensor(
                reward_output.rewards if hasattr(reward_output, 'rewards') else reward_output,
                device=self.accelerator.device,
                dtype=torch.float32
            )
            
            rewards.append(reward_tensor)

        rewards = torch.cat(rewards, dim=0)
                
        # Add rewards to samples
        for sample, reward in zip(samples, rewards):
            sample.extra_kwargs['reward'] = reward

        return rewards

    def compute_advantages(self, samples: List[BaseSample]) -> torch.Tensor:
        """Compute advantages for GRPO."""

        # 1. Get rewards
        rewards = torch.stack([sample.extra_kwargs['reward'] for sample in samples], dim=0)
        rewards = torch.as_tensor(rewards, device=self.accelerator.device)
        gathered_rewards = self.accelerator.gather(rewards).cpu().numpy()

        # 2. Gather prompt ids
        # Pad if necessary
        if hasattr(self.adapter.tokenizer, 'pad_token_id') and self.adapter.tokenizer.pad_token_id is not None:
            pad_token_id = self.adapter.tokenizer.pad_token_id
        elif hasattr(self.adapter.tokenizer, 'eos_token_id') and self.adapter.tokenizer.eos_token_id is not None:
            pad_token_id = self.adapter.tokenizer.eos_token_id
        else:
            pad_token_id = 0

        prompt_ids_list = [sample.prompt_ids.to(self.accelerator.device) for sample in samples]
        prompt_ids = pad_sequence(prompt_ids_list, batch_first=True, padding_value=pad_token_id)

        if self.accelerator.num_processes > 1:
            local_max_len = torch.tensor(prompt_ids.shape[1], device=self.accelerator.device)
            global_max_len = self.accelerator.reduce(local_max_len, reduction="max")
            
            if local_max_len < global_max_len:
                padding_length = global_max_len - local_max_len
                prompt_ids = torch.nn.functional.pad(prompt_ids, (0, padding_length), value=pad_token_id)

        # Gather across processes
        gathered_prompt_ids = self.accelerator.gather(prompt_ids).cpu().numpy()
        decoded_prompts = self.adapter.tokenizer.batch_decode(
            gathered_prompt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # 3. Group rewards by prompt ids and compute advantages
        unique_prompt_ids, group_indices = np.unique(gathered_prompt_ids, axis=0, return_inverse=True)

        advantages = np.zeros_like(gathered_rewards, dtype=np.float64)

        if self.training_args.global_std:
            std = max(np.std(gathered_rewards, axis=0, keepdims=True), 1e-6)

        for group_id in np.unique(group_indices):
            mask = (group_indices == group_id)
            group_rewards = gathered_rewards[mask]

            assert len(group_rewards) == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {len(group_rewards)}"

            mean = np.mean(group_rewards, axis=0, keepdims=True)
            if not self.training_args.global_std:
                std = max(np.std(group_rewards, axis=0, keepdims=True), 1e-6)
            
            advantages[mask] = (group_rewards - mean) / std

        # 4. Log statistics
        self.log_data(
            {
                'train/reward_mean': np.mean(gathered_rewards),
                'train/reward_std': np.std(gathered_rewards),
                'train/adv_max': np.max(advantages),
                'train/adv_min': np.min(advantages),
                'train/adv_abs_mean': np.mean(np.abs(advantages)),
                'train_samples': samples[:30],
            },
            step=self.step,
        )

        # 5. Scatter advantages back to samples
        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        # Add advantages to samples
        for sample, adv in zip(samples, advantages):
            sample.extra_kwargs['advantage'] = adv

        return advantages

    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        self.adapter.train()
        # Compute rewards and advantages for samples
        rewards = self.compute_rewards(samples)
        advantages = self.compute_advantages(samples)
        
        sample_batches : List[List[BaseSample]] = [
            samples[i:i + self.training_args.per_device_batch_size]
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]

        loss_info = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(
            sample_batches,
            total=len(sample_batches),
            desc=f'Epoch {self.epoch} Training',
            position=0,
            disable=not self.accelerator.is_local_main_process,
        )):
            with self.accelerator.accumulate(self.adapter.transformer):
                for idx, timestep_index in enumerate(tqdm(
                    self.adapter.scheduler.train_timesteps,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.accelerator.is_local_main_process,
                )):
                        # Get old log prob
                        old_log_prob = torch.stack(
                            [sample.log_probs[timestep_index] for sample in batch],
                            dim=0
                        )
                        adv = torch.stack(
                            [sample.extra_kwargs['advantage'] for sample in batch],
                            dim=0
                        )

                        with self.autocast():
                            # Forward pass
                            if self.enable_kl_penalty:
                                if self.training_args.kl_type == 'v-based':
                                    return_kwargs = ['log_prob', 'noise_pred', 'std_dev_t', 'dt']
                                elif self.training_args.kl_type == 'x-based':
                                    return_kwargs = ['log_prob', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt']
                            else:
                                return_kwargs = ['log_prob', 'std_dev_t', 'dt']
                            output = self.adapter.forward(
                                batch,
                                timestep_index=timestep_index,
                                compute_log_prob=True,
                                return_kwargs=return_kwargs,
                            )

                        # Clip advantages
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # PPO-style clipped loss
                        ratio = torch.exp(output.log_prob - old_log_prob)
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        if self.enable_kl_penalty:
                            with self.autocast(), torch.no_grad(), self.adapter.use_ref_parameters():
                                if self.training_args.kl_type == 'v-based':
                                    # KL in velocity space
                                    ref_output = self.adapter.forward(
                                        batch,
                                        timestep_index=timestep_index,
                                        compute_log_prob=False,
                                        return_kwargs=['noise_pred'],
                                    )
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True
                                    ) / (2 * output.std_dev_t ** 2 + 1e-7)
                                elif self.training_args.kl_type == 'x-based':
                                    # KL in latent space
                                    ref_output = self.adapter.forward(
                                        batch,
                                        timestep_index=timestep_index,
                                        compute_log_prob=False,
                                        return_kwargs=['next_latents_mean'],
                                    )
                                    kl_div = torch.mean(
                                        ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True
                                    ) / (2 * output.std_dev_t ** 2 + 1e-7)
                            
                            kl_div = torch.mean(kl_div)
                            kl_penalty = self.training_args.kl_beta * kl_div
                            loss += kl_penalty
                            loss_info['kl_div'].append(kl_div.detach())
                            loss_info['kl_penalty'].append(kl_penalty.detach())


                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        loss_info["clip_frac_high"].append(torch.mean((ratio > 1.0 + ratio_clip_range[1]).float()))
                        loss_info["clip_frac_low"].append(torch.mean((ratio < 1.0 + ratio_clip_range[0]).float()))

                        # Backward
                        self.accelerator.backward(loss)
                    
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        self.training_args.max_grad_norm,
                    )
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
                
                self.optimizer.step()
                self.optimizer.zero_grad()

    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return
        
        self.adapter.eval()
        with self.adapter.use_ema_parameters():
            all_samples : List[BaseSample] = []
            
            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.accelerator.is_local_main_process,
            ):
                generator = create_generator(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                with torch.no_grad(), self.autocast():
                        samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
            
            # Compute rewards
            rewards = self.compute_rewards(all_samples)
            gathered_rewards = self.accelerator.gather(rewards).cpu().numpy()
            
            # Log statistics
            if self.accelerator.is_main_process:
                for sample, reward in zip(all_samples, gathered_rewards):
                    sample.extra_kwargs['reward'] = reward
                self.log_data(
                    {
                        'eval/reward': np.mean(gathered_rewards),
                        'eval/reward_std': np.std(gathered_rewards),
                        'eval_samples' : all_samples,
                    },
                    step=self.step,
                )
            self.accelerator.wait_for_everyone()



class GRPOGuardTrainer(GRPOTrainer):
    """
    GRPOGuard Trainer with reweighted loss.
    References:
    [1] GRPO-Guard: https://arxiv.org/abs/2510.22319
    [2] Temp-FlowGRPO: https://arxiv.org/abs/2508.04324
    """
    
    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        self.adapter.train()
        # Compute rewards and advantages for samples
        rewards = self.compute_rewards(samples)
        advantages = self.compute_advantages(samples)
        
        sample_batches : List[List[BaseSample]] = [
            samples[i:i + self.training_args.per_device_batch_size]
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]

        loss_info = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(
            sample_batches,
            total=len(sample_batches),
            desc=f'Epoch {self.epoch} Training',
            position=0,
            disable=not self.accelerator.is_local_main_process,
        )):
            with self.accelerator.accumulate(self.adapter.transformer):
                for idx, timestep_index in enumerate(tqdm(
                    self.adapter.scheduler.train_timesteps,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.accelerator.is_local_main_process,
                )):
                        # Get old log probs
                        old_log_prob = torch.stack(
                            [sample.log_probs[timestep_index] for sample in batch],
                            dim=0
                        )
                        adv = torch.stack(
                            [sample.extra_kwargs['advantage'] for sample in batch],
                            dim=0
                        )

                        with self.autocast():
                            # Forward pass
                            return_kwargs = ['log_prob', 'next_latents_mean', 'std_dev_t', 'dt']
                            output = self.adapter.forward(
                                batch,
                                timestep_index=timestep_index,
                                compute_log_prob=True,
                                return_kwargs=return_kwargs,
                            )

                        # Clip advantages
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                        # Reweighted ratio
                        scale_factor = torch.sqrt(-output.dt) * output.std_dev_t
                        old_next_latents_mean = torch.stack([sample.next_latents_mean[timestep_index] for sample in batch], dim=0)
                        mse = (output.next_latents_mean - old_next_latents_mean).flatten(1).pow(2).mean(dim=1)
                        ratio = torch.exp((output.log_prob - old_log_prob) * scale_factor + mse / (2 * scale_factor))

                        # PPO-style clipped loss
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        if self.enable_kl_penalty:
                            with self.autocast(), torch.no_grad(), self.adapter.use_ref_parameters():
                                if self.training_args.kl_type == 'v-based':
                                    # KL in velocity space
                                    ref_output = self.adapter.forward(
                                        batch,
                                        timestep_index=timestep_index,
                                        compute_log_prob=False,
                                        return_kwargs=['noise_pred'],
                                    )
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True
                                    ) / (2 * output.std_dev_t ** 2 + 1e-7)
                                elif self.training_args.kl_type == 'x-based':
                                    # KL in latent space
                                    ref_output = self.adapter.forward(
                                        batch,
                                        timestep_index=timestep_index,
                                        compute_log_prob=False,
                                        return_kwargs=['next_latents_mean'],
                                    )
                                    kl_div = torch.mean(
                                        ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True
                                    ) / (2 * output.std_dev_t ** 2 + 1e-7)
                            
                            kl_penalty = self.training_args.kl_beta * kl_div
                            loss += kl_penalty
                            loss_info['kl_div'].append(kl_div.detach())
                            loss_info['kl_penalty'].append(kl_penalty.detach())

                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info["clip_frac_high"].append(torch.mean((ratio > 1.0 + ratio_clip_range[1]).float()))
                        loss_info["clip_frac_low"].append(torch.mean((ratio < 1.0 + ratio_clip_range[0]).float()))

                        # Backward
                        self.accelerator.backward(loss)
                    
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        self.training_args.max_grad_norm,
                    )
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
                
                self.optimizer.step()
                self.optimizer.zero_grad()