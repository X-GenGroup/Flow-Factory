# src/flow_factory/trainers/grpo_trainer.py
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
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .trainer import BaseTrainer
from ..models.adapter import BaseSample
from ..utils.base import filter_kwargs, create_generator


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("flow_factory.train")

class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        """Main training loop."""
        while True:
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.training_args.save_freq > 0 and 
                self.epoch % self.training_args.save_freq == 0 and 
                self.training_args.save_dir
            ):
                save_path = os.path.join(self.training_args.save_dir, self.config.run_name, f"epoch_{self.epoch}")
                self.save_checkpoint(save_path)

            # Evaluation
            if (self.training_args.eval_args.eval_freq > 0 and self.epoch % self.training_args.eval_args.eval_freq == 0):
                self.evaluate()

            samples = self.sample()
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)

            self.epoch += 1

    def sample(self, **kwargs) -> List[BaseSample]:
        """Generate rollouts for GRPO."""
        self.adapter.train()
        self.adapter.transformer.eval()
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
                        'compute_log_probs': True,
                        **self.training_args.to_dict(),
                    }
                    sample_kwargs.update(**batch)
                    sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                    sample_batch = self.adapter.inference(**sample_kwargs)
            
            samples.extend(sample_batch)

        self.adapter.transformer.train()
        return samples

    def compute_rewards(self, samples: List[BaseSample]) -> torch.Tensor:
        """Compute rewards using the reward model."""
        rewards = []
        
        filtered_key_fields = filter_kwargs(self.reward_model.forward, **samples[0])
        
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
                
        return rewards

    def compute_advantages(self, samples: List[BaseSample]) -> torch.Tensor:
        """Compute advantages for GRPO."""

        # Compute rewards first
        rewards = self.compute_rewards(samples)
        prompt_ids = torch.stack([sample.prompt_ids for sample in samples], dim=0)
        prompt_ids = torch.as_tensor(prompt_ids, device=self.accelerator.device)
        for sample, reward in zip(samples, rewards):
            sample.extra_kwargs['reward'] = reward

        # Gather across processes
        gathered_prompt_ids = self.accelerator.gather(prompt_ids).cpu().numpy()
        gathered_rewards = self.accelerator.gather(rewards).cpu().numpy()

        # Compute advantages
        unique_prompt_ids, group_indices = np.unique(gathered_prompt_ids, axis=0, return_inverse=True)

        advantages = np.zeros_like(gathered_rewards, dtype=np.float64)

        if self.training_args.global_std:
            std = max(np.std(gathered_rewards, axis=0, keepdims=True), 1e-6)

        for group_id in np.unique(group_indices):
            mask = (group_indices == group_id)
            group_rewards = gathered_rewards[mask]

            assert len(group_rewards) == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {len(group_rewards)}"

            mean = np.mean(group_rewards, keepdims=True)
            if not self.training_args.global_std:
                std = max(np.std(group_rewards, axis=0, keepdims=True), 1e-6)
            
            advantages[mask] = (group_rewards - mean) / std

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

        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        return advantages

    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        self.adapter.train()
        advantages = self.compute_advantages(samples)

        batched_samples = [
            samples[i:i + self.training_args.per_device_batch_size]
            for i in range(0, len(samples), self.training_args.per_device_batch_size)
        ]
        batched_advantages = advantages.reshape(
            -1, self.training_args.per_device_batch_size
        )

        loss_info = defaultdict(list)

        for batch_idx, (batch_samples, batch_advantages) in enumerate(tqdm(
            zip(batched_samples, batched_advantages),
            total=len(batched_samples),
            desc=f'Epoch {self.epoch} Training',
            position=0,
            disable=not self.accelerator.is_local_main_process,
        )):
            with self.accelerator.accumulate(self.adapter.transformer):
                for idx, timestep_index in enumerate(tqdm(
                    self.adapter.scheduler.current_noise_steps,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.accelerator.is_local_main_process,
                )):
                        # Get old log probs
                        old_log_probs = torch.stack(
                            [sample.log_probs[timestep_index] for sample in batch_samples],
                            dim=0
                        )

                        with self.autocast():
                            # Forward pass
                            output = self.adapter.forward(
                                batch_samples,
                                timestep_index=timestep_index,
                                return_log_prob=True
                            )

                        # Clip advantages
                        adv_clip_range = self.training_args.adv_clip_range         
                        batch_advantages = torch.clamp(batch_advantages, adv_clip_range[0], adv_clip_range[1])

                        # PPO-style clipped loss
                        ratio = torch.exp(output.log_prob - old_log_probs)
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -batch_advantages * ratio
                        clipped_loss = -batch_advantages * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss
                        loss_info['ratio'].append(ratio.detach().cpu().mean().item())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach().cpu().mean().item())
                        loss_info['clipped_loss'].append(clipped_loss.detach().cpu().mean().item())
                        loss_info['policy_loss'].append(policy_loss.detach().cpu().item())
                        loss_info["clip_frac_high"].append(torch.mean((ratio > 1.0 + ratio_clip_range[1]).float()).item())
                        loss_info["clip_frac_low"].append(torch.mean((ratio < 1.0 + ratio_clip_range[0]).float()).item())

                        # Other normalization strategies:
                        # 1. Temp-FlowGRPO
                        # 2. GRPO-Guard

                        # Backward
                        self.accelerator.backward(loss)
                    
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        self.training_args.max_grad_norm,
                    )
                    self.log_data(
                        {f'train/{k}': np.mean(v) for k, v in loss_info.items()},
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
                    'compute_log_probs': False,
                    'generator': generator,
                    **self.training_args.eval_args.to_dict(),
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