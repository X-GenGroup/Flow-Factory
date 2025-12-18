# src/flow_factory/models/grpo_trainer.py
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from PIL import Image
from ..hparams.training_args import TrainingArguments
from .trainer import BaseTrainer
from ..scheduler.flow_matching import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput
from ..models.adapter import BaseAdapter, BaseSample

class GRPOTrainer(BaseTrainer):
    """
    Abstract Base Class for Flow-Factory trainers.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        """Main run function to start sampling/training/evaluation."""
        epoch = 0
        while True:
            if self.training_args.eval_args.eval_freq > 0 and epoch % self.training_args.eval_args.eval_freq == 0:
                self.evaluate()
            samples = self.sample()
            self.compute_loss(samples)

            epoch += 1

    def sample(self, **kwargs) -> List[BaseSample]:
        """Sampling function."""
        self.adapter.eval()
        samples = []
        data_iter = iter(self.dataloader)
        for batch_index in range(self.training_args.num_batches_per_epoch):
            batch = next(data_iter)
            with torch.no_grad():
                sample_batch = self.adapter.inference(**batch, **kwargs)
            samples.extend(sample_batch)

        return samples

    def compute_rewards(self, samples: List[BaseSample]) -> torch.Tensor:
        """Compute rewards for GRPO."""
        rewards = []
        # Filter key fields for reward model
        filtered_key_fields = samples[0].keys() & set(self.reward_model.__call__.__code__.co_varnames)
        for i in range(0, len(samples), self.reward_args.batch_size):
            batch_samples = [
                {key: getattr(sample, key) for key in filtered_key_fields}
                for sample in samples[i:i + self.reward_args.batch_size]
            ]
            batch_samples = {
                key : torch.stack([sample[key] for sample in batch_samples], dim=0)
                if isinstance(batch_samples[0][key], torch.Tensor) else
                [sample[key] for sample in batch_samples]
                for key in filtered_key_fields
            }
            reward_output = self.reward_model(**batch_samples)
            rewards.append(torch.tensor(reward_output.rewards, device=self.accelerator.device, dtype=torch.float32))

        rewards = torch.cat(rewards, dim=0)
        return rewards

    def compute_loss(self, samples: List[BaseSample]) -> None:
        """Main training loop."""

        rewards = self.compute_rewards(samples)  # (sample_num, )
        prompt_ids = torch.cat([sample.prompt_ids for sample in samples], dim=0)  # (sample_num, seq_len)

        # Gather all rewards across processes if in distributed setting
        gathered_prompt_ids = self.accelerator.gather(prompt_ids).cpu().numpy()        
        gathered_rewards = self.accelerator.gather(rewards).cpu().numpy()
        # Compute advantages
        
        _, group_indices = np.unique(gathered_prompt_ids, axis=0, return_inverse=True)

        advantages = np.zeros_like(gathered_rewards, dtype=np.float64)

        if self.training_args.global_std:
            std = np.std(gathered_rewards, axis=0, keepdims=True) + 1e-8

        for group_id in np.unique(group_indices):
            # Create a mask for the current group
            mask = (group_indices == group_id)
            
            # Extract rewards for this specific prompt
            group_rewards = gathered_rewards[mask]
            
            # GRPO Logic: Normalize rewards within the group
            # (reward - mean) / (std + epsilon)
            assert len(group_rewards) == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {len(group_rewards)}"

            mean = np.mean(group_rewards, keepdims=True)
            if not self.training_args.global_std:
                std = np.std(group_rewards, keepdims=True) + 1e-8
            advantages[mask] = (group_rewards - mean) / std

        # Convert advantages back to tensor and split
        advantages = torch.tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        self.adapter.train()
        # Compute loss using advantages
        with self.accelerator.accumulate(self.adapter):
            for timestep_index in self.adapter.scheduler.current_noise_steps:
                # Batch samples according to training args - per_device_batch_size
                batched_samples = [
                    samples[i:i + self.training_args.per_device_batch_size]
                    for i in range(0, len(samples), self.training_args.per_device_batch_size)
                ]
                batched_advantages = advantages.reshape(-1, self.training_args.per_device_batch_size)  # (batch_size, num_batches_per_epoch)

                for samples, advantages in zip(batched_samples, batched_advantages):
                    prev_samples = torch.stack([sample.all_latents[timestep_index + 1] for sample in samples], dim=0)
                    old_log_probs = torch.stack([sample.log_probs[timestep_index] for sample in samples], dim=0)

                    output = self.adapter.forward(samples, timestep_index=timestep_index, return_log_prob=True)

                    advantages = torch.clamp(advantages, -self.training_args.adv_clip_range[0], self.training_args.adv_clip_range[1])

                    ratio = torch.exp(output.log_prob - old_log_probs)
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - self.training_args.clip_range[0],
                        1.0 + self.training_args.clip_range[1],
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    # KL loss requires a copy of original model, requiring more memory, TODO later

                    loss = policy_loss

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.adapter.parameters(),
                            self.training_args.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return
        self.adapter.eval()
        data_iter = iter(self.test_dataloader)
        all_samples = []
        for batch_index in range(len(self.test_dataloader)):
            batch = next(data_iter)
            with torch.no_grad():
                samples = self.adapter.inference(**batch, compute_log_probs=False)
            all_samples.extend(samples)
    
        rewards = self.compute_rewards(all_samples).cpu().numpy()
        gathered_rewards = self.accelerator.gather(torch.tensor(rewards)).cpu().numpy()
        avg_reward = np.mean(gathered_rewards)
        std_reward = np.std(gathered_rewards)
        if self.accelerator.is_main_process:
            print(f"Evaluation - Average Reward: {avg_reward:.4f}, Std Reward: {std_reward:.4f}")
        