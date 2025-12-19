# src/flow_factory/trainers/grpo_trainer.py
"""
Group Relative Policy Optimization (GRPO) Trainer.
Implements GRPO algorithm for flow matching models.
"""
import os
from typing import List
from functools import partial
import inspect
import logging
import numpy as np
import torch
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .trainer import BaseTrainer
from ..models.adapter import BaseSample
from ..rewards.reward_model import BaseRewardModel

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("flow_factory.train")

class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        """Main training loop."""
        while True:
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.training_args.save_freq > 0 and 
                self.epoch % self.training_args.save_freq == 0 and 
                self.training_args.save_dir
            ):
                save_path = os.path.join(self.training_args.save_dir, self.training_args.run_name, f"epoch_{self.epoch}")
                self.save_checkpoint(save_path)

            # Evaluation
            if (self.training_args.eval_args.eval_freq > 0 and self.epoch % self.training_args.eval_args.eval_freq == 0):
                self.evaluate()
            
            self.memory_profiler.snapshot(f"before_epoch_{self.epoch}_sampling")
            samples = self.sample()
            self.memory_profiler.snapshot(f"after_epoch_{self.epoch}_sampling")
            
            # Track samples memory
            self.memory_profiler.track_samples(
                [s.to_dict() for s in samples], 
                stage=f"epoch_{self.epoch}_samples"
            )
            
            self.memory_profiler.snapshot(f"before_epoch_{self.epoch}_training")
            self.compute_loss(samples)
            self.memory_profiler.snapshot(f"after_epoch_{self.epoch}_training")
            
            # Print detailed report every epoch
            if self.epoch % 5 == 0:
                self.memory_profiler.print_full_report(stage=f"epoch_{self.epoch}")
            
            self.epoch += 1

    def sample(self, **kwargs) -> List[BaseSample]:
        """Generate rollouts for GRPO."""
        self.adapter.eval()
        samples = []
        data_iter = iter(self.dataloader)
        
        for batch_index in tqdm(
            range(self.training_args.num_batches_per_epoch),
            desc=f'Epoch {self.epoch} Sampling',
            disable=not self.accelerator.is_local_main_process,
        ):
            batch = next(data_iter)
            
            # Track input batch
            self.memory_profiler.track_tensors(
                batch, 
                stage=f"epoch_{self.epoch}_batch_{batch_index}_input"
            )
            
            with torch.no_grad():
                with self.autocast():
                    sample_batch = self.adapter.inference(**batch, compute_log_probs=True, **kwargs)
            
            # Track output samples
            self.memory_profiler.track_samples(
                [s.to_dict() for s in sample_batch],
                stage=f"epoch_{self.epoch}_batch_{batch_index}_output"
            )
            
            samples.extend(sample_batch)

        return samples

    def compute_rewards(self, samples: List[BaseSample]) -> torch.Tensor:
        """Compute rewards using the reward model."""
        rewards = []
        
        signature = inspect.signature(self.reward_model.__call__)
        reward_params = list(signature.parameters.keys())
        
        filtered_key_fields = [
            k for k in reward_params 
            if k in samples[0].keys() and k != 'self'
        ]
        
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
            
            # Track reward input batch
            self.memory_profiler.track_tensors(
                batch_samples,
                stage=f"epoch_{self.epoch}_reward_batch_{i//self.reward_args.batch_size}"
            )
            
            reward_output = self.reward_model(**batch_samples)
            reward_tensor = torch.as_tensor(
                reward_output.rewards if hasattr(reward_output, 'rewards') else reward_output,
                device=self.accelerator.device,
                dtype=torch.float32
            )
            
            # Track reward output
            self.memory_profiler.track_tensors(
                {'rewards': reward_tensor},
                stage=f"epoch_{self.epoch}_reward_output_{i//self.reward_args.batch_size}"
            )
            
            rewards.append(reward_tensor)

        rewards = torch.cat(rewards, dim=0)
        
        # Track final rewards
        self.memory_profiler.track_tensors(
            {'rewards_final': rewards},
            stage=f"epoch_{self.epoch}_rewards"
        )
        
        return rewards

    def print_memory_breakdown(self):
        """Prints the memory footprint of model components AND Optimizer."""
        components = {
            "Transformer": getattr(self.adapter.pipeline, "transformer", None),
            "VAE": getattr(self.adapter.pipeline, "vae", None),
            "Text Encoder 1": getattr(self.adapter.pipeline, "text_encoder", None),
            "Text Encoder 2": getattr(self.adapter.pipeline, "text_encoder_2", None),
        }

        logger.info("=" * 40)
        logger.info(f"GPU Memory Occupation on {self.accelerator.device}")
        
        total_mem = 0
        
        for name, module in components.items():
            if module is None: continue
            mem_params = sum(p.nelement() * p.element_size() for p in module.parameters())
            mem_buffers = sum(b.nelement() * b.element_size() for b in module.buffers())
            total_mb = (mem_params + mem_buffers) / 1024**2
            
            is_on_gpu = False
            try:
                if len(list(module.parameters())) > 0:
                    is_on_gpu = next(module.parameters()).is_cuda
            except: pass
            
            status = "(GPU)" if is_on_gpu else "(CPU/Offloaded)"
            logger.info(f"{name:<15}: {total_mb:>8.2f} MB {status}")
            if is_on_gpu: total_mem += total_mb

        opt_mem = 0
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        opt_mem += v.nelement() * v.element_size()
        
        opt_mb = opt_mem / 1024**2
        logger.info(f"{'Optimizer':<15}: {opt_mb:>8.2f} MB (GPU)")
        total_mem += opt_mb

        logger.info("-" * 40)
        logger.info(f"{'Total Tracked':<15}: {total_mem:>8.2f} MB")
        logger.info(f"{'Torch Allocated':<15}: {torch.cuda.memory_allocated()/1024**2:>8.2f} MB")
        logger.info(f"{'Torch Reserved':<15}: {torch.cuda.memory_reserved()/1024**2:>8.2f} MB")
        logger.info("=" * 40)

    def compute_loss(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute advantages and update policy."""
        
        # Compute rewards
        rewards = self.compute_rewards(samples)
        prompt_ids = torch.stack([sample.prompt_ids for sample in samples], dim=0)
        prompt_ids = torch.as_tensor(prompt_ids, device=self.accelerator.device)

        # Track rewards and prompt_ids
        self.memory_profiler.track_tensors(
            {'rewards': rewards, 'prompt_ids': prompt_ids},
            stage=f"epoch_{self.epoch}_compute_loss_inputs"
        )

        # Gather across processes
        gathered_prompt_ids = self.accelerator.gather(prompt_ids).cpu().numpy()
        gathered_rewards = self.accelerator.gather(rewards).cpu().numpy()

        # Compute advantages
        _, group_indices = np.unique(gathered_prompt_ids, axis=0, return_inverse=True)
        advantages = np.zeros_like(gathered_rewards, dtype=np.float64)

        if self.training_args.global_std:
            std = np.std(gathered_rewards, axis=0, keepdims=True) + 1e-8

        for group_id in np.unique(group_indices):
            mask = (group_indices == group_id)
            group_rewards = gathered_rewards[mask]

            assert len(group_rewards) == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {len(group_rewards)}"

            mean = np.mean(group_rewards, keepdims=True)
            if not self.training_args.global_std:
                std = np.std(group_rewards, keepdims=True) + 1e-8
            
            advantages[mask] = (group_rewards - mean) / std

        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        # Track advantages
        self.memory_profiler.track_tensors(
            {'advantages': advantages},
            stage=f"epoch_{self.epoch}_advantages"
        )

        # Training loop
        self.adapter.train()
        self.print_memory_breakdown()

        with self.accelerator.accumulate(self.adapter):
            batched_samples = [
                samples[i:i + self.training_args.per_device_batch_size]
                for i in range(0, len(samples), self.training_args.per_device_batch_size)
            ]
            batched_advantages = advantages.reshape(
                -1, self.training_args.per_device_batch_size
            )

            for batch_idx, (batch_samples, batch_advantages) in enumerate(tqdm(
                zip(batched_samples, batched_advantages),
                total=len(batched_samples),
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.accelerator.is_local_main_process,
            )):
                # Track batch advantages
                self.memory_profiler.track_tensors(
                    {'batch_advantages': batch_advantages},
                    stage=f"epoch_{self.epoch}_batch_{batch_idx}"
                )
                
                for timestep_idx, timestep_index in enumerate(tqdm(
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
                    
                    # Track old log probs
                    self.memory_profiler.track_tensors(
                        {'old_log_probs': old_log_probs},
                        stage=f"epoch_{self.epoch}_batch_{batch_idx}_timestep_{timestep_idx}"
                    )

                    with self.autocast():
                        # Forward pass
                        output = self.adapter.forward(
                            batch_samples, 
                            timestep_index=timestep_index, 
                            return_log_prob=True
                        )
                    
                    # Track forward output
                    self.memory_profiler.track_tensors(
                        {
                            'new_log_probs': output.log_prob,
                            'prev_sample': output.prev_sample
                        },
                        stage=f"epoch_{self.epoch}_batch_{batch_idx}_timestep_{timestep_idx}_output"
                    )

                    # Clip advantages
                    adv_clip_range = self.training_args.adv_clip_range         
                    batch_advantages = torch.clamp(batch_advantages, adv_clip_range[0], adv_clip_range[1])

                    # PPO-style clipped loss
                    ratio = torch.exp(output.log_prob - old_log_probs)
                    ratio_clip_range = self.training_args.clip_range

                    unclipped_loss = -batch_advantages * ratio
                    clipped_loss = -batch_advantages * torch.clamp(ratio, ratio_clip_range[0], ratio_clip_range[1])
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    
                    # Track loss components
                    self.memory_profiler.track_tensors(
                        {
                            'ratio': ratio,
                            'unclipped_loss': unclipped_loss,
                            'clipped_loss': clipped_loss,
                            'policy_loss': policy_loss
                        },
                        stage=f"epoch_{self.epoch}_batch_{batch_idx}_timestep_{timestep_idx}_loss"
                    )

                    # Backward and step
                    self.accelerator.backward(policy_loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            self.training_args.max_grad_norm,
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                # Snapshot after each batch
                if batch_idx % 10 == 0:
                    self.memory_profiler.snapshot(
                        f"epoch_{self.epoch}_after_batch_{batch_idx}"
                    )

    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return
        
        self.adapter.eval()
        all_samples = []
        
        for batch in tqdm(
            self.test_dataloader,
            desc='Evaluating',
            disable=not self.accelerator.is_local_main_process,
        ):
            with torch.no_grad():
                samples = self.adapter.inference(**batch, compute_log_probs=False)
            all_samples.extend(samples)
        
        # Compute rewards
        rewards = self.compute_rewards(all_samples)
        gathered_rewards = self.accelerator.gather(rewards).cpu().numpy()
        
        # Log statistics
        if self.accelerator.is_main_process:
            avg_reward = np.mean(gathered_rewards)
            std_reward = np.std(gathered_rewards)
            print(f"Evaluation - Avg Reward: {avg_reward:.4f}, Std Reward: {std_reward:.4f}")