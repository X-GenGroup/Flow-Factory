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
Reference: https://arxiv.org/abs/2509.16117
"""
import os
from typing import List, Dict, Optional
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
from ..rewards import BaseRewardModel
from ..utils.base import filter_kwargs, create_generator
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

class DiffusionNFTTrainer(BaseTrainer):
    """
    DiffusionNFT Trainer.
    Reference: https://arxiv.org/abs/2509.16117
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

            with self.adapter.use_ema_parameters():
                # Sampling with the EMA model
                samples = self.sample()

            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)

            self.epoch += 1

    def sample(self, **kwargs) -> List[BaseSample]:
        """Generate rollouts for DiffusionNFT."""
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
                        'compute_log_prob': False, # No need to compute log probs during sampling
                        'extra_call_back_kwargs': ['noise_pred'],
                        **self.training_args,
                    }
                    sample_kwargs.update(**batch)
                    sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                    sample_batch = self.adapter.inference(**sample_kwargs)
            
            samples.extend(sample_batch)

        return samples

    def compute_rewards(self, samples: List[BaseSample], reward_models: Dict[str, BaseRewardModel], store_to_samples : bool = True) -> Dict[str, torch.Tensor]:
        """Compute rewards using multiple reward models."""
        
        name_to_rewards = {}

        for reward_name, reward_model in reward_models.items():
            rewards = []
            
            filtered_key_fields = filter_kwargs(reward_model.__call__, **samples[0])
            stackable_keys = None
            
            for i in tqdm(
                range(0, len(samples), reward_model.config.batch_size),
                desc=f'Epoch {self.epoch} Computing Rewards: {reward_name}',
                disable=not self.accelerator.is_local_main_process,
            ):
                batch_samples = [
                    {key: getattr(sample, key) for key in filtered_key_fields}
                    for sample in samples[i:i + reward_model.config.batch_size]
                ]

                if stackable_keys is None:
                    stackable_keys = {
                        key for key in filtered_key_fields
                        if isinstance(batch_samples[0][key], torch.Tensor)
                        and all(s[key].shape == batch_samples[0][key].shape for s in batch_samples)
                    }
                
                batch_samples = {
                    key: torch.stack([sample[key] for sample in batch_samples], dim=0) if key in stackable_keys
                    else [sample[key] for sample in batch_samples]
                    for key in filtered_key_fields
                }
                
                reward_output = reward_model(**batch_samples)
                reward_tensor = torch.as_tensor(
                    reward_output.rewards if hasattr(reward_output, 'rewards') else reward_output,
                    device='cpu',
                    dtype=torch.float32
                )
                rewards.append(reward_tensor)

            rewards = torch.cat(rewards, dim=0)
            name_to_rewards[reward_name] = rewards

        # Store `rewards` as a `Dict[str, Tensor(cpu)]` in extra_kwargs
        if store_to_samples:
            for i, sample in enumerate(samples):
                sample.extra_kwargs['rewards'] = {
                    key: value[i] for key, value in name_to_rewards.items()
                }
        
        return name_to_rewards

    def compute_advantages(self, samples: List[BaseSample], rewards: Dict[str, torch.Tensor], store_to_samples: bool = True) -> torch.Tensor:
        """Compute advantages with multi-reward support."""
        # 1. Gather rewards
        rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
        gathered_rewards = {
            key: self.accelerator.gather(value).cpu().numpy()
            for key, value in rewards.items()
        }

        # 2. Aggregate rewards with weights
        aggregated_rewards = np.zeros_like(next(iter(gathered_rewards.values())), dtype=np.float64)
        for key, reward_array in gathered_rewards.items():
            aggregated_rewards += reward_array * self.reward_models[key].config.weight

        # 3. Group by unique_ids
        unique_ids = torch.tensor([s.unique_id for s in samples], dtype=torch.int64, device=self.accelerator.device)
        gathered_ids = self.accelerator.gather(unique_ids).cpu().numpy()
        _unique_ids, group_indices = np.unique(gathered_ids, return_inverse=True)

        # 4. Compute group-wise advantages
        advantages = np.zeros_like(aggregated_rewards, dtype=np.float64)

        if self.training_args.global_std:
            std = max(np.std(aggregated_rewards, axis=0, keepdims=True), 1e-6)

        for group_id in np.unique(group_indices):
            mask = (group_indices == group_id)
            group_rewards = aggregated_rewards[mask]
            assert len(group_rewards) == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {len(group_rewards)}"

            mean = np.mean(group_rewards, axis=0, keepdims=True)
            if not self.training_args.global_std:
                std = max(np.std(group_rewards, axis=0, keepdims=True), 1e-6)
            
            advantages[mask] = (group_rewards - mean) / std

        # 5. Log statistics
        _log_data = {f'train/reward_{key}_mean': np.mean(value) for key, value in gathered_rewards.items()}
        _log_data.update({f'train/reward_{key}_std': np.std(value) for key, value in gathered_rewards.items()})
        _log_data.update({
            'train/reward_mean': np.mean(aggregated_rewards),
            'train/reward_std': np.std(aggregated_rewards),
            'train/adv_max': np.max(advantages),
            'train/adv_min': np.min(advantages),
            'train/adv_abs_mean': np.mean(np.abs(advantages)),
        })
        # Add rewards to sample for logging
        for sample, i in zip(samples, range(len(samples))):
            sample.extra_kwargs['rewards'] = {
                key: value for key, value in zip(gathered_rewards.keys(), [arr[i] for arr in gathered_rewards.values()])
            }
        _log_data['train_samples'] = samples[:30]
        
        self.log_data(_log_data, step=self.step)

        # 6. Scatter back
        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        # Store advantages to samples' extra_kwargs
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs['advantage'] = adv
        
        return advantages

    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop: compute loss and update policy."""
        self.adapter.train()
        # Compute rewards and advantages for samples
        rewards = self.compute_rewards(samples, self.reward_models)
        advantages = self.compute_advantages(samples, rewards)
    
        # Add advantages to samples
        for sample, adv in zip(samples, advantages):
            sample.extra_kwargs['advantage'] = adv
        
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
                        with self.autocast():
                            # Forward pass
                            return_kwargs = ['noise_pred', 'next_latents', 'latents', 'dt']
                            forward_kwargs = {
                                'samples': batch,
                                'timestep_index': timestep_index,
                                'compute_log_prob': False,
                                'return_kwargs': return_kwargs,
                                **self.training_args,
                            }
                            forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
                            output = self.adapter.forward(**forward_kwargs)

                        # 1. Prepare variables                        
                        nft_beta = self.training_args.nft_beta if hasattr(self.training_args, 'nft_beta') else 1 # 1 as default
                        timestep = samples[0].timesteps[timestep_index].to(self.accelerator.device)
                        t = (timestep / 1000).view(-1, *([1] * (output.noise_pred.dim() - 1)))
                        x0 = torch.stack([
                            sample.all_latents[-1] for sample in batch
                        ], dim=0)
                        xt = torch.stack([
                            sample.all_latents[timestep_index] for sample in batch
                        ], dim=0)
                        old_v_pred = torch.stack([
                            sample.noise_pred[timestep_index] for sample in batch
                        ], dim=0)
                        new_v_pred = output.noise_pred

                        # 2. Compute adv
                        adv = torch.stack([
                            sample.extra_kwargs['advantage'] for sample in batch
                        ], dim=0)
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        normalized_advantages_clip = (adv / adv_clip_range[1]) / 2.0 + 0.5
                        r = torch.clamp(normalized_advantages_clip, 0, 1)

                        # 3. Compute loss
                        positive_prediction = nft_beta * new_v_pred + (1 - nft_beta) * old_v_pred.detach()
                        implicit_negative_prediction = (1.0 + nft_beta) * old_v_pred.detach() - nft_beta * new_v_pred

                        x0_prediction = xt - t * positive_prediction
                        with torch.no_grad():
                            weight_factor = (
                                torch.abs(x0_prediction.double() - x0.double())
                                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                .clip(min=0.00001)
                            )
                        positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))


                        negative_x0_prediction = xt - t * implicit_negative_prediction
                        with torch.no_grad():
                            negative_weight_factor = (
                                torch.abs(negative_x0_prediction.double() - x0.double())
                                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                .clip(min=0.00001)
                            )
                        negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                            dim=tuple(range(1, x0.ndim))
                        )

                        ori_policy_loss = (r * positive_loss  + (1.0 - r) * negative_loss) / nft_beta
                        policy_loss = (ori_policy_loss * adv_clip_range[1]).mean()

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

                        loss_info["policy_loss"].append(policy_loss.detach())
                        loss_info["unweighted_policy_loss"].append(ori_policy_loss.mean().detach())
                        loss_info["loss"].append(loss.detach())

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
            all_samples: List[BaseSample] = []
            
            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating', 
                disable=not self.accelerator.is_local_main_process
            ):
                generator = create_generator(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    **self.eval_args,
                    **batch,
                }
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                with torch.no_grad(), self.autocast():
                    samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
            
            # Multi-reward evaluation
            rewards = self.compute_rewards(all_samples, self.eval_reward_models)
            rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
            gathered_rewards = {key: self.accelerator.gather(value).cpu().numpy() for key, value in rewards.items()}
            
            if self.accelerator.is_main_process:
                _log_data = {f'eval/reward_{key}_mean': np.mean(value) for key, value in gathered_rewards.items()}
                _log_data.update({f'eval/reward_{key}_std': np.std(value) for key, value in gathered_rewards.items()})
                _log_data['eval_samples'] = all_samples
                self.log_data(_log_data, step=self.step)
            
            self.accelerator.wait_for_everyone()