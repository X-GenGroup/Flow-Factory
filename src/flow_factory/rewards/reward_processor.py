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

# src/flow_factory/rewards/reward_processor.py
"""
Unified Reward Processor for handling multiple reward models.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Set, Union
import torch
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator

from .abc import (
    BaseRewardModel,
    PointwiseRewardModel,
    GroupwiseRewardModel,
    RewardModelOutput,
)
from ..models.samples import BaseSample
from ..utils.dist import gather_samples
from ..utils.base import filter_kwargs
from ..utils.image import standardize_image_batch
from ..utils.video import standardize_video_batch

# ============================ Reward Processor ============================
class RewardProcessor:
    """
    Unified reward processor bound to specific reward models.
    
    Handles both PointwiseRewardModel and GroupwiseRewardModel seamlessly.
    """
    MEDIA_FIELDS = {'image', 'video', 'condition_images', 'condition_videos'} # Fields that may contain media data, requiring format conversion

    def __init__(
        self,
        accelerator: Accelerator,
        reward_models: Dict[str, BaseRewardModel],
        tokenizer: Optional[Any] = None,
    ):
        self.accelerator = accelerator
        self.reward_models = reward_models
        self.tokenizer = tokenizer
        
        # Pre-categorize models by type
        self._pointwise_models : Dict[str, PointwiseRewardModel] = {
            k: v for k, v in reward_models.items()
            if isinstance(v, PointwiseRewardModel)
        }
        self._groupwise_models : Dict[str, GroupwiseRewardModel] = {
            k: v for k, v in reward_models.items()
            if isinstance(v, GroupwiseRewardModel)
        }

    # ============================ Media Format Conversion ============================
    def _convert_media_to_pil(self, batch_input: Dict[str, Any], model: BaseRewardModel) -> Dict[str, Any]:
        """Convert tensor media fields to PIL format (unless model opts out)."""
        if getattr(model, 'use_tensor_inputs', False):
            output_type = 'pt'
        else:
            output_type = 'pil'
        
        result = {}
        for k, v in batch_input.items():
            if k not in self.MEDIA_FIELDS or v is None:
                result[k] = v
                continue
            if k == 'image':
                result[k] = standardize_image_batch(v, output_type=output_type)
            elif k == 'video':
                result[k] = standardize_video_batch(v, output_type=output_type)
            elif k == 'condition_images':
                result[k] = [
                    standardize_image_batch(imgs, output_type=output_type)
                    for imgs in v
                ]
            elif k == 'condition_videos':
                result[k] = [
                    standardize_video_batch(videos, output_type=output_type)
                    for videos in v
                ]

        return result
    
    # ============================ Public API ============================
    def compute_rewards(
        self,
        samples: List[BaseSample],
        store_to_samples: bool = True,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards using bound reward models.
        
        Args:
            samples: Local samples on this rank
            store_to_samples: Whether to store rewards in sample.extra_kwargs
            epoch: Current epoch for progress bar display
        
        Returns:
            Dict mapping reward_name -> rewards tensor aligned with local samples
        """
        results: Dict[str, torch.Tensor] = {}
        
        # Pointwise: local computation
        if self._pointwise_models:
            results.update(self._compute_pointwise_rewards(samples, epoch))
        
        # Groupwise: gather -> compute -> scatter
        if self._groupwise_models:
            results.update(self._compute_groupwise_rewards(samples, epoch))
        
        # Store to samples
        if store_to_samples:
            for i, sample in enumerate(samples):
                sample.extra_kwargs['rewards'] = {
                    k: v[i] for k, v in results.items()
                }
        
        return results

    # ============================ Pointwise Computation ============================
    def _compute_pointwise_rewards(
        self,
        samples: List[BaseSample],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for all PointwiseRewardModels."""
        results: Dict[str, torch.Tensor] = {}
        
        for name, model in self._pointwise_models.items():
            rewards = []
            batch_size = model.config.batch_size
            
            # Get required fields from model signature
            filtered_fields = filter_kwargs(model.__call__, **samples[0])
            
            for i in tqdm(
                range(0, len(samples), batch_size),
                desc=f'Epoch {epoch} Pointwise Rewards: {name}',
                disable=not self.accelerator.is_local_main_process,
            ):
                # Prepare batch input
                batch_samples = samples[i : i + batch_size]
                # Filter out fields with None values in any sample
                batch_input : Dict[str, List[Any]] = {
                    k: [getattr(s, k) for s in batch_samples]
                    for k in filtered_fields
                    if all(getattr(s, k) is not None for s in batch_samples)
                }
                # Convert media formats
                batch_input = self._convert_media_to_pil(batch_input, model)
                
                output = model(**batch_input)
                reward_tensor = torch.as_tensor(
                    output.rewards if hasattr(output, 'rewards') else output,
                    device='cpu', dtype=torch.float32
                )
                rewards.append(reward_tensor)
            
            results[name] = torch.cat(rewards, dim=0)
        
        return results

    # ============================ Groupwise Computation ============================
    def _compute_groupwise_rewards(
        self,
        samples: List[BaseSample],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for all GroupwiseRewardModels."""
        device = self.accelerator.device
        
        # 1. Collect required fields from all groupwise models
        required_fields: Set[str] = {'unique_id'}
        for model in self._groupwise_models.values():
            required_fields.update(model.required_fields)
        
        # Optimize: use prompt_ids instead of prompt strings for communication
        needs_decode = False
        if 'prompt' in required_fields:
            if hasattr(samples[0], 'prompt_ids') and samples[0].prompt_ids is not None:
                required_fields.discard('prompt')
                required_fields.add('prompt_ids')
                needs_decode = True
        
        # 2. Gather samples from all ranks
        gathered = gather_samples(
            accelerator=self.accelerator,
            samples=samples,
            field_names=list(required_fields),
            device=device,
        )
        
        # Decode prompts if needed
        if needs_decode:
            prompts = self._decode_prompts([s.prompt_ids for s in gathered])
            for i, s in enumerate(gathered):
                s.prompt = prompts[i]
        
        # 3. Group by unique_id
        groups, inverse = self.group_samples(gathered, key='unique_id', return_inverse=True)
        
        # 4. Compute rewards per group
        num_gathered = len(gathered)
        results: Dict[str, torch.Tensor] = {}
        
        for name, model in self._groupwise_models.items():
            all_rewards = torch.zeros(num_gathered, dtype=torch.float32)
            
            for idx, (uid, group_list) in enumerate(tqdm(
                groups.items(),
                desc=f'Epoch {epoch} Groupwise Rewards: {name}',
                disable=not self.accelerator.is_local_main_process,
            )):
                # Prepare group input
                fields = filter_kwargs(model.__call__, **group_list[0])
                # Filter out fields with None values in any sample
                group_input = {
                    k: [getattr(s, k) for s in group_list]
                    for k in fields
                    if all(getattr(s, k) is not None for s in group_list)
                }

                # Convert media formats
                group_input = self._convert_media_to_pil(group_input, model)
                
                output = model(**group_input)
                group_rewards = torch.as_tensor(
                    output.rewards if hasattr(output, 'rewards') else output,
                    dtype=torch.float32
                )
                
                # Assign to correct positions
                all_rewards[inverse == idx] = group_rewards
            
            results[name] = all_rewards
        
        # 5. Scatter back to local rank
        results = {
            k: v.chunk(self.accelerator.num_processes)[self.accelerator.process_index]
            for k, v in results.items()
        }
        
        return results

    # ============================ Prompt Encoding/Decoding ============================
    def _decode_prompts(self, prompt_ids_list: List[torch.Tensor]) -> List[str]:
        """Decode prompt_ids to strings."""
        if self.tokenizer is None:
            raise ValueError("Cannot decode prompts: tokenizer not provided")
        
        return [
            self.tokenizer.decode(
                ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids,
                skip_special_tokens=True
            )
            for ids in prompt_ids_list
        ]

    def _encode_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        """Encode strings to prompt_ids."""
        if self.tokenizer is None:
            raise ValueError("Cannot encode prompts: tokenizer not provided")
        
        return [
            self.tokenizer(text, return_tensors='pt', padding=False, truncation=True)
            .input_ids.squeeze(0)
            for text in prompts
        ]
    
    # ============================ Helper Functions ============================
    @staticmethod
    def compute_group_zero_std_ratio(
        rewards: np.ndarray, 
        group_indices: np.ndarray, 
        eps: float = 1e-6
    ) -> float:
        """
        Compute the fraction of groups with near-zero standard deviation.
        
        Args:
            rewards: Array of reward values
            group_indices: Array mapping each sample to its group
            eps: Threshold for considering std as zero
            
        Returns:
            Fraction of groups with std < eps
        """
        unique_groups = np.unique(group_indices)
        zero_std_count = sum(
            1 for gid in unique_groups 
            if np.std(rewards[group_indices == gid]) < eps
        )
        return zero_std_count / len(unique_groups)

    @staticmethod
    def group_samples(
        samples: List[BaseSample],
        key: str = 'unique_id',
        return_inverse: bool = False,
    ) -> Union[Dict[Any, List[BaseSample]], Tuple[Dict[Any, List[BaseSample]], np.ndarray]]:
        """
        Group samples by a key field, similar to np.unique.
        
        Args:
            samples: List of BaseSample instances
            key: Field name to group by (default: 'unique_id')
            return_inverse: If True, return indices to reconstruct original order
            return_index: If True, return first occurrence index for each group
        
        Returns:
            groups: Dict mapping key_value -> List[BaseSample]
            inverse: (optional) Array where inverse[i] gives group index for samples[i]
            index: (optional) Array of first occurrence indices for each unique key
        """
        keys = np.array([getattr(s, key) for s in samples])
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        
        groups: Dict[Any, List[BaseSample]] = {k: [] for k in unique_keys}
        for sample, k in zip(samples, keys):
            groups[k].append(sample)
        
        return (groups, inverse) if return_inverse else groups
