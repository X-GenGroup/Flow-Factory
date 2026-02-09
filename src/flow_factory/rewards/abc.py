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

# src/flow_factory/rewards/abc.py
"""
Abstract Base Class for Reward Models
Provides common interface for all reward models.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, Literal
from enum import Enum, auto
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from accelerate import Accelerator
from diffusers.utils.outputs import BaseOutput

from ..hparams import RewardArguments


@dataclass
class RewardModelOutput(BaseOutput):
    """Output for reward models."""
    rewards: Union[torch.Tensor, np.ndarray, List[float]]
    extra_info: Optional[Dict[str, Any]] = None


class BaseRewardModel(ABC):
    """
    Abstract base class for all reward models.
    Provides common interface and utilities.
    """    
    # Fields required from `Sample` for reward computation
    required_fields: Tuple[str, ...] = ()

    # Whether the model accepts tensor inputs directly
    use_tensor_inputs: bool = False # True: the input media are torch.Tensors; False: the input media are PIL Images
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        self.accelerator = accelerator
        self.config = config
        self.device = (
            accelerator.device
            if config.device == torch.device('cuda')
            else config.device
        )
        self.dtype = config.dtype
        self.model: Optional[nn.Module] = None
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> RewardModelOutput:
        """Compute rewards. Signature varies by subclass."""
        pass
    
    def to(self, device: torch.device) -> BaseRewardModel:
        """Move model to device."""
        if self.model is not None:
            self.model.to(device)
        self.device = device
        return self


class PointwiseRewardModel(BaseRewardModel):
    """
    Reward model that computes independent rewards for each sample.
    This is the original behavior - each sample's reward depends only on itself.
    
    Usage:
        rewards = model(prompt=prompts, image=images)
    """
    required_fields: Tuple[str, ...] = ('image', 'prompt')
    use_tensor_inputs: bool = False # True: the input media are torch.Tensors; False: the input media are PIL Images
    
    @abstractmethod
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """
        Compute per-sample rewards.
        
        Args:
            prompt: List of text prompts (batch_size,)
            image: List[Image.Image]: each element is a generated PIL image. If `use_tensor_inputs` is True, this will be a list of torch.Tensors (C, H, W).
            video: List[List[Image.Image]]: each element is a list of frames (PIL Images) for a generated video. If `use_tensor_inputs` is True, this will be a list of torch.Tensors (T, C, H, W).
            condition_images
                - List[List[Image.Image]]: each inner list corresponds to one prompt, and contains multiple condition images (PIL Images).
                - If `use_tensor_inputs` is True and **all condition images are resized the same**, this will be a list of torch.Tensors (num_conditions, C, H, W).
                - If `use_tensor_inputs` is True and **condition images have varying sizes**, this will be a list of lists of torch.Tensors (C, H, W).
            condition_videos:
                - List[List[List[Image.Image]]]: each innermost list corresponds to one condition video (PIL Images).
                - If `use_tensor_inputs` is True and **all condition videos are resized the same**, this will be a list of torch.Tensors (num_conditions, T, C, H, W).
                - If `use_tensor_inputs` is True and **condition videos have varying sizes**, this will be a list of lists of torch.Tensors (T, C, H, W).
            **kwargs: Additional fields from Sample
        Returns:
            RewardModelOutput with rewards shape (batch_size,)
        """
        pass


class GroupwiseRewardModel(BaseRewardModel):
    """
    Reward model that computes rewards considering the entire group.
    Used for pairwise preferences, ranking losses, or contrastive rewards.
    
    The model receives all samples belonging to the same unique_id group.
    
    Usage:
        # Called once per group with all samples in that group
        rewards = model(
            group_id=unique_id,
            prompts=[sample.prompt for sample in group_samples],
            images=[sample.image for sample in group_samples],
            ...
        )
    """
    required_fields: Tuple[str, ...] = ('image', 'prompt')
    use_tensor_inputs: bool = False # True: the input media are torch.Tensors; False: the input media are PIL Images

    @abstractmethod
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """
        Compute group-aware rewards. Pairwise or ranking rewards can be computed here.
        
        Args:
            prompt: List of text prompts (batch_size,)
            image: List[Image.Image]: each element is a generated PIL image. If `use_tensor_inputs` is True, this will be a list of torch.Tensors (C, H, W).
            video: List[List[Image.Image]]: each element is a list of frames (PIL Images) for a generated video. If `use_tensor_inputs` is True, this will be a list of torch.Tensors (T, C, H, W).
            condition_images
                - List[List[Image.Image]]: each inner list corresponds to one prompt, and contains multiple condition images (PIL Images).
                - If `use_tensor_inputs` is True and **all condition images are resized the same**, this will be a list of torch.Tensors (num_conditions, C, H, W).
                - If `use_tensor_inputs` is True and **condition images have varying sizes**, this will be a list of lists of torch.Tensors (C, H, W).
            condition_videos:
                - List[List[List[Image.Image]]]: each innermost list corresponds to one condition video (PIL Images).
                - If `use_tensor_inputs` is True and **all condition videos are resized the same**, this will be a list of torch.Tensors (num_conditions, T, C, H, W).
                - If `use_tensor_inputs` is True and **condition videos have varying sizes**, this will be a list of lists of torch.Tensors (T, C, H, W).
            **kwargs: Additional fields from Sample        
        Returns:
            RewardModelOutput with rewards shape (group_size,)
            Rewards must align with the order of input samples.

        NOTE: self.config.batch_size is ignored in this method, but it can be handled internally if needed.
        """
        pass

class GlobalwiseRewardModel(BaseRewardModel):
    """
        Reward model that computes rewards considering all samples globally - maybe can be merged with `advantage computation` stage.
        A placeholder for future extension.
    """
    pass