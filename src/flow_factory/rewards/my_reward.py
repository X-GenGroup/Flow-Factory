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

# src/flow_factory/rewards/my_reward.py
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
from typing import Optional, List, Union
from PIL import Image
from contextlib import nullcontext
import torch

from .abc import PointwiseRewardModel, GroupwiseRewardModel, RewardModelOutput
from ..hparams import *

class MyPointwiseRewardModel(PointwiseRewardModel):
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # `super().__init__` gives you:
        # self.accelerator = accelerator
        # self.config = config
        # self.device = self.accelerator.device if config.device == torch.device('cuda') else config.device
        # self.dtype = config.dtype

        # Implement your custom reward model initialization here
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt : List[str],
        image : Optional[List[Image.Image]] = None,
        video : Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """
        Compute rewards for given prompts and images.
        Args:
            prompt (list[str]): List of text prompts.
            image (list[Image.Image]): List of generated images corresponding to the prompts.
            video (list[list[Image.Image]]): List of generated videos (each video is a list of frames) corresponding to the prompts.
            condition_images (Optional[List[List[Image.Image] | torch.Tensor]]): Optional list of condition images
                - each element is a list of images. If only one condition image per prompt, this will be a list of single-element lists.
                - each element is a tensor with batch dimension, scaled in [0, 1].
            condition_videos (Optional[List[List[List[Image.Image]]] | torch.Tensor]): Optional list of condition videos
                - each element is a list of videos, where each video is a list of frames. If only one condition video per prompt, this will be a list of single-element lists.
                - each element is a tensor with batch dimension, scaled in [0, 1].
        Returns:
            RewardModelOutput: Contains rewards tensor and any extra information.
        """

        # Ensure inputs are lists, each of length `self.config.batch_size`
        # Implement your custom reward computation here
        rewards = torch.zeros(len(prompt), device=self.device)


        # Wrap rewards in RewardModelOutput
        return RewardModelOutput(
            rewards=rewards,
            extra_info={}, # Add any extra info if needed
        )


class MyGroupwiseRewardModel(PointwiseRewardModel):
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # `super().__init__` gives you:
        # self.accelerator = accelerator
        # self.config = config
        # self.device = self.accelerator.device if config.device == torch.device('cuda') else config.device
        # self.dtype = config.dtype

        # Implement your custom reward model initialization here
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt : List[str],
        image : Optional[List[Image.Image]] = None,
        video : Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """
        Compute rewards for given prompts and images.
        Args:
            prompt (list[str]): List of text prompts. The length is equal to `group_size`.
            image (list[Image.Image]): List of generated images corresponding to the prompts. The length is equal to `group_size`.
            video (list[list[Image.Image]]): List of generated videos (each video is a list of frames) corresponding to the prompts. The length is equal to `group_size`.
            condition_images (Optional[List[List[Image.Image] | torch.Tensor]]): Optional list of condition images. The length is equal to `group_size`.
                - each element is a list of images. If only one condition image per prompt, this will be a list of single-element lists.
                - each element is a tensor with batch dimension, scaled in [0, 1].
            condition_videos (Optional[List[List[List[Image.Image]]] | torch.Tensor]): Optional list of condition videos. The length is equal to `group_size`.
                - each element is a list of videos, where each video is a list of frames. If only one condition video per prompt, this will be a list of single-element lists.
                - each element is a tensor with batch dimension, scaled in [0, 1].
        Returns:
            RewardModelOutput: Contains rewards tensor and any extra information.
        """

        # Ensure inputs are lists, each of length `group_size`
        # Implement your custom reward computation here.
        rewards = torch.arange(len(prompt)) # A trivia reward assignment (0, 1, 2, .... group_size - 1)


        # Wrap rewards in RewardModelOutput, make sure the order of `rewards` align the original prompt
        return RewardModelOutput(
            rewards=rewards,
            extra_info={}, # Add any extra info if needed
        )