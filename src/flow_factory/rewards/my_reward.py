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
from contextlib import nullcontext
from typing import List, Optional, Union

import torch
from accelerate import Accelerator
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..hparams import *
from .abc import GroupwiseRewardModel, PointwiseRewardModel, RewardModelOutput


class MyPointwiseRewardModel(PointwiseRewardModel):
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # `super().__init__` sets `self.accelerator`, `self.config`, `self.dtype`
        # (`config.dtype`), `self.device` (the accelerator device when `config.device`
        # is cuda, else `config.device`), and `self.model = None`.

        # Implement your custom reward model initialization here
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """Compute rewards for given prompts and images.

        Args:
            prompt (list[str]): List of text prompts.
            image (list[Image.Image]): List of generated images corresponding to the prompts.
            video (list[list[Image.Image]]): List of generated videos (each video is a list of
                frames) corresponding to the prompts.
            condition_images (Optional[List[List[Image.Image] | torch.Tensor]]): Optional list
                of condition images. Each element is either
                - a list of images (a single-element list when a prompt has one condition
                  image), or
                - a tensor with batch dimension, scaled in [0, 1].
            condition_videos (Optional[List[List[List[Image.Image]]] | torch.Tensor]): Optional
                list of condition videos. Each element is either
                - a list of videos, where each video is a list of frames (a single-element
                  list when a prompt has one condition video), or
                - a tensor with batch dimension, scaled in [0, 1].

        Note:
            Dataset metadata is available via ``metadata: Optional[List[str]] = None``
            parameter. Each element is a JSON string containing all non-preprocess
            JSONL columns for that sample. Parse with ``json.loads(metadata[i])``
            to access fields like ``include``, ``tag``, etc. See GenEvalRewardModel
            for a full example.

        Returns:
            RewardModelOutput: Contains rewards tensor and any extra information.
        """

        # Inputs are lists of at most `self.config.batch_size` samples
        # Implement your custom reward computation here
        rewards = torch.zeros(len(prompt), device=self.device)

        # Wrap rewards in RewardModelOutput
        return RewardModelOutput(
            rewards=rewards,
            extra_info={},  # Add any extra info if needed
        )


class MyGroupwiseRewardModel(GroupwiseRewardModel):
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # `super().__init__` sets `self.accelerator`, `self.config`, `self.dtype`
        # (`config.dtype`), `self.device` (the accelerator device when `config.device`
        # is cuda, else `config.device`), and `self.model = None`.

        # Implement your custom reward model initialization here
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """Compute rewards for given prompts and images within a group.

        The whole group arrives in one call; handle `self.config.batch_size` yourself here.

        Args:
            prompt (list[str]): List of text prompts. The length is equal to `group_size`.
            image (list[Image.Image]): List of generated images corresponding to the prompts.
                The length is equal to `group_size`.
            video (list[list[Image.Image]]): List of generated videos (each video is a list of
                frames) corresponding to the prompts. The length is equal to `group_size`.
            condition_images (Optional[List[List[Image.Image] | torch.Tensor]]): Optional list
                of condition images. The length is equal to `group_size`. Each element is
                either
                - a list of images (a single-element list when a prompt has one condition
                  image), or
                - a tensor with batch dimension, scaled in [0, 1].
            condition_videos (Optional[List[List[List[Image.Image]]] | torch.Tensor]): Optional
                list of condition videos. The length is equal to `group_size`. Each element is
                either
                - a list of videos, where each video is a list of frames (a single-element
                  list when a prompt has one condition video), or
                - a tensor with batch dimension, scaled in [0, 1].

        Note:
            Dataset metadata is available via ``metadata: Optional[List[str]] = None``
            parameter. Each element is a JSON string containing all non-preprocess
            JSONL columns for that sample. Parse with ``json.loads(metadata[i])``
            to access fields. See GenEvalRewardModel for a full example.

        Returns:
            RewardModelOutput: Contains rewards tensor and any extra information.
        """

        # Ensure inputs are lists, each of length `group_size`
        # Implement your custom reward computation here.
        rewards = torch.arange(
            len(prompt)
        )  # A trivial reward assignment (0, 1, 2, ..., group_size - 1)

        # Wrap rewards in RewardModelOutput; the order of `rewards` must match the input prompts
        return RewardModelOutput(
            rewards=rewards,
            extra_info={},  # Add any extra info if needed
        )
