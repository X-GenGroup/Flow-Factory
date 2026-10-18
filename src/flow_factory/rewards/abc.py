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
"""Abstract base classes and the shared output type for reward models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers.utils.outputs import BaseOutput
from PIL import Image

from ..hparams import RewardArguments


@dataclass
class RewardModelOutput(BaseOutput):
    """Output for reward models."""

    rewards: Union[torch.Tensor, np.ndarray, List[float]]
    extra_info: Optional[Dict[str, Any]] = None


class BaseRewardModel(ABC):
    """Abstract base class providing the common interface for all reward models."""

    # Fields required from `BaseSample` for reward computation
    required_fields: Tuple[str, ...] = ()

    # Whether the model accepts tensor inputs directly
    use_tensor_inputs: bool = (
        False  # True: the input media are torch.Tensors; False: the input media are PIL Images
    )

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        self.accelerator = accelerator
        self.config = config
        self.device = accelerator.device if config.device == torch.device("cuda") else config.device
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
    """Reward model that computes independent rewards for each sample.

    Each sample's reward depends only on itself.

    Example:
        rewards = model(prompt=prompts, image=images)
    """

    required_fields: Tuple[str, ...] = ("image", "prompt")
    use_tensor_inputs: bool = (
        False  # True: the input media are torch.Tensors; False: the input media are PIL Images
    )

    @abstractmethod
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        audio: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute per-sample rewards.

        Args:
            prompt: List of text prompts (batch_size,)
            image: Generated PIL images. If `use_tensor_inputs` is True, a list of
                floating [0, 1] torch.Tensors (C, H, W).
            video: Generated videos, each a list of PIL frames. If `use_tensor_inputs` is
                True, a list of floating [0, 1] torch.Tensors (T, C, H, W).
            audio: Optional list of audio waveforms. Each element is a torch.Tensor
                of shape (C, T), float32 in [-1, 1]. If `use_tensor_inputs` is False,
                each element is an np.ndarray (C, T) instead.
            condition_images: Per-prompt lists of PIL condition images. If
                `use_tensor_inputs` is True, each entry is a list of floating [0, 1] image
                tensors or one stacked (num_conditions, C, H, W) tensor, depending on how
                the sample stores its condition images.
            condition_videos: Per-prompt lists of condition videos, each a list of PIL
                frames. If `use_tensor_inputs` is True, each entry is a list of floating
                [0, 1] video tensors or one stacked (num_conditions, T, C, H, W) tensor,
                depending on how the sample stores its condition videos.
            **kwargs: Additional fields from sample.extra_kwargs, passed automatically
                by RewardProcessor via filter_kwargs. Dataset metadata is delivered
                as a single ``metadata: str`` (JSON-encoded dict) containing all
                non-preprocess JSONL columns. Reward models that need metadata
                fields should accept a ``metadata`` parameter and call
                ``json.loads(metadata)`` to parse. See GenEvalRewardModel for a
                full example.

        Returns:
            RewardModelOutput with rewards shape (batch_size,)
        """
        pass


class GroupwiseRewardModel(BaseRewardModel):
    """Reward model that computes rewards considering the entire group.

    Used for pairwise preferences, ranking losses, or contrastive rewards.

    The model receives all samples belonging to the same sampler-assigned
    ``(source_id, unique_id)`` group.

    Example:
        # Called once per group with all samples in that group
        rewards = model(
            prompt=[sample.prompt for sample in group_samples],
            image=[sample.image for sample in group_samples],
            ...
        )
    """

    required_fields: Tuple[str, ...] = ("image", "prompt")
    use_tensor_inputs: bool = (
        False  # True: the input media are torch.Tensors; False: the input media are PIL Images
    )

    @abstractmethod
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        audio: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute group-aware rewards. Pairwise or ranking rewards can be computed here.

        Args:
            prompt: List of text prompts (group_size,)
            image: Generated PIL images. If `use_tensor_inputs` is True, a list of
                floating [0, 1] torch.Tensors (C, H, W).
            video: Generated videos, each a list of PIL frames. If `use_tensor_inputs` is
                True, a list of floating [0, 1] torch.Tensors (T, C, H, W).
            audio: Optional list of audio waveforms. Each element is a torch.Tensor
                of shape (C, T), float32 in [-1, 1]. If `use_tensor_inputs` is False,
                each element is an np.ndarray (C, T) instead.
            condition_images: Per-prompt lists of PIL condition images. If
                `use_tensor_inputs` is True, each entry is a list of floating [0, 1] image
                tensors or one stacked (num_conditions, C, H, W) tensor, depending on how
                the sample stores its condition images.
            condition_videos: Per-prompt lists of condition videos, each a list of PIL
                frames. If `use_tensor_inputs` is True, each entry is a list of floating
                [0, 1] video tensors or one stacked (num_conditions, T, C, H, W) tensor,
                depending on how the sample stores its condition videos.
            **kwargs: Additional fields from sample.extra_kwargs, passed automatically
                by RewardProcessor via filter_kwargs. Dataset metadata is delivered
                as a single ``metadata: str`` (JSON-encoded dict). Parse with
                ``json.loads(metadata)`` to access fields. See GenEvalRewardModel
                for a full example.

        Returns:
            RewardModelOutput with rewards shape (group_size,), aligned with the order of
            the input samples.

        Note:
            `self.config.batch_size` is not applied to groupwise calls; batch internally
            if needed.
        """
        pass


class GlobalwiseRewardModel(BaseRewardModel):
    """Placeholder for reward models that score all samples globally.

    `RewardProcessor` does not dispatch this type.
    """

    pass
