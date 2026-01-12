# src/flow_factory/rewards/my_reward.py
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
from typing import Optional, List, Union
from PIL import Image
from contextlib import nullcontext
import torch

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import *

class MyRewardModel(BaseRewardModel):
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
        **kwargs,
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
        Returns:
            RewardModelOutput: Contains rewards tensor and any extra information.
        """

        # Ensure inputs are lists, each of length reward_args.batch_size
        # Implement your custom reward computation here
        rewards = torch.zeros(len(prompt), device=self.device)


        # Wrap rewards in RewardModelOutput
        return RewardModelOutput(
            rewards=rewards,
            extra_info={}, # Add any extra info if needed
        )