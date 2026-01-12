# src/flow_factory/rewards/abc.py
"""
Abstract Base Class for Reward Models
Provides common interface for all reward models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from functools import partial
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from PIL import Image
from contextlib import nullcontext

from accelerate import Accelerator
from diffusers.utils.outputs import BaseOutput
from ..hparams import *

@dataclass
class RewardModelOutput(BaseOutput):
    """
    Output class for Reward models.
    
    Args:
        rewards: Reward values (can be tensor, numpy array, or list)
        extra_info: Optional additional information
    """
    rewards: Union[torch.FloatTensor, np.ndarray, List[float]]
    extra_info: Optional[Dict[str, Any]] = None



class BaseRewardModel(ABC):
    """
    Abstract base class for reward models.
    
    Subclasses must implement the `forward` method. 
    """
    def __init__(self, config: RewardArguments, accelerator : Accelerator):
        """
        Args:
            config: Configuration object containing `reward_args`.
            accelerator: Accelerator instance for distributed setup.
        """
        super().__init__()
        self.accelerator = accelerator
        self.config = config
        self.device = self.accelerator.device if config.device == torch.device('cuda') else config.device
        self.dtype = config.dtype
        self.model = None  # To be defined in subclasses

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, RewardModelOutput]:
        """
        Implement the forward pass of the reward model.
        """
        pass