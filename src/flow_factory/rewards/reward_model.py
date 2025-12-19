# src/flow_factory/rewards/reward_model.py
"""
Base class for reward models.
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
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

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



class BaseRewardModel(ABC, nn.Module):
    """
    Abstract base class for reward models.
    
    Subclasses must implement the `forward` method. 
    This class handles DeepSpeed ZeRO-3 parameter gathering and no_grad contexts automatically.
    """
    model : nn.Module
    def __init__(self, config: Arguments, accelerator : Accelerator):
        """
        Args:
            config: Configuration object containing `reward_args`.
            accelerator: Accelerator instance for distributed setup.
        """
        super().__init__()
        self.accelerator = accelerator
        reward_args = config.reward_args
        self.reward_args = reward_args
        self.device = reward_args.device
        self.dtype = reward_args.dtype

    def __call__(self, *args, **kwargs):
        """
        Wraps `forward` to automatically handle `torch.no_grad` and DeepSpeed ZeRO-3 parameter gathering.
        """
        # if self.accelerator.state.deepspeed_plugin and self.accelerator.state.deepspeed_plugin.zero_stage == 3:
        #     # Bind self.model dynamically
        #     stage3_context = lambda: GatheredParameters(self.model.parameters(), modifier_rank=0)
        # else:
        #     stage3_context = nullcontext

        stage3_context = nullcontext()
        with torch.no_grad(), stage3_context:
            return super().__call__(*args, **kwargs)

    
    @abstractmethod
    def forward(self, **kwargs) -> Union[RewardModelOutput, torch.Tensor, np.ndarray, List[float]]:
        """
        Compute rewards for the given inputs.
        
        Args:
            **inputs: Model-specific inputs (e.g., pixel_values, input_ids).
            
        Returns:
            The computed rewards in the specified format.
        """
        pass