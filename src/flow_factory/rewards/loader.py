# src/flow_factory/rewards/loader.py
"""
Reward Model Loader
Factory function using registry pattern for extensibility.
"""
from typing import Optional
from accelerate import Accelerator

from .reward_model import BaseRewardModel
from .registry import get_reward_model_class, list_registered_reward_models
from ..hparams import Arguments


def load_reward_model(
    config: Arguments,
    accelerator: Accelerator,
) -> BaseRewardModel:
    """
    Load and initialize the appropriate reward model based on configuration.
    
    Uses registry pattern for automatic reward model discovery and loading.
    Supports both built-in models and custom backends via python paths.
    
    Args:
        config: Configuration containing reward_model_cls identifier
        accelerator: Accelerator instance for distributed setup
    
    Returns:
        Reward model instance
    
    Raises:
        ImportError: If the reward model is not registered or cannot be imported
    
    Examples:
        # Using built-in reward model
        config.reward_args.reward_model_cls = "PickScore"
        reward_model = load_reward_model(config, accelerator)
        
        # Using custom reward model
        config.reward_args.reward_model_cls = "my_package.rewards.ImageReward"
        reward_model = load_reward_model(config, accelerator)
    """
    reward_model_identifier = config.reward_args.reward_model
    
    try:
        # Get reward model class from registry or direct import
        reward_model_class = get_reward_model_class(reward_model_identifier)
        
        # Instantiate reward model
        reward_model = reward_model_class(config=config, accelerator=accelerator)
        
        return reward_model
        
    except ImportError as e:
        registered_models = list(list_registered_reward_models().keys())
        raise ImportError(
            f"Failed to load reward model '{reward_model_identifier}'. "
            f"Available models: {registered_models}"
        ) from e