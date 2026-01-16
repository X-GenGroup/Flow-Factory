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

# src/flow_factory/rewards/registry.py
"""
Reward Model Registry System
Centralized registry for reward models with dynamic loading.
"""
from typing import Type, Dict
import importlib
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# Reward Model Registry Storage
_REWARD_MODEL_REGISTRY: Dict[str, str] = {
    'pickscore': 'flow_factory.rewards.pick_score.PickScoreRewardModel',
    'clip': 'flow_factory.rewards.clip.CLIPRewardModel',
    'consistent_editing': 'flow_factory.rewards.consistent_editing.ConsistentEditingRewardModel'
}
_REWARD_MODEL_REGISTRY = {k.lower(): v for k, v in _REWARD_MODEL_REGISTRY.items()}


def register_reward_model(name: str):
    """
    Decorator for registering reward models.
    
    Usage:
        @register_reward_model('PickScore')
        class PickScoreRewardModel(BaseRewardModel):
            ...
    
    Args:
        name: Reward model identifier (e.g., 'PickScore', 'ImageReward')
    
    Returns:
        Decorator function that registers the class
    """
    def decorator(cls):
        _REWARD_MODEL_REGISTRY[name] = f"{cls.__module__}.{cls.__name__}"
        logger.info(f"Registered reward model: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_reward_model_class(identifier: str) -> Type:
    """
    Resolve and import a reward model class from registry or python path.
    
    Supports two modes:
    1. Registry lookup: 'PickScore' -> PickScoreRewardModel
    2. Direct import: 'my_package.rewards.CustomReward' -> CustomReward
    
    Args:
        identifier: Reward model name or fully qualified class path
    
    Returns:
        Reward model class
    
    Raises:
        ImportError: If the reward model cannot be loaded
    
    Examples:
        >>> cls = get_reward_model_class('PickScore')
        >>> reward_model = cls(config, accelerator)
        
        >>> cls = get_reward_model_class('my_lib.rewards.ImageReward')
        >>> reward_model = cls(config, accelerator)
    """
    # Check registry first (case-insensitive for convenience)
    identifier_lower = identifier.lower()
    if identifier_lower in _REWARD_MODEL_REGISTRY:
        class_path = _REWARD_MODEL_REGISTRY[identifier_lower]
    else:
        # Assume it's a direct python path
        class_path = identifier
    
    # Dynamic import
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        reward_model_class = getattr(module, class_name)
        
        logger.debug(f"Loaded reward model: {identifier} -> {class_name}")
        return reward_model_class
        
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not load reward model '{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered reward model: {list(_REWARD_MODEL_REGISTRY.keys())}\n"
            f"  2. A valid python path (e.g., 'my_package.rewards.CustomReward')\n"
            f"Error: {e}"
        ) from e


def list_registered_reward_models() -> Dict[str, str]:
    """
    Get all registered reward models.
    
    Returns:
        Dictionary mapping reward model names to their class paths
    """
    return _REWARD_MODEL_REGISTRY.copy()