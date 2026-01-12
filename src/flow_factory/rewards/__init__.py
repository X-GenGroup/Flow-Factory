# src/flow_factory/rewards/__init__.py
"""
Reward models module for evaluating generated content.

Provides interfaces for single and multi-reward model loading and evaluation.
"""
from .abc import BaseRewardModel, RewardModelOutput
from .registry import get_reward_model_class, list_registered_reward_models
from .loader import load_reward_model, MultiRewardLoader, RewardModelHandle


__all__ = [
    # Base classes
    'BaseRewardModel',
    'RewardModelOutput',
    # Registry
    'get_reward_model_class',
    'list_registered_reward_models',
    # Loaders
    'load_reward_model',
    'MultiRewardLoader',
    'RewardModelHandle',
]