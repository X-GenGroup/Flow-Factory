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

# src/flow_factory/rewards/loader.py
"""
Reward Model Loader

Factory functions using registry pattern for extensibility.
Supports both single reward and multi-reward loading with automatic deduplication.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
import logging

from accelerate import Accelerator

from .abc import BaseRewardModel
from .registry import get_reward_model_class, list_registered_reward_models
from ..hparams import RewardArguments, MultiRewardArguments

logger = logging.getLogger(__name__)


# =============================================================================
# Single Reward Loader
# =============================================================================

def load_reward_model(
    config: RewardArguments,
    accelerator: Accelerator,
) -> BaseRewardModel:
    """
    Load and initialize the appropriate reward model based on configuration.
    
    Args:
        config: Reward model configuration arguments.
        accelerator: Accelerator instance for distributed setup.
    
    Returns:
        Reward model instance.
    
    Raises:
        ImportError: If the reward model is not registered or cannot be imported.
    """
    reward_model_identifier = config.reward_model
    
    try:
        reward_model_class = get_reward_model_class(reward_model_identifier)
        reward_model = reward_model_class(config=config, accelerator=accelerator)
        return reward_model
        
    except ImportError as e:
        registered_models = list(list_registered_reward_models().keys())
        raise ImportError(
            f"Failed to load reward model '{reward_model_identifier}'. "
            f"Available models: {registered_models}"
        ) from e


# =============================================================================
# Multi-Reward Loader with Deduplication
# =============================================================================

@dataclass
class RewardModelHandle:
    """
    Handle to a loaded reward model with associated metadata.
    
    Attributes:
        model: The loaded reward model instance.
        config: The original RewardArguments used to load this model.
        names: List of all reward configuration names that share this model.
    """
    model: BaseRewardModel
    config: RewardArguments
    names: List[str] = field(default_factory=list)


class MultiRewardLoader:
    """
    Load multiple reward models with automatic deduplication.
    
    Holds both training and evaluation reward configurations, performs
    deduplication analysis before loading, and provides separate interfaces
    for accessing training and evaluation models.
    
    Examples:
        >>> loader = MultiRewardLoader(
        ...     reward_args=config.reward_args,
        ...     eval_reward_args=config.eval_reward_args,
        ...     accelerator=accelerator,
        ... ).load()
        >>> 
        >>> train_reward_models = loader.get_training_models()
        >>> eval_reward_models = loader.get_eval_models()
    """
    
    def __init__(
        self,
        reward_args: MultiRewardArguments,
        accelerator: Accelerator,
        eval_reward_args: Optional[MultiRewardArguments] = None,
    ):
        """
        Initialize the MultiRewardLoader.
        
        Args:
            reward_args: Training reward configurations.
            accelerator: Accelerator instance for distributed setup.
            eval_reward_args: Evaluation reward configurations.
                If None, training rewards are used for evaluation.
        """
        self.reward_args = reward_args
        self.eval_reward_args = eval_reward_args
        self.accelerator = accelerator
        
        # Internal state
        self._cache: Dict[tuple, RewardModelHandle] = {}
        self._training_name_to_key: Dict[str, tuple] = {}
        self._eval_name_to_key: Dict[str, tuple] = {}
        self._loaded = False
    
    def load(self) -> MultiRewardLoader:
        """
        Load all reward models with deduplication.
        
        Analyzes both training and eval configurations, identifies unique
        models, loads each unique model once, and maps names to models.
        
        Returns:
            Self for method chaining.
        
        Raises:
            ImportError: If any reward model fails to load.
        """
        if self._loaded:
            logger.warning("Models already loaded, skipping reload")
            return self
        
        # Collect all unique configs
        all_configs: List[tuple[RewardArguments, str, Dict[str, tuple]]] = []
        
        # Add training rewards
        for config in self.reward_args.reward_configs:
            all_configs.append((config, 'train', self._training_name_to_key))
        
        # Add eval rewards
        if self.eval_reward_args:
            for config in self.eval_reward_args.reward_configs:
                all_configs.append((config, 'eval', self._eval_name_to_key))
        
        # Load with deduplication
        for config, source, name_to_key in all_configs:
            if config.reward_model is None:
                logger.warning(f"Skipping reward '{config.name}': no reward_model specified")
                continue
            
            identity_key = config.get_identity_key()
            
            if identity_key in self._cache:
                # Reuse existing model
                handle = self._cache[identity_key]
                handle.names.append(f"{config.name}({source})")
                # logger.info(f"Reusing '{handle.config.name}' for '{config.name}' ({source})")
            else:
                # Load new model
                model = load_reward_model(config, self.accelerator)
                handle = RewardModelHandle(
                    model=model, 
                    config=config, 
                    names=[f"{config.name}({source})"]
                )
                self._cache[identity_key] = handle
                # logger.info(f"Loaded reward model: {config.name} ({config.reward_model}) for {source}")
            
            name_to_key[config.name] = identity_key
        
        # If no eval rewards specified, use training rewards for eval
        if not self.eval_reward_args or len(self.eval_reward_args) == 0:
            self._eval_name_to_key = self._training_name_to_key.copy()
        
        self._loaded = True
        # logger.info(self.summary())
        return self
    
    def get_rewards_models(self, split : Literal['train', 'eval']) -> Dict[str, BaseRewardModel]:
        """
        Get reward models for the specified split.
        
        Args:
            split: 'train' or 'eval' to specify which group.
        """
        assert split in ['train', 'eval'], "Reward model split must be 'train' or 'eval'"
        self._ensure_loaded()
        name_to_key = self._training_name_to_key if split == 'train' else self._eval_name_to_key
        return {
            name: self._cache[key].model 
            for name, key in name_to_key.items()
        }
        
    def get_training_reward_models(self) -> Dict[str, BaseRewardModel]:
        """Get training reward models."""
        self._ensure_loaded()
        return {
            name: self._cache[key].model 
            for name, key in self._training_name_to_key.items()
        }
    
    def get_eval_reward_models(self) -> Dict[str, BaseRewardModel]:
        """Get evaluation reward models."""
        self._ensure_loaded()
        return {
            name: self._cache[key].model 
            for name, key in self._eval_name_to_key.items()
        }
    
    def get(self, name: str, source: str = 'train') -> Optional[BaseRewardModel]:
        """
        Get a loaded reward model by name.
        
        Args:
            name: The name of the reward configuration.
            source: 'train' or 'eval' to specify which group.
        
        Returns:
            The loaded model instance, or None if not found.
        """
        self._ensure_loaded()
        name_to_key = self._training_name_to_key if source == 'train' else self._eval_name_to_key
        key = name_to_key.get(name)
        return self._cache[key].model if key else None
    
    def get_args(self, name: str, source: str = 'train') -> Optional[RewardArguments]:
        """
        Get the RewardArguments for a loaded model by name.
        
        Args:
            name: The name of the reward configuration.
            source: 'train' or 'eval' to specify which group.
        
        Returns:
            The RewardArguments instance, or None if not found.
        """
        self._ensure_loaded()
        name_to_key = self._training_name_to_key if source == 'train' else self._eval_name_to_key
        key = name_to_key.get(name)
        return self._cache[key].config if key else None
    
    def get_unique_model_count(self) -> int:
        """Get the number of unique loaded models."""
        return len(self._cache)
    
    def get_total_config_count(self) -> int:
        """Get the total number of reward configurations."""
        return len(self._training_name_to_key) + len(self._eval_name_to_key)
    
    def summary(self) -> str:
        """
        Get a summary string of the loading results.
        
        Returns:
            Human-readable summary of loaded models and deduplication savings.
        """
        unique = len(self._cache)
        train_count = len(self._training_name_to_key)
        eval_count = len(self._eval_name_to_key)
        total = train_count + eval_count
        saved = total - unique
        
        parts = [f"Loaded {unique} unique models"]
        parts.append(f"(train: {train_count}, eval: {eval_count})")
        if saved > 0:
            parts.append(f"(saved {saved} duplicates)")
        
        return " ".join(parts)
    
    def _ensure_loaded(self):
        """Ensure models have been loaded."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
    
    def clear(self):
        """Clear all loaded models from cache."""
        self._cache.clear()
        self._training_name_to_key.clear()
        self._eval_name_to_key.clear()
        self._loaded = False