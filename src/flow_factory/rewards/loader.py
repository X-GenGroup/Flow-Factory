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
"""Reward model loader.

Factory functions using registry pattern for extensibility.
Supports both single reward and multi-reward loading with automatic deduplication.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, ContextManager, Dict, List, Literal, Optional, Tuple

from accelerate import Accelerator

from ..hparams import MultiRewardArguments, RewardArguments
from .abc import BaseRewardModel
from .registry import get_reward_model_class, list_registered_reward_models

logger = logging.getLogger(__name__)


# =============================================================================
# Single Reward Loader
# =============================================================================


def load_reward_model(
    config: RewardArguments,
    accelerator: Accelerator,
) -> BaseRewardModel:
    """Load and initialize the appropriate reward model based on configuration.

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
    """Handle to a loaded reward model with associated metadata.

    Attributes:
        model: The loaded reward model instance.
        config: The original RewardArguments used to load this model.
        names: List of all reward configuration names that share this model.
    """

    model: BaseRewardModel
    config: RewardArguments
    names: List[str] = field(default_factory=list)


class MultiRewardLoader:
    """Load multiple reward models with automatic deduplication.

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
        >>> train_reward_models = loader.get_training_reward_models()
        >>> eval_reward_models = loader.get_eval_reward_models()
    """

    def __init__(
        self,
        reward_args: MultiRewardArguments,
        accelerator: Accelerator,
        training_dataset_names: Optional[List[str]] = None,
        eval_reward_args: Optional[MultiRewardArguments] = None,
        eval_dataset_names: Optional[List[str]] = None,
        load_context: Optional[Callable[[], ContextManager[None]]] = None,
    ):
        """Initialize the MultiRewardLoader.

        Args:
            reward_args: Training reward configurations.
            accelerator: Accelerator instance for distributed setup.
            training_dataset_names: Names of training datasets for per-training-
                dataset reward routing. When provided, builds per-dataset
                mappings using the ``applicable_datasets`` field in each training
                reward config. When None, no per-training-dataset routing is built
                (legacy single-source mode — every reward applies to every
                training sample).
            eval_reward_args: Evaluation reward configurations.
                If None, training rewards are used for evaluation.
            eval_dataset_names: Names of eval datasets for per-eval-dataset
                reward routing. When provided, builds per-dataset mappings
                using the ``applicable_datasets`` field in each eval reward config.
            load_context: Backend-owned context entered around each unique
                reward model construction.
        """
        self.reward_args = reward_args
        self.eval_reward_args = eval_reward_args
        self.accelerator = accelerator
        self._training_dataset_names = training_dataset_names or []
        self._eval_dataset_names = eval_dataset_names or []
        self._load_context = load_context or nullcontext

        # Internal state
        self._cache: Dict[tuple, RewardModelHandle] = {}
        self._training_name_to_key: Dict[str, tuple] = {}
        self._eval_name_to_key: Dict[str, tuple] = {}
        self._training_name_to_config: Dict[str, RewardArguments] = {}
        self._eval_name_to_config: Dict[str, RewardArguments] = {}
        # Per-training-dataset mappings: dataset_name → {reward_name → identity_key}
        self._training_dataset_reward_keys: Dict[str, Dict[str, tuple]] = {}
        self._training_dataset_reward_configs: Dict[str, Dict[str, RewardArguments]] = {}
        # Per-eval-dataset mappings: dataset_name → {reward_name → identity_key}
        self._eval_dataset_reward_keys: Dict[str, Dict[str, tuple]] = {}
        self._eval_dataset_reward_configs: Dict[str, Dict[str, RewardArguments]] = {}
        self._loaded = False

    def load(self) -> MultiRewardLoader:
        """Load all reward models with deduplication.

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
            all_configs.append((config, "train", self._training_name_to_key))

        # Add eval rewards
        if self.eval_reward_args:
            for config in self.eval_reward_args.reward_configs:
                all_configs.append((config, "eval", self._eval_name_to_key))

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
            else:
                # Load new model
                with self._load_context():
                    model = load_reward_model(config, self.accelerator)
                handle = RewardModelHandle(
                    model=model, config=config, names=[f"{config.name}({source})"]
                )
                self._cache[identity_key] = handle

            name_to_key[config.name] = identity_key
            if source == "train":
                self._training_name_to_config[config.name] = config
            else:
                self._eval_name_to_config[config.name] = config

        # If no eval rewards specified, use training rewards for eval
        if not self.eval_reward_args or len(self.eval_reward_args) == 0:
            self._eval_name_to_key = self._training_name_to_key.copy()
            self._eval_name_to_config = self._training_name_to_config.copy()

        # Build per-training-dataset reward mappings
        self._training_dataset_reward_keys, self._training_dataset_reward_configs = (
            self._build_dataset_reward_mappings(
                self._training_dataset_names,
                self._training_name_to_key,
                self._training_name_to_config,
            )
        )
        # Build per-eval-dataset reward mappings
        self._eval_dataset_reward_keys, self._eval_dataset_reward_configs = (
            self._build_dataset_reward_mappings(
                self._eval_dataset_names,
                self._eval_name_to_key,
                self._eval_name_to_config,
            )
        )

        self._loaded = True
        return self

    @staticmethod
    def _build_dataset_reward_mappings(
        dataset_names: List[str],
        name_to_key: Dict[str, tuple],
        name_to_config: Dict[str, "RewardArguments"],
    ) -> Tuple[Dict[str, Dict[str, tuple]], Dict[str, Dict[str, "RewardArguments"]]]:
        """Build per-dataset reward routing for a set of dataset names.

        For each dataset name, determines which rewards apply to it
        (using each ``RewardArguments.applicable_datasets`` field).  ``None`` is
        accepted defensively as "applies to every source" for callers
        that construct a ``MultiRewardLoader`` outside the ``Arguments``
        flow.

        Returns:
            Tuple of (keys_mapping, configs_mapping) where each maps
            dataset_name -> {reward_name -> identity_key/config}.
        """
        keys_mapping: Dict[str, Dict[str, tuple]] = {}
        configs_mapping: Dict[str, Dict[str, "RewardArguments"]] = {}

        for dataset_name in dataset_names:
            ds_keys: Dict[str, tuple] = {}
            ds_configs: Dict[str, "RewardArguments"] = {}

            for reward_name, identity_key in name_to_key.items():
                reward_cfg = name_to_config[reward_name]
                applies = (
                    reward_cfg.applicable_datasets is None
                    or dataset_name in reward_cfg.applicable_datasets
                )
                if applies:
                    ds_keys[reward_name] = identity_key
                    ds_configs[reward_name] = reward_cfg

            keys_mapping[dataset_name] = ds_keys
            configs_mapping[dataset_name] = ds_configs

        return keys_mapping, configs_mapping

    def get_rewards_models(self, split: Literal["train", "eval"]) -> Dict[str, BaseRewardModel]:
        """Get reward models for the specified split.

        Args:
            split: 'train' or 'eval' to specify which group.
        """
        assert split in ["train", "eval"], "Reward model split must be 'train' or 'eval'"
        self._ensure_loaded()
        name_to_key = self._training_name_to_key if split == "train" else self._eval_name_to_key
        return {name: self._cache[key].model for name, key in name_to_key.items()}

    def get_reward_configs(self, split: Literal["train", "eval"]) -> Dict[str, RewardArguments]:
        """Get reward argument configs for the specified split.

        Args:
            split: 'train' or 'eval' to specify which group.
        """
        assert split in ["train", "eval"], "Reward config split must be 'train' or 'eval'"
        self._ensure_loaded()
        name_to_config = (
            self._training_name_to_config if split == "train" else self._eval_name_to_config
        )
        return name_to_config.copy()

    def get_training_reward_models(self) -> Dict[str, BaseRewardModel]:
        """Get training reward models."""
        self._ensure_loaded()
        return {name: self._cache[key].model for name, key in self._training_name_to_key.items()}

    def get_eval_reward_models(self) -> Dict[str, BaseRewardModel]:
        """Get evaluation reward models."""
        self._ensure_loaded()
        return {name: self._cache[key].model for name, key in self._eval_name_to_key.items()}

    def get_training_dataset_reward_models(self, dataset_name: str) -> Dict[str, BaseRewardModel]:
        """Get TRAINING reward models applicable to a specific training dataset.

        Args:
            dataset_name: Name of the training dataset (as declared in
                ``data.datasets[*].name``).

        Returns:
            Dict mapping reward name → model instance for rewards that
            apply to the given training source.

        Raises:
            KeyError: If ``dataset_name`` was not registered during loading
                (i.e., it isn't in ``training_dataset_names`` passed to
                ``__init__``).
        """
        self._ensure_loaded()
        if dataset_name not in self._training_dataset_reward_keys:
            raise KeyError(
                f"Unknown training dataset '{dataset_name}'. "
                f"Known datasets: {sorted(self._training_dataset_reward_keys.keys())}"
            )
        return {
            name: self._cache[key].model
            for name, key in self._training_dataset_reward_keys[dataset_name].items()
        }

    def get_training_dataset_reward_configs(
        self, dataset_name: str
    ) -> Dict[str, "RewardArguments"]:
        """Get TRAINING reward configs applicable to a specific training dataset.

        Mirror of :meth:`get_eval_dataset_reward_configs` for the
        training side.
        """
        self._ensure_loaded()
        if dataset_name not in self._training_dataset_reward_configs:
            raise KeyError(
                f"Unknown training dataset '{dataset_name}'. "
                f"Known datasets: {sorted(self._training_dataset_reward_configs.keys())}"
            )
        return self._training_dataset_reward_configs[dataset_name].copy()

    def get_eval_dataset_reward_models(self, dataset_name: str) -> Dict[str, BaseRewardModel]:
        """Get reward models applicable to a specific eval dataset.

        Args:
            dataset_name: Name of the eval dataset (as declared in eval_datasets config).

        Returns:
            Dict mapping reward name → model instance for rewards that apply
            to the given dataset.

        Raises:
            KeyError: If dataset_name was not registered during loading.
        """
        self._ensure_loaded()
        if dataset_name not in self._eval_dataset_reward_keys:
            raise KeyError(
                f"Unknown eval dataset '{dataset_name}'. "
                f"Known datasets: {sorted(self._eval_dataset_reward_keys.keys())}"
            )
        return {
            name: self._cache[key].model
            for name, key in self._eval_dataset_reward_keys[dataset_name].items()
        }

    def get_eval_dataset_reward_configs(self, dataset_name: str) -> Dict[str, "RewardArguments"]:
        """Get reward configs applicable to a specific eval dataset.

        Args:
            dataset_name: Name of the eval dataset.

        Returns:
            Dict mapping reward name → RewardArguments for rewards that apply
            to the given dataset.
        """
        self._ensure_loaded()
        if dataset_name not in self._eval_dataset_reward_configs:
            raise KeyError(
                f"Unknown eval dataset '{dataset_name}'. "
                f"Known datasets: {sorted(self._eval_dataset_reward_configs.keys())}"
            )
        return self._eval_dataset_reward_configs[dataset_name].copy()

    def get(self, name: str, source: str = "train") -> Optional[BaseRewardModel]:
        """Get a loaded reward model by name.

        Args:
            name: The name of the reward configuration.
            source: 'train' or 'eval' to specify which group.

        Returns:
            The loaded model instance, or None if not found.
        """
        self._ensure_loaded()
        name_to_key = self._training_name_to_key if source == "train" else self._eval_name_to_key
        key = name_to_key.get(name)
        return self._cache[key].model if key else None

    def get_args(self, name: str, source: str = "train") -> Optional[RewardArguments]:
        """Get the RewardArguments for a loaded model by name.

        Args:
            name: The name of the reward configuration.
            source: 'train' or 'eval' to specify which group.

        Returns:
            The RewardArguments instance, or None if not found.
        """
        self._ensure_loaded()
        name_to_key = self._training_name_to_key if source == "train" else self._eval_name_to_key
        key = name_to_key.get(name)
        return self._cache[key].config if key else None

    def get_unique_model_count(self) -> int:
        """Get the number of unique loaded models."""
        return len(self._cache)

    def get_total_config_count(self) -> int:
        """Get the total number of reward configurations."""
        return len(self._training_name_to_key) + len(self._eval_name_to_key)

    def summary(self) -> str:
        """Get a summary string of the loading results.

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
        self._training_name_to_config.clear()
        self._eval_name_to_config.clear()
        self._eval_dataset_reward_keys.clear()
        self._eval_dataset_reward_configs.clear()
        self._training_dataset_reward_keys.clear()
        self._training_dataset_reward_configs.clear()
        self._loaded = False
