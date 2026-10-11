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

# src/flow_factory/trainers/registry.py
"""Centralized trainer registry with dynamic class loading."""

import importlib
import logging
from typing import Dict, Type

from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# Trainer Registry Storage
_TRAINER_REGISTRY: Dict[str, str] = {
    "grpo": "flow_factory.trainers.rl.grpo.GRPOTrainer",
    "grpo-guard": "flow_factory.trainers.rl.grpo.GRPOGuardTrainer",
    "dppo": "flow_factory.trainers.rl.dppo.DPPOTrainer",
    "nft": "flow_factory.trainers.rl.nft.DiffusionNFTTrainer",
    "awm": "flow_factory.trainers.rl.awm.AWMTrainer",
    "dgpo": "flow_factory.trainers.rl.dgpo.DGPOTrainer",
    "dpo": "flow_factory.trainers.rl.dpo.DPOTrainer",
    "crd": "flow_factory.trainers.rl.crd.CRDTrainer",
    "diffusion-opd": "flow_factory.trainers.distillation.opd.trainer.DiffusionOPDTrainer",
    "dmd2": "flow_factory.trainers.distillation.dmd2.DMD2Trainer",
    "tdm": "flow_factory.trainers.distillation.tdm.TDMTrainer",
    "tdm-r1": "flow_factory.trainers.distillation.tdm_r1.TDMR1Trainer",
    "sft": "flow_factory.trainers.offline.sft.SFTTrainer",
    "offline-dpo": "flow_factory.trainers.offline.offline_dpo.OfflineDPOTrainer",
}


def register_trainer(name: str):
    """Return a decorator that registers a trainer class under ``name``.

    Example:
        @register_trainer('grpo')
        class GRPOTrainer(BaseTrainer):
            ...

    Args:
        name: Trainer algorithm identifier (e.g., 'grpo', 'ppo', 'dpo').

    Returns:
        Decorator function that registers the class.
    """

    def decorator(cls):
        _TRAINER_REGISTRY[name] = f"{cls.__module__}.{cls.__name__}"
        logger.info(f"Registered trainer: {name} -> {cls.__name__}")
        return cls

    return decorator


def get_trainer_class(identifier: str) -> Type:
    """Resolve and import a trainer class from the registry or a Python path.

    Supports two modes:
    1. Registry lookup: 'grpo' -> GRPOTrainer
    2. Direct import: 'my_package.trainers.CustomTrainer' -> CustomTrainer

    Args:
        identifier: Trainer algorithm name or fully qualified class path.

    Returns:
        Trainer class.

    Raises:
        ImportError: If the trainer cannot be loaded.

    Example:
        >>> cls = get_trainer_class('grpo')
        >>> trainer = cls(config=config, accelerator=accelerator, adapter=adapter)

        >>> cls = get_trainer_class('my_lib.trainers.PPOTrainer')
        >>> trainer = cls(config=config, accelerator=accelerator, adapter=adapter)
    """
    identifier_lower = identifier.lower()

    # Check registry first
    if identifier_lower in _TRAINER_REGISTRY:
        class_path = _TRAINER_REGISTRY[identifier_lower]
    else:
        # Assume it's a direct python path
        class_path = identifier

    # Dynamic import
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)

        logger.debug(f"Loaded trainer: {identifier} -> {class_name}")
        return trainer_class

    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not load trainer '{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered trainer: {list(_TRAINER_REGISTRY.keys())}\n"
            f"  2. A valid python path (e.g., 'my_package.trainers.CustomTrainer')\n"
            f"Error: {e}"
        ) from e


def list_registered_trainers() -> Dict[str, str]:
    """Return a copy of the trainer registry.

    Returns:
        Dictionary mapping trainer names to their class paths.
    """
    return _TRAINER_REGISTRY.copy()
