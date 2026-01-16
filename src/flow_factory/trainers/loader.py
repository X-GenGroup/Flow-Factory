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

# src/flow_factory/trainers/loader.py
"""
Trainer loader factory for extensibility.
Supports multiple RL algorithms via registry pattern.
"""
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
import logging

from ..models.loader import load_model
from .trainer import BaseTrainer
from .registry import get_trainer_class, list_registered_trainers
from ..hparams import Arguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # Fix issue that Qwen-Image uses different cache context for CFG forwards

def load_trainer(config: Arguments) -> BaseTrainer:
    """
    Factory function to instantiate trainer based on algorithm type.
    
    Uses registry pattern for automatic trainer discovery and loading.
    Supports both built-in trainers and custom algorithms via python paths.
    
    Args:
        config: Configuration containing trainer_type and all hyperparameters
    
    Returns:
        An instance of a BaseTrainer subclass
    
    Raises:
        ImportError: If the trainer is not registered or cannot be imported
    
    Examples:
        # Using built-in trainer
        config.training_args.trainer_type = "grpo"
        trainer = load_trainer(config)
        
        # Using custom trainer
        config.training_args.trainer_type = "my_package.trainers.PPOTrainer"
        trainer = load_trainer(config)
    """
    # Initialize Accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.log_args.save_dir, config.run_name),
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(config.training_args.seed, device_specific=True)

    # Initialize model adapter
    adapter = load_model(config=config, accelerator=accelerator)

    # Get trainer class from registry
    trainer_type = config.training_args.trainer_type
    
    try:
        trainer_cls = get_trainer_class(trainer_type)
    except ImportError as e:
        registered_trainers = list(list_registered_trainers().keys())
        raise ImportError(
            f"Failed to load trainer '{trainer_type}'. "
            f"Available trainers: {registered_trainers}"
        ) from e
    
    return trainer_cls(
        config=config,
        accelerator=accelerator,
        adapter=adapter,
    )