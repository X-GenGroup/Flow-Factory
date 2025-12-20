# src/flow_factory/trainers/loader.py
"""
Trainer loader factory for extensibility.
Supports multiple RL algorithms and can be easily extended.
"""
import os
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from typing import Literal
import logging

from ..models.loader import load_model
from .trainer import BaseTrainer
from .grpo_trainer import GRPOTrainer
from ..hparams import *


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


def load_trainer(
    config : Arguments,
) -> BaseTrainer:
    """
    Factory function to instantiate the correct trainer based on algorithm type.
    
    Args:
        trainer_type: Algorithm type (grpo, ppo, dpo, reinforce, etc.)
        data_args: Data configuration
        training_args: Training configuration
        reward_args: Reward model configuration
        adapter: Model adapter instance
    
    Returns:
        An instance of a subclass of BaseTrainer
    """
    # Initialize Accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.training_args.save_dir, config.run_name),
        automatic_checkpoint_naming=True,
    )
    accelerator = Accelerator(
        mixed_precision=config.training_args.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.training_args.gradient_accumulation_steps,
    )
    set_seed(config.training_args.seed, device_specific=True)

    # Initialize model adapter    
    adapter = load_model(config=config)

    # Initialize trainer
    trainer_type = config.training_args.trainer_type.lower()
    trainer_mapping = {
        "grpo": GRPOTrainer,
    }
    
    if trainer_type not in trainer_mapping:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Supported: {list(trainer_mapping.keys())}"
        )
    
    trainer_cls = trainer_mapping[trainer_type]
    
    return trainer_cls(
        config=config,
        accelerator=accelerator,
        adapter=adapter,
    )