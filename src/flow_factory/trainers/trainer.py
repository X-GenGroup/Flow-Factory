# src/flow_factory/models/trainer.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

from ..hparams import *
from ..models.adapter import BaseAdapter
from ..data_utils.loader import get_dataloader
from ..rewards.reward_model import BaseRewardModel

class BaseTrainer(ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """
    def __init__(
            self,
            accelerator: Accelerator,
            config : Arguments,
            adapter : BaseAdapter,
        ):
        self.accelerator = accelerator
        self.config = config
        self.data_args = config.data_args
        self.training_args = config.training_args
        self.reward_args = config.reward_args
        self.adapter = adapter
        self.epoch = 0
        self.step = 0

        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
        )

        self._initialization()
        self._init_logging_backend()

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()


    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)

    @property
    def transformer(self) -> nn.Module:
        return self.adapter.transformer

    @property
    def unwrapped_transformer(self) -> BaseAdapter:
        return self.accelerator.unwrap_model(self.adapter.transformer)
    
    def _init_logging_backend(self):
        if not self.accelerator.is_main_process:
            self.logger = None
            return
        """Initialize logging backend if specified."""
        if self.config.logging_backend == 'wandb':
            from ..logger import WandbLogger
            self.logger = WandbLogger(config=self.config)
        elif self.config.logging_backend == 'swanlab':
            from ..logger import SwanlabLogger
            self.logger = SwanlabLogger(config=self.config)
        else:
            self.logger = None

    def _init_reward_model(self) -> BaseRewardModel:
        """Initialize reward model from configuration."""
        reward_model_cls = self.reward_args.reward_model_cls
        self.reward_model = reward_model_cls(config=self.config, accelerator=self.accelerator)
        return self.reward_model

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        # Only the first process loads and preprocesses the dataset
        with self.accelerator.main_process_first():
            # Move text-encoder & vae to GPU for dataloader encoding
            if self.accelerator.is_local_main_process:
                self.adapter.on_load_text_encoder(self.accelerator.device)
            dataloader, test_dataloader = get_dataloader(
                self.config,
                text_encode_func=self.adapter.encode_prompt,
                image_encode_func=self.adapter.encode_image,
                video_encode_func=self.adapter.encode_video,
            )
            # Offload text-encoder after dataloader encoding
            if self.accelerator.is_local_main_process:
                self.adapter.off_load_text_encoder()

        return dataloader, test_dataloader
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _initialization(self):
        # Init dataloader and optimizer
        self.adapter.on_load(self.accelerator.device)
        self.dataloader, self.test_dataloader = self._init_dataloader()
        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        to_prepare = [self.adapter.transformer, self.optimizer, self.test_dataloader]
        to_prepare = [x for x in to_prepare if x is not None]
        prepared = self.accelerator.prepare(*to_prepare)
        self.adapter.transformer, self.optimizer = prepared[:2]
        if len(prepared) > 2:
            self.test_dataloader = prepared[2]

        # Load Vae for image decoding
        self.adapter.on_load_vae(self.accelerator.device)
        
        # Initialize reward model
        self._init_reward_model()

    @abstractmethod
    def run(self):
        """Main training loop."""
        pass

    @abstractmethod
    def compute_loss(self):
        """Compute loss for a training step."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation for one epoch."""
        pass

    def save_checkpoint(self, path: str):
        """Save trainer state to a specific path."""
        if self.accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            self.adapter.save_checkpoint(path, transformer_override=self.unwrapped_transformer)

        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: str):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(path)
        self.accelerator.wait_for_everyone()