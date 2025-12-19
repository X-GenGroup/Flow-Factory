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

from ..utils.memory_tracker import MemoryProfiler

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

        self.memory_profiler = MemoryProfiler(self.accelerator, enable_tensor_accumulation=True, log_file='./memory_log.log')
        self.memory_profiler.register_model(self.adapter.transformer, 'adapter_transformer')
        self.memory_profiler.register_model(self.adapter.pipeline.text_encoder_2, 'adapter_text_encoder_2')
        self._initialization()
        self.memory_profiler.snapshot("after_init")

        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
        )

    @property
    def transformer(self) -> nn.Module:
        return self.adapter.transformer

    @property
    def unwrapped_transformer(self) -> BaseAdapter:
        return self.accelerator.unwrap_model(self.adapter.transformer)
    
    def _init_reward_model(self) -> BaseRewardModel:
        """Initialize reward model from configuration."""
        reward_model_cls = self.reward_args.reward_model_cls
        self.reward_model = reward_model_cls(self.reward_args)
        return self.reward_model

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        # Only the first process loads and preprocesses the dataset
        with self.accelerator.main_process_first():
            # Move text-encoder & vae to GPU for dataloader encoding
            if self.accelerator.is_local_main_process:
                self.adapter.on_load_text_encoder(self.accelerator.device)
            dataloader, test_dataloader = get_dataloader(
                data_args=self.data_args,
                training_args=self.training_args,
                text_encode_func=self.adapter.encode_prompts,
                image_encode_func=self.adapter.encode_images,
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
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _initialization(self):
        # Init dataloader and optimizer
        self.memory_profiler.snapshot("before_init")
        self.adapter.on_load(self.accelerator.device)
        self.memory_profiler.snapshot("before_dataloader_init")
        self.dataloader, self.test_dataloader = self._init_dataloader()
        self.memory_profiler.snapshot("before_optimizer_init")
        self.optimizer = self._init_optimizer()
        self.memory_profiler.track_optimizer(self.optimizer)
        self.memory_profiler.snapshot("after_optimizer_init")
        self.memory_profiler.snapshot("before_accelerator_prepare")
        # Prepare everything with accelerator
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        if self.test_dataloader is not None:
            self.adapter.transformer, self.optimizer, self.test_dataloader = self.accelerator.prepare(
                self.adapter.transformer,
                self.optimizer,
                self.test_dataloader,
            )
        else:
            self.adapter.transformer, self.optimizer = self.accelerator.prepare(
                self.adapter.transformer,
                self.optimizer,
            )
        self.memory_profiler.snapshot("after_accelerator_prepare")
        # Load Vae for image decoding
        self.adapter.on_load_vae(self.accelerator.device)
        self.memory_profiler.snapshot("after_vae_load")
        
        # Initialize reward model
        self._init_reward_model()
        self.memory_profiler.snapshot("after_reward_model_init")

    @abstractmethod
    def run(self):
        """Main training loop."""
        pass

    @abstractmethod
    def compute_loss(self):
        """Main training loop."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation loop."""
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