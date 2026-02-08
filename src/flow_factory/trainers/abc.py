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

# src/flow_factory/trainers/abc.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataclasses import dataclass
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

from ..hparams import *
from ..models.abc import BaseAdapter
from ..data_utils.loader import get_dataloader
from ..rewards import load_reward_model, BaseRewardModel, MultiRewardLoader, RewardProcessor
from ..logger import load_logger, LogFormatter
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

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
        self.log_args = config.log_args
        self.model_args = config.model_args

        self.training_args = config.training_args
        self.eval_args = config.eval_args

        self.reward_args = config.reward_args
        self.eval_reward_args = config.eval_reward_args or config.reward_args # If `eval_reward_args` is not given, use `reward_args`

        self.adapter = adapter
        self.epoch = 0
        self.step = 0

        self._initialization()
        self.adapter.post_init()
        self._init_logging_backend()

        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.log_args.verbose and self.accelerator.is_local_main_process

    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)
        
        # Print summary to console
        if self.accelerator.is_local_main_process:
            metrics = {k: v for k, v in ((k, LogFormatter.to_scalar(v)) for k, v in data.items()) if v is not None}
            if metrics:
                parts = [f"[Step {step:04d} | Epoch {self.epoch:03d}]"]
                parts.extend(
                    f"{k}={int(v)}" if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                    else f"{k}={v:.4f}"
                    for k, v in metrics.items()
                )
                logger.info(" ".join(parts))
    
    def _init_logging_backend(self):
        if not self.accelerator.is_main_process:
            self.logger = None
            return
        """Initialize logging backend if specified."""
        self.logger = load_logger(self.config)

    def _init_reward_model(self) -> Tuple[Dict[str, BaseRewardModel], Dict[str, BaseRewardModel]]:
        """Initialize reward model from configuration."""

        # If DeepSpeed ZeRO-3 is enabled, the reward model will be somehow sharded.
        # We need to disable ZeRO-3 init context when loading the model to avoid issues
        # NOTE: This bug persists even with this context manager. DONOT USE ZeRO-3.
        # A possible solution: use DeepSpeed GatherParamter manually in the reward_model's `forward`.

        # Initialize all reward model instances
        self.reward_loader = MultiRewardLoader(
            reward_args=self.config.reward_args,
            accelerator=self.accelerator,
            eval_reward_args=self.config.eval_reward_args,
        ).load()
        # Get training & eval reward models
        self.reward_models = self.reward_loader.get_training_reward_models()
        self.eval_reward_models = self.reward_loader.get_eval_reward_models()
        # Initialize reward processor
        self.reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.reward_models,
            tokenizer=self.adapter.tokenizer, # For prompt encoding/decoding,
            verbose=self.log_args.verbose,
        )
        self.eval_reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.eval_reward_models,
            tokenizer=self.adapter.tokenizer, # For prompt encoding/decoding
            verbose=self.log_args.verbose,
        )
            
        return self.reward_models, self.eval_reward_models

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        # Move text-encoder & vae to GPU for dataloader encoding
        self.adapter.on_load_components(
            components=self.adapter.preprocessing_modules,
            device=self.accelerator.device
        )
        dataloader, test_dataloader = get_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )
        # Offload text-encoder after dataloader encoding
        self.adapter.off_load_components(
            components=self.adapter.preprocessing_modules,
        )

        self.accelerator.wait_for_everyone()

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

    def _load_inference_components(self, trainable_module_names: List[str]):
        """
        Load non-trainable components needed at runtime to the accelerator device.
        
        Trainable modules are already on-device via `accelerator.prepare()`.
        This loads the remaining modules required for inference and,
        when preprocessing is disabled, also loads encoding components
        that would otherwise stay offloaded.
        """
        prepared_names = set(trainable_module_names)
        
        modules_to_load = list(self.adapter.inference_modules)
        
        if not self.config.data_args.enable_preprocess:
            modules_to_load.extend(self.adapter.preprocessing_modules)
        
        # Resolve group names → concrete names, then deduplicate & exclude prepared
        resolved = self.adapter._resolve_component_names(modules_to_load)
        resolved = [m for m in resolved if m not in prepared_names]
        
        if resolved:
            self.adapter.on_load_components(
                components=resolved,
                device=self.accelerator.device,
            )

    def _initialization(self):
        # Fix for FSDP, synchronize frozen components like text encoder & VAE.
        # Otherwise they may be uninitialized on Rank > 0.
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            # self.adapter.on_load(self.accelerator.device)
            self._synchronize_frozen_components()

        # Init dataloader and optimizer
        self.dataloader, self.test_dataloader = self._init_dataloader()
        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        # Dynamically get all trainable modules from target_module_map
        trainable_module_names = list(self.adapter.target_module_map.keys())
        trainable_modules = [
            getattr(self.adapter, name) 
            for name in trainable_module_names 
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]
        # Prepare trainable modules + optimizer + test_dataloader
        to_prepare = trainable_modules + [self.optimizer]
        if self.test_dataloader is not None:
            to_prepare.append(self.test_dataloader)

        prepared = self.accelerator.prepare(*to_prepare)
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])

        self.optimizer = prepared[len(trainable_modules)]
        if self.test_dataloader is not None:
            self.test_dataloader = prepared[len(trainable_modules) + 1]

        # Load inference modules, excluding already-prepared ones
        self._load_inference_components(trainable_module_names)
        
        # Initialize reward model
        self._init_reward_model()

    def _synchronize_frozen_components(self):
        if self.accelerator.num_processes <= 1:
            return
        
        # Synchronize all non-prepared components
        all_names = self.adapter._resolve_component_names()
        for name in all_names:
            if self.adapter._should_manage_device(name):
                comp = self.adapter.get_component(name)
                if comp is not None:
                    for param in comp.parameters():
                        param.data = param.data.to(self.accelerator.device)
                        dist.broadcast(param.data, src=0)

        # Barrier to ensure everyone is done
        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {self.accelerator.process_index}] Frozen components synchronized.")

    @abstractmethod
    def start(self, *args, **kwargs):
        """Start training process."""
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Update policy model"""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation for one epoch."""
        pass

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save trainer state to a specific path."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        self.adapter.save_checkpoint(
            save_directory=save_directory,
            model_only=self.log_args.save_model_only,
        )

        self.accelerator.wait_for_everyone()

    def load_checkpoint(
            self,
            path: str,
            resume_type: Optional[Literal['lora', 'full', 'state']] = None,
        ):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(
            path=path,
            strict=True,
            resume_type=resume_type,
        )
        self.accelerator.wait_for_everyone()