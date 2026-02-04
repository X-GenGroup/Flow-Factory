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

# src/flow_factory/models/abc.py
import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal, Iterable, Set
from dataclasses import dataclass, field, asdict, fields
from contextlib import contextmanager, nullcontext, ExitStack
import logging
import hashlib
import glob

import torch
import torch.nn as nn
import torch.distributed as dist

from PIL import Image
import numpy as np
from safetensors.torch import save_file, load_file
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from peft import get_peft_model, LoraConfig, PeftModel

from huggingface_hub import split_torch_state_dict_into_shards
from accelerate import Accelerator, DistributedType
from accelerate.state import PartialState
from accelerate.utils.modeling import (
    get_state_dict_offloaded_model,
)
from accelerate.utils import (
    WEIGHTS_NAME,
    WEIGHTS_PATTERN_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_PATTERN_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    has_offloaded_params,
    clean_state_dict_for_safetensors,
)

from ..utils.checkpoint import (
    mapping_lora_state_dict,
    infer_lora_config,
    infer_target_modules,
)
from .samples import BaseSample
from ..ema import EMAModuleWrapper
from ..scheduler import (
    load_scheduler as _load_scheduler,
    SDESchedulerOutput,
    SDESchedulerMixin,
)
from ..hparams import *
from ..utils.base import filter_kwargs, is_tensor_list
from ..utils.logger_utils import setup_logger

# Constants
CONFIG_NAME = "config.json"
DIFFUSION_WEIGHTS_NAME = "diffusion_pytorch_model.bin"
DIFFUSION_WEIGHTS_PATTERN_NAME = "diffusion_pytorch_model{suffix}.bin"
DIFFUSION_WEIGHTS_INDEX_NAME = f"{DIFFUSION_WEIGHTS_NAME}.index.json"
SAFE_DIFFUSION_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
SAFE_DIFFUSION_WEIGHTS_PATTERN_NAME = "diffusion_pytorch_model{suffix}.safetensors"
SAFE_DIFFUSION_WEIGHTS_INDEX_NAME = f"{SAFE_DIFFUSION_WEIGHTS_NAME}.index.json"
LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"

logger = setup_logger(__name__)

@dataclass
class NamedParametersInfo:
    """Metadata for named parameters snapshot."""
    target_components: List[str]
    ema_wrapper: EMAModuleWrapper

class BaseAdapter(ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """

    lora_keys: List[str] = [
            "lora_A", "lora_B",
            "lora_magnitude_vector",  # DoRA
            "lora_embedding_A", "lora_embedding_B",  # Embedding LoRA
            "modules_to_save",  # Additional modules marked for saving
        ]
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.eval_args = config.eval_args
        self._mode : str = 'train' # ['train', 'eval', 'rollout']
        self._named_parameters : Dict[str, NamedParametersInfo] = {}

        # Load pipeline and scheduler (delegated to subclasses)
        self.pipeline = self.load_pipeline()
        self.pipeline.scheduler = self.load_scheduler()
        
        # Initialize prepared components cache
        self._prepared_components: Dict[str, torch.nn.Module] = {}

        # Cache target module mapping
        self.target_module_map = self._init_target_module_map()

        # Load checkpoint
        if self.model_args.resume_path:
            self.load_checkpoint(
                self.model_args.resume_path,
                resume_type=self.model_args.resume_type
            )

        # Freeze non-trainable components
        self._freeze_components()

        # Apply LoRA if needed
        if self.model_args.finetune_type == 'lora':
            self.apply_lora(
                target_modules=self.model_args.target_modules,
                components=self.model_args.target_components,
                overwrite=False, # Do not overwrite existing adapters
            )

        # Set precision
        self._mix_precision()

        # Set attention backend for all transformers
        self._set_attention_backend()

        # Enable gradient checkpointing if needed
        if self.training_args.enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ================================== Post Init =================================
    def post_init(self):
        """Hook for additional initialization after main trainer's `accelerator.prepare`."""
        self._init_ema()
        self._init_ref_parameters()

    # ============================== Loading Components ==============================
    @abstractmethod
    def load_pipeline(self) -> DiffusionPipeline:
        """Load and return the diffusion pipeline. Must be implemented by subclasses."""
        pass

    def load_scheduler(self) -> SDESchedulerMixin:
        """Load and return the scheduler."""
        scheduler = _load_scheduler(
            pipeline_scheduler=self.pipeline.scheduler,
            scheduler_args=self.config.scheduler_args,
        )
        return scheduler

    # ============================== Component Accessors ==============================
    # ---------------------------------- Wrappers ----------------------------------
    def _unwrap(self, model: torch.nn.Module) -> torch.nn.Module:
        """Get the unwrapped model from accelerator."""
        return self.accelerator.unwrap_model(model)

    def set_prepared(self, name: str, module: torch.nn.Module):
        """Mark a component as prepared (after accelerator.prepare)."""
        self._prepared_components[name] = module
    
    def get_component(self, name: str) -> torch.nn.Module:
        """Get a component, preferring the prepared version if available."""
        return self._prepared_components.get(name) or getattr(self.pipeline, name)

    def get_component_unwrapped(self, name: str) -> torch.nn.Module:
        """Get the original unwrapped component."""
        return getattr(self.pipeline, name)
    
    def get_component_config(self, name: str):
        """Get the config of a component."""
        return getattr(self.pipeline, name).config

    def prepare_components(self, accelerator: Accelerator, component_names: List[str]):
        """Prepare specified components with the accelerator."""
        components = [getattr(self.pipeline, name) for name in component_names]
        prepared = accelerator.prepare(*components)
        for name, module in zip(component_names, prepared):
            self.set_prepared(name, module)
        return prepared

    # ------------------------------ Text Encoders & Tokenizers ------------------------------
    @property
    def text_encoders(self) -> List[torch.nn.Module]:
        """Collect all text encoders from pipeline."""
        encoders = []
        for attr_name, attr_value in vars(self.pipeline).items():
            if (
                'text_encoder' in attr_name 
                and not attr_name.startswith('_')  # Filter private attr
                and isinstance(attr_value, torch.nn.Module)
            ):
                encoders.append((attr_name, attr_value))
        
        encoders.sort(key=lambda x: x[0])
        return [enc for _, enc in encoders]

    @property
    def text_encoder(self) -> torch.nn.Module:
        """Get the first text encoder."""
        encoders = self.text_encoders
        if len(encoders) == 0:
            raise ValueError("No text encoder found in the pipeline.")
        return encoders[0]
    
    @text_encoder.setter
    def text_encoder(self, module: torch.nn.Module):
        encoders = self.text_encoders
        if len(encoders) == 0:
            raise ValueError("No text encoder found in the pipeline.")
        first_encoder_name = [name for name, _ in vars(self.pipeline).items() if 'text_encoder' in name][0]
        setattr(self.pipeline, first_encoder_name, module)

    @property
    def tokenizers(self) -> List[Any]:
        """Collect all tokenizers from pipeline."""
        tokenizers = []
        for attr_name, attr_value in vars(self.pipeline).items():
            if (
                'tokenizer' in attr_name 
                and not attr_name.startswith('_')  # Filter private attr
            ):
                tokenizers.append((attr_name, attr_value))
        
        tokenizers.sort(key=lambda x: x[0])
        return [tok for _, tok in tokenizers]
    
    @property
    def tokenizer(self) -> Any:
        """Get the first tokenizer. Or use the one with longest context length."""
        tokenizers = self.tokenizers
        if len(tokenizers) == 0:
            raise ValueError("No tokenizer found in the pipeline.")
        return tokenizers[0]

    # -------------------------------------- VAE --------------------------------------
    @property
    def vae(self) -> torch.nn.Module:
        return self.pipeline.vae
    
    @vae.setter
    def vae(self, module: torch.nn.Module):
        self.pipeline.vae = module
        
    # ---------------------------------- Transformers ----------------------------------
    @property
    def transformer_names(self) -> List[str]:
        """Get all transformer component names."""
        names = [
            name for name, value in vars(self.pipeline).items()
            if 'transformer' in name 
            and not name.startswith('_')
            and isinstance(value, torch.nn.Module)
        ]
        return sorted(names)

    @property
    def transformers(self) -> List[torch.nn.Module]:
        """Collect all transformers, preferring prepared versions."""
        return [self.get_component(name) for name in self.transformer_names]
    
    @property
    def transformer(self) -> torch.nn.Module:
        return self.get_component('transformer')

    @transformer.setter
    def transformer(self, module: torch.nn.Module):
        self.set_prepared('transformer', module)

    @property
    def transformer_config(self):
        return self.get_component_config('transformer')

    # ------------------------------------ Scheduler ------------------------------------
    @property
    def scheduler(self) -> SDESchedulerMixin:
        return self.pipeline.scheduler

    @scheduler.setter
    def scheduler(self, scheduler: Union[SDESchedulerMixin, SchedulerMixin]):
        self.pipeline.scheduler = scheduler

    # ---------------------------------- Device & Dtype ----------------------------------
    @property
    def device(self) -> torch.device:
        return self.accelerator.device
    
    @property
    def _inference_dtype(self) -> torch.dtype:
        """Get inference dtype based on mixed precision setting."""
        if self.config.mixed_precision == "fp16":
            return torch.float16
        elif self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    # ============================== Mode Management ==============================

    @property
    def mode(self) -> str:
        """Get current mode."""
        return self._mode

    def eval(self):
        """Set model to evaluation mode."""
        self._mode = 'eval'

        for transformer in self.transformers:
            transformer.eval()

        if hasattr(self.scheduler, 'eval'):
            self.scheduler.eval()

    def rollout(self, *args, **kwargs):
        """Set the model to rollout mode if applicable. Base implementation sets `transformer` to eval mode and try to set scheduler to rollout mode."""
        self._mode = 'rollout'

        for transformer in self.transformers:
            transformer.eval()
        
        if hasattr(self.scheduler, 'rollout'):
            self.scheduler.rollout(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set model to training mode."""
        self._mode = 'train' if mode else 'eval'
        for transformer in self.transformers:
            transformer.train(mode)

        if hasattr(self.scheduler, 'train'):
            self.scheduler.train(mode=mode)

    # ============================== Target Modules ==============================
    @property
    def default_target_modules(self) -> List[str]:
        """Default target modules for training."""
        return ['to_q', 'to_k', 'to_v', 'to_out.0']

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules that are requires for preprocessing"""
        return ['text_encoders', 'vae']
    
    @property
    def inference_modules(self) -> List[str]:
        """Modules that are required for inference and forward"""
        return ['transformer', 'vae']

    @property
    def trainable_component_names(self) -> List[str]:
        """Names of components with trainable parameters."""
        return [comp for comp, mods in self.target_module_map.items() if mods]

    @property
    def trainable_components(self) -> List[torch.nn.Module]:
        """Prepared model objects with trainable parameters."""
        return [self.get_component(name) for name in self.trainable_component_names]

    def _merge_module_pattern(
        self,
        current_pattern: Union[str, List[str], Set[str]],
        new_pattern: str
    ) -> Union[str, Set[str]]:
        """
        Resolve pattern and merge into current modules.
        
        Args:
            current: Current state ('all' or list of modules)
            pattern: New pattern to merge ('all', 'default', or module name)
        
        Returns:
            'all' or updated module list
        """
        # 'all' is absorbing - once set, stays 'all'
        if current_pattern == 'all' or new_pattern == 'all':
            return 'all'
        
        # Resolve pattern to module list
        new_modules = self.default_target_modules if new_pattern == 'default' else [new_pattern]
        new_pattern_set = set(current_pattern) | set(new_modules)
        return new_pattern_set

    def _parse_target_modules(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]]
    ) -> Dict[str, Union[List[str], None]]:
        """
        Parse target_modules config into component-specific mapping.
        
        Args:
            target_modules: 
                - 'default': Use self.default_target_modules
                - 'all': Unfreeze all parameters
                - str: Single module pattern
                - List[str]: Module patterns with optional component prefix
            components: Union[str, List[str]]
                - Component(s) to apply target_modules to.
        
        Returns:
            Dict mapping component names to their target modules.
            Example: {
                'transformer': ['attn.to_q', 'attn.to_k'],
                'transformer_2': 'all',
                'transformer_3': None
            }
        """
        # Normalize components to list
        if isinstance(components, str):
            components = [components]
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        
        component_map = {comp: set() for comp in components}
        
        for module in target_modules:
            parts = module.split('.', 1)
            if len(parts) == 2 and parts[0] in components:
                component_map[parts[0]] = self._merge_module_pattern(component_map[parts[0]], parts[1])
            else:
                for comp in components:
                    component_map[comp] = self._merge_module_pattern(component_map[comp], module)

        # Remove duplicates and handle empty lists
        component_map = {
            comp: ('all' if mods == 'all' else sorted(mods) if mods else None) # Keep None here, to enable `accelerator.prepare` for non-trainable module to save mem.
            for comp, mods in component_map.items()
        }
        
        return component_map
    
    def _init_target_module_map(self) -> Dict[str, Union[List[str], None]]:
        """
        Initialize and cache target module mapping from config.
        
        Returns:
            Dict mapping component names to their target modules.
        """
        component_map = self._parse_target_modules(
            target_modules=self.model_args.target_modules,
            components=self.model_args.target_components
        )
                
        return component_map

    # ============================== EMA Management ==============================
    def _init_ema(self):
        """Initialize EMA wrapper for the transformer."""
        if self.training_args.ema_decay > 0:
            ema_device = (
                self.accelerator.device 
                if self.training_args.ema_device == "cuda" 
                else torch.device("cpu")
            )
            self.ema_wrapper = EMAModuleWrapper(
                parameters=self.get_trainable_parameters(),
                decay=self.training_args.ema_decay,
                update_step_interval=self.training_args.ema_update_interval,
                device=ema_device,
            )
        else:
            self.ema_wrapper = None
    
    def ema_step(self, step : int):
        """Update EMA parameters."""
        if hasattr(self, 'ema_wrapper') and self.ema_wrapper is not None:
            self.ema_wrapper.step(
                self.get_trainable_parameters(),
                optimization_step=step
            )


    @contextmanager
    def use_ema_parameters(self):
        if hasattr(self, 'ema_wrapper') and self.ema_wrapper is not None:
            trainable_params = self.get_trainable_parameters()
            with self.ema_wrapper.use_ema_parameters(trainable_params):
                yield
        else:
            yield

    # ============================== Reference Parameters ==============================
    def _init_ref_parameters(self):
        """
            Initialize reference parameters for target components.
            Used for KL regularization during training.
        """
        if (
            self.training_args.kl_beta > 0.0
            and self.model_args.finetune_type in ['full']
        ):
            ref_param_device = (
                self.accelerator.device 
                if self.training_args.ref_param_device == "cuda" 
                else torch.device("cpu")
            )
            self._ref_ema = EMAModuleWrapper(
                parameters=self.get_trainable_parameters(),
                decay=0.0,  # No decay,
                update_step_interval=0,  # No updates, just store original weights
                device=ref_param_device,
            )
        else:
            self._ref_ema = None

    @contextmanager
    def use_ref_parameters(self):
        """Context manager to use reference parameters."""
        if self.model_args.finetune_type == 'lora':
            # Use ExitStack to manage multiple context managers (one per component)
            with ExitStack() as stack:
                enabled_any = False
                for comp_name in self.target_module_map.keys():
                    if hasattr(self, comp_name):
                        component = getattr(self, comp_name)
                        unwrapped = self._unwrap(component)

                        # Handle Compiled Models (torch.compile)
                        if hasattr(unwrapped, "_orig_mod"):
                            unwrapped = unwrapped._orig_mod

                        if isinstance(unwrapped, PeftModel):
                            # Enter disable_adapter context for each component
                            stack.enter_context(unwrapped.disable_adapter())
                            enabled_any = True
                if not enabled_any:
                    logger.warning("No LoRA adapters found to disable in use_ref_parameters")

                yield

        elif self._ref_ema is not None:
            trainable_params = self.get_trainable_parameters()
            # If ref_ema is on CPU, this line will be very slow!
            with self._ref_ema.use_ema_parameters(trainable_params):
                yield
        else:
            yield


    # ============================== Named Parameters Snapshot ==============================
    """
        These utilities help to snapshot and restore named parameters for target components.
        NOTE: `use_ref_parameters` always refers to the original model weights before any fine-tuning.

        For algorithms like DPO, GARDO, they requires to update reference model weights frequently.
        So we need more flexible utilities to manage multiple named parameter snapshots.
        The functions below help to store, use, update, and remove named parameter snapshots.
    """
    def _get_component_parameters(
        self, 
        target_modules: List[str]
    ) -> List[torch.nn.Parameter]:
        """Get trainable parameters from specified components."""
        params = []
        for comp_name in target_modules:
            if hasattr(self, comp_name):
                component = getattr(self, comp_name)
                params.extend(p for p in component.parameters() if p.requires_grad)
            else:
                logger.warning(f"Component '{comp_name}' not found in the model. Skipping.")
        return params
    
    def add_named_parameters(
        self,
        name: str,
        target_components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Store current trainable parameters snapshot under a name.
        
        Args:
            name: Identifier for this parameter snapshot
            target_modules: Component names to store. Defaults to components with trainable params.
            device: Storage device (defaults to 'cpu')
            overwrite: Whether to overwrite existing snapshot
        """
        if name in self._named_parameters and not overwrite:
            raise KeyError(f"Named parameters '{name}' exists. Use overwrite=True.")
        
        # Normalize target_modules - filter only those with trainable params
        if target_components is None:
            target_components = [k for k, v in self.target_module_map.items() if v]
        elif isinstance(target_components, str):
            target_components = [target_components]
        
        # Validate
        invalid = set(target_components) - set(self.target_module_map.keys())
        if invalid:
            raise ValueError(f"Invalid target_modules: {invalid}. Valid: {list(self.target_module_map.keys())}")
        
        device = torch.device(device) if device else torch.device('cpu')
        params = self._get_component_parameters(target_components)
        
        if not params:
            raise ValueError(f"No trainable parameters found in {target_components}")
        
        self._named_parameters[name] = NamedParametersInfo(
            target_components=target_components,
            ema_wrapper=EMAModuleWrapper(
                parameters=params,
                decay=0.0,
                update_step_interval=0,
                device=device,
            ),
        )
        logger.info(f"Stored named parameters '{name}' for {target_components} on {device}")
    
    @contextmanager
    def use_named_parameters(self, name: str):
        """
        Context manager to temporarily use named parameters.
        
        Args:
            name: Name of stored parameters snapshot
            
        Usage:
            adapter.add_named_parameters('init')
            # ... training ...
            with adapter.use_named_parameters('init'):
                evaluate(model)  # Uses stored weights
            # Current weights restored
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found. Available: {self.list_named_parameters()}")
        
        info = self._named_parameters[name]
        params = self._get_component_parameters(info.target_components)
        
        with info.ema_wrapper.use_ema_parameters(params):
            yield

    def update_named_parameters(
        self,
        name: str,
        target_components: Optional[Union[str, List[str]]] = None,
        new_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> None:
        """
        Update existing named parameters with specified or current values.
        
        Args:
            name: Name of snapshot to update
            target_modules: Components to update. Defaults to originally stored components.
            new_parameters: Parameters to copy from. Defaults to current model parameters.
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")
        
        info = self._named_parameters[name]
        
        # Resolve target_modules
        if target_components is None:
            target_components = info.target_components
        elif isinstance(target_components, str):
            target_components = [target_components]
        
        if not set(target_components).issubset(set(info.target_components)):
            raise ValueError(f"Must be subset of original: {info.target_components}")
        
        # Resolve parameters
        if new_parameters is None:
            new_parameters = self._get_component_parameters(target_components)
        else:
            new_parameters = list(new_parameters)
        
        # Validate param count
        if len(new_parameters) != len(info.ema_wrapper.ema_parameters):
            raise ValueError(
                f"Parameter count mismatch: got {len(new_parameters)}, "
                f"expected {len(info.ema_wrapper.ema_parameters)}"
            )
        
        # Update
        with torch.no_grad():
            for ema_param, param in zip(info.ema_wrapper.ema_parameters, new_parameters, strict=True):
                ema_param.data.copy_(param.detach().to(ema_param.device))
        
        logger.info(f"Updated named parameters '{name}'")

    def remove_named_parameters(self, name: str) -> None:
        """Remove named parameters."""
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")
        del self._named_parameters[name]
        logger.info(f"Removed named parameters '{name}'")

    def list_named_parameters(self) -> List[str]:
        """List all stored parameter names."""
        return list(self._named_parameters.keys())
    
    def get_named_parameters_info(self, name: str) -> Dict[str, Any]:
        """Get info about a named parameter snapshot."""
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")
        info = self._named_parameters[name]
        return {
            "name": name,
            "target_components": info.target_components,
            "num_params": len(info.ema_wrapper.ema_parameters),
            "device": str(info.ema_wrapper.device),
        }

    # ============================== Gradient Checkpointing ==============================
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for target components."""
        for comp_name in self.model_args.target_components:
            if hasattr(self, comp_name):
                component = getattr(self, comp_name)
                if hasattr(component, 'enable_gradient_checkpointing'):
                    component.enable_gradient_checkpointing()
                    logger.info(f"Enabled gradient checkpointing for {comp_name}")
                else:
                    logger.warning(f"{comp_name} does not support gradient checkpointing")

    # ============================== Attention Backend ==============================
    def _set_attention_backend(self) -> None:
        """
        Set attention backend for all transformer components.

        Refer to https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends#available-backends
        to see supported backends.
        """
        backend = self.model_args.attn_backend
        if backend is None:
            return
        
        for transformer_name in self.transformer_names:
            transformer = self.get_component(transformer_name)
            if hasattr(transformer, 'set_attention_backend'):
                transformer.set_attention_backend(backend)
                if self.accelerator.is_main_process:
                    logger.info(f"Set attention backend '{backend}' for {transformer_name}")

    # ============================== Precision Management ==============================
    def _mix_precision(self):
        """Apply mixed precision to all components."""
        inference_dtype = self._inference_dtype
        
        # Text encoders and VAE always use inference dtype
        for encoder in self.text_encoders:
            encoder.to(dtype=inference_dtype)
        
        self.vae.to(dtype=inference_dtype)
        
        # Transformer: inference dtype first (will be updated later for trainable params)
        for transformer in self.transformers:
            transformer.to(dtype=inference_dtype)

        master_dtype = self.model_args.master_weight_dtype
        
        if master_dtype == inference_dtype:
            return
        
        trainable_count = 0

        for comp_name in self.model_args.target_components:
            if hasattr(self, comp_name):
                component = getattr(self, comp_name)
                for name, param in component.named_parameters():
                    if param.requires_grad:
                        param.data = param.data.to(dtype=master_dtype)
                        trainable_count += 1
        
        if trainable_count > 0:
            logger.info(f"Set {trainable_count} trainable parameters to {master_dtype}")

    # ============================== LoRA Management ==============================
    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = 'transformer',
        overwrite: bool = False,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        """
        Apply LoRA adapters to specified components with prefix-based module targeting.
        
        Args:
            target_modules: Module patterns with optional component prefix
                - 'to_q': Apply to all components in `components`
                - 'transformer.to_q': Apply only to transformer
                - 'transformer_2.to_v': Apply only to transformer_2
                - ['to_q', 'transformer.to_k']: Mixed specification
            components: Component(s) to apply LoRA
            overwrite: When applying LoRA to a component that already has LoRA adapters:
                If True, delete existing 'default' adapter and create new one.
                If False, skip components that already have LoRA adapters.
        """
        # Normalize components to list
        if isinstance(components, str):
            components = [components]
        
        # Parse with explicit target_modules
        component_modules = self._parse_target_modules(target_modules, components)
        # Apply LoRA to each component
        results = {}
        for comp in components:
            modules = component_modules.get(comp)
            
            # Handle special cases
            if modules == 'default':
                modules = self.default_target_modules
            elif modules == 'all':
                modules = 'all' # Keep as 'all' for PEFT
            elif not modules:
                logger.warning(f"No target modules for {comp}, skipping LoRA")
                continue

            lora_config = LoraConfig(
                r=self.model_args.lora_rank,
                lora_alpha=self.model_args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=modules,
            )

            model_component = getattr(self, comp)

            if isinstance(model_component, PeftModel):
                # Already a PeftModel, check for existing adapter
                has_default = "default" in model_component.peft_config
                if has_default and not overwrite:
                    logger.info(f"Component {comp} already has 'default' adapter. Skipping.")
                    continue

                if has_default and overwrite:
                    # Overwrite: delete existing adapter and reinitialize
                    logger.info(f"Overwriting existing 'default' adapter for {comp}")
                    model_component.delete_adapter("default")

                # Add `default` adapter to existing PeftModel
                model_component.add_adapter("default", lora_config)
            else:
                # Not a PeftModel, initialize directly
                lora_config = LoraConfig(
                    r=self.model_args.lora_rank,
                    lora_alpha=self.model_args.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=modules,
                )
                model_component = get_peft_model(model_component, lora_config)
                # Set back to attribute
                setattr(self, comp, model_component)
         
            # Activate the adapter
            model_component.set_adapter("default")
            results[comp] = model_component
            
            logger.info(f"Applied LoRA to {comp} with modules: {modules}")
        
        if not results:
            logger.warning("No LoRA adapters were applied")
            return {}

        return next(iter(results.values())) if len(results) == 1 else results

    # ============================== Distributed Utils ==================================

    # ------------------------------ Dist Types -----------------------------------------
    @property
    def _distributed_type(self) -> DistributedType:
        """Get current distributed type."""
        return self.accelerator.distributed_type

    def _is_deepspeed(self) -> bool:
        """Check if DeepSpeed is enabled."""
        return self._distributed_type == DistributedType.DEEPSPEED

    def _is_fsdp(self) -> bool:
        """Check if FSDP (v1) is enabled."""
        return self._distributed_type == DistributedType.FSDP

    def _is_fsdp2(self) -> bool:
        """Check if FSDP2 is enabled."""
        return getattr(self.accelerator, 'is_fsdp2', False)

    def _is_fsdp_cpu_efficient_loading(self) -> bool:
        """Check if FSDP efficient loading is enabled."""
        if not self._is_fsdp():
            return False
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        return fsdp_plugin is not None and getattr(fsdp_plugin, "cpu_ram_efficient_loading", False)

    # ------------------------------ Shard Strategies ---------------------------------
    def _is_zero3(self) -> bool:
        """Check if DeepSpeed ZeRO Stage 3 (parameter sharding) is enabled."""
        if not self._is_deepspeed():
            return False
        ds_plugin = self.accelerator.state.deepspeed_plugin
        return ds_plugin is not None and ds_plugin.zero_stage == 3

    def _is_fsdp_param_sharded(self) -> bool:
        """Check if FSDP shards parameters across ranks (FULL_SHARD or HYBRID)."""
        if not self._is_fsdp():
            return False
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        if fsdp_plugin is None:
            return False
        from torch.distributed.fsdp import ShardingStrategy
        return fsdp_plugin.sharding_strategy in (
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        )

    # ------------------------------ FSDP Views ----------------------------------------
    def _fsdp_state_dict_type(self):
        """Get FSDP state_dict_type, returns None if not FSDP."""
        if not self._is_fsdp():
            return None
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        return fsdp_plugin.state_dict_type if fsdp_plugin else None

    def _is_fsdp_collective_state_dict(self) -> bool:
        """Check if FSDP state_dict_type requires collective operations."""
        from torch.distributed.fsdp import StateDictType
        state_dict_type = self._fsdp_state_dict_type()
        if state_dict_type is None:
            return False
        # LOCAL_STATE_DICT does not requires communication while others do
        return state_dict_type != StateDictType.LOCAL_STATE_DICT

    def _is_param_sharded(self) -> bool:
        """Check if parameters are sharded across ranks."""
        return self._is_zero3() or self._is_fsdp2() or self._is_fsdp_param_sharded()


    def _requires_collective_state_dict(self) -> bool:
        """
        Check if state_dict gathering requires all ranks to participate.
        
        This is True when:
        - DeepSpeed ZeRO-3 (parameters sharded)
        - FSDP2 (always uses collective ops)
        - FSDP with FULL/SHARDED_STATE_DICT (collective save)
        - FSDP with FULL_SHARD (parameters sharded, must gather)
        """
        if self._is_zero3():
            return True
        if self._is_fsdp2():
            return True
        if self._is_fsdp() and (
            self._is_fsdp_param_sharded()
            or self._is_fsdp_collective_state_dict()
        ):
            return True
        return False

    # ============================== Checkpoint Management ==============================


    # ------------------------------ State Dict ------------------------------------------

    def get_state_dict(
        self,
        model,
        unwrap=True,
        state_dict_keys: Optional[Iterable[str]] = None,
        ignore_frozen_params : bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        **Modified from `Accelerator.get_state_dict`**
        Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
        precision.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model sent through [`Accelerator.prepare`]
            unwrap (`bool`, *optional*, defaults to `True`):
                Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict
                (e.g. for DeepSpeed or FSDP models).
            state_dict_keys (`List[str]`, *optional*):
                If provided, only return the parameters with these keys in the state dict. This is useful for saving with FSDP
                when you only want to save the trainable parameters.
            ignore_frozen_params (`bool`, *optional*, defaults to `False`):
                For FSDP2 only. If `True`, frozen parameters (i.e., those with `requires_grad=False`) will be ignored when saving the state dict.
                
        Returns:
            `dict`: The state dictionary of the model potentially without full precision.
        ```
        """
        def is_param_match_key(name, keys, strict=True):
            # Later to add re logic maybe
            return (
                strict and (keys is not None and any(k == name for k in keys)) # Strict: `keys` is None returns False
                or (not strict) and keys is None or any(k in name for k in keys) # Non-strict: `keys` is None returns True
            )

        state_dict_keys = set(state_dict_keys) if state_dict_keys is not None else None

        from accelerate.utils import compare_versions

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            zero3_sharding = self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3
            tp_sharding = self.accelerator.deepspeed_config.get("tensor_parallel", {}).get("autotp_size", 0) > 1
            if zero3_sharding or tp_sharding:
                if model.zero_gather_16bit_weights_on_model_save():
                    ver_min_required = "0.16.4"
                    if tp_sharding and not compare_versions("deepspeed", ">=", ver_min_required):
                        raise ImportError(
                            f"Deepspeed TP requires deepspeed>={ver_min_required}. Please update DeepSpeed via `pip install deepspeed -U`."
                        )
                    state_dict = (
                        model._consolidated_16bit_state_dict()
                        if tp_sharding
                        else model._zero3_consolidated_16bit_state_dict()
                    )
                else:
                    raise ValueError(
                        "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                        "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                        "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                        "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                    )
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

                state_dict = clone_tensors_for_torch_save(self.accelerator.unwrap_model(model).state_dict())
        elif self.accelerator.is_fsdp2:
            # FSDP/FSDP2
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            if state_dict_keys is not None:
                # Temporarily mark unwanted params as frozen
                # This `requires_grad` trick does not work correctly. Don't know why.
                original_state = {}
                
                # Freeze unwanted params
                for name, param in model.named_parameters():
                    original_state[name] = param.requires_grad
                    param.requires_grad = is_param_match_key(name, state_dict_keys)
                
                options = StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    cpu_offload=True,
                    ignore_frozen_params=True,
                )
                state_dict = get_model_state_dict(model, options=options)
                
                # Restore original state
                for name, param in model.named_parameters():
                    param.requires_grad = original_state[name]
            else:
                options = StateDictOptions(
                    full_state_dict=True, 
                    broadcast_from_rank0=True, 
                    cpu_offload=True, 
                    ignore_frozen_params=ignore_frozen_params
                )
                state_dict = get_model_state_dict(model, options=options)
        elif self.accelerator.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict as fsdp_get_state_dict,
                get_model_state_dict as fsdp_get_model_state_dict
            )

            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = model.state_dict()
        else:
            if unwrap:
                model = self.accelerator.unwrap_model(model)
            state_dict = model.state_dict()

        # Filter by keys.
        state_dict = {
            k: v for k, v in state_dict.items()
            if is_param_match_key(k, state_dict_keys, strict=False)
        }

        return state_dict
    
    @classmethod
    def _filter_lora_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        adapter_name: str = "default",
    ) -> Dict[str, torch.Tensor]:
        """
        Filter state dict to only include LoRA parameters.
        
        Args:
            state_dict: Full model state dict
            adapter_name: Name of the LoRA adapter (default: "default")
        
        Returns:
            State dict containing only LoRA-related weights
        """
        return {
            k: v for k, v in state_dict.items()
            if any(lk in k for lk in cls.lora_keys)
        }
    
    # -------------------------------------------- Save ------------------------------------
    def _save_lora(
        self,
        model: torch.nn.Module,
        save_directory: str,
    ) -> None:
        """Save LoRA adapter with distributed training support."""        
        unwrapped = self.accelerator.unwrap_model(model)
        
        if not isinstance(unwrapped, PeftModel):
            logger.warning(f"Model is not a PeftModel, falling back to full save.")
            self._save_full_model(
                model,
                save_directory,
                safe_serialization=True,
            )
            return

        # If not sharded save, use standard save_pretrained
        if self._requires_collective_state_dict():
            # Handle sharded save
            # Gather all params before saving
            state_dict = self.get_state_dict(
                model,
                unwrap=True,
                state_dict_keys=self.lora_keys,
                ignore_frozen_params=True,
            )
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    state_dict=state_dict,
                )
        else:
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(save_directory)

        self.accelerator.wait_for_everyone()

    def _save_full_model(
        self,
        model: torch.nn.Module,
        save_directory: str,
        max_shard_size: str = "10GB",
        safe_serialization: bool = True,
        dtype : Optional[Union[torch.dtype, str]] = None,
    ) -> None:
        """
        **Modified from `Accelerator.save_model`**
        Save full model weights with distributed training support.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        
        # Normalize dtype
        if isinstance(dtype, str):
            dtype = {
                'bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32,
            }.get(dtype.lower(), torch.bfloat16)

        unwrapped = self.accelerator.unwrap_model(model)

        # Check if casting is needed
        cast_needed = False
        if dtype is not None:
            # Try to get model dtype, falling back to parameter inspection
            model_dtype = getattr(unwrapped, "dtype", None)
            if model_dtype is None:
                try:
                    model_dtype = next(unwrapped.parameters()).dtype
                except StopIteration:
                    # Empty model, assume no cast needed
                    model_dtype = dtype 
            
            if model_dtype != dtype:
                cast_needed = True
        
        # Check offload
        is_offloaded = any(has_offloaded_params(module) for module in unwrapped.modules())
        
        # No shard, no casting, no offload, save directyly
        if (
            not self._requires_collective_state_dict
            and not cast_needed
            and not is_offloaded
        ):
            # Standard save
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    max_shard_size=max_shard_size,
                    safe_serialization=safe_serialization,
                )
            self.accelerator.wait_for_everyone()
            return

        # Get the state_dict of the model
        if is_offloaded:
            state_dict = get_state_dict_offloaded_model(model)
        else:
            if any(param.device == torch.device("meta") for param in model.parameters()):
                raise RuntimeError("You can't save the model since some parameters are on the meta device.")
            state_dict = self.get_state_dict(model, unwrap=True, ignore_frozen_params=False)

        # Case: DeepSpeed zero3 gets gathered and `state_dict` is empty
        if state_dict is None:
            return

        # Dtype casting
        if dtype is not None:
            for k in state_dict.keys():
                state_dict[k] = state_dict[k].to(device='cpu', dtype=dtype)

        os.makedirs(save_directory, exist_ok=True)

        if safe_serialization:
            state_dict = clean_state_dict_for_safetensors(state_dict)

        weights_name = SAFE_DIFFUSION_WEIGHTS_NAME if safe_serialization else DIFFUSION_WEIGHTS_NAME
        filename_pattern = SAFE_DIFFUSION_WEIGHTS_PATTERN_NAME if safe_serialization else DIFFUSION_WEIGHTS_PATTERN_NAME

        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and reg.fullmatch(filename_no_suffix) is not None
                and PartialState().is_main_process
            ):
                os.remove(full_filename)

        # Save the model
        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            self.accelerator.save(shard, os.path.join(save_directory, filename), safe_serialization=safe_serialization)

        # Save the config file
        if hasattr(unwrapped, 'config') and unwrapped.config is not None:
            config_save_file = os.path.join(save_directory, CONFIG_NAME)
            if hasattr(unwrapped.config, 'save_pretrained'):
                unwrapped.config.save_pretrained(save_directory)
            else:
                # Handle dict-like configs (e.g., FrozenDict from diffusers)
                with open(config_save_file, 'w', encoding='utf-8') as f:
                    json.dump(dict(unwrapped.config), f, indent=2, sort_keys=True)

            if self.accelerator.is_main_process:
                logger.info(f"Model config saved in {config_save_file}")

        # Save index if sharded
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_DIFFUSION_WEIGHTS_INDEX_NAME if safe_serialization else DIFFUSION_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            if self.accelerator.is_main_process:
                logger.info(
                    f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                    f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                    f"index located at {save_index_file}."
                )
        else:
            path_to_weights = os.path.join(save_directory, weights_name)
            if self.accelerator.is_main_process:
                logger.info(f"Model weights saved in {path_to_weights}")

    def save_checkpoint(
        self,
        save_directory: str,
        max_shard_size: str = "10GB",
        dtype: Union[torch.dtype, str] = torch.bfloat16,
        save_ema: bool = True,
        model_only : bool = True,
        safe_serialization: bool = True,
        **kwargs,
    ):
        """
        Save checkpoint for target components.
        """
        # Normalize dtype
        if isinstance(dtype, str):
            dtype = {
                'bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32,
            }.get(dtype.lower(), torch.bfloat16)
            
        # 1. Save the training state if not model_only
        if not model_only:
            if self.accelerator.is_main_process:
                logger.info(f"Saving training state (resume-ready) to {save_directory}...")
            
            self.accelerator.save_state(save_directory, safe_serialization=safe_serialization, **kwargs)
            
            if self.accelerator.is_main_process:
                logger.info(f"Training state saved.")
            return

        # 2. Save only model
        # Setup EMA context
        save_context = self.use_ema_parameters if save_ema else nullcontext
        
        with save_context():
            for comp_name, target_modules in self.target_module_map.items():
                if not hasattr(self, comp_name):
                    logger.warning(f"Component {comp_name} not found, skipping save")
                    continue
                
                if not target_modules:
                    logger.info(f"No target modules applied to {comp_name}, skip saving")
                    continue

                component = getattr(self, comp_name)
                
                # Determine save path
                comp_path = (
                    os.path.join(save_directory, comp_name) 
                    if len(self.model_args.target_components) > 1 
                    else save_directory
                )
                
                os.makedirs(comp_path, exist_ok=True)
                
                # Dispatch to appropriate save method
                if self.model_args.finetune_type == 'lora':
                    if self.accelerator.is_main_process:
                        logger.info(f"Saving LoRA weights for {comp_name} to {comp_path}")
                    self._save_lora(component, comp_path)
                else:
                    if self.accelerator.is_main_process:
                        logger.info(f"Saving full weights for {comp_name} to {comp_path}")
                    self._save_full_model(
                        component,
                        comp_path,
                        max_shard_size=max_shard_size,
                        safe_serialization=safe_serialization,
                        dtype=dtype,
                    )
            
            # Sync after saving
            self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            logger.info(f"Checkpoint saved successfully to {save_directory}")

    # -------------------------------------------- Load -------------------------------------------
    @staticmethod
    def load_sharded_checkpoint(checkpoint_dir: str, index_file: str) -> Dict[str, torch.Tensor]:
        """Load sharded safetensors checkpoint."""
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        state_dict = {}
        loaded_files = set()
        
        for param_name, filename in index["weight_map"].items():
            if filename not in loaded_files:
                shard_path = os.path.join(checkpoint_dir, filename)
                shard = load_file(shard_path)
                state_dict.update(shard)
                loaded_files.add(filename)
        
        return state_dict

    def _load_lora(self, path: str) -> None:
        """Load LoRA adapters for target components with auto-format detection."""
        for comp_name in self.model_args.target_components:
            if not hasattr(self, comp_name):
                logger.warning(f"Component {comp_name} not found, skipping")
                continue
            
            component = getattr(self, comp_name)
            comp_path = (
                os.path.join(path, comp_name) 
                if len(self.model_args.target_components) > 1 
                else path
            )
            
            unwrapped = self.accelerator.unwrap_model(component)
            
            # Auto-detect checkpoint format
            adapter_config_path = os.path.join(comp_path, LORA_ADAPTER_CONFIG_NAME)
            has_config_file = os.path.exists(adapter_config_path)
            
            if has_config_file:
                # Standard PeftModel format
                if not isinstance(unwrapped, PeftModel):
                    unwrapped = PeftModel.from_pretrained(
                        unwrapped, comp_path, is_trainable=True
                    )
                    unwrapped.set_adapter("default")
                    setattr(self, comp_name, unwrapped)
                else:
                    unwrapped.load_adapter(comp_path, unwrapped.active_adapter)
            else:
                # No config file found, manual `state_dict` loading with key mapping
                # Detect `safetensors` or `bin` format with `safetensors` preferred
                safetensors_files = glob.glob(os.path.join(comp_path, "*.safetensors"))
                if safetensors_files:
                    state_dict_path = sorted(safetensors_files)[0]
                    state_dict = load_file(state_dict_path)
                else:
                    bin_files = glob.glob(os.path.join(comp_path, "*.bin"))
                    if bin_files:
                        state_dict_path = sorted(bin_files)[0]
                        state_dict = torch.load(state_dict_path, map_location='cpu')
                    else:
                        logger.error(f"No checkpoint file (.safetensors or .bin) found at {comp_path}")
                        continue
                
                if self.accelerator.is_main_process:
                    logger.info(
                        f"Loaded LoRA `state_dict` from: {state_dict_path}. "
                        f"If this is not wanted, please make sure the directory contains only single checkpoint file. "
                    )

                state_dict = load_file(state_dict_path) if state_dict_path.endswith('.safetensors') else torch.load(state_dict_path)
                
                # Apply key mapping for legacy format
                state_dict = mapping_lora_state_dict(state_dict)
                
                # Infer LoRA configuration from state_dict
                lora_rank, lora_alpha = infer_lora_config(state_dict)
                lora_alpha = self.model_args.lora_alpha or lora_alpha # Use model arg if given
                if self.model_args.target_modules in [None, 'default']:
                    # If default, infer target modules
                    target_modules = infer_target_modules(state_dict)
                else:
                    target_modules = self.model_args.target_modules
                
                if self.accelerator.is_main_process:
                    logger.info(
                        f"Inferred LoRA config for {comp_name}: "
                        f"rank={lora_rank}, alpha={lora_alpha}, target_modules={target_modules[:5]}..."
                    )
                
                # Create PeftModel if not already
                if not isinstance(unwrapped, PeftModel):
                    lora_config = LoraConfig(
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        init_lora_weights="gaussian",
                        target_modules=target_modules,
                    )
                    
                    unwrapped = get_peft_model(unwrapped, lora_config)
                    unwrapped.set_adapter("default")
                
                # Load mapped state_dict
                missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)

                # Filter missing keys to LoRA only
                missing = [k for k in missing if any(lk in k for lk in self.lora_keys)]
                
                if self.accelerator.is_main_process:
                    if missing:
                        logger.warning(f"Missing keys: {missing[:5]}...")
                    if unexpected:
                        logger.warning(f"Unexpected keys: {unexpected[:5]}...")
                
                setattr(self, comp_name, unwrapped)
            
            if self.accelerator.is_main_process:
                logger.info(f"LoRA adapter loaded for {comp_name} from {comp_path}")

    def _load_full_model(self, path: str, strict: bool = True) -> None:
        """Load full model weights for target components."""
        for comp_name in self.model_args.target_components:
            if not hasattr(self, comp_name):
                logger.warning(f"Component {comp_name} not found, skipping")
                continue
            
            component = getattr(self, comp_name)
            comp_path = (
                os.path.join(path, comp_name) 
                if len(self.model_args.target_components) > 1 
                else path
            )
            
            unwrapped = self.accelerator.unwrap_model(component)
            component_class = unwrapped.__class__
        
            # Try from_pretrained first
            try:
                new_component = component_class.from_pretrained(comp_path)
                setattr(self, comp_name, new_component)
                if self.accelerator.is_main_process:
                    logger.info(f"Loaded {comp_name} via from_pretrained from {comp_path}")
                continue
            except Exception as e:
                if self.accelerator.is_main_process:
                    logger.debug(f"from_pretrained failed for {comp_name}: {e}, trying manual load...")

            # Detect the checkpoint type
            index_file = os.path.join(comp_path, SAFE_DIFFUSION_WEIGHTS_INDEX_NAME)
            weights_file = os.path.join(comp_path, SAFE_DIFFUSION_WEIGHTS_NAME)

            if os.path.exists(index_file):
                state_dict = self.load_sharded_checkpoint(comp_path, index_file)
            elif os.path.exists(weights_file):
                state_dict = load_file(weights_file)
            else:
                logger.error(f"No valid checkpoint found for {comp_name} at {comp_path}")
                continue
            
            # Load state_dict
            missing, unexpected = unwrapped.load_state_dict(state_dict, strict=strict)
            
            if self.accelerator.is_main_process:
                if missing:
                    logger.warning(f"Missing keys for {comp_name}: {missing[:5]}...")
                if unexpected:
                    logger.warning(f"Unexpected keys for {comp_name}: {unexpected[:5]}...")
                logger.info(f"Full model weights loaded for {comp_name} from {comp_path}")

    def _load_training_state(self, path: str) -> None:
        """Load full training state for resuming training."""
        if self.accelerator.is_main_process:
            logger.info(f"Loading training state from {path}...")
        
        self.accelerator.load_state(path)
        
        if self.accelerator.is_main_process:
            logger.info("Training state loaded successfully.")

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        resume_type: Optional[Literal['lora', 'full', 'state']] = None,
    ) -> None:
        """
        Load checkpoint for target components.
        
        Args:
            path: Checkpoint directory path
            model_only: If True, load only model weights. If False, load full training state
                        (model, optimizer, scheduler, RNG states) for resuming training.
            strict: Whether to strictly enforce state_dict key matching (only for full model).
            resume_type: Type of checkpoint to load.
                - 'lora': Load LoRA adapters only
                - 'full': Load full model weights  
                - 'state': Load full training state (model + optimizer + scheduler + RNG)
                - None: Auto-detect based on finetune_type
        """
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        
        # Auto-detect if not specified
        if resume_type is None:
            resume_type = self.model_args.finetune_type  # 'lora' or 'full'
        
        if resume_type == 'state':
            self._load_training_state(path)
        elif resume_type == 'lora':
            self._load_lora(path)
        elif resume_type == 'full':
            self._load_full_model(path, strict=strict)
        else:
            raise ValueError(f"Invalid resume_type: {resume_type}. Available: ['lora', 'full', 'state'].")
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            logger.info(f"Checkpoint loaded successfully from {path} (type={resume_type})")

    # ============================== Freezing Components ==============================
    def _freeze_text_encoders(self):
        """Freeze all text encoders."""
        for i, encoder in enumerate(self.text_encoders):
            encoder.requires_grad_(False)
            encoder.eval()

    def _freeze_vae(self):
        """Freeze VAE."""
        self.vae.requires_grad_(False)
        self.vae.eval()

    def _freeze_transformer(self, trainable_modules: Optional[Union[str, List[str]]] = None):
        """
        Freeze transformer with optional selective unfreezing.
        
        Args:
            target_modules:
                - 'all': Unfreeze all parameters
                - 'default': Use self.default_target_modules
                - List[str]: Custom module name patterns to unfreeze
                - None / Empty list []: Freeze all (for LoRA)
        """
        if trainable_modules == 'all':
            logger.info("Unfreezing ALL transformer parameters")
            self.transformer.requires_grad_(True)
            return
        
        if isinstance(trainable_modules, str):
            if trainable_modules == 'default':
                trainable_modules = self.default_target_modules
            else:
                trainable_modules = [trainable_modules]

        # Freeze all first
        self.transformer.requires_grad_(False)
        
        # Early return if no modules to unfreeze
        if not trainable_modules:
            logger.info("Froze ALL transformer parameters")
            return
        
        # Selectively unfreeze
        trainable_count = 0
        for name, param in self.transformer.named_parameters():
            if any(target in name for target in trainable_modules):
                param.requires_grad = True
                trainable_count += 1
        
        if trainable_count == 0:
            logger.warning(f"No parameters matched target_modules: {trainable_modules}")
        else:
            logger.info(f"Unfroze {trainable_count} parameters in modules: {trainable_modules}")

    def _freeze_components(self):
        """Freeze strategy using cached target_module_map."""
        # Always freeze text encoders and VAE
        self._freeze_text_encoders()
        self._freeze_vae()
        
        # Freeze target components based on cached map
        for comp_name in self.model_args.target_components:
            if not hasattr(self, comp_name):
                logger.warning(f"Component {comp_name} not found, skipping freeze")
                continue
            
            trainable_modules = self.target_module_map.get(comp_name)
            
            # LoRA mode: freeze all (LoRA adapters will be trainable)
            if self.model_args.finetune_type == 'lora':
                trainable_modules = None
            
            self._freeze_component(comp_name, trainable_modules=trainable_modules)

    def _freeze_component(self, component_name: str, trainable_modules: Optional[Union[str, List[str]]] = None):
        """Freeze a specific component with optional selective unfreezing."""
        component = getattr(self, component_name)
        
        if trainable_modules == 'all':
            logger.info(f"Unfreezing ALL {component_name} parameters")
            component.requires_grad_(True)
            return
        
        if isinstance(trainable_modules, str):
            if trainable_modules == 'default':
                trainable_modules = self.default_target_modules
            else:
                trainable_modules = [trainable_modules]

        # Freeze all first
        component.requires_grad_(False)
        
        if not trainable_modules:
            logger.info(f"Froze ALL {component_name} parameters")
            return
        
        # Selectively unfreeze
        trainable_count = 0
        for name, param in component.named_parameters():
            if any(target in name for target in trainable_modules):
                param.requires_grad = True
                trainable_count += 1
        
        if trainable_count == 0:
            logger.warning(f"No parameters in {component_name} matched: {trainable_modules}")
        else:
            logger.info(f"Unfroze {trainable_count} parameters in {component_name}")


    # ============================== Trainable Parameters ==============================
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters from all target components."""
        params = []
        for comp_name in self.model_args.target_components:
            if hasattr(self, comp_name):
                component = getattr(self, comp_name)
                params.extend(filter(lambda p: p.requires_grad, component.parameters()))
        return params

    def log_trainable_parameters(self):
        """Log trainable parameter statistics for all target components."""
        for comp_name in self.model_args.target_components:
            if not hasattr(self, comp_name):
                continue
            
            component = getattr(self, comp_name)
            total_params = 0
            trainable_params = 0
            total_size_bytes = 0
            trainable_size_bytes = 0
            
            for param in component.parameters():
                param_count = param.numel()
                param_size = param.element_size() * param_count
                
                total_params += param_count
                total_size_bytes += param_size
                
                if param.requires_grad:
                    trainable_params += param_count
                    trainable_size_bytes += param_size
            
            total_size_gb = total_size_bytes / (1024 ** 3)
            trainable_size_gb = trainable_size_bytes / (1024 ** 3)
            trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
            
            logger.info("=" * 70)
            logger.info(f"{comp_name.capitalize()} Trainable Parameters:")
            logger.info(f"  Total:      {total_params:>15,d} ({total_size_gb:>6.2f} GB)")
            logger.info(f"  Trainable:  {trainable_params:>15,d} ({trainable_size_gb:>6.2f} GB)")
            logger.info(f"  Percentage: {trainable_percentage:>14.2f}%")
            logger.info("=" * 70)

    # ============================== Device Management ==============================
    def off_load_components(self, components: Optional[Union[str, List[str]]] = None):
        """Off-load specified components to CPU."""
        if components is None:
            if hasattr(self.pipeline, 'model_cpu_offload_seq'):
                components = self.pipeline.model_cpu_offload_seq.split('->')
            else:
                components = ['text_encoders', 'vae', 'transformers']
        elif isinstance(components, str):
            components = [components]
        
        for comp in components:
            if comp == 'text_encoders':
                self.off_load_text_encoders()
            elif comp == 'vae':
                self.off_load_vae()
            elif comp == 'transformers':
                self.off_load_transformers()
            else:
                component = self.get_component(comp)
                if component is not None and hasattr(component, 'to'):
                    component.to('cpu')
                    logger.info(f"Off-loaded {comp} to CPU")

    def on_load_components(self, components: Optional[Union[str, List[str]]] = None, device: Union[torch.device, str] = None):
        """Load specified components to device."""
        device = device or self.device
        
        if components is None:
            if hasattr(self.pipeline, 'model_cpu_offload_seq'):
                components = self.pipeline.model_cpu_offload_seq.split('->')
                components = components[::-1] # Reverse the order
            else:
                components = ['transformers', 'vae', 'text_encoders']
        elif isinstance(components, str):
            components = [components]
        
        for comp in components:
            if comp == 'text_encoders':
                self.on_load_text_encoders(device)
            elif comp == 'vae':
                self.on_load_vae(device)
            elif comp == 'transformers':
                self.on_load_transformers(device)
            else:
                component = self.get_component(comp)
                if component is not None and hasattr(component, 'to'):
                    component.to(device)
                    logger.info(f"Loaded {comp} to {device}")

    def off_load_text_encoders(self):
        """Off-load all text encoders to CPU."""
        for encoder in self.text_encoders:
            encoder.to("cpu")

    def off_load_vae(self):
        """Off-load VAE to CPU."""
        self.vae.to("cpu")

    def off_load_transformers(self):
        """Off-load Transformer to CPU."""
        for transformer in self.transformers:
            transformer.to("cpu")

    def off_load(self):
        """Off-load all components to CPU."""
        self.off_load_text_encoders()
        self.off_load_vae()
        self.off_load_transformers()

    def on_load_text_encoders(self, device: Union[torch.device, str] = None):
        """Load text encoders to device."""
        device = device or self.device
        for encoder in self.text_encoders:
            encoder.to(device)

    def on_load_vae(self, device: Union[torch.device, str] = None):
        """Load VAE to device."""
        device = device or self.device
        self.vae.to(device)

    def on_load_transformers(self, device: Union[torch.device, str] = None):
        """Load Transformer to device."""
        device = device or self.device
        for transformer in self.transformers:
            transformer.to(device)

    def on_load(self, device: Union[torch.device, str] = None):
        """Load all components to device."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.on_load_text_encoders(device)
        self.on_load_vae(device)
        self.on_load_transformers(device)


    # ============================== Preprocessing ==============================
    def preprocess_func(
        self,
        prompt : Optional[List[str]] = None,
        images : Optional[List[Union[Image.Image, List[Image.Image]]]] = None,
        videos : Optional[List[Union[List[Image.Image], List[List[Image.Image]]]]] = None,
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Preprocess input prompt, image, and video into model-compatible embeddings/tensors.
        Always process a batch of inputs.
        Args:
            prompt: List of text prompts. A batch of text inputs.
            images: 
                - None: no image input.
                - List[Image.Image]: list of images (a batch of single images)
                - List[List[Image.Image]]: list of list of images (a batch of a list images, each image list can be empty)
            videos: 
                - None: no video input.
                - List[Video]: list of videos (a batch of single videos)
                - List[List[Video]]: list of list of videos (a batch of a list videos, each video list can be empty)
            **kwargs: Additional keyword arguments for encoder methods.

        """
        results = {}
        
        for input, encoder_method in [
            (prompt, self.encode_prompt),
            (images, self.encode_image),
            (videos, self.encode_video),
        ]:
            if input is not None:
                res = encoder_method(
                        input,
                        **(filter_kwargs(encoder_method, **kwargs))
                    )

                if res is None:
                    # No preprocess needed
                    continue

                if (
                    isinstance(res, dict)
                    and res
                    and all(isinstance(v, (list, torch.Tensor, np.ndarray)) for v in res.values())
                ):
                    results.update(res)
                else:
                    raise ValueError(
                        f"Encoder method {encoder_method.__name__} should return a non-empty dict and each key maps to a list or tensor, " 
                        f"but got {type(res)} with values types {[type(v) for v in res.values()]}"
                    )

        return results

    @abstractmethod
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Tokenizes input text prompts into model-compatible embeddings/tensors.
        Args:
            prompt: Single or a batch of text prompts.
            **kwargs: Additional keyword arguments for tokenization/encoding.
        """
        pass

    @abstractmethod
    def encode_image(
        self,
        images : Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Encodes input images into latent representations if applicable.
        Args:
            images:
                - Single Image.Image
                - List[Image.Image]: list of images (a batch of single images)
                - List[List[Image.Image]]: list of list of images (a batch of multiple condition images)

        NOTE:
            The determination of input `images` type is based on:
                - if isinstance(images, Image.Image): single image
                - elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images): list of single images
                - elif isinstance(images, list) and all(isinstance(imgs, list) for imgs in images): list of list of images
            This function should return a dict containing the encoded representations,
                especially, the key `condition_images` should map to the processed PIL images, i.e., a batch of (list of) PIL images.
        """
        pass

    @abstractmethod
    def encode_video(
        self,
        videos: Union[List[Image.Image], List[List[Image.Image]], List[List[List[Image.Image]]]],
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Encodes input videos into latent representations if applicable.
        Args:
            videos:
                - List[Image.Image]: Single video input
                - List[List[Image.Image]]: list of videos (A batch of videos)
                - List[List[List[Image.Image]]]: list of list of videos (A batch of multiple condition videos)
        NOTE:
            The determination of input `videos` type should be based on:
                - if isinstance(videos, list) and all(isinstance(frame, Image.Image) for frame in videos): single video
                - elif isinstance(videos, list) and all(isinstance(frames, list) for frames in videos): list of videos
                - elif isinstance(videos, list) and all(isinstance(videos_list, list) for videos_list in videos): list of list of videos

            This function should return a dict containing the encoded representations,
            especially, the key `condition_videos` should map to the processed video frames, i.e., a batch of (list of) list of PIL images.
        """
        pass

    # ======================================= Postprocessing =======================================
    @abstractmethod
    def decode_latents(
        self,
        latents: torch.Tensor,
        **kwargs,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Decodes latent representations back into images/videos if applicable.
        """
        pass

    # ======================================= Sampling & Training =======================================
    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ) -> SDESchedulerOutput:
        """
        Calculates the log-probability of the action (image/latent) given inputs.
        """
        pass

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> List[BaseSample]:
        """
        Execute the generation process (Integration/Sampling).
        Returns a list of BaseSample instances.
        """
        pass