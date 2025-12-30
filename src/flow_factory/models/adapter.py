# src/flow_factory/models/adapter.py
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal
from dataclasses import dataclass, field, asdict, fields
from contextlib import contextmanager, nullcontext, ExitStack
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from safetensors.torch import save_file, load_file
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin
from peft import get_peft_model, LoraConfig, PeftModel
from accelerate import Accelerator, DistributedType

from ..ema import EMAModuleWrapper
from ..scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput
from ..hparams import *
from ..utils.base import filter_kwargs, is_tensor_list
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)
@dataclass
class BaseSample(BaseOutput):
    """
    Base output class for Adapter models.
    The tensors are without batch dimension.
    """
    all_latents : torch.FloatTensor
    timesteps : torch.FloatTensor
    prompt_ids : torch.LongTensor
    height : Optional[int] = None
    width : Optional[int] = None
    image: Optional[Image.Image] = None
    prompt : Optional[str] = None
    negative_prompt : Optional[str] = None
    negative_prompt_ids : Optional[torch.LongTensor] = None
    prompt_embeds : Optional[torch.FloatTensor] = None
    negative_prompt_embeds : Optional[torch.FloatTensor] = None
    log_probs : Optional[torch.FloatTensor] = None
    image_ids : Optional[torch.Tensor] = None
    extra_kwargs : Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for memory tracking, excluding non-tensor fields."""
        result = asdict(self)
        extra = result.pop('extra_kwargs', {})
        result.update(extra)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseSample":
        """Create instance from dict, putting unknown fields into extra_kwargs."""
        field_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in d.items() if k in field_names and k != 'extra_kwargs'}
        extra = {k: v for k, v in d.items() if k not in field_names}
        assert not (set(extra) & field_names), f"Key collision: {set(extra) & field_names} when creating BaseSample from dict."
        extra.update(d.get('extra_kwargs', {}))
        return cls(**known, extra_kwargs=extra)
    
    def __getattr__(self, key: str) -> Any:
        """Access attributes. Check extra_kwargs if not found."""
        extra = object.__getattribute__(self, 'extra_kwargs')
        if key in extra:
            return extra[key]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def short_rep(self) -> Dict[str, Any]:
        """Short representation for logging."""
        def long_tensor_to_shape(t : torch.Tensor):
            if isinstance(t, torch.Tensor) and t.numel() > 16:
                return t.shape
            else:
                return t

        d = self.to_dict()
        d = {k: long_tensor_to_shape(v) for k,v in d.items()}
        return d

    def to(self, device: Union[torch.device, str], depth : int = 1) -> "BaseSample":
        """Move all tensor fields to specified device."""
        assert 0 <= depth <= 1, "Only depth 0 and 1 are supported."
        device = torch.device(device)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))
            elif depth == 1 and is_tensor_list(value):
                setattr(
                    self,
                    field.name,
                    [t.to(device) if isinstance(t, torch.Tensor) else t for t in value]
                )
            
        return self



class BaseAdapter(ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.eval_args = config.eval_args
        self._mode : str = 'train' # ['train', 'eval', 'rollout']

        # Load pipeline and scheduler (delegated to subclasses)
        self.pipeline = self.load_pipeline()
        self.pipeline.scheduler = self.load_scheduler()
        
        # Initialize prepared components cache
        self._prepared_components: Dict[str, torch.nn.Module] = {}

        # Cache target module mapping
        self.target_module_map = self._init_target_module_map()

        # Freeze non-trainable components
        self._freeze_components()

        # Load checkpoint or apply LoRA
        if self.model_args.resume_path:
            self.load_checkpoint(self.model_args.resume_path)
        elif self.model_args.finetune_type == 'lora':
            self.apply_lora(
                target_modules=self.model_args.target_modules,
                components=self.model_args.target_components,
            )

        # Set precision
        self._mix_precision()

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

    def load_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """Load and return the scheduler."""
        return FlowMatchEulerDiscreteSDEScheduler(
            noise_level=self.training_args.noise_level,
            train_steps=self.training_args.train_steps,
            num_train_steps=self.training_args.num_train_steps,
            seed=self.training_args.seed,
            dynamics_type=self.training_args.dynamics_type,
            **self.pipeline.scheduler.config.__dict__,
        )

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
    def scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        return self.pipeline.scheduler

    @scheduler.setter
    def scheduler(self, scheduler: FlowMatchEulerDiscreteSDEScheduler):
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

        # Normalize target_modules
        if target_modules == 'default':
            base_modules = self.default_target_modules
        elif target_modules == 'all':
            # Return 'all' for each component
            return {comp: 'all' for comp in components}
        elif isinstance(target_modules, str):
            base_modules = [target_modules]
        else:
            base_modules = target_modules
        
        # Initialize component map
        component_map = {comp: [] for comp in components}
        
        # Parse each module pattern
        for module in base_modules:
            # Split only on first dot to check for component prefix
            parts = module.split('.', 1)
            
            if len(parts) == 2 and parts[0] in components:
                # Component-specific: 'transformer.attn.to_q' -> transformer: ['attn.to_q']
                component_map[parts[0]].append(parts[1])
            else:
                # Shared: 'attn.to_q' -> apply to all components
                for comp in components:
                    component_map[comp].append(module)
        
        # Remove duplicates and handle empty lists
        component_map = {
            comp: sorted(list(set(mods))) if mods else None
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
            self.ema_wrapper = EMAModuleWrapper(
                parameters=self.get_trainable_parameters(),
                decay=self.training_args.ema_decay,
                update_step_interval=self.training_args.ema_update_interval,
                device=self.device,
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
        """Initialize reference parameters for target components."""
        if (
            self.training_args.kl_beta > 0.0
            and self.model_args.finetune_type in ['full']
        ):
            ref_param_device = self.accelerator.device if self.training_args.ref_param_device == 'same_as_model' else torch.device(self.training_args.ref_param_device)
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
            if modules == 'all':
                modules = self.default_target_modules
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
            model_component = get_peft_model(model_component, lora_config)
            model_component.set_adapter("default")
            setattr(self, comp, model_component)
            results[comp] = model_component
            
            logger.info(f"Applied LoRA to {comp} with modules: {modules}")
        
        return results[components[0]] if len(results) == 1 else results

    # ============================== Checkpoint Management ==============================

    # -------------------------------- Helpers --------------------------------

    @property
    def _distributed_type(self) -> DistributedType:
        """Get current distributed type."""
        return self.accelerator.distributed_type

    def _is_deepspeed(self) -> bool:
        """Check if DeepSpeed is enabled."""
        return self._distributed_type == DistributedType.DEEPSPEED


    def _is_fsdp(self) -> bool:
        """Check if FSDP is enabled."""
        return self._distributed_type == DistributedType.FSDP


    def _is_zero3(self) -> bool:
        """Check if DeepSpeed ZeRO Stage 3 is enabled."""
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return False
        ds_plugin = self.accelerator.state.deepspeed_plugin
        return ds_plugin is not None and ds_plugin.zero_stage == 3


    def _is_fsdp_full_shard(self) -> bool:
        """Check if FSDP Full Shard is enabled."""
        if self.accelerator.distributed_type != DistributedType.FSDP:
            return False
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        if fsdp_plugin is None:
            return False
        # Check for full sharding strategy
        from torch.distributed.fsdp import ShardingStrategy
        return fsdp_plugin.sharding_strategy in (
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        )
    
    @contextmanager
    def _fsdp_full_shard_save_context(self, model: torch.nn.Module):
        """Context manager for FSDP full state dict gathering."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        original_timeout = os.environ.get('NCCL_TIMEOUT') # Default 10 minutes
        # Extend NCCL timeout for large models if needed
        timeout_minutes = 60
        os.environ['NCCL_TIMEOUT'] = str(timeout_minutes * 60)
        
        try:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                yield
        finally:
            if original_timeout is not None:
                os.environ['NCCL_TIMEOUT'] = original_timeout
            else:
                os.environ.pop('NCCL_TIMEOUT', None)

    @contextmanager
    def _deepspeed_zero3_save_context(self, model: torch.nn.Module):
        """Context manager for DeepSpeed ZeRO-3 state dict gathering."""
        # Not Tested. Use with caution.
        from deepspeed.runtime.zero import GatheredParameters
        original_timeout = os.environ.get('NCCL_TIMEOUT')
        # Extend NCCL timeout for large models if needed
        timeout_minutes = 10
        os.environ['NCCL_TIMEOUT'] = str(timeout_minutes * 60)

        try:
            with GatheredParameters(list(model.parameters()), modifier_rank=0):
                yield
        finally:
            if original_timeout is not None:
                os.environ['NCCL_TIMEOUT'] = original_timeout
            else:
                os.environ.pop('NCCL_TIMEOUT', None)

    def _get_state_dict(
        self,
        model: torch.nn.Module,
        dtype: Optional[torch.dtype] = None,
    ) -> dict:
        """
        Get state dict with proper handling for distributed strategies.
        
        Handles:
            - Standard DDP / single GPU
            - DeepSpeed ZeRO Stage 3 (gathers sharded params)
            - FSDP Full Shard (gathers sharded params)
        """
        if self._is_fsdp_full_shard():
            with self._fsdp_full_shard_save_context(model):
                state_dict = model.state_dict()
        elif self._is_zero3():
            with self._deepspeed_zero3_save_context(model):
                state_dict = model.state_dict()
        else:
            # Standard case: unwrap and get state dict directly
            unwrapped = self.accelerator.unwrap_model(model)
            state_dict = unwrapped.state_dict()
        
        # Cast to target dtype if specified (only on main process with valid state_dict)
        if dtype is not None and state_dict is not None:
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        
        self.accelerator.wait_for_everyone()
        return state_dict

    @staticmethod
    def _filter_lora_state_dict(state_dict: dict) -> dict:
        """Filter state dict to only include LoRA parameters."""
        return {
            k: v for k, v in state_dict.items()
            if 'lora_' in k or 'modules_to_save' in k
        }
    
    # -------------------------------------------- Save ------------------------------------
    def _save_lora(
        self,
        model: torch.nn.Module,
        save_path: str,
        dtype: torch.dtype,
    ) -> None:
        """Save LoRA adapter with distributed training support."""        
        unwrapped = self.accelerator.unwrap_model(model)
        
        if not isinstance(unwrapped, PeftModel):
            logger.warning(f"Model is not a PeftModel, falling back to full save.")
            self._save_full_model(
                model,
                save_path,
                dtype=dtype,
            )
            return
        
        if self._is_zero3() or self._is_fsdp_full_shard():
            # Gather all params before saving
            state_dict = self._get_state_dict(model, dtype=dtype)
            # Filter LoRA params
            lora_state_dict = self._filter_lora_state_dict(state_dict)
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_path,
                    state_dict=lora_state_dict,
                )
        else:
            # Standard save
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(save_path)

    def _save_full_model(
        self,
        model: torch.nn.Module,
        save_path: str,
        dtype: torch.dtype,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save full model weights with distributed training support."""
        unwrapped = self.accelerator.unwrap_model(model)
        state_dict = self._get_state_dict(model, dtype=dtype)
        
        # Only main process saves
        if not self.accelerator.is_main_process:
            return
        
        if state_dict is None:
            logger.warning("State dict is None, skipping save")
            return
        
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(
                save_path,
                state_dict=state_dict,
                max_shard_size=max_shard_size,
                safe_serialization=True,
            )
        else:
            # Fallback to torch save
            weight_path = os.path.join(save_path, "model.safetensors")
            save_file(state_dict, weight_path)


    def save_checkpoint(
        self,
        path: str,
        max_shard_size: str = "5GB",
        dtype: Union[torch.dtype, str] = torch.bfloat16,
        save_ema: bool = True,
    ):
        """
        Save checkpoint for target components.
        
        Properly handles:
            - Standard DDP / single GPU
            - DeepSpeed ZeRO Stage 3
            - FSDP Full Shard
            - LoRA adapters
            - EMA weights
        """
        # Normalize dtype
        if isinstance(dtype, str):
            dtype = {
                'bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32,
            }.get(dtype.lower(), torch.bfloat16)
        
        # Setup EMA context
        save_context = self.use_ema_parameters if save_ema else nullcontext
        
        with save_context():            
            for comp_name in self.target_module_map.keys():
                if not hasattr(self, comp_name):
                    logger.warning(f"Component {comp_name} not found, skipping save")
                    continue
                
                component = getattr(self, comp_name)
                
                # Determine save path
                comp_path = (
                    os.path.join(path, comp_name) 
                    if len(self.model_args.target_components) > 1 
                    else path
                )
                
                if self.accelerator.is_main_process:
                    os.makedirs(comp_path, exist_ok=True)
                
                self.accelerator.wait_for_everyone()
                
                # Dispatch to appropriate save method
                if self.model_args.finetune_type == 'lora':
                    if self.accelerator.is_main_process:
                        logger.info(f"Saving LoRA weights for {comp_name} to {comp_path}")
                    self._save_lora(component, comp_path, dtype)
                else:
                    if self.accelerator.is_main_process:
                        logger.info(f"Saving full weights for {comp_name} to {comp_path}")
                    self._save_full_model(component, comp_path, dtype, max_shard_size)
            
            # Sync after saving
            self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            logger.info(f"Checkpoint saved successfully to {path}")

    # -------------------------------------------- Load -------------------------------------------
    def load_checkpoint(self, path: str):
        """
        Loads safetensors checkpoints. Detects if the path contains LoRA adapters,
        a sharded full model, or a single safetensor file.
        """
        logger.info(f"Attempting to load checkpoint from {path}")

        if self.model_args.finetune_type == 'lora':
            logger.info("Loading LoRA checkpoint...")
            for comp_name in self.model_args.target_components:
                if not hasattr(self, comp_name):
                    logger.warning(f"Component {comp_name} not found, skipping")
                    continue
                
                component = getattr(self, comp_name)
                comp_path = os.path.join(path, comp_name) if len(self.model_args.target_components) > 1 else path
                
                if not isinstance(component, PeftModel):
                    component = PeftModel.from_pretrained(component, comp_path, is_trainable=True)
                    component.set_adapter("default")
                    setattr(self, comp_name, component)
                else:
                    component.load_adapter(comp_path, component.active_adapter)
                
                logger.info(f"LoRA adapter loaded for {comp_name}")
        else:
            for comp_name in self.model_args.target_components:
                if not hasattr(self, comp_name):
                    logger.warning(f"Component {comp_name} not found, skipping")
                    continue
                
                component = getattr(self, comp_name)
                comp_path = os.path.join(path, comp_name) if len(self.model_args.target_components) > 1 else path
                component_class = type(component)
                component = component_class.from_pretrained(comp_path)
                setattr(self, comp_name, component)
                logger.info(f"Weights loaded for {comp_name}")

        # Move model back to target device
        self.on_load()

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
                component = getattr(self, comp, None)
                if component is not None and hasattr(component, 'to'):
                    component.to('cpu')
                    logger.info(f"Off-loaded {comp} to CPU")

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
                results.update(
                    encoder_method(
                        input,
                        **(filter_kwargs(encoder_method, **kwargs))
                    )
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
        Decodes latent representations back into images if applicable.
        For Flow Matching models, this might be identity.
        """
        pass

    # ======================================= Sampling & Training =======================================
    @abstractmethod
    def forward(
        self,
        samples : List[BaseSample],
        timestep_index : Union[int, torch.IntTensor, torch.LongTensor],
        compute_log_prob: bool = True,
        **kwargs,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        Calculates the log-probability of the action (image/latent) given the batch of samples.
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