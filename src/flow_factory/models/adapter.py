# src/flow_factory/models/adapter.py
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal
from dataclasses import dataclass, field, asdict, fields
from contextlib import contextmanager, nullcontext
import logging

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file, load_file
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin
from peft import get_peft_model, LoraConfig, PeftModel


from ..ema import EMAModuleWrapper
from ..scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput
from ..hparams import *
from ..utils.base import filter_kwargs, is_tensor_list

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

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


class BaseAdapter(nn.Module, ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """
    
    pipeline: DiffusionPipeline

    def __init__(self, config: Arguments):
        super().__init__()
        self.config = config
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.eval_args = config.eval_args

        # Load pipeline and scheduler (delegated to subclasses)
        self.pipeline = self.load_pipeline()
        self.pipeline.scheduler = self.load_scheduler()

        self.target_module_map = self._init_target_module_map()

        # Freeze non-trainable components
        self._freeze_components()

        # Load checkpoint or apply LoRA
        if self.model_args.resume_path:
            self.load_checkpoint(self.model_args.resume_path)
        elif self.model_args.finetune_type == 'lora':
            self.apply_lora(
                components=self.model_args.target_components,
                target_modules=self.model_args.target_modules
            )

        # Set precision
        self._mix_precision()
        self._set_trainable_params_dtype()

        self._init_ema()

        # Enable gradient checkpointing if needed
        if self.training_args.enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()

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

    @property
    def vae(self) -> torch.nn.Module:
        return self.pipeline.vae
    
    @vae.setter
    def vae(self, module: torch.nn.Module):
        self.pipeline.vae = module
    
    @property
    def transformer(self) -> torch.nn.Module:
        return self.pipeline.transformer
    
    @property
    def transformers(self) -> List[torch.nn.Module]:
        """Collect all transformers from pipeline."""
        transformers = []
        for attr_name, attr_value in vars(self.pipeline).items():
            if (
                'transformer' in attr_name 
                and not attr_name.startswith('_')  # Filter private attr
                and isinstance(attr_value, torch.nn.Module)
            ):
                transformers.append((attr_name, attr_value))
        
        transformers.sort(key=lambda x: x[0])
        return [trans for _, trans in transformers]
    
    @transformer.setter
    def transformer(self, module: torch.nn.Module):
        self.pipeline.transformer = module

    @property
    def scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        return self.pipeline.scheduler

    @scheduler.setter
    def scheduler(self, scheduler: FlowMatchEulerDiscreteSDEScheduler):
        self.pipeline.scheduler = scheduler

    @property
    def device(self) -> torch.device:
        return self.transformer.device
    
    @property
    def _inference_dtype(self) -> torch.dtype:
        """Get inference dtype based on mixed precision setting."""
        if self.config.mixed_precision == "fp16":
            return torch.float16
        elif self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32
    
    # ============================== Mode Management ==============================
    def eval(self, transformer_only: bool = True):
        """Set model to evaluation mode."""
        super().eval()
        
        if not transformer_only:
            # Set all components to eval mode
            for encoder in self.text_encoders:
                encoder.eval()
            self.vae.eval()

        for transformer in self.transformers:
            transformer.eval()

        if hasattr(self.scheduler, 'eval'):
            self.scheduler.eval()

    def rollout(self, *args, **kwargs):
        """Set the model to rollout mode if applicable. Base implementation sets `transformer` to eval mode and try to set scheduler to rollout mode."""
        for transformer in self.transformers:
            transformer.eval()
        
        if hasattr(self.scheduler, 'rollout'):
            self.scheduler.rollout(*args, **kwargs)

    def train(self, mode: bool = True, transformer_only: bool = True):
        """Set model to training mode."""
        super().train(mode)
        
        # Set all components to training mode
        if not transformer_only:
            for encoder in self.text_encoders:
                encoder.train(mode)
            self.vae.train(mode)

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
        target_modules: Union[str, List[str], None],
        components: List[str]
    ) -> Dict[str, Union[str, List[str], None]]:
        """
        Parse target_modules config into component-specific mapping.
        
        Args:
            target_modules: 
                - 'default': Use self.default_target_modules
                - 'all': Unfreeze all parameters
                - str: Single module pattern
                - List[str]: Module patterns with optional component prefix
                - None: Use self.default_target_modules
            components: List of component names to map modules to
        
        Returns:
            Dict mapping component names to their target modules.
            Example: {
                'transformer': ['attn.to_q', 'attn.to_k'],
                'transformer_2': 'all',
                'transformer_3': None
            }
        """
        # Normalize target_modules
        if target_modules == 'default' or target_modules is None:
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
        for comp in components:
            if component_map[comp]:
                component_map[comp] = list(set(component_map[comp]))
            else:
                component_map[comp] = None
        
        return component_map
    
    def _init_target_module_map(self) -> Dict[str, Union[str, List[str], None]]:
        """
        Initialize and cache target module mapping from config.
        
        Returns:
            Dict mapping component names to their target modules.
        """
        component_map = self._parse_target_modules(
            target_modules=self.model_args.target_modules,
            components=self.model_args.target_components
        )
        
        logger.info("Target module map:")
        for comp, modules in component_map.items():
            logger.info(f"  {comp}: {modules}")
        
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
            with self.ema_wrapper.use_ema_parameters(self.get_trainable_parameters()):
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

    def _set_trainable_params_dtype(self):
        """Set trainable parameters to master weight dtype."""
        master_dtype = self.model_args.master_weight_dtype
        inference_dtype = self._inference_dtype
        
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
        components: Union[str, List[str]] = 'transformer',
        target_modules: Optional[Union[str, List[str]]] = None
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        """
        Apply LoRA adapters to specified components with prefix-based module targeting.
        
        Args:
            components: Component(s) to apply LoRA
            target_modules: Module patterns with optional component prefix
                - 'to_q': Apply to all components in `components`
                - 'transformer.to_q': Apply only to transformer
                - 'transformer_2.to_v': Apply only to transformer_2
                - ['to_q', 'transformer.to_k']: Mixed specification
        """
        # Normalize components to list
        if isinstance(components, str):
            components = [components]
        
        if target_modules is None:
            # Use cached map, filtering for requested components
            component_modules = {
                comp: self.target_module_map.get(comp, self.default_target_modules)
                for comp in components
            }
        else:
            # Re-parse with explicit target_modules
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
            setattr(self, comp, model_component)
            results[comp] = model_component
            
            logger.info(f"Applied LoRA to {comp} with modules: {modules}")
        
        return results[components[0]] if len(results) == 1 else results

    # ============================== Checkpoint Management ==============================
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
                component.from_pretrained(comp_path)
                logger.info(f"Weights loaded for {comp_name}")

        # Move model back to target device
        self.on_load()

    def save_checkpoint(
            self,
            path: str,
            max_shard_size: str = "5GB",
            dtype: Union[torch.dtype, str] = torch.bfloat16,
            save_ema: bool = True,
        ):
        """Save checkpoint for target components."""
        save_context = nullcontext if not save_ema else self.use_ema_parameters

        with save_context():
            os.makedirs(path, exist_ok=True)

            if isinstance(dtype, str):
                dtype = {
                    'bfloat16': torch.bfloat16,
                    'float16': torch.float16,
                    'float32': torch.float32
                }.get(dtype.lower(), torch.bfloat16)

        for comp_name in self.model_args.target_components:
            if not hasattr(self, comp_name):
                logger.warning(f"Component {comp_name} not found, skipping save")
                continue

            component = getattr(self, comp_name)
            comp_path = os.path.join(path, comp_name) if len(self.model_args.target_components) > 1 else path
            os.makedirs(comp_path, exist_ok=True)

            if self.model_args.finetune_type == 'lora' and isinstance(component, PeftModel):
                logger.info(f"Saving LoRA adapter for {comp_name} to {comp_path}")
                component.save_pretrained(comp_path)
            else:
                logger.info(f"Saving full weights for {comp_name} to {comp_path}")
                component.to(dtype).save_pretrained(comp_path, max_shard_size=max_shard_size)

        logger.info(f"Checkpoint saved successfully to {path}")

    # ============================== Freezing Components ==============================
    def _freeze_text_encoders(self):
        """Freeze all text encoders."""
        for i, encoder in enumerate(self.text_encoders):
            encoder.requires_grad_(False)

    def _freeze_vae(self):
        """Freeze VAE."""
        self.vae.requires_grad_(False)

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