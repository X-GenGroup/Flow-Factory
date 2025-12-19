# src/flow_factory/models/adapter.py
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file, load_file
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin
from peft import get_peft_model, LoraConfig, PeftModel


from ..scheduler.flow_matching import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput
from ..hparams import *

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
    prompt_ids : Optional[torch.FloatTensor]
    height : Optional[int] = None
    width : Optional[int] = None
    prompt : Optional[str] = None
    image: Optional[Image.Image] = None
    prompt_embeds : Optional[torch.FloatTensor] = None
    pooled_prompt_embeds : Optional[torch.FloatTensor] = None
    log_probs : Optional[torch.FloatTensor] = None
    text_ids : Optional[torch.Tensor] = None
    image_ids : Optional[torch.Tensor] = None
    extra_kwargs : Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(nn.Module, ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """
    
    pipeline: DiffusionPipeline

    def __init__(
            self,
            config: Arguments,
        ):
        super().__init__()
        self.config = config
        self.model_args = config.model_args
        self.training_args = config.training_args

    @property
    def transformer(self) -> torch.nn.Module:
        return self.pipeline.transformer

    @property
    def scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        return self.pipeline.scheduler
    
    def eval(self):
        """Set model to evaluation mode."""
        super().eval()

    def train(self, mode: bool = True):
        """Set model to training mode."""
        super().train(mode)

    @property
    def default_lora_target_modules(self) -> List[str]:
        """Default target modules for LoRA adaptation."""
        return []

    def apply_lora(self):
        """Apply LoRA adapters to the model if specified."""
        lora_config = LoraConfig(
            r=self.model_args.lora_rank,
            lora_alpha=self.model_args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.default_lora_target_modules,
        )
        transformer = get_peft_model(self.pipeline.transformer, lora_config)
        transformer.set_adapter("default")
        self.pipeline.transformer = transformer
        
        return transformer

    def load_checkpoint(self, path: str):
        """
        Loads safetensors checkpoints. Detects if the path contains LoRA adapters,
        a sharded full model, or a single safetensor file.
        """
        logger.info(f"Attempting to load checkpoint from {path}")

        if self.model_args.finetune_type == 'lora':
            logger.info("Detected LoRA checkpoint. Wrapping model...")
            if not isinstance(self.transformer, PeftModel):
                self.transformer = PeftModel.from_pretrained(self.transformer, path, is_trainable=True)
            else:
                self.transformer.load_adapter(path, self.transformer.active_adapter)
            logger.info("LoRA adapter loaded successfully.")
        else:
            self.transformer.from_pretrained(path)
            logger.info("Model weights loaded successfully from checkpoint.")

        # Move model back to target device
        self.on_load_transformer()

    def save_checkpoint(self, path: str, transformer_override=None, max_shard_size: str = "5GB"):
        """
        Saves the transformer checkpoint using safetensors. 
        Supports sharding for full-parameter tuning and native PEFT saving for LoRA.
        """
        model_to_save = transformer_override if transformer_override is not None else self.transformer
        os.makedirs(path, exist_ok=True)

        if self.model_args.finetune_type == 'lora' and isinstance(model_to_save, PeftModel):
            logger.info(f"Saving LoRA adapter safetensors to {path}")
            self.transformer.save_pretrained(path)
        else:
            logger.info(f"Preparing to save full-parameter shards to {path} (Max shard size: {max_shard_size})")
            model_to_save.save_pretrained(path, max_shard_size=max_shard_size)

        logger.info(f"Model shards saved successfully to {path}")
        

    @abstractmethod
    def off_load_text_encoder(self):
        """Off-load text encoder to CPU to save GPU memory."""
        pass

    @abstractmethod
    def off_load_vae(self):
        """Off-load VAE to CPU to save GPU memory."""
        pass

    @abstractmethod
    def off_load_transformer(self):
        """Off-load Transformer to CPU to save GPU memory."""
        pass

    def off_load(self):
        """Off-load all components to CPU."""
        self.off_load_text_encoder()
        self.off_load_vae()
        self.off_load_transformer()

    @abstractmethod
    def on_load_text_encoder(self, device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Load text encoder back to specified device. Defaults to model device."""
        pass

    @abstractmethod
    def on_load_vae(self, device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Load VAE back to specified device. Defaults to model device."""
        pass

    @abstractmethod
    def on_load_transformer(self, device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Load Transformer back to specified device. Defaults to model device."""
        pass

    def on_load(self, device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Load all components back to specified device."""
        self.on_load_text_encoder(device)
        self.on_load_vae(device)
        self.on_load_transformer(device)

    @abstractmethod
    def encode_prompts(
        self,
        prompts: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input text prompts into model-compatible embeddings/tensors.
        """
        pass

    @abstractmethod
    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes input images into latent representations if applicable.
        For Flow Matching models, this might be identity.
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

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.pipeline.transformer, 'enable_gradient_checkpointing'):
            self.pipeline.transformer.enable_gradient_checkpointing()
    
    @abstractmethod
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Returns generator for optimizer."""
        pass