# src/flow_factory/models/adapter.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin

from ..scheduler.flow_matching import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput
from ..hparams import *

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
    extra_kwargs : Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(nn.Module, ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """
    
    pipeline: DiffusionPipeline
    scheduler: FlowMatchEulerDiscreteSDEScheduler
    transformer: Union[nn.Module, ModelMixin]

    def __init__(
            self,
            config: Arguments,
        ):
        super().__init__()
        self.config = config
        self.model_args = config.model_args
        self.training_args = config.training_args

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

    @abstractmethod
    def apply_lora(self):
        """Apply LoRA adapters to the model if specified."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load weights from a specific path (including LoRA adapters)."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save weights to a specific path (including LoRA adapters)."""
        pass

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

    @abstractmethod 
    def enable_gradient_checkpointing(self):
        """Default implementation for memory efficiency."""
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Returns generator for optimizer."""
        pass