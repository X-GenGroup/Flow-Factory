# src/flow_factory/models/adapter.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ..scheduler.flow_matching import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput
from ..hparams.training_args import TrainingArguments
from ..hparams.model_args import ModelArguments

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
    extra_kwargs : Dict[str, Any] = {}


class BaseAdapter(nn.Module, ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """
    def __init__(
            self,
            model_args : ModelArguments,
            training_args: TrainingArguments
        ):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.pipeline : DiffusionPipeline = DiffusionPipeline.from_pretrained(
            model_args.model_name_or_path,
        )
        self.scheduler = FlowMatchEulerDiscreteSDEScheduler(
            noise_level=training_args.noise_level,
            noise_steps=training_args.noise_steps,
            num_noise_steps=training_args.num_noise_steps,
        )
        self.pipeline.scheduler = self.scheduler

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        if self.pipeline is not None:
            self.pipeline.eval()

    def train(self, mode: bool = True):
        """Set model to training mode."""
        super().train(mode)
        if self.pipeline is not None:
            self.pipeline.train()

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