# src/flow_factory/rewards/clip.py
from accelerate import Accelerator
from typing import Optional, List, Union
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

from .abc import BaseRewardModel, RewardModelOutput
from ..hparams import RewardArguments


class CLIPRewardModel(BaseRewardModel):
    """
    CLIP-based reward model that computes image-text similarity.
    """
    
    DEFAULT_MODEL = "openai/clip-vit-large-patch32"
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        
        model_name_or_path = config.extra_kwargs.get("model_name_or_path", self.DEFAULT_MODEL)
        
        self.model = CLIPModel.from_pretrained(model_name_or_path, torch_dtype=self.dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """
        Compute CLIP similarity rewards for given prompts and images.
        
        Args:
            prompt: List of text prompts.
            image: List of generated images corresponding to the prompts.
            video: List of videos (uses first frame of each video).
        
        Returns:
            RewardModelOutput with cosine similarity scores as rewards.
        """
        # Handle video input (use first frame)
        if image is None and video is not None:
            image = [v[0] for v in video]
        
        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")
        
        assert len(prompt) == len(image), \
            f"Mismatch: {len(prompt)} prompts vs {len(image)} images"
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.model(**inputs)
        
        # Compute per-pair cosine similarity (diagonal of similarity matrix)
        # logits_per_image shape: [batch, batch] - we want diagonal elements
        image_embeds = outputs.image_embeds  # [batch, dim]
        text_embeds = outputs.text_embeds    # [batch, dim]
        
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # Per-sample cosine similarity (element-wise dot product)
        rewards = (image_embeds * text_embeds).sum(dim=-1)  # [batch]
        
        return RewardModelOutput(
            rewards=rewards.float().cpu(),
            extra_info={},
        )