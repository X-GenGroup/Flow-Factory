# src/flow_factory/models/wan/wan2_v2v.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import torch
from diffusers.pipelines.wan.pipeline_wan_video2video import WanVideoToVideoPipeline, prompt_clean
from PIL import Image
from accelerate import Accelerator
from peft import PeftModel

from ..adapter import BaseAdapter
from ..samples import V2VSample
from ...hparams import *
from ...scheduler import UniPCMultistepSDESchedulerOutput, set_scheduler_timesteps, UniPCMultistepSDEScheduler
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)



@dataclass
class WanV2VSample(V2VSample):
    video : Optional[Union[np.ndarray, torch.Tensor, List[Image.Image]]] = None


class Wan2_V2V_Adapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
    
    def load_pipeline(self) -> WanVideoToVideoPipeline:
        return WanVideoToVideoPipeline.from_pretrained(
            self.model_args.model_name_or_path,
        )
    
    def load_scheduler(self) -> UniPCMultistepSDEScheduler:
        """Load and return the scheduler."""
        sde_config_keys = ['noise_level', 'train_steps', 'num_train_steps', 'seed', 'dynamics_type']
        # Check keys:
        for k in sde_config_keys:
            if not hasattr(self.training_args, k):
                logger.warning(f"Missing SDE config key '{k}' in training_args, using default value")

        sde_config = {
            k: getattr(self.training_args, k)
            for k in sde_config_keys
            if hasattr(self.training_args, k)
        }
        scheduler_config = self.pipeline.scheduler.config.__dict__.copy()
        scheduler_config.update(sde_config)
        return UniPCMultistepSDEScheduler(**scheduler_config)
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # --- Self Attention ---
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            
            # --- Cross Attention ---
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",

            # --- Feed Forward Network ---
            "ffn.net.0.proj", "ffn.net.2"
        ]
    
    @property
    def inference_modules(self) -> List[str]:
        """Modules taht are requires for inference and forward"""
        if self.pipeline.config.boundary_ratio is None or self.pipeline.config.boundary_ratio <= 0:
            return ['transformer', 'vae']

        if self.pipeline.config.boundary_ratio >= 1:
            return ['transformer_2', 'vae']

        return ['transformer', 'transformer_2', 'vae']