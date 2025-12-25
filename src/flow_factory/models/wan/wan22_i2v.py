# src/flow_factory/models/wan/wan22_i2v.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from PIL import Image
import logging

from ..adapter import BaseAdapter, BaseSample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)



@dataclass
class WanSample(BaseSample):
    pass


class Wan22_I2V_Adapter(BaseAdapter):
    def __init__(self, config: Arguments):
        super().__init__(config)
    
    def load_pipeline(self) -> WanPipeline:
        return WanPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return []        