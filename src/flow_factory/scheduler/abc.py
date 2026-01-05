from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Literal
from argparse import Namespace
import logging
from dataclasses import dataclass, field, fields, asdict
import math

import torch
import numpy as np
from diffusers.utils.outputs import BaseOutput
from ..utils.logger_utils import setup_logger

@dataclass
class SDESchedulerOutput(BaseOutput):
    """
    Output class for a single SDE step in Flow Matching.
    """

    next_latents: Optional[torch.FloatTensor] = None
    next_latents_mean: Optional[torch.FloatTensor] = None
    std_dev_t: Optional[torch.FloatTensor] = None
    dt: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None
    noise_pred: Optional[torch.FloatTensor] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDESchedulerOutput":
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)