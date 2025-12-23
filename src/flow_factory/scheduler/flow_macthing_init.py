from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Literal
from argparse import Namespace
import logging
from dataclasses import dataclass
import math

import torch
import numpy as np
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ..utils.base import to_broadcast_tensor


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

def calculate_shift(
    image_seq_len : int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def set_scheduler_timesteps(
    scheduler,
    num_inference_steps: int,
    seq_len: int,
    sigmas: Optional[Union[List[float], np.ndarray]] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    # 5. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
    mu = calculate_shift(
        seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps

@dataclass
class FlowMatchEulerDiscreteSDESchedulerOutput(BaseOutput):
    """
    Output class for a single SDE step in Flow Matching.
    """

    prev_sample: torch.FloatTensor
    prev_sample_mean: torch.FloatTensor
    std_dev_t: torch.FloatTensor
    dt: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None
