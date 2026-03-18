# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/utils/dmdr_utils.py
"""
DMDR (Distribution Matching Distillation + RL) sampling and loss utilities.
- SiT-style: model(latents, time_input, y, lora_scale=0) -> velocity.
- T2I adapter-style: adapter.forward(t, latents, prompt_embeds, ..., return_kwargs=['noise_pred']).
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision.transforms import transforms
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from ...utils.base import filter_kwargs


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    return x.mean(dim=tuple(range(1, x.ndim)))


def sample_continue(
    B: int,
    alpha: float = 4.0,
    beta: float = 1.5,
    s_type: str = "logit_normal",
    step: int = 0,
    dynamic_step: int = 1000,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Sample continuous timesteps in (0, 1).
    Returns tensor of shape (B, 1, 1, 1) for broadcasting.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32

    if s_type == "uniform":
        t = torch.rand(B, device=device, dtype=dtype) * (1.0 - 0.001) + 0.001
        return t.reshape(B, 1, 1, 1)
    if s_type == "logit_normal":
        alpha_f, beta_f = alpha, beta
        if dynamic_step > 0:
            progress = min(float(step) / float(dynamic_step), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
            alpha_f = 1.0 + (alpha - 1.0) * cosine_decay
            beta_f = 1.0 + (beta - 1.0) * cosine_decay
        t = torch.distributions.Beta(alpha_f, beta_f).sample((B,)).to(device=device, dtype=dtype)
        return t.reshape(B, 1, 1, 1)
    raise ValueError(f"Unsupported s_type: {s_type}. Choose from ['logit_normal', 'uniform'].")


def sample_discrete(
    B: int,
    timesteps: torch.Tensor,
    alpha: float = 4.0,
    beta: float = 1.0,
    s_type: str = "logit_normal",
    step: int = 0,
    dynamic_step: int = 1000,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Sample discrete timesteps from the given schedule.
    timesteps: 1D tensor of shape (T,).
    Returns tensor of shape (B, 1, 1, 1).
    """
    if device is None:
        device = timesteps.device
    if dtype is None:
        dtype = timesteps.dtype

    if s_type == "uniform":
        indices = torch.randint(0, len(timesteps), (B,), device=device)
        discrete = timesteps[indices]
        return discrete.reshape(B, 1, 1, 1)
    if s_type == "logit_normal":
        alpha_f, beta_f = alpha, beta
        if dynamic_step > 0:
            progress = min(float(step) / float(dynamic_step), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
            alpha_f = 1.0 + (alpha - 1.0) * cosine_decay
            beta_f = 1.0 + (beta - 1.0) * cosine_decay
        t = torch.distributions.Beta(alpha_f, beta_f).sample((B,)).to(device=device, dtype=dtype)
        timesteps_1 = timesteps.reshape(1, -1).to(device=device, dtype=dtype)
        distances = (t.unsqueeze(-1) - timesteps_1).abs()
        closest = distances.argmin(dim=-1)
        discrete = timesteps_1[0, closest]
        return discrete.reshape(B, 1, 1, 1)
    raise ValueError(f"Unsupported s_type: {s_type}. Choose from ['logit_normal', 'uniform'].")


def get_sample(
    x0_all: list,
    sample_t: torch.Tensor,
    t_steps: torch.Tensor,
) -> torch.Tensor:
    """
    Index into list of x0 tensors by sampled timestep.
    x0_all: list of tensors, each (B, C, H, W).
    sample_t: (B, 1, 1, 1) or (B,) of timestep values.
    t_steps: 1D tensor of length len(x0_all).
    Returns tensor (B, C, H, W).
    """
    stacked = torch.stack(x0_all, dim=0)
    t_steps_1 = t_steps.reshape(1, -1).to(sample_t.device)
    b = sample_t.shape[0]
    if sample_t.ndim > 1:
        sample_t = sample_t.flatten()
    sample_t_ = sample_t.unsqueeze(-1)
    distances = (sample_t_ - t_steps_1).abs()
    indices = distances.argmin(dim=-1)
    return stacked[indices, torch.arange(b, device=stacked.device)]

def v2x0_sampler_adapter(
    adapter: Any,
    latents: torch.Tensor,
    embeddings_batch: Dict[str, Any],
    num_steps: int = 20,
    shift: float = 1.0,
    guidance_scale: float = 0.0,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Backward (flow-matching) sampler using adapter.forward.
    Uses adapter.scheduler for timesteps; each step predicts x0 (t_next=0) then chains.
    adapter.forward(t, latents, ..., t_next, next_latents=None, return_kwargs=['next_latents_mean'])
    Returns: (x_next, all_x0, t_steps_normalized) with t in [0, 1].
    """
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=latents.dtype, device=latents.device)
    t_steps = shift * t_steps / (1 + (shift - 1) * t_steps)
    t_steps[-1] = 0.0

    batch_size = latents.shape[0]
    device, dtype = latents.device, latents.dtype
    x_next = latents
    all_x0: List[torch.Tensor] = []

    zero_ts = torch.zeros(batch_size, device=device, dtype=torch.float32)
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        t_cur_b = torch.full((x_next.size(0),), t_cur.item(), device=device, dtype=torch.float32)
        forward_kwargs = {
            "t": t_cur_b,
            "latents": x_next,
            "t_next": zero_ts,
            "next_latents": None,
            "compute_log_prob": False,
            "return_kwargs": ["next_latents_mean", "noise_pred"],
            "guidance_scale": guidance_scale,
            **embeddings_batch,
        }
        forward_kwargs = filter_kwargs(adapter.forward, **forward_kwargs)
        out = adapter.forward(**forward_kwargs)
        x0 = out.next_latents_mean
        all_x0.append(x0)
        x_next = (1.0 - t_next) * x0 + t_next * torch.randn_like(x0, device=device, dtype=dtype)

    return x_next, all_x0, t_steps[:-1]



class DINOv2ProcessorWithGrad(BaseImageProcessor):
    def __init__(
        self,
        res,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.crop_size = res
        self.transforms = transforms.Compose([
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            transforms.Resize(size=(self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        ])

    def preprocess(
        self,
        images: ImageInput,
        **kwargs,
    ):
        images = self.transforms(images)
        return images