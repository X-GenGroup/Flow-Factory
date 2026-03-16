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

from .base import filter_kwargs


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


def pred_v(
    model: torch.nn.Module,
    latents: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
    lora_scale: float = 0,
    num_classes: int = 1000,
) -> torch.Tensor:
    """
    Predict velocity with optional classifier-free guidance.
    model(latents, time_input, y, lora_scale=...) -> velocity.
    """
    do_cfg = cfg_scale > 1.0
    if do_cfg:
        latents = torch.cat([latents, latents], dim=0)
        t = torch.cat([t, t], dim=0)
        y_null = torch.full((y.size(0),), num_classes, device=y.device, dtype=y.dtype)
        y = torch.cat([y_null, y], dim=0)
    time_input = t.flatten()
    out = model(latents, time_input, y, lora_scale=lora_scale)
    if do_cfg:
        v_uncond, v_cond = out.chunk(2)
        out = v_uncond + cfg_scale * (v_cond - v_uncond)
    return out.to(latents.dtype)


def v2x0_sampler(
    model: torch.nn.Module,
    latents: torch.Tensor,
    y: torch.Tensor,
    num_steps: int = 20,
    shift: float = 1.0,
):
    """
    Backward (v-prediction) sampler that returns final x0 and list of per-step x0.
    Returns: (x_next, all_x0, t_steps)
    - x_next: final latent (B, C, H, W)
    - all_x0: list of x0 at each step (length num_steps)
    - t_steps: timesteps used (length num_steps)
    """
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=latents.dtype, device=latents.device)
    t_steps = shift * t_steps / (1 + (shift - 1) * t_steps)
    t_steps[-1] = 0.0

    x_next = latents
    device = x_next.device
    all_x0 = []
    dtype = latents.dtype

    with torch.no_grad():
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            time_input = torch.full((x_next.size(0),), t_cur.item(), device=device, dtype=dtype)
            d_cur = model(x_next, time_input, y)
            x_0 = x_next + (0.0 - t_cur) * d_cur
            all_x0.append(x_0)
            x_next = (1.0 - t_next) * x_0 + t_next * torch.randn_like(x_0, device=device, dtype=dtype)
    return x_next, all_x0, t_steps[:-1]


# --------------- T2I Adapter-based API (reuse existing Flow-Factory adapters) ---------------

def _timesteps_normalized(scheduler) -> torch.Tensor:
    """Return scheduler timesteps normalized to [0, 1] for DMDR interpolation."""
    t = scheduler.timesteps
    if isinstance(t, torch.Tensor):
        t = t.float()
        t_max = t.max().item()
        if t_max > 1.0:
            t = t / t_max
    else:
        t = torch.tensor(t, dtype=torch.float32)
        t = t / t.max().item()
    return t


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
    scheduler = adapter.scheduler
    if hasattr(scheduler, "set_timesteps"):
        try:
            scheduler.set_timesteps(num_steps, device=latents.device)
        except TypeError:
            scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, device=latents.device, dtype=latents.dtype)
    else:
        timesteps = timesteps.to(device=latents.device, dtype=latents.dtype)
    t_max = float(timesteps.max().item()) or 1.0
    if t_max > 1.0:
        timesteps_norm = timesteps / t_max
    else:
        timesteps_norm = timesteps.float()
    if shift != 1.0:
        timesteps_norm = shift * timesteps_norm / (1 + (shift - 1) * timesteps_norm)
        timesteps_norm[-1] = 0.0
    batch_size = latents.shape[0]
    device, dtype = latents.device, latents.dtype
    x_next = latents
    all_x0: List[torch.Tensor] = []

    zero_ts = torch.zeros(batch_size, device=device, dtype=timesteps.dtype)
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_cur_b = t_cur.expand(batch_size) if (isinstance(t_cur, torch.Tensor) and t_cur.ndim == 0) else torch.full((batch_size,), t_cur.item() if isinstance(t_cur, torch.Tensor) else t_cur, device=device, dtype=timesteps.dtype)
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
        sigma_next = timesteps_norm[i + 1]
        if isinstance(sigma_next, torch.Tensor):
            sigma_next = sigma_next.item() if sigma_next.ndim == 0 else sigma_next[0].item()
        x_next = (1.0 - sigma_next) * x0 + sigma_next * torch.randn_like(x0, device=device, dtype=dtype)

    t_steps_out = timesteps_norm[:-1]
    if t_steps_out.dim() == 0:
        t_steps_out = t_steps_out.unsqueeze(0)
    return x_next, all_x0, t_steps_out


def pred_velocity_adapter(
    adapter: Any,
    latents: torch.Tensor,
    t: torch.Tensor,
    embeddings_batch: Dict[str, Any],
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """
    Predict velocity (noise_pred) using adapter.forward.
    t: (B,) or (B,1,1,1) in [0,1]; will be scaled to scheduler scale if needed.
    """
    batch_size = latents.shape[0]
    device, dtype = latents.device, latents.dtype
    if t.ndim > 1:
        t_flat = t.flatten()
    else:
        t_flat = t
    scheduler = adapter.scheduler
    t_max = 1000.0
    if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None:
        tt = scheduler.timesteps
        if isinstance(tt, torch.Tensor):
            t_max = float(tt.max().item()) or 1000.0
        else:
            t_max = float(max(tt)) or 1000.0
    t_scaled = t_flat.to(device=device, dtype=dtype) * t_max
    forward_kwargs = {
        "t": t_scaled,
        "latents": latents,
        "t_next": None,
        "next_latents": None,
        "compute_log_prob": False,
        "return_kwargs": ["noise_pred"],
        "guidance_scale": guidance_scale,
        **embeddings_batch,
    }
    forward_kwargs = filter_kwargs(adapter.forward, **forward_kwargs)
    out = adapter.forward(**forward_kwargs)
    return out.noise_pred.to(latents.dtype)
