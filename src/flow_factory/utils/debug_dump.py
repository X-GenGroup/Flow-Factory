# src/flow_factory/utils/debug_dump.py
"""
Debug tensor dumping utilities for train-inference consistency analysis.

Controlled by environment variable `FLOW_FACTORY_DEBUG_OUTPUT_DIR`.
When set, per-step SDE tensors are saved during both sampling and training
for offline comparison.

Directory structure (organized by rank and batch_idx for exact correspondence):
    {debug_dir}/
    ├── sampling/
    │   ├── config.json                         # Scheduler config snapshot
    │   ├── rank_000/
    │   │   ├── batch_000/
    │   │   │   ├── metadata.pt                 # {prompts, num_samples, ...}
    │   │   │   ├── step_000/
    │   │   │   │   ├── noise_pred.pt            # v(x_t, t) velocity prediction
    │   │   │   │   ├── latents.pt               # x_t   — step input latent
    │   │   │   │   ├── next_latents.pt           # x_{t-1} — step output (with SDE noise)
    │   │   │   │   ├── next_latents_mean.pt      # μ(x_t, v) — deterministic ODE mean
    │   │   │   │   ├── log_prob.pt
    │   │   │   │   ├── sigma.pt
    │   │   │   │   ├── sigma_prev.pt
    │   │   │   │   ├── std_dev_t.pt
    │   │   │   │   ├── dt.pt
    │   │   │   │   └── noise_level.pt
    │   │   │   ├── step_001/
    │   │   │   └── ...
    │   │   ├── batch_001/
    │   │   └── ...
    │   ├── rank_001/
    │   └── ...
    └── training/
        ├── rank_000/
        │   ├── batch_000/
        │   │   ├── step_000/
        │   │   │   ├── noise_pred.pt            # v(x_t, t) velocity prediction
        │   │   │   ├── latents.pt               # x_t   — same as sampling (on-policy)
        │   │   │   ├── next_latents.pt           # x_{t-1} — same as sampling (on-policy)
        │   │   │   ├── next_latents_mean.pt      # μ(x_t, v) — from training forward
        │   │   │   ├── new_log_prob.pt
        │   │   │   ├── old_log_prob.pt
        │   │   │   ├── ratio.pt
        │   │   │   ├── sigma.pt
        │   │   │   ├── sigma_prev.pt
        │   │   │   ├── std_dev_t.pt
        │   │   │   ├── dt.pt
        │   │   │   └── noise_level.pt
        │   │   ├── step_001/
        │   │   └── ...
        │   ├── batch_001/
        │   └── ...
        ├── rank_001/
        └── ...
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Global flag to avoid repeated env lookups
_DEBUG_DIR_CACHE = None
_DEBUG_DIR_CHECKED = False


def get_debug_output_dir() -> Optional[str]:
    """Get debug output directory from environment variable, with caching."""
    global _DEBUG_DIR_CACHE, _DEBUG_DIR_CHECKED
    if not _DEBUG_DIR_CHECKED:
        _DEBUG_DIR_CACHE = os.environ.get("FLOW_FACTORY_DEBUG_OUTPUT_DIR")
        _DEBUG_DIR_CHECKED = True
    return _DEBUG_DIR_CACHE


def is_debug_enabled() -> bool:
    """Check if debug tensor dumping is enabled."""
    return get_debug_output_dir() is not None


def _get_step_dir(base_dir: str, rank: int, batch_idx: int, step_idx: int) -> str:
    """Construct the per-rank/per-batch/per-step directory path."""
    return os.path.join(base_dir, f"rank_{rank:03d}", f"batch_{batch_idx:03d}", f"step_{step_idx:03d}")


def save_debug_tensor(
    base_dir: str,
    rank: int,
    batch_idx: int,
    step_idx: int,
    name: str,
    tensor: torch.Tensor,
) -> None:
    """
    Save a single debug tensor to disk.

    All ranks save their own data (organized by rank directory) to enable
    per-rank consistency comparison.

    Args:
        base_dir: Base directory (e.g., "{debug_dir}/sampling" or "{debug_dir}/training")
        rank: Current process rank
        batch_idx: Current batch index within the epoch
        step_idx: Denoising step index
        name: Tensor name (used as filename without extension)
        tensor: The tensor to save
    """
    step_dir = _get_step_dir(base_dir, rank, batch_idx, step_idx)
    os.makedirs(step_dir, exist_ok=True)

    save_path = os.path.join(step_dir, f"{name}.pt")
    try:
        torch.save(tensor.detach().cpu().float(), save_path)
    except Exception as e:
        logger.warning(f"Failed to save debug tensor {name} at rank={rank} batch={batch_idx} step={step_idx}: {e}")


def save_debug_metadata(
    base_dir: str,
    rank: int,
    batch_idx: int,
    metadata: Dict[str, Any],
) -> None:
    """Save per-batch metadata (e.g., prompts, sample indices) for verification."""
    batch_dir = os.path.join(base_dir, f"rank_{rank:03d}", f"batch_{batch_idx:03d}")
    os.makedirs(batch_dir, exist_ok=True)
    metadata_path = os.path.join(batch_dir, "metadata.pt")
    try:
        # Convert non-serializable types to basic Python types
        serializable = {}
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                serializable[k] = v.detach().cpu()
            else:
                serializable[k] = v
        torch.save(serializable, metadata_path)
    except Exception as e:
        logger.warning(f"Failed to save debug metadata at rank={rank} batch={batch_idx}: {e}")


def save_debug_config(
    base_dir: str,
    config: Dict[str, Any],
    rank: int = 0,
) -> None:
    """Save debug configuration as JSON. Only rank 0 saves to avoid contention."""
    if rank != 0:
        return

    os.makedirs(base_dir, exist_ok=True)
    config_path = os.path.join(base_dir, "config.json")
    try:
        with open(config_path, "w") as f:
            # Convert non-serializable types
            serializable = {}
            for k, v in config.items():
                if isinstance(v, torch.Tensor):
                    v = v.tolist()
                elif hasattr(v, '__iter__') and not isinstance(v, (str, list, dict)):
                    v = list(v)
                serializable[k] = v
            json.dump(serializable, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save debug config: {e}")


def dump_sampling_step(
    debug_dir: str,
    step_idx: int,
    rank: int,
    batch_idx: int = 0,
    noise_pred: Optional[torch.Tensor] = None,
    latents: Optional[torch.Tensor] = None,
    next_latents: Optional[torch.Tensor] = None,
    next_latents_mean: Optional[torch.Tensor] = None,
    log_prob: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    sigma_prev: Optional[torch.Tensor] = None,
    std_dev_t: Optional[torch.Tensor] = None,
    dt: Optional[torch.Tensor] = None,
    noise_level: Optional[float] = None,
) -> None:
    """
    Dump all relevant sampling tensors for a single SDE step.

    Naming convention: parameter names match the scheduler's `step()` variables exactly.

    SDE transition diagram for one step:
        latents (x_t)  ──[model]──> noise_pred v(x_t,t)
                                       │
                          ┌────────────┴────────────┐
                          │  ODE mean (deterministic)│
                          │  next_latents_mean = ... │
                          └────────────┬────────────┘
                                       │ + std_dev_t * sqrt(-dt) * z
                                       ▼
                              next_latents (x_{t-1})

    Args:
        debug_dir: Root debug directory
        step_idx: Current denoising step index
        rank: Current process rank
        batch_idx: Current batch index within the epoch
        noise_pred: v(x_t, t) — model velocity prediction
        latents: x_t — input latent at current timestep (scheduler param name)
        next_latents: x_{t-1} — output latent after SDE step (with noise)
        next_latents_mean: μ(x_t, v) — deterministic ODE mean of transition
        log_prob: log p(x_{t-1} | x_t, v) — transition log probability
        sigma: t / 1000 — current noise schedule value
        sigma_prev: t_next / 1000 — next noise schedule value
        std_dev_t: SDE diffusion coefficient
        dt: sigma_prev - sigma — step size (negative)
        noise_level: η — noise injection level
    """
    sampling_dir = os.path.join(debug_dir, "sampling")

    tensor_map = {
        "noise_pred": noise_pred,
        "latents": latents,
        "next_latents": next_latents,
        "next_latents_mean": next_latents_mean,
        "log_prob": log_prob,
        "sigma": sigma,
        "sigma_prev": sigma_prev,
        "std_dev_t": std_dev_t,
        "dt": dt,
    }

    for name, tensor in tensor_map.items():
        if tensor is not None:
            save_debug_tensor(sampling_dir, rank, batch_idx, step_idx, name, tensor)

    # Save noise_level as a scalar tensor
    if noise_level is not None:
        save_debug_tensor(
            sampling_dir, rank, batch_idx, step_idx, "noise_level",
            torch.tensor([noise_level], dtype=torch.float32),
        )


def dump_training_step(
    debug_dir: str,
    step_idx: int,
    rank: int,
    batch_idx: int = 0,
    noise_pred: Optional[torch.Tensor] = None,
    latents: Optional[torch.Tensor] = None,
    next_latents: Optional[torch.Tensor] = None,
    next_latents_mean: Optional[torch.Tensor] = None,
    new_log_prob: Optional[torch.Tensor] = None,
    old_log_prob: Optional[torch.Tensor] = None,
    ratio: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    sigma_prev: Optional[torch.Tensor] = None,
    std_dev_t: Optional[torch.Tensor] = None,
    dt: Optional[torch.Tensor] = None,
    noise_level: Optional[float] = None,
) -> None:
    """
    Dump all relevant training tensors for a single SDE step.

    Naming convention: parameter names match the scheduler's `step()` and
    grpo.py optimizer variables exactly.

    Args:
        debug_dir: Root debug directory
        step_idx: Current denoising step index
        rank: Current process rank
        batch_idx: Current batch index within the epoch
        noise_pred: v(x_t, t) — model velocity from training forward
        latents: x_t — input latent (same as sampling on-policy)
        next_latents: x_{t-1} — output latent from sampling trajectory (stored in sample)
        next_latents_mean: μ(x_t, v) — deterministic ODE mean from training forward
        new_log_prob: log p_new(x_{t-1} | x_t, v_θ) — from training forward
        old_log_prob: log p_old(x_{t-1} | x_t, v_θ) — from sampling (transported)
        ratio: exp(new_log_prob - old_log_prob) — importance sampling ratio
        sigma: t / 1000 — current noise schedule value
        sigma_prev: t_next / 1000 — next noise schedule value
        std_dev_t: SDE diffusion coefficient
        dt: sigma_prev - sigma — step size (negative)
        noise_level: η — noise injection level
    """
    training_dir = os.path.join(debug_dir, "training")

    tensor_map = {
        "noise_pred": noise_pred,
        "latents": latents,
        "next_latents": next_latents,
        "next_latents_mean": next_latents_mean,
        "new_log_prob": new_log_prob,
        "old_log_prob": old_log_prob,
        "ratio": ratio,
        "sigma": sigma,
        "sigma_prev": sigma_prev,
        "std_dev_t": std_dev_t,
        "dt": dt,
    }

    for name, tensor in tensor_map.items():
        if tensor is not None:
            save_debug_tensor(training_dir, rank, batch_idx, step_idx, name, tensor)

    if noise_level is not None:
        save_debug_tensor(
            training_dir, rank, batch_idx, step_idx, "noise_level",
            torch.tensor([noise_level], dtype=torch.float32),
        )
