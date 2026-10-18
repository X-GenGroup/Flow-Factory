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

import base64
import hashlib
import inspect
import math

# src/flow_factory/utils/base.py
import re
from contextlib import contextmanager
from io import BytesIO
from itertools import chain, combinations, permutations
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from PIL import Image

from .audio import *
from .image import *
from .video import *

# ------------------------------------Function Utils-------------------------------------


def filter_kwargs(func: Callable, **kwargs: Any) -> dict[str, Any]:
    """Filter kwargs to only include parameters accepted by func.

    Args:
        func: Target function
        **kwargs: Keyword arguments to filter

    Returns:
        Filtered kwargs containing only valid parameters
    """
    sig = inspect.signature(func)

    # Check if function accepts **kwargs
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    # If has **kwargs, accept all
    if has_var_keyword:
        return kwargs

    # Otherwise, filter to valid parameter names
    valid_keys = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def split_kwargs(funcs: list[Callable], **kwargs: Any) -> list[dict[str, Any]]:
    """Split kwargs among multiple functions by their signatures.

    Earlier functions have priority for overlapping params.

    Returns:
        List of filtered kwargs dicts, one per function
    """
    results = []
    remaining = kwargs.copy()

    for func in funcs:
        sig = inspect.signature(func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if has_var_keyword:
            results.append(remaining.copy())
        else:
            valid_keys = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
            matched = {k: v for k, v in remaining.items() if k in valid_keys}
            results.append(matched)
            # Remove matched keys so they don't go to later functions
            for k in matched:
                remaining.pop(k, None)

    return results


def json_default(o: Any) -> Any:
    """``json.dumps(obj, default=json_default)`` fallback for non-serializable values.

    Converts torch Tensors and numpy scalars to native Python values
    (e.g. 0-dim Tensors produced by HF datasets' torch formatter).

    Args:
        o: Object that the default JSON encoder cannot serialize

    Returns:
        A JSON-serializable equivalent of ``o``

    Raises:
        TypeError: If ``o`` is not a supported type
    """
    if isinstance(o, torch.Tensor):
        return o.item() if o.dim() == 0 else o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ------------------------------------Random Utils---------------------------------------
def create_generator(
    *args: int,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Generator:
    """Create a reproducible torch Generator seeded by combining integer keys.

    The seed is derived from ``hash(args) % 2**32``; tuples of ints have a
    stable hash in CPython (unlike tuples containing strings, which depend on
    ``PYTHONHASHSEED``), so the same integer keys reliably produce the same
    generator within and across runs.

    Args:
        *args: Any number of integers (e.g., base_seed, epoch, rank). Order
            matters — different orderings seed different generators.
        device: Target device for the generator. ``None`` (default) creates a
            CPU generator; pass ``accelerator.device`` / ``'cuda'`` to sample
            directly on GPU when feeding ``torch.rand`` / ``torch.randn`` /
            ``randn_tensor`` without a CPU↔GPU copy.

    Returns:
        A seeded ``torch.Generator`` on the requested device.
    """
    seed = hash(args) % (2**32)
    generator = torch.Generator(device=device) if device is not None else torch.Generator()
    generator.manual_seed(seed)
    return generator


def create_generator_by_prompt(prompts: List[str], base_seed: int) -> List[torch.Generator]:
    generators = []
    for batch_pos, prompt in enumerate(prompts):
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(
            hash_digest[:4], "big"
        )  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31)  # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


@contextmanager
def isolated_rng(seed: int):
    """Seed the global RNG inside a block and restore original state on exit.

    Useful when a third-party API (e.g. ``transformers`` ``.generate()``) only
    accepts global seeding via ``torch.manual_seed()`` and does not support
    passing a ``torch.Generator``.

    Saves and restores both CPU and all CUDA device RNG states so that
    downstream random operations (noise sampling, SDE steps, etc.) are
    completely unaffected.
    """
    cpu_state = torch.random.get_rng_state()
    gpu_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else None
    )
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if gpu_states is not None:
            torch.cuda.set_rng_state_all(gpu_states)


# ------------------------------------Combination Utils---------------------------------------


def num_to_base_tuple(num, base, length):
    """Convert ``num`` to a ``length``-tuple of base-``base`` digits, left-padded with zeros."""
    result = np.zeros(length, dtype=int)
    for i in range(length - 1, -1, -1):
        result[i] = num % base
        num //= base
    return tuple(result.tolist())


# ----------------------------------- Hash Utils --------------------------------------


def hash_pil_image(image: Image.Image, size: Optional[int] = None) -> str:
    """Generate a hash string for a PIL Image.

    Args:
        image: PIL Image object
        size: Optional thumbnail size for faster hashing. None uses full image.

    Returns:
        str: MD5 hash hex string
    """
    if size is not None:
        image = image.copy()
        image.thumbnail((size, size))
    return hashlib.md5(image.tobytes()).hexdigest()


def hash_tensor(tensor: torch.Tensor, max_elements: int = 1024) -> str:
    """Generate a stable hash string for a torch Tensor.

    Floating values are quantized at 1/255 resolution after sampling, so float precision noise
    does not change the hash and a ``[0, 1]`` pixel tensor hashes like its ``uint8`` bytes. The
    value range is never inferred and nothing is clamped.

    Args:
        tensor: Input tensor; floating pixels follow the ``[0, 1]`` sample-media contract.
        max_elements: Max elements to hash (for efficiency)

    Returns:
        str: MD5 hash hex string
    """
    flat = tensor.detach().flatten()
    n = flat.numel()

    # 1. Sample first (before quantization for efficiency)
    if n > max_elements:
        step = n // max_elements
        flat = flat[::step][:max_elements]

    # 2. Quantize floats at byte resolution (eliminates float precision issues)
    if flat.is_floating_point():
        quantized = (flat.double() * 255).round().to(torch.int64)
    else:
        quantized = flat.to(torch.int64)

    return hashlib.md5(quantized.cpu().numpy().tobytes()).hexdigest()


def hash_pil_image_list(images: List[Image.Image], size: int = 32) -> str:
    """Generate a combined hash for a list of PIL Images.

    Args:
        images: List of PIL Image objects
        size: Thumbnail size per image

    Returns:
        str: Combined MD5 hash hex string
    """
    hasher = hashlib.md5()
    for img in images:
        hasher.update(hash_pil_image(img, size=size).encode())
    return hasher.hexdigest()


def hash_tensor_list(tensors: List[torch.Tensor], max_elements_per_tensor: int = 1024) -> str:
    """Generate a combined hash for a list of torch Tensors.

    Args:
        tensors: List of input tensors
        max_elements_per_tensor: Max elements to hash per tensor for efficiency

    Returns:
        str: Combined MD5 hash hex string
    """
    hasher = hashlib.md5()
    for tensor in tensors:
        # Compute the per-tensor hash
        t_hash = hash_tensor(tensor, max_elements=max_elements_per_tensor)
        # Update the master hasher with this tensor's hash
        hasher.update(t_hash.encode())
    return hasher.hexdigest()


# ------------------------------------ Grid Latents --------------------------------------------


def divide_latents(latents: torch.Tensor, H: int, W: int, h: int, w: int) -> torch.Tensor:
    """Divide latents into sub-latents based on the specified sub-image size (h, w).

    Args:
        latents: The input latents tensor of shape (B, seq_len, C).
        H: Height of the original image.
        W: Width of the original image.
        h: Height of each sub-image.
        w: Width of each sub-image.

    Returns:
        A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
    """
    batch_size, image_seq_len, channels = latents.shape
    assert H % h == 0 and W % w == 0, "H and W must be divisible by h and w respectively."

    # Compute downsampling factor
    total_pixels = H * W
    downsampling_factor = total_pixels // image_seq_len

    # Check if downsampling factor is a perfect square
    downsample_ratio = int(math.sqrt(downsampling_factor))
    if downsample_ratio * downsample_ratio != downsampling_factor:
        raise ValueError(
            f"The downsampling ratio cannot be determined. Image pixels {total_pixels} and sequence length {image_seq_len} do not match."
        )

    # Calculate latent dimensions
    latent_H = H // downsample_ratio
    latent_W = W // downsample_ratio
    latent_h = h // downsample_ratio
    latent_w = w // downsample_ratio

    # Match check
    assert (
        latent_H * latent_W == image_seq_len
    ), f"Calculated latent dimensions {latent_H}x{latent_W} do not match sequence length {image_seq_len}"

    rows = latent_H // latent_h
    cols = latent_W // latent_w

    # Reshape latents to (B, latent_H, latent_W, C)
    latents = latents.view(batch_size, latent_H, latent_W, channels)

    # split into sub-grids: (B, rows, latent_h, cols, latent_w, C)
    latents = latents.view(batch_size, rows, latent_h, cols, latent_w, channels)

    # (B, rows, latent_h, cols, latent_w, C) -> (B, rows, cols, latent_h, latent_w, C)
    sub_latents = latents.permute(0, 1, 3, 2, 4, 5).contiguous()

    # (B, rows, cols, latent_h, latent_w, C) -> (B, rows, cols, sub_seq_len, C)
    sub_latents = sub_latents.view(batch_size, rows, cols, latent_h * latent_w, channels)

    return sub_latents


def merge_latents(sub_latents: torch.Tensor, H: int, W: int, h: int, w: int) -> torch.Tensor:
    """Merge sub-latents back into the original latents tensor.

    Args:
        sub_latents: A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
        H: Height of the original image.
        W: Width of the original image.
        h: Height of each sub-image.
        w: Width of each sub-image.

    Returns:
        The merged latents tensor of shape (B, seq_len, C).
    """
    batch_size, rows, cols, sub_seq_len, channels = sub_latents.shape

    vae_scale_factor = int(math.sqrt(h * w // sub_seq_len))
    # Calculate latent dimensions using the explicit parameters
    latent_h = h // vae_scale_factor
    latent_w = w // vae_scale_factor
    latent_H = H // vae_scale_factor
    latent_W = W // vae_scale_factor

    # Verify dimensions match
    assert (
        latent_h * latent_w == sub_seq_len
    ), f"sub_seq_len {sub_seq_len} does not match calculated sub-latent size {latent_h}x{latent_w}"
    assert rows * cols == (latent_H // latent_h) * (
        latent_W // latent_w
    ), f"Grid size {rows}x{cols} does not match expected grid size"

    # Reshape sub_latents to (B, rows, cols, latent_h, latent_w, C)
    sub_latents = sub_latents.view(batch_size, rows, cols, latent_h, latent_w, channels)

    # Merge by rearranging dimensions
    # (B, rows, cols, latent_h, latent_w, C) -> (B, rows, latent_h, cols, latent_w, C)
    merged = sub_latents.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape to (B, latent_H, latent_W, C)
    merged = merged.view(batch_size, latent_H, latent_W, channels)

    # Final reshape to (B, seq_len, C)
    merged = merged.view(batch_size, latent_H * latent_W, channels)

    return merged


# -----------------------------------Tensor Utils---------------------------------------


def to_broadcast_tensor(
    value: Union[int, float, List[int], List[float], torch.Tensor], ref_tensor: torch.Tensor
) -> torch.Tensor:
    """Convert a scalar, list, or tensor to a tensor broadcastable with ``ref_tensor``.

    The result has ``ref_tensor``'s device and dtype and shape ``(N, 1, ..., 1)`` with
    ``ref_tensor.ndim - 1`` trailing singleton dimensions, where ``N`` is
    ``ref_tensor.shape[0]`` for single-element inputs and ``value.numel()`` otherwise.
    """
    # Convert to tensor if not already a tensor
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value if isinstance(value, list) else [value])

    # Move to the correct device and data type
    value = value.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    # If scalar, expand to batch size
    if value.numel() == 1:
        value = value.expand(ref_tensor.shape[0])

    # Adjust shape for broadcasting
    return value.view(-1, *([1] * (len(ref_tensor.shape) - 1)))


def expand_per_sample(
    value: Optional[torch.Tensor], batch_size: int, name: str
) -> Optional[torch.Tensor]:
    """Return a per-sample ``(batch_size,)`` view of a shared or per-sample tensor.

    Distillation replays pass one boundary time shared by the whole batch, while
    rollouts pass one time per sample; per-sample forward loops accept both.

    Args:
        value: A 0-dim or ``(1,)`` tensor shared by the batch, a ``(batch_size,)``
            tensor, or ``None``.
        batch_size: Number of samples in the batch.
        name: Argument name reported in errors.

    Returns:
        ``None`` when ``value`` is ``None``, otherwise a ``(batch_size,)`` tensor.

    Raises:
        TypeError: If ``value`` is neither ``None`` nor a tensor.
        ValueError: If ``value`` is neither shared nor shaped ``(batch_size,)``.
    """
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected {name} as a torch.Tensor, received {type(value).__name__}")
    if value.ndim == 0 or tuple(value.shape) == (1,):
        return value.reshape(()).expand(batch_size)
    if tuple(value.shape) == (batch_size,):
        return value
    raise ValueError(
        f"expected {name} as a 0-dim, (1,), or ({batch_size},) tensor, "
        f"received shape {tuple(value.shape)}"
    )


def is_tensor_list(tensor_list: List[torch.Tensor]) -> bool:
    """Check if the input is a list of torch Tensors.

    Args:
        tensor_list: List to check.

    Returns:
        True if all elements are torch Tensors, False otherwise.
    """
    return isinstance(tensor_list, list) and all(isinstance(t, torch.Tensor) for t in tensor_list)


def map_tensor_leaves(
    value: Any,
    fn: Callable[[torch.Tensor], Any],
    max_depth: Optional[int] = None,
) -> Any:
    """Rebuild a nested list/tuple/dict, applying ``fn`` to each tensor leaf.

    Non-tensor leaves (PIL, str, int, ``np.ndarray``, etc.) pass through
    unchanged; containers are reconstructed immutably. ``max_depth`` bounds
    recursion: ``None`` unlimited, ``0`` acts only when ``value`` is a Tensor,
    ``N`` descends up to ``N`` container levels.
    """
    if isinstance(value, torch.Tensor):
        return fn(value)
    if max_depth == 0:
        return value
    next_depth = None if max_depth is None else max_depth - 1
    if isinstance(value, list):
        return [map_tensor_leaves(item, fn, next_depth) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensor_leaves(item, fn, next_depth) for item in value)
    if isinstance(value, dict):
        return {k: map_tensor_leaves(v, fn, next_depth) for k, v in value.items()}
    return value


def visit_tensor_leaves(
    value: Any,
    fn: Callable[[torch.Tensor], None],
    max_depth: Optional[int] = None,
) -> None:
    """Call ``fn`` on each tensor leaf of a nested list/tuple/dict (side effect)."""
    if isinstance(value, torch.Tensor):
        fn(value)
        return
    if max_depth == 0:
        return
    next_depth = None if max_depth is None else max_depth - 1
    if isinstance(value, (list, tuple)):
        for v in value:
            visit_tensor_leaves(v, fn, next_depth)
    elif isinstance(value, dict):
        for v in value.values():
            visit_tensor_leaves(v, fn, next_depth)


def move_tensors_to_device(
    value: Any,
    device: Union[torch.device, str],
    max_depth: Optional[int] = None,
) -> Any:
    """Recursively move tensor leaves of a nested container onto ``device``.

    Thin wrapper over :func:`map_tensor_leaves`; non-tensor leaves pass through
    unchanged and the original input is not modified.
    """
    return map_tensor_leaves(value, lambda t: t.to(device), max_depth)
