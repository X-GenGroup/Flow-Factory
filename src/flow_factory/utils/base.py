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

# src/flow_factory/utils/base.py
import re
import base64
import inspect
from io import BytesIO
from typing import List, Union, Optional, Dict, Callable, Any
from itertools import permutations, combinations, chain
import math
import hashlib

import torch.distributed as dist
from PIL import Image
import torch
import numpy as np
from accelerate import Accelerator

from .image import *
from .video import *

# ------------------------------------Function Utils-------------------------------------

def filter_kwargs(func: Callable, **kwargs: Any) -> dict[str, Any]:
    """
    Filter kwargs to only include parameters accepted by func.
    
    Args:
        func: Target function
        **kwargs: Keyword arguments to filter
    
    Returns:
        Filtered kwargs containing only valid parameters
    """
    sig = inspect.signature(func)
    
    # Check if function accepts **kwargs
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD 
        for p in sig.parameters.values()
    )
    
    # If has **kwargs, accept all
    if has_var_keyword:
        return kwargs
    
    # Otherwise, filter to valid parameter names
    valid_keys = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_keys}

def split_kwargs(funcs: list[Callable], **kwargs: Any) -> list[dict[str, Any]]:
    """
    Split kwargs among multiple functions by their signatures.
    Earlier functions have priority for overlapping params.
    
    Returns:
        List of filtered kwargs dicts, one per function
    """
    results = []
    remaining = kwargs.copy()
    
    for func in funcs:
        sig = inspect.signature(func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD 
            for p in sig.parameters.values()
        )
        
        if has_var_keyword:
            results.append(remaining.copy())
        else:
            valid_keys = set(sig.parameters.keys()) - {'self', 'args', 'kwargs'}
            matched = {k: v for k, v in remaining.items() if k in valid_keys}
            results.append(matched)
            # Remove matched keys so they don't go to later functions
            for k in matched:
                remaining.pop(k, None)
    
    return results

# ------------------------------------Random Utils---------------------------------------
def create_generator(prompts : List[str], base_seed : int) -> List[torch.Generator]:
    generators = []
    for batch_pos, prompt in enumerate(prompts):
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

# ------------------------------------Combination Utils---------------------------------------

def num_to_base_tuple(num, base, length):
    """
        Convert a `num` to given `base` and pad left with 0 to form a `length`-tuple
    """
    result = np.zeros(length, dtype=int)
    for i in range(length - 1, -1, -1):
        result[i] = num % base
        num //= base
    return tuple(result.tolist())

# ----------------------------------- Hash Utils --------------------------------------

def hash_pil_image(image: Image.Image, size: Optional[int] = None) -> str:
    """
    Generate a hash string for a PIL Image.
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
    """
    Generate a stable hash string for a torch Tensor.
    
    Quantizes to uint8 after sampling to ensure consistent hashing
    regardless of float precision issues.
    
    Args:
        tensor: Input tensor (supports [0,1], [-1,1], or [0,255] ranges)
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
    
    # 2. Quantize to uint8 (eliminates float precision issues)
    if flat.dtype == torch.uint8:
        uint8_flat = flat
    else:
        min_val, max_val = flat.min().item(), flat.max().item()
        if min_val >= -1.0 and max_val <= 1.0 and min_val < 0:
            # [-1, 1] -> [0, 255]
            uint8_flat = ((flat + 1) * 127.5).round().clamp(0, 255).byte()
        elif min_val >= 0 and max_val <= 1.0:
            # [0, 1] -> [0, 255]
            uint8_flat = (flat * 255).round().clamp(0, 255).byte()
        else:
            # Assume [0, 255]
            uint8_flat = flat.round().clamp(0, 255).byte()
    
    return hashlib.md5(uint8_flat.cpu().numpy().tobytes()).hexdigest()

def hash_pil_image_list(images: List[Image.Image], size: int = 32) -> str:
    """
    Generate a combined hash for a list of PIL Images.
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
    """
    Generate a combined hash for a list of torch Tensors.
    Args:
        tensors: List of input tensors
        max_elements_per_tensor: Max elements to hash per tensor for efficiency
    Returns:
        str: Combined MD5 hash hex string
    """
    hasher = hashlib.md5()
    for tensor in tensors:
        # Compute individual hash using your existing hash_tensor function
        t_hash = hash_tensor(tensor, max_elements=max_elements_per_tensor)
        # Update the master hasher with this tensor's hash
        hasher.update(t_hash.encode())
    return hasher.hexdigest()

# ------------------------------------ Grid Latents --------------------------------------------

def divide_latents(latents: torch.Tensor, H: int, W: int, h: int, w: int) -> torch.Tensor:
    """
    Divide latents into sub-latents based on the specified sub-image size (h, w).
    Args:
        latents (torch.Tensor): The input latents tensor of shape (B, seq_len, C).
        H (int): Height of the original image.
        W (int): Width of the original image.
        h (int): Height of each sub-image.
        w (int): Width of each sub-image.

    Returns:
        torch.Tensor: A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
    """
    batch_size, image_seq_len, channels = latents.shape
    assert H % h == 0 and W % w == 0, "H and W must be divisible by h and w respectively."
    
    # Compute downsampling factor
    total_pixels = H * W
    downsampling_factor = total_pixels // image_seq_len

    # Check if downsampling factor is a perfect square
    downsample_ratio = int(math.sqrt(downsampling_factor))
    if downsample_ratio * downsample_ratio != downsampling_factor:
        raise ValueError(f"The downsampling ratio cannot be determined. Image pixels {total_pixels} and sequence length {image_seq_len} do not match.")
    
    # Calculate latent dimensions
    latent_H = H // downsample_ratio
    latent_W = W // downsample_ratio
    latent_h = h // downsample_ratio
    latent_w = w // downsample_ratio
    
    # Match check
    assert latent_H * latent_W == image_seq_len, f"Calculated latent dimensions {latent_H}x{latent_W} do not match sequence length {image_seq_len}"
    
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
    """
    Merge sub-latents back into the original latents tensor.
    Args:
        sub_latents (torch.Tensor): A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
        H (int): Height of the original image.
        W (int): Width of the original image.
        h (int): Height of each sub-image.
        w (int): Width of each sub-image.
    Returns:
        torch.Tensor: The merged latents tensor of shape (B, seq_len, C).
    """
    batch_size, rows, cols, sub_seq_len, channels = sub_latents.shape
    
    vae_scale_factor = int(math.sqrt(h * w // sub_seq_len))
    # Calculate latent dimensions using the explicit parameters
    latent_h = h // vae_scale_factor
    latent_w = w // vae_scale_factor
    latent_H = H // vae_scale_factor
    latent_W = W // vae_scale_factor
    
    # Verify dimensions match
    assert latent_h * latent_w == sub_seq_len, f"sub_seq_len {sub_seq_len} does not match calculated sub-latent size {latent_h}x{latent_w}"
    assert rows * cols == (latent_H // latent_h) * (latent_W // latent_w), f"Grid size {rows}x{cols} does not match expected grid size"
    
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

def to_broadcast_tensor(value : Union[int, float, List[int], List[float], torch.Tensor], ref_tensor : torch.Tensor) -> torch.Tensor:
    """
    Convert a scalar, list, or tensor to a tensor that can be broadcasted with ref_tensor.
    The returned tensor will have shape (batch_size, 1, 1, ..., 1) where batch_size is the first dimension of ref_tensor,
    and the number of trailing singleton dimensions is equal to the number of dimensions in ref_tensor minus one.
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



def is_tensor_list(tensor_list: List[torch.Tensor]) -> bool:
    """
    Check if the input is a list of torch Tensors.
    Args:
        tensor_list (List[torch.Tensor]): list to check
    Returns:
        bool: True if all elements are torch Tensors, False otherwise
    """
    return isinstance(tensor_list, list) and all(isinstance(t, torch.Tensor) for t in tensor_list)