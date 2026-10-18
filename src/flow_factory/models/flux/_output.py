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

"""Role-neutral VAE encoding shared by the FLUX pipeline variants."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, List, Optional, Tuple

import torch

from ..configured_image_output import (
    EncodedImageTensor,
    VAELatentSampleMode,
    encode_shift_scale_vae_image,
    retrieve_vae_latents,
)


def encode_flux1_vae_image(
    adapter: Any,
    pixel_values: torch.Tensor,
    *,
    sample_mode: VAELatentSampleMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply explicit posterior selection shared by FLUX.1 roles.

    Args:
        adapter: FLUX.1 adapter exposing the canonical VAE.
        pixel_values: Preprocessed BCHW image tensor.
        sample_mode: ``sample`` for targets or the official condition selection.
        generator: Generator forwarded unchanged for posterior sampling.

    Returns:
        Shifted and scaled convolutional VAE latents.
    """
    return encode_shift_scale_vae_image(
        adapter,
        pixel_values,
        sample_mode=sample_mode,
        generator=generator,
    )


def prepare_flux1_output_latents(
    adapter: Any,
    pixel_values: torch.Tensor,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample and pack FLUX.1 target latents with their target-first IDs.

    Args:
        adapter: FLUX.1 adapter exposing the canonical VAE and pipeline helpers.
        pixel_values: Preprocessed BCHW target images.
        generator: Generator forwarded unchanged to target posterior sampling.

    Returns:
        Packed clean target latents and their unbatched position identifiers.
    """
    latents = encode_flux1_vae_image(
        adapter,
        pixel_values,
        sample_mode="sample",
        generator=generator,
    )
    if latents.ndim != 4:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.1 target VAE expected BCHW latents, "
            f"received shape {tuple(latents.shape)}"
        )
    batch_size, channels, latent_height, latent_width = latents.shape
    if latent_height % 2 or latent_width % 2:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.1 target packing expected even latent height/width, "
            f"received {(latent_height, latent_width)}"
        )
    packed = adapter.pipeline._pack_latents(
        latents,
        batch_size,
        channels,
        latent_height,
        latent_width,
    )
    target_ids = adapter.pipeline._prepare_latent_image_ids(
        batch_size,
        latent_height // 2,
        latent_width // 2,
        adapter.device,
        packed.dtype,
    )
    return packed, target_ids


def patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Convert 4D VAE latents ``(B, C, H, W)`` to 2x2 patchified ``(B, C*4, H//2, W//2)``."""
    if latents.ndim != 4:
        raise ValueError(
            f"expected 4D BCHW latents for 2x2 patchify, received shape {tuple(latents.shape)}"
        )
    batch_size, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError(
            f"expected even latent height/width for 2x2 patchify, received {(height, width)}"
        )
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, channels * 4, height // 2, width // 2)


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Convert 2x2 patchified latents ``(B, C*4, H, W)`` back to ``(B, C, H*2, W*2)``."""
    if latents.ndim != 4:
        raise ValueError(
            f"expected 4D BCHW latents for 2x2 unpatchify, received shape {tuple(latents.shape)}"
        )
    batch_size, channels, height, width = latents.shape
    if channels % 4:
        raise ValueError(
            f"expected channel count divisible by 4 for 2x2 unpatchify, received {channels}"
        )
    latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def pack_flux3_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack 4D patchified latents ``(B, C, H, W)`` into tokens ``(B, H*W, C)``."""
    if latents.ndim != 4:
        raise ValueError(
            f"expected 4D BCHW latents for FLUX.3 packing, received shape {tuple(latents.shape)}"
        )
    return latents.flatten(2).transpose(1, 2)


def unpack_flux3_latents(
    latents: torch.Tensor,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Unpack FLUX.3 latent tokens ``(B, H*W, C)`` onto a 2D grid ``(B, C, H, W)``."""
    if latents.ndim != 3:
        raise ValueError(
            f"expected 3D (B, seq_len, C) latents for FLUX.3 unpacking, "
            f"received shape {tuple(latents.shape)}"
        )
    batch_size, seq_len, channels = latents.shape
    if latent_height <= 0 or latent_width <= 0 or latent_height * latent_width != seq_len:
        raise ValueError(
            f"FLUX.3 latent token count {seq_len} does not match "
            f"latent grid {latent_height} x {latent_width}"
        )
    return latents.view(batch_size, latent_height, latent_width, channels).permute(0, 3, 1, 2)


def prepare_flux3_image_ids(
    latent_height: int,
    latent_width: int,
    *,
    image_rope_scale: float = 0.5,
    l_coord: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build 4D position IDs ``(t, h, w, l)`` of shape ``(H*W, 4)`` for FLUX.3 image tokens."""
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError(
            f"expected positive FLUX.3 latent grid dimensions, received {(latent_height, latent_width)}"
        )
    t_coords = torch.arange(1)
    h_coords = torch.arange(latent_height)
    w_coords = torch.arange(latent_width)
    l_coords = torch.tensor([l_coord])
    ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords).to(dtype=torch.float32)
    ids[:, 1:3] = ids[:, 1:3] * float(image_rope_scale)
    if device is not None:
        ids = ids.to(device)
    return ids


def encode_flux2_output_images(
    adapter: Any,
    pixel_values: torch.Tensor,
    generator: Optional[torch.Generator],
) -> EncodedImageTensor:
    """Sample the FLUX.2 target posterior, normalize, and pack tokens.

    Args:
        adapter: FLUX.2 adapter exposing pipeline packing primitives.
        pixel_values: Preprocessed BCHW target images.
        generator: Generator forwarded unchanged to target posterior sampling.

    Returns:
        Packed clean latents with their position identifiers.
    """
    latents = encode_flux2_vae_image(
        adapter,
        pixel_values,
        sample_mode="sample",
        generator=generator,
    )
    latent_ids = adapter.pipeline._prepare_latent_ids(latents).to(adapter.device)
    packed = adapter.pipeline._pack_latents(latents)
    return EncodedImageTensor(
        latents=packed,
        forward_context={"latent_ids": latent_ids},
        decode_context={"latent_ids": latent_ids},
    )


def encode_flux3_output_images(
    adapter: Any,
    pixel_values: torch.Tensor,
    generator: Optional[torch.Generator],
    *,
    image_rope_scale: float = 0.5,
) -> EncodedImageTensor:
    """Sample the FLUX.3 target posterior, normalize, and pack tokens with scaled RoPE IDs."""
    latents = encode_flux2_vae_image(
        adapter,
        pixel_values,
        sample_mode="sample",
        generator=generator,
    )
    batch_size = latents.shape[0]
    latent_height, latent_width = latents.shape[-2:]
    latent_ids = (
        prepare_flux3_image_ids(
            latent_height,
            latent_width,
            image_rope_scale=image_rope_scale,
            l_coord=0,
            device=adapter.device,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    packed = pack_flux3_latents(latents).to(adapter.device)
    return EncodedImageTensor(
        latents=packed,
        forward_context={"latent_ids": latent_ids},
        decode_context={"latent_ids": latent_ids},
    )


def encode_flux2_vae_image(
    adapter: Any,
    pixel_values: torch.Tensor,
    *,
    sample_mode: VAELatentSampleMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the exact FLUX.2/FLUX.3 transform shared by condition and target roles.

    Args:
        adapter: FLUX.2 or FLUX.3 adapter exposing the canonical VAE and patchify primitive.
        pixel_values: Preprocessed BCHW image tensor.
        sample_mode: Explicit posterior selection for the calling pipeline role.
        generator: Generator forwarded unchanged for posterior sampling.

    Returns:
        Patchified and BatchNorm-normalized convolutional latents.
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 VAE expected BCHW input, "
            f"received shape {tuple(pixel_values.shape)}"
        )
    vae = adapter.vae
    latents = retrieve_vae_latents(
        vae.encode(pixel_values),
        sample_mode=sample_mode,
        generator=generator,
        source=f"{type(adapter).__name__} FLUX.2 VAE encode",
    )
    pipeline_patchify = getattr(getattr(adapter, "pipeline", None), "_patchify_latents", None)
    if callable(pipeline_patchify):
        latents = pipeline_patchify(latents)
    else:
        latents = patchify_flux2_latents(latents)
    if latents.ndim != 4:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 patchify expected BCHW output, "
            f"received shape {tuple(latents.shape)}"
        )

    batch_norm = getattr(vae, "bn", None)
    running_mean = getattr(batch_norm, "running_mean", None)
    running_var = getattr(batch_norm, "running_var", None)
    if not isinstance(running_mean, torch.Tensor) or not isinstance(running_var, torch.Tensor):
        raise TypeError(
            f"{type(adapter).__name__} FLUX.2 VAE must expose BatchNorm running_mean/running_var"
        )
    channels = latents.shape[1]
    if running_mean.ndim != 1 or running_var.ndim != 1:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 BatchNorm statistics must be rank 1, "
            f"received mean={tuple(running_mean.shape)}, var={tuple(running_var.shape)}"
        )
    if running_mean.numel() != channels or running_var.numel() != channels:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 BatchNorm expected {channels} values, "
            f"received mean={running_mean.numel()}, var={running_var.numel()}"
        )
    eps = getattr(getattr(vae, "config", None), "batch_norm_eps", None)
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError(
            f"{type(adapter).__name__} FLUX.2 VAE config expected numeric batch_norm_eps, "
            f"received {type(eps).__name__}: {eps!r}"
        )
    if not math.isfinite(float(eps)) or eps < 0:
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 VAE batch_norm_eps must be finite and >= 0, "
            f"received {eps!r}"
        )
    mean_values = running_mean.view(1, -1, 1, 1)
    variance_values = running_var.view(1, -1, 1, 1)
    if not torch.isfinite(mean_values).all() or not torch.isfinite(variance_values).all():
        raise ValueError(f"{type(adapter).__name__} FLUX.2 BatchNorm statistics must be finite")
    if torch.any(variance_values + float(eps) <= 0):
        raise ValueError(
            f"{type(adapter).__name__} FLUX.2 BatchNorm variance plus epsilon must be positive"
        )
    mean = mean_values.to(device=latents.device, dtype=latents.dtype)
    std = torch.sqrt(variance_values + float(eps)).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    return (latents - mean) / std


def prepare_flux2_condition_latents(
    adapter: Any,
    images: List[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose FLUX.2 condition tokens around the shared VAE transform.

    Args:
        adapter: FLUX.2 adapter exposing condition packing primitives.
        images: One or more preprocessed BCHW condition tensors for one sample.
        batch_size: Number of prompt rows that reuse the condition sequence.
        device: Target device for condition tensors and identifiers.
        dtype: VAE input dtype.

    Returns:
        Packed condition latents and their repeated position identifiers.
    """
    if not images:
        raise ValueError(f"{type(adapter).__name__} requires at least one condition image")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError(
            f"{type(adapter).__name__} condition batch_size must be a positive int, "
            f"received {batch_size!r}"
        )
    image_latents = [
        encode_flux2_vae_image(
            adapter,
            image.to(device=device, dtype=dtype),
            sample_mode="argmax",
        )
        for image in images
    ]
    image_latent_ids = adapter.pipeline._prepare_image_ids(image_latents)
    packed_latents = [adapter.pipeline._pack_latents(latent).squeeze(0) for latent in image_latents]
    packed = torch.cat(packed_latents, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
    image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1).to(device)
    return packed, image_latent_ids


def prepare_flux3_condition_latents(
    adapter: Any,
    images: List[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    image_rope_scale: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose FLUX.3 reference-image tokens and ``l = -(1 + idx)`` position IDs.

    Args:
        adapter: FLUX.3 adapter exposing the canonical VAE.
        images: One or more preprocessed BCHW condition tensors for one sample.
        batch_size: Number of prompt rows that reuse the condition sequence.
        device: Target device for condition tensors and identifiers.
        dtype: Target compute dtype for packed condition tokens.
        image_rope_scale: Spatial RoPE coordinate multiplier (0.5 in FLUX.3).

    Returns:
        Packed condition latents ``(B, cond_seq_len, C)`` and position IDs ``(B, cond_seq_len, 4)``.
    """
    if not images:
        raise ValueError(f"{type(adapter).__name__} requires at least one condition image")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError(
            f"{type(adapter).__name__} condition batch_size must be a positive int, "
            f"received {batch_size!r}"
        )
    vae_dtype = getattr(adapter.vae, "dtype", dtype)
    cond_tokens: List[torch.Tensor] = []
    cond_ids: List[torch.Tensor] = []
    for index, image in enumerate(images):
        encoded = encode_flux2_vae_image(
            adapter,
            image.to(device=device, dtype=vae_dtype),
            sample_mode="argmax",
        )
        latent_height, latent_width = encoded.shape[-2:]
        cond_tokens.append(pack_flux3_latents(encoded).squeeze(0))
        cond_ids.append(
            prepare_flux3_image_ids(
                latent_height,
                latent_width,
                image_rope_scale=image_rope_scale,
                l_coord=-(1 + index),
                device=device,
            )
        )
    packed = torch.cat(cond_tokens, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
    packed = packed.to(device=device, dtype=dtype)
    image_latent_ids = torch.cat(cond_ids, dim=0).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    return packed, image_latent_ids


__all__ = [
    "encode_flux1_vae_image",
    "encode_flux2_output_images",
    "encode_flux2_vae_image",
    "encode_flux3_output_images",
    "pack_flux3_latents",
    "patchify_flux2_latents",
    "prepare_flux1_output_latents",
    "prepare_flux2_condition_latents",
    "prepare_flux3_condition_latents",
    "prepare_flux3_image_ids",
    "unpack_flux3_latents",
    "unpatchify_flux2_latents",
]
