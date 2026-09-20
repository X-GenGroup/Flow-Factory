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

"""Independent Qwen-Image 2.1 VAE and offline-output encoding helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

import torch

from ..configured_image_output import EncodedImageTensor

QWEN_IMAGE_21_DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"


def _retrieve_vae_latents(
    encoder_output: Any,
    *,
    sample_mode: Literal["sample", "argmax"],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Retrieve one explicit posterior realization from a Diffusers VAE output.

    Modified from
    ``diffusers.pipelines.qwenimage21.pipeline_qwenimage21.retrieve_latents``
    at :data:`QWEN_IMAGE_21_DIFFUSERS_COMMIT`. The explicit mode is kept at
    the Flow-Factory semantic boundary: conditions use ``argmax`` while
    supervised targets use ``sample``.
    """
    latent_dist = getattr(encoder_output, "latent_dist", None)
    if latent_dist is not None:
        if sample_mode == "sample":
            return latent_dist.sample(generator)
        return latent_dist.mode()
    latents = getattr(encoder_output, "latents", None)
    if isinstance(latents, torch.Tensor):
        return latents
    raise AttributeError("Qwen-Image 2.1 VAE output does not expose latent_dist or latents")


def _channel_statistics(
    values: Any,
    channels: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
    if tensor.numel() != channels:
        raise ValueError(
            f"Qwen-Image 2.1 VAE {name} expected {channels} values, " f"received {tensor.numel()}"
        )
    return tensor.reshape(1, channels, 1, 1, 1)


def encode_qwen_image_21_vae(
    adapter: Any,
    video_values: torch.Tensor,
    *,
    sample_mode: Literal["sample", "argmax"],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode and normalize RGBA image-as-video tensors with the 2.1 VAE."""
    latents = _retrieve_vae_latents(
        adapter.vae.encode(video_values),
        sample_mode=sample_mode,
        generator=generator,
    )
    channels = latents.shape[1]
    means = _channel_statistics(adapter.vae.config.latents_mean, channels, latents, "latents_mean")
    stds = _channel_statistics(adapter.vae.config.latents_std, channels, latents, "latents_std")
    if torch.any(stds <= 0):
        raise ValueError("Qwen-Image 2.1 VAE latents_std must be positive")
    return (latents - means) / stds


def _shape_tuple(value: Any, source: str) -> tuple[int, int, int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise TypeError(f"{source} must be a length-3 sequence, received {value!r}")
    shape = tuple(int(item) for item in value)
    if any(item <= 0 for item in shape):
        raise ValueError(f"{source} must contain positive dimensions, received {shape!r}")
    return shape


def normalize_condition_img_shapes(
    value: Any,
    *,
    batch_size: int,
) -> list[list[tuple[int, int, int]]]:
    """Normalize cached ragged condition shapes without borrowing old Qwen helpers."""
    if value is None:
        return [[] for _ in range(batch_size)]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if (
        batch_size == 1
        and isinstance(value, Sequence)
        and (not value or (isinstance(value[0], Sequence) and len(value[0]) == 3))
    ):
        value = [value]
    if not isinstance(value, Sequence) or len(value) != batch_size:
        raise ValueError(
            "Qwen-Image 2.1 condition_img_shapes batch mismatch: "
            f"expected {batch_size}, received {value!r}"
        )
    normalized = []
    for sample_index, shapes in enumerate(value):
        if isinstance(shapes, torch.Tensor):
            shapes = shapes.detach().cpu().tolist()
        if not isinstance(shapes, Sequence) or isinstance(shapes, (str, bytes)):
            raise TypeError(
                f"condition_img_shapes[{sample_index}] must be a sequence, received {shapes!r}"
            )
        normalized.append(
            [
                _shape_tuple(shape, f"condition_img_shapes[{sample_index}][{shape_index}]")
                for shape_index, shape in enumerate(shapes)
            ]
        )
    return normalized


def encode_qwen_image_21_output(
    adapter: Any,
    pixel_values: torch.Tensor,
    condition: Mapping[str, Any],
    generator: Optional[torch.Generator],
) -> EncodedImageTensor:
    """Encode configured-resolution targets with target-last 2.1 metadata."""
    video_values = pixel_values.unsqueeze(2)
    latents = encode_qwen_image_21_vae(
        adapter,
        video_values,
        sample_mode="sample",
        generator=generator,
    )
    batch_size, channels = latents.shape[:2]
    latent_height, latent_width = latents.shape[-2:]
    packed = adapter._pack_latents(
        latents,
        batch_size=batch_size,
        num_channels_latents=channels,
        height=latent_height,
        width=latent_width,
    )
    condition_shapes = normalize_condition_img_shapes(
        condition.get("condition_img_shapes"),
        batch_size=batch_size,
    )
    target_shape = (1, latent_height, latent_width)
    img_shapes = [shapes + [target_shape] for shapes in condition_shapes]
    return EncodedImageTensor(
        latents=packed,
        forward_context={"img_shapes": img_shapes},
        decode_context={},
    )


__all__ = [
    "QWEN_IMAGE_21_DIFFUSERS_COMMIT",
    "encode_qwen_image_21_output",
    "encode_qwen_image_21_vae",
    "normalize_condition_img_shapes",
]
