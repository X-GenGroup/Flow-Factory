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

"""Wan image-to-video condition realization shared by offline and rollout paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image import is_image, is_image_batch, is_multi_image_batch, standardize_image_batch
from ...utils.media import pixels_to_unit, unit_to_signed_unit_pixels
from ..condition_state import PreparedConditionState
from ..configured_image_output import retrieve_vae_latents
from ._output import configured_wan_video_output_geometry, normalize_wan_video_latents

WanConditionImageRows = Tuple[Tuple[Image.Image, ...], ...]


@dataclass(frozen=True, slots=True)
class WanI2VConditionTensors:
    """Hold the transformer condition channels and optional expanded-time mask."""

    condition: torch.Tensor
    first_frame_mask: Optional[torch.Tensor]


def normalize_wan_i2v_image_rows(
    images: Any,
    *,
    expected_batch_size: Optional[int] = None,
) -> WanConditionImageRows:
    """Normalize first/optional-last images while preserving within-sample order."""
    if expected_batch_size is not None and (
        type(expected_batch_size) is not int or expected_batch_size <= 0
    ):
        raise ValueError(
            "Wan I2V expected_batch_size must be a positive integer or None, "
            f"received {expected_batch_size!r}"
        )

    rows: list[tuple[Image.Image, ...]]
    if is_multi_image_batch(images):
        rows = []
        if isinstance(images, (torch.Tensor, np.ndarray)):
            iterable = list(images)
        else:
            iterable = images
        for row_index, row in enumerate(iterable):
            if isinstance(row, list) and not row:
                raise ValueError(f"Wan I2V sample {row_index} requires at least one input image")
            standardized = standardize_image_batch(row, output_type="pil")
            rows.append(tuple(standardized))
    elif is_image(images):
        rows = [(standardize_image_batch(images, output_type="pil")[0],)]
    elif is_image_batch(images):
        standardized = tuple(standardize_image_batch(images, output_type="pil"))
        if expected_batch_size == 1 and len(standardized) <= 2:
            rows = [standardized]
        else:
            rows = [(image,) for image in standardized]
    else:
        raise TypeError(
            "Wan I2V images must be one image, an image batch, or a multi-image batch, "
            f"received {type(images).__name__}"
        )

    if expected_batch_size is not None and len(rows) != expected_batch_size:
        raise ValueError(
            "Wan I2V condition-image batch size mismatch: "
            f"expected {expected_batch_size}, received {len(rows)}"
        )
    for row_index, row in enumerate(rows):
        if len(row) not in (1, 2):
            raise ValueError(
                "Wan I2V requires one first-frame image and at most one optional "
                f"last-frame image per sample, received {len(row)} at sample {row_index}"
            )
    return tuple(rows)


def append_wan_i2v_last_images(
    rows: WanConditionImageRows,
    last_images: Any,
) -> WanConditionImageRows:
    """Bind a legacy separate last-image argument to normalized first-frame rows."""
    if last_images is None:
        return rows
    if any(len(row) != 1 for row in rows):
        raise ValueError(
            "Wan I2V last_image cannot be combined with image rows that already contain "
            "an optional last frame"
        )
    normalized_last = normalize_wan_i2v_image_rows(
        last_images,
        expected_batch_size=len(rows),
    )
    if any(len(row) != 1 for row in normalized_last):
        raise ValueError("Wan I2V last_image must provide exactly one image per sample")
    return tuple((row[0], last_row[0]) for row, last_row in zip(rows, normalized_last))


def preprocess_wan_i2v_image_rows(
    adapter: Any,
    rows: WanConditionImageRows,
    *,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the official Wan video processor to ordered first/last images."""
    first_images = [row[0] for row in rows]
    first = adapter.pipeline.video_processor.preprocess(
        first_images,
        height=height,
        width=width,
    )
    if not isinstance(first, torch.Tensor):
        raise TypeError(
            "Wan video_processor.preprocess must return torch.Tensor, "
            f"received {type(first).__name__}"
        )
    first = first.to(device=adapter.device, dtype=torch.float32)

    has_last = [len(row) == 2 for row in rows]
    if any(has_last) and not all(has_last):
        raise ValueError("Wan I2V batches cannot mix first-only and first/last conditions")
    if not any(has_last):
        return first, None

    last = adapter.pipeline.video_processor.preprocess(
        [row[1] for row in rows],
        height=height,
        width=width,
    )
    if not isinstance(last, torch.Tensor):
        raise TypeError(
            "Wan video_processor.preprocess must return torch.Tensor, "
            f"received {type(last).__name__}"
        )
    return first, last.to(device=adapter.device, dtype=torch.float32)


def restore_wan_i2v_condition_pixels(
    condition_images: Any,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Restore Arrow-cached per-sample ``[0, 1]`` first/last pixels as ``[-1, 1]`` VAE input."""
    rows: list[torch.Tensor]
    if isinstance(condition_images, torch.Tensor):
        if condition_images.ndim == 5:
            if condition_images.shape[0] != batch_size:
                raise ValueError(
                    "Wan cached condition_images batch mismatch: "
                    f"expected {batch_size}, received {condition_images.shape[0]}"
                )
            rows = list(condition_images.unbind(0))
        elif condition_images.ndim == 4 and batch_size == 1:
            rows = [condition_images]
        elif condition_images.ndim == 4 and condition_images.shape[0] == batch_size:
            rows = [value.unsqueeze(0) for value in condition_images.unbind(0)]
        else:
            raise ValueError(
                "Wan cached condition_images tensor must be BNCHW or one sample's NCHW, "
                f"received shape {tuple(condition_images.shape)}"
            )
    elif isinstance(condition_images, Sequence) and not isinstance(condition_images, (str, bytes)):
        if len(condition_images) != batch_size:
            raise ValueError(
                "Wan cached condition_images sequence batch mismatch: "
                f"expected {batch_size}, received {len(condition_images)}"
            )
        rows = []
        for index, value in enumerate(condition_images):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "Wan cached condition_images entries must be tensors, "
                    f"received {type(value).__name__} at sample {index}"
                )
            rows.append(value.unsqueeze(0) if value.ndim == 3 else value)
    else:
        raise TypeError(
            "Wan cached condition_images must be a tensor or per-sample tensor sequence, "
            f"received {type(condition_images).__name__}"
        )

    expected_tail = (3, height, width)
    image_counts = []
    for index, row in enumerate(rows):
        if row.ndim != 4 or row.shape[0] not in (1, 2) or tuple(row.shape[1:]) != expected_tail:
            raise ValueError(
                "Wan cached condition_images must contain one first and optional last "
                f"pixel tensor shaped {expected_tail}; received {tuple(row.shape)} "
                f"at sample {index}"
            )
        image_counts.append(row.shape[0])
        rows[index] = pixels_to_unit(row, source=f"Wan cached condition_images[{index}]")
    if any(count != image_counts[0] for count in image_counts):
        raise ValueError("Wan I2V batches cannot mix first-only and first/last conditions")

    first = torch.stack([row[0] for row in rows], dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    if image_counts[0] == 1:
        return unit_to_signed_unit_pixels(first), None
    last = torch.stack([row[1] for row in rows], dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    return unit_to_signed_unit_pixels(first), unit_to_signed_unit_pixels(last)


def prepare_wan_i2v_condition_tensors(
    adapter: Any,
    image: torch.Tensor,
    *,
    height: int,
    width: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    last_image: Optional[torch.Tensor] = None,
) -> WanI2VConditionTensors:
    """Encode ordered input frames with posterior mode and build Wan condition channels.

    Args:
        adapter: Realized Wan I2V adapter owning the pipeline and VAE.
        image: First-frame pixels shaped ``(B, 3, H, W)``.
        height: Configured output height.
        width: Configured output width.
        num_frames: Configured output frame count.
        dtype: Floating dtype for returned condition tensors.
        device: Device used for VAE encoding and returned tensors.
        last_image: Optional last-frame pixels with the same shape as ``image``.

    Returns:
        Encoded condition channels and, for expanded-timestep checkpoints, the separate
        first-frame mask.

    Raises:
        TypeError: If the requested output dtype or VAE dtype is invalid.
        ValueError: If input tensors, pixel/latent geometry, endpoint support, or temporal
            divisibility are incompatible with the realized checkpoint.
    """
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError(f"Wan I2V condition dtype must be floating, received {dtype!r}")
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(
            "Wan I2V first-frame pixels must be rank-4 BCHW, "
            f"received {type(image).__name__} with shape {getattr(image, 'shape', None)}"
        )
    batch_size = image.shape[0]
    expected_pixels = (batch_size, 3, height, width)
    if tuple(image.shape) != expected_pixels:
        raise ValueError(
            f"Wan I2V first-frame pixels must have shape {expected_pixels}, "
            f"received {tuple(image.shape)}"
        )
    if last_image is not None:
        if adapter.pipeline.config.expand_timesteps:
            raise ValueError(
                "Wan I2V expand_timesteps does not support an optional last-frame image; "
                "Diffusers would otherwise ignore it"
            )
        if num_frames < 2:
            raise ValueError(
                "Wan I2V optional last-frame conditioning requires num_frames >= 2, "
                f"received {num_frames}"
            )
        if not isinstance(last_image, torch.Tensor) or tuple(last_image.shape) != expected_pixels:
            raise ValueError(
                f"Wan I2V last-frame pixels must have shape {expected_pixels}, "
                f"received {type(last_image).__name__} with shape "
                f"{getattr(last_image, 'shape', None)}"
            )

    temporal_scale = adapter.pipeline.vae_scale_factor_temporal
    spatial_scale = adapter.pipeline.vae_scale_factor_spatial
    if (num_frames - 1) % temporal_scale:
        raise ValueError(
            "Wan I2V num_frames must satisfy "
            f"(num_frames - 1) % {temporal_scale} == 0, received {num_frames}"
        )
    num_latent_frames = (num_frames - 1) // temporal_scale + 1
    latent_height = height // spatial_scale
    latent_width = width // spatial_scale

    image = image.unsqueeze(2)
    if adapter.pipeline.config.expand_timesteps:
        video_condition = image
    elif last_image is None:
        video_condition = torch.cat(
            [
                image,
                image.new_zeros(batch_size, image.shape[1], num_frames - 1, height, width),
            ],
            dim=2,
        )
    else:
        video_condition = torch.cat(
            [
                image,
                image.new_zeros(batch_size, image.shape[1], num_frames - 2, height, width),
                last_image.unsqueeze(2),
            ],
            dim=2,
        )

    vae = adapter.vae
    vae_dtype = getattr(vae, "dtype", None)
    if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
        raise TypeError(
            "Wan I2V condition encoder expected VAE to expose a floating dtype, "
            f"received {vae_dtype!r}"
        )
    encoded = vae.encode(video_condition.to(device=device, dtype=vae_dtype))
    latent_condition = retrieve_vae_latents(
        encoded,
        sample_mode="argmax",
        source="Wan input video condition",
    ).to(device=device, dtype=dtype)
    latent_condition = normalize_wan_video_latents(adapter, latent_condition)

    # The condition VAE time axis belongs to the encoded source. Expanded-timestep checkpoints
    # encode one frame and broadcast it across the target via first_frame_mask, so it must not be
    # validated against the rollout's num_latent_frames.
    condition_latent_frames = (video_condition.shape[2] - 1) // temporal_scale + 1
    expected_latents = (
        batch_size,
        vae.config.z_dim,
        condition_latent_frames,
        latent_height,
        latent_width,
    )
    if tuple(latent_condition.shape) != expected_latents:
        raise ValueError(
            "Wan I2V condition latent geometry mismatch: "
            f"expected {expected_latents}, received {tuple(latent_condition.shape)}"
        )

    if adapter.pipeline.config.expand_timesteps:
        first_frame_mask = torch.ones(
            batch_size,
            1,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
            device=device,
        )
        first_frame_mask[:, :, 0] = 0
        return WanI2VConditionTensors(latent_condition, first_frame_mask)

    mask = torch.ones(
        batch_size,
        1,
        num_frames,
        latent_height,
        latent_width,
        dtype=dtype,
        device=device,
    )
    if last_image is None:
        mask[:, :, 1:] = 0
    else:
        mask[:, :, 1:-1] = 0
    first = torch.repeat_interleave(mask[:, :, :1], dim=2, repeats=temporal_scale)
    mask = torch.cat([first, mask[:, :, 1:]], dim=2)
    mask = mask.view(
        batch_size,
        -1,
        temporal_scale,
        latent_height,
        latent_width,
    ).transpose(1, 2)
    condition = torch.cat([mask, latent_condition], dim=1)
    return WanI2VConditionTensors(condition, None)


def normalize_wan_image_embeds(image_embeds: Any, *, batch_size: int) -> torch.Tensor:
    """Pack cached per-sample CLIP embeddings into Diffusers' forward layout."""
    if isinstance(image_embeds, torch.Tensor):
        if image_embeds.ndim == 2:
            if batch_size != 1:
                raise ValueError(
                    "unbatched Wan image_embeds require batch_size=1, " f"received {batch_size}"
                )
            return image_embeds.unsqueeze(0)
        if image_embeds.ndim == 3:
            return image_embeds
        if image_embeds.ndim == 4:
            if image_embeds.shape[0] != batch_size:
                raise ValueError(
                    "Wan image_embeds batch mismatch: "
                    f"expected {batch_size}, received {image_embeds.shape[0]}"
                )
            return image_embeds.flatten(0, 1)
        raise ValueError(
            "Wan image_embeds must be rank 2, 3, or 4, "
            f"received shape {tuple(image_embeds.shape)}"
        )
    if isinstance(image_embeds, Sequence) and not isinstance(image_embeds, (str, bytes)):
        if len(image_embeds) != batch_size:
            raise ValueError(
                "Wan image_embeds sequence batch mismatch: "
                f"expected {batch_size}, received {len(image_embeds)}"
            )
        packed = []
        for index, value in enumerate(image_embeds):
            if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
                raise ValueError(
                    "Wan image_embeds sequence entries must be rank-2 or rank-3 tensors, "
                    f"received {type(value).__name__} with shape "
                    f"{getattr(value, 'shape', None)} at sample {index}"
                )
            packed.append(value.unsqueeze(0) if value.ndim == 2 else value)
        return torch.cat(packed, dim=0)
    raise TypeError(
        "Wan image_embeds must be a tensor or per-sample tensor sequence, "
        f"received {type(image_embeds).__name__}"
    )


def split_wan_image_embeds(
    image_embeds: torch.Tensor,
    image_counts: Sequence[int],
) -> Tuple[torch.Tensor, ...]:
    """Split Diffusers' packed CLIP layout back into per-sample replay tensors."""
    if not isinstance(image_embeds, torch.Tensor) or image_embeds.ndim != 3:
        raise ValueError(
            "packed Wan image_embeds must be rank-3, "
            f"received {type(image_embeds).__name__} with shape "
            f"{getattr(image_embeds, 'shape', None)}"
        )
    counts = tuple(image_counts)
    if not counts or any(type(count) is not int or count not in (1, 2) for count in counts):
        raise ValueError(
            "Wan image_counts must contain one or two images per sample, " f"received {counts!r}"
        )
    if sum(counts) != image_embeds.shape[0]:
        raise ValueError(
            "packed Wan image_embeds count mismatch: "
            f"expected {sum(counts)}, received {image_embeds.shape[0]}"
        )
    per_sample = []
    offset = 0
    for count in counts:
        value = image_embeds[offset : offset + count]
        per_sample.append(value[0] if count == 1 else value)
        offset += count
    return tuple(per_sample)


@dataclass(frozen=True, slots=True)
class WanI2VConditionStatePreparer:
    """Realize input-owned Wan VAE conditions once per offline batch."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedConditionState:
        """Bind cached images to configured geometry with deterministic VAE mode."""
        del generator
        if "condition_images" not in condition:
            raise ValueError(
                "Wan I2V cached condition is missing preprocessed 'condition_images'; "
                "rebuild the input-condition cache with this adapter version"
            )
        height, width, num_frames, _ = configured_wan_video_output_geometry(self.adapter)
        first, last = restore_wan_i2v_condition_pixels(
            condition["condition_images"],
            batch_size=1,
            height=height,
            width=width,
            device=self.adapter.device,
        )
        realized = prepare_wan_i2v_condition_tensors(
            self.adapter,
            first,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.adapter.device,
            last_image=last,
        )

        cached_condition = dict(condition)
        cached_condition.pop("condition_images")
        cached_condition.pop("images", None)
        forward_context: dict[str, Any] = {"latent_condition": realized.condition}
        image_embeds = cached_condition.pop("image_embeds", None)
        if image_embeds is not None:
            packed_image_embeds = normalize_wan_image_embeds(
                image_embeds,
                batch_size=first.shape[0],
            )
            expected_image_embeds = first.shape[0] * (2 if last is not None else 1)
            if packed_image_embeds.shape[0] != expected_image_embeds:
                raise ValueError(
                    "Wan cached image_embeds count disagrees with condition_images: "
                    f"expected {expected_image_embeds}, received "
                    f"{packed_image_embeds.shape[0]}"
                )
            forward_context["image_embeds"] = packed_image_embeds

        output_context: dict[str, Any] = {}
        if realized.first_frame_mask is not None:
            forward_context["first_frame_mask"] = realized.first_frame_mask
            output_context["first_frame_mask"] = realized.first_frame_mask
        return PreparedConditionState(
            condition=cached_condition,
            forward_context=forward_context,
            output_context=output_context,
        )


__all__ = [
    "WanConditionImageRows",
    "WanI2VConditionStatePreparer",
    "WanI2VConditionTensors",
    "append_wan_i2v_last_images",
    "normalize_wan_i2v_image_rows",
    "normalize_wan_image_embeds",
    "prepare_wan_i2v_condition_tensors",
    "preprocess_wan_i2v_image_rows",
    "restore_wan_i2v_condition_pixels",
    "split_wan_image_embeds",
]
