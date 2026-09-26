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

"""Shared on-the-fly clean-video encoding for Wan video adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
import torch

from ...contracts import MediaType
from ...samples import LatentState
from ...utils.video import (
    decoded_video_to_unit_float,
    require_decoded_video_frames,
    require_finite_bcfhw_video,
)
from ..configured_image_output import retrieve_vae_latents
from ..output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)


def configured_wan_video_output_geometry(adapter: Any) -> Tuple[int, int, int, float]:
    """Return configured Wan geometry after exact latent-grid validation."""
    geometry = []
    for name in ("height", "width", "num_frames"):
        value = getattr(adapter.training_args, name, None)
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"Wan output geometry requires positive integer train.{name}, "
                f"received {value!r}"
            )
        geometry.append(value)
    frame_rate = getattr(adapter.training_args, "frame_rate", None)
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, Real):
        raise TypeError(
            "Wan output geometry requires finite positive train.frame_rate, "
            f"received {type(frame_rate).__name__}: {frame_rate!r}"
        )
    frame_rate = float(frame_rate)
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(
            "Wan output geometry requires finite positive train.frame_rate, "
            f"received {frame_rate!r}"
        )

    height, width, num_frames = geometry
    temporal_scale = adapter.pipeline.vae_scale_factor_temporal
    spatial_scale = adapter.pipeline.vae_scale_factor_spatial
    if (num_frames - 1) % temporal_scale:
        raise ValueError(
            "Wan output num_frames must satisfy "
            f"(num_frames - 1) % {temporal_scale} == 0, received {num_frames}"
        )
    transformer = (
        adapter.pipeline.transformer
        if adapter.pipeline.transformer is not None
        else adapter.pipeline.transformer_2
    )
    if transformer is None:
        raise RuntimeError("Wan output geometry requires one materialized transformer")
    patch_size = transformer.config.patch_size
    height_multiple = spatial_scale * patch_size[1]
    width_multiple = spatial_scale * patch_size[2]
    if height % height_multiple or width % width_multiple:
        raise ValueError(
            "Wan output height/width must be divisible by transformer latent-grid "
            f"multiples {(height_multiple, width_multiple)}, received {(height, width)}"
        )
    return height, width, num_frames, frame_rate


def resample_wan_output_video(
    video: np.ndarray,
    *,
    source_fps: Optional[float],
    target_frames: int,
    target_fps: float,
) -> np.ndarray:
    """Select deterministic nearest-time frames for configured target cadence."""
    video = require_decoded_video_frames(video, source="Wan decoded target video")
    if isinstance(source_fps, bool) or not isinstance(source_fps, Real):
        raise TypeError(
            "Wan target video requires source fps metadata, "
            f"received {type(source_fps).__name__}: {source_fps!r}"
        )
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Wan target video requires positive finite fps, got {source_fps!r}")
    indices = np.rint(np.arange(target_frames, dtype=np.float64) * source_fps / target_fps).astype(
        np.int64
    )
    if indices[-1] >= video.shape[0]:
        required_duration = (target_frames - 1) / target_fps
        available_duration = (video.shape[0] - 1) / source_fps
        raise ValueError(
            "Wan target video is too short for configured temporal geometry: "
            f"requires {required_duration:.6f}s, has {available_duration:.6f}s"
        )
    return np.ascontiguousarray(video[indices])


def normalize_wan_video_latents(adapter: Any, latents: torch.Tensor) -> torch.Tensor:
    """Apply the exact inverse of Wan's existing decode normalization."""
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError(
            "Wan VAE video latents must be rank-5 BCFHW, "
            f"received {type(latents).__name__} with shape "
            f"{getattr(latents, 'shape', None)}"
        )
    config = adapter.vae.config
    z_dim = config.z_dim
    if latents.shape[1] != z_dim:
        raise ValueError(
            f"Wan VAE video latent channels must equal z_dim={z_dim}, "
            f"received {latents.shape[1]}"
        )
    latents_mean = torch.as_tensor(
        config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, z_dim, 1, 1, 1)
    inverse_std = (
        torch.as_tensor(
            config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        )
        .reciprocal()
        .view(1, z_dim, 1, 1, 1)
    )
    return (latents - latents_mean) * inverse_std


def validate_wan_encoded_output_geometry(
    adapter: Any,
    media_batch: DecodedMediaBatch,
    condition: Mapping[str, Any],
    encoded: EncodedOutputState,
) -> None:
    """Require encoded signatures and decode metadata to match train geometry."""
    del condition
    height, width, num_frames, frame_rate = configured_wan_video_output_geometry(adapter)
    if len(encoded.geometry_signatures) != len(media_batch):
        raise ValueError(
            "Wan output codec must return one geometry signature per sample, "
            f"received {len(encoded.geometry_signatures)} for {len(media_batch)}"
        )
    for sample_index, signature in enumerate(encoded.geometry_signatures):
        geometry = signature.media[0]
        received = (
            geometry.height,
            geometry.width,
            geometry.frames,
            geometry.fps,
        )
        expected = (height, width, num_frames, frame_rate)
        if received != expected:
            raise ValueError(
                "Wan encoded output geometry disagrees with configured geometry for "
                f"sample {sample_index}: expected {expected}, received {received}"
            )
    expected_context = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
    }
    for name, expected in expected_context.items():
        if encoded.decode_context.get(name) != expected:
            raise ValueError(
                f"Wan decode_context {name!r} must equal {expected!r}, "
                f"received {encoded.decode_context.get(name)!r}"
            )


@dataclass(frozen=True, slots=True)
class WanVideoOutputCodec:
    """Encode configured Wan target videos without retaining pixels or latents."""

    adapter: Any
    bind_condition_active_mask: bool = False
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Preprocess, VAE-sample, and normalize one video per sample."""
        height, width, num_frames, frame_rate = configured_wan_video_output_geometry(self.adapter)
        first_frame_mask = None
        if self.bind_condition_active_mask:
            first_frame_mask = condition.get("first_frame_mask")
            expand_timesteps = bool(self.adapter.pipeline.config.expand_timesteps)
            if expand_timesteps and first_frame_mask is None:
                raise ValueError(
                    "Wan I2V expand_timesteps target encoding requires first_frame_mask "
                    "from the prepared input condition"
                )
            if not expand_timesteps and first_frame_mask is not None:
                raise ValueError(
                    "Wan I2V non-expanded target encoding must not receive first_frame_mask"
                )
            if first_frame_mask is not None:
                if not isinstance(first_frame_mask, torch.Tensor):
                    raise TypeError(
                        "Wan I2V first_frame_mask must be torch.Tensor, "
                        f"received {type(first_frame_mask).__name__}"
                    )
                if not torch.all((first_frame_mask == 0) | (first_frame_mask == 1)):
                    raise ValueError("Wan I2V first_frame_mask must contain only zero and one")
        videos = []
        for sample_index, candidate in enumerate(media_batch):
            if len(candidate) != 1:
                raise ValueError(
                    "Wan output codec expected one video per sample, "
                    f"received {len(candidate)} for sample {sample_index}"
                )
            media = candidate[0]
            videos.append(
                resample_wan_output_video(
                    media.payload,
                    source_fps=media.fps,
                    target_frames=num_frames,
                    target_fps=frame_rate,
                )
            )

        pixel_values = self.adapter.pipeline.video_processor.preprocess_video(
            [
                decoded_video_to_unit_float(video, source="Wan sampled target video")
                for video in videos
            ],
            height=height,
            width=width,
        )
        require_finite_bcfhw_video(
            pixel_values,
            source="Wan video_processor.preprocess_video",
            batch_size=len(videos),
            frames=num_frames,
            height=height,
            width=width,
        )

        vae = self.adapter.vae
        vae_dtype = getattr(vae, "dtype", None)
        if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
            raise TypeError(
                "Wan output codec expected VAE to expose a floating dtype, "
                f"received {vae_dtype!r}"
            )
        pixel_values = pixel_values.to(device=self.adapter.device, dtype=vae_dtype)
        encoded = vae.encode(pixel_values)
        latents = retrieve_vae_latents(
            encoded,
            sample_mode="sample",
            generator=generator,
            source="Wan target video",
        )
        latents = normalize_wan_video_latents(self.adapter, latents)

        temporal_scale = self.adapter.pipeline.vae_scale_factor_temporal
        spatial_scale = self.adapter.pipeline.vae_scale_factor_spatial
        expected_latent_shape = (
            len(videos),
            getattr(vae.config, "z_dim", latents.shape[1]),
            (num_frames - 1) // temporal_scale + 1,
            height // spatial_scale,
            width // spatial_scale,
        )
        if tuple(latents.shape) != expected_latent_shape:
            raise ValueError(
                "Wan VAE target latent geometry mismatch: "
                f"expected {expected_latent_shape}, received {tuple(latents.shape)}"
            )

        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.VIDEO,
                    height=height,
                    width=width,
                    frames=num_frames,
                    fps=frame_rate,
                ),
            )
        )
        active_masks = None
        if first_frame_mask is not None:
            active_mask = first_frame_mask.to(device=latents.device, dtype=torch.bool)
            active_masks = {"latent": active_mask}

        return EncodedOutputState(
            clean_state=LatentState({"latent": latents}, active_masks=active_masks),
            forward_context={},
            decode_context={
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
            },
            geometry_signatures=tuple(signature for _ in videos),
        )


__all__ = [
    "WanVideoOutputCodec",
    "configured_wan_video_output_geometry",
    "normalize_wan_video_latents",
    "resample_wan_output_video",
    "validate_wan_encoded_output_geometry",
]
