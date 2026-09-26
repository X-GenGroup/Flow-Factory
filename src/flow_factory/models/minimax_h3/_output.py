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

"""On-the-fly audiovisual target encoding for all MiniMax H3 workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...contracts import MediaGeometry, MediaType
from ...samples import LatentState
from ...utils.audio import convert_audio, require_decoded_audio_waveform
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
)
from ._common import pack_audio_latents, pack_video_latents, validate_target_state
from .workflow import _normalize_geometry, _normalize_layout

_GEOMETRY_FIELDS = (
    "height",
    "width",
    "num_frames",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
)
_LAYOUT_FIELDS = (
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
    "num_condition_video_rows",
    "num_condition_audio_rows",
)
_RELEASED_PATCH_SIZE = (1, 2, 2)
_RELEASED_VIDEO_LATENT_CHANNELS = 24
_RELEASED_AUDIO_CHANNELS = 2
_RELEASED_AUDIO_LATENT_CHANNELS = 32
_RELEASED_AUDIO_SAMPLE_RATE = 32000
_RELEASED_AUDIO_HOP_LENGTH = 800


@dataclass(frozen=True, slots=True)
class _H3ModelShape:
    patch_size: Tuple[int, int, int]
    video_latent_channels: int
    audio_channels: int
    audio_latent_channels: int


@dataclass(frozen=True, slots=True)
class MiniMaxH3AVOutputCodec:
    """Encode one configured H3 video/audio target into packed target rows."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("vae", "audio_vae")

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Encode the exact ``(video, audio)`` target sequence for one sample.

        Diffusers defines H3's conditioning encoder, but not an offline target
        encoder. Both target posteriors therefore use their deterministic modes,
        matching the released H3 SFT data flow. The condition-only float16 rounding
        step is intentionally not applied to clean targets. Both components are
        normalized and packed exactly like generated target rows.

        Args:
            media_batch: One decoded video/audio pair.
            condition: Cached H3 condition and its authoritative packed layout.
            generator: Accepted for the shared codec protocol; H3 target modes are
                deterministic and do not consume it.

        Returns:
            Detached clean video/audio rows plus the output-owned decode context.
        """
        if len(media_batch) != 1:
            raise ValueError(
                f"MiniMax H3 output encoding requires B=1, received B={len(media_batch)}"
            )
        candidate = media_batch[0]
        if len(candidate) != 2:
            raise ValueError(
                "MiniMax H3 output codec expected exact (video, audio) media, "
                f"received {len(candidate)} items"
            )
        video_media, audio_media = candidate
        geometry = resolve_h3_output_geometry(self.adapter, condition)
        model_shape = _resolve_h3_model_shape(self.adapter)

        video_pixels = prepare_h3_target_video(
            video_media.payload,
            source_fps=video_media.fps,
            target_frames=geometry["num_frames"],
            target_fps=float(self.adapter.pipeline.fps),
            height=geometry["height"],
            width=geometry["width"],
        )
        del generator
        video_latents = encode_h3_target_video(self.adapter, video_pixels)
        expected_video_shape = (
            1,
            model_shape.video_latent_channels,
            geometry["num_latent_frames"],
            geometry["latent_height"],
            geometry["latent_width"],
        )
        if tuple(video_latents.shape) != expected_video_shape:
            raise ValueError(
                "MiniMax H3 target video latent geometry mismatch: "
                f"expected {expected_video_shape}, received {tuple(video_latents.shape)}"
            )

        audio_vae = self.adapter.get_component("audio_vae")
        sample_rate = _positive_int(
            getattr(audio_vae.config, "sampling_rate", None),
            "audio_vae.config.sampling_rate",
        )
        hop_length = _positive_int(
            getattr(audio_vae, "hop_length", None),
            "audio_vae.hop_length",
        )
        target_audio_samples = geometry["num_audio_latents"] * hop_length
        waveform = prepare_h3_target_audio(
            audio_media.payload,
            source_sample_rate=audio_media.sample_rate,
            target_sample_rate=sample_rate,
            target_samples=target_audio_samples,
            target_duration_seconds=geometry["num_frames"] / float(self.adapter.pipeline.fps),
        )
        audio_latents = encode_h3_target_audio(self.adapter, waveform)
        expected_audio_shape = (
            model_shape.audio_channels,
            model_shape.audio_latent_channels,
            geometry["num_audio_latents"],
        )
        if tuple(audio_latents.shape) != expected_audio_shape:
            raise ValueError(
                "MiniMax H3 target audio latent geometry mismatch: "
                f"expected {expected_audio_shape}, received {tuple(audio_latents.shape)}"
            )

        clean_state = LatentState(
            {
                "video": pack_video_latents(video_latents.to(torch.float32)),
                "audio": pack_audio_latents(audio_latents.to(torch.float32).unsqueeze(0)),
            }
        )
        validate_target_state(clean_state)
        _validate_h3_input_layout(condition, clean_state)
        frame_rate = float(self.adapter.pipeline.fps)
        signature = GeometrySignature(
            media=(
                MediaGeometry(
                    type=MediaType.VIDEO,
                    height=geometry["height"],
                    width=geometry["width"],
                    frames=geometry["num_frames"],
                    fps=frame_rate,
                ),
                MediaGeometry(
                    type=MediaType.AUDIO,
                    samples=target_audio_samples,
                    sample_rate=sample_rate,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=clean_state,
            forward_context={},
            decode_context={"geometry": geometry},
            geometry_signatures=(signature,),
        )


def resolve_h3_output_geometry(adapter: Any, condition: Mapping[str, Any]) -> Dict[str, int]:
    """Resolve and validate configured geometry from the cached H3 condition.

    Args:
        adapter: Active MiniMax H3 adapter with configured pipeline components.
        condition: Cached input condition carrying flat H3 geometry fields.

    Returns:
        Canonical positive integer geometry validated against the training config.
    """
    geometry = _normalize_geometry(condition)
    missing = tuple(field for field in _GEOMETRY_FIELDS if field not in geometry)
    if missing:
        raise ValueError(f"MiniMax H3 offline condition geometry missing fields={missing}")
    for field in _GEOMETRY_FIELDS:
        _positive_int(geometry[field], f"condition.{field}")

    pipeline = adapter.pipeline
    model_shape = _resolve_h3_model_shape(adapter)
    frame_rate = _positive_real(getattr(pipeline, "fps", None), "pipeline.fps")
    if frame_rate != 24.0:
        raise ValueError(
            f"MiniMax H3 target encoding requires fixed 24 fps, received {frame_rate!r}"
        )
    configured_frame_rate = _positive_real(
        getattr(adapter.training_args, "frame_rate", None),
        "training_args.frame_rate",
    )
    if configured_frame_rate != frame_rate:
        raise ValueError(
            "MiniMax H3 configured frame rate must match the model clock: "
            f"expected {frame_rate}, received {configured_frame_rate}"
        )
    configured_height = _positive_int(
        getattr(adapter.training_args, "height", None),
        "training_args.height",
    )
    configured_width = _positive_int(
        getattr(adapter.training_args, "width", None),
        "training_args.width",
    )
    if (geometry["height"], geometry["width"]) != (
        configured_height,
        configured_width,
    ):
        raise ValueError(
            "MiniMax H3 cached output canvas does not match the current configured geometry: "
            f"expected {(configured_height, configured_width)}, received "
            f"{(geometry['height'], geometry['width'])}"
        )

    frames_per_chunk = _positive_int(
        getattr(pipeline, "vae_frames_per_chunk", None),
        "pipeline.vae_frames_per_chunk",
    )
    latents_per_chunk = _positive_int(
        getattr(pipeline, "vae_latents_per_chunk", None),
        "pipeline.vae_latents_per_chunk",
    )
    if latents_per_chunk >= frames_per_chunk:
        raise ValueError(
            "MiniMax H3 video VAE chunk geometry requires latents_per_chunk < "
            f"frames_per_chunk, received {(latents_per_chunk, frames_per_chunk)}"
        )
    configured_num_frames = _positive_int(
        getattr(adapter.training_args, "num_frames", None),
        "training_args.num_frames",
    )
    aligned_configured_num_frames = configured_num_frames + (
        (latents_per_chunk - configured_num_frames) % frames_per_chunk
    )
    if geometry["num_frames"] != aligned_configured_num_frames:
        raise ValueError(
            "MiniMax H3 cached output frame count does not match the current configured "
            "frame count after official VAE alignment: "
            f"expected {aligned_configured_num_frames} from configured "
            f"num_frames={configured_num_frames}, received {geometry['num_frames']}"
        )

    min_duration = _positive_real(
        getattr(pipeline, "min_duration", None),
        "pipeline.min_duration",
    )
    max_duration = _positive_real(
        getattr(pipeline, "max_duration", None),
        "pipeline.max_duration",
    )
    if min_duration > max_duration:
        raise ValueError(
            "MiniMax H3 pipeline duration bounds are inverted: "
            f"min_duration={min_duration}, max_duration={max_duration}"
        )
    duration = geometry["num_frames"] / frame_rate
    if not min_duration <= duration <= max_duration:
        raise ValueError(
            "MiniMax H3 configured output duration is outside the pipeline contract: "
            f"expected [{min_duration}, {max_duration}] seconds, received {duration} "
            f"from {geometry['num_frames']} frames at {frame_rate} fps"
        )

    spatial_ratio = _positive_int(
        getattr(pipeline, "vae_spatial_compression_ratio", None),
        "pipeline.vae_spatial_compression_ratio",
    )
    expected_latent_height = geometry["height"] // spatial_ratio
    expected_latent_width = geometry["width"] // spatial_ratio
    if geometry["height"] % spatial_ratio or geometry["width"] % spatial_ratio:
        raise ValueError(
            "MiniMax H3 output height/width must be divisible by the video VAE spatial ratio "
            f"{spatial_ratio}, received {(geometry['height'], geometry['width'])}"
        )
    if (geometry["latent_height"], geometry["latent_width"]) != (
        expected_latent_height,
        expected_latent_width,
    ):
        raise ValueError(
            "MiniMax H3 cached spatial latent geometry mismatch: expected "
            f"{(expected_latent_height, expected_latent_width)}, received "
            f"{(geometry['latent_height'], geometry['latent_width'])}"
        )
    _, patch_height, patch_width = model_shape.patch_size
    if geometry["latent_height"] % patch_height or geometry["latent_width"] % patch_width:
        raise ValueError(
            "MiniMax H3 cached latent height/width must be divisible by transformer patch "
            f"{model_shape.patch_size}, received "
            f"{(geometry['latent_height'], geometry['latent_width'])}"
        )

    num_frames = geometry["num_frames"]
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            "MiniMax H3 target num_frames must satisfy the video VAE chunk geometry "
            f"F % {frames_per_chunk} == {latents_per_chunk}, received {num_frames}"
        )
    expected_video_latents = (
        num_frames - latents_per_chunk
    ) // frames_per_chunk * latents_per_chunk + 2
    if geometry["num_latent_frames"] != expected_video_latents:
        raise ValueError(
            "MiniMax H3 cached temporal latent geometry mismatch: expected "
            f"{expected_video_latents}, received {geometry['num_latent_frames']}"
        )
    patch_time = model_shape.patch_size[0]
    if geometry["num_latent_frames"] % patch_time:
        raise ValueError(
            "MiniMax H3 cached latent frame count must be divisible by transformer temporal "
            f"patch {patch_time}, received {geometry['num_latent_frames']}"
        )

    audio_vae = adapter.get_component("audio_vae")
    sample_rate = _positive_int(
        getattr(audio_vae.config, "sampling_rate", None),
        "audio_vae.config.sampling_rate",
    )
    hop_length = _positive_int(
        getattr(audio_vae, "hop_length", None),
        "audio_vae.hop_length",
    )
    pipeline_sample_rate = _positive_int(
        getattr(pipeline, "audio_sampling_rate", None),
        "pipeline.audio_sampling_rate",
    )
    if (
        sample_rate,
        pipeline_sample_rate,
        hop_length,
    ) != (
        _RELEASED_AUDIO_SAMPLE_RATE,
        _RELEASED_AUDIO_SAMPLE_RATE,
        _RELEASED_AUDIO_HOP_LENGTH,
    ):
        raise ValueError(
            "MiniMax H3 output codec requires the released audio clock "
            f"sample_rate/hop_length={(_RELEASED_AUDIO_SAMPLE_RATE, _RELEASED_AUDIO_HOP_LENGTH)}, "
            f"received config/pipeline/hop={(sample_rate, pipeline_sample_rate, hop_length)}"
        )
    expected_audio_latents = int(round(num_frames / frame_rate * sample_rate / hop_length))
    if geometry["num_audio_latents"] != expected_audio_latents:
        raise ValueError(
            "MiniMax H3 cached audio latent geometry mismatch: expected "
            f"{expected_audio_latents}, received {geometry['num_audio_latents']}"
        )
    return geometry


def prepare_h3_target_video(
    payload: Any,
    *,
    source_fps: Any,
    target_frames: int,
    target_fps: float,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resample decoded RGB frames onto the configured H3 video grid.

    Args:
        payload: Decoded uint8 RGB frames shaped ``(F, H, W, 3)``.
        source_fps: Logical source frame rate from the media manifest.
        target_frames: Exact aligned output frame count.
        target_fps: Fixed model frame rate.
        height: Configured output height.
        width: Configured output width.

    Returns:
        Float32 pixels shaped ``(1, 3, F, H, W)`` in the unit interval.
    """
    payload = require_decoded_video_frames(payload, source="MiniMax H3 target video")
    source_fps = _positive_real(source_fps, "target video fps")
    target_fps = _positive_real(target_fps, "model video fps")
    frames = payload
    if source_fps != target_fps:
        # Match H3's official ffmpeg-style fps filter: each source frame is held
        # until the rounded slot of the next frame, including the stream endpoint.
        scale = target_fps / source_fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        endpoint = math.floor(frames.shape[0] * scale + 0.5)
        frames = np.repeat(frames, np.diff(slots, append=endpoint), axis=0)
    if frames.shape[0] < target_frames:
        required_duration = target_frames / target_fps
        available_duration = frames.shape[0] / target_fps
        raise ValueError(
            "MiniMax H3 target video is too short for configured temporal geometry: "
            f"requires {target_frames} frames/{required_duration:.6f}s after rate conversion, "
            f"has {frames.shape[0]} frames/{available_duration:.6f}s"
        )
    frames = np.ascontiguousarray(frames[:target_frames])
    if frames.shape[1:3] != (height, width):
        frames = np.stack(
            [
                np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
                for frame in frames
            ]
        )
    unit_frames = decoded_video_to_unit_float(frames, source="MiniMax H3 sampled target video")
    pixels = torch.from_numpy(unit_frames).permute(3, 0, 1, 2).unsqueeze(0)
    return require_finite_bcfhw_video(
        pixels,
        source="MiniMax H3 unit target pixels",
        batch_size=1,
        frames=target_frames,
        height=height,
        width=width,
    )


def encode_h3_target_video(
    adapter: Any,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Encode clean H3 targets through the deterministic video posterior mode.

    Diffusers specifies a fixed-seed sample followed by float16 rounding only for
    H3 *conditions*. Its released pipeline has no offline target encoder, so this
    path follows the H3 SFT reference data flow: take the posterior mean/mode and
    retain float32 precision before normalization.

    Args:
        adapter: Active MiniMax H3 adapter exposing the video VAE.
        pixel_values: Normalized-shape pixels ``(1, 3, F, H, W)`` in ``[0, 1]``.
    Returns:
        Normalized video latents shaped ``(1, 24, F', H', W')``.
    """
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
        raise ValueError(
            "MiniMax H3 target video pixels expected BCFHW tensor, "
            f"received {type(pixel_values).__name__}/{getattr(pixel_values, 'shape', None)}"
        )
    vae = adapter.get_component("vae")
    device = torch.device(adapter.device)
    pixel_mean = torch.as_tensor(
        adapter.pipeline.pixel_mean,
        device=device,
        dtype=torch.float32,
    ).view(1, -1, 1, 1, 1)
    pixel_std = torch.as_tensor(
        adapter.pipeline.pixel_std,
        device=device,
        dtype=torch.float32,
    ).view(1, -1, 1, 1, 1)
    normalized_pixels = (
        pixel_values.to(device=device, dtype=torch.float32) - pixel_mean
    ) / pixel_std
    encoded = vae.encode(normalized_pixels)
    latents = retrieve_vae_latents(
        encoded,
        sample_mode="argmax",
        source="MiniMax H3 target video",
    ).to(torch.float32)
    latent_mean, latent_std = _latent_statistics(
        vae.config,
        channels=latents.shape[1],
        rank=5,
        device=latents.device,
        source="MiniMax H3 video VAE",
    )
    return (latents - latent_mean) / latent_std


def prepare_h3_target_audio(
    payload: Any,
    *,
    source_sample_rate: Any,
    target_sample_rate: int,
    target_samples: int,
    target_duration_seconds: float,
) -> torch.Tensor:
    """Convert one waveform to exact stereo H3 audio-grid geometry.

    The source-clock truncation precedes resampling, matching Diffusers' H3
    reference normalization. The final trim/pad then accounts for audio-latent
    rounding so the waveform lands on the exact model grid.

    Args:
        payload: Decoded mono or stereo float waveform shaped ``(C, S)``.
        source_sample_rate: Logical source clock from the media manifest.
        target_sample_rate: Audio VAE sample rate.
        target_samples: Exact number of samples required by the audio latent grid.
        target_duration_seconds: Aligned AV duration used for source-clock truncation.

    Returns:
        Contiguous float32 stereo waveform shaped ``(2, target_samples)``.
    """
    payload = require_decoded_audio_waveform(payload, source="MiniMax H3 target audio")
    if payload.shape[0] not in (1, 2):
        raise ValueError(
            "MiniMax H3 target audio must be mono or stereo, "
            f"received {payload.shape[0]} channels"
        )
    source_sample_rate = _positive_int(source_sample_rate, "target audio sample_rate")
    target_sample_rate = _positive_int(target_sample_rate, "model audio sample_rate")
    target_samples = _positive_int(target_samples, "target audio samples")
    target_duration_seconds = _positive_real(
        target_duration_seconds,
        "target audio duration",
    )
    source_samples = int(target_duration_seconds * source_sample_rate)
    if source_samples < 1:
        raise ValueError(
            "MiniMax H3 target audio duration resolves to fewer than one source sample: "
            f"duration={target_duration_seconds}, sample_rate={source_sample_rate}"
        )
    source_waveform = payload.detach().to(device="cpu", dtype=torch.float32)[:, :source_samples]
    waveform = convert_audio(
        source_waveform,
        from_rate=source_sample_rate,
        to_rate=target_sample_rate,
        to_channels=2,
    )
    if waveform.shape[-1] >= target_samples:
        waveform = waveform[:, :target_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
    return waveform.contiguous()


def encode_h3_target_audio(adapter: Any, waveform: torch.Tensor) -> torch.Tensor:
    """Take and normalize the official H3 audio posterior mode.

    Args:
        adapter: Active MiniMax H3 adapter exposing the audio VAE.
        waveform: Exact stereo model-rate waveform shaped ``(2, S)``.

    Returns:
        Normalized channel-major audio latents shaped ``(2, 32, F)``.
    """
    audio_vae = adapter.get_component("audio_vae")
    device = torch.device(adapter.device)
    posterior = audio_vae.encode(waveform.to(device=device)[:, None])
    latents = retrieve_vae_latents(
        posterior,
        sample_mode="argmax",
        source="MiniMax H3 target audio",
    ).to(torch.float32)
    latent_mean, latent_std = _latent_statistics(
        audio_vae.config,
        channels=latents.shape[1],
        rank=3,
        device=latents.device,
        source="MiniMax H3 audio VAE",
    )
    return (latents - latent_mean) / latent_std


def _validate_h3_input_layout(
    condition: Mapping[str, Any],
    clean_state: LatentState,
) -> None:
    """Validate target rows against an authoritative H3 input layout."""
    layout = _normalize_layout(condition)
    missing = tuple(field for field in _LAYOUT_FIELDS if field not in layout)
    if missing:
        raise ValueError(f"MiniMax H3 offline condition layout missing fields={missing}")

    position_ids = layout["position_ids"]
    token_tags = layout["token_tags"]
    index_tensors = [layout[field] for field in ("video_indices", "audio_indices", "text_indices")]
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.ndim != 2
        or position_ids.shape[-1] != 3
        or position_ids.dtype != torch.float64
    ):
        raise ValueError(
            "MiniMax H3 position_ids expected float64 shape (N,3), "
            f"received {type(position_ids).__name__}/{getattr(position_ids, 'shape', None)}/"
            f"{getattr(position_ids, 'dtype', None)}"
        )
    if (
        not isinstance(token_tags, torch.Tensor)
        or token_tags.ndim != 1
        or token_tags.dtype != torch.long
    ):
        raise ValueError(
            "MiniMax H3 token_tags expected one-dimensional torch.long, "
            f"received {type(token_tags).__name__}/{getattr(token_tags, 'shape', None)}/"
            f"{getattr(token_tags, 'dtype', None)}"
        )
    for field, values in zip(("video_indices", "audio_indices", "text_indices"), index_tensors):
        if not isinstance(values, torch.Tensor) or values.ndim != 1 or values.dtype != torch.long:
            raise ValueError(
                f"MiniMax H3 {field} expected one-dimensional torch.long, "
                f"received {type(values).__name__}/{getattr(values, 'shape', None)}/"
                f"{getattr(values, 'dtype', None)}"
            )

    for component in ("video", "audio"):
        count_field = f"num_condition_{component}_rows"
        condition_rows = layout[count_field]
        if condition_rows < 0:
            raise ValueError(
                f"MiniMax H3 offline layout requires non-negative {count_field}, "
                f"received {condition_rows}"
            )
        component_indices = layout[f"{component}_indices"]
        target_rows = clean_state.components[component].shape[1]
        expected_total_rows = condition_rows + target_rows
        if component_indices.numel() != expected_total_rows:
            raise ValueError(
                f"MiniMax H3 {component} layout expected {condition_rows} condition + "
                f"{target_rows} target rows, received {component_indices.numel()} indices"
            )

    sequence_length = sum(values.numel() for values in index_tensors)
    if position_ids.shape[0] != sequence_length or token_tags.numel() != sequence_length:
        raise ValueError(
            "MiniMax H3 flat layout sequence lengths disagree: "
            f"indices={sequence_length}, position_ids={position_ids.shape[0]}, "
            f"token_tags={token_tags.numel()}"
        )
    devices = {
        position_ids.device,
        token_tags.device,
        *(values.device for values in index_tensors),
    }
    if len(devices) != 1:
        raise ValueError(f"MiniMax H3 flat layout tensors must share one device, got {devices}")
    permutation = torch.cat(index_tensors).sort().values
    expected_permutation = torch.arange(
        sequence_length,
        dtype=torch.long,
        device=permutation.device,
    )
    if not torch.equal(permutation, expected_permutation):
        raise ValueError("MiniMax H3 video/audio/text indices must partition the packed sequence")


def validate_h3_encoded_output_geometry(
    adapter: Any,
    media_batch: DecodedMediaBatch,
    condition: Mapping[str, Any],
    encoded: EncodedOutputState,
) -> None:
    """Prove codec geometry and contexts agree with the cached H3 layout.

    Args:
        adapter: Active MiniMax H3 adapter.
        media_batch: Validated exact audiovisual target batch.
        condition: Cached input condition with geometry and packed-row layout.
        encoded: Encoded structured target state to validate.

    Returns:
        None after every geometry, component, and context invariant is proven.
    """
    geometry = resolve_h3_output_geometry(adapter, condition)
    model_shape = _resolve_h3_model_shape(adapter)
    audio_vae = adapter.get_component("audio_vae")
    sample_rate = _positive_int(
        getattr(audio_vae.config, "sampling_rate", None),
        "audio_vae.config.sampling_rate",
    )
    hop_length = _positive_int(
        getattr(audio_vae, "hop_length", None),
        "audio_vae.hop_length",
    )
    expected_signature = GeometrySignature(
        media=(
            MediaGeometry(
                type=MediaType.VIDEO,
                height=geometry["height"],
                width=geometry["width"],
                frames=geometry["num_frames"],
                fps=float(adapter.pipeline.fps),
            ),
            MediaGeometry(
                type=MediaType.AUDIO,
                samples=geometry["num_audio_latents"] * hop_length,
                sample_rate=sample_rate,
            ),
        )
    )
    if len(media_batch) != 1 or encoded.geometry_signatures != (expected_signature,):
        raise ValueError(
            "MiniMax H3 encoded output geometry disagrees with configured audiovisual geometry: "
            f"expected {(expected_signature,)!r}, received {encoded.geometry_signatures!r}"
        )
    if encoded.decode_context.get("geometry") != geometry:
        raise ValueError(
            "MiniMax H3 decode_context geometry disagrees with cached condition geometry: "
            f"expected {geometry!r}, received {encoded.decode_context.get('geometry')!r}"
        )

    validate_target_state(encoded.clean_state)
    if encoded.clean_state.component_names != ("video", "audio"):
        raise ValueError(
            "MiniMax H3 clean target components must be ordered ('video', 'audio'), "
            f"received {encoded.clean_state.component_names}"
        )
    patch_time, patch_height, patch_width = model_shape.patch_size
    expected_shapes = {
        "video": (
            1,
            geometry["num_latent_frames"]
            // patch_time
            * (geometry["latent_height"] // patch_height)
            * (geometry["latent_width"] // patch_width),
            model_shape.video_latent_channels * patch_time * patch_height * patch_width,
        ),
        "audio": (
            1,
            model_shape.audio_channels * geometry["num_audio_latents"],
            model_shape.audio_latent_channels,
        ),
    }
    for component, expected_shape in expected_shapes.items():
        received_shape = tuple(encoded.clean_state.components[component].shape)
        if received_shape != expected_shape:
            raise ValueError(
                f"MiniMax H3 clean {component} rows expected shape {expected_shape}, "
                f"received {received_shape}"
            )

    if encoded.forward_context:
        raise ValueError(
            "MiniMax H3 output codec must not duplicate input-owned layout/prefix fields, "
            f"received keys={tuple(encoded.forward_context)}"
        )
    _validate_h3_input_layout(condition, encoded.clean_state)


def _resolve_h3_model_shape(adapter: Any) -> _H3ModelShape:
    """Validate the packing dimensions of the released H3 checkpoint contract."""
    pipeline = adapter.pipeline
    patch_size = getattr(pipeline, "patch_size", None)
    if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 3:
        raise TypeError(
            "pipeline.patch_size expected a length-3 tuple/list, "
            f"received {type(patch_size).__name__}: {patch_size!r}"
        )
    patch_size = tuple(
        _positive_int(value, f"pipeline.patch_size[{index}]")
        for index, value in enumerate(patch_size)
    )
    video_latent_channels = _positive_int(
        getattr(pipeline, "vae_latent_channels", None),
        "pipeline.vae_latent_channels",
    )
    audio_channels = _positive_int(
        getattr(pipeline, "audio_channels", None),
        "pipeline.audio_channels",
    )
    audio_latent_channels = _positive_int(
        getattr(pipeline, "audio_latent_channels", None),
        "pipeline.audio_latent_channels",
    )
    received = (
        patch_size,
        video_latent_channels,
        audio_channels,
        audio_latent_channels,
    )
    expected = (
        _RELEASED_PATCH_SIZE,
        _RELEASED_VIDEO_LATENT_CHANNELS,
        _RELEASED_AUDIO_CHANNELS,
        _RELEASED_AUDIO_LATENT_CHANNELS,
    )
    if received != expected:
        raise ValueError(
            "MiniMax H3 output codec supports the released packing contract "
            "patch/video_channels/audio_channels/audio_latent_channels="
            f"{expected!r}, received {received!r}"
        )
    return _H3ModelShape(
        patch_size=patch_size,
        video_latent_channels=video_latent_channels,
        audio_channels=audio_channels,
        audio_latent_channels=audio_latent_channels,
    )


def _latent_statistics(
    config: Any,
    *,
    channels: int,
    rank: int,
    device: torch.device,
    source: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_mean = getattr(config, "latents_mean", None)
    raw_std = getattr(config, "latents_std", None)
    if raw_mean is None or raw_std is None:
        raise ValueError(f"{source} config must define latents_mean and latents_std")
    mean = torch.as_tensor(raw_mean, device=device, dtype=torch.float32)
    std = torch.as_tensor(raw_std, device=device, dtype=torch.float32)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError(
            f"{source} expected per-channel latent statistics shaped ({channels},), "
            f"received mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError(f"{source} latent statistics must be finite with strictly positive std")
    shape = [1, channels, *([1] * (rank - 2))]
    return mean.view(shape), std.view(shape)


def _positive_int(value: Any, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{source} expected positive int, received {value!r}")
    return value


def _positive_real(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{source} expected positive finite real, received {type(value).__name__}: {value!r}"
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{source} expected positive finite real, received {value!r}")
    return value


__all__ = [
    "MiniMaxH3AVOutputCodec",
    "encode_h3_target_audio",
    "encode_h3_target_video",
    "prepare_h3_target_audio",
    "prepare_h3_target_video",
    "resolve_h3_output_geometry",
    "validate_h3_encoded_output_geometry",
]
