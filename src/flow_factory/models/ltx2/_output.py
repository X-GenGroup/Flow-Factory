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

"""On-the-fly audiovisual target encoding shared by the LTX2 adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Optional, Tuple

import numpy as np
import torch
import torchaudio

from ...contracts import MediaGeometry, MediaType
from ...samples import LatentState
from ...utils.audio import convert_audio, require_decoded_audio_waveform
from ...utils.video import (
    decoded_video_to_unit_float,
    require_decoded_video_frames,
    require_finite_bcfhw_video,
)
from ..condition_state import PreparedConditionState
from ..configured_image_output import retrieve_vae_latents
from ..output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
)

LTX2_OFFLINE_FORWARD_OVERRIDES = MappingProxyType(
    {
        "guidance_scale": 1.0,
        "audio_guidance_scale": 1.0,
        "guidance_rescale": 0.0,
        "audio_guidance_rescale": 0.0,
        "stg_scale": 0.0,
        "audio_stg_scale": 0.0,
        "spatio_temporal_guidance_blocks": None,
        "modality_scale": 1.0,
        "audio_modality_scale": 1.0,
        "preserve_raw_model_velocity": True,
    }
)


@dataclass(frozen=True, slots=True)
class LTX2VideoGeometry:
    """Canonical configured video and packed-latent geometry."""

    height: int
    width: int
    num_frames: int
    frame_rate: float
    latent_frames: int
    latent_height: int
    latent_width: int
    latent_channels: int
    patch_size: int
    patch_size_t: int
    sequence_length: int
    feature_dim: int


@dataclass(frozen=True, slots=True)
class LTX2AudioGeometry:
    """Canonical configured waveform, mel, and packed-latent geometry."""

    sample_rate: int
    hop_length: int
    waveform_channels: int
    target_samples: int
    mel_bins: int
    latent_mel_bins: int
    latent_channels: int
    latent_frames: int
    temporal_compression_ratio: int
    mel_compression_ratio: int
    sequence_length: int
    feature_dim: int


@dataclass(frozen=True, slots=True)
class LTX2OutputGeometry:
    """One aligned LTX2 video/audio output geometry."""

    video: LTX2VideoGeometry
    audio: LTX2AudioGeometry


@dataclass(frozen=True, slots=True)
class LTX2FirstFrameConditionPreparer:
    """Encode an I2AV condition image once for all offline target candidates."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedConditionState:
        """Realize the deterministic VideoVAE posterior mode for the first frame."""
        del generator
        if "condition_images" not in condition:
            raise ValueError(
                "LTX2 I2AV offline condition requires cached `condition_images` pixels"
            )
        video_geometry = resolve_ltx2_video_geometry(self.adapter)
        pixels = condition["condition_images"]
        if not isinstance(pixels, torch.Tensor):
            raise TypeError(
                "LTX2 I2AV condition_images must be a torch.Tensor, "
                f"received {type(pixels).__name__}"
            )
        expected_shape = (
            pixels.shape[0] if pixels.ndim == 4 else None,
            3,
            video_geometry.height,
            video_geometry.width,
        )
        if pixels.ndim != 4 or tuple(pixels.shape) != expected_shape:
            raise ValueError(
                "LTX2 I2AV condition_images must use configured BCHW geometry: "
                f"expected {expected_shape}, received {tuple(pixels.shape)}"
            )
        if not pixels.is_floating_point():
            raise TypeError(
                "LTX2 I2AV condition_images must be floating pixels, " f"received {pixels.dtype}"
            )
        _require_finite_tensor(pixels, "LTX2 I2AV condition_images")

        vae = self.adapter.get_component("vae")
        vae_dtype = _floating_module_dtype(vae, "LTX2 VideoVAE")
        encoded = vae.encode(pixels.to(device=self.adapter.device, dtype=vae_dtype).unsqueeze(2))
        condition_latents = retrieve_vae_latents(
            encoded,
            sample_mode="argmax",
            source="LTX2 I2AV condition image",
        ).to(device=self.adapter.device, dtype=torch.float32)
        expected_latent_shape = (
            pixels.shape[0],
            video_geometry.latent_channels,
            1,
            video_geometry.latent_height,
            video_geometry.latent_width,
        )
        if tuple(condition_latents.shape) != expected_latent_shape:
            raise ValueError(
                "LTX2 I2AV condition VideoVAE latent geometry mismatch: "
                f"expected {expected_latent_shape}, received {tuple(condition_latents.shape)}"
            )
        _require_finite_tensor(condition_latents, "LTX2 I2AV condition latents")

        # Pixels are an encoder input, not a transformer input. Dropping them from
        # the realized condition also avoids retaining a full-resolution tensor
        # through both chosen and rejected offline forwards.
        model_condition = {
            key: value for key, value in condition.items() if key != "condition_images"
        }
        return PreparedConditionState(
            condition=model_condition,
            forward_context={},
            output_context={"condition_video_latents": condition_latents.detach()},
        )


@dataclass(frozen=True, slots=True)
class LTX2AVOutputCodec:
    """Encode exact ``(video, audio)`` targets into LTX2's joint latent state."""

    adapter: Any
    conditioned: bool = False
    required_components: ClassVar[Tuple[str, ...]] = ("vae", "audio_vae")

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Encode VideoVAE and AudioVAE posterior modes on the configured AV grid."""
        del generator
        geometry = resolve_ltx2_output_geometry(self.adapter, conditioned=self.conditioned)
        videos = []
        waveforms = []
        for sample_index, candidate in enumerate(media_batch):
            if len(candidate) != 2:
                raise ValueError(
                    "LTX2 output codec expected exact (video, audio) media, "
                    f"received {len(candidate)} items for sample {sample_index}"
                )
            video_media, audio_media = candidate
            videos.append(
                prepare_ltx2_target_video(
                    video_media.payload,
                    source_fps=video_media.fps,
                    geometry=geometry.video,
                )
            )
            waveforms.append(
                prepare_ltx2_target_audio(
                    audio_media.payload,
                    source_sample_rate=audio_media.sample_rate,
                    geometry=geometry.audio,
                    duration_seconds=geometry.video.num_frames / geometry.video.frame_rate,
                )
            )

        video_latents = encode_ltx2_target_video(self.adapter, videos, geometry.video)
        if self.conditioned:
            condition_latents = condition.get("condition_video_latents")
            expected_condition_shape = (
                len(media_batch),
                geometry.video.latent_channels,
                1,
                geometry.video.latent_height,
                geometry.video.latent_width,
            )
            if not isinstance(condition_latents, torch.Tensor):
                raise TypeError(
                    "LTX2 I2AV output binding requires condition_video_latents from "
                    "prepare_condition_state()"
                )
            if tuple(condition_latents.shape) != expected_condition_shape:
                raise ValueError(
                    "LTX2 I2AV condition latent batch/geometry mismatch: "
                    f"expected {expected_condition_shape}, received "
                    f"{tuple(condition_latents.shape)}"
                )
            condition_latents = condition_latents.to(
                device=self.adapter.device,
                dtype=torch.float32,
            )
            _require_finite_tensor(condition_latents, "LTX2 I2AV condition latents")
            video_latents = torch.cat(
                [condition_latents, video_latents[:, :, 1:]],
                dim=2,
            )

        packed_video = normalize_and_pack_ltx2_video(
            self.adapter,
            video_latents,
            geometry.video,
        )
        packed_audio = encode_ltx2_target_audio(
            self.adapter,
            torch.stack(waveforms),
            geometry.audio,
        )

        forward_context = _ltx2_forward_context(geometry)
        active_masks = None
        if self.conditioned:
            unpacked_mask = packed_video.new_zeros(
                (
                    len(media_batch),
                    1,
                    geometry.video.latent_frames,
                    geometry.video.latent_height,
                    geometry.video.latent_width,
                )
            )
            unpacked_mask[:, :, 0] = 1.0
            conditioning_mask = self.adapter.pipeline._pack_latents(
                unpacked_mask,
                geometry.video.patch_size,
                geometry.video.patch_size_t,
            )
            if conditioning_mask.shape[-1] != 1:
                raise ValueError(
                    "LTX2 I2AV conditioning mask requires one scalar per packed token, "
                    f"received packed shape {tuple(conditioning_mask.shape)}"
                )
            conditioning_mask = conditioning_mask.squeeze(-1)
            forward_context["conditioning_mask"] = conditioning_mask
            active_masks = {
                "video": (~conditioning_mask.bool()).unsqueeze(-1),
                "audio": torch.ones(
                    (len(media_batch), geometry.audio.sequence_length, 1),
                    device=packed_audio.device,
                    dtype=torch.bool,
                ),
            }

        signature = GeometrySignature(
            media=(
                MediaGeometry(
                    type=MediaType.VIDEO,
                    height=geometry.video.height,
                    width=geometry.video.width,
                    frames=geometry.video.num_frames,
                    fps=geometry.video.frame_rate,
                ),
                MediaGeometry(
                    type=MediaType.AUDIO,
                    samples=geometry.audio.target_samples,
                    sample_rate=geometry.audio.sample_rate,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState(
                {"video": packed_video.detach(), "audio": packed_audio.detach()},
                active_masks=active_masks,
            ),
            forward_context=forward_context,
            decode_context={
                "height": geometry.video.height,
                "width": geometry.video.width,
                "num_frames": geometry.video.num_frames,
                "frame_rate": geometry.video.frame_rate,
            },
            geometry_signatures=tuple(signature for _ in media_batch),
        )


def resolve_ltx2_video_geometry(adapter: Any) -> LTX2VideoGeometry:
    """Resolve configured video geometry from runtime VAE and transformer metadata."""
    height = _positive_int(getattr(adapter.training_args, "height", None), "training_args.height")
    width = _positive_int(getattr(adapter.training_args, "width", None), "training_args.width")
    num_frames = _positive_int(
        getattr(adapter.training_args, "num_frames", None),
        "training_args.num_frames",
    )
    frame_rate = _positive_real(
        getattr(adapter.training_args, "frame_rate", None),
        "training_args.frame_rate",
    )
    pipeline = adapter.pipeline
    spatial_ratio = _positive_int(
        getattr(pipeline, "vae_spatial_compression_ratio", None),
        "pipeline.vae_spatial_compression_ratio",
    )
    temporal_ratio = _positive_int(
        getattr(pipeline, "vae_temporal_compression_ratio", None),
        "pipeline.vae_temporal_compression_ratio",
    )
    if height % spatial_ratio or width % spatial_ratio:
        raise ValueError(
            "LTX2 configured height/width must be divisible by the VideoVAE spatial "
            f"compression ratio {spatial_ratio}, received {(height, width)}"
        )
    if (num_frames - 1) % temporal_ratio:
        raise ValueError(
            "LTX2 configured num_frames must satisfy "
            f"(num_frames - 1) % {temporal_ratio} == 0, received {num_frames}"
        )
    latent_frames = (num_frames - 1) // temporal_ratio + 1
    latent_height = height // spatial_ratio
    latent_width = width // spatial_ratio
    patch_size = _positive_int(
        getattr(pipeline, "transformer_spatial_patch_size", None),
        "pipeline.transformer_spatial_patch_size",
    )
    patch_size_t = _positive_int(
        getattr(pipeline, "transformer_temporal_patch_size", None),
        "pipeline.transformer_temporal_patch_size",
    )
    if latent_frames % patch_size_t or latent_height % patch_size or latent_width % patch_size:
        raise ValueError(
            "LTX2 latent video grid must be divisible by transformer patches "
            f"(time={patch_size_t}, spatial={patch_size}), received "
            f"{(latent_frames, latent_height, latent_width)}"
        )
    vae = adapter.get_component("vae")
    latent_channels = _positive_int(
        getattr(getattr(vae, "config", None), "latent_channels", None),
        "vae.config.latent_channels",
    )
    feature_dim = latent_channels * patch_size_t * patch_size * patch_size
    sequence_length = (
        latent_frames // patch_size_t * (latent_height // patch_size) * (latent_width // patch_size)
    )
    return LTX2VideoGeometry(
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_channels=latent_channels,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
    )


def resolve_ltx2_output_geometry(
    adapter: Any,
    *,
    conditioned: bool,
) -> LTX2OutputGeometry:
    """Resolve and cross-check the complete config-driven LTX2 AV geometry."""
    video = resolve_ltx2_video_geometry(adapter)
    if conditioned and (video.patch_size != 1 or video.patch_size_t != 1):
        raise ValueError(
            "LTX2 I2AV's official scalar conditioning mask requires video patch_size=1 "
            f"and patch_size_t=1, received {(video.patch_size, video.patch_size_t)}"
        )

    pipeline = adapter.pipeline
    audio_vae = adapter.get_component("audio_vae")
    audio_config = getattr(audio_vae, "config", None)
    sample_rate = _positive_int(
        getattr(audio_config, "sample_rate", None),
        "audio_vae.config.sample_rate",
    )
    hop_length = _positive_int(
        getattr(audio_config, "mel_hop_length", None),
        "audio_vae.config.mel_hop_length",
    )
    mel_bins = _positive_int(
        getattr(audio_config, "mel_bins", None),
        "audio_vae.config.mel_bins",
    )
    waveform_channels = _positive_int(
        getattr(audio_config, "in_channels", None),
        "audio_vae.config.in_channels",
    )
    if waveform_channels not in (1, 2):
        raise ValueError(
            "LTX2 audio frontend supports mono or stereo AudioVAE inputs, "
            f"received in_channels={waveform_channels}"
        )
    latent_channels = _positive_int(
        getattr(audio_config, "latent_channels", None),
        "audio_vae.config.latent_channels",
    )
    temporal_ratio = _positive_int(
        getattr(pipeline, "audio_vae_temporal_compression_ratio", None),
        "pipeline.audio_vae_temporal_compression_ratio",
    )
    mel_ratio = _positive_int(
        getattr(pipeline, "audio_vae_mel_compression_ratio", None),
        "pipeline.audio_vae_mel_compression_ratio",
    )
    if mel_bins % mel_ratio:
        raise ValueError(
            "LTX2 AudioVAE mel bins must be divisible by mel compression ratio, "
            f"received mel_bins={mel_bins}, ratio={mel_ratio}"
        )
    pipeline_sample_rate = _positive_int(
        getattr(pipeline, "audio_sampling_rate", None),
        "pipeline.audio_sampling_rate",
    )
    pipeline_hop_length = _positive_int(
        getattr(pipeline, "audio_hop_length", None),
        "pipeline.audio_hop_length",
    )
    if (sample_rate, hop_length) != (pipeline_sample_rate, pipeline_hop_length):
        raise ValueError(
            "LTX2 pipeline and AudioVAE audio clocks disagree: "
            f"pipeline={(pipeline_sample_rate, pipeline_hop_length)}, "
            f"audio_vae={(sample_rate, hop_length)}"
        )

    duration_seconds = video.num_frames / video.frame_rate
    target_samples = max(round(duration_seconds * sample_rate), 1)
    latent_frames = round(duration_seconds * sample_rate / hop_length / temporal_ratio)
    if latent_frames < 1:
        raise ValueError(
            "LTX2 configured AV duration resolves to no audio latent frames: "
            f"duration={duration_seconds}, sample_rate={sample_rate}, "
            f"hop_length={hop_length}, temporal_ratio={temporal_ratio}"
        )
    latent_mel_bins = mel_bins // mel_ratio
    feature_dim = latent_channels * latent_mel_bins
    transformer_config = _component_config(adapter, "transformer")
    transformer_video_dim = _positive_int(
        getattr(transformer_config, "in_channels", None),
        "transformer.config.in_channels",
    )
    transformer_audio_dim = _positive_int(
        getattr(transformer_config, "audio_in_channels", None),
        "transformer.config.audio_in_channels",
    )
    if video.feature_dim != transformer_video_dim:
        raise ValueError(
            "LTX2 packed video feature width disagrees with transformer config: "
            f"expected {transformer_video_dim}, resolved {video.feature_dim}"
        )
    if feature_dim != transformer_audio_dim:
        raise ValueError(
            "LTX2 packed audio feature width disagrees with transformer config: "
            f"expected {transformer_audio_dim}, resolved {feature_dim}"
        )
    audio_patch_size = _positive_int(
        getattr(transformer_config, "audio_patch_size", 1),
        "transformer.config.audio_patch_size",
    )
    audio_patch_size_t = _positive_int(
        getattr(transformer_config, "audio_patch_size_t", 1),
        "transformer.config.audio_patch_size_t",
    )
    if (audio_patch_size, audio_patch_size_t) != (1, 1):
        raise ValueError(
            "LTX2 Diffusers 0.40 packs audio as one full-mel token per latent time; "
            "non-unit audio patching is not supported by the online pipeline, received "
            f"{(audio_patch_size, audio_patch_size_t)}"
        )

    audio = LTX2AudioGeometry(
        sample_rate=sample_rate,
        hop_length=hop_length,
        waveform_channels=waveform_channels,
        target_samples=target_samples,
        mel_bins=mel_bins,
        latent_mel_bins=latent_mel_bins,
        latent_channels=latent_channels,
        latent_frames=latent_frames,
        temporal_compression_ratio=temporal_ratio,
        mel_compression_ratio=mel_ratio,
        sequence_length=latent_frames,
        feature_dim=feature_dim,
    )
    return LTX2OutputGeometry(video=video, audio=audio)


def prepare_ltx2_target_video(
    payload: Any,
    *,
    source_fps: Any,
    geometry: LTX2VideoGeometry,
) -> np.ndarray:
    """Select deterministic nearest-time RGB frames on the configured cadence."""
    payload = require_decoded_video_frames(payload, source="LTX2 target video")
    source_fps = _positive_real(source_fps, "target video fps")
    indices = np.rint(
        np.arange(geometry.num_frames, dtype=np.float64) * source_fps / geometry.frame_rate
    ).astype(np.int64)
    if indices[-1] >= payload.shape[0]:
        required_duration = (geometry.num_frames - 1) / geometry.frame_rate
        available_duration = (payload.shape[0] - 1) / source_fps
        raise ValueError(
            "LTX2 target video is too short for configured temporal geometry: "
            f"requires {required_duration:.6f}s, has {available_duration:.6f}s"
        )
    return np.ascontiguousarray(payload[indices])


def prepare_ltx2_target_audio(
    payload: Any,
    *,
    source_sample_rate: Any,
    geometry: LTX2AudioGeometry,
    duration_seconds: float,
) -> torch.Tensor:
    """Convert one waveform to the exact official LTX2 model-rate audio clock."""
    payload = require_decoded_audio_waveform(payload, source="LTX2 target audio")
    if payload.shape[0] not in (1, 2):
        raise ValueError(
            "LTX2 target audio must be mono or stereo, " f"received {payload.shape[0]} channels"
        )
    source_sample_rate = _positive_int(source_sample_rate, "target audio sample_rate")
    duration_seconds = _positive_real(duration_seconds, "target AV duration")
    source_samples = int(duration_seconds * source_sample_rate)
    if source_samples < 1:
        raise ValueError("LTX2 target AV duration resolves to fewer than one source audio sample")
    source_waveform = payload.detach().to(device="cpu", dtype=torch.float32)[:, :source_samples]
    waveform = convert_audio(
        source_waveform,
        from_rate=source_sample_rate,
        to_rate=geometry.sample_rate,
        to_channels=geometry.waveform_channels,
    )
    if waveform.shape[-1] >= geometry.target_samples:
        waveform = waveform[:, : geometry.target_samples]
    else:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, geometry.target_samples - waveform.shape[-1]),
        )
    return waveform.contiguous()


def ltx2_log_mel_spectrogram(
    waveforms: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    mel_bins: int,
) -> torch.Tensor:
    """Apply Lightricks' official magnitude Slaney log-mel frontend.

    The frontend is intentionally independent from the vocoder's inverse-STFT
    helpers: those use a different FFT and hop for bandwidth extension and are
    not the AudioVAE training representation.
    """
    if not isinstance(waveforms, torch.Tensor) or waveforms.ndim != 3:
        raise ValueError(
            "LTX2 log-mel frontend expected waveform tensor shaped (B,C,S), "
            f"received {type(waveforms).__name__}/{getattr(waveforms, 'shape', None)}"
        )
    if not waveforms.is_floating_point():
        raise TypeError(
            f"LTX2 log-mel frontend expected floating waveform, received {waveforms.dtype}"
        )
    _require_finite_tensor(waveforms, "LTX2 log-mel waveform")
    sample_rate = _positive_int(sample_rate, "LTX2 log-mel sample_rate")
    hop_length = _positive_int(hop_length, "LTX2 log-mel hop_length")
    mel_bins = _positive_int(mel_bins, "LTX2 log-mel mel_bins")
    if waveforms.shape[-1] <= 512:
        raise ValueError(
            "LTX2 official centered 1024-point log-mel frontend requires more than "
            f"512 waveform samples for reflect padding, received {waveforms.shape[-1]}"
        )
    mel_spectrogram = getattr(getattr(torchaudio, "transforms", None), "MelSpectrogram", None)
    if not callable(mel_spectrogram):
        raise RuntimeError(
            "LTX2 target audio encoding requires torchaudio.transforms.MelSpectrogram"
        )
    frontend = mel_spectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=hop_length,
        f_min=0.0,
        f_max=sample_rate / 2,
        n_mels=mel_bins,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
    ).to(device=waveforms.device, dtype=torch.float32)
    magnitude_mel = frontend(waveforms.to(torch.float32))
    log_mel = magnitude_mel.clamp_min(1e-5).log().permute(0, 1, 3, 2).contiguous()
    _require_finite_tensor(log_mel, "LTX2 log-mel spectrogram")
    return log_mel


def encode_ltx2_target_video(
    adapter: Any,
    videos: list[np.ndarray],
    geometry: LTX2VideoGeometry,
) -> torch.Tensor:
    """Preprocess videos and take the deterministic VideoVAE posterior mode."""
    pixels = adapter.pipeline.video_processor.preprocess_video(
        [
            decoded_video_to_unit_float(video, source="LTX2 sampled target video")
            for video in videos
        ],
        height=geometry.height,
        width=geometry.width,
    )
    require_finite_bcfhw_video(
        pixels,
        source="LTX2 video_processor.preprocess_video",
        batch_size=len(videos),
        frames=geometry.num_frames,
        height=geometry.height,
        width=geometry.width,
    )
    vae = adapter.get_component("vae")
    encoded = vae.encode(
        pixels.to(
            device=adapter.device,
            dtype=_floating_module_dtype(vae, "LTX2 VideoVAE"),
        )
    )
    latents = retrieve_vae_latents(
        encoded,
        sample_mode="argmax",
        source="LTX2 target video",
    ).to(device=adapter.device, dtype=torch.float32)
    expected_latent_shape = (
        len(videos),
        geometry.latent_channels,
        geometry.latent_frames,
        geometry.latent_height,
        geometry.latent_width,
    )
    if tuple(latents.shape) != expected_latent_shape:
        raise ValueError(
            "LTX2 target VideoVAE latent geometry mismatch: "
            f"expected {expected_latent_shape}, received {tuple(latents.shape)}"
        )
    _require_finite_tensor(latents, "LTX2 target video latents")
    return latents


def normalize_and_pack_ltx2_video(
    adapter: Any,
    latents: torch.Tensor,
    geometry: LTX2VideoGeometry,
) -> torch.Tensor:
    """Apply Diffusers' LTX2 video normalization and patch packing."""
    vae = adapter.get_component("vae")
    scaling_factor = _positive_real(
        getattr(getattr(vae, "config", None), "scaling_factor", None),
        "vae.config.scaling_factor",
    )
    normalized = adapter.pipeline._normalize_latents(
        latents.to(torch.float32),
        vae.latents_mean,
        vae.latents_std,
        scaling_factor,
    )
    packed = adapter.pipeline._pack_latents(
        normalized,
        geometry.patch_size,
        geometry.patch_size_t,
    ).to(device=adapter.device, dtype=torch.float32)
    expected_shape = (
        latents.shape[0],
        geometry.sequence_length,
        geometry.feature_dim,
    )
    if tuple(packed.shape) != expected_shape:
        raise ValueError(
            "LTX2 packed target video geometry mismatch: "
            f"expected {expected_shape}, received {tuple(packed.shape)}"
        )
    _require_finite_tensor(packed, "LTX2 packed target video")
    return packed


def encode_ltx2_target_audio(
    adapter: Any,
    waveforms: torch.Tensor,
    geometry: LTX2AudioGeometry,
) -> torch.Tensor:
    """Apply the official frontend, AudioVAE mode, conformance, packing, and normalization."""
    expected_waveform_shape = (
        waveforms.shape[0] if waveforms.ndim == 3 else None,
        geometry.waveform_channels,
        geometry.target_samples,
    )
    if waveforms.ndim != 3 or tuple(waveforms.shape) != expected_waveform_shape:
        raise ValueError(
            "LTX2 target waveform batch geometry mismatch: "
            f"expected {expected_waveform_shape}, received {tuple(waveforms.shape)}"
        )
    log_mel = ltx2_log_mel_spectrogram(
        waveforms.to(device="cpu", dtype=torch.float32),
        sample_rate=geometry.sample_rate,
        hop_length=geometry.hop_length,
        mel_bins=geometry.mel_bins,
    )
    audio_vae = adapter.get_component("audio_vae")
    encoded = audio_vae.encode(
        log_mel.to(
            device=adapter.device,
            dtype=_floating_module_dtype(audio_vae, "LTX2 AudioVAE"),
        )
    )
    latents = retrieve_vae_latents(
        encoded,
        sample_mode="argmax",
        source="LTX2 target audio",
    ).to(device=adapter.device, dtype=torch.float32)
    if (
        latents.ndim != 4
        or latents.shape[0] != waveforms.shape[0]
        or latents.shape[1] != geometry.latent_channels
        or latents.shape[3] != geometry.latent_mel_bins
    ):
        raise ValueError(
            "LTX2 target AudioVAE latent geometry mismatch: expected "
            f"(B={waveforms.shape[0]}, C={geometry.latent_channels}, T, "
            f"M={geometry.latent_mel_bins}), received {tuple(latents.shape)}"
        )
    if latents.shape[2] >= geometry.latent_frames:
        latents = latents[:, :, : geometry.latent_frames]
    else:
        latents = torch.nn.functional.pad(
            latents,
            (0, 0, 0, geometry.latent_frames - latents.shape[2]),
        )
    packed = adapter.pipeline._pack_audio_latents(latents)
    normalized = adapter.pipeline._normalize_audio_latents(
        packed,
        audio_vae.latents_mean,
        audio_vae.latents_std,
    ).to(device=adapter.device, dtype=torch.float32)
    expected_shape = (
        waveforms.shape[0],
        geometry.sequence_length,
        geometry.feature_dim,
    )
    if tuple(normalized.shape) != expected_shape:
        raise ValueError(
            "LTX2 packed target audio geometry mismatch: "
            f"expected {expected_shape}, received {tuple(normalized.shape)}"
        )
    _require_finite_tensor(normalized, "LTX2 packed target audio")
    return normalized


def validate_ltx2_encoded_output_geometry(
    adapter: Any,
    media_batch: DecodedMediaBatch,
    condition: Mapping[str, Any],
    encoded: EncodedOutputState,
    *,
    conditioned: bool,
) -> None:
    """Validate self-reported codec geometry against runtime configs and I2AV binding."""
    geometry = resolve_ltx2_output_geometry(adapter, conditioned=conditioned)
    batch_size = len(media_batch)
    expected_video_shape = (
        batch_size,
        geometry.video.sequence_length,
        geometry.video.feature_dim,
    )
    expected_audio_shape = (
        batch_size,
        geometry.audio.sequence_length,
        geometry.audio.feature_dim,
    )
    video = encoded.clean_state.components["video"]
    audio = encoded.clean_state.components["audio"]
    if tuple(video.shape) != expected_video_shape or tuple(audio.shape) != expected_audio_shape:
        raise ValueError(
            "LTX2 encoded component geometry mismatch: expected "
            f"video={expected_video_shape}, audio={expected_audio_shape}; received "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )

    expected_forward = _ltx2_forward_context(geometry)
    for key, value in expected_forward.items():
        if encoded.forward_context.get(key) != value:
            raise ValueError(
                f"LTX2 encoded forward_context[{key!r}] mismatch: "
                f"expected {value!r}, received {encoded.forward_context.get(key)!r}"
            )
    expected_decode = {
        "height": geometry.video.height,
        "width": geometry.video.width,
        "num_frames": geometry.video.num_frames,
        "frame_rate": geometry.video.frame_rate,
    }
    for key, value in expected_decode.items():
        if encoded.decode_context.get(key) != value:
            raise ValueError(
                f"LTX2 encoded decode_context[{key!r}] mismatch: "
                f"expected {value!r}, received {encoded.decode_context.get(key)!r}"
            )

    expected_signature = GeometrySignature(
        media=(
            MediaGeometry(
                type=MediaType.VIDEO,
                height=geometry.video.height,
                width=geometry.video.width,
                frames=geometry.video.num_frames,
                fps=geometry.video.frame_rate,
            ),
            MediaGeometry(
                type=MediaType.AUDIO,
                samples=geometry.audio.target_samples,
                sample_rate=geometry.audio.sample_rate,
            ),
        )
    )
    if encoded.geometry_signatures != tuple(expected_signature for _ in media_batch):
        raise ValueError("LTX2 encoded geometry signatures do not match the configured AV geometry")

    if not conditioned:
        if encoded.clean_state.active_masks is not None:
            raise ValueError("LTX2 T2AV clean targets must not carry active masks")
        return

    condition_latents = condition.get("condition_video_latents")
    expected_condition_shape = (
        batch_size,
        geometry.video.latent_channels,
        1,
        geometry.video.latent_height,
        geometry.video.latent_width,
    )
    if (
        not isinstance(condition_latents, torch.Tensor)
        or tuple(condition_latents.shape) != expected_condition_shape
    ):
        raise ValueError(
            "LTX2 I2AV encoded output is not bound to the prepared first-frame condition: "
            f"expected {expected_condition_shape}, received "
            f"{getattr(condition_latents, 'shape', None)}"
        )
    conditioning_mask = encoded.forward_context.get("conditioning_mask")
    if not isinstance(conditioning_mask, torch.Tensor) or tuple(conditioning_mask.shape) != (
        batch_size,
        geometry.video.sequence_length,
    ):
        raise ValueError(
            "LTX2 I2AV conditioning_mask geometry mismatch: expected "
            f"{(batch_size, geometry.video.sequence_length)}, received "
            f"{getattr(conditioning_mask, 'shape', None)}"
        )
    first_frame_tokens = (
        geometry.video.latent_height
        // geometry.video.patch_size
        * (geometry.video.latent_width // geometry.video.patch_size)
    )
    expected_conditioning_mask = torch.zeros_like(conditioning_mask, dtype=torch.bool)
    expected_conditioning_mask[:, :first_frame_tokens] = True
    if not torch.equal(conditioning_mask.bool(), expected_conditioning_mask):
        raise ValueError(
            "LTX2 I2AV conditioning_mask must pin exactly the first latent video frame"
        )
    active_masks = encoded.clean_state.active_masks
    if active_masks is None:
        raise ValueError("LTX2 I2AV clean targets require video/audio active masks")
    if not torch.equal(
        active_masks["video"].reshape_as(conditioning_mask),
        ~expected_conditioning_mask,
    ):
        raise ValueError("LTX2 I2AV video active mask must be the inverse conditioning mask")
    if not bool(active_masks["audio"].all()):
        raise ValueError("LTX2 I2AV audio target must remain fully active")


def reduce_ltx2_flow_matching_objective_values(
    adapter: Any,
    values: Mapping[str, torch.Tensor],
    *,
    state: Optional[LatentState],
) -> torch.Tensor:
    """Sum official per-modality means only for the offline flow objective.

    LTX2 trains the video and audio denoisers as two equally weighted terms.
    Keeping this policy behind the flow-matching-specific adapter hook preserves
    the element-weighted joint likelihood used by online RL and distillation.
    """
    component_means = adapter.reduce_component_latent_values(values, state=state)
    return component_means["video"] + component_means["audio"]


def decode_ltx2_output_state(
    adapter: Any,
    encoded: EncodedOutputState,
    *,
    output_type: Literal["pil", "pt", "np"],
) -> Any:
    """Route the shared video/audio state through the existing LTX2 decoder."""
    if encoded.clean_state.component_names != ("video", "audio"):
        raise ValueError(
            "LTX2 output decoding requires component order ('video', 'audio'), "
            f"received {encoded.clean_state.component_names}"
        )
    context = encoded.decode_context
    return adapter.decode_latents(
        encoded.clean_state.components["video"],
        encoded.clean_state.components["audio"],
        height=context["height"],
        width=context["width"],
        num_frames=context["num_frames"],
        frame_rate=context["frame_rate"],
        output_type=output_type,
    )


def _ltx2_forward_context(geometry: LTX2OutputGeometry) -> dict[str, Any]:
    return {
        "height": geometry.video.height,
        "width": geometry.video.width,
        "num_frames": geometry.video.num_frames,
        "frame_rate": geometry.video.frame_rate,
        "video_seq_len": geometry.video.sequence_length,
        "audio_num_frames": geometry.audio.latent_frames,
    }


def _component_config(adapter: Any, name: str) -> Any:
    getter = getattr(adapter, "get_component_config", None)
    if callable(getter):
        return getter(name)
    component = adapter.get_component(name)
    config = getattr(component, "config", None)
    if config is None:
        raise TypeError(f"LTX2 component {name!r} does not expose config")
    return config


def _floating_module_dtype(module: Any, identifier: str) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return dtype
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        first = next(iter(parameters()), None)
        if isinstance(first, torch.Tensor) and first.dtype.is_floating_point:
            return first.dtype
    raise TypeError(f"{identifier} must expose a floating dtype, received {dtype!r}")


def _positive_int(value: Any, identifier: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"expected positive int for {identifier}, received "
            f"{type(value).__name__}: {value!r}"
        )
    if value <= 0:
        raise ValueError(f"expected positive int for {identifier}, received {value}")
    return value


def _positive_real(value: Any, identifier: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"expected positive real for {identifier}, received "
            f"{type(value).__name__}: {value!r}"
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"expected positive finite real for {identifier}, received {value!r}")
    return value


def _require_finite_tensor(value: torch.Tensor, identifier: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{identifier} contains non-finite values")


__all__ = [
    "LTX2AVOutputCodec",
    "LTX2FirstFrameConditionPreparer",
    "LTX2OutputGeometry",
    "LTX2_OFFLINE_FORWARD_OVERRIDES",
    "decode_ltx2_output_state",
    "ltx2_log_mel_spectrogram",
    "reduce_ltx2_flow_matching_objective_values",
    "resolve_ltx2_output_geometry",
    "resolve_ltx2_video_geometry",
    "validate_ltx2_encoded_output_geometry",
]
