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

"""CPU tests for LTX2's config-driven offline audiovisual boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
import torchaudio
from diffusers.video_processor import VideoProcessor

from flow_factory.contracts import BatchCapability, GeometrySource, MediaType
from flow_factory.models.ltx2._output import (
    LTX2AVOutputCodec,
    LTX2FirstFrameConditionPreparer,
    decode_ltx2_output_state,
    encode_ltx2_target_video,
    ltx2_log_mel_spectrogram,
    resolve_ltx2_output_geometry,
    validate_ltx2_encoded_output_geometry,
)
from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.models.output_state import validate_encoded_output_state

BATCH_SIZE = 2
HEIGHT = 8
WIDTH = 8
NUM_FRAMES = 3
FRAME_RATE = 4.0
VIDEO_LATENT_CHANNELS = 2
LATENT_FRAMES = 2
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
VIDEO_SEQUENCE_LENGTH = LATENT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH
AUDIO_SAMPLE_RATE = 8000
AUDIO_HOP_LENGTH = 80
AUDIO_MEL_BINS = 8
AUDIO_LATENT_CHANNELS = 2
AUDIO_LATENT_MEL_BINS = 4
AUDIO_FEATURE_DIM = AUDIO_LATENT_CHANNELS * AUDIO_LATENT_MEL_BINS
AUDIO_TARGET_SAMPLES = 6000
AUDIO_LATENT_FRAMES = 38


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _ModeOnlyPosterior:
    """Posterior fake proving target and condition codecs never sample."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.mode_calls = 0
        self.sample_calls = 0

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self.value

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        del generator
        self.sample_calls += 1
        raise AssertionError("LTX2 offline targets must use posterior mode")


class _VideoVAEFake:
    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(latent_channels=VIDEO_LATENT_CHANNELS, scaling_factor=2.0)
        self.latents_mean = torch.tensor([0.5, -0.5])
        self.latents_std = torch.tensor([2.0, 4.0])
        self.posteriors: list[_ModeOnlyPosterior] = []
        self.encoded_pixels: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> Any:
        self.encoded_pixels.append(pixels)
        if pixels.shape[2] == 1:
            latents = pixels[:, :VIDEO_LATENT_CHANNELS, :, ::2, ::2]
        else:
            latents = pixels[:, :VIDEO_LATENT_CHANNELS, ::2, ::2, ::2]
        posterior = _ModeOnlyPosterior(latents + 0.25)
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


class _AudioVAEFake:
    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            sample_rate=AUDIO_SAMPLE_RATE,
            mel_hop_length=AUDIO_HOP_LENGTH,
            mel_bins=AUDIO_MEL_BINS,
            in_channels=2,
            latent_channels=AUDIO_LATENT_CHANNELS,
        )
        self.latents_mean = torch.linspace(-0.4, 0.3, AUDIO_FEATURE_DIM)
        self.latents_std = torch.linspace(1.0, 1.7, AUDIO_FEATURE_DIM)
        self.posteriors: list[_ModeOnlyPosterior] = []

    def encode(self, log_mel: torch.Tensor) -> Any:
        latents = log_mel[:, :AUDIO_LATENT_CHANNELS, ::2, ::2]
        posterior = _ModeOnlyPosterior(latents)
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


class _VideoProcessorFake:
    def __init__(self) -> None:
        self.videos: list[list[np.ndarray]] = []

    def preprocess_video(
        self,
        videos: list[np.ndarray],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        assert (height, width) == (HEIGHT, WIDTH)
        self.videos.append(videos)
        stacked = np.stack(videos)
        return torch.from_numpy(stacked).permute(0, 4, 1, 2, 3).float().mul(2.0).sub(1.0)


class _PipelineFake:
    vae_spatial_compression_ratio = 2
    vae_temporal_compression_ratio = 2
    transformer_spatial_patch_size = 1
    transformer_temporal_patch_size = 1
    audio_vae_temporal_compression_ratio = 2
    audio_vae_mel_compression_ratio = 2
    audio_sampling_rate = AUDIO_SAMPLE_RATE
    audio_hop_length = AUDIO_HOP_LENGTH

    def __init__(self) -> None:
        self.video_processor = _VideoProcessorFake()

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        mean = latents_mean.view(1, -1, 1, 1, 1).to(latents)
        std = latents_std.view(1, -1, 1, 1, 1).to(latents)
        return (latents - mean) * scaling_factor / std

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = latents.shape
        latents = latents.reshape(
            batch,
            -1,
            frames // patch_size_t,
            patch_size_t,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        batch = latents.shape[0]
        latents = latents.reshape(
            batch,
            num_frames,
            height,
            width,
            -1,
            patch_size_t,
            patch_size,
            patch_size,
        )
        return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)

    @staticmethod
    def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
        return latents.transpose(1, 2).flatten(2, 3)

    @staticmethod
    def _normalize_audio_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        return (latents - latents_mean.to(latents)) / latents_std.to(latents)


class _AdapterFake:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.training_args = SimpleNamespace(
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=FRAME_RATE,
        )
        self.pipeline = _PipelineFake()
        self.components = {
            "vae": _VideoVAEFake(),
            "audio_vae": _AudioVAEFake(),
            "transformer": SimpleNamespace(
                config=SimpleNamespace(
                    in_channels=VIDEO_LATENT_CHANNELS,
                    audio_in_channels=AUDIO_FEATURE_DIM,
                    audio_patch_size=1,
                    audio_patch_size_t=1,
                    # LTX-2.3-only topology flags do not alter the AV target contract.
                    gated_attn=True,
                    cross_attn_mod=True,
                )
            ),
        }
        self.decode_calls: list[tuple[Any, ...]] = []

    def get_component(self, name: str) -> Any:
        return self.components[name]

    def get_component_config(self, name: str) -> Any:
        return self.components[name].config

    def decode_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.decode_calls.append((video_latents, audio_latents, kwargs))
        return video_latents, audio_latents


def _media_batch() -> tuple[tuple[_DecodedMedia, _DecodedMedia], ...]:
    samples = []
    time = torch.arange(AUDIO_TARGET_SAMPLES, dtype=torch.float32) / AUDIO_SAMPLE_RATE
    for index in range(BATCH_SIZE):
        frames = np.full(
            (NUM_FRAMES, HEIGHT, WIDTH, 3),
            fill_value=32 + index * 48,
            dtype=np.uint8,
        )
        waveform = torch.sin(2 * torch.pi * (220 + index * 30) * time).unsqueeze(0)
        samples.append(
            (
                _DecodedMedia(type="video", payload=frames, fps=FRAME_RATE),
                _DecodedMedia(
                    type="audio",
                    payload=waveform,
                    sample_rate=AUDIO_SAMPLE_RATE,
                ),
            )
        )
    return tuple(samples)


def _condition_images() -> torch.Tensor:
    values = torch.linspace(-0.8, 0.8, BATCH_SIZE * 3 * HEIGHT * WIDTH)
    return values.reshape(BATCH_SIZE, 3, HEIGHT, WIDTH)


def test_official_lightricks_log_mel_frontend_matches_torchaudio() -> None:
    waveforms = torch.linspace(-1.0, 1.0, 2 * 2048).reshape(1, 2, 2048)

    actual = ltx2_log_mel_spectrogram(
        waveforms,
        sample_rate=AUDIO_SAMPLE_RATE,
        hop_length=AUDIO_HOP_LENGTH,
        mel_bins=AUDIO_MEL_BINS,
    )
    frontend = torchaudio.transforms.MelSpectrogram(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_fft=1024,
        win_length=1024,
        hop_length=AUDIO_HOP_LENGTH,
        f_min=0.0,
        f_max=AUDIO_SAMPLE_RATE / 2,
        n_mels=AUDIO_MEL_BINS,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
    )
    expected = frontend(waveforms).clamp_min(1e-5).log().permute(0, 1, 3, 2)

    assert torch.equal(actual, expected)


def test_t2av_codec_encodes_config_driven_joint_mode_state() -> None:
    adapter = _AdapterFake()
    geometry = resolve_ltx2_output_geometry(adapter, conditioned=False)

    with torch.no_grad():
        encoded = LTX2AVOutputCodec(adapter).encode_output_state(_media_batch(), {})

    assert geometry.video.sequence_length == VIDEO_SEQUENCE_LENGTH
    assert geometry.audio.latent_frames == AUDIO_LATENT_FRAMES
    assert geometry.audio.target_samples == AUDIO_TARGET_SAMPLES
    assert encoded.clean_state.component_names == ("video", "audio")
    assert encoded.clean_state.components["video"].shape == (
        BATCH_SIZE,
        VIDEO_SEQUENCE_LENGTH,
        VIDEO_LATENT_CHANNELS,
    )
    assert encoded.clean_state.components["audio"].shape == (
        BATCH_SIZE,
        AUDIO_LATENT_FRAMES,
        AUDIO_FEATURE_DIM,
    )
    assert encoded.clean_state.active_masks is None
    assert encoded.forward_context["video_seq_len"] == VIDEO_SEQUENCE_LENGTH
    assert encoded.forward_context["audio_num_frames"] == AUDIO_LATENT_FRAMES
    assert encoded.geometry_signatures[0].media[0].type is MediaType.VIDEO
    assert encoded.geometry_signatures[0].media[1].type is MediaType.AUDIO
    assert encoded.geometry_signatures[0].media[1].samples == AUDIO_TARGET_SAMPLES
    assert all(posterior.mode_calls == 1 for posterior in adapter.components["vae"].posteriors)
    assert all(
        posterior.mode_calls == 1 for posterior in adapter.components["audio_vae"].posteriors
    )

    validate_encoded_output_state(
        encoded,
        contract=LTX2_T2AV_Adapter.pipeline_io_contract,
        expected_component_order=("video", "audio"),
        expected_batch_size=BATCH_SIZE,
        device="cpu",
    )
    validate_ltx2_encoded_output_geometry(
        adapter,
        _media_batch(),
        {},
        encoded,
        conditioned=False,
    )


@pytest.mark.parametrize("channels", [(0, 0, 0), (255, 255, 255), (0, 127, 255)])
def test_ltx2_target_video_normalizes_pixels_with_real_video_processor(
    channels: tuple[int, int, int],
) -> None:
    """Cross the decoded-byte boundary once before Diffusers VAE normalization."""
    adapter = _AdapterFake()
    adapter.pipeline.video_processor = VideoProcessor(vae_scale_factor=2)
    geometry = resolve_ltx2_output_geometry(adapter, conditioned=False).video
    source = np.broadcast_to(
        np.array(channels, dtype=np.uint8),
        (NUM_FRAMES, HEIGHT, WIDTH, 3),
    ).copy()
    original = source.copy()

    encode_ltx2_target_video(adapter, [source], geometry)

    pixels = adapter.components["vae"].encoded_pixels[-1]
    expected = torch.tensor(channels, dtype=torch.float32) / 255.0 * 2.0 - 1.0
    assert pixels.shape == (1, 3, NUM_FRAMES, HEIGHT, WIDTH)
    torch.testing.assert_close(pixels, expected.view(1, 3, 1, 1, 1).expand_as(pixels))
    np.testing.assert_array_equal(source, original)


def test_i2av_preparer_binds_and_masks_the_exact_first_latent_frame() -> None:
    adapter = _AdapterFake()
    cached_condition = {
        "connector_prompt_embeds": torch.zeros(BATCH_SIZE, 2, 3),
        "condition_images": _condition_images(),
    }

    with torch.no_grad():
        prepared = LTX2FirstFrameConditionPreparer(adapter).prepare_condition_state(
            cached_condition
        )
        encoded = LTX2AVOutputCodec(adapter, conditioned=True).encode_output_state(
            _media_batch(),
            prepared.output_codec_condition(),
        )

    assert "condition_images" not in prepared.condition
    assert prepared.forward_context == {}
    condition_latents = prepared.output_context["condition_video_latents"]
    assert condition_latents.shape == (
        BATCH_SIZE,
        VIDEO_LATENT_CHANNELS,
        1,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    conditioning_mask = encoded.forward_context["conditioning_mask"]
    first_frame_tokens = LATENT_HEIGHT * LATENT_WIDTH
    assert torch.equal(
        conditioning_mask[:, :first_frame_tokens],
        torch.ones(BATCH_SIZE, first_frame_tokens),
    )
    assert not bool(conditioning_mask[:, first_frame_tokens:].any())
    assert torch.equal(
        encoded.clean_state.active_masks["video"].squeeze(-1),
        ~conditioning_mask.bool(),
    )
    assert bool(encoded.clean_state.active_masks["audio"].all())

    unpacked = adapter.pipeline._unpack_latents(
        encoded.clean_state.components["video"],
        LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    expected_first_frame = adapter.pipeline._normalize_latents(
        condition_latents,
        adapter.components["vae"].latents_mean,
        adapter.components["vae"].latents_std,
        adapter.components["vae"].config.scaling_factor,
    )
    assert torch.equal(unpacked[:, :, :1], expected_first_frame)

    validate_encoded_output_state(
        encoded,
        contract=LTX2_I2AV_Adapter.pipeline_io_contract,
        expected_component_order=("video", "audio"),
        expected_batch_size=BATCH_SIZE,
        device="cpu",
    )
    validate_ltx2_encoded_output_geometry(
        adapter,
        _media_batch(),
        prepared.output_codec_condition(),
        encoded,
        conditioned=True,
    )


def test_shared_decoder_routes_both_components_and_configured_geometry() -> None:
    adapter = _AdapterFake()
    with torch.no_grad():
        encoded = LTX2AVOutputCodec(adapter).encode_output_state(_media_batch(), {})

    decoded = decode_ltx2_output_state(adapter, encoded, output_type="pt")

    assert decoded == (
        encoded.clean_state.components["video"],
        encoded.clean_state.components["audio"],
    )
    assert adapter.decode_calls[0][2] == {
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "frame_rate": FRAME_RATE,
        "output_type": "pt",
    }


def test_adapters_declare_complete_offline_av_capability() -> None:
    LTX2_T2AV_Adapter.validate_offline_output_capability()
    LTX2_I2AV_Adapter.validate_offline_output_capability()

    assert LTX2_T2AV_Adapter.component_load_dtype_defaults == {"audio_vae": torch.float32}
    assert LTX2_I2AV_Adapter.component_load_dtype_defaults == {"audio_vae": torch.float32}
    assert LTX2_T2AV_Adapter.pipeline_io_contract.geometry_source is GeometrySource.CONFIGURED
    assert LTX2_I2AV_Adapter.pipeline_io_contract.geometry_source is GeometrySource.CONFIGURED
    assert LTX2_T2AV_Adapter.pipeline_io_contract.batch_capability is BatchCapability.UNIFORM
    assert LTX2_I2AV_Adapter.pipeline_io_contract.batch_capability is BatchCapability.UNIFORM
    assert LTX2_T2AV_Adapter.pipeline_io_contract.input_media.rules == ()
    assert LTX2_I2AV_Adapter.pipeline_io_contract.input_media.rules[0].min_count == 1
    assert LTX2_I2AV_Adapter.pipeline_io_contract.input_media.rules[0].max_count == 1
    assert LTX2_T2AV_Adapter.output_state_codec_unavailable_reason is None
    assert LTX2_I2AV_Adapter.output_state_codec_unavailable_reason is None
    assert LTX2_T2AV_Adapter.offline_training_forward_overrides == {
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

    with pytest.raises(TypeError):
        LTX2_T2AV_Adapter.offline_training_forward_overrides["guidance_scale"] = 2.0


@pytest.mark.parametrize(
    ("adapter_class", "conditioned"),
    [(LTX2_T2AV_Adapter, False), (LTX2_I2AV_Adapter, True)],
)
def test_offline_flow_objective_sums_modalities_without_changing_joint_reducer(
    adapter_class: type,
    conditioned: bool,
) -> None:
    codec_adapter = _AdapterFake()
    condition: Any = {}
    if conditioned:
        condition = LTX2FirstFrameConditionPreparer(codec_adapter).prepare_condition_state(
            {"condition_images": _condition_images()}
        )
        condition = condition.output_codec_condition()
    with torch.no_grad():
        encoded = LTX2AVOutputCodec(codec_adapter, conditioned=conditioned).encode_output_state(
            _media_batch(),
            condition,
        )
    values = {
        "video": torch.full_like(encoded.clean_state.components["video"], 2.0),
        "audio": torch.full_like(encoded.clean_state.components["audio"], 5.0),
    }
    if conditioned:
        values["video"] = values["video"].masked_fill(
            ~encoded.clean_state.active_masks["video"],
            1000.0,
        )
    adapter = object.__new__(adapter_class)

    offline = adapter.reduce_flow_matching_objective_values(
        values,
        state=encoded.clean_state,
    )
    joint = adapter.reduce_latent_values(values, state=encoded.clean_state)

    assert torch.equal(offline, torch.full((BATCH_SIZE,), 7.0))
    assert not torch.equal(joint, offline)
