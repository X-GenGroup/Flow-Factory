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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor
from PIL import Image

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaOrder,
    MediaType,
    RateRequirement,
)
from flow_factory.data_utils.offline_condition_cache import build_offline_condition_cache
from flow_factory.data_utils.offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    DecodedMedia,
    _collate_condition_mappings,
)
from flow_factory.data_utils.schema import normalize_v2_record
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.wan._conditioning import (
    WanI2VConditionStatePreparer,
    normalize_wan_i2v_image_rows,
    normalize_wan_image_embeds,
    split_wan_image_embeds,
)
from flow_factory.models.wan._output import WanVideoOutputCodec
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter, WanI2VSample
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter


class _Posterior:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latents = latents
        self.generators: list[torch.Generator | None] = []
        self.mode_calls = 0

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        self.generators.append(generator)
        return self.latents

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self.latents


class _VAE:
    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            z_dim=3,
            latents_mean=[1.0, 2.0, 3.0],
            latents_std=[2.0, 4.0, 5.0],
        )
        raw_channels = torch.tensor([3.0, 6.0, 8.0]).view(1, 3, 1, 1, 1)
        self.full_latents = raw_channels.expand(1, 3, 2, 2, 2).clone()
        self.posterior = _Posterior(self.full_latents)
        self.encoded_pixels: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> Any:
        self.encoded_pixels.append(pixels)
        self.posterior.latents = (
            self.full_latents[:, :, :1] if pixels.shape[2] == 1 else self.full_latents
        )
        return SimpleNamespace(latent_dist=self.posterior)


class _VideoProcessor:
    def __init__(self) -> None:
        self.videos: list[list[np.ndarray]] = []
        self.images: list[list[Image.Image]] = []

    def preprocess(
        self,
        images: list[Image.Image],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.images.append(images)
        channels = [
            torch.tensor(np.asarray(image, dtype=np.float32)[0, 0] / 255.0)
            .view(3, 1, 1)
            .expand(3, height, width)
            for image in images
        ]
        return torch.stack(channels, dim=0)

    def preprocess_video(
        self,
        videos: list[np.ndarray],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.videos.append(videos)
        return torch.zeros(len(videos), 3, videos[0].shape[0], height, width)


class _ProcessorOutput(dict):
    def to(self, device: torch.device) -> "_ProcessorOutput":
        del device
        return self


class _ImageProcessor:
    def __call__(self, *, images: list[Image.Image], return_tensors: str) -> _ProcessorOutput:
        assert return_tensors == "pt"
        return _ProcessorOutput(pixel_values=torch.zeros(len(images), 3, 2, 2))


class _ImageEncoder:
    device = torch.device("cpu")

    def __call__(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool,
    ) -> Any:
        assert output_hidden_states is True
        count = pixel_values.shape[0]
        embeds = torch.arange(count * 3 * 5, dtype=torch.float32).view(count, 3, 5)
        return SimpleNamespace(hidden_states=(embeds, torch.zeros_like(embeds)))


class _Adapter:
    _configured_video_output_geometry = Wan2_T2V_Adapter._configured_video_output_geometry
    _resample_output_video = staticmethod(Wan2_T2V_Adapter._resample_output_video)
    _normalize_output_video_latents = Wan2_T2V_Adapter._normalize_output_video_latents

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.training_args = SimpleNamespace(
            height=16,
            width=16,
            num_frames=5,
            frame_rate=4.0,
        )
        self.vae = _VAE()
        self.pipeline = SimpleNamespace(
            vae_scale_factor_temporal=4,
            vae_scale_factor_spatial=8,
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
            transformer_2=None,
            video_processor=_VideoProcessor(),
            config=SimpleNamespace(expand_timesteps=False),
        )


class _WanPreprocessHarness:
    """Exercise BaseAdapter preprocessing with Wan's real image encoder hook."""

    preprocess_func = BaseAdapter.preprocess_func
    encode_image = Wan2_I2V_Adapter.encode_image
    encode_video = BaseAdapter.encode_video
    encode_audio = BaseAdapter.encode_audio
    python_format_columns = frozenset()
    supports_ordered_references = False

    def __init__(self, *, with_clip: bool = False) -> None:
        self.device = torch.device("cpu")
        self.training_args = SimpleNamespace(height=16, width=16)
        self.image_encoder = _ImageEncoder()
        self.pipeline = SimpleNamespace(
            transformer=SimpleNamespace(config=SimpleNamespace(image_dim=8 if with_clip else None)),
            video_processor=_VideoProcessor(),
            image_processor=_ImageProcessor(),
            image_encoder=self.image_encoder,
        )

    def encode_prompt(self, prompt: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        return {"prompt_embeds": torch.ones(len(prompt), 3, 4)}


def _media(video: np.ndarray, fps: float = 8.0):
    return (
        (
            DecodedMedia(
                type="video",
                path="target.mp4",
                payload=video,
                fps=fps,
            ),
        ),
    )


def _condition_pixels(count: int = 1) -> torch.Tensor:
    pixels = [torch.zeros(3, 16, 16)]
    if count == 2:
        pixels.append(torch.ones(3, 16, 16))
    return torch.stack(pixels, dim=0).unsqueeze(0)


def test_wan_t2v_declares_required_video_output_semantics() -> None:
    contract = Wan2_T2V_Adapter.pipeline_io_contract

    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.output_media.items[0].type is MediaType.VIDEO
    assert contract.output_media.items[0].fps is RateRequirement.REQUIRED
    assert contract.batch_capability is BatchCapability.SINGLE_SAMPLE
    Wan2_T2V_Adapter.validate_offline_output_capability()

    adapter = object.__new__(Wan2_T2V_Adapter)
    codec = adapter.build_output_state_codec()
    assert isinstance(codec, WanVideoOutputCodec)
    assert codec.required_components == ("vae",)


def test_wan_codec_resamples_preprocesses_and_samples_target_latents() -> None:
    adapter = _Adapter()
    codec = WanVideoOutputCodec(adapter)
    source = np.arange(9 * 4 * 4 * 3, dtype=np.uint8).reshape(9, 4, 4, 3)
    generator = torch.Generator().manual_seed(7)

    encoded = codec.encode_output_state(_media(source), {}, generator)

    selected = adapter.pipeline.video_processor.videos[0][0]
    np.testing.assert_allclose(selected, source[[0, 2, 4, 6, 8]].astype(np.float32) / 255.0)
    assert adapter.vae.posterior.generators == [generator]
    torch.testing.assert_close(
        encoded.clean_state.components["latent"],
        torch.ones(1, 3, 2, 2, 2),
    )
    assert encoded.forward_context == {}
    assert dict(encoded.decode_context) == {
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "frame_rate": 4.0,
    }
    geometry = encoded.geometry_signatures[0].media[0]
    assert (geometry.type, geometry.height, geometry.width, geometry.frames, geometry.fps) == (
        MediaType.VIDEO,
        16,
        16,
        5,
        4.0,
    )


@pytest.mark.parametrize(
    ("pixels", "error_type", "message"),
    [
        (torch.zeros(1, 3, 5, 16, 16, dtype=torch.uint8), TypeError, "dtype floating"),
        (torch.full((1, 3, 5, 16, 16), float("nan")), ValueError, "non-finite values"),
        (torch.zeros(1, 4, 5, 16, 16), ValueError, "3 channels in BCFHW"),
    ],
)
def test_wan_codec_rejects_invalid_model_pixel_tensor(
    pixels: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    adapter = _Adapter()
    adapter.pipeline.video_processor.preprocess_video = lambda *args, **kwargs: pixels

    with pytest.raises(error_type, match=message):
        WanVideoOutputCodec(adapter).encode_output_state(
            _media(np.zeros((9, 4, 4, 3), dtype=np.uint8)),
            {},
        )

    assert adapter.vae.encoded_pixels == []


def test_wan_codec_rejects_insufficient_duration_and_invalid_latent_grid() -> None:
    adapter = _Adapter()
    codec = WanVideoOutputCodec(adapter)
    short = np.zeros((8, 4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="too short"):
        codec.encode_output_state(_media(short), {})

    adapter.training_args.num_frames = 6
    with pytest.raises(ValueError, match="num_frames must satisfy"):
        adapter._configured_video_output_geometry()


def test_wan_geometry_validator_rejects_output_context_drift() -> None:
    adapter = _Adapter()
    encoded = WanVideoOutputCodec(adapter).encode_output_state(
        _media(np.zeros((9, 4, 4, 3), dtype=np.uint8)),
        {},
    )

    Wan2_T2V_Adapter._validate_encoded_output_geometry(
        adapter, _media(np.zeros((9, 1, 1, 3), dtype=np.uint8)), {}, encoded
    )

    drifted = type(encoded)(
        clean_state=encoded.clean_state,
        forward_context=encoded.forward_context,
        decode_context={**dict(encoded.decode_context), "frame_rate": 8.0},
        geometry_signatures=encoded.geometry_signatures,
    )
    with pytest.raises(ValueError, match="decode_context 'frame_rate'"):
        Wan2_T2V_Adapter._validate_encoded_output_geometry(
            adapter,
            _media(np.zeros((9, 1, 1, 3), dtype=np.uint8)),
            {},
            drifted,
        )


def test_wan_i2v_resolves_checkpoint_specific_endpoint_contract() -> None:
    contract = Wan2_I2V_Adapter.pipeline_io_contract

    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.batch_capability is BatchCapability.SINGLE_SAMPLE
    assert contract.input_media.order is InputMediaOrder.WITHIN_TYPE
    assert len(contract.input_media.rules) == 1
    rule = contract.input_media.rules[0]
    assert (rule.format.type, rule.min_count, rule.max_count) == (MediaType.IMAGE, 1, 2)
    assert dict(Wan2_I2V_Adapter.offline_training_forward_overrides) == {
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
    }
    Wan2_I2V_Adapter.validate_offline_output_capability()

    adapter = object.__new__(Wan2_I2V_Adapter)
    assert isinstance(adapter.build_condition_state_preparer(), WanI2VConditionStatePreparer)
    codec = adapter.build_output_state_codec()
    assert isinstance(codec, WanVideoOutputCodec)
    assert codec.bind_condition_active_mask is True

    runtime = object.__new__(Wan2_I2V_Adapter)
    runtime.pipeline = _Adapter().pipeline
    runtime.pipeline.transformer.config.image_dim = None
    runtime.pipeline.transformer.config.pos_embed_seq_len = None
    runtime.pipeline.transformer_2 = SimpleNamespace(
        config=SimpleNamespace(image_dim=None, pos_embed_seq_len=None)
    )
    effective = Wan2_I2V_Adapter._resolve_pipeline_io_contract(runtime)
    rule = effective.input_media.rules[0]
    assert (rule.min_count, rule.max_count) == (1, 2)

    runtime.pipeline.transformer_2 = None
    runtime.pipeline.transformer.config.image_dim = 1280
    effective = Wan2_I2V_Adapter._resolve_pipeline_io_contract(runtime)
    rule = effective.input_media.rules[0]
    assert (rule.min_count, rule.max_count) == (1, 1)

    runtime.pipeline.transformer.config.pos_embed_seq_len = 514
    effective = Wan2_I2V_Adapter._resolve_pipeline_io_contract(runtime)
    rule = effective.input_media.rules[0]
    assert (rule.min_count, rule.max_count) == (2, 2)
    assert rule.required_slots == ("first_frame", "last_frame")

    runtime.pipeline.config.expand_timesteps = True
    effective = Wan2_I2V_Adapter._resolve_pipeline_io_contract(runtime)
    assert effective.input_media.rules[0].max_count == 1


def test_wan_i2v_condition_preparer_preserves_order_and_uses_posterior_mode() -> None:
    adapter = _Adapter()
    first = Image.new("RGB", (4, 4), color="red")
    last = Image.new("RGB", (4, 4), color="blue")
    image_embeds = torch.ones(1, 2, 4, 6)

    prepared = WanI2VConditionStatePreparer(adapter).prepare_condition_state(
        {
            "images": [[first, last]],
            "condition_images": _condition_pixels(2),
            "prompt_embeds": torch.ones(1, 3, 5),
            "image_embeds": image_embeds,
        }
    )

    assert tuple(prepared.condition) == ("prompt_embeds",)
    assert tuple(prepared.forward_context) == ("latent_condition", "image_embeds")
    assert prepared.output_context == {}
    assert prepared.forward_context["image_embeds"].shape == (2, 4, 6)
    condition = prepared.forward_context["latent_condition"]
    assert condition.shape == (1, 7, 2, 2, 2)
    torch.testing.assert_close(condition[:, :4, 0], torch.ones(1, 4, 2, 2))
    expected_last_mask = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 4, 1, 1)
    torch.testing.assert_close(
        condition[:, :4, 1],
        expected_last_mask.expand(1, 4, 2, 2),
    )
    assert adapter.vae.posterior.mode_calls == 1
    assert adapter.vae.posterior.generators == []
    encoded_pixels = adapter.vae.encoded_pixels[0]
    torch.testing.assert_close(encoded_pixels[:, :, 0], torch.zeros(1, 3, 16, 16))
    torch.testing.assert_close(encoded_pixels[:, :, -1], torch.ones(1, 3, 16, 16))


def test_wan_i2v_first_only_non_expanded_condition_has_no_output_mask() -> None:
    adapter = _Adapter()
    first = Image.new("RGB", (4, 4))

    prepared = WanI2VConditionStatePreparer(adapter).prepare_condition_state(
        {"images": [[first]], "condition_images": _condition_pixels()}
    )

    condition = prepared.forward_context["latent_condition"]
    torch.testing.assert_close(condition[:, :4, 0], torch.ones(1, 4, 2, 2))
    torch.testing.assert_close(condition[:, :4, 1], torch.zeros(1, 4, 2, 2))
    assert "first_frame_mask" not in prepared.forward_context
    assert prepared.output_context == {}


def test_wan_i2v_expand_condition_binds_target_active_mask() -> None:
    adapter = _Adapter()
    adapter.pipeline.config.expand_timesteps = True
    first = Image.new("RGB", (4, 4))
    prepared = WanI2VConditionStatePreparer(adapter).prepare_condition_state(
        {"images": [[first]], "condition_images": _condition_pixels()}
    )

    condition = prepared.forward_context["latent_condition"]
    mask = prepared.forward_context["first_frame_mask"]
    assert condition.shape == (1, 3, 1, 2, 2)
    assert mask.shape == (1, 1, 2, 2, 2)
    assert adapter.vae.encoded_pixels[0].shape[2] == 1
    rollout_latents = torch.full((1, 3, 2, 2, 2), 2.0)
    blended = (1 - mask) * condition + mask * rollout_latents
    assert blended.shape == rollout_latents.shape
    torch.testing.assert_close(blended[:, :, 0], condition[:, :, 0])
    torch.testing.assert_close(blended[:, :, 1], rollout_latents[:, :, 1])
    torch.testing.assert_close(mask[:, :, 0], torch.zeros(1, 1, 2, 2))
    torch.testing.assert_close(mask[:, :, 1], torch.ones(1, 1, 2, 2))
    assert prepared.output_context["first_frame_mask"] is mask

    source = np.zeros((9, 4, 4, 3), dtype=np.uint8)
    encoded = WanVideoOutputCodec(adapter, bind_condition_active_mask=True).encode_output_state(
        _media(source),
        prepared.output_codec_condition(),
    )

    active_mask = encoded.clean_state.active_masks["latent"]
    assert active_mask.dtype is torch.bool
    torch.testing.assert_close(active_mask, mask.bool())
    assert adapter.vae.posterior.mode_calls == 1
    assert adapter.vae.posterior.generators == [None]


def test_wan_i2v_expand_rejects_last_frame_instead_of_ignoring_it() -> None:
    adapter = _Adapter()
    adapter.pipeline.config.expand_timesteps = True
    first = Image.new("RGB", (4, 4))
    last = Image.new("RGB", (4, 4))

    with pytest.raises(ValueError, match="expand_timesteps does not support"):
        WanI2VConditionStatePreparer(adapter).prepare_condition_state(
            {"images": [[first, last]], "condition_images": _condition_pixels(2)}
        )

    assert adapter.vae.encoded_pixels == []


def test_wan_i2v_expand_target_requires_prepared_active_mask_before_vae() -> None:
    adapter = _Adapter()
    adapter.pipeline.config.expand_timesteps = True

    with pytest.raises(ValueError, match="requires first_frame_mask"):
        WanVideoOutputCodec(adapter, bind_condition_active_mask=True).encode_output_state(
            _media(np.zeros((9, 4, 4, 3), dtype=np.uint8)),
            {},
        )

    assert adapter.vae.encoded_pixels == []
    assert adapter.vae.posterior.generators == []


def test_wan_i2v_normalization_never_truncates_optional_last_frame() -> None:
    first = Image.new("RGB", (4, 4), color="red")
    last = Image.new("RGB", (4, 4), color="blue")

    rows = normalize_wan_i2v_image_rows([[first, last]], expected_batch_size=1)
    sample = WanI2VSample(condition_images=list(rows[0]))
    stacked = WanI2VSample.stack([sample])

    assert rows == ((first, last),)
    assert len(sample.condition_images) == 2
    torch.testing.assert_close(sample.condition_images[0][:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(sample.condition_images[1][:, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert len(stacked["condition_images"]) == 1
    assert len(stacked["condition_images"][0]) == 2
    torch.testing.assert_close(
        stacked["condition_images"][0][0][:, 0, 0], torch.tensor([1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        stacked["condition_images"][0][1][:, 0, 0], torch.tensor([0.0, 0.0, 1.0])
    )


def test_wan_i2v_online_prepare_latents_reuses_condition_mode_path() -> None:
    adapter = _Adapter()
    first_pixels = torch.zeros(1, 3, 16, 16)
    last_pixels = torch.ones(1, 3, 16, 16)
    rollout_noise = torch.zeros(1, 3, 2, 2, 2)

    latents, condition = Wan2_I2V_Adapter.prepare_latents(
        adapter,
        first_pixels,
        batch_size=1,
        num_channels_latents=3,
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=rollout_noise,
        last_image=last_pixels,
    )

    assert latents is rollout_noise
    assert condition.shape == (1, 7, 2, 2, 2)
    assert adapter.vae.posterior.mode_calls == 1
    assert adapter.vae.posterior.generators == []


def test_wan_v2_condition_cache_keeps_ordered_pixels_through_prepare(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (4, 4), color="red").save(tmp_path / "first.png")
    Image.new("RGB", (4, 4), color="blue").save(tmp_path / "last.png")
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {
                "prompt": "first and last",
                "media": [
                    {"type": "image", "path": "first.png"},
                    {"type": "image", "path": "last.png"},
                ],
            },
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "video", "path": "target.mp4"}]},
            },
        },
        dataset_dir=tmp_path,
    )
    cache = build_offline_condition_cache(
        [record],
        source_name="wan-i2v",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=_WanPreprocessHarness(with_clip=True).preprocess_func,
        preprocessing_batch_size=1,
        force_reprocess=True,
    )
    cache_row = dict(cache[0])
    cache_row.pop(OFFLINE_CONDITION_ID_COLUMN)

    condition = _collate_condition_mappings([cache_row])
    assert len(condition["images"]) == 1
    assert len(condition["images"][0]) == 2
    assert condition["condition_images"].shape == (1, 2, 3, 16, 16)
    torch.testing.assert_close(
        condition["condition_images"][0, :, :, 0, 0],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert condition["image_embeds"].shape == (1, 2, 3, 5)
    prepared = WanI2VConditionStatePreparer(_Adapter()).prepare_condition_state(condition)

    assert "images" not in prepared.condition
    assert prepared.forward_context["latent_condition"].shape == (1, 7, 2, 2, 2)
    assert prepared.forward_context["image_embeds"].shape == (2, 3, 5)


def test_wan_i2v_two_sample_first_last_embeds_round_trip_through_replay_stack() -> None:
    packed = torch.arange(4 * 3 * 5, dtype=torch.float32).view(4, 3, 5)
    per_sample = split_wan_image_embeds(packed, (2, 2))
    samples = [
        WanI2VSample(
            image_embeds=per_sample[index],
            latent_condition=torch.zeros(3, 2, 2, 2),
        )
        for index in range(2)
    ]

    replay_batch = WanI2VSample.stack(samples)
    assert replay_batch["image_embeds"].shape == (2, 2, 3, 5)
    restored = normalize_wan_image_embeds(replay_batch["image_embeds"], batch_size=2)

    torch.testing.assert_close(restored, packed)


@pytest.mark.parametrize("channels", [(0, 0, 0), (255, 255, 255), (0, 127, 255)])
def test_wan_codec_normalizes_pixels_with_real_video_processor(
    channels: tuple[int, int, int],
) -> None:
    """Convert decoded bytes once before Diffusers normalizes VAE pixels."""
    adapter = _Adapter()
    processor = VideoProcessor(vae_scale_factor=8)
    received = []

    def preprocess_video(videos: list[np.ndarray], **kwargs: Any) -> torch.Tensor:
        received.extend(videos)
        return processor.preprocess_video(videos, **kwargs)

    adapter.pipeline.video_processor = SimpleNamespace(preprocess_video=preprocess_video)
    source = np.broadcast_to(np.array(channels, dtype=np.uint8), (9, 16, 16, 3)).copy()
    original = source.copy()
    WanVideoOutputCodec(adapter).encode_output_state(_media(source), {})
    pixels = adapter.vae.encoded_pixels[0]
    assert pixels.shape == (1, 3, 5, 16, 16)
    expected = torch.tensor(channels, dtype=torch.float32) / 255.0 * 2.0 - 1.0
    torch.testing.assert_close(pixels, expected.view(1, 3, 1, 1, 1).expand_as(pixels))
    assert received[0].dtype == np.float32
    np.testing.assert_allclose(received[0], source[[0, 2, 4, 6, 8]].astype(np.float32) / 255.0)
    np.testing.assert_array_equal(source, original)
