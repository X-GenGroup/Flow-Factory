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

"""Fake-only coverage for FLUX.2 and Qwen offline output codecs."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from PIL import Image

from diffusers import Flux2Pipeline, QwenImageEditPlusPipeline
from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaOrder,
    MediaType,
)
from flow_factory.models.flux._output import (
    encode_flux2_vae_image,
    prepare_flux2_condition_latents,
)
from flow_factory.models.flux.flux2 import Flux2Adapter
from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter
from flow_factory.models.qwen_image._output import encode_qwen_vae_image
from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter
from flow_factory.utils.image import is_multi_image_batch


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _Processor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, bool]] = []

    def preprocess(
        self,
        images: list[Image.Image],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.calls.append((len(images), height, width, torch.is_grad_enabled()))
        return torch.arange(
            len(images) * 3 * height * width,
            dtype=torch.float32,
        ).reshape(len(images), 3, height, width)


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.mode_calls = 0
        self.sample_generators: list[Optional[torch.Generator]] = []

    def mode(self) -> torch.Tensor:
        self.mode_calls += 1
        return self.value

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        self.sample_generators.append(generator)
        return self.value + 7.0


class _Flux2VAE:
    dtype = torch.bfloat16
    config = SimpleNamespace(batch_norm_eps=0.0)
    bn = SimpleNamespace(running_mean=torch.zeros(2), running_var=torch.ones(2))

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []
        self.posteriors: list[_Posterior] = []

    def encode(self, image: torch.Tensor) -> Any:
        self.inputs.append(image)
        posterior = _Posterior(image[:, :2, ::8, ::8])
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


class _QwenVAE:
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []
        self.posteriors: list[_Posterior] = []

    def encode(self, values: torch.Tensor) -> Any:
        self.inputs.append(values)
        latent = torch.cat(
            [
                values[:, :1, :, ::8, ::8] + 5.0,
                values[:, 1:2, :, ::8, ::8] + 10.0,
            ],
            dim=1,
        )
        posterior = _Posterior(latent)
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


class _DeclarationRuntime:
    materialized_component_names = ()
    override_components: dict[str, Any] = {}
    declared_component_names = ("vae",)


def _media_batch(batch_size: int = 2) -> tuple[tuple[_DecodedMedia, ...], ...]:
    return tuple(
        (
            _DecodedMedia(
                type="image",
                payload=Image.new("RGB", (11 + index, 13 + index)),
            ),
        )
        for index in range(batch_size)
    )


def _install_adapter_runtime(
    adapter: Any,
    *,
    pipeline: Any,
    vae: Any,
    height: int = 32,
    width: int = 32,
) -> Any:
    adapter.training_args = SimpleNamespace(
        height=height,
        width=width,
        latent_storage_dtype=None,
    )
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = pipeline
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: vae)
    adapter._output_state_encoding_modules = ("vae",)
    adapter._output_state_codec = adapter.build_output_state_codec()
    return adapter


def _pack_qwen_latents(
    latents: torch.Tensor,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    assert tuple(latents.shape) == (batch_size, channels, 1, height, width)
    return (
        latents.reshape(batch_size, channels, 1, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(batch_size, height // 2 * (width // 2), channels * 4)
    )


@pytest.mark.parametrize(
    "adapter_cls",
    [
        Flux2Adapter,
        Flux2KleinAdapter,
        QwenImageAdapter,
        QwenImageEditPlusAdapter,
        QwenImage21Adapter,
    ],
)
def test_modern_image_codec_declarations_require_only_logical_vae(
    adapter_cls: type,
) -> None:
    """Codec construction is declaration-only and names the logical VAE route."""
    adapter_cls.validate_offline_output_capability()
    adapter = object.__new__(adapter_cls)
    adapter.training_args = SimpleNamespace(height=32, width=32)
    adapter.pipeline = SimpleNamespace(vae_scale_factor=8)
    adapter.component_runtime = _DeclarationRuntime()

    codec = adapter._build_output_state_codec_declaration()
    adapter._output_state_codec = codec

    assert codec is not None
    assert codec.required_components == ("vae",)
    assert adapter._validate_output_state_codec_lifecycle() == ("vae",)
    assert adapter.component_runtime.materialized_component_names == ()
    assert adapter.component_runtime.override_components == {}


@pytest.mark.parametrize("adapter_cls", [Flux2Adapter, Flux2KleinAdapter])
def test_flux2_target_samples_while_condition_encoding_uses_argmax(
    adapter_cls: type,
) -> None:
    """FLUX.2 shares patchify/BN/packing but keeps role-specific posterior policy."""
    processor = _Processor()
    vae = _Flux2VAE()

    def pack(latents: torch.Tensor) -> torch.Tensor:
        return latents.flatten(2).transpose(1, 2)

    pipeline = SimpleNamespace(
        image_processor=processor,
        vae_scale_factor=8,
        _patchify_latents=lambda latents: latents,
        _prepare_latent_ids=lambda latents: torch.zeros(
            latents.shape[0],
            latents.shape[-2] * latents.shape[-1],
            4,
        ),
        _prepare_image_ids=lambda latents: torch.zeros(
            1,
            sum(item.shape[-2] * item.shape[-1] for item in latents),
            4,
        ),
        _pack_latents=pack,
    )
    adapter = _install_adapter_runtime(
        object.__new__(adapter_cls),
        pipeline=pipeline,
        vae=vae,
    )
    generator = torch.Generator().manual_seed(17)

    encoded = adapter.encode_output_state(_media_batch(), {}, generator)

    assert vae.inputs[0].dtype is torch.bfloat16
    assert vae.posteriors[0].sample_generators == [generator]
    assert vae.posteriors[0].mode_calls == 0
    assert encoded.clean_state.components["latent"].shape == (2, 16, 2)
    assert encoded.forward_context["latent_ids"] is encoded.decode_context["latent_ids"]

    condition_latents, condition_ids = prepare_flux2_condition_latents(
        adapter,
        [torch.zeros(1, 3, 32, 32)],
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert vae.posteriors[1].mode_calls == 1
    assert vae.posteriors[1].sample_generators == []
    assert condition_latents.shape == (1, 16, 2)
    assert condition_ids.shape == (1, 16, 4)


def test_flux2_condition_transform_matches_pinned_diffusers() -> None:
    """The role-neutral FLUX.2 argmax transform stays aligned with Diffusers."""
    pixels = torch.randn(2, 3, 32, 32, dtype=torch.bfloat16)
    expected_vae = _Flux2VAE()
    expected = Flux2Pipeline._encode_vae_image(
        SimpleNamespace(
            vae=expected_vae,
            _patchify_latents=lambda latents: latents,
        ),
        pixels,
        torch.Generator().manual_seed(31),
    )
    actual_vae = _Flux2VAE()
    actual = encode_flux2_vae_image(
        SimpleNamespace(
            vae=actual_vae,
            pipeline=SimpleNamespace(_patchify_latents=lambda latents: latents),
        ),
        pixels,
        sample_mode="argmax",
    )

    assert torch.equal(actual, expected)
    assert actual_vae.posteriors[0].mode_calls == 1


@pytest.mark.parametrize("adapter_cls", [Flux2Adapter, Flux2KleinAdapter])
def test_flux2_condition_encoding_preserves_mixed_t2i_i2i_slots(
    adapter_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional condition images use None latents without shifting batch rows."""
    adapter = object.__new__(adapter_cls)
    adapter.pipeline = SimpleNamespace(
        vae=SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32),
        image_processor=SimpleNamespace(postprocess=lambda image, output_type: [image.squeeze(0)]),
    )
    adapter._standardize_image_input = lambda images, output_type: images
    adapter._resize_condition_images = lambda condition_images, condition_image_size: [
        torch.ones(1, 3, 2, 2)
    ]
    module = __import__(adapter_cls.__module__, fromlist=["prepare_flux2_condition_latents"])
    monkeypatch.setattr(
        module,
        "prepare_flux2_condition_latents",
        lambda *args, **kwargs: (torch.ones(1, 4, 2), torch.zeros(1, 4, 4)),
    )
    image = Image.new("RGB", (8, 8))

    encoded = adapter.encode_image([[], [image]])

    assert is_multi_image_batch([[], [image]])
    assert encoded["condition_images"][0] == []
    assert encoded["image_latents"][0] is None
    assert encoded["image_latent_ids"][0] is None
    assert encoded["image_latents"][1].shape == (4, 2)
    assert encoded["image_latent_ids"][1].shape == (4, 4)


@pytest.mark.parametrize("adapter_cls", [Flux2Adapter, Flux2KleinAdapter])
def test_flux2_preprocess_keeps_optional_image_columns_for_empty_chunk(adapter_cls: type) -> None:
    """Source-column presence fixes the cache schema even for an all-empty chunk."""
    adapter = object.__new__(adapter_cls)
    adapter.encode_prompt = lambda prompt, **kwargs: {"prompt_embeds": torch.ones(len(prompt), 2)}
    adapter.encode_image = lambda images, **kwargs: {
        "condition_images": [[] for _ in images],
        "image_latents": [None for _ in images],
        "image_latent_ids": [None for _ in images],
    }

    encoded = adapter.preprocess_func(prompt=["first", "second"], images=[[], []])

    assert set(encoded) == {
        "prompt_embeds",
        "condition_images",
        "image_latents",
        "image_latent_ids",
    }
    assert encoded["condition_images"] == [[], []]
    assert encoded["image_latents"] == [None, None]
    assert encoded["image_latent_ids"] == [None, None]


def test_qwen_target_codec_samples_five_dimensional_latents() -> None:
    """Qwen T2I targets sample the posterior before normalization and 2x2 packing."""
    processor = _Processor()
    vae = _QwenVAE()
    pipeline = SimpleNamespace(
        image_processor=processor,
        vae_scale_factor=8,
        _pack_latents=_pack_qwen_latents,
    )
    adapter = _install_adapter_runtime(
        object.__new__(QwenImageAdapter),
        pipeline=pipeline,
        vae=vae,
    )
    generator = torch.Generator().manual_seed(19)

    encoded = adapter.encode_output_state(_media_batch(), {}, generator)

    assert vae.inputs[0].shape == (2, 3, 1, 32, 32)
    assert vae.posteriors[0].sample_generators == [generator]
    assert vae.posteriors[0].mode_calls == 0
    assert encoded.clean_state.components["latent"].shape == (2, 4, 8)
    assert encoded.forward_context["img_shapes"] == [[(1, 2, 2)], [(1, 2, 2)]]


def test_qwen_condition_transform_matches_pinned_diffusers() -> None:
    """The role-neutral Qwen argmax transform stays aligned with Diffusers."""
    pixels = torch.randn(2, 3, 1, 32, 32)
    expected_vae = _QwenVAE()
    expected = QwenImageEditPlusPipeline._encode_vae_image(
        SimpleNamespace(vae=expected_vae, latent_channels=2),
        pixels,
        torch.Generator().manual_seed(37),
    )
    actual_vae = _QwenVAE()
    actual = encode_qwen_vae_image(
        SimpleNamespace(vae=actual_vae),
        pixels,
        sample_mode="argmax",
    )

    assert torch.equal(actual, expected)
    assert actual_vae.posteriors[0].mode_calls == 1


def test_qwen_edit_keeps_condition_argmax_and_target_first_ordered_shapes() -> None:
    """Qwen Edit separates posterior roles and retains ordered reference geometry."""
    processor = _Processor()
    vae = _QwenVAE()
    pipeline = SimpleNamespace(
        image_processor=processor,
        vae_scale_factor=8,
        latent_channels=2,
        _pack_latents=_pack_qwen_latents,
    )
    adapter = _install_adapter_runtime(
        object.__new__(QwenImageEditPlusAdapter),
        pipeline=pipeline,
        vae=vae,
        height=64,
        width=64,
    )

    condition_latents = adapter.prepare_image_latents(
        images=[
            torch.zeros(1, 3, 1, 32, 32),
            torch.full((1, 3, 1, 32, 64), 10.0),
        ],
        batch_size=1,
        num_channels_latents=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(23),
    )

    assert condition_latents.shape == (1, 12, 8)
    assert [posterior.mode_calls for posterior in vae.posteriors] == [1, 1]
    assert all(not posterior.sample_generators for posterior in vae.posteriors)

    condition = {
        "condition_image_sizes": torch.tensor([[[32, 32], [64, 32]]]),
        "vae_image_sizes": torch.tensor([[[32, 32], [64, 32]]]),
    }
    generator = torch.Generator().manual_seed(29)
    encoded = adapter.encode_output_state(_media_batch(1), condition, generator)

    assert vae.posteriors[2].sample_generators == [generator]
    assert vae.posteriors[2].mode_calls == 0
    assert processor.calls == [(1, 32, 96, False)]
    assert dict(encoded.decode_context) == {"height": 32, "width": 96}
    assert encoded.forward_context["img_shapes"] == [[(1, 2, 6), (1, 2, 2), (1, 2, 4)]]


@pytest.mark.parametrize(
    ("adapter_cls", "geometry_source", "batch_capability", "input_order"),
    [
        (
            Flux2Adapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.WITHIN_TYPE,
        ),
        (
            Flux2KleinAdapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.WITHIN_TYPE,
        ),
        (
            QwenImageAdapter,
            GeometrySource.CONFIGURED,
            BatchCapability.UNIFORM,
            InputMediaOrder.INSENSITIVE,
        ),
        (
            QwenImageEditPlusAdapter,
            GeometrySource.INPUT_MEDIA,
            BatchCapability.SINGLE_SAMPLE,
            InputMediaOrder.WITHIN_TYPE,
        ),
    ],
)
def test_modern_image_pipeline_contracts_are_explicit(
    adapter_cls: type,
    geometry_source: GeometrySource,
    batch_capability: BatchCapability,
    input_order: InputMediaOrder,
) -> None:
    """Algorithm code can reason about media and batching without adapter checks."""
    contract = adapter_cls.pipeline_io_contract

    assert contract is not None
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.IMAGE,)
    assert contract.geometry_source is geometry_source
    assert contract.batch_capability is batch_capability
    assert contract.input_media.order is input_order
