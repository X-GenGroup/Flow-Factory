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

"""Fake-only coverage for classic image-family offline output codecs."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from PIL import Image

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    MediaType,
    NegativePromptPolicy,
)
from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.models.z_image.z_image import ZImageAdapter

HEIGHT = 32
WIDTH = 32


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _Processor:
    def __init__(self) -> None:
        self.preprocess_calls: list[tuple[int, int, int, bool]] = []

    def get_default_height_width(self, image: Image.Image) -> tuple[int, int]:
        del image
        return HEIGHT, WIDTH

    def resize(
        self,
        images: list[Image.Image],
        height: int,
        width: int,
    ) -> list[Image.Image]:
        del height, width
        return images

    def preprocess(
        self,
        images: list[Image.Image],
        height: int,
        width: int,
    ) -> torch.Tensor:
        self.preprocess_calls.append((len(images), height, width, torch.is_grad_enabled()))
        return torch.arange(
            len(images) * 3 * height * width,
            dtype=torch.float32,
        ).reshape(len(images), 3, height, width)

    def postprocess(self, values: torch.Tensor, *, output_type: str) -> torch.Tensor:
        assert output_type == "pt"
        return values


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


class _ConvVAE:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self, *, shift: float = 1.5, scale: float = 2.0) -> None:
        self.config = SimpleNamespace(shift_factor=shift, scaling_factor=scale)
        self.encode_inputs: list[torch.Tensor] = []
        self.posteriors: list[_Posterior] = []

    def encode(self, values: torch.Tensor) -> Any:
        self.encode_inputs.append(values)
        posterior = _Posterior(values[:, :2, ::8, ::8] + len(self.encode_inputs))
        self.posteriors.append(posterior)
        return SimpleNamespace(latent_dist=posterior)


class _Runtime:
    def __init__(self, vae: _ConvVAE) -> None:
        self.vae = vae
        self.lookups: list[str] = []

    def get_component(self, name: str) -> _ConvVAE:
        self.lookups.append(name)
        assert name == "vae"
        return self.vae


class _FluxPipeline:
    vae_scale_factor = 8

    def __init__(self, processor: _Processor, vae: _ConvVAE) -> None:
        self.image_processor = processor
        self.vae = vae
        self.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=8))

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        assert tuple(latents.shape) == (batch_size, channels, height, width)
        return (
            latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, height // 2 * (width // 2), channels * 4)
        )

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del batch_size
        ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
        return ids.reshape(height * width, 3)


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


def _install_runtime(
    adapter_cls: type,
    *,
    pipeline: Any,
    vae: _ConvVAE,
) -> tuple[Any, _Runtime]:
    adapter = object.__new__(adapter_cls)
    adapter.training_args = SimpleNamespace(
        height=HEIGHT,
        width=WIDTH,
        latent_storage_dtype=None,
    )
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = pipeline
    runtime = _Runtime(vae)
    adapter.component_runtime = runtime
    adapter._output_state_encoding_modules = ("vae",)
    adapter._output_state_codec = adapter.build_output_state_codec()
    return adapter, runtime


@pytest.mark.parametrize(
    ("adapter_cls", "negative_prompt"),
    [
        (SD3_5Adapter, NegativePromptPolicy.OPTIONAL),
        (Flux1Adapter, NegativePromptPolicy.UNSUPPORTED),
        (Flux1KontextAdapter, NegativePromptPolicy.UNSUPPORTED),
        (ZImageAdapter, NegativePromptPolicy.OPTIONAL),
    ],
)
def test_classic_image_adapters_declare_static_offline_capability(
    adapter_cls: type,
    negative_prompt: NegativePromptPolicy,
) -> None:
    """The class preflight proves image output semantics without loading weights."""
    adapter_cls.validate_offline_output_capability()
    contract = adapter_cls.pipeline_io_contract

    assert contract is not None
    assert contract.negative_prompt is negative_prompt
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.IMAGE,)
    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.batch_capability is BatchCapability.UNIFORM


def test_kontext_contract_requires_exactly_one_condition_image() -> None:
    """Offline manifests cannot inherit the online first-image fallback."""
    contract = Flux1KontextAdapter.pipeline_io_contract

    assert contract is not None
    assert len(contract.input_media.rules) == 1
    rule = contract.input_media.rules[0]
    assert rule.format.type is MediaType.IMAGE
    assert rule.min_count == 1
    assert rule.max_count == 1


@pytest.mark.parametrize(
    "adapter_cls",
    [SD3_5Adapter, Flux1Adapter, Flux1KontextAdapter, ZImageAdapter],
)
def test_codec_construction_only_declares_logical_vae_requirement(adapter_cls: type) -> None:
    """Building a codec performs no component lookup, movement, or materialization."""
    processor = _Processor()
    vae = _ConvVAE()
    pipeline: Any = _FluxPipeline(processor, vae)
    if adapter_cls is SD3_5Adapter:
        pipeline.patch_size = 2
    adapter, runtime = _install_runtime(adapter_cls, pipeline=pipeline, vae=vae)

    assert runtime.lookups == []
    assert adapter.output_state_codec.required_components == ("vae",)
    assert adapter.output_state_encoding_modules == ("vae",)


@pytest.mark.parametrize(
    ("adapter_cls", "shift", "scale"),
    [
        (SD3_5Adapter, 1.5, 2.0),
        (ZImageAdapter, 0.25, 1.75),
    ],
)
def test_conv_image_targets_are_sampled_on_the_fly_with_the_caller_generator(
    adapter_cls: type,
    shift: float,
    scale: float,
) -> None:
    """SD3.5 and Z-Image re-encode every target instead of reading a latent cache."""
    processor = _Processor()
    vae = _ConvVAE(shift=shift, scale=scale)
    pipeline = _FluxPipeline(processor, vae)
    pipeline.patch_size = 2
    adapter, _ = _install_runtime(adapter_cls, pipeline=pipeline, vae=vae)
    generator = torch.Generator().manual_seed(17)

    first = adapter.encode_output_state(_media_batch(), {}, generator)
    second = adapter.encode_output_state(_media_batch(), {}, generator)

    assert len(vae.encode_inputs) == 2
    assert processor.preprocess_calls == [
        (2, HEIGHT, WIDTH, False),
        (2, HEIGHT, WIDTH, False),
    ]
    assert all(posterior.mode_calls == 0 for posterior in vae.posteriors)
    assert all(posterior.sample_generators == [generator] for posterior in vae.posteriors)
    expected_first = (vae.posteriors[0].value + 7.0 - shift) * scale
    expected_second = (vae.posteriors[1].value + 7.0 - shift) * scale
    assert torch.equal(first.clean_state.components["latent"], expected_first)
    assert torch.equal(second.clean_state.components["latent"], expected_second)
    assert not torch.equal(expected_first, expected_second)
    assert dict(first.forward_context) == {}
    assert dict(first.decode_context) == {"height": HEIGHT, "width": WIDTH}


def test_flux1_targets_are_sampled_then_packed_with_target_ids() -> None:
    """The target codec uses FLUX packing and exposes the IDs consumed by forward."""
    processor = _Processor()
    vae = _ConvVAE()
    pipeline = _FluxPipeline(processor, vae)
    adapter, _ = _install_runtime(Flux1Adapter, pipeline=pipeline, vae=vae)
    generator = torch.Generator().manual_seed(19)

    encoded = adapter.encode_output_state(_media_batch(), {}, generator)

    posterior = vae.posteriors[0]
    expected = (posterior.value + 7.0 - 1.5) * 2.0
    expected = pipeline._pack_latents(expected, 2, 2, 4, 4)
    assert posterior.mode_calls == 0
    assert posterior.sample_generators == [generator]
    assert torch.equal(encoded.clean_state.components["latent"], expected)
    assert tuple(encoded.forward_context) == ("img_ids",)
    assert encoded.forward_context["img_ids"].shape == (4, 3)
    assert dict(encoded.decode_context) == {"height": HEIGHT, "width": WIDTH}


def test_kontext_targets_prepend_target_ids_to_shared_condition_ids() -> None:
    """Target token IDs precede condition IDs exactly as Kontext forward concatenates states."""
    processor = _Processor()
    vae = _ConvVAE()
    pipeline = _FluxPipeline(processor, vae)
    adapter, _ = _install_runtime(Flux1KontextAdapter, pipeline=pipeline, vae=vae)
    generator = torch.Generator().manual_seed(23)
    condition_ids = torch.ones(2, 5, 3)

    encoded = adapter.encode_output_state(
        _media_batch(),
        {"image_ids": condition_ids},
        generator,
    )

    ids = encoded.forward_context["latent_ids"]
    assert vae.posteriors[0].mode_calls == 0
    assert vae.posteriors[0].sample_generators == [generator]
    assert ids.shape == (9, 3)
    assert torch.equal(ids[:4], pipeline._prepare_latent_image_ids(2, 2, 2, ids.device, ids.dtype))
    assert torch.equal(ids[4:], condition_ids[0])


def test_kontext_rejects_nonuniform_batched_condition_ids() -> None:
    """One shared forward ID sequence cannot represent differing condition geometry."""
    processor = _Processor()
    vae = _ConvVAE()
    pipeline = _FluxPipeline(processor, vae)
    adapter, _ = _install_runtime(Flux1KontextAdapter, pipeline=pipeline, vae=vae)
    condition_ids = torch.zeros(2, 5, 3)
    condition_ids[1, :, 0] = 1

    with pytest.raises(ValueError, match=r"must be identical"):
        adapter.encode_output_state(_media_batch(), {"image_ids": condition_ids})


def test_kontext_condition_encoding_uses_explicit_posterior_argmax() -> None:
    """Cached input conditions remain deterministic while output targets sample."""
    processor = _Processor()
    vae = _ConvVAE()
    pipeline = _FluxPipeline(processor, vae)
    adapter, _ = _install_runtime(Flux1KontextAdapter, pipeline=pipeline, vae=vae)
    adapter._standardize_image_input = lambda images, output_type: images
    generator = torch.Generator().manual_seed(29)
    images = [Image.new("RGB", (WIDTH, HEIGHT)), Image.new("RGB", (WIDTH, HEIGHT))]

    condition = adapter.encode_image(
        images,
        condition_image_size=(HEIGHT, WIDTH),
        generator=generator,
    )

    posterior = vae.posteriors[0]
    assert posterior.mode_calls == 1
    assert posterior.sample_generators == []
    assert condition["image_latents"].shape == (2, 4, 8)
    assert condition["image_ids"].shape == (2, 4, 3)
    assert torch.equal(condition["image_ids"][..., 0], torch.ones(2, 4))


def test_kontext_flattens_one_condition_image_per_offline_sample() -> None:
    """GeneralDataset's nested single-image batch remains valid for Kontext."""
    adapter = object.__new__(Flux1KontextAdapter)
    adapter.accelerator = SimpleNamespace(num_processes=1, process_index=0)
    adapter._has_warned_multi_image = False
    first = Image.new("RGB", (WIDTH, HEIGHT), color="red")
    second = Image.new("RGB", (WIDTH, HEIGHT), color="blue")

    standardized = adapter._standardize_image_input([[first], [second]], output_type="pil")

    assert standardized == [first, second]
    assert adapter._has_warned_multi_image is False


def test_z_image_keeps_precision_aware_transformer_loading() -> None:
    """Offline codec support does not weaken the precision branch's model contract."""
    assert ZImageAdapter.component_load_dtype_defaults == {"transformer": torch.float32}
