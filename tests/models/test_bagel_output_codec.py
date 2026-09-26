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

"""Fake-component coverage for Bagel's offline image output codec."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
import torch.nn as nn
from PIL import Image

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    MediaType,
    NegativePromptPolicy,
)
from flow_factory.models.bagel._output import encode_bagel_vae_image
from flow_factory.models.output_state import GeometrySignature, MediaGeometrySignature


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _FakeBagelTransform:
    def __init__(self, output_shapes: dict[tuple[int, int], tuple[int, int]]) -> None:
        self.output_shapes = output_shapes
        self.calls: list[tuple[str, tuple[int, int], bool]] = []

    def _shape(self, image: Image.Image) -> tuple[int, int]:
        return self.output_shapes[image.size]

    def resize_transform(self, image: Image.Image) -> Image.Image:
        height, width = self._shape(image)
        self.calls.append(("resize", image.size, torch.is_grad_enabled()))
        return image.resize((width, height))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = self.resize_transform(image)
        self.calls.append(("tensor", image.size, torch.is_grad_enabled()))
        height, width = resized.height, resized.width
        return torch.arange(3 * height * width, dtype=torch.float32).reshape(
            3,
            height,
            width,
        )


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.inputs.append(pixel_values)
        mean = torch.cat(
            (
                pixel_values[:, :1, ::8, ::8] / 64,
                pixel_values[:, 1:2, ::8, ::8] / 32,
            ),
            dim=1,
        )
        logvar = torch.zeros_like(mean)
        moments = torch.cat((mean, logvar), dim=1)
        self.outputs.append(moments)
        return moments


class _FakeReg:
    def __init__(self) -> None:
        self.sample = False
        self.chunk_dim = 1
        self.selections: list[tuple[bool, Optional[torch.Generator]]] = []

    def select(
        self,
        moments: torch.Tensor,
        *,
        sample: bool,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self.selections.append((sample, generator))
        mean, logvar = torch.chunk(moments, 2, dim=self.chunk_dim)
        if not sample:
            return mean
        noise = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        return mean + torch.exp(0.5 * logvar) * noise


class _FakeBagelVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.encoder = _FakeEncoder()
        self.reg = _FakeReg()
        self.scale_factor = 0.5
        self.shift_factor = 0.25
        self.encode_calls = 0

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        moments = self.encoder(pixel_values)
        latents = self.reg.select(moments, sample=self.reg.sample)
        return self.normalize_latents(latents)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.scale_factor * (latents - self.shift_factor)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents / self.scale_factor + self.shift_factor


class _Runtime:
    def __init__(self, bagel: Any, vae: nn.Module) -> None:
        self.bagel = bagel
        self.vae = vae

    def get_component(self, name: str) -> Any:
        return {"bagel": self.bagel, "vae": self.vae}[name]


class _DeclarationRuntime:
    materialized_component_names = ()
    override_components: dict[str, Any] = {}
    declared_component_names = ("bagel", "vae")


def _load_bagel_adapter(monkeypatch: pytest.MonkeyPatch) -> type:
    import flow_factory.utils.imports as import_utils

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
    cv2 = types.ModuleType("cv2")
    cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(import_utils, "is_flash_attn_available", lambda *args: True)
    monkeypatch.setattr(import_utils, "get_flash_attn_version", lambda: "test")
    return importlib.import_module("flow_factory.models.bagel.bagel").BagelAdapter


def _load_bagel_image_transform(monkeypatch: pytest.MonkeyPatch) -> type:
    cv2 = types.ModuleType("cv2")
    cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return importlib.import_module("flow_factory.models.bagel.data.transforms").ImageTransform


def _install_adapter(
    adapter_cls: type,
    *,
    transform: _FakeBagelTransform,
    vae: _FakeBagelVAE,
    patch_size: int = 2,
) -> Any:
    adapter = object.__new__(adapter_cls)
    bagel = SimpleNamespace(
        latent_patch_size=patch_size,
        latent_channel=2,
        latent_downsample=16,
    )
    adapter.training_args = SimpleNamespace(latent_storage_dtype=None)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.vae_transform = transform
    adapter.pipeline = SimpleNamespace(bagel=bagel, vae=vae)
    adapter.component_runtime = _Runtime(bagel, vae)
    adapter._output_state_encoding_modules = ("bagel", "vae")
    adapter._output_state_codec = adapter.build_output_state_codec()
    return adapter


def _media_batch(*sizes: tuple[int, int]) -> tuple[tuple[_DecodedMedia, ...], ...]:
    return tuple((_DecodedMedia(type="image", payload=Image.new("RGB", size)),) for size in sizes)


def _manual_patchify(latents: torch.Tensor) -> torch.Tensor:
    packed_samples = []
    for sample in latents:
        tokens = []
        for latent_h in range(sample.shape[-2] // 2):
            for latent_w in range(sample.shape[-1] // 2):
                token = []
                for patch_h in range(2):
                    for patch_w in range(2):
                        for channel in range(sample.shape[0]):
                            token.append(
                                sample[
                                    channel,
                                    latent_h * 2 + patch_h,
                                    latent_w * 2 + patch_w,
                                ]
                            )
                tokens.append(torch.stack(token))
        packed_samples.append(torch.stack(tokens))
    return torch.stack(packed_samples)


def test_bagel_official_transform_maps_rgb_bytes_to_model_pixel_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bagel's released transform performs one ToTensor conversion before normalization."""
    transform_cls = _load_bagel_image_transform(monkeypatch)
    transform = transform_cls(max_image_size=2, min_image_size=2, image_stride=1)
    image = Image.new("RGB", (2, 2), color=(0, 127, 255))

    pixels = transform(image)

    expected_channels = torch.tensor(
        [-1.0, 2.0 * 127.0 / 255.0 - 1.0, 1.0],
        dtype=torch.float32,
    )
    expected = expected_channels.view(3, 1, 1).expand(3, 2, 2)
    torch.testing.assert_close(pixels, expected, rtol=0, atol=1e-7)


def test_bagel_pipeline_contract_covers_t2i_and_ordered_multi_image_i2i(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_cls = _load_bagel_adapter(monkeypatch)
    adapter_cls.validate_offline_output_capability()
    contract = adapter_cls.pipeline_io_contract

    assert contract.negative_prompt is NegativePromptPolicy.UNSUPPORTED
    assert contract.geometry_source is GeometrySource.OUTPUT_MEDIA
    assert contract.batch_capability is BatchCapability.UNIFORM
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.WITHIN_TYPE
    assert len(contract.input_media.rules) == 1
    assert contract.input_media.rules[0].format.type is MediaType.IMAGE
    assert contract.input_media.rules[0].min_count == 0
    assert contract.input_media.rules[0].max_count is None
    assert tuple(item.type for item in contract.output_media.items) == (MediaType.IMAGE,)


def test_bagel_offline_training_disables_both_cfg_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_cls = _load_bagel_adapter(monkeypatch)

    assert dict(adapter_cls.offline_training_forward_overrides) == {
        "cfg_text_scale": 1.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": (0.0, 1.0),
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
    }


def test_bagel_condition_encoding_preserves_mixed_t2i_i2i_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty optional-image slot remains aligned with its prompt row."""
    adapter_cls = _load_bagel_adapter(monkeypatch)
    adapter = object.__new__(adapter_cls)
    image = Image.new("RGB", (8, 8))

    encoded = adapter.encode_image([[], [image]])

    assert encoded["condition_images"][0] == []
    assert len(encoded["condition_images"][1]) == 1
    assert encoded["condition_images"][1][0].shape == (3, 8, 8)


def test_bagel_codec_declaration_is_logical_and_does_not_touch_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_cls = _load_bagel_adapter(monkeypatch)
    adapter = object.__new__(adapter_cls)
    adapter.component_runtime = _DeclarationRuntime()

    codec = adapter._build_output_state_codec_declaration()
    adapter._output_state_codec = codec

    assert codec.required_components == ("bagel", "vae")
    assert adapter._validate_output_state_codec_lifecycle() == ("bagel", "vae")
    assert adapter.component_runtime.materialized_component_names == ()
    assert adapter.component_runtime.override_components == {}


def test_bagel_target_samples_encoder_moments_without_mutating_condition_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_cls = _load_bagel_adapter(monkeypatch)
    transform = _FakeBagelTransform({(7, 9): (16, 32), (11, 13): (16, 32)})
    vae = _FakeBagelVAE()
    adapter = _install_adapter(adapter_cls, transform=transform, vae=vae)
    generator = torch.Generator().manual_seed(17)

    encoded = adapter.encode_output_state(
        _media_batch((7, 9), (11, 13)),
        {"prompt": ["first", "second"]},
        generator,
    )

    assert vae.reg.sample is False
    assert vae.encode_calls == 0
    assert vae.reg.selections == [(True, generator)]
    assert len(vae.encoder.inputs) == 1
    assert vae.encoder.inputs[0].dtype is torch.bfloat16
    assert all(not grad_enabled for kind, _, grad_enabled in transform.calls if kind == "tensor")

    mean, logvar = torch.chunk(vae.encoder.outputs[0], 2, dim=1)
    expected_noise = torch.randn(
        mean.shape,
        generator=torch.Generator().manual_seed(17),
        dtype=mean.dtype,
    )
    sampled = mean + torch.exp(0.5 * logvar) * expected_noise
    normalized = vae.scale_factor * (sampled - vae.shift_factor)
    assert torch.equal(encoded.clean_state.components["latent"], _manual_patchify(normalized))
    assert encoded.clean_state.components["latent"].shape == (2, 2, 8)
    assert dict(encoded.forward_context) == {"image_shape": (16, 32)}
    assert dict(encoded.decode_context) == {"image_shape": (16, 32)}
    signature = GeometrySignature(
        media=(
            MediaGeometrySignature(
                type=MediaType.IMAGE,
                height=16,
                width=32,
            ),
        )
    )
    assert encoded.geometry_signatures == (signature, signature)


def test_bagel_target_codec_requires_canonical_decoded_rgb_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_cls = _load_bagel_adapter(monkeypatch)
    transform = _FakeBagelTransform({(7, 9): (16, 32)})
    vae = _FakeBagelVAE()
    adapter = _install_adapter(adapter_cls, transform=transform, vae=vae)
    media_batch = ((_DecodedMedia(type="image", payload=Image.new("L", (7, 9))),),)

    with pytest.raises(ValueError, match="RGB mode"):
        adapter.encode_output_state(media_batch, {})

    assert vae.encoder.inputs == []


def test_bagel_role_neutral_primitive_keeps_argmax_and_sample_explicit() -> None:
    vae = _FakeBagelVAE()
    pixels = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    generator = torch.Generator().manual_seed(23)

    argmax = encode_bagel_vae_image(vae, pixels, posterior_mode="argmax")
    sampled = encode_bagel_vae_image(
        vae,
        pixels,
        posterior_mode="sample",
        generator=generator,
    )

    mean, logvar = torch.chunk(vae.encoder.outputs[0], 2, dim=1)
    expected_noise = torch.randn(
        mean.shape,
        generator=torch.Generator().manual_seed(23),
    )
    assert torch.equal(argmax, vae.scale_factor * (mean - vae.shift_factor))
    assert torch.equal(
        sampled,
        vae.scale_factor * (mean + torch.exp(0.5 * logvar) * expected_noise - vae.shift_factor),
    )
    assert vae.reg.sample is False


def test_bagel_custom_diagonal_gaussian_preserves_global_condition_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_bagel_adapter(monkeypatch)
    autoencoder = importlib.import_module("flow_factory.models.bagel.modeling.autoencoder")
    reg = autoencoder.DiagonalGaussian(sample=False)
    mean = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    logvar = torch.zeros_like(mean)
    moments = torch.cat((mean, logvar), dim=1)
    generator = torch.Generator().manual_seed(31)

    sampled = reg.select(moments, sample=True, generator=generator)
    condition = reg(moments)
    expected_noise = torch.randn(
        mean.shape,
        generator=torch.Generator().manual_seed(31),
    )

    assert torch.equal(sampled, mean + expected_noise)
    assert torch.equal(condition, mean)
    assert reg.sample is False


def test_bagel_target_codec_rejects_ragged_post_transform_geometry_before_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_cls = _load_bagel_adapter(monkeypatch)
    transform = _FakeBagelTransform({(7, 9): (16, 32), (11, 13): (32, 16)})
    vae = _FakeBagelVAE()
    adapter = _install_adapter(adapter_cls, transform=transform, vae=vae)

    with pytest.raises(ValueError, match="batch size 1 or batch targets by the geometry"):
        adapter.encode_output_state(_media_batch((7, 9), (11, 13)), {})

    assert vae.encoder.inputs == []


def test_bagel_target_codec_requires_the_official_two_by_two_patch_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_cls = _load_bagel_adapter(monkeypatch)
    transform = _FakeBagelTransform({(7, 9): (16, 32)})
    vae = _FakeBagelVAE()
    adapter = _install_adapter(
        adapter_cls,
        transform=transform,
        vae=vae,
        patch_size=1,
    )

    with pytest.raises(ValueError, match="latent_patch_size=2"):
        adapter.encode_output_state(_media_batch((7, 9)), {})
