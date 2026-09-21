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

"""Fake-only contract and parity tests for the Qwen-Image 2.1 adapter."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from PIL import Image

from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from flow_factory.contracts import BatchCapability, GeometrySource, InputMediaOrder
from flow_factory.hparams import Arguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.qwen_image_21.output_codec import encode_qwen_image_21_output
from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter
from flow_factory.models.registry import get_model_adapter_class


class _Handle:
    def remove(self) -> None:
        return None


class _Norm:
    def register_forward_hook(self, hook: Any) -> _Handle:
        assert callable(hook)
        return _Handle()


class _TextEncoder:
    device = torch.device("cpu")
    model = SimpleNamespace(language_model=SimpleNamespace(norm=_Norm()))

    def __init__(self, hidden_states: torch.Tensor) -> None:
        self.hidden_states = hidden_states
        self.calls = 0

    def __call__(self, **kwargs: Any) -> Any:
        self.calls += 1
        assert kwargs["output_hidden_states"] is True
        return SimpleNamespace(hidden_states=(self.hidden_states,))


class _ProcessorInputs(SimpleNamespace):
    def to(self, device: torch.device) -> "_ProcessorInputs":
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[list[str]] = []

    def __call__(self, **kwargs: Any) -> _ProcessorInputs:
        self.calls += 1
        self.texts.append(list(kwargs["text"]))
        assert kwargs["padding_side"] == "left"
        assert len(kwargs["images"]) == 1
        return _ProcessorInputs(
            input_ids=torch.tensor([[0, 10, 11, 99, 12]]),
            attention_mask=torch.tensor([[0, 1, 1, 1, 1]]),
            pixel_values=torch.ones(1, 3, 2, 2),
            image_grid_thw=torch.ones(1, 3, dtype=torch.long),
        )


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.sample_generators: list[Optional[torch.Generator]] = []

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        self.sample_generators.append(generator)
        return self.value

    def mode(self) -> torch.Tensor:
        return self.value - 1


class _VAE:
    config = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])

    def __init__(self) -> None:
        self.posterior: Optional[_Posterior] = None

    def encode(self, values: torch.Tensor) -> Any:
        latent = torch.stack(
            [values[:, 0, :, ::16, ::16], values[:, 1, :, ::16, ::16]],
            dim=1,
        )
        self.posterior = _Posterior(latent)
        return SimpleNamespace(latent_dist=self.posterior)


class _ImageProcessor:
    def __init__(self) -> None:
        self.resize_calls: list[tuple[int, int]] = []

    def resize(self, image: Image.Image, *, width: int, height: int) -> Image.Image:
        self.resize_calls.append((width, height))
        return image.resize((width, height))

    def preprocess(self, image: Image.Image, *, width: int, height: int) -> torch.Tensor:
        return torch.zeros(1, 4, height, width)


class _Transformer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> tuple[torch.Tensor]:
        self.calls.append(kwargs)
        marker = kwargs["encoder_hidden_states"][0, 0, 0]
        return (torch.full_like(kwargs["hidden_states"], marker),)


def _adapter_with_components(
    *,
    text_encoder: Any = None,
    transformer: Any = None,
    vae: Any = None,
) -> Any:
    adapter = object.__new__(QwenImage21Adapter)
    components = {
        "text_encoder": text_encoder,
        "transformer": transformer,
        "vae": vae,
    }
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: components[name])
    adapter._warned_cfg_without_negative = False
    adapter._warned_negative_without_cfg = False
    return adapter


def test_registry_and_io_contract_are_explicit() -> None:
    assert get_model_adapter_class("qwen-image-2.1") is QwenImage21Adapter
    QwenImage21Adapter.validate_offline_output_capability()
    contract = QwenImage21Adapter.pipeline_io_contract
    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.batch_capability is BatchCapability.SINGLE_SAMPLE
    assert contract.input_media.order is InputMediaOrder.WITHIN_TYPE
    assert contract.input_media.rules[0].min_count == 0
    assert contract.input_media.rules[0].max_count is None
    assert not QwenImage21Adapter.supports_diffusers_cache
    assert BaseAdapter in QwenImage21Adapter.__bases__
    assert not any(
        base.__module__.startswith("flow_factory.models.qwen_image.")
        for base in QwenImage21Adapter.__mro__
    )


def test_grpo_example_parses_through_production_config() -> None:
    root = Path(__file__).resolve().parents[2]
    config = Arguments.load_from_yaml(str(root / "examples/grpo/lora/qwen_image_2_1/default.yaml"))

    assert config.model_args.model_type == "qwen-image-2.1"
    assert config.model_args.model_name_or_path == "Qwen/Qwen-Image-2.1"
    assert config.training_args.trainer_type == "grpo"
    assert config.training_args.per_device_batch_size == 1
    assert config.training_args.guidance_scale == 1.0
    assert config.training_args.condition_image_size == [384, 384]
    assert config.eval_args.per_device_batch_size == 1
    assert config.eval_args.guidance_scale == 1.0


def test_prompt_encoding_returns_ids_from_the_same_processor_pass() -> None:
    processor = _Processor()
    hidden_states = torch.arange(5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    text_encoder = _TextEncoder(hidden_states)
    adapter = _adapter_with_components(text_encoder=text_encoder)
    adapter.pipeline = SimpleNamespace(
        text_encoder=text_encoder,
        transformer=SimpleNamespace(dtype=torch.float32),
        processor=processor,
        prompt_template_t2i="system:{}",
        prompt_template_ti2i="<image1><|vision_start|><|image_pad|><|vision_end|>{}",
        _drop_idx=1,
        _img_token_id=99,
    )

    encoded = adapter._encode_qwen_image_21_condition(
        "edit this",
        [Image.new("RGBA", (2, 2), (1, 2, 3, 128))],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert processor.calls == 1
    assert text_encoder.calls == 1
    assert encoded["prompt_ids"].tolist() == [[11, 99, 12]]
    assert encoded["prompt_embeds"].shape == (1, 3, 4)
    assert encoded["prompt_embeds_mask"].tolist() == [[1, 1, 1]]
    assert encoded["image_pad_mask"].tolist() == [[False, True, False]]


def test_cfg_defaults_missing_negative_prompt_to_empty_text() -> None:
    processor = _Processor()
    hidden_states = torch.arange(5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    text_encoder = _TextEncoder(hidden_states)
    adapter = _adapter_with_components(text_encoder=text_encoder)
    adapter.pipeline = SimpleNamespace(
        text_encoder=text_encoder,
        transformer=SimpleNamespace(dtype=torch.float32),
        processor=processor,
        prompt_template_t2i="system:{}",
        prompt_template_ti2i="<image1><|vision_start|><|image_pad|><|vision_end|>{}",
        _drop_idx=1,
        _img_token_id=99,
    )

    encoded = adapter.encode_prompt(
        "edit this",
        guidance_scale=2.0,
        images=[Image.new("RGBA", (2, 2), (1, 2, 3, 128))],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert processor.calls == 2
    assert text_encoder.calls == 2
    assert processor.texts[1] == ["<image1><|vision_start|><|image_pad|><|vision_end|> "]
    assert encoded["negative_prompt_ids"].tolist() == [[11, 99, 12]]
    assert encoded["negative_prompt_embeds"].shape == (1, 3, 4)
    assert encoded["negative_prompt_embeds_mask"].tolist() == [[1, 1, 1]]
    assert encoded["negative_image_pad_mask"].tolist() == [[False, True, False]]


def test_non_square_condition_size_preserves_area_budget() -> None:
    image_processor = _ImageProcessor()
    vae = _VAE()
    adapter = _adapter_with_components(vae=vae)
    adapter.pipeline = SimpleNamespace(image_processor=image_processor)

    encoded = adapter._prepare_condition_images(
        [Image.new("RGBA", (400, 300), (1, 2, 3, 255))],
        condition_image_size=(384, 512),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert image_processor.resize_calls == [(512, 384)]
    assert encoded["condition_image_sizes"] == [(512, 384)]
    assert encoded["condition_img_shapes"] == [(1, 24, 32)]


def test_target_codec_uses_plain_packing_and_target_last_shapes() -> None:
    vae = _VAE()
    adapter = SimpleNamespace(
        vae=vae,
        _pack_latents=QwenImage21Adapter._pack_latents,
    )
    generator = torch.Generator().manual_seed(3)
    pixel_values = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(1, 4, 32, 32)

    encoded = encode_qwen_image_21_output(
        adapter,
        pixel_values,
        {"condition_img_shapes": [[(1, 3, 5), (1, 2, 4)]]},
        generator,
    )

    assert encoded.latents.shape == (1, 4, 2)
    assert encoded.forward_context["img_shapes"] == [[(1, 3, 5), (1, 2, 4), (1, 2, 2)]]
    assert vae.posterior is not None
    assert vae.posterior.sample_generators == [generator]


def test_prediction_is_condition_first_target_last_and_uses_plain_cfg() -> None:
    transformer = _Transformer()
    adapter = _adapter_with_components(transformer=transformer)
    target = torch.zeros(1, 4, 3)
    condition = torch.ones(1, 4, 3)
    positive = torch.full((1, 3, 6), 3.0)
    negative = torch.full((1, 3, 6), 1.0)
    mask = torch.ones(1, 3, dtype=torch.long)
    image_mask = torch.tensor([[False, True, False]])

    velocity = adapter._predict_velocity_one(
        t=torch.tensor([500.0]),
        latents=target,
        prompt_embeds=positive,
        prompt_embeds_mask=mask,
        image_pad_mask=image_mask,
        img_shapes=[(1, 2, 2), (1, 2, 2)],
        condition_image_latents=condition,
        negative_prompt_embeds=negative,
        negative_prompt_embeds_mask=mask,
        negative_image_pad_mask=image_mask,
        guidance_scale=2.0,
        attention_kwargs=None,
        use_kv_cache=False,
    )

    assert len(transformer.calls) == 2
    assert torch.equal(transformer.calls[0]["hidden_states"][:, :4], condition)
    assert torch.equal(transformer.calls[0]["hidden_states"][:, 4:], target)
    assert transformer.calls[0]["img_shapes"] == [[(1, 2, 2), (1, 2, 2)]]
    assert transformer.calls[0]["img_mask"].tolist() == [[False, True, False, True]]
    assert transformer.calls[0]["encoder_hidden_states_mask"] is None
    assert transformer.calls[1]["encoder_hidden_states_mask"] is None
    assert torch.equal(velocity, torch.full_like(target, 5.0))


def test_real_diffusers_transformer_api_preserves_training_gradients() -> None:
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    transformer.enable_gradient_checkpointing()
    adapter = _adapter_with_components(transformer=transformer)
    latents = torch.randn(1, 4, 8)
    prompt_embeds = torch.randn(1, 3, 16)
    condition_latents = torch.randn(1, 4, 8)
    prediction_kwargs = dict(
        t=torch.tensor([500.0]),
        latents=latents,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=torch.ones(1, 3, dtype=torch.long),
        image_pad_mask=torch.tensor([[False, True, False]]),
        img_shapes=[(1, 2, 2), (1, 2, 2)],
        condition_image_latents=condition_latents,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_image_pad_mask=None,
        guidance_scale=1.0,
        attention_kwargs=None,
    )
    with torch.no_grad():
        cache_free_velocity = adapter._predict_velocity_one(
            **prediction_kwargs,
            use_kv_cache=False,
        )
        rollout_velocity = adapter._predict_velocity_one(
            **prediction_kwargs,
            use_kv_cache=True,
        )
    velocity = adapter._predict_velocity_one(
        **prediction_kwargs,
        use_kv_cache=True,
    )

    torch.testing.assert_close(rollout_velocity, cache_free_velocity, rtol=0, atol=1e-6)
    assert torch.equal(velocity.detach(), rollout_velocity)
    velocity.square().mean().backward()

    assert velocity.shape == (1, 4, 8)
    assert torch.isfinite(velocity).all()
    assert transformer.transformer_blocks[0].attn.to_q.weight.grad is not None
    assert transformer.transformer_blocks[0].attn.to_k.weight.grad is not None
    assert transformer.txt_in.in_layer.weight.grad is not None


def test_minimal_prefill_matches_full_target_prefix_and_detaches_rollout_cache() -> None:
    torch.manual_seed(7)
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    ).eval()
    adapter = _adapter_with_components(transformer=transformer)
    target = torch.randn(1, 16, 8)
    condition = torch.randn(1, 4, 8)
    prompt = torch.randn(1, 3, 16)
    image_mask = torch.tensor([[False, True, False]])
    img_shapes = [(1, 2, 2), (1, 4, 4)]

    with torch.no_grad():
        minimal_cache = adapter._prefill_kv_cache_one(
            latents=target,
            prompt_embeds=prompt,
            prompt_embeds_mask=None,
            image_pad_mask=image_mask,
            img_shapes=img_shapes,
            condition_image_latents=condition,
            attention_kwargs=None,
            context_name="cond",
        )
        full_cache = QwenImage21KVCache(len(transformer.transformer_blocks))
        with transformer.cache_context("cond"):
            transformer(
                hidden_states=torch.cat([condition, target], dim=1),
                timestep=torch.tensor([0.5]),
                encoder_hidden_states=prompt,
                encoder_hidden_states_mask=None,
                img_shapes=[img_shapes],
                img_mask=torch.tensor([[False, True, False, True, True, True, True]]),
                kv_cache=full_cache,
                kv_cache_mode="extract",
                return_dict=False,
            )

    for minimal_layer, full_layer in zip(
        minimal_cache.layer_caches,
        full_cache.layer_caches,
        strict=True,
    ):
        assert minimal_layer.k is not None and minimal_layer.v is not None
        assert minimal_layer.k.requires_grad and minimal_layer.k.grad_fn is None
        assert minimal_layer.v.requires_grad and minimal_layer.v.grad_fn is None
        torch.testing.assert_close(minimal_layer.k, full_layer.k, rtol=0, atol=0)
        torch.testing.assert_close(minimal_layer.v, full_layer.v, rtol=0, atol=0)


def test_pack_unpack_round_trip_preserves_unpatched_latents() -> None:
    latents = torch.arange(1 * 3 * 1 * 2 * 4, dtype=torch.float32).reshape(1, 3, 1, 2, 4)
    packed = QwenImage21Adapter._pack_latents(latents, 1, 3, 2, 4)
    unpacked = QwenImage21Adapter._unpack_latents(
        packed,
        height=32,
        width=64,
        vae_scale_factor=16,
    )
    assert packed.shape == (1, 8, 3)
    assert torch.equal(unpacked, latents)
