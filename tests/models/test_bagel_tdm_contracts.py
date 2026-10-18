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

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import FullyShardedDataParallelPlugin
from peft import LoraConfig, get_peft_model

import flow_factory.utils.imports as import_utils
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.samples import BaseSample
from flow_factory.trainers.distillation.distillation_runtime import (
    validate_media_free_rollout,
    without_media_decoding,
)


def _load_bagel_types(monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
    """Load Bagel behind the same optional-kernel seam as its adapter tests."""
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
    cv2 = types.ModuleType("cv2")
    cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(import_utils, "is_flash_attn_available", lambda *args: True)
    monkeypatch.setattr(import_utils, "get_flash_attn_version", lambda: "test")

    module = importlib.import_module("flow_factory.models.bagel.bagel")
    return module.BagelAdapter, module.BagelSample


def _load_bagel_qwen_types(monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
    """Load Bagel's Qwen2-NaViT model behind the optional-kernel seam."""
    _load_bagel_types(monkeypatch)
    module = importlib.import_module("flow_factory.models.bagel.modeling.bagel.qwen2_navit")
    return module.Qwen2Config, module.Qwen2ForCausalLM


def _load_bagel_model_type(monkeypatch: pytest.MonkeyPatch) -> type:
    """Load the vendored Bagel container behind the optional-kernel seam."""
    _load_bagel_types(monkeypatch)
    module = importlib.import_module("flow_factory.models.bagel.modeling.bagel.bagel")
    return module.Bagel


class _FakeRotaryEmbedding(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids
        values = torch.ones_like(hidden_states).unsqueeze(0)
        return values, values


class _CheckpointedDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        *,
        packed_query_sequence: torch.Tensor,
        past_key_values: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        del kwargs
        self.calls += 1
        return packed_query_sequence.square(), past_key_values


def _checkpointing_qwen_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[torch.nn.Module, _CheckpointedDecoderLayer]:
    qwen_config_type, qwen_causal_lm_type = _load_bagel_qwen_types(monkeypatch)
    config = qwen_config_type(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        layer_module="Qwen2DecoderLayer",
        qk_norm=False,
        _attn_implementation="eager",
        pad_token_id=0,
    )
    causal_lm = qwen_causal_lm_type(config)
    model = causal_lm.model
    decoder = _CheckpointedDecoderLayer()
    model.layers = torch.nn.ModuleList([decoder])
    model.rotary_emb = _FakeRotaryEmbedding()
    model.norm = torch.nn.Identity()
    model.use_moe = False
    causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, decoder


def _raw_token_qwen_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RoutedComponentProxy, torch.nn.Module]:
    """Build the real PEFT/bundle route with a lightweight decoder body."""
    qwen_config_type, qwen_causal_lm_type = _load_bagel_qwen_types(monkeypatch)
    config = qwen_config_type(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        layer_module="Qwen2DecoderLayer",
        qk_norm=False,
        _attn_implementation="eager",
        pad_token_id=0,
    )
    base_model = qwen_causal_lm_type(config)
    model = get_peft_model(
        base_model,
        LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"]),
    )
    routed_base = model.get_base_model()
    routed_base.model.layers = torch.nn.ModuleList([_CheckpointedDecoderLayer()])
    routed_base.model.rotary_emb = _FakeRotaryEmbedding()
    routed_base.model.norm = torch.nn.Identity()
    routed_base.model.use_moe = False
    bundle = ModelBundle({"transformer": model})
    return RoutedComponentProxy(bundle, "transformer", model), routed_base


def _adapter(adapter_type: type, decoder: Any) -> Any:
    adapter = object.__new__(adapter_type)
    adapter.decode_latents = MethodType(decoder, adapter)
    return adapter


def _result(batch_size: int = 2) -> dict[str, Any]:
    initial = torch.zeros(batch_size, 4, 8)
    terminal = torch.ones(batch_size, 4, 8)
    return {
        "final_latents": terminal,
        "all_latents": [initial, terminal],
        "all_log_probs": None,
        "timesteps": torch.tensor([1000.0]),
        "latent_index_map": torch.tensor([0, 1]),
        "log_prob_index_map": None,
        "callback_results": {},
        "callback_index_map": None,
    }


def test_bagel_qwen_gradient_checkpointing_recomputes_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, decoder = _checkpointing_qwen_model(monkeypatch)

    packed = torch.tensor([[2.0, 3.0]], requires_grad=True)
    cache = object()
    output = model.forward_inference(
        packed_query_sequence=packed,
        query_lens=torch.tensor([1]),
        packed_query_position_ids=torch.tensor([0]),
        packed_query_indexes=torch.tensor([0]),
        past_key_values=cache,
        key_values_lens=torch.tensor([0]),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        update_past_key_values=False,
        is_causal=False,
    )
    output.packed_query_sequence.sum().backward()

    assert model.gradient_checkpointing is True
    assert decoder.calls == 2
    assert output.past_key_values is cache
    torch.testing.assert_close(packed.grad, torch.tensor([[4.0, 6.0]]))


def test_bagel_qwen_checkpointing_does_not_replay_cache_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, decoder = _checkpointing_qwen_model(monkeypatch)

    packed = torch.tensor([[2.0, 3.0]], requires_grad=True)
    output = model.forward_inference(
        packed_query_sequence=packed,
        query_lens=torch.tensor([1]),
        packed_query_position_ids=torch.tensor([0]),
        packed_query_indexes=torch.tensor([0]),
        past_key_values=object(),
        key_values_lens=torch.tensor([0]),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        update_past_key_values=True,
        is_causal=False,
    )
    output.packed_query_sequence.sum().backward()

    assert decoder.calls == 1


def test_bagel_qwen_routes_raw_text_embedding_through_outer_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer, base_model = _raw_token_qwen_proxy(monkeypatch)
    with torch.no_grad():
        base_model.model.embed_tokens.weight.copy_(
            torch.arange(64, dtype=torch.float32).reshape(8, 8) / 10
        )

    text_ids = torch.tensor([1, 2])
    output = transformer(
        packed_query_sequence=None,
        query_lens=torch.tensor([2]),
        packed_query_position_ids=torch.tensor([0, 1]),
        packed_query_indexes=torch.tensor([4, 5]),
        past_key_values=object(),
        key_values_lens=torch.tensor([4]),
        packed_key_value_indexes=torch.arange(4),
        update_past_key_values=True,
        is_causal=True,
        packed_text_ids=text_ids,
        # These are merged-cache indexes for text prefill, not query-local
        # insertion indexes. The sequence=None path must ignore them.
        packed_text_indexes=torch.tensor([4, 5]),
    )

    expected = base_model.model.embed_tokens(text_ids).square()
    torch.testing.assert_close(output.packed_query_sequence, expected)


def test_bagel_qwen_inserts_raw_text_without_detaching_aux_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer, base_model = _raw_token_qwen_proxy(monkeypatch)
    base_model.model.embed_tokens.weight.requires_grad_(True)
    with torch.no_grad():
        base_model.model.embed_tokens.weight.copy_(
            torch.arange(64, dtype=torch.float32).reshape(8, 8) / 10
        )

    aux_sequence = torch.full((3, 8), 2.0, requires_grad=True)
    text_ids = torch.tensor([1])
    output = transformer(
        packed_query_sequence=aux_sequence,
        query_lens=torch.tensor([3]),
        packed_query_position_ids=torch.tensor([0, 0, 0]),
        packed_query_indexes=torch.tensor([0, 1, 2]),
        past_key_values=object(),
        key_values_lens=torch.tensor([0]),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        update_past_key_values=False,
        is_causal=False,
        packed_text_ids=text_ids,
        packed_text_indexes=torch.tensor([1]),
    )
    output.packed_query_sequence.sum().backward()

    expected_input = aux_sequence.detach().clone()
    expected_input[1] = base_model.model.embed_tokens(text_ids).detach()[0]
    torch.testing.assert_close(output.packed_query_sequence, expected_input.square())
    torch.testing.assert_close(aux_sequence.grad[0], torch.full((8,), 4.0))
    torch.testing.assert_close(aux_sequence.grad[1], torch.zeros(8))
    torch.testing.assert_close(aux_sequence.grad[2], torch.full((8,), 4.0))
    assert base_model.model.embed_tokens.weight.grad is not None
    assert base_model.model.embed_tokens.weight.grad[1].abs().sum() > 0


def test_bagel_text_cache_uses_injected_language_model_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bagel_type = _load_bagel_model_type(monkeypatch)
    bagel = object.__new__(bagel_type)
    torch.nn.Module.__init__(bagel)
    bagel.use_moe = False

    class ExplodingPhysicalLanguageModel(torch.nn.Module):
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError("the physical language model must not be called")

        @property
        def model(self) -> Any:
            raise AssertionError("the physical embedding must not be accessed")

    bagel.language_model = ExplodingPhysicalLanguageModel()
    returned_cache = object()
    calls: list[dict[str, Any]] = []

    def routed_forward(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(past_key_values=returned_cache)

    result = bagel.forward_cache_update_text(
        past_key_values=object(),
        packed_text_ids=torch.tensor([1, 2]),
        packed_text_position_ids=torch.tensor([0, 1]),
        text_token_lens=torch.tensor([2]),
        packed_text_indexes=torch.tensor([0, 1]),
        packed_key_value_indexes=torch.tensor([], dtype=torch.long),
        key_values_lens=torch.tensor([0]),
        language_model_forward=routed_forward,
    )

    assert result is returned_cache
    assert len(calls) == 1
    assert calls[0]["packed_query_sequence"] is None
    assert torch.equal(calls[0]["packed_text_ids"], torch.tensor([1, 2]))


def test_bagel_adapter_injects_prepared_route_into_all_cache_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_type, _ = _load_bagel_types(monkeypatch)
    routed_transformer = object()
    forwarded_routes: list[Any] = []

    class FakeBagel:
        def prepare_prompts(self, **kwargs: Any) -> tuple[dict[str, Any], list[int], list[int]]:
            del kwargs
            return {}, [1], [1]

        def forward_cache_update_text(
            self, past_key_values: Any, *, language_model_forward: Any
        ) -> Any:
            del past_key_values
            forwarded_routes.append(language_model_forward)
            return "text-cache"

        def prepare_vae_images(self, **kwargs: Any) -> tuple[dict[str, Any], list[int], list[int]]:
            del kwargs
            return {}, [2], [2]

        def forward_cache_update_vae(
            self, vae_model: Any, past_key_values: Any, *, language_model_forward: Any
        ) -> Any:
            del vae_model, past_key_values
            forwarded_routes.append(language_model_forward)
            return "vae-cache"

        def prepare_vit_images(self, **kwargs: Any) -> tuple[dict[str, Any], list[int], list[int]]:
            del kwargs
            return {}, [3], [3]

        def forward_cache_update_vit(
            self, past_key_values: Any, *, language_model_forward: Any
        ) -> Any:
            del past_key_values
            forwarded_routes.append(language_model_forward)
            return "vit-cache"

    holder = SimpleNamespace(
        pipeline=SimpleNamespace(bagel=FakeBagel(), vae=object()),
        device=torch.device("cpu"),
        transformer=routed_transformer,
        _tokenizer=object(),
        new_token_ids={},
        vae_transform=object(),
        vit_transform=object(),
    )
    context = {"kv_lens": [0], "ropes": [0], "past_key_values": object()}

    text_context = adapter_type._update_context_text(holder, ["prompt"], context)
    image_context = adapter_type._update_context_image(holder, [torch.zeros(3, 8, 8)], context)

    assert text_context["past_key_values"] == "text-cache"
    assert image_context["past_key_values"] == "vit-cache"
    assert forwarded_routes == [routed_transformer] * 3


@pytest.mark.parametrize(
    "layer_module",
    ["Qwen2DecoderLayer", "Qwen2MoEDecoderLayer", "Qwen2MoTDecoderLayer"],
)
def test_bagel_qwen_fsdp_wrap_policy_matches_decoder_variant(
    monkeypatch: pytest.MonkeyPatch,
    layer_module: str,
) -> None:
    qwen_config_type, qwen_causal_lm_type = _load_bagel_qwen_types(monkeypatch)
    config = qwen_config_type(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        layer_module=layer_module,
        qk_norm=False,
        _attn_implementation="eager",
        pad_token_id=0,
    )
    base_model = qwen_causal_lm_type(config)
    decoder_type = type(base_model.model.layers[0])
    model = get_peft_model(
        base_model,
        LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"]),
    )
    bundle = ModelBundle({"transformer": model})
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
    )

    plugin.set_auto_wrap_policy(bundle)

    assert set(model._no_split_modules) == {decoder_type.__name__}
    assert set(base_model.model._no_split_modules) == {decoder_type.__name__}
    assert set(bundle._no_split_modules) == {decoder_type.__name__}
    assert plugin.auto_wrap_policy.keywords["transformer_layer_cls"] == {decoder_type}


def test_bagel_assembles_samples_from_one_batched_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_type, sample_type = _load_bagel_types(monkeypatch)
    calls: list[tuple[tuple[int, ...], tuple[int, int] | None]] = []

    def decode(
        _self: Any,
        latents: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> list[torch.Tensor]:
        calls.append((tuple(latents.shape), image_shape))
        return [torch.full((3, 8, 8), float(index)) for index in range(latents.shape[0])]

    adapter = _adapter(adapter_type, decode)
    samples = adapter._assemble_samples(
        _result(),
        prompts=["first", "second"],
        condition_images_list=None,
        height=64,
        width=64,
    )

    assert calls == [((2, 4, 8), (64, 64))]
    assert all(isinstance(sample, sample_type) for sample in samples)
    assert torch.equal(samples[0].image, torch.zeros(3, 8, 8))
    assert torch.equal(samples[1].image, torch.ones(3, 8, 8))


def test_bagel_media_free_samples_keep_replay_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_type, _ = _load_bagel_types(monkeypatch)

    def decode(
        _self: Any,
        latents: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> list[torch.Tensor]:
        del latents, image_shape
        raise AssertionError("TDM must not invoke the real Bagel decoder")

    adapter = _adapter(adapter_type, decode)
    validate_media_free_rollout(adapter, algorithm_name="TDM")

    with without_media_decoding(adapter, algorithm_name="TDM"):
        samples = adapter._assemble_samples(
            _result(),
            prompts=["first", "second"],
            condition_images_list=None,
            height=64,
            width=64,
        )

    assert [sample.image for sample in samples] == [None, None]
    batch = BaseSample.stack(samples)
    replay = adapter.get_replay_step(batch, 0)
    assert replay.state.components["latent"].shape == (2, 4, 8)
    assert replay.next_state.components["latent"].shape == (2, 4, 8)

    with pytest.raises(AssertionError, match="real Bagel decoder"):
        adapter.decode_latents(torch.zeros(2, 4, 8), image_shape=(64, 64))


def test_bagel_maps_reference_guidance_to_text_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_type, _ = _load_bagel_types(monkeypatch)
    adapter = object.__new__(adapter_type)

    assert adapter.reference_guidance_kwargs(4.0) == {"cfg_text_scale": 4.0}
