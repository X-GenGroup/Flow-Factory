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

"""Qwen-2.1 prefix, masked-padding and fail-closed parallel contracts."""

from dataclasses import asdict

import pytest
import torch

from diffusers import QwenImage21Transformer2DModel
from flow_factory.models.qwen_image_21._parallel import configure_processors, joint_attention_mask
from flow_factory.models.qwen_image_21.qwen_image_21 import (
    QwenImage21Adapter,
    QwenImage21ContextParallelLayout,
)


def _model():
    torch.manual_seed(9)
    return QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )


def test_capability_has_model_owned_serialized_layout():
    capability = QwenImage21Adapter.context_parallel_capability
    assert capability.layout is QwenImage21ContextParallelLayout.PREFIX_BLOCK_CAUSAL
    assert asdict(capability) == {
        "layout": "qwen21_prefix_block_causal",
        "transport": "ulysses",
        "output_gradient": "local_slice",
        "version": 1,
        "tensor_parallel": True,
        "required_precision_policy": None,
    }
    assert not QwenImage21Adapter.sequence_parallel_by_default


@pytest.mark.parametrize("heads,cp,tp", [(6, 2, 2), (8, 3, 1)])
def test_head_geometry_rejected_before_module_mutation(heads, cp, tp):
    config = dict(_model().config)
    config["num_attention_heads"] = heads
    with pytest.raises(ValueError, match="divisible"):
        QwenImage21Adapter._validate_context_parallel_config(config, cp, tp)


@pytest.mark.parametrize(
    "change", [{"causal_condition": False}, {"num_layers": 0}, {"patch_size": 2}]
)
def test_non_native_layout_is_rejected(change):
    config = dict(_model().config)
    config.update(change)
    with pytest.raises(ValueError, match="Qwen-2.1"):
        QwenImage21Adapter._validate_context_parallel_config(config, 2, 2)


def test_mask_preserves_causal_text_ordered_images_and_excludes_padding():
    segments = [(0, 3, True), (3, 7, False), (7, 9, True), (9, 13, False)]
    key_valid = torch.ones(1, 17, dtype=torch.bool)
    key_valid[:, 2] = False
    observed = joint_attention_mask(
        segments, key_valid, batch=1, length=20, actual_length=17, device="cpu"
    )[0, 0]
    for query in range(17):
        for key in range(20):
            expected = key < 17 and key != 2
            for start, end, text in segments:
                if start <= query < end:
                    expected = expected and key < end and (not text or key <= query)
            assert bool(observed[query, key]) == expected, (query, key)
    assert not observed[:, 17:].any()
    assert observed[17:, 0].all()


def test_mask_refuses_overlapping_or_out_of_range_segments():
    with pytest.raises(ValueError, match="prefix segment"):
        joint_attention_mask(
            [(0, 4, False), (3, 7, True)], None, batch=1, length=8, actual_length=8, device="cpu"
        )


def test_model_scopes_clean_up_after_failure_and_do_not_leak_between_models():
    first, second = _model(), _model()
    configure_processors(first)
    configure_processors(second)
    assert first._ff_qwen21_parallel_state is not second._ff_qwen21_parallel_state
    with pytest.raises((ValueError, RuntimeError)):
        first(
            hidden_states=torch.randn(1, 4, 8),
            encoder_hidden_states=torch.randn(1, 3, 16),
            timestep=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2)]],
            img_mask=torch.tensor([[False, False, False, False]]),
            return_dict=False,
        )
    assert first._ff_qwen21_parallel_state.call.get() is None
    assert first._ff_qwen21_parallel_state.tokens.get() == ()
    assert second._ff_qwen21_parallel_state.call.get() is None


def test_reusing_one_parallel_plan_keeps_full_variants_in_distinct_scopes():
    from types import SimpleNamespace

    first, second = _model(), _model()
    plan = QwenImage21Adapter.sequence_parallel_plan()
    mesh = SimpleNamespace(size=lambda: 2)
    plan[""]._apply(first, mesh)
    plan[""]._apply(second, mesh)
    assert first._ff_qwen21_parallel_state is not second._ff_qwen21_parallel_state
    for model in (first, second):
        assert all(
            child._ff_qwen21_parallel_state is model._ff_qwen21_parallel_state
            for child in model.modules()
        )


def test_processor_validation_precedes_all_parallel_mutation():
    model = _model()
    original = model.transformer_blocks[0].attn.processor
    model.transformer_blocks[1].attn.set_processor(object())
    with pytest.raises(ValueError, match="native segmented"):
        configure_processors(model)
    assert model.transformer_blocks[0].attn.processor is original
    assert not hasattr(model, "_ff_qwen21_parallel_state")


@pytest.mark.parametrize("mode,cp_size", [("extract", 2), (None, 1), ("cached", 1), (None, 2)])
def test_native_prefix_backend_ownership_and_cached_selection(monkeypatch, mode, cp_size):
    from types import SimpleNamespace

    from diffusers.models import attention_dispatch
    from diffusers.models.transformers import transformer_qwenimage21 as native

    from flow_factory.models.qwen_image_21._parallel import _Call, _ParallelProcessor, _State

    calls = []

    def dispatch(query, key, value, **kwargs):
        calls.append(
            (kwargs.get("backend"), kwargs.get("parallel_config"), kwargs.get("attn_mask"))
        )
        return torch.zeros_like(query)

    monkeypatch.setattr(attention_dispatch, "dispatch_attention_fn", dispatch)
    monkeypatch.setattr(native, "dispatch_attention_fn", dispatch)
    state = _State()
    state.cp_config = SimpleNamespace(ulysses_degree=cp_size) if cp_size > 1 else None
    state.call.set(_Call(mode, 4 * cp_size))
    processor = _ParallelProcessor(state)
    processor._attention_backend = "native" if cp_size > 1 else "flash"
    config = object() if cp_size > 1 else None
    processor._parallel_config = config
    attention = _model().transformer_blocks[0].attn
    cache = SimpleNamespace(get=lambda: (torch.zeros(1, 2, 4, 16), torch.zeros(1, 2, 4, 16)))
    processor(
        attention,
        torch.randn(1, 4, 64),
        kv_cache_mode=mode,
        layer_cache=cache if mode == "cached" else None,
        segments=None if mode == "cached" else [(0, 2, True)],
    )
    expected = (
        processor._attention_backend
        if mode == "cached" or (cp_size > 1 and mode != "extract")
        else None
    )
    assert calls and all(backend == expected for backend, _, _ in calls)
    assert all(parallel is (None if mode == "extract" else config) for _, parallel, _ in calls)
    if mode == "cached" and cp_size == 1:
        assert all(mask is None for _, _, mask in calls)
    if mode == "extract":
        assert processor.serial._attention_backend == processor._attention_backend
        assert processor.serial._parallel_config is None
