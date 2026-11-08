# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Bagel's new parallel admission preserves its legacy MLP-only TP contract."""

import json
from dataclasses import asdict

import pytest
import torch
from test_bagel_tdm_contracts import _load_bagel_qwen_types, _load_bagel_types


@pytest.mark.parametrize(
    "cp,tp,sp", [(1, 2, True), (1, 4, True), (2, 2, True), (2, 8, False), (4, 8, False)]
)
def test_bagel_parallel_geometry_accepts_native_gqa_and_legacy_mlp_degrees(monkeypatch, cp, tp, sp):
    adapter, _ = _load_bagel_types(monkeypatch)
    adapter._validate_parallel_config(
        dict(num_attention_heads=28, num_key_value_heads=4, num_hidden_layers=2), cp, tp, sp
    )


@pytest.mark.parametrize("cp,tp,sp", [(1, 8, True), (1, 1, True), (2, 4, True), (3, 2, False)])
def test_bagel_parallel_geometry_rejects_incompatible_heads_before_loading(monkeypatch, cp, tp, sp):
    adapter, _ = _load_bagel_types(monkeypatch)
    with pytest.raises(ValueError, match="Bagel"):
        adapter._validate_parallel_config(
            dict(num_attention_heads=28, num_key_value_heads=4, num_hidden_layers=2), cp, tp, sp
        )


def test_bagel_capability_and_sp_overlay_preserve_legacy_tp_weights(monkeypatch):
    adapter, _ = _load_bagel_types(monkeypatch)
    from flow_factory.models.bagel.bagel import BagelContextParallelLayout

    capability = adapter.context_parallel_capability
    assert capability.layout is BagelContextParallelLayout.PACKED_PREFIX_MOT
    assert json.loads(json.dumps(asdict(capability))) == dict(
        layout="bagel_packed_prefix_mot",
        transport="ulysses",
        output_gradient="local_slice",
        version=1,
        tensor_parallel=True,
        required_precision_policy=None,
    )
    assert adapter.context_parallel_diffusers_revision is None
    assert adapter.sequence_parallel_by_default is False
    assert all("mlp" in path for path in adapter.tensor_parallel_plan())
    assert any("self_attn.q_proj" in path for path in adapter.sequence_parallel_plan())


@pytest.mark.parametrize("sp", [False, True], ids=["legacy-mlp", "explicit-attention-sp"])
def test_bagel_native_meta_parameter_coverage(monkeypatch, sp):
    """Count native base parameters against actual styles, without allocating weights."""
    from torch.distributed.tensor.parallel import RowwiseParallel

    from flow_factory.models.tensor_parallel import (
        SharedInputColwiseParallel,
        build_total_plan,
        resolve_parallel_plan,
    )

    adapter, _ = _load_bagel_types(monkeypatch)
    config_type, model_type = _load_bagel_qwen_types(monkeypatch)
    # Released BAGEL-7B-MoT llm_config.json plus the native pipeline's MoT,
    # QK-norm and untied-embedding choices; no pretrained weights are loaded.
    hidden, intermediate, layers, heads, kv_heads, vocab = 3584, 18944, 28, 28, 4, 152064
    with torch.device("meta"):
        model = model_type(
            config_type(
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_hidden_layers=layers,
                num_attention_heads=heads,
                num_key_value_heads=kv_heads,
                vocab_size=vocab,
                pad_token_id=151645,
                layer_module="Qwen2MoTDecoderLayer",
                qk_norm=True,
                tie_word_embeddings=False,
                _attn_implementation="eager",
            )
        )
    plan = build_total_plan(model, resolve_parallel_plan(adapter, sequence_parallel=sp))
    totals = {
        (placement, kind): 0
        for placement in ("sharded", "replicated")
        for kind in ("weight", "bias")
    }
    tensors = {"sharded": 0, "replicated": 0}
    replicated = set()
    for name, parameter in model.named_parameters():
        assert parameter.is_meta
        owner, kind = name.rsplit(".", 1)
        assert kind in ("weight", "bias")
        style = plan[owner]
        sharded = isinstance(style, SharedInputColwiseParallel) or (
            isinstance(style, RowwiseParallel) and kind == "weight"
        )
        # Assert the placement of every parameter, not just aggregate coverage.
        expected_sharded = ".mlp" in name or (sp and ".self_attn." in name and "_proj" in name)
        assert sharded is expected_sharded, name
        placement = "sharded" if sharded else "replicated"
        totals[placement, kind] += parameter.numel()
        tensors[placement] += 1
        if not sharded:
            replicated.add(name)

    head_dim = hidden // heads
    kv_width = kv_heads * head_dim
    mlp = 6 * layers * hidden * intermediate
    attention_weights = layers * (4 * hidden**2 + 4 * hidden * kv_width)
    attention_biases = layers * (2 * hidden + 4 * kv_width)
    norms = layers * (4 * hidden + 4 * head_dim) + 2 * hidden
    vocabulary = 2 * vocab * hidden
    assert (
        sum(totals.values())
        == mlp + attention_weights + attention_biases + norms + vocabulary
        == 14_141_252_608
    )
    assert totals == {
        ("sharded", "weight"): mlp + (attention_weights if sp else 0),
        ("sharded", "bias"): attention_biases if sp else 0,
        ("replicated", "weight"): norms + vocabulary + (0 if sp else attention_weights),
        ("replicated", "bias"): 0 if sp else attention_biases,
    }
    assert tensors == {"sharded": 560 if sp else 168, "replicated": 228 if sp else 620}
    assert {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.norm_moe_gen.weight",
    } <= replicated
    if sp:
        assert all(
            "norm" in name or name in ("model.embed_tokens.weight", "lm_head.weight")
            for name in replicated
        )


def test_bagel_cp_rejects_foreign_root_before_installation(monkeypatch):
    adapter, _ = _load_bagel_types(monkeypatch)
    module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="native Qwen2ForCausalLM"):
        adapter.validate_context_parallel_module(module, 2, 2)
    assert not hasattr(module, "_ff_bagel_parallel_state")


def test_bagel_packed_call_rejects_dropped_roles_and_reordered_cache():
    from flow_factory.models.bagel._parallel import _State

    state = _State()
    arguments = dict(
        mode="gen",
        update_past_key_values=False,
        is_causal=False,
        query_lens=torch.tensor([4]),
        key_values_lens=torch.tensor([3]),
        packed_query_indexes=torch.arange(3, 7),
        packed_key_value_indexes=torch.arange(3),
        packed_text_indexes=torch.tensor([0, 3]),
        packed_vae_token_indexes=torch.tensor([1, 2]),
    )
    assert state.describe(arguments).active
    with pytest.raises(ValueError, match="partition"):
        state.describe({**arguments, "packed_vae_token_indexes": torch.tensor([1])})
    with pytest.raises(ValueError, match="ordered prefix"):
        state.describe({**arguments, "packed_query_indexes": torch.tensor([3, 5, 4, 6])})


def test_bagel_packed_call_rejects_cross_sample_prefix_permutation():
    from flow_factory.models.bagel._parallel import _State

    state = _State()
    arguments = dict(
        mode="gen",
        update_past_key_values=False,
        is_causal=False,
        query_lens=torch.tensor([1, 1]),
        key_values_lens=torch.tensor([1, 1]),
        packed_query_indexes=torch.tensor([1, 3]),
        packed_key_value_indexes=torch.tensor([0, 2]),
        packed_text_indexes=torch.empty(0, dtype=torch.long),
        packed_vae_token_indexes=torch.tensor([0, 1]),
    )
    assert state.describe(arguments).active
    # Native scatter moves these cached rows between samples. The parallel
    # kernel splits canonical cache rows by length and must reject that mapping.
    with pytest.raises(ValueError, match="canonical packed prefix cache indexes"):
        state.describe({**arguments, "packed_key_value_indexes": torch.tensor([2, 0])})
    with pytest.raises(ValueError, match="canonical packed prefix cache indexes"):
        state.describe({**arguments, "packed_key_value_indexes": None})


@pytest.mark.parametrize("prefix_lengths", [None, torch.tensor([0, 0])])
@pytest.mark.parametrize("prefix_indexes", [None, torch.empty(0, dtype=torch.long)])
def test_bagel_packed_call_accepts_absent_or_empty_no_prefix(prefix_lengths, prefix_indexes):
    from flow_factory.models.bagel._parallel import _State

    arguments = dict(
        mode="gen",
        update_past_key_values=False,
        is_causal=False,
        query_lens=torch.tensor([2, 2]),
        key_values_lens=prefix_lengths,
        packed_query_indexes=torch.arange(4),
        packed_key_value_indexes=prefix_indexes,
        packed_text_indexes=torch.empty(0, dtype=torch.long),
        packed_vae_token_indexes=torch.arange(4),
    )
    assert _State().describe(arguments).active
