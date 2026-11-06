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

"""Qwen CP metadata, masked conditioning and native fixture invariants on CPU."""

import copy
from types import SimpleNamespace

import pytest
import torch
from qwen_cp_fixture import (
    adapter_class,
    build_fixture,
    build_transformer,
    inactive_parameter_names,
    install_lora,
    model_inputs,
)

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter


def test_qwen_native_layout_declaration_and_module_guard():
    capability = QwenImageAdapter.context_parallel_capability
    assert capability.layout == "qwen_joint_image_text" and not capability.tensor_parallel
    model = build_transformer()
    QwenImageAdapter.validate_context_parallel_module(model, 2)
    QwenImageAdapter.validate_context_parallel_module(model, 4)
    assert set(model._cp_plan) == {
        "transformer_blocks.0",
        "transformer_blocks.*",
        "pos_embed",
        "proj_out",
    }
    with pytest.raises(ValueError, match="QwenImageTransformer2DModel"):
        QwenImageAdapter.validate_context_parallel_module(torch.nn.Linear(1, 1), 2)
    model.transformer_blocks[0].attn.set_processor(torch.nn.Identity())
    with pytest.raises(ValueError, match="native QwenDoubleStream"):
        QwenImageAdapter.validate_context_parallel_module(model, 2)


@pytest.mark.parametrize(
    "patch,tp,error",
    [
        ({"guidance_embeds": True}, 1, "guidance_embeds=False"),
        ({"zero_cond_t": True}, 1, "zero_cond_t=False"),
        ({"use_additional_t_cond": True}, 1, "use_additional_t_cond=False"),
        ({"use_layer3d_rope": True}, 1, "use_layer3d_rope=False"),
        ({"num_attention_heads": 3}, 1, "num_attention_heads=3"),
        ({"num_layers": 0}, 1, "native transformer blocks"),
        ({"patch_size": 1}, 1, "packed input"),
        ({"out_channels": 16}, 1, "packed input"),
        ({}, 2, r"TP\+CP"),
    ],
)
def test_qwen_invalid_metadata_fails_before_weight_loading(monkeypatch, patch, tp, error):
    config = dict(build_transformer().config)
    config.update(patch)
    adapter = object.__new__(QwenImageAdapter)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2, effective_tensor_parallel_size=tp
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    monkeypatch.setattr(QwenImageTransformer2DModel, "load_config", lambda *a, **kw: config)
    adapter._load_diffusers_pipeline = lambda *a, **kw: pytest.fail("loaded before guard")
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_native_masked_both_streams_have_only_declared_unused_gradients(mode):
    model = build_transformer()
    if mode == "lora":
        model = install_lora(model)
    output = model(**model_inputs(0, 0, "cpu"), return_dict=False)[0]
    assert output.shape == (2, 16, 16)
    output.square().mean().backward()
    missing = {name for name, p in model.named_parameters() if p.requires_grad and p.grad is None}
    assert missing == inactive_parameter_names(model)
    assert len(missing) == 6
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    for branch in ("img_mlp", "txt_mlp", "attn.to_q", "attn.add_q_proj"):
        assert any(
            branch in name and p.grad is not None and torch.count_nonzero(p.grad)
            for name, p in model.named_parameters()
        )


def _adapter(model, monkeypatch):
    adapter = object.__new__(QwenImageAdapter)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = SimpleNamespace(transformer=model)
    adapter.scheduler = SimpleNamespace(step=lambda **kwargs: kwargs["velocity"])
    monkeypatch.setattr(
        QwenImageAdapter, "transformer", property(lambda self: self.pipeline.transformer)
    )
    return adapter


@pytest.mark.parametrize("cfg", [False, True])
def test_qwen_cp_padding_preserves_masked_tokens_cfg_and_gradients(monkeypatch, cfg):
    baseline = build_transformer().float()
    padded = copy.deepcopy(baseline)
    inputs = model_inputs(0, 0, "cpu")
    args = dict(
        t=torch.tensor([600.0]),
        latents=inputs["hidden_states"][:1],
        prompt_embeds=inputs["encoder_hidden_states"][:1, :7],
        prompt_embeds_mask=torch.tensor([[1, 0, 1, 1, 0, 1, 1]]),
        img_shapes=[[(1, 4, 4)]],
        guidance_scale=2.0 if cfg else 1.0,
    )
    if cfg:
        args.update(
            negative_prompt_embeds=inputs["encoder_hidden_states"][1:, :5],
            negative_prompt_embeds_mask=torch.tensor([[1, 1, 0, 1, 1]]),
        )
    results = []
    for cp_size, model in ((1, baseline), (2, padded)):
        monkeypatch.setattr(
            "flow_factory.models.qwen_image.qwen_image.context_parallel_geometry",
            lambda _: (cp_size, 0),
        )
        adapter = _adapter(model, monkeypatch)
        seen = []
        handle = model.register_forward_pre_hook(lambda m, a, kw: seen.append(kw), with_kwargs=True)
        output = adapter.forward(**args)
        handle.remove()
        output.square().mean().backward()
        results.append(output)
        expected = 7 if cp_size == 1 else 8
        assert seen[0]["encoder_hidden_states"].shape[1] == expected
        mask = seen[0]["encoder_hidden_states_mask"]
        assert mask[0, :7].tolist() == [1, 0, 1, 1, 0, 1, 1]
        if cp_size == 2:
            assert mask[:, -1].count_nonzero() == 0
        if cfg:
            assert mask[1, :5].tolist() == [1, 1, 0, 1, 1]
    torch.testing.assert_close(results[0], results[1], rtol=3e-5, atol=2e-6)
    for (name, left), (_, right) in zip(baseline.named_parameters(), padded.named_parameters()):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("latents", torch.zeros(1, 15, 16), "token count=15"),
        ("latents", torch.zeros(2, 16, 16), "batch=1"),
        ("prompt_embeds", torch.zeros(1, 8, 31), "width=32"),
        ("prompt_embeds_mask", torch.ones(1, 7), "mask must match"),
        ("img_shapes", [[(1, 2, 8), (1, 2, 8)]], "one.*image"),
        ("img_shapes", [[(1, 2, 7)]], "matching packed"),
        ("attention_kwargs", {"scale": 0.5}, "attention_kwargs"),
    ],
)
def test_qwen_invalid_inputs_fail_before_cp_collectives(monkeypatch, field, value, error):
    monkeypatch.setattr(
        "flow_factory.models.qwen_image.qwen_image.context_parallel_geometry", lambda _: (2, 0)
    )
    adapter = _adapter(build_transformer(), monkeypatch)
    inputs = dict(
        latents=torch.zeros(1, 16, 16),
        prompt_embeds=torch.zeros(1, 8, 32),
        prompt_embeds_mask=torch.ones(1, 8),
        img_shapes=[[(1, 4, 4)]],
        attention_kwargs=None,
    )
    adapter._validate_context_parallel_inputs(**inputs)
    inputs[field] = value
    with pytest.raises(ValueError, match=error):
        adapter._validate_context_parallel_inputs(**inputs)


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("mask_length", [4, 6])
def test_qwen_rejects_original_mask_mismatch_before_padding(
    monkeypatch, negative, as_list, mask_length
):
    monkeypatch.setattr(
        "flow_factory.models.qwen_image.qwen_image.context_parallel_geometry", lambda _: (2, 0)
    )
    adapter = _adapter(build_transformer(), monkeypatch)
    adapter._pad_batch_prompt = lambda **kwargs: pytest.fail(
        "normalized before original mask guard"
    )
    embeddings = torch.zeros(1, 5, 32)
    mask = torch.ones(1, mask_length)
    invalid_embeddings = list(embeddings.unbind(0)) if as_list else embeddings
    invalid_mask = list(mask.unbind(0)) if as_list else mask
    args = dict(
        t=torch.tensor([600.0]),
        latents=torch.zeros(1, 16, 16),
        prompt_embeds=embeddings,
        prompt_embeds_mask=torch.ones(1, 5),
        img_shapes=[[(1, 4, 4)]],
        guidance_scale=2.0 if negative else 1.0,
    )
    prefix = "negative_" if negative else ""
    args[prefix + "prompt_embeds"] = invalid_embeddings
    args[prefix + "prompt_embeds_mask"] = invalid_mask
    with pytest.raises(ValueError, match="mask must match original embeddings"):
        adapter.forward(**args)


def test_qwen_fixture_roundtrip_uses_native_text_vae_and_transformer(tmp_path):
    build_fixture(tmp_path)
    pipeline = QwenImagePipeline.from_pretrained(tmp_path / "model", dtype=torch.float32)
    embeddings, mask = pipeline.encode_prompt("a red image")
    assert embeddings.shape == (1, 190, 32)
    assert mask is None
    output = pipeline.vae.decode(torch.ones(1, 4, 1, 8, 8), return_dict=False)[0]
    assert output.shape == (1, 3, 1, 64, 64) and torch.isfinite(output).all()
    assert len(pipeline.transformer.transformer_blocks) == 2


def test_qwen_recipe_binds_cfg_and_existing_guard_owner(monkeypatch, tmp_path):
    from qwen_coupled_fixture import make_config

    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "4")
    options = dict(algorithm="grpo-guard", training_mode="full", cp_size=2, tp_size=1, dp_size=2)
    config = make_config(tmp_path, tmp_path, options)
    assert config.training_args.guidance_scale == 2.0
    assert config.training_args.gradient_accumulation_steps == 4
    assert config.scheduler_args.sde_steps == [0, 1]
    assert config.model_args.model_type == "qwen-image"
    validate_context_parallel_config(config, QwenImageAdapter, get_trainer_class("grpo-guard"))
