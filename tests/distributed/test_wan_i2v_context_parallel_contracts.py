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

"""Wan2.1 I2V CP metadata, native image conditioning and fail-fast CPU controls."""

import copy
import json
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from wan_i2v_cp_fixture import build_fixture, build_transformer, install_lora, model_inputs

from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
from flow_factory.trainers.common.runtime_identity import _schema_digest
from flow_factory.utils.base import filter_kwargs


def _adapter(model, monkeypatch):
    adapter = object.__new__(Wan2_I2V_Adapter)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.config = SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=True))
    adapter.pipeline = SimpleNamespace(
        transformer=model,
        transformer_2=None,
        config=SimpleNamespace(expand_timesteps=False, boundary_ratio=None),
    )
    monkeypatch.setattr(
        Wan2_I2V_Adapter, "transformer", property(lambda self: self.pipeline.transformer)
    )
    return adapter


def _forward_inputs():
    values = model_inputs(0, 0, "cpu")
    return dict(
        t=values["timestep"],
        latents=values["hidden_states"][:, :4],
        latent_condition=values["hidden_states"][:, 4:],
        prompt_embeds=values["encoder_hidden_states"],
        image_embeds=values["encoder_hidden_states_image"],
        negative_prompt_embeds=None,
        attention_kwargs=None,
    )


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("cfg", [False, True])
def test_native_adapter_cfg_matches_independent_image_conditioned_forward(monkeypatch, mode, cfg):
    model = build_transformer()
    if mode == "lora":
        model = install_lora(model)
    reference = copy.deepcopy(model)
    adapter = _adapter(model, monkeypatch)
    args = _forward_inputs()
    scale = 2.0 if cfg else 1.0
    if cfg:
        args["negative_prompt_embeds"] = -args["prompt_embeds"]
    output = adapter.forward(
        **args, guidance_scale=scale, compute_log_prob=False, return_kwargs=("velocity",)
    ).velocity
    inputs = model_inputs(0, 0, "cpu")
    expected = reference(**inputs, return_dict=False)[0]
    if cfg:
        uncond = reference(
            **{**inputs, "encoder_hidden_states": -inputs["encoder_hidden_states"]},
            return_dict=False,
        )[0]
        expected = uncond + scale * (expected - uncond)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    expected.square().mean().backward()
    for (name, actual), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
        if actual.requires_grad:
            assert actual.grad is not None and target.grad is not None, name
            torch.testing.assert_close(actual.grad, target.grad, rtol=0, atol=0, msg=name)
        else:
            assert actual.grad is None and target.grad is None, name
    if mode == "full":
        for path in ("attn2.add_k_proj", "attn2.add_v_proj", "image_embedder", "patch_embedding"):
            assert any(
                path in name and p.grad is not None and torch.count_nonzero(p.grad)
                for name, p in model.named_parameters()
            ), path
    with torch.no_grad():
        no_image = reference(
            **{
                **inputs,
                "encoder_hidden_states_image": torch.zeros_like(
                    inputs["encoder_hidden_states_image"]
                ),
            },
            return_dict=False,
        )[0]
        no_condition = inputs["hidden_states"].clone()
        no_condition[:, 4:] = 0
        no_latent_condition = reference(
            **{**inputs, "hidden_states": no_condition}, return_dict=False
        )[0]
    conditional = reference(**inputs, return_dict=False)[0]
    assert not torch.allclose(conditional, no_image)
    assert not torch.allclose(conditional, no_latent_condition)


def test_wan_i2v_declares_native_conditioned_layout_and_rejects_other_processors():
    capability = Wan2_I2V_Adapter.context_parallel_capability
    assert capability.layout == "wan_image_conditioned_video" and capability.tensor_parallel
    assert capability.required_precision_policy == "cross_attention_fp32_v1"
    assert type(capability.layout).__module__ == "flow_factory.models.wan.wan2_i2v"
    legacy = {
        "layout": "wan_image_conditioned_video",
        "transport": "ulysses",
        "output_gradient": "local_slice",
        "version": 1,
        "tensor_parallel": True,
        "required_precision_policy": "cross_attention_fp32_v1",
    }
    assert json.dumps(asdict(capability), sort_keys=True) == json.dumps(legacy, sort_keys=True)
    assert _schema_digest(asdict(capability)) == _schema_digest(legacy)
    model = build_transformer()
    Wan2_I2V_Adapter.validate_context_parallel_module(model, 2, 2)
    with pytest.raises(ValueError, match="WanTransformer3DModel"):
        Wan2_I2V_Adapter.validate_context_parallel_module(torch.nn.Linear(1, 1), 2)
    model.blocks[0].attn2.set_processor(torch.nn.Identity())
    with pytest.raises(ValueError, match="native unfused"):
        Wan2_I2V_Adapter.validate_context_parallel_module(model, 2)


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"image_dim": None}, "CLIP image KV"),
        ({"added_kv_proj_dim": None}, "CLIP image KV"),
        ({"added_kv_proj_dim": 16}, "CLIP image KV"),
        ({"pos_embed_seq_len": 512}, "FLF2V"),
        ({"in_channels": 4}, "in_channels=2"),
        ({"num_layers": 0}, "native transformer blocks"),
        ({"num_attention_heads": 3}, "num_attention_heads=3"),
    ],
)
def test_unsupported_transformer_layout_fails_before_weights(monkeypatch, patch, error):
    config = dict(build_transformer().config)
    config.update(patch)
    adapter = object.__new__(Wan2_I2V_Adapter)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2, effective_tensor_parallel_size=2
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    monkeypatch.setattr(WanImageToVideoPipeline, "load_config", lambda *a, **kw: {})
    monkeypatch.setattr(WanTransformer3DModel, "load_config", lambda *a, **kw: config)
    adapter._load_diffusers_pipeline = lambda *a, **kw: pytest.fail("loaded before guard")
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


@pytest.mark.parametrize(
    "patch,error",
    [
        (
            {"expand_timesteps": True, "transformer_2": ["diffusers", "WanTransformer3DModel"]},
            "single TI2V",
        ),
        ({"boundary_ratio": float("nan")}, "boundary_ratio"),
        ({"boundary_ratio": 0.9}, "boundary_ratio"),
    ],
)
def test_unsupported_schedule_fails_before_transformer_metadata(monkeypatch, patch, error):
    adapter = object.__new__(Wan2_I2V_Adapter)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(effective_context_parallel_size=2)
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    monkeypatch.setattr(WanImageToVideoPipeline, "load_config", lambda *a, **kw: patch)
    monkeypatch.setattr(
        WanTransformer3DModel, "load_config", lambda *a, **kw: pytest.fail("loaded config")
    )
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("t", torch.tensor([600.0, 400.0]), "one sample timestep"),
        ("latents", torch.zeros(2, 4, 2, 4, 4), "one video"),
        ("latent_condition", torch.zeros(1, 4, 2, 4, 4), "latent_condition"),
        ("prompt_embeds", torch.zeros(1, 511, 16), "512 text tokens"),
        ("negative_prompt_embeds", torch.zeros(1, 511, 16), "512 text tokens"),
        ("image_embeds", None, "257"),
        ("image_embeds", torch.zeros(1, 514, 16), "257"),
        ("attention_kwargs", {"scale": 0.5}, "attention_kwargs"),
    ],
)
def test_invalid_conditions_fail_before_collectives(monkeypatch, field, value, error):
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_i2v.context_parallel_geometry", lambda _: (2, 0)
    )
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_i2v.tensor_parallel_geometry", lambda _: (2, 0)
    )
    adapter = _adapter(build_transformer(), monkeypatch)
    args = _forward_inputs()
    adapter._validate_context_parallel_inputs(**args)
    args[field] = value
    with pytest.raises(ValueError, match=error):
        adapter._validate_context_parallel_inputs(**args)


@pytest.mark.parametrize("shape,error", [((2, 3, 4), "patch_size"), ((1, 2, 4), "CP/SP degree=4")])
def test_i2v_generated_token_geometry_guard(monkeypatch, shape, error):
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_i2v.context_parallel_geometry", lambda _: (2, 0)
    )
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_i2v.tensor_parallel_geometry", lambda _: (2, 0)
    )
    adapter = _adapter(build_transformer(), monkeypatch)
    args = {
        **_forward_inputs(),
        "latents": torch.zeros(1, 4, *shape),
        "latent_condition": torch.zeros(1, 8, *shape),
    }
    with pytest.raises(ValueError, match=error):
        adapter._validate_context_parallel_inputs(**args)


def test_i2v_fixture_has_native_condition_encoders_and_finite_records(tmp_path):
    build_fixture(tmp_path)
    pipeline = WanImageToVideoPipeline.from_pretrained(
        tmp_path / "model", torch_dtype=torch.float32
    )
    image = pipeline.encode_image(Image.open(tmp_path / "data/0.png"))
    prompt, negative = pipeline.encode_prompt("a red video", max_sequence_length=512)
    assert image.shape == (1, 257, 16)
    assert prompt.shape == negative.shape == (1, 512, 16)
    rows = [json.loads(line) for line in (tmp_path / "data/train.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and all(row["input"]["media"][0]["slot"] == "first_frame" for row in rows)
    assert all(row["supervision"]["type"] == "demonstration" for row in rows)


def test_i2v_tdm_recipe_preserves_guidance_and_role_schedule(monkeypatch, tmp_path):
    from wan_i2v_tdm_fixture import make_config

    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = {"dp_size": 2, "tp_size": 2, "cp_size": 2, "mode": "full", "algorithm": "tdm"}
    config = make_config(tmp_path, tmp_path, options)
    assert config.model_args.model_type == "wan2_i2v"
    assert config.training_args.guidance_scale == config.training_args.real_guidance_scale == 2.0
    assert config.training_args.gradient_accumulation_steps == 4
    validate_context_parallel_config(config, Wan2_I2V_Adapter, get_trainer_class("tdm"))


def test_i2v_observation_preserves_bridge_forward_filtering(monkeypatch):
    import wan_i2v_tdm_fixture as fixture

    monkeypatch.setattr(fixture.tdm, "observe_training", lambda *args: nullcontext())
    adapter = _adapter(build_transformer(), monkeypatch)
    original = adapter.forward
    records = {}
    with fixture.observe_training(SimpleNamespace(adapter=adapter), records):
        kwargs = filter_kwargs(
            adapter.forward,
            **_forward_inputs(),
            height=32,
            width=32,
            guidance_scale=1.0,
            compute_log_prob=False,
            return_kwargs=("velocity",),
        )
        assert "height" not in kwargs and "width" not in kwargs
        assert adapter.forward(**kwargs).velocity.shape == (1, 4, 2, 4, 4)
    assert adapter.forward == original
    assert len(records["image_conditions"]) == 1
