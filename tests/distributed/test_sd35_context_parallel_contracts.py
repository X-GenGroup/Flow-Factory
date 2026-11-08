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

"""Native SD3 pipeline packing, processor parity and early CP rejection."""

import copy

import pytest
import torch
from sd35_cp_fixture import build_pipeline
from test_sd35_context_parallel import build_transformer, model_inputs

from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter


@pytest.mark.parametrize("family", ["medium", "large"])
def test_native_processor_preserves_ordinary_forward_and_every_gradient(family):
    native = build_transformer(family)
    adapted = copy.deepcopy(native)
    SD3_5Adapter.validate_context_parallel_module(adapted, 2, 2)
    SD3_5Adapter.configure_context_parallel_module(adapted)
    expected, actual = native(**model_inputs(0))[0], adapted(**model_inputs(0))[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, p), (_, q) in zip(native.named_parameters(), adapted.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


def test_sd35_native_pipeline_roundtrip_keeps_odd_text_packing_and_cfg(tmp_path):
    from diffusers import StableDiffusion3Pipeline

    pipeline = build_pipeline("large")
    pipeline.save_pretrained(tmp_path / "model")
    loaded = StableDiffusion3Pipeline.from_pretrained(tmp_path / "model", torch_dtype=torch.float32)
    with torch.no_grad():
        text, negative, pooled, negative_pooled = loaded.encode_prompt(
            prompt="a red image",
            prompt_2="a red image",
            prompt_3="a red image",
            negative_prompt="blue",
            do_classifier_free_guidance=True,
            max_sequence_length=512,
        )
    assert text.shape == negative.shape == (1, 589, 32)
    assert pooled.shape == negative_pooled.shape == (1, 16)
    assert not torch.equal(text, negative)
    assert loaded.transformer.config.num_attention_heads == 38
    SD3_5Adapter.validate_context_parallel_module(loaded.transformer, 2, 2)
    with torch.no_grad():
        result = loaded.transformer(
            hidden_states=torch.zeros(1, 16, 8, 8),
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=torch.tensor([500.0]),
            return_dict=False,
        )[0]
    assert result.shape == (1, 16, 8, 8) and torch.isfinite(result).all()


def test_sd35_rejects_foreign_processor_before_mutation():
    module = build_transformer("medium")
    old = module.transformer_blocks[0].attn.processor
    module.transformer_blocks[-1].attn.processor = object()
    with pytest.raises(ValueError, match="requires native joint processors"):
        SD3_5Adapter.validate_context_parallel_module(module, 2, 2)
    assert module.transformer_blocks[0].attn.processor is old


def test_sd35_native_lifecycle_uses_public_cp_sp_and_cfg_contract(monkeypatch, tmp_path):
    import sd35_coupled_fixture as fixture
    from context_parallel_overlap_runner import make_overlap_config

    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = dict(
        algorithm="dppo",
        dp_size=2,
        tp_size=2,
        cp_size=2,
        training_mode="full",
        scheduling_mode="ordered",
        sampler_type="rank_local",
    )
    config = make_overlap_config(
        fixture.make_config(tmp_path, tmp_path, options), tmp_path, tmp_path, options
    )
    validate_context_parallel_config(config, SD3_5Adapter, get_trainer_class("dppo"))
    assert config.parallelism_args.sequence_parallel
    assert config.training_args.guidance_scale > 1
    assert config.training_args.reward_optimization_overlap


@pytest.mark.parametrize(
    "patch,fragment",
    [
        ({"num_attention_heads": 37}, "real attention heads"),
        ({"patch_size": 1}, "patch_size"),
        ({"num_layers": 0}, "MMDiT"),
        ({"out_channels": 8}, "matching"),
    ],
)
def test_invalid_metadata_fails_before_pipeline_weights(monkeypatch, patch, fragment):
    from types import SimpleNamespace

    from diffusers import SD3Transformer2DModel

    config = dict(build_transformer("large").config)
    config.update(patch)
    monkeypatch.setattr(SD3Transformer2DModel, "load_config", lambda *args, **kwargs: config)
    adapter = object.__new__(SD3_5Adapter)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2, effective_tensor_parallel_size=2
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    monkeypatch.setattr(
        adapter,
        "_load_diffusers_pipeline",
        lambda *args, **kwargs: pytest.fail("weights loaded before metadata rejection"),
    )
    with pytest.raises(ValueError, match=fragment):
        adapter.load_pipeline()


@pytest.mark.parametrize(
    "patch,fragment",
    [
        ({"hidden_states": torch.zeros(1, 16, 7, 8)}, "patch-aligned"),
        ({"encoder_hidden_states": torch.zeros(1, 0, 32)}, "nonempty"),
        ({"block_controlnet_hidden_states": []}, "ControlNet"),
        ({"joint_attention_kwargs": {"scale": 0.5}}, "custom attention"),
        ({"skip_layers": [0]}, "skipped blocks"),
    ],
)
def test_invalid_inputs_fail_before_native_collectives(monkeypatch, patch, fragment):
    from types import SimpleNamespace

    module = build_transformer("large")
    monkeypatch.setattr(module, "enable_parallelism", lambda **kwargs: None)
    SD3_5Adapter.install_context_parallel_module(module, SimpleNamespace(ulysses_degree=2))
    args = model_inputs(0)
    args.update(patch)
    with pytest.raises(ValueError, match=fragment):
        module(**args)
