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

"""Z-Image native padding, ragged CFG and early CP layout guards."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_z_image_context_parallel import build_transformer, model_inputs
from z_image_cp_fixture import build_pipeline

from flow_factory.models.z_image.z_image import ZImageAdapter


def test_native_processor_preserves_outputs_and_every_gradient():
    native = build_transformer().double()
    adapted = copy.deepcopy(native)
    ZImageAdapter.validate_context_parallel_module(adapted, 2, 2)
    ZImageAdapter.configure_context_parallel_module(adapted)
    expected = torch.stack(native(**model_inputs(0))[0])
    actual = torch.stack(adapted(**model_inputs(0))[0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, p), (_, q) in zip(native.named_parameters(), adapted.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


def test_native_pipeline_roundtrip_encodes_text_and_decodes_pixels(tmp_path):
    from diffusers import ZImagePipeline

    pipeline = build_pipeline()
    pipeline.save_pretrained(tmp_path / "model")
    loaded = ZImagePipeline.from_pretrained(tmp_path / "model", torch_dtype=torch.float32)
    adapter = object.__new__(ZImageAdapter)
    adapter.pipeline = loaded
    prompts = ["a red image", "a blue small image"]
    negative_prompts = ["blue", "red"]
    with torch.no_grad():
        encoded = adapter.encode_prompt(
            prompts,
            device=torch.device("cpu"),
            negative_prompt=negative_prompts,
            max_sequence_length=64,
        )
        assert prompts == ["a red image", "a blue small image"]
        assert negative_prompts == ["blue", "red"]
        repeated = adapter.encode_prompt(
            prompts,
            device=torch.device("cpu"),
            negative_prompt=negative_prompts,
            max_sequence_length=64,
        )
        for field in ("prompt_embeds", "negative_prompt_embeds"):
            for first, second in zip(encoded[field], repeated[field]):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert len(encoded["prompt_embeds"]) == 2
        assert encoded["prompt_embeds"][0].shape[0] != encoded["prompt_embeds"][1].shape[0]
        assert encoded["negative_prompt_embeds"][0].shape[-1] == 32
        args = model_inputs(0, dtype=torch.float32)
        args["cap_feats"] = encoded["prompt_embeds"]
        output = torch.stack(loaded.transformer(**args)[0]).squeeze(2)
        pixels = loaded.vae.decode(
            output / loaded.vae.config.scaling_factor + loaded.vae.config.shift_factor,
            return_dict=False,
        )[0]
    assert pixels.shape == (2, 3, 64, 64) and torch.isfinite(pixels).all()
    ZImageAdapter.validate_context_parallel_module(loaded.transformer, 2, 2)


@pytest.mark.parametrize(
    "patch,fragment",
    [
        ({"n_kv_heads": 10}, "equal Q/KV"),
        ({"all_patch_size": [1]}, "patch_size"),
        ({"n_layers": 0}, "nonempty"),
    ],
)
def test_invalid_metadata_fails_before_pipeline_weights(monkeypatch, patch, fragment):
    from diffusers import ZImageTransformer2DModel

    config = dict(build_transformer().config)
    config.update(patch)
    monkeypatch.setattr(ZImageTransformer2DModel, "load_config", lambda *args, **kwargs: config)
    adapter = object.__new__(ZImageAdapter)
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
        ({"x": [[]]}, "omni"),
        ({"cap_feats": [torch.zeros(0, 32)]}, "nonempty"),
        ({"controlnet_block_samples": {}}, "ControlNet"),
        ({"patch_size": 1}, "patch_size"),
        ({"x": [torch.zeros(16, 1, 7, 8)] * 2}, "patch-aligned"),
    ],
)
def test_unsupported_input_fails_before_native_collectives(monkeypatch, patch, fragment):
    module = build_transformer()
    monkeypatch.setattr(module, "enable_parallelism", lambda **kwargs: None)
    ZImageAdapter.install_context_parallel_module(module, SimpleNamespace(ulysses_degree=2))
    args = model_inputs(0, dtype=torch.float32)
    args.update(patch)
    with pytest.raises(ValueError, match=fragment):
        module(**args)


def test_foreign_processor_rejected_before_mutation():
    module = build_transformer()
    before = module.layers[0].attention.processor
    module.layers[-1].attention.processor = object()
    with pytest.raises(ValueError, match="native attention"):
        ZImageAdapter.validate_context_parallel_module(module, 2, 2)
    assert module.layers[0].attention.processor is before


def test_native_lifecycle_uses_cp_sp_guard_and_cfg(monkeypatch, tmp_path):
    import z_image_coupled_fixture as fixture
    from context_parallel_overlap_runner import make_overlap_config

    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = dict(
        algorithm="grpo-guard",
        dp_size=2,
        tp_size=2,
        cp_size=2,
        training_mode="lora",
        scheduling_mode="ordered",
        sampler_type="rank_local",
    )
    config = make_overlap_config(
        fixture.make_config(tmp_path, tmp_path, options), tmp_path, tmp_path, options
    )
    validate_context_parallel_config(config, ZImageAdapter, get_trainer_class("grpo-guard"))
    assert config.parallelism_args.sequence_parallel and config.training_args.guidance_scale > 0
    assert config.training_args.reward_optimization_overlap
