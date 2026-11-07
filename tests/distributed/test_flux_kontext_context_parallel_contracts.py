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

"""CPU contracts for native target/reference-image CP layout and crop ownership."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from flux_kontext_cp_fixture import (
    build_fixture,
    encoded_conditions,
    flux_transformer,
    install_lora,
    make_adapter,
    model_inputs,
    reference_forward,
    scheduler,
)
from PIL import Image

from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("kontext-native")
    build_fixture(root)
    return root


def test_kontext_layout_identity_is_separate_and_sp_remains_unavailable():
    from flow_factory.models.tensor_parallel import resolve_parallel_plan
    from flow_factory.trainers.common.runtime_identity import _schema_digest

    capability = asdict(Flux1KontextAdapter.context_parallel_capability)
    assert capability == {
        "layout": "flux_reference_image_text",
        "transport": "ulysses",
        "output_gradient": "local_slice",
        "version": 1,
        "tensor_parallel": True,
        "required_precision_policy": None,
    }
    assert _schema_digest(capability) != _schema_digest(
        asdict(Flux1Adapter.context_parallel_capability)
    )
    assert Flux1KontextAdapter.sequence_parallel_plan() == {}
    assert not Flux1KontextAdapter.sequence_parallel_by_default
    with pytest.raises(ValueError, match="declares no sequence-parallel plan"):
        resolve_parallel_plan(Flux1KontextAdapter, sequence_parallel=True)


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"guidance_embeds": False}, "guidance_embeds"),
        ({"num_attention_heads": 2}, "tp_size=2"),
        ({"num_layers": 0}, "dual-stream"),
        ({"num_single_layers": 0}, "single-stream"),
        ({"out_channels": 8}, "equal packed"),
        ({"patch_size": 2}, "patch_size=1"),
    ],
)
def test_metadata_fails_before_weights(monkeypatch, patch, error):
    adapter = object.__new__(Flux1KontextAdapter)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2, effective_tensor_parallel_size=2
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    config = {**dict(flux_transformer().config), **patch}
    monkeypatch.setattr(FluxTransformer2DModel, "load_config", lambda *a, **kw: config)
    adapter._load_diffusers_pipeline = lambda *a, **kw: pytest.fail(
        "weights loaded before rejection"
    )
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


def test_native_transformer_and_processor_guards():
    model = flux_transformer()
    Flux1KontextAdapter.validate_context_parallel_module(model, 2, 2)
    assert set(model._cp_plan) == {"", "proj_out"}
    assert model._cp_plan[""]["hidden_states"].split_dim == 1
    assert model._cp_plan["proj_out"].gather_dim == 1
    model.transformer_blocks[0].attn.set_processor(torch.nn.Identity())
    with pytest.raises(ValueError, match="native FluxAttnProcessor"):
        Flux1KontextAdapter.validate_context_parallel_module(model, 2, 2)


@pytest.fixture
def guarded(monkeypatch, native_fixture):
    for name in ("flux1", "flux1_kontext"):
        monkeypatch.setattr(
            f"flow_factory.models.flux.{name}.context_parallel_geometry", lambda _: (2, 0)
        )
    pipeline = FluxKontextPipeline.from_pretrained(
        native_fixture / "model", torch_dtype=torch.float32
    )
    adapter = make_adapter(pipeline)
    inputs = model_inputs(native_fixture, 0, 0, "cpu")
    return adapter, inputs


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("latents", torch.zeros(2, 16, 16), "latents"),
        ("latents", torch.zeros(1, 0, 16), "latents"),
        ("image_latents", None, "image_latents"),
        ("image_latents", torch.zeros(1, 0, 16), "image_latents"),
        ("image_latents", torch.zeros(1, 8, 8), "image_latents"),
        ("image_latents", torch.zeros(1, 7, 16), "token count=23"),
        ("latent_ids", None, "tensor latent_ids"),
        ("latent_ids", torch.zeros(1, 24, 3), "img_ids"),
        ("latent_ids", torch.zeros(23, 3), "img_ids"),
        ("latent_ids", torch.zeros(24, 3), "single-reference tag 1"),
        ("latent_ids", torch.full((24, 3), float("nan")), "finite latent_ids"),
        ("prompt_embeds", torch.zeros(1, 511, 16), "text token count=511"),
        ("pooled_prompt_embeds", torch.zeros(1, 16), "pooled_prompt_embeds"),
        ("joint_attention_kwargs", {"ip_adapter_image_embeds": torch.zeros(1)}, "IP adapters"),
    ],
)
def test_inputs_rejected_before_collectives(guarded, key, value, error):
    adapter, inputs = guarded
    adapter.pipeline.transformer.forward = lambda *a, **kw: pytest.fail(
        "collectives before validation"
    )
    inputs[key] = value
    with pytest.raises(ValueError, match=error):
        adapter.forward(**inputs)


def test_only_combined_image_sequence_needs_cp_divisibility(guarded):
    adapter, inputs = guarded
    inputs["latents"] = inputs["latents"][:, :15]
    inputs["image_latents"] = torch.cat(
        [inputs["image_latents"], inputs["image_latents"][:, :1]], dim=1
    )
    inputs["latent_ids"][15, 0] = 1
    adapter._validate_context_parallel_inputs(
        **{k: v for k, v in inputs.items() if k not in ("t", "t_next")}, joint_attention_kwargs=None
    )


def test_cp1_preserves_existing_input_policy(guarded, monkeypatch):
    adapter, inputs = guarded
    monkeypatch.setattr(
        "flow_factory.models.flux.flux1_kontext.context_parallel_geometry", lambda _: (1, 0)
    )
    adapter._validate_context_parallel_inputs(None, None, None, None, None, {"scale": 0.5})


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_real_reference_encoding_crop_and_gradients(native_fixture, mode):
    pipeline = FluxKontextPipeline.from_pretrained(
        native_fixture / "model", torch_dtype=torch.float32
    )
    if mode == "lora":
        pipeline.transformer = install_lora(pipeline.transformer)
    pipeline.scheduler = scheduler()
    adapter = make_adapter(pipeline)
    inputs = model_inputs(native_fixture, 0, 0, "cpu")
    actual = adapter.forward(**inputs, return_kwargs=["velocity"]).velocity
    expected = reference_forward(pipeline.transformer, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 16, 16)
    for field in ("image_latents", "latent_ids"):
        changed = {**inputs, field: inputs[field].clone()}
        if field == "image_latents":
            changed[field].zero_()
        else:
            changed[field][16:, 0] = 0
        assert not torch.allclose(
            reference_forward(pipeline.transformer, changed), expected, rtol=1e-5, atol=1e-7
        ), field
    actual.square().mean().backward()
    active = {name: p for name, p in pipeline.transformer.named_parameters() if p.requires_grad}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active.values())
    for stack in ("transformer_blocks.0.", "single_transformer_blocks.0."):
        assert any(stack in name and torch.count_nonzero(p.grad) for name, p in active.items())
    if mode == "lora":
        assert all(p.grad is None for p in pipeline.transformer.parameters() if not p.requires_grad)
    prompt, image = encoded_conditions(str(native_fixture), 0)
    pixels = pipeline.image_processor.preprocess(
        Image.open(native_fixture / "data/0.png"), height=32, width=64
    )
    with torch.no_grad():
        raw = pipeline.vae.encode(pixels).latent_dist.mode()
    packed = pipeline._pack_latents(raw, 1, 4, 4, 8)
    torch.testing.assert_close(image["image_latents"], packed, rtol=0, atol=0)
    assert image["condition_images"][0].shape == (3, 32, 64)
    assert prompt["prompt_embeds"].shape == (1, 512, 16)


def test_reference_recipe_has_real_images_and_shared_capability(monkeypatch, native_fixture):
    from flux_kontext_coupled_fixture import dataset_record, make_config

    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "8")
    config = make_config(
        native_fixture,
        native_fixture,
        {
            "algorithm": "grpo-guard",
            "training_mode": "full",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
        },
    )
    assert config.model_args.model_type == "flux1-kontext"
    assert (
        config.training_args.condition_image_size
        == config.eval_args.condition_image_size
        == [32, 64]
    )
    assert not config.parallelism_args.sequence_parallel
    row = dataset_record(native_fixture, "a red image", 9)
    assert row["prompt"] == "a red image"
    assert len(row["images"]) == 1 and Image.open(row["images"][0]).size == (64, 64)
    validate_context_parallel_config(config, Flux1KontextAdapter, get_trainer_class("grpo-guard"))


def test_shared_input_seam_keeps_real_reference_records_and_condition_size(
    monkeypatch, native_fixture
):
    import json

    import flux_kontext_coupled_fixture as fixture
    from context_parallel_overlap_runner import make_overlap_config, write_overlap_datasets

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = {
        "algorithm": "grpo-guard",
        "training_mode": "full",
        "scheduling_mode": "ordered",
        "dp_size": 2,
        "tp_size": 2,
        "cp_size": 2,
    }
    write_overlap_datasets(native_fixture, fixture)
    base = fixture.make_config(native_fixture, native_fixture, options)
    config = make_overlap_config(base, native_fixture, native_fixture, options)
    assert (
        config.training_args.condition_image_size
        == config.eval_args.condition_image_size
        == [32, 64]
    )
    assert config.eval_args.eval_freq == 1
    for directory, split, count in (("tiny", "train", 16), ("odd", "test", 3)):
        rows = [
            json.loads(line)
            for line in (native_fixture / directory / f"{split}.jsonl").read_text().splitlines()
        ]
        assert len(rows) == count
        for row in rows:
            assert len(row["images"]) == 1
            with Image.open(row["images"][0]) as image:
                assert image.size == (64, 64)


@pytest.mark.parametrize("reference_count", [0, 2])
def test_cp_rejects_reference_count_before_image_encoding(guarded, reference_count):
    adapter, _ = guarded
    images = [[Image.new("RGB", (64, 64)) for _ in range(reference_count)]]
    with pytest.raises(ValueError, match="exactly one reference image"):
        adapter.encode_image(images, condition_image_size=(32, 64))
