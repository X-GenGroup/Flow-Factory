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

"""Native remaining Wan metadata, schedule, conditioning and gradient contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from wan_remaining_cp_fixture import (
    VARIANTS,
    adapter_class,
    build_transformer,
    forward_inputs,
    install_lora,
    native_velocity,
    pipeline_config,
)

from flow_factory.models.wan._cp_contract import (
    validate_wan_context_parallel_configs,
    wan_context_parallel_components,
)


def adapter_for(variant, monkeypatch, *, mode="full"):
    cls = adapter_class(variant)
    module = build_transformer(variant)
    secondary = build_transformer(variant, seed=71) if variant.endswith("dual") else None
    if mode == "lora":
        module = install_lora(module, variant)
        secondary = install_lora(secondary, variant) if secondary is not None else None
    adapter = object.__new__(cls)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.config = SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=True))
    adapter.pipeline = SimpleNamespace(
        transformer=module, transformer_2=secondary, config=pipeline_config(variant)
    )
    adapter.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    monkeypatch.setattr(cls, "transformer", property(lambda self: self.pipeline.transformer))
    monkeypatch.setattr(cls, "transformer_2", property(lambda self: self.pipeline.transformer_2))
    return adapter


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ("full", "lora"))
def test_native_variant_cfg_schedule_and_every_parameter_gradient(monkeypatch, variant, mode):
    adapter = adapter_for(variant, monkeypatch, mode=mode)
    modules = [adapter.pipeline.transformer]
    if adapter.pipeline.transformer_2 is not None:
        modules.append(adapter.pipeline.transformer_2)
    references = [copy.deepcopy(module) for module in modules]
    for index, (module, reference) in enumerate(zip(modules, references)):
        args = forward_inputs(variant, timestep=960.0 if index == 0 else 400.0)
        output = adapter.forward(**args).velocity
        expected = native_velocity(reference, variant, args)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().mean().backward()
        expected.square().mean().backward()
        for (name, actual), (_, target) in zip(
            module.named_parameters(), reference.named_parameters()
        ):
            if actual.requires_grad:
                assert actual.grad is not None and target.grad is not None, name
                torch.testing.assert_close(actual.grad, target.grad, rtol=0, atol=0, msg=name)
            else:
                assert actual.grad is None and target.grad is None, name
        if len(modules) == 2 and index == 0:
            assert all(parameter.grad is None for parameter in modules[1].parameters())
        module.zero_grad(set_to_none=True)
    assert any(
        parameter.grad is not None
        for reference in references
        for parameter in reference.parameters()
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_published_layout_metadata_and_runtime_inputs_admit_cp(monkeypatch, variant):
    adapter = adapter_for(variant, monkeypatch)
    cls = adapter_class(variant)
    namespace = cls.__module__
    monkeypatch.setattr(f"{namespace}.context_parallel_geometry", lambda _: (2, 0))
    monkeypatch.setattr(f"{namespace}.tensor_parallel_geometry", lambda _: (2, 0))
    config = vars(pipeline_config(variant)) | {"transformer_2": adapter.pipeline.transformer_2}
    configs = {
        name: getattr(adapter.pipeline, name).config
        for name in wan_context_parallel_components(config)
    }
    validate_wan_context_parallel_configs(
        configs, config, 2, 2, image_conditioned=not variant.startswith("t2v")
    )
    for module in (adapter.pipeline.transformer, adapter.pipeline.transformer_2):
        if module is not None:
            cls.validate_context_parallel_module(module, 2, 2)
    args = forward_inputs(variant)
    if variant.startswith("t2v"):
        adapter._validate_context_parallel_inputs(args["latents"], args["prompt_embeds"], None)
    else:
        adapter._validate_context_parallel_inputs(
            args["t"],
            args["latents"],
            args["latent_condition"],
            args["prompt_embeds"],
            args["image_embeds"],
            args["negative_prompt_embeds"],
            None,
            args.get("first_frame_mask"),
        )


@pytest.mark.parametrize(
    "patch",
    (
        {"boundary_ratio": 0.9},
        {"boundary_ratio": float("nan")},
        {"boundary_ratio": True},
        {"boundary_ratio": 2},
        {"expand_timesteps": 1},
        {"expand_timesteps": True, "transformer_2": ["diffusers", "WanTransformer3DModel"]},
    ),
)
def test_invalid_schedules_fail_before_model_loading(patch):
    with pytest.raises(ValueError, match="Wan CP"):
        wan_context_parallel_components(patch)


def test_dual_metadata_requires_identical_condition_geometry():
    primary = dict(build_transformer("i2v_dual").config)
    secondary = primary | {"text_dim": 32}
    with pytest.raises(ValueError, match="transformer_2.*text_dim"):
        validate_wan_context_parallel_configs(
            {"transformer": primary, "transformer_2": secondary},
            {"boundary_ratio": 0.9, "transformer_2": ["diffusers", "WanTransformer3DModel"]},
            2,
            2,
            image_conditioned=True,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("first_frame_mask", None),
        ("first_frame_mask", torch.zeros(1, 1, 3, 4, 4)),
        ("latent_condition", torch.zeros(1, 48, 3, 4, 4)),
        ("image_embeds", torch.zeros(1, 257, 16)),
    ),
)
def test_expanded_condition_mismatch_fails_before_collectives(monkeypatch, field, value):
    adapter = adapter_for("i2v_expanded", monkeypatch)
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_i2v.context_parallel_geometry", lambda _: (2, 0)
    )
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_i2v.tensor_parallel_geometry", lambda _: (2, 0)
    )
    args = forward_inputs("i2v_expanded")
    args[field] = value
    with pytest.raises(ValueError, match="Wan I2V"):
        adapter.forward(**args)


def test_first_last_image_order_changes_native_conditioning():
    variant = "flf2v"
    module = build_transformer(variant)
    # Official position table is learned; a nonuniform table makes endpoint order observable.
    with torch.no_grad():
        module.condition_embedder.image_embedder.pos_embed.normal_(0, 0.2)
    args = forward_inputs(variant)
    with torch.no_grad():
        expected = native_velocity(module, variant, args)
        reversed_image = native_velocity(
            module, variant, args | {"image_embeds": args["image_embeds"].flip(0)}
        )
    assert not torch.allclose(expected, reversed_image)


@pytest.mark.parametrize("variant", ("i2v_dual", "i2v_expanded", "flf2v"))
def test_real_native_encoders_reload_and_rollout_offline_condition_identity(
    tmp_path, monkeypatch, variant
):
    from PIL import Image
    from wan_remaining_cp_fixture import build_fixture

    from diffusers import WanImageToVideoPipeline
    from flow_factory.models.wan._conditioning import restore_wan_i2v_condition_pixels

    build_fixture(tmp_path, variant)
    pipeline = WanImageToVideoPipeline.from_pretrained(
        tmp_path / "model", torch_dtype=torch.float32
    )
    adapter = adapter_for(variant, monkeypatch)
    adapter.pipeline = pipeline
    cls = adapter_class(variant)
    monkeypatch.setattr(cls, "vae", property(lambda self: self.pipeline.vae))
    monkeypatch.setattr(cls, "image_encoder", property(lambda self: self.pipeline.image_encoder))
    size = 64 if variant == "i2v_expanded" else 32
    adapter.training_args = SimpleNamespace(height=size, width=size, num_frames=9, frame_rate=4.0)
    images = [Image.open(tmp_path / "data/0-first.png")]
    if variant != "i2v_expanded":
        images.append(Image.open(tmp_path / "data/0-last.png"))
    encoded = adapter.encode_image([images], height=size, width=size)
    prepared = adapter.build_condition_state_preparer().prepare_condition_state(encoded)
    first, last = restore_wan_i2v_condition_pixels(
        encoded["condition_images"],
        batch_size=1,
        height=size,
        width=size,
        device=torch.device("cpu"),
    )
    rollout = adapter.prepare_latents(
        first,
        1,
        pipeline.vae.config.z_dim,
        size,
        size,
        9,
        generator=torch.Generator().manual_seed(41),
        last_image=last,
    )
    torch.testing.assert_close(
        rollout[1], prepared.forward_context["latent_condition"], rtol=0, atol=0
    )
    assert rollout[0].shape == (1, pipeline.vae.config.z_dim, 3, 4, 4)
    if variant == "i2v_expanded":
        torch.testing.assert_close(
            rollout[2], prepared.forward_context["first_frame_mask"], rtol=0, atol=0
        )
        assert rollout[1].shape == (1, 48, 1, 4, 4)
    else:
        assert rollout[1].shape == (1, 20, 3, 4, 4)
        reversed_encoded = adapter.encode_image([images[::-1]], height=size, width=size)
        reversed_condition = adapter.build_condition_state_preparer().prepare_condition_state(
            reversed_encoded
        )
        assert not torch.equal(rollout[1], reversed_condition.forward_context["latent_condition"])
    if variant == "flf2v":
        assert prepared.forward_context["image_embeds"].shape == (2, 257, 16)
        torch.testing.assert_close(
            prepared.forward_context["image_embeds"].flip(0),
            reversed_condition.forward_context["image_embeds"],
            rtol=0,
            atol=0,
        )
    else:
        assert "image_embeds" not in prepared.forward_context
    prompt = adapter.encode_prompt(
        "a red video", negative_prompt="a blue video", guidance_scale=2.0
    )
    assert prompt["prompt_embeds"].shape == prompt["negative_prompt_embeds"].shape == (1, 512, 16)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_scheduled_denoiser_metadata_is_checked_before_weights(monkeypatch, variant):
    from diffusers import WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel

    cls = adapter_class(variant)
    adapter = object.__new__(cls)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2, effective_tensor_parallel_size=2
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="metadata-only")
    manifest = vars(pipeline_config(variant)) | {
        "transformer_2": (
            ["diffusers", "WanTransformer3DModel"] if variant.endswith("dual") else None
        )
    }
    pipeline_cls = WanPipeline if variant.startswith("t2v") else WanImageToVideoPipeline
    monkeypatch.setattr(pipeline_cls, "load_config", lambda *args: manifest)
    config = dict(build_transformer(variant).config)
    reads = []

    def load_config(path, subfolder):
        assert path == "metadata-only"
        reads.append(subfolder)
        return config

    monkeypatch.setattr(WanTransformer3DModel, "load_config", load_config)
    marker = object()
    adapter._load_diffusers_pipeline = lambda *args: marker
    assert adapter.load_pipeline() is marker
    assert reads == list(wan_context_parallel_components(manifest))
    if variant.endswith("dual"):
        monkeypatch.setattr(
            WanTransformer3DModel,
            "load_config",
            lambda path, subfolder: config
            | ({"text_dim": 32} if subfolder == "transformer_2" else {}),
        )
        adapter._load_diffusers_pipeline = lambda *args: pytest.fail(
            "weights loaded before secondary geometry rejection"
        )
        with pytest.raises(ValueError, match="transformer_2.*text_dim"):
            adapter.load_pipeline()


@pytest.mark.parametrize("variant", ("i2v_dual", "i2v_expanded", "flf2v"))
def test_conditioned_lifecycle_recipe_preserves_dual_targets_and_generation_kwargs(
    tmp_path, monkeypatch, variant
):
    import wan_remaining_coupled_fixture as recipe
    from context_parallel_overlap_runner import make_overlap_config

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = dict(
        dp_size=2,
        tp_size=2,
        cp_size=2,
        training_mode="full",
        algorithm="grpo",
        scheduling_mode="ordered",
    )
    base = recipe.make_config(tmp_path, tmp_path, options, variant)
    config = make_overlap_config(base, tmp_path, tmp_path, options)
    assert config.model_args.model_type == "wan2_i2v"
    assert config.model_args.target_components == (
        ["transformer", "transformer_2"] if variant.endswith("dual") else ["transformer"]
    )
    if variant == "i2v_dual":
        # Keeping the frozen sibling's canonical key is what reaches shared CP preparation.
        adapter = object.__new__(adapter_class(variant))
        targets = adapter._parse_target_modules(
            ["transformer.*"], config.model_args.target_components
        )
        assert targets == {"transformer": ["*"], "transformer_2": None}
        assert adapter._parse_target_modules(["transformer.*"], ["transformer"]) == {
            "transformer": ["*"]
        }
    assert config.training_args.num_frames == config.eval_args.num_frames == 9
    assert config.training_args.guidance_scale == config.eval_args.guidance_scale == 2.0
    assert config.training_args.extra_kwargs["guidance_scale_2"] == 3.0
    assert config.eval_args.extra_kwargs["guidance_scale_2"] == 3.0
    assert config.training_args.extra_kwargs["negative_prompt"] == "blue"
    row = recipe.dataset_record(tmp_path, "a video", 3, variant)
    assert row["images"][0].endswith("3-first.png")
    if variant != "i2v_expanded":
        assert row["images"][1].endswith("3-last.png")
