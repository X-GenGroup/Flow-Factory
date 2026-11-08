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

"""Flux2 native metadata, CP layouts and public scheduler derivative contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from flux2_cp_fixture import (
    VARIANTS,
    adapter_class,
    build_transformer,
    fixed_inputs,
    install_lora,
    make_adapter,
    native_velocity,
)

from flow_factory.models.flux._flux2_parallel import Flux2PackedAttentionMetadata


@pytest.mark.parametrize("variant", VARIANTS)
def test_native_metadata_and_missing_dev_guidance_default(variant):
    cls = adapter_class(variant)
    config = dict(build_transformer(variant).config)
    cls._validate_context_parallel_config(config, 2, 2)
    if variant == "dev":
        config.pop("guidance_embeds")
        cls._validate_context_parallel_config(config, 2, 2)
    config["num_attention_heads"] = 30
    with pytest.raises(ValueError, match="num_attention_heads=30"):
        cls._validate_context_parallel_config(config, 2, 2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_native_public_velocity_and_every_parameter_gradient(variant):
    model = build_transformer(variant)
    reference = copy.deepcopy(model)
    args = fixed_inputs(variant)
    adapter = make_adapter(model, variant)
    actual = adapter.forward(
        **args, compute_log_prob=False, return_kwargs=["velocity"], noise_level=0.35
    ).velocity
    expected = native_velocity(reference, args, variant)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, parameter), (_, target) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert parameter.grad is not None and target.grad is not None, name
        torch.testing.assert_close(parameter.grad, target.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("variant", VARIANTS)
def test_native_packed_metadata_localizes_once_after_reload(tmp_path, variant):
    from diffusers import Flux2Transformer2DModel

    original = build_transformer(variant)
    original.save_pretrained(tmp_path)
    model = Flux2Transformer2DModel.from_pretrained(tmp_path)
    attention = model.single_transformer_blocks[0].attn
    width = attention.mlp_hidden_dim
    blocks = tuple(attention.to_qkv_mlp_proj._tp_packed_col_blocks)
    style = Flux2PackedAttentionMetadata()
    style._apply(attention, SimpleNamespace(size=lambda: 2))
    style._apply(attention, SimpleNamespace(size=lambda: 2))
    assert attention.mlp_hidden_dim == width // 2
    assert tuple(attention.to_qkv_mlp_proj._tp_packed_col_blocks) == blocks
    assert attention.inner_dim == original.single_transformer_blocks[0].attn.inner_dim
    with pytest.raises(ValueError, match="localized"):
        style._apply(attention, SimpleNamespace(size=lambda: 4))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("fields", ["all", "velocity"])
def test_native_singleton_preserves_guard_graph_and_parameter_gradients(variant, mode, fields):
    from coupled_objective_reference import assert_objective

    from flow_factory.samples import ComponentTimes, LatentState, MultiModalStepOutput, ReplayStep
    from flow_factory.scheduler import (
        FlowMatchEulerDiscreteSDESchedulerOutput,
        SchedulerGroup,
    )
    from flow_factory.trainers.rl.grpo import GRPOGuardTrainer

    model = build_transformer(variant)
    if mode == "lora":
        model = install_lora(model, variant)
    adapter = make_adapter(model, variant)
    adapter.scheduler_group = SchedulerGroup({"latent": adapter.scheduler}, primary_name="latent")
    inputs = fixed_inputs(variant)
    inputs["noise_level"] = 0.35
    if fields == "velocity":
        inputs.update(compute_log_prob=False, return_kwargs=["velocity"])
    native_outputs = []
    original_step = adapter.scheduler.step

    def observe_step(**kwargs):
        result = original_step(**kwargs)
        native_outputs.append(result)
        return result

    adapter.scheduler.step = observe_step
    output = adapter.forward(**inputs)
    (native,) = native_outputs
    native_fields, public_fields = native.to_dict(), output.to_dict()
    assert type(output) is FlowMatchEulerDiscreteSDESchedulerOutput
    assert public_fields.keys() == native_fields.keys()
    for name, expected in native_fields.items():
        actual = public_fields[name]
        if expected is None:
            assert actual is None
        else:
            assert actual is expected
            assert type(actual) is type(expected) and actual.dtype == expected.dtype
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    if fields == "velocity":
        assert output.next_latents_mean is None and output.log_prob is None
        return

    # The public mean and velocity must be ancestors of the public log probability,
    # including the contribution independently checked by the coupled-loss oracle.
    upstreams = torch.autograd.grad(
        output.log_prob.sum(), (output.next_latents_mean, output.velocity), retain_graph=True
    )
    assert all(torch.count_nonzero(value) for value in upstreams)
    concatenated = type(native).from_dict(
        {
            name: torch.cat([value], dim=0) if value is not None else None
            for name, value in native_fields.items()
        }
    )
    trainer = object.__new__(GRPOGuardTrainer)
    trainer.adapter = adapter
    trainer.training_args = SimpleNamespace(kl_type="v-based")
    old_log_prob = native.log_prob.detach() - 0.02
    replay = ReplayStep(
        state=LatentState({"latent": inputs["latents"]}),
        next_state=LatentState({"latent": inputs["next_latents"]}),
        times=ComponentTimes(
            timestep={"latent": inputs["t"]}, next_timestep={"latent": inputs["t_next"]}
        ),
        log_prob=old_log_prob,
        component_log_probs={"latent": old_log_prob},
    )
    old_mean = LatentState({"latent": native.next_latents_mean.detach() + 0.01})
    reference = MultiModalStepOutput(
        velocity=LatentState({"latent": native.velocity.detach() + 0.03})
    )
    advantage, beta = torch.tensor([0.7]), 0.01

    def guard_loss(raw):
        step = MultiModalStepOutput(
            next_state_mean=LatentState({"latent": raw.next_latents_mean}),
            velocity=LatentState({"latent": raw.velocity}),
            std_dev_t={"latent": raw.std_dev_t},
            dt={"latent": raw.dt},
            log_prob=raw.log_prob,
            component_log_probs={"latent": raw.log_prob},
        )
        ratio = trainer._guard_ratio(step, replay, old_mean)
        policy = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2)).mean()
        return policy + beta * trainer._reference_kl_divergence(step, reference, replay)

    loss, legacy_loss = guard_loss(output), guard_loss(concatenated)
    torch.testing.assert_close(loss, legacy_loss, rtol=0, atol=0)
    parameters = [value for value in model.parameters() if value.requires_grad]
    actual_gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    legacy_gradients = torch.autograd.grad(
        legacy_loss, parameters, allow_unused=True, retain_graph=True
    )
    for actual, expected in zip(actual_gradients, legacy_gradients):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mean_gradient, velocity_gradient = torch.autograd.grad(
        loss * 0.5, (output.next_latents_mean, output.velocity), retain_graph=True
    )
    assert_objective(
        {
            "algorithm": "grpo-guard",
            "mean": output.next_latents_mean.detach(),
            "velocity": output.velocity.detach(),
            "following": inputs["next_latents"],
            "old_mean": old_mean.components["latent"],
            "reference": reference.velocity.components["latent"],
            "log_prob": output.log_prob.detach(),
            "old_log_prob": old_log_prob,
            "std": output.std_dev_t,
            "dt": output.dt,
            "sigma": inputs["t"] / 1000,
            "advantage": advantage,
            "adv_clip_range": (-3, 3),
            "clip_range": (-0.2, 0.2),
            "kl_type": "v-based",
            "kl_beta": beta,
            "cp_size": 2,
            "gas": 4,
            "mean_gradient": mean_gradient,
            "velocity_gradient": velocity_gradient,
            "loss": loss.detach(),
        }
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_saved_native_text_vae_reference_conditions(tmp_path, variant):
    from flux2_cp_fixture import build_fixture, load_pipeline, pipeline_adapter
    from PIL import Image

    build_fixture(tmp_path, variant)
    pipeline = load_pipeline(tmp_path, variant)
    adapter = pipeline_adapter(pipeline, variant)
    prompt_options = {} if variant == "dev" else {"negative_prompt": "blue", "guidance_scale": 2.0}
    with torch.no_grad():
        prompt = adapter.encode_prompt("a red image", max_sequence_length=8, **prompt_options)
        image = adapter.encode_image(
            Image.open(tmp_path / "data" / "0.png"), condition_image_size=(32, 64)
        )
        assert prompt["prompt_embeds"].shape == (1, 8, 24)
        assert image["image_latents"][0].shape == (8, 128)
        assert image["image_latent_ids"][0].shape == (8, 4)
        assert image["condition_images"][0][0].shape == (3, 32, 64)
        assert torch.count_nonzero(image["image_latents"][0])
        native_prompt = pipeline.encode_prompt("a red image", max_sequence_length=8)
        torch.testing.assert_close(prompt["prompt_embeds"], native_prompt[0], rtol=0, atol=0)
        if variant != "dev":
            torch.testing.assert_close(
                prompt["negative_prompt_embeds"],
                pipeline.encode_prompt("blue", max_sequence_length=8)[0],
                rtol=0,
                atol=0,
            )
        pixels = pipeline.image_processor.preprocess(
            Image.open(tmp_path / "data" / "0.png"), height=32, width=64
        )
        native_image, native_ids = pipeline.prepare_image_latents(
            images=[pixels],
            batch_size=1,
            generator=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(image["image_latents"][0], native_image[0], rtol=0, atol=0)
        torch.testing.assert_close(image["image_latent_ids"][0], native_ids[0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"guidance_embeds": False}, "guidance_embeds"),
        ({"num_attention_heads": 30}, "num_attention_heads=30"),
        ({"num_layers": 0}, "both native"),
        ({"num_single_layers": 0}, "both native"),
        ({"axes_dims_rope": (2, 2, 4)}, "four positive even"),
        ({"axes_dims_rope": (1, 3, 2, 2)}, "four positive even"),
        ({"out_channels": 64}, "equal positive"),
        ({"patch_size": 2}, "patch_size=1"),
        ({"mlp_ratio": float("nan")}, "finite mlp_ratio"),
    ],
)
def test_cp_metadata_rejected_before_weights(monkeypatch, patch, error):
    from diffusers import Flux2Transformer2DModel

    cls = adapter_class("dev")
    adapter = object.__new__(cls)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2, effective_tensor_parallel_size=2
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    config = {**dict(build_transformer("dev").config), **patch}
    monkeypatch.setattr(Flux2Transformer2DModel, "load_config", lambda *a, **kw: config)
    adapter._load_diffusers_pipeline = lambda *a, **kw: pytest.fail(
        "weights loaded before rejection"
    )
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


@pytest.mark.parametrize("variant", VARIANTS)
def test_cp_dense_native_processor_and_boundary_ownership(variant):
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.models._modeling_parallel import ContextParallelOutput

    cls, model = adapter_class(variant), build_transformer(variant)
    cls.validate_context_parallel_module(model, 2, 2)
    split = ContextParallelSplitHook(copy.deepcopy(model._cp_plan[""]), None)
    gather = ContextParallelGatherHook([model._cp_plan["proj_out"]], None)
    model._diffusers_hook = SimpleNamespace(get_hook={"cp_input---": split}.get)
    model.proj_out._diffusers_hook = SimpleNamespace(get_hook={"cp_output---proj_out": gather}.get)
    assert cls.compose_context_parallel_sequence_boundaries(model) == ()
    assert model._diffusers_hook.get_hook("cp_input---") is split
    assert model.proj_out._diffusers_hook.get_hook("cp_output---proj_out") is gather
    gather.metadata = [ContextParallelOutput(gather_dim=2, expected_dims=3)]
    with pytest.raises(ValueError, match="native root"):
        cls.compose_context_parallel_sequence_boundaries(model)
    model.single_transformer_blocks[0].attn.processor = torch.nn.Identity()
    with pytest.raises(ValueError, match="without reference KV caches"):
        cls.validate_context_parallel_module(model, 2, 2)


@pytest.fixture
def guarded_flux2(monkeypatch):
    owner = "flow_factory.models.flux._flux2_parallel"
    monkeypatch.setattr(f"{owner}.context_parallel_geometry", lambda _: (2, 0))
    monkeypatch.setattr(f"{owner}.tensor_parallel_geometry", lambda _: (2, 0))
    adapter = make_adapter(build_transformer("klein4b"), "klein4b")
    return adapter, fixed_inputs("klein4b")


def _validate(adapter, inputs):
    names = (
        "t",
        "latents",
        "latent_ids",
        "prompt_embeds",
        "text_ids",
        "image_latents",
        "image_latent_ids",
    )
    adapter._validate_flux2_context_parallel_inputs(
        *(inputs[name] for name in names),
        inputs.get("joint_attention_kwargs"),
        negative_prompt_embeds=inputs.get("negative_prompt_embeds"),
        negative_text_ids=inputs.get("negative_text_ids"),
    )


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("t", torch.zeros(2), "one sample timestep"),
        ("latents", torch.zeros(2, 16, 128), "target must have shape"),
        ("latent_ids", torch.zeros(16, 4), "target IDs"),
        ("image_latents", torch.zeros(1, 0, 128), "reference must have shape"),
        ("image_latents", torch.zeros(1, 8, 64), "reference must have shape"),
        ("image_latents", None, "IDs require reference latents"),
        ("image_latent_ids", torch.zeros(1, 8, 3), "reference IDs"),
        ("prompt_embeds", torch.zeros(1, 7, 24), "text IDs"),
        ("negative_prompt_embeds", None, "IDs require negative prompt"),
        ("negative_text_ids", None, "negative text IDs"),
        ("joint_attention_kwargs", {"attention_mask": torch.ones(8)}, "attention masks"),
    ],
)
def test_cp_input_guards_precede_collectives(guarded_flux2, key, value, error):
    adapter, inputs = guarded_flux2
    inputs[key] = value
    adapter.pipeline.transformer.forward = lambda *a, **kw: pytest.fail("collective before guard")
    with pytest.raises(ValueError, match=error):
        _validate(adapter, inputs)


def test_cp_combined_reference_span_and_each_text_divisibility(guarded_flux2, monkeypatch):
    adapter, inputs = guarded_flux2
    for field in ("latents", "latent_ids"):
        inputs[field] = inputs[field][:, :15]
    for field in ("image_latents", "image_latent_ids"):
        inputs[field] = torch.cat([inputs[field], inputs[field][:, :1]], dim=1)
    _validate(adapter, inputs)
    inputs["image_latents"] = inputs["image_latents"][:, :7]
    inputs["image_latent_ids"] = inputs["image_latent_ids"][:, :7]
    with pytest.raises(ValueError, match="combined target/reference tokens=22"):
        _validate(adapter, inputs)
    adapter.config.parallelism_args.sequence_parallel = False
    _validate(adapter, inputs)
    inputs["negative_prompt_embeds"] = inputs["negative_prompt_embeds"][:, :7]
    inputs["negative_text_ids"] = inputs["negative_text_ids"][:, :7]
    with pytest.raises(ValueError, match="negative text tokens=7"):
        _validate(adapter, inputs)
    monkeypatch.setattr(
        "flow_factory.models.flux._flux2_parallel.context_parallel_geometry", lambda _: (1, 0)
    )
    _validate(adapter, {key: None for key in inputs})


@pytest.mark.parametrize(
    "variant,trainer_name,mode", [("dev", "grpo-guard", "full"), ("klein4b", "dppo", "lora")]
)
def test_reference_lifecycle_recipe_admission(monkeypatch, tmp_path, variant, trainer_name, mode):
    import flux2_coupled_fixture as fixture
    from context_parallel_overlap_runner import make_overlap_config

    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = {
        "algorithm": trainer_name,
        "training_mode": mode,
        "variant": variant,
        "dp_size": 2,
        "tp_size": 2,
        "cp_size": 2,
        "scheduling_mode": "ordered",
    }
    config = make_overlap_config(
        fixture.make_config(tmp_path, tmp_path, options), tmp_path, tmp_path, options
    )
    assert config.parallelism_args.sequence_parallel is True
    assert config.training_args.extra_kwargs["condition_image_size"] == [32, 64]
    assert config.eval_args.extra_kwargs["condition_image_size"] == [32, 64]
    assert config.training_args.extra_kwargs["max_sequence_length"] == 8
    assert config.eval_args.eval_freq == 1 and config.training_args.reward_optimization_overlap
    if variant != "dev":
        assert config.training_args.extra_kwargs["negative_prompt"] == "blue"
    validate_context_parallel_config(
        config, adapter_class(variant), get_trainer_class(trainer_name)
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_cached_reference_inference_native_trajectory_and_decode(tmp_path, variant):
    from flux2_cp_fixture import build_fixture, encoded_conditions, load_pipeline, pipeline_adapter

    from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler

    build_fixture(tmp_path, variant)
    pipeline = load_pipeline(tmp_path, variant)
    adapter = pipeline_adapter(pipeline, variant)
    adapter.scheduler = FlowMatchEulerDiscreteSDEScheduler(dynamics_type="ODE")
    prompt, image, _ = encoded_conditions(str(tmp_path), variant, 0)
    condition = {
        **prompt,
        "condition_images": image["condition_images"],
        "image_latents": torch.stack(image["image_latents"]),
        "image_latent_ids": torch.stack(image["image_latent_ids"]),
    }
    with torch.no_grad():
        (sample,) = adapter.inference(
            prompt=["a blue image"],
            height=64,
            width=64,
            num_inference_steps=3,
            guidance_scale=2.0,
            max_sequence_length=8,
            generator=torch.Generator().manual_seed(173),
            **condition,
        )
        assert sample.all_latents.shape == (4, 16, 128)
        assert sample.image.shape == (3, 64, 64)
        assert sample.image_latents.shape == (8, 128)
        for step, timestep in enumerate(sample.timesteps):
            values = {
                **fixed_inputs(variant),
                **{key: value for key, value in condition.items() if key in fixed_inputs(variant)},
                "latents": sample.all_latents[step].unsqueeze(0),
                "latent_ids": sample.latent_ids.unsqueeze(0),
                "t": timestep.reshape(1),
            }
            velocity = native_velocity(pipeline.transformer, values, variant)
            following = adapter.scheduler.step(
                velocity=velocity,
                timestep=timestep,
                timestep_next=sample.timesteps[step + 1] if step < 2 else torch.tensor(0),
                latents=values["latents"],
                noise_level=0.0,
                compute_log_prob=False,
                return_dict=True,
            ).next_latents
            torch.testing.assert_close(following[0], sample.all_latents[step + 1], rtol=0, atol=0)
        spatial = pipeline._unpack_latents_with_ids(
            sample.all_latents[-1].unsqueeze(0), sample.latent_ids.unsqueeze(0)
        )
        mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1)
        std = (
            pipeline.vae.bn.running_var.view(1, -1, 1, 1) + pipeline.vae.config.batch_norm_eps
        ).sqrt()
        raw = pipeline._unpatchify_latents(spatial * std + mean)
        decoded = pipeline.vae.decode(raw, return_dict=False)[0]
        expected = pipeline.image_processor.postprocess(decoded, output_type="pt")[0]
        torch.testing.assert_close(sample.image, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_latents_positions_and_guidance_are_effective(variant):
    model = build_transformer(variant)
    inputs = fixed_inputs(variant)
    with torch.no_grad():
        expected = native_velocity(model, inputs, variant)
        fields = ["image_latents", "image_latent_ids"]
        if variant != "dev":
            fields.append("negative_prompt_embeds")
        for field in fields:
            changed = {**inputs, field: torch.zeros_like(inputs[field])}
            assert not torch.allclose(
                native_velocity(model, changed, variant), expected, rtol=1e-5, atol=1e-7
            ), field
        assert not torch.allclose(
            native_velocity(model, {**inputs, "guidance_scale": 1.0}, variant),
            expected,
            rtol=1e-5,
            atol=1e-7,
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_public_multi_sample_aggregation_retains_scheduler_values(variant):
    adapter = make_adapter(build_transformer(variant), variant)
    inputs = fixed_inputs(variant)
    batched = {
        key: torch.cat([value, value], dim=0) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    outputs = []
    original = adapter.scheduler.step

    def observe(**kwargs):
        result = original(**kwargs)
        outputs.append(result)
        return result

    adapter.scheduler.step = observe
    output = adapter.forward(**batched, noise_level=0.35)
    assert len(outputs) == 2
    for key, value in output.to_dict().items():
        expected = [item.to_dict()[key] for item in outputs]
        if value is None:
            assert expected == [None, None]
        else:
            torch.testing.assert_close(value, torch.cat(expected, dim=0), rtol=0, atol=0)
