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

"""Flux3 native metadata, CP layouts and public scheduler derivative contracts."""

import copy
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from flux3_cp_fixture import (
    VARIANTS,
    build_transformer,
    fixed_inputs,
    install_lora,
    make_adapter,
    native_velocity,
)

from flow_factory.models.flux.flux3 import Flux3Adapter


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
def test_published_base_default_and_distilled_native_metadata(variant):
    config = dict(build_transformer(variant).config)
    Flux3Adapter._validate_context_parallel_config(config, 2, 2)
    if variant == "base":
        config.pop("flow_map")
        Flux3Adapter._validate_context_parallel_config(config, 2, 2)
    assert Flux3Adapter.context_parallel_diffusers_revision is None


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"num_attention_heads": 30}, "num_attention_heads=30"),
        ({"num_layers": 0}, "all native"),
        ({"num_single_layers": 0}, "all native"),
        ({"axes_dims_rope": (2, 2, 4)}, "four positive even"),
        ({"axes_dims_rope": (1, 3, 2, 2)}, "four positive even"),
        ({"cond_in_channels": 64}, "equal positive"),
        ({"flow_map": 1}, "flow_map must be boolean"),
        ({"mlp_ratio": float("nan")}, "positive and finite"),
    ],
)
def test_cp_metadata_fails_before_loading_weights(monkeypatch, patch, error):
    from diffusers import Flux3Transformer2DModel, ModularPipeline

    adapter = object.__new__(Flux3Adapter)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2, effective_tensor_parallel_size=2
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    config = {**dict(build_transformer("base").config), **patch}
    monkeypatch.setattr(Flux3Transformer2DModel, "load_config", lambda *a, **kw: config)
    monkeypatch.setattr(
        ModularPipeline, "from_pretrained", lambda *a, **kw: pytest.fail("weights before guard")
    )
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


@pytest.mark.parametrize("variant", VARIANTS)
def test_cp_plan_owns_all_mode_streams_ids_and_target_gather(variant):
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.hooks.hooks import HookRegistry

    from flow_factory.models.flux._flux3_parallel import flux3_context_parallel_plan

    model = build_transformer(variant)
    Flux3Adapter.validate_context_parallel_module(model, 2, 2)
    plan = flux3_context_parallel_plan()
    assert set(plan[""]) == {
        "hidden_states",
        "encoder_hidden_states",
        "image_cond_hidden_states",
        "img_ids",
        "txt_ids",
        "image_cond_ids",
    }
    assert all(
        value.split_dim == (0 if name.endswith("ids") else 1) for name, value in plan[""].items()
    )
    split = ContextParallelSplitHook(copy.deepcopy(plan[""]), None)
    gather = ContextParallelGatherHook([plan["proj_out"]], None)
    HookRegistry.check_if_exists_or_initialize(model).register_hook(split, "cp_input---")
    HookRegistry.check_if_exists_or_initialize(model.proj_out).register_hook(
        gather, "cp_output---proj_out"
    )
    boundaries = Flux3Adapter.compose_context_parallel_sequence_boundaries(model)
    assert len(boundaries) == 1 and boundaries[0].module is model.proj_out
    assert boundaries[0].phase == "output"
    assert Flux3Adapter.compose_context_parallel_sequence_boundaries(model) == boundaries
    assert model.proj_out._diffusers_hook.get_hook("cp_output---proj_out") is None
    split.metadata.pop("image_cond_ids")
    with pytest.raises(ValueError, match="owned native root"):
        Flux3Adapter.compose_context_parallel_sequence_boundaries(model)
    split.metadata["image_cond_ids"] = plan[""]["image_cond_ids"]
    boundaries[0].handle.remove()
    with pytest.raises(ValueError, match="owned output gather"):
        Flux3Adapter.compose_context_parallel_sequence_boundaries(model)
    model.mode_blocks["image_cond"][0].attn.processor = torch.nn.Identity()
    with pytest.raises(ValueError, match="native dense"):
        Flux3Adapter.validate_context_parallel_module(model, 2, 2)


@pytest.fixture
def guarded_flux3(monkeypatch):
    owner = "flow_factory.models.flux._flux3_parallel"
    monkeypatch.setattr(f"{owner}.context_parallel_geometry", lambda _: (2, 0))
    monkeypatch.setattr(f"{owner}.tensor_parallel_geometry", lambda _: (2, 0))
    adapter = make_adapter(build_transformer("base"), "base")
    inputs = fixed_inputs("base")
    for name in ("latent_ids", "image_latent_ids", "text_ids", "negative_text_ids"):
        inputs[name] = inputs[name][0]
    return adapter, inputs


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
    adapter._validate_flux3_context_parallel_inputs(
        *(inputs.get(name) for name in names),
        inputs.get("attention_kwargs"),
        inputs.get("negative_prompt_embeds"),
        inputs.get("negative_text_ids"),
    )


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("t", torch.zeros(2), "one sample timestep"),
        ("latents", torch.zeros(2, 16, 128), "target must have shape"),
        ("latent_ids", torch.zeros(1, 16, 4), "target IDs"),
        ("image_latents", torch.zeros(1, 0, 128), "reference must have shape"),
        ("image_latents", torch.zeros(1, 8, 64), "reference must have shape"),
        ("image_latents", None, "IDs require reference latents"),
        ("image_latent_ids", torch.full((8, 4), float("nan")), "reference IDs"),
        ("prompt_embeds", torch.zeros(1, 78, 64), "tokens=78"),
        ("negative_prompt_embeds", None, "IDs require negative embeddings"),
        ("attention_kwargs", {"attention_mask": torch.ones(8)}, "masks, caches"),
    ],
)
def test_cp_input_guards_precede_collectives(guarded_flux3, key, value, error):
    adapter, inputs = guarded_flux3
    inputs[key] = value
    adapter.pipeline.transformer.forward = lambda *a, **kw: pytest.fail("collective before guard")
    with pytest.raises(ValueError, match=error):
        _validate(adapter, inputs)


def test_cp_per_mode_divisibility_optional_reference_and_negative_fallback(guarded_flux3):
    adapter, inputs = guarded_flux3
    _validate(adapter, inputs)
    # The combined24 tokens divide CP×TP, but both native early mode stacks need their own split.
    bad = {
        **inputs,
        "latents": inputs["latents"][:, :15],
        "latent_ids": inputs["latent_ids"][:15],
        "image_latents": torch.zeros(1, 9, 128),
        "image_latent_ids": torch.zeros(9, 4),
    }
    with pytest.raises(ValueError, match="target tokens=15"):
        _validate(adapter, bad)
    _validate(
        adapter,
        {**inputs, "image_latents": None, "image_latent_ids": None, "negative_text_ids": None},
    )
    adapter.config.parallelism_args.sequence_parallel = False
    _validate(
        adapter,
        {
            **inputs,
            "prompt_embeds": inputs["prompt_embeds"][:, :78],
            "text_ids": inputs["text_ids"][:78],
        },
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_mode_specific_target_time_reference_and_negative_conditions_are_effective(variant):
    model = build_transformer(variant)
    inputs = fixed_inputs(variant)
    with torch.no_grad():
        baseline = native_velocity(model, inputs, variant)
        changed = native_velocity(
            model, {**inputs, "image_latents": inputs["image_latents"] + 0.5}, variant
        )
        assert not torch.allclose(baseline, changed, rtol=1e-5, atol=1e-6)
        time_changed = native_velocity(model, {**inputs, "t_next": inputs["t_next"] + 100}, variant)
        if variant == "distilled":
            assert not torch.allclose(baseline, time_changed, rtol=1e-5, atol=1e-6)
        else:
            torch.testing.assert_close(baseline, time_changed, rtol=0, atol=0)
            neg = native_velocity(
                model,
                {**inputs, "negative_prompt_embeds": inputs["negative_prompt_embeds"] + 0.3},
                variant,
            )
            assert not torch.allclose(baseline, neg, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_saved_real_qwen3vl_vae_and_native_trajectory(tmp_path, variant):
    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks, Flux3DistilledAutoBlocks
    from diffusers.modular_pipelines.flux3.before_denoise import (
        Flux3DistilledTextRoPEInputsStep,
        Flux3PrepareImageCondLatentsStep,
        Flux3PrepareLatentsStep,
        Flux3SetTimestepsStep,
        Flux3TextRoPEInputsStep,
    )
    from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep, Flux3UnpackLatentsStep
    from diffusers.modular_pipelines.flux3.denoise import (
        Flux3DenoiseStep,
        Flux3DistilledDenoiseStep,
    )
    from diffusers.modular_pipelines.flux3.encoders import (
        Flux3ProcessImagesInputStep,
        Flux3TextEncoderStep,
        Flux3VaeEncoderStep,
    )
    from diffusers.modular_pipelines.flux3.inputs import (
        Flux3DistilledTextInputStep,
        Flux3ImageCondInputStep,
        Flux3TextInputStep,
    )
    from flux3_cp_fixture import build_fixture, load_pipeline, pipeline_adapter
    from PIL import Image

    from diffusers import ClassifierFreeGuidance, FlowMatchEulerDiscreteScheduler

    build_fixture(tmp_path, variant)
    pipeline = load_pipeline(tmp_path, variant)
    assert pipeline.transformer.config.flow_map is (variant == "distilled")
    adapter = pipeline_adapter(pipeline, variant)
    adapter._mode = "eval"
    from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler

    adapter.pipeline.scheduler = FlowMatchEulerDiscreteSDEScheduler(
        shift=6.93, use_dynamic_shifting=False, noise_level=0.0, dynamics_type="ODE"
    )
    adapter.scheduler_group = adapter.build_scheduler_group()
    flow_map = variant == "distilled"
    scale = 1.0 if flow_map else 2.0
    original = Image.open(tmp_path / "data" / "0.png")
    assert original.size == (128, 64)
    with torch.no_grad():
        prompt = adapter.encode_prompt(
            "a red image",
            negative_prompt=None if flow_map else "blue",
            guidance_scale=scale,
            max_sequence_length=80,
        )
        image = adapter.encode_image(original)
        assert prompt["prompt_embeds"].shape == (1, 80, 64)
        assert image["image_latents"][0].shape == (32, 128)
        assert image["condition_images"][0][0].shape == (3, 64, 128)
        for text, key in (("a red image", "prompt_embeds"), ("blue", "negative_prompt_embeds")):
            if flow_map and key.startswith("negative"):
                continue
            native_prompt = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                pipeline.text_encoder, pipeline.tokenizer, text, max_sequence_length=80
            )
            torch.testing.assert_close(prompt[key], native_prompt, rtol=0, atol=0)
        blocks = Flux3DistilledAutoBlocks if flow_map else Flux3AutoBlocks
        native = blocks().init_pipeline()
        components = dict(
            transformer=adapter.transformer,
            vae=adapter.vae,
            image_processor=pipeline.image_processor,
            text_encoder=adapter.text_encoder,
            tokenizer=adapter.processor,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False),
        )
        if not flow_map:
            components["guider"] = ClassifierFreeGuidance(guidance_scale=scale)
        native.update_components(**components)
        state = PipelineState(
            values=dict(
                prompt_embeds=prompt["prompt_embeds"],
                negative_prompt_embeds=prompt.get("negative_prompt_embeds"),
                image=original,
                height=64,
                width=64,
                num_images_per_prompt=1,
                generator=torch.Generator().manual_seed(123),
                num_inference_steps=3,
                timesteps=None,
                sigmas=None,
                latents=None,
                attention_kwargs=None,
                output_type="pt",
            )
        )
        steps = [
            Flux3ProcessImagesInputStep(),
            Flux3VaeEncoderStep(),
            Flux3DistilledTextInputStep() if flow_map else Flux3TextInputStep(),
            Flux3ImageCondInputStep(),
            Flux3PrepareLatentsStep(default_height=64, default_width=64, image_rope_scale=0.5),
            Flux3PrepareImageCondLatentsStep(image_rope_scale=0.5),
            Flux3SetTimestepsStep(shift=6.93, num_inference_steps=3),
            Flux3DistilledTextRoPEInputsStep() if flow_map else Flux3TextRoPEInputsStep(),
        ]
        for step in steps:
            native, state = step(native, state)
        torch.testing.assert_close(
            image["image_latents"][0], state.values["image_cond_hidden_states"][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            image["image_latent_ids"][0], state.values["image_cond_ids"], rtol=0, atol=0
        )
        trajectory = [state.values["latents"].clone()]
        original_step = native.scheduler.step

        def record_step(*args, **kwargs):
            output = original_step(*args, **kwargs)
            trajectory.append(
                (output[0] if isinstance(output, tuple) else output.prev_sample).clone()
            )
            return output

        native.scheduler.step = record_step
        for step in [
            Flux3DistilledDenoiseStep() if flow_map else Flux3DenoiseStep(),
            Flux3UnpackLatentsStep(),
            Flux3DecodeStep(),
        ]:
            native, state = step(native, state)
        samples = adapter.inference(
            images=[[original]],
            prompt=["a red image"],
            negative_prompt=None if flow_map else ["blue"],
            height=64,
            width=64,
            num_inference_steps=3,
            guidance_scale=scale,
            max_sequence_length=80,
            generator=torch.Generator().manual_seed(123),
            compute_log_prob=False,
        )
        assert len(samples) == 1 and samples[0].all_latents.shape == (4, 16, 128)
        torch.testing.assert_close(samples[0].all_latents, torch.cat(trajectory), rtol=0, atol=1e-6)
        torch.testing.assert_close(samples[0].image, state.values["images"][0], rtol=0, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_multi_sample_reference_aggregation_preserves_native_rows(variant):
    adapter = make_adapter(build_transformer(variant), variant)
    operands = [fixed_inputs(variant, replica=index) for index in range(2)]
    expected = [adapter._forward(**row, noise_level=0.35) for row in operands]
    batch = {
        key: torch.cat([row[key] for row in operands]) if isinstance(value, torch.Tensor) else value
        for key, value in operands[0].items()
    }
    actual = adapter.forward(**batch, noise_level=0.35)
    for key, value in actual.to_dict().items():
        if value is not None:
            torch.testing.assert_close(
                value, torch.cat([row.to_dict()[key] for row in expected]), rtol=0, atol=0
            )


def test_cp_native_dependency_accepts_the_released_package():
    from importlib.metadata import version

    from packaging.version import Version

    assert Flux3Adapter.context_parallel_diffusers_revision is None
    installed = Version(version("diffusers"))
    assert installed >= Version("0.42.1")
    assert not installed.is_prerelease


@pytest.mark.parametrize("backend", ["serial", "ddp", "single_fsdp2"])
def test_fsdp_schedule_guard_has_no_serial_or_ddp_collectives(monkeypatch, backend):
    adapter = object.__new__(Flux3Adapter)
    if backend != "serial":
        adapter.accelerator = SimpleNamespace(
            is_fsdp2=backend == "single_fsdp2", num_processes=1 if backend == "single_fsdp2" else 4
        )
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **kw: pytest.fail("unexpected collective"))
    adapter._assert_flux3_fsdp_schedule((1, 0), "native forward")


def _fsdp_schedule_worker(rank, store_path, result_dir, options):
    from accelerate.parallelism_config import ParallelismConfig

    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=4)
    result = {"rank": rank, "mesh": options["mesh"], "scope": "cpu_native_schedule_contract"}
    entries = []
    try:
        model = build_transformer("base")
        adapter = make_adapter(model, "base")
        accelerator = adapter.accelerator
        accelerator.is_fsdp2 = True
        accelerator.num_processes = 4
        if options["mesh"] == "world":
            accelerator.torch_device_mesh = None
        else:
            accelerator.parallelism_config = ParallelismConfig(
                dp_replicate_size=1 if options["mesh"] == "tp" else 2,
                dp_shard_size=2,
                tp_size=2 if options["mesh"] == "tp" else 1,
                cp_size=1,
                sp_size=1,
            )
            accelerator.torch_device_mesh = accelerator.parallelism_config.build_device_mesh("cpu")
        for name, module in model.named_modules():
            if type(module).__name__ in (
                "Flux3ModeTransformerBlock",
                "Flux3SingleTransformerBlock",
            ):
                module.register_forward_pre_hook(lambda _m, _args, name=name: entries.append(name))
        inputs = fixed_inputs("base")
        cases = []

        def repeat_batch(values):
            return {
                key: (
                    value.repeat((2,) + (1,) * (value.ndim - 1))
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in values.items()
            }

        for mismatch in (
            "reference",
            "cfg",
            "forward_count",
            "inference_count",
            "inference_steps",
            "empty_inference",
        ):
            values = dict(inputs)
            if rank >= 2:
                if mismatch == "reference":
                    values.update(image_latents=None, image_latent_ids=None)
                elif mismatch == "cfg":
                    values.update(negative_prompt_embeds=None, negative_text_ids=None)
                elif mismatch in ("forward_count", "inference_count"):
                    values = repeat_batch(values)
            entries.clear()
            if mismatch.startswith("inference") or mismatch == "empty_inference":
                inference = {
                    k: values[k]
                    for k in ("prompt_embeds", "text_ids", "image_latents", "image_latent_ids")
                }
                if mismatch == "empty_inference" and rank >= 2:
                    inference = {k: [] for k in inference}
                calls = []
                original = adapter._inference
                adapter._inference = lambda **kw: calls.append(kw) or pytest.fail(
                    "native inference entered"
                )
                try:
                    with pytest.raises(ValueError, match="identical native module-call schedules"):
                        adapter.inference(
                            **inference,
                            num_inference_steps=(
                                3 if mismatch == "inference_steps" and rank >= 2 else 2
                            ),
                        )
                    assert calls == []
                finally:
                    adapter._inference = original
            else:
                with pytest.raises(ValueError, match="identical native module-call schedules"):
                    adapter.forward(**values, return_kwargs=["velocity"], noise_level=0.35)
            assert entries == [], (mismatch, entries)
            cases.append(mismatch)

        # Uniform T2I/I2I remain legal, including different reference lengths on DP peers.
        routes = {}
        for conditioned in (False, True):
            values = dict(inputs)
            if not conditioned:
                values.update(image_latents=None, image_latent_ids=None)
            elif rank >= 2:
                values["image_latents"] = values["image_latents"][:, :4]
                values["image_latent_ids"] = values["image_latent_ids"][:, :4]
            entries.clear()
            output = adapter.forward(
                **values, return_kwargs=["velocity"], noise_level=0.35
            ).velocity
            assert torch.isfinite(output).all()
            assert (
                any(name.startswith("mode_blocks.image_cond.") for name in entries) == conditioned
            )
            routes[str(conditioned)] = list(entries)

        if options["mesh"] == "tp":
            # Transport control only: unrelated TP coordinates are not merged into a WORLD check.
            adapter._assert_flux3_fsdp_schedule((rank % 2,), "independent FSDP groups")
            if rank % 2 == 0:
                with pytest.raises(ValueError, match="identical native module-call schedules"):
                    adapter._assert_flux3_fsdp_schedule((rank // 2,), "one mismatched shard group")
            else:
                adapter._assert_flux3_fsdp_schedule((1,), "one matched shard group")
        else:
            # In a hybrid mesh this mismatch starts in only one axis group, then must
            # propagate through both axes before any peer raises.
            with pytest.raises(ValueError, match="identical native module-call schedules"):
                adapter._assert_flux3_fsdp_schedule((int(rank == 0),), "one mismatched axis group")
        adapter._assert_flux3_fsdp_schedule((1, 1), "recovery after rejection")

        # Negative control: disabling only admission reveals the real native branch
        # difference without launching mismatched FSDP collectives in this CPU test.
        original_guard = adapter._assert_flux3_fsdp_schedule
        adapter._assert_flux3_fsdp_schedule = lambda *a, **kw: None
        values = dict(inputs)
        if rank >= 2:
            values.update(image_latents=None, image_latent_ids=None)
        entries.clear()
        try:
            adapter.forward(**values, return_kwargs=["velocity"], noise_level=0.35)
        finally:
            adapter._assert_flux3_fsdp_schedule = original_guard
        assert any(name.startswith("mode_blocks.image_cond.") for name in entries) == (rank < 2)
        result.update(
            status="pass",
            rejected_before_native=cases,
            uniform_native_routes=routes,
            admission_disabled_native_route=list(entries),
            recovery_collective=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("mesh", ["tp", "hybrid", "world"])
def test_fsdp_schedule_guard_rejects_every_peer_before_native_work(tmp_path, mesh):
    rows = run_gloo_ranks(_fsdp_schedule_worker, tmp_path, 4, {"mesh": mesh})
    assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows
    print(json.dumps({"flux3_fsdp_schedule": mesh, "ranks": rows}))
