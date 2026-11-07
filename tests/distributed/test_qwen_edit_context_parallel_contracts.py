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

"""Edit-specific native VL/reference encoding, CFG gradients and CP rejection."""

import copy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from qwen_edit_cp_fixture import (
    build_pipeline,
    build_transformer,
    inactive_parameter_names,
    install_lora,
    model_inputs,
)

from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.models.runtime.classic import ClassicPipelineRuntime
from flow_factory.scheduler import FlowMatchEulerDiscreteSDESchedulerOutput

OWNER = "flow_factory.models.qwen_image.qwen_image_edit_plus"


def _adapter(model=None, pipeline=None):
    adapter = object.__new__(QwenImageEditPlusAdapter)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.config = SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=True))
    adapter.training_args = SimpleNamespace(height=64, width=64)
    adapter.pipeline = pipeline or SimpleNamespace(
        transformer=model, components={"transformer": model}
    )
    adapter.component_runtime = ClassicPipelineRuntime(adapter.pipeline)
    if pipeline is None:
        adapter.scheduler = SimpleNamespace(
            step=lambda **kwargs: FlowMatchEulerDiscreteSDESchedulerOutput(
                velocity=kwargs["velocity"]
            )
        )
    return adapter


def _forward_inputs():
    native = model_inputs(0, 0, "cpu")
    return dict(
        t=torch.tensor([600.0]),
        latents=native["hidden_states"][:1, :8],
        image_latents=native["hidden_states"][:1, 8:],
        img_shapes=native["img_shapes"][:1],
        prompt_embeds=native["encoder_hidden_states"][:1, :7],
        prompt_embeds_mask=torch.tensor([[1, 0, 1, 1, 0, 1, 1]]),
        negative_prompt_embeds=native["encoder_hidden_states"][1:, :5],
        negative_prompt_embeds_mask=torch.tensor([[1, 1, 0, 1, 1]]),
    )


@pytest.mark.parametrize("zero_cond_t", [False, True], ids=["2509", "2511"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("cfg", [False, True])
def test_native_edit_cfg_target_crop_and_gradients_match_independent_passes(
    monkeypatch, zero_cond_t, mode, cfg
):
    native = build_transformer(zero_cond_t)
    if mode == "lora":
        native = install_lora(native)
    inputs = _forward_inputs()
    scale = 2.0 if cfg else 1.0
    hidden = torch.cat([inputs["latents"], inputs["image_latents"]], dim=1)

    def reference_pass(prompt, mask):
        return native(
            hidden_states=hidden,
            timestep=inputs["t"] / 1000,
            encoder_hidden_states=prompt,
            encoder_hidden_states_mask=mask,
            img_shapes=inputs["img_shapes"],
            return_dict=False,
        )[0][:, :8]

    conditional = reference_pass(inputs["prompt_embeds"], inputs["prompt_embeds_mask"])
    expected = conditional
    if cfg:
        negative = reference_pass(
            inputs["negative_prompt_embeds"], inputs["negative_prompt_embeds_mask"]
        )
        combined = negative + scale * (conditional - negative)
        expected = combined * (
            conditional.norm(dim=-1, keepdim=True) / combined.norm(dim=-1, keepdim=True)
        )
    expected.square().mean().backward()
    inactive = inactive_parameter_names(native)
    for cp_size in (1, 2):
        model = copy.deepcopy(native)
        model.zero_grad(set_to_none=True)
        monkeypatch.setattr(f"{OWNER}.context_parallel_geometry", lambda _: (cp_size, 0))
        monkeypatch.setattr(f"{OWNER}.tensor_parallel_geometry", lambda _: (2, 0))
        adapter = _adapter(model)
        seen = []
        hook = model.register_forward_pre_hook(lambda m, a, kw: seen.append(kw), with_kwargs=True)
        actual = adapter.forward(**inputs, guidance_scale=scale).velocity
        hook.remove()
        assert actual.shape == (1, 8, 16)
        assert seen[0]["hidden_states"].shape[1] == 24
        assert seen[0]["encoder_hidden_states"].shape[1] == (7 if cp_size == 1 else 8)
        assert seen[0]["encoder_hidden_states_mask"][0, :7].tolist() == [1, 0, 1, 1, 0, 1, 1]
        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=3e-6)
        actual.square().mean().backward()
        missing = {
            name
            for name, value in model.named_parameters()
            if value.requires_grad and value.grad is None
        }
        assert missing == inactive
        zero_gradients = set()
        for (name, left), (_, right) in zip(native.named_parameters(), model.named_parameters()):
            if left.grad is None:
                assert right.grad is None
            else:
                if not torch.count_nonzero(left.grad):
                    # Final text queries have no path to the cropped image output,
                    # but native concatenation propagates explicit zero gradients.
                    assert any(
                        prefix in name
                        for prefix in (
                            "transformer_blocks.1.attn.add_q_proj.",
                            "transformer_blocks.1.attn.norm_added_q.",
                        )
                    ), name
                    zero_gradients.add(name)
                torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=3e-6, msg=name)
        assert len(zero_gradients) == (3 if mode == "full" else 2)


@pytest.mark.parametrize("zero_cond_t", [False, True], ids=["2509", "2511"])
def test_native_edit_reference_order_and_zero_timestep_metadata(zero_cond_t):
    model = build_transformer(zero_cond_t)
    inputs = model_inputs(0, 0, "cpu")
    observed = []
    hook = model.transformer_blocks[0].register_forward_pre_hook(
        lambda m, a, kw: observed.append(kw), with_kwargs=True
    )
    with torch.no_grad():
        output = model(**inputs, return_dict=False)[0][:, :8]
    hook.remove()
    index = observed[0]["modulate_index"]
    if zero_cond_t:
        assert index.tolist() == [[0] * 8 + [1] * 16] * 2
        assert observed[0]["temb"].shape[0] == 4
    else:
        assert index is None and observed[0]["temb"].shape[0] == 2
    with torch.no_grad():
        no_reference = inputs["hidden_states"].clone()
        no_reference[:, 8:] = 0
        ablated = model(**{**inputs, "hidden_states": no_reference}, return_dict=False)[0][:, :8]
        swapped = torch.cat(
            [
                inputs["hidden_states"][:, :8],
                inputs["hidden_states"][:, 14:],
                inputs["hidden_states"][:, 8:14],
            ],
            dim=1,
        )
        shapes = [[row[0], row[2], row[1]] for row in inputs["img_shapes"]]
        reordered = model(
            **{**inputs, "hidden_states": swapped, "img_shapes": shapes}, return_dict=False
        )[0][:, :8]
    assert not torch.allclose(output, ablated)
    assert not torch.allclose(output, reordered)


def test_native_vl_reference_encoding_and_saved_edit_pipeline_roundtrip(tmp_path):
    pipeline = build_pipeline()
    adapter = _adapter(pipeline=pipeline)
    images = [Image.new("RGB", (64, 64), (200, 20, 40)), Image.new("RGB", (64, 64), (30, 160, 80))]
    # Match the native pipeline's condition resize before comparing its encoder;
    # feeding the original pixels would exercise a different VL patch grid.
    native_conditions = [pipeline.image_processor.resize(image, 384, 384) for image in images]
    with torch.no_grad():
        encoded = adapter.preprocess_func(
            prompt=["make picture red"],
            images=[images],
            negative_prompt=["make picture blue"],
            condition_image_size=64,
            max_sequence_length=512,
        )
        expected, mask = pipeline._get_qwen_prompt_embeds(
            prompt=["make picture red"], image=native_conditions, device=torch.device("cpu")
        )
    torch.testing.assert_close(encoded["prompt_embeds"][0], expected[0], rtol=0, atol=0)
    assert torch.equal(encoded["prompt_embeds_mask"][0], mask[0])
    assert pipeline.prompt_template_encode_start_idx == 64
    assert expected.shape[1] > 64 and expected.shape[1] <= 512
    assert encoded["prompt_ids"][0].eq(pipeline.text_encoder.config.image_token_id).sum() == 32
    assert encoded["condition_image_sizes"] == [[(384, 384), (384, 384)]]
    assert encoded["vae_image_sizes"] == [[(64, 64), (64, 64)]]
    assert encoded["image_latents"][0].shape == (1, 32, 16)
    reference_parts = []
    with torch.no_grad():
        for image in encoded["vae_images"][0]:
            latent = pipeline.vae.encode(image).latent_dist.mode()
            reference_parts.append(pipeline._pack_latents(latent, 1, 4, 8, 8))
        swapped = adapter.preprocess_func(
            prompt=["make picture red"],
            images=[list(reversed(images))],
            condition_image_size=64,
            max_sequence_length=512,
        )
    torch.testing.assert_close(
        encoded["image_latents"][0], torch.cat(reference_parts, dim=1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        swapped["image_latents"][0],
        torch.cat(list(reversed(reference_parts)), dim=1),
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(encoded["prompt_embeds"][0], swapped["prompt_embeds"][0])
    common = min(
        encoded["prompt_embeds"][0].shape[0], encoded["negative_prompt_embeds"][0].shape[0]
    )
    assert not torch.allclose(
        encoded["prompt_embeds"][0][:common], encoded["negative_prompt_embeds"][0][:common]
    )
    pipeline.save_pretrained(tmp_path / "model")
    restored = QwenImageEditPlusPipeline.from_pretrained(tmp_path / "model", local_files_only=True)
    with torch.no_grad():
        actual, actual_mask = restored._get_qwen_prompt_embeds(
            prompt=["make picture red"], image=native_conditions, device=torch.device("cpu")
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_mask, mask)
    assert restored.transformer.config.zero_cond_t is True


@pytest.mark.parametrize("zero_cond_t", [False, True])
def test_edit_owns_native_cp_layout_for_both_checkpoint_variants(zero_cond_t):
    capability = QwenImageEditPlusAdapter.context_parallel_capability
    assert capability.layout == "qwen_edit_ordered_reference_image_text"
    assert capability.tensor_parallel
    assert type(capability.layout).__module__ == OWNER
    QwenImageEditPlusAdapter.validate_context_parallel_module(build_transformer(zero_cond_t), 2, 2)


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"num_attention_heads": 3}, "num_attention_heads"),
        ({"num_layers": 0}, "native transformer blocks"),
        ({"guidance_embeds": True}, "guidance_embeds=False"),
        ({"zero_cond_t": "true"}, "boolean"),
        ({"use_additional_t_cond": True}, "use_additional_t_cond=False"),
        ({"use_layer3d_rope": True}, "use_layer3d_rope=False"),
        ({"axes_dims_rope": [2, 2, 2]}, "rotary axes"),
        ({"out_channels": 8}, "packed input/output"),
    ],
)
def test_edit_layout_rejection_precedes_weight_loading(monkeypatch, patch, error):
    config = {**dict(build_transformer().config), **patch}
    adapter = object.__new__(QwenImageEditPlusAdapter)
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=2,
            effective_tensor_parallel_size=2,
        )
    )
    monkeypatch.setattr(QwenImageTransformer2DModel, "load_config", lambda *a, **kw: config)
    adapter._load_diffusers_pipeline = lambda *a, **kw: pytest.fail("loaded before guard")
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("image_latents", None, "image_latents"),
        ("image_latents", torch.zeros(1, 16, 8), "image_latents"),
        ("img_shapes", [[(1, 2, 4)]], "at least one reference"),
        ("img_shapes", [[(1, 2, 3), (1, 2, 3), (1, 2, 5)]], "ordered shapes"),
        ("img_shapes", [[(2, 1, 4), (1, 2, 3), (1, 2, 5)]], "target-first"),
        ("prompt_embeds_mask", torch.ones(1, 3), "original embeddings"),
        ("attention_kwargs", {"scale": 1.0}, "attention_kwargs"),
    ],
)
def test_edit_invalid_operands_fail_before_native_forward(monkeypatch, field, value, error):
    monkeypatch.setattr(f"{OWNER}.context_parallel_geometry", lambda _: (2, 0))
    monkeypatch.setattr(f"{OWNER}.tensor_parallel_geometry", lambda _: (2, 0))
    model = build_transformer()
    model.forward = lambda *a, **kw: pytest.fail("native forward reached before guard")
    adapter = _adapter(model)
    args = {**_forward_inputs(), field: value}
    with pytest.raises(ValueError, match=error):
        adapter._forward(**args, guidance_scale=2.0)


def test_edit_cp_sp_token_guard_checks_concatenated_reference_sequence(monkeypatch):
    monkeypatch.setattr(f"{OWNER}.context_parallel_geometry", lambda _: (2, 0))
    monkeypatch.setattr(f"{OWNER}.tensor_parallel_geometry", lambda _: (2, 0))
    model = build_transformer()
    model.forward = lambda *a, **kw: pytest.fail("native forward reached before guard")
    adapter = _adapter(model)
    args = _forward_inputs()
    args.update(
        image_latents=torch.zeros(1, 18, 16), img_shapes=[[(1, 2, 4), (1, 2, 3), (1, 2, 6)]]
    )
    with pytest.raises(ValueError, match="tokens=26, cp_size=2, sp_size=2"):
        adapter._forward(**args, guidance_scale=2.0)


def test_edit_lifecycle_recipe_keeps_real_reference_records_and_encoding_options(
    monkeypatch, tmp_path
):
    import json

    import qwen_edit_coupled_fixture as fixture
    from context_parallel_overlap_runner import make_overlap_config, write_overlap_datasets

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = dict(
        algorithm="grpo-guard",
        dp_size=2,
        cp_size=2,
        tp_size=2,
        training_mode="full",
        scheduling_mode="ordered",
    )
    base = fixture.make_config(tmp_path, tmp_path, options)
    config = make_overlap_config(base, tmp_path, tmp_path, options)
    assert config.model_args.model_type == "qwen-image-edit-plus"
    assert config.parallelism_args.sequence_parallel is True
    for arguments in (config.training_args, config.eval_args):
        assert arguments.extra_kwargs["condition_image_size"] == 64
        assert arguments.extra_kwargs["max_sequence_length"] == 512
        assert arguments.extra_kwargs["negative_prompt"] == "blue"
    write_overlap_datasets(tmp_path, fixture)
    for relative in ("tiny/train.jsonl", "odd/test.jsonl"):
        rows = [json.loads(line) for line in (tmp_path / relative).read_text().splitlines()]
        assert rows
        assert all(
            row["images"] == [str(tmp_path / f"reference-{index}.png") for index in range(2)]
            for row in rows
        )


@pytest.mark.parametrize("corruption", ["processor", "rotary", "module_time", "block_time"])
def test_edit_realized_module_rejects_unqualified_native_layout(corruption):
    model = build_transformer(True)
    if corruption == "processor":
        model.transformer_blocks[0].attn.processor = object()
        error = "native QwenDoubleStreamAttnProcessor2_0"
    elif corruption == "rotary":
        model.pos_embed = torch.nn.Identity()
        error = "native ordered rotary"
    elif corruption == "module_time":
        model.zero_cond_t = False
        error = "zero_cond_t must match every native block"
    else:
        model.transformer_blocks[-1].zero_cond_t = False
        error = "zero_cond_t must match every native block"
    with pytest.raises(ValueError, match=error):
        QwenImageEditPlusAdapter.validate_context_parallel_module(model, 2, 2)


def test_edit_text_token_guard_reports_concrete_divisibility(monkeypatch):
    monkeypatch.setattr(f"{OWNER}.context_parallel_geometry", lambda _: (2, 0))
    adapter = _adapter(build_transformer())
    inputs = _forward_inputs()
    with pytest.raises(
        ValueError, match="must be divisible by cp_size; received text_tokens=3, cp_size=2"
    ):
        adapter._validate_context_parallel_inputs(
            inputs["latents"],
            inputs["image_latents"],
            inputs["prompt_embeds"][:, :3],
            inputs["prompt_embeds_mask"][:, :3],
            inputs["img_shapes"],
            None,
        )


def _cached_edit_inference_inputs(zero_cond_t=True):
    from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler

    pipeline = build_pipeline(zero_cond_t)
    adapter = _adapter(pipeline=pipeline)
    adapter.scheduler = FlowMatchEulerDiscreteSDEScheduler(noise_level=0.0, dynamics_type="ODE")
    adapter._warned_cfg_no_neg_prompt = False
    adapter._warned_no_cfg = False
    adapter._has_warned_inference_auto_resize = False
    images = [Image.new("RGB", (64, 64), (200, 20, 40)), Image.new("RGB", (64, 64), (30, 160, 80))]
    with torch.no_grad():
        encoded = adapter.preprocess_func(
            prompt=["make picture red"],
            images=[images],
            negative_prompt=["make picture blue"],
            condition_image_size=64,
            max_sequence_length=512,
        )
    return adapter, {
        **encoded,
        "images": [images],
        "prompt": ["make picture red"],
        "negative_prompt": ["make picture blue"],
        "height": 64,
        "width": 64,
        "num_inference_steps": 2,
        "guidance_scale": 2.0,
        "auto_resize": False,
    }


@pytest.mark.parametrize("zero_cond_t", [False, True], ids=["2509", "2511"])
@pytest.mark.parametrize("cp_size", [1, 2], ids=["cp1", "cp2"])
def test_cached_edit_inference_normalizes_collated_integer_geometry(
    monkeypatch, zero_cond_t, cp_size
):
    from datasets import Dataset

    from flow_factory.data_utils.dataset import GeneralDataset

    monkeypatch.setattr(f"{OWNER}.context_parallel_geometry", lambda _: (cp_size, 0))
    monkeypatch.setattr(f"{OWNER}.tensor_parallel_geometry", lambda _: (2, 0))
    adapter, inputs = _cached_edit_inference_inputs(zero_cond_t)
    dataset = Dataset.from_dict({"vae_image_sizes": inputs["vae_image_sizes"]}).with_format("torch")
    collated = GeneralDataset.collate_fn([dataset[0]])
    assert collated["vae_image_sizes"].shape == (1, 2, 2)
    assert collated["vae_image_sizes"].dtype == torch.int64
    expected = adapter.inference(**inputs, generator=torch.Generator().manual_seed(71))[0]
    actual = adapter.inference(
        **{**inputs, **collated}, generator=torch.Generator().manual_seed(71)
    )[0]
    assert actual.img_shapes == expected.img_shapes == [(1, 4, 4)] * 3
    assert all(type(value) is int for shape in actual.img_shapes for value in shape)
    torch.testing.assert_close(actual.all_latents, expected.all_latents, rtol=0, atol=0)
    torch.testing.assert_close(actual.image, expected.image, rtol=0, atol=0)
    torch.testing.assert_close(actual.image_latents, expected.image_latents, rtol=0, atol=0)


@pytest.mark.parametrize(
    "geometry,error",
    [
        (torch.full((1, 2, 2), 64.0), "expected integer geometry tensor"),
        (torch.ones((1, 2, 2), dtype=torch.bool), "expected integer geometry tensor"),
        (torch.full((1, 2, 2, 1), 64, dtype=torch.int64), "expected rank 2"),
        (torch.tensor([[[0, 64], [64, 64]]]), "expected positive integer geometry"),
    ],
    ids=["float", "bool", "non-scalar-dimension", "nonpositive"],
)
def test_cached_edit_geometry_rejects_invalid_metadata_before_latent_preparation(
    monkeypatch, geometry, error
):
    monkeypatch.setattr(f"{OWNER}.context_parallel_geometry", lambda _: (2, 0))
    adapter, inputs = _cached_edit_inference_inputs()
    adapter.prepare_latents = lambda **kwargs: pytest.fail(
        "invalid geometry reached latent preparation"
    )
    with pytest.raises((TypeError, ValueError), match=error):
        adapter.inference(**{**inputs, "vae_image_sizes": geometry})


@pytest.mark.parametrize("zero_cond_t", [False, True], ids=["2509", "2511"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("fields", ["all", "velocity"])
def test_native_edit_singleton_preserves_guard_graph_and_parameter_gradients(
    monkeypatch, zero_cond_t, mode, fields
):
    from coupled_objective_reference import assert_objective

    from flow_factory.samples import ComponentTimes, LatentState, MultiModalStepOutput, ReplayStep
    from flow_factory.scheduler import (
        FlowMatchEulerDiscreteSDEScheduler,
        SchedulerGroup,
        set_scheduler_timesteps,
    )
    from flow_factory.trainers.rl.grpo import GRPOGuardTrainer

    monkeypatch.setattr(f"{OWNER}.context_parallel_geometry", lambda _: (2, 0))
    monkeypatch.setattr(f"{OWNER}.tensor_parallel_geometry", lambda _: (2, 0))
    model = build_transformer(zero_cond_t)
    if mode == "lora":
        model = install_lora(model)
    adapter = _adapter(model)
    adapter.scheduler = FlowMatchEulerDiscreteSDEScheduler(noise_level=0.35)
    adapter.scheduler_group = SchedulerGroup({"latent": adapter.scheduler}, primary_name="latent")
    set_scheduler_timesteps(adapter.scheduler, num_inference_steps=3, seq_len=8, device="cpu")
    inputs = _forward_inputs()
    inputs.update(
        t_next=torch.tensor([400.0]),
        next_latents=inputs["latents"] + 0.137,
        noise_level=0.35,
        guidance_scale=2.0,
    )
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
