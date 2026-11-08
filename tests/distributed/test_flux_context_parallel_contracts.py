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

"""CPU contracts for Flux's native CP layout and serialized acceptance fixture."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from flux_coupled_fixture import make_config
from flux_sft_fixture import build_flux_sft_fixture, flux_inputs, flux_lora, flux_transformer
from PIL import Image

from diffusers import FluxPipeline, FluxTransformer2DModel
from flow_factory.models.flux._output import prepare_flux1_output_latents
from flow_factory.models.flux.flux1 import Flux1Adapter


def test_flux_cp_declaration_preserves_tp_and_precision_boundaries():
    capability = Flux1Adapter.context_parallel_capability
    assert asdict(capability) == {
        "layout": "flux_joint_image_text",
        "transport": "ulysses",
        "output_gradient": "local_slice",
        "version": 1,
        "tensor_parallel": True,
        "required_precision_policy": None,
    }
    assert not Flux1Adapter.sequence_parallel_by_default
    assert Flux1Adapter.sequence_parallel_plan()
    assert Flux1Adapter.tensor_parallel_plan()


def test_flux_tp_cp_keeps_sequence_parallel_opt_in_and_kontext_distinct():
    from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
    from flow_factory.models.tensor_parallel import resolve_parallel_plan

    assert (
        Flux1KontextAdapter.context_parallel_capability.layout
        != Flux1Adapter.context_parallel_capability.layout
    )
    assert resolve_parallel_plan(Flux1Adapter, sequence_parallel=True)


@pytest.mark.parametrize(
    "patch,degree,tp,error",
    [
        ({"guidance_embeds": False}, 2, 1, "guidance_embeds=True"),
        ({"num_attention_heads": 3}, 2, 1, "num_attention_heads=3"),
        ({"num_layers": 0}, 2, 1, "dual-stream and single-stream"),
        ({"num_single_layers": 0}, 2, 1, "dual-stream and single-stream"),
        ({"out_channels": 8}, 2, 1, "equal packed"),
        ({"in_channels": 15}, 2, 1, "divisible by four"),
        ({"patch_size": 2}, 2, 1, "patch_size=1"),
        ({"num_attention_heads": 2}, 2, 2, r"cp_size=2 \* tp_size=2"),
        ({"num_attention_heads": 6}, 2, 2, r"num_attention_heads=6"),
    ],
)
def test_flux_metadata_rejects_before_loading(patch, degree, tp, error, monkeypatch):
    config = dict(flux_transformer().config)
    config.update(patch)
    adapter = object.__new__(Flux1Adapter)
    adapter.config = SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=degree,
            effective_tensor_parallel_size=tp,
        )
    )
    adapter.model_args = SimpleNamespace(model_name_or_path="unused")
    monkeypatch.setattr(FluxTransformer2DModel, "load_config", lambda *args, **kwargs: config)
    adapter._load_diffusers_pipeline = lambda *a, **kw: pytest.fail(
        "weights loaded before rejection"
    )
    with pytest.raises(ValueError, match=error):
        adapter.load_pipeline()


def test_flux_module_validation_and_native_plan():
    model = flux_transformer()
    Flux1Adapter.validate_context_parallel_module(model, 2)
    Flux1Adapter.validate_context_parallel_module(model, 4)
    Flux1Adapter.validate_context_parallel_module(model, 2, 2)
    with pytest.raises(ValueError, match="FluxTransformer2DModel"):
        Flux1Adapter.validate_context_parallel_module(torch.nn.Linear(2, 2), 2)
    assert set(model._cp_plan[""]) == {
        "hidden_states",
        "encoder_hidden_states",
        "img_ids",
        "txt_ids",
    }
    assert set(model._cp_plan) == {"", "proj_out"}
    for name, declaration in model._cp_plan[""].items():
        expected = (0, 2) if name.endswith("_ids") else (1, 3)
        assert (declaration.split_dim, declaration.expected_dims) == expected
        assert not declaration.split_output
    assert (model._cp_plan["proj_out"].gather_dim, model._cp_plan["proj_out"].expected_dims) == (
        1,
        3,
    )
    model.single_transformer_blocks = torch.nn.ModuleList()
    with pytest.raises(ValueError, match="dual-stream and single-stream"):
        Flux1Adapter.validate_context_parallel_module(model, 2)


@pytest.fixture
def input_guard(monkeypatch):
    monkeypatch.setattr(
        "flow_factory.models.flux.flux1.context_parallel_geometry", lambda _: (2, 0)
    )
    adapter = object.__new__(Flux1Adapter)
    adapter.accelerator = object()
    adapter.config = SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=False))
    adapter.pipeline = SimpleNamespace(transformer=flux_transformer())
    return adapter, {
        "latents": torch.zeros(1, 16, 16),
        "prompt_embeds": torch.zeros(1, 512, 16),
        "pooled_prompt_embeds": torch.zeros(1, 8),
        "img_ids": torch.zeros(16, 3),
        "joint_attention_kwargs": None,
    }


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("latents", torch.zeros(2, 16, 16), "latents"),
        ("latents", torch.zeros(1, 15, 16), "image token count=15"),
        ("latents", torch.zeros(1, 16, 8), "width=16"),
        ("prompt_embeds", torch.zeros(1, 511, 16), "text token count=511"),
        ("prompt_embeds", torch.zeros(1, 0, 16), "text token count=0"),
        ("prompt_embeds", torch.zeros(1, 512, 8), "width=16"),
        ("pooled_prompt_embeds", torch.zeros(1, 1, 8), "pooled_prompt_embeds"),
        ("img_ids", torch.zeros(1, 16, 3), "img_ids"),
        ("img_ids", torch.zeros(8, 3), "img_ids"),
        ("joint_attention_kwargs", {"ip_adapter_image_embeds": torch.zeros(1)}, "IP adapters"),
    ],
)
def test_flux_inputs_fail_before_transformer_collectives(input_guard, field, value, error):
    adapter, inputs = input_guard
    adapter._validate_context_parallel_inputs(**inputs)
    inputs[field] = value
    with pytest.raises(ValueError, match=error):
        adapter.forward(t=torch.tensor([600.0]), **inputs)


def test_cp_off_preserves_existing_forward_input_policy(input_guard, monkeypatch):
    adapter, inputs = input_guard
    monkeypatch.setattr(
        "flow_factory.models.flux.flux1.context_parallel_geometry", lambda _: (1, 0)
    )
    inputs["latents"] = torch.zeros(2, 15, 16)
    inputs["joint_attention_kwargs"] = {"scale": 0.5}
    adapter._validate_context_parallel_inputs(**inputs)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_flux_native_both_streams_have_gradients(mode):
    model = flux_transformer()
    if mode == "lora":
        model = flux_lora(model)
    output = model(**flux_inputs(0, 0, "cpu"), return_dict=False)[0]
    assert output.shape == (1, 16, 16)
    output.square().mean().backward()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    for stack in ("transformer_blocks.0.", "single_transformer_blocks.0."):
        assert any(stack in name and torch.count_nonzero(p.grad) for name, p in trainable.items())
    if mode == "lora":
        assert all("lora_" in name for name in trainable)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_flux_fixture_roundtrip_and_real_codec(tmp_path):
    build_flux_sft_fixture(tmp_path)
    pipeline = FluxPipeline.from_pretrained(tmp_path / "model", dtype=torch.float32)
    assert len(pipeline.transformer.transformer_blocks) == 1
    assert len(pipeline.transformer.single_transformer_blocks) == 1
    prompt, pooled, text_ids = pipeline.encode_prompt("a red image")
    assert prompt.shape == (1, 512, 16) and pooled.shape == (1, 8) and text_ids.shape == (512, 3)
    records = [
        json.loads(line) for line in (tmp_path / "data/train.jsonl").read_text().splitlines()
    ]
    assert len(records) == 8
    pixels = pipeline.image_processor.preprocess(
        Image.open(tmp_path / "data/0.png"), height=64, width=64
    )
    adapter = SimpleNamespace(pipeline=pipeline, vae=pipeline.vae, device=torch.device("cpu"))
    packed, ids = prepare_flux1_output_latents(adapter, pixels, torch.Generator().manual_seed(17))
    assert packed.shape == (1, 16, 16) and ids.shape == (16, 3)
    unpacked = pipeline._unpack_latents(packed, 64, 64, pipeline.vae_scale_factor)
    assert unpacked.shape == (1, 4, 8, 8)
    decoded = pipeline.vae.decode(unpacked, return_dict=False)[0]
    assert decoded.shape == (1, 3, 64, 64) and torch.isfinite(decoded).all()
    output = pipeline.transformer(
        hidden_states=packed,
        encoder_hidden_states=prompt,
        pooled_projections=pooled,
        timestep=torch.tensor([0.6]),
        guidance=torch.tensor([3.5]),
        img_ids=ids,
        txt_ids=text_ids,
        return_dict=False,
    )[0]
    assert output.shape == packed.shape
    output.square().mean().backward()
    assert all(p.grad is not None for p in pipeline.transformer.parameters())


def test_image_reward_depends_on_decoded_pixels():
    from flux_online_reward import ImageReward

    reward = object.__new__(ImageReward)
    ImageReward.scored, ImageReward.values = [], []
    first, second = torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)
    second[0, :32] = 1
    assert reward([first, second], [7, 8]).rewards.tolist() == [0.0, 1.0]
    assert ImageReward.scored == [7, 8]


def test_flux_rejects_non_native_attention_processor():
    model = flux_transformer()
    model.transformer_blocks[0].attn.set_processor(torch.nn.Identity())
    with pytest.raises(ValueError, match="native FluxAttnProcessor"):
        Flux1Adapter.validate_context_parallel_module(model, 2)


@pytest.mark.parametrize("trainer_type,mode", [("grpo-guard", "full"), ("dppo", "lora")])
@pytest.mark.parametrize("dp_size", [2, 4])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_flux_coupled_recipe_uses_native_conditioning_and_shared_topology(
    monkeypatch, tmp_path, trainer_type, mode, dp_size, tp_size
):
    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", str(2 * dp_size * tp_size))
    config = make_config(
        tmp_path,
        tmp_path,
        {
            "algorithm": trainer_type,
            "training_mode": mode,
            "cp_size": 2,
            "tp_size": tp_size,
            "dp_size": dp_size,
        },
    )
    assert config.model_args.model_type == "flux1"
    assert config.model_args.compute_precision_policy is None
    assert config.training_args.guidance_scale == 3.5
    assert config.training_args.num_inner_epochs == 1
    assert config.training_args.gradient_accumulation_steps == 4
    assert config.parallelism_args.effective_tensor_parallel_size == tp_size
    assert not config.parallelism_args.sequence_parallel
    validate_context_parallel_config(config, Flux1Adapter, get_trainer_class(trainer_type))


def test_flux_media_reward_has_independent_pixel_reference():
    from flux_coupled_fixture import reference_media_score, score_media

    media = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(5))
    assert score_media(media) == pytest.approx(reference_media_score(media), abs=1e-12, rel=0)
    changed = media.clone()
    changed[0, :32] += 0.25
    assert score_media(changed) - score_media(media) == pytest.approx(0.25, abs=1e-8, rel=0)


@pytest.mark.parametrize(
    "node,trainer_type,mode,scheduling,tp_size",
    [
        ("test_cuda_flux_guard_ordered_full", "grpo-guard", "full", "ordered", 1),
        ("test_cuda_flux_dppo_ready_lora", "dppo", "lora", "ready", 1),
        ("test_cuda_flux_tp_cp_guard_ordered_full", "grpo-guard", "full", "ordered", 2),
        ("test_cuda_flux_tp_cp_dppo_ready_lora", "dppo", "lora", "ready", 2),
    ],
)
def test_flux_native_leaves_bind_execution_capability(
    monkeypatch, tmp_path, node, trainer_type, mode, scheduling, tp_size
):
    import sys

    import test_flux_coupled_overlap

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "context_parallel_overlap_runner",
        SimpleNamespace(qualify_overlap=lambda path, options: calls.append((path, options))),
    )
    getattr(test_flux_coupled_overlap, node)(tmp_path)
    assert len(calls) == 1 and calls[0][0] == tmp_path
    options = calls[0][1]
    assert options == {
        "case_module": "flux_coupled_fixture",
        "model_family": "flux",
        "algorithm": trainer_type,
        "training_mode": mode,
        "scheduling_mode": scheduling,
        "dp_size": 2,
        "cp_size": 2,
        "tp_size": tp_size,
    }


@pytest.mark.parametrize(
    "tp_size,reference_ranks", [(1, [0, 0, 1, 1]), (2, [0, 1, 0, 1, 2, 3, 2, 3])]
)
def test_fixed_operand_controls_preserve_logical_dp_and_tp_peer_identity(
    tmp_path, monkeypatch, tp_size, reference_ranks
):
    import context_parallel_model_control as control

    launches, comparisons = [], []

    def run(worker, directory, world, options, **kwargs):
        launches.append((world, options))
        directory.mkdir(parents=True)
        for rank in range(world):
            torch.save({"rank": rank}, directory / f"observations-{rank}.pt")
        return [{"status": "pass"}] * world

    def compare(actual, expected, precision):
        comparisons.append((actual["rank"], expected["rank"], precision))
        return []

    monkeypatch.setattr(control, "run_gloo_ranks", run)
    monkeypatch.setattr(control, "compare_model_controls", compare)
    reports = control.qualify_model_control(
        tmp_path,
        tmp_path,
        {
            "fixture_module": "flux_cp_model_control",
            "training_mode": "full",
            "tp_size": tp_size,
            "cp_size": 2,
            "dp_size": 2,
        },
    )
    assert [world for world, _ in launches] == [2 * tp_size, 4 * tp_size] * 2
    assert all(options["tp_size"] == tp_size for _, options in launches)
    assert comparisons == [
        (rank, reference, precision)
        for precision in ("no", "bf16")
        for rank, reference in enumerate(reference_ranks)
    ]
    assert len(reports) == 2 * len(reference_ranks)


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_fixed_operand_control_checks_adam_and_rejects_wrong_scaling(precision):
    import copy

    from flux_cp_model_control import compare_model_controls
    from test_context_parallel_training import _optimizer_fixture

    expected = _optimizer_fixture([torch.tensor([0.25, -0.5, 0.125])])
    expected.update(
        inputs={"fixed": torch.tensor([1.0])},
        output=torch.tensor([0.3, -0.2]),
        fixed_parameters=expected["initial_parameters"],
    )
    actual = copy.deepcopy(expected)
    actual["unscaled_gradients"] = {
        name: value / 2 for name, value in expected["updates"][0]["gradients"].items()
    }
    compare_model_controls(actual, expected, precision)
    corrupted = copy.deepcopy(actual)
    corrupted["updates"][0]["gradients"]["p"] *= 0.5
    with pytest.raises(AssertionError):
        compare_model_controls(corrupted, expected, precision)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_packed_sft_reduction_uses_existing_full_output_backward(monkeypatch, cp_size):
    from flow_factory.samples import LatentState, NoisedState
    from flow_factory.trainers.common import context_parallel
    from flow_factory.trainers.common.flow_matching import flow_matching_per_sample_loss
    from flow_factory.trainers.offline.sft import SFTTrainer

    prediction = torch.nn.Parameter(torch.linspace(-0.4, 0.8, 128).reshape(2, 4, 16))
    target = torch.linspace(0.9, -0.3, 128).reshape_as(prediction)
    state = LatentState({"latent": prediction})
    noised = NoisedState(
        state=state,
        target_velocity=LatentState({"latent": target}),
        noise=LatentState({"latent": torch.zeros_like(target)}),
    )
    adapter = object.__new__(Flux1Adapter)
    trainer = object.__new__(SFTTrainer)
    trainer.accelerator = SimpleNamespace(backward=lambda loss: (loss / 2).backward())
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    per_sample = flow_matching_per_sample_loss(adapter, state, noised)
    expected = (prediction.detach().double() - target.double()).square().flatten(1).mean(1)
    assert per_sample.shape == (2,)
    torch.testing.assert_close(per_sample.double(), expected, rtol=2e-7, atol=1e-8)
    trainer._backward_full_output_loss(per_sample.mean())
    gradient = 2 * (prediction.detach() - target) / prediction.numel() * cp_size / 2
    assert prediction.grad.shape == (2, 4, 16)
    torch.testing.assert_close(prediction.grad, gradient, rtol=2e-7, atol=1e-8)
    for factor in (0.0, 0.5, 2.0, -1.0):
        assert not torch.allclose(prediction.grad * factor, gradient, rtol=2e-7, atol=1e-8)


def test_model_control_clipping_preserves_prepared_root_identity_and_order():
    from types import MethodType

    from accelerate import Accelerator
    from accelerate.utils import DistributedType
    from flux_cp_model_control import clip_prepared_model

    model = torch.nn.Module()
    model.register_parameter("z_matrix", torch.nn.Parameter(torch.ones(2, 3)))
    model.register_parameter("m_frozen", torch.nn.Parameter(torch.ones(5), requires_grad=False))
    model.register_parameter("a_vector", torch.nn.Parameter(torch.ones(2)))
    model.z_matrix.grad = torch.ones_like(model.z_matrix)
    model.a_vector.grad = torch.ones_like(model.a_vector)
    expected = list(model.parameters())
    calls = []

    def clip(parameters, max_norm, norm_type):
        assert len(parameters) == len(expected)
        assert all(left is right for left, right in zip(parameters, expected))
        calls.append(tuple(id(parameter) for parameter in parameters))
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        is_fsdp2=True,
        _models=[model],
        unscale_gradients=lambda: None,
        _clip_grad_norm_dtensor_aware=clip,
    )
    # Execute Accelerate's real prepared-model list matching on CPU. The sorted,
    # map used by the failed full probe reaches tensor equality on differently
    # shaped parameters. Filtering frozen entries also loses prepared-root identity.
    accelerator.clip_grad_norm_ = MethodType(Accelerator.clip_grad_norm_, accelerator)
    wrong = [parameter for _, parameter in sorted(model.named_parameters())]
    with pytest.raises(RuntimeError):
        accelerator.clip_grad_norm_(wrong, 1.0)
    with pytest.raises(AssertionError):
        accelerator.clip_grad_norm_(
            [parameter for parameter in expected if parameter.requires_grad], 1.0
        )
    norm = clip_prepared_model(accelerator, model)
    torch.testing.assert_close(norm, torch.tensor(8.0).sqrt())
    assert calls == [tuple(id(parameter) for parameter in expected)]
    assert model.m_frozen.grad is None


def test_flux_cp_accepts_odd_text_only_with_explicit_sp_padding(input_guard):
    adapter, inputs = input_guard
    inputs["prompt_embeds"] = inputs["prompt_embeds"][:, :5]
    with pytest.raises(ValueError, match="text token count=5"):
        adapter._validate_context_parallel_inputs(**inputs)
    adapter.config.parallelism_args.sequence_parallel = True
    adapter._validate_context_parallel_inputs(**inputs)
