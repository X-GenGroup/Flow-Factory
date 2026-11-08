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

"""Native Flux continuous image/text SP boundaries and numerical controls."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from flux_sft_fixture import flux_inputs, flux_lora, flux_transformer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.tensor_parallel import (
    apply_tensor_parallel,
    peel_peft_wrapper,
    resolve_parallel_plan,
)


def _assert_added_placements(native):
    """Check physical base and LoRA shards, not merely plan pattern membership."""
    paths = {
        "transformer_blocks.0.norm1.linear": 1,
        "transformer_blocks.0.norm1_context.linear": 1,
        "single_transformer_blocks.0.norm.linear": 1,
        "norm_out.linear": 1,
        "single_transformer_blocks.0.attn.to_q": 0,
        "single_transformer_blocks.0.attn.to_k": 0,
        "single_transformer_blocks.0.attn.to_v": 0,
        "single_transformer_blocks.0.proj_mlp": 0,
        "single_transformer_blocks.0.proj_out": 1,
    }
    observed = {}
    for path, dimension in paths.items():
        module = native.get_submodule(path)
        base = getattr(module, "base_layer", module)
        weight = base.weight
        assert isinstance(weight, DTensor) and weight.placements == (Shard(dimension),), path
        shape = list(weight.shape)
        shape[dimension] //= 2
        assert list(weight.to_local().shape) == shape, path
        if base.bias is not None:
            assert base.bias.placements == ((Shard(0),) if dimension == 0 else (Replicate(),)), path
        if hasattr(module, "lora_A"):
            a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
            assert a.placements == ((Shard(1),) if dimension == 1 else (Replicate(),)), path
            assert b.placements == ((Shard(0),) if dimension == 0 else (Replicate(),)), path
        observed[path] = {"global": list(weight.shape), "local": shape, "shard_dim": dimension}
    return observed


def _worker(rank, store_path, result_dir, options):
    cp_size = options.get("cp_size", 1)
    world_size = 2 * cp_size
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, world_size), rank=rank, world_size=world_size
    )
    result = {"rank": rank, **options}
    try:
        if cp_size > 1:
            from torch.distributed.device_mesh import DeviceMesh

            mesh = DeviceMesh(
                "cpu",
                torch.arange(world_size).reshape(cp_size, 2).T.unsqueeze(1),
                mesh_dim_names=("tp", "ring", "ulysses"),
            )
            tp_mesh = mesh["tp"]
        else:
            mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
            tp_mesh = mesh
        model = flux_transformer(num_layers=2, num_single_layers=2).double()
        if options["mode"] == "lora":
            model = flux_lora(model).double()
        if options["mode"] == "expanded_lora":
            from peft import LoraConfig, get_peft_model

            # Exercise new packed and modulation factors without widening public defaults.
            model = get_peft_model(
                model,
                LoraConfig(
                    r=2,
                    lora_alpha=2,
                    target_modules=[
                        name
                        for name, child in model.named_modules()
                        if isinstance(child, torch.nn.Linear)
                        and (
                            name == "proj_out"
                            or name.startswith(
                                ("transformer_blocks.", "single_transformer_blocks.", "norm_out.")
                            )
                        )
                    ],
                    init_lora_weights=False,
                ),
            ).double()
        model.set_attention_backend("native")
        reference = copy.deepcopy(model)
        if options["checkpointing"]:
            model.enable_gradient_checkpointing()
            reference.enable_gradient_checkpointing()
        apply_tensor_parallel(
            peel_peft_wrapper(model),
            tp_mesh,
            resolve_parallel_plan(Flux1Adapter, sequence_parallel=options.get("sp", True)),
        )
        observed = []
        native = peel_peft_wrapper(model)
        if cp_size > 1:
            from unittest.mock import patch

            from diffusers.models._modeling_parallel import ContextParallelConfig

            # Released Diffusers probes accelerator discovery even for an explicit
            # CPU mesh. Keep its hooks and collectives intact in this Gloo control.
            with (
                patch("torch._C._get_accelerator", return_value=torch.device("cpu")),
                patch("torch.accelerator.device_count", return_value=1),
            ):
                native.enable_parallelism(
                    config=ContextParallelConfig(ring_degree=1, ulysses_degree=cp_size, mesh=mesh)
                )
            boundaries = Flux1Adapter.compose_context_parallel_sequence_boundaries(native)
            assert len(boundaries) == 1 and boundaries[0].module is native.proj_out
        placements = _assert_added_placements(native)

        def observe(name):
            return lambda _module, args, kwargs: observed.append((name, args[0].shape[1]))

        for index, block in enumerate(native.transformer_blocks):
            block.norm1.register_forward_pre_hook(observe(f"dual_image_{index}"), with_kwargs=True)
            block.norm1_context.register_forward_pre_hook(
                observe(f"dual_text_{index}"), with_kwargs=True
            )
        for index, block in enumerate(native.single_transformer_blocks):
            block.norm.register_forward_pre_hook(observe(f"single_joint_{index}"), with_kwargs=True)
        native.norm_out.register_forward_pre_hook(observe("final_image"), with_kwargs=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        for step in range(3):
            inputs = {name: value.double() for name, value in flux_inputs(0, step, "cpu").items()}
            text_tokens = options.get("text_tokens", 8)
            inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"][
                :, :text_tokens
            ].clone()
            inputs["txt_ids"] = inputs["txt_ids"][:text_tokens].clone()
            differentiable = ("hidden_states", "encoder_hidden_states", "pooled_projections")
            for name in differentiable:
                inputs[name].requires_grad_(True)
            reference_inputs = {
                name: value.detach().clone().requires_grad_(value.requires_grad)
                for name, value in inputs.items()
            }
            if "lora_scale" in options:
                inputs["joint_attention_kwargs"] = {"scale": options["lora_scale"]}
                reference_inputs["joint_attention_kwargs"] = {"scale": options["lora_scale"]}
            expected = reference(**reference_inputs, return_dict=False)[0]
            actual = model(**inputs, return_dict=False)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
            expected.square().mean().backward()
            actual.square().mean().backward()
            for name in differentiable:
                if cp_size > 1:
                    dist.all_reduce(inputs[name].grad, group=mesh["ulysses"].get_group())
                torch.testing.assert_close(
                    inputs[name].grad,
                    reference_inputs[name].grad,
                    rtol=2e-7,
                    atol=1e-9,
                    msg=f"input derivative: {name}",
                )
            names = dict(reference.named_parameters())
            actual_parameters = dict(model.named_parameters())
            assert names.keys() == actual_parameters.keys()
            active = 0
            for name, target in names.items():
                parameter = actual_parameters[name]
                assert (target.grad is None) == (parameter.grad is None), name
                if target.grad is not None:
                    if cp_size > 1:
                        dist.all_reduce(
                            parameter.grad.to_local(), group=mesh["ulysses"].get_group()
                        )
                    torch.testing.assert_close(
                        parameter.grad.full_tensor(), target.grad, rtol=2e-7, atol=1e-9, msg=name
                    )
                    active += 1
            assert active > 0
            optimizer.step()
            reference_optimizer.step()
            for name, target in names.items():
                parameter = actual_parameters[name]
                torch.testing.assert_close(
                    parameter.full_tensor(), target, rtol=2e-7, atol=1e-8, msg=name
                )
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in reference_optimizer.state[target]:
                        torch.testing.assert_close(
                            optimizer.state[parameter][key].full_tensor(),
                            reference_optimizer.state[target][key],
                            rtol=2e-7,
                            atol=1e-9,
                            msg=name,
                        )
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        sp = options.get("sp", True)
        image_tokens = (8 if sp else 16) // cp_size
        degree = cp_size * (2 if sp else 1)
        local_text = (text_tokens + degree - 1) // degree
        expected_shapes = {("final_image", image_tokens)}
        for index in range(2):
            expected_shapes.update(
                {
                    (f"dual_image_{index}", image_tokens),
                    (f"dual_text_{index}", local_text),
                    (f"single_joint_{index}", image_tokens + local_text),
                }
            )
        assert set(observed) == expected_shapes, observed
        # Ordinary full state preserves native 2-D packed order and all PEFT names.
        canonical = {name: value.full_tensor() for name, value in model.state_dict().items()}
        assert canonical.keys() == reference.state_dict().keys()
        for name, value in reference.state_dict().items():
            assert canonical[name].shape == value.shape, name
        checkpoint = Path(result_dir, f"canonical-{rank}.pt")
        torch.save(canonical, checkpoint)
        restored = copy.deepcopy(reference)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(
            restored(**inputs, return_dict=False)[0],
            model(**inputs, return_dict=False)[0],
            rtol=1e-8,
            atol=1e-9,
        )
        result.update(
            status="pass",
            active_parameters=active,
            updates=3,
            observed=observed,
            placements=placements,
            canonical_reload=True,
            input_derivatives=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_flux_sequence_parallel_gradients(tmp_path, mode, checkpointing):
    results = run_gloo_ranks(_worker, tmp_path, 2, {"mode": mode, "checkpointing": checkpointing})
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("mode", ["full", "expanded_lora"])
def test_native_flux_complete_tp_placements_derivatives_and_state(tmp_path, mode, sp):
    results = run_gloo_ranks(_worker, tmp_path, 2, {"mode": mode, "checkpointing": True, "sp": sp})
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    assert all(row["canonical_reload"] and len(row["placements"]) == 9 for row in results)


@pytest.mark.parametrize("mode", ["full", "lora", "expanded_lora"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_flux_complete_sp_masks_odd_text_without_changing_gradients(
    tmp_path, mode, checkpointing
):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        {"mode": mode, "checkpointing": checkpointing, "sp": True, "text_tokens": 5},
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


@pytest.mark.parametrize("mode", ["full", "expanded_lora"])
def test_native_flux_complete_sp_cp_masks_odd_text_gradients_and_state(tmp_path, mode):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        4,
        {"mode": mode, "checkpointing": True, "sp": True, "text_tokens": 5, "cp_size": 2},
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


def test_kontext_reuses_only_the_same_transformer_sequence_region():
    expected = Flux1Adapter.sequence_parallel_plan()
    actual = Flux1KontextAdapter.sequence_parallel_plan()
    assert expected.keys() == actual.keys()
    assert all(type(expected[key]) is type(actual[key]) for key in expected)
    assert not Flux1Adapter.sequence_parallel_by_default
    assert not Flux1KontextAdapter.sequence_parallel_by_default


def _cp_boundaries():
    from types import SimpleNamespace

    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.models._modeling_parallel import ContextParallelOutput

    module = flux_transformer()
    split = ContextParallelSplitHook(copy.deepcopy(module._cp_plan[""]), None)
    gather = ContextParallelGatherHook([ContextParallelOutput(gather_dim=1, expected_dims=3)], None)
    root_hooks = {"cp_input---": split}
    output_hooks = {"cp_output---proj_out": gather}
    module._diffusers_hook = SimpleNamespace(get_hook=root_hooks.get)
    module.proj_out._diffusers_hook = SimpleNamespace(
        get_hook=output_hooks.get,
        remove_hook=lambda name, recurse=False: output_hooks.pop(name),
    )
    return module, split, gather


@pytest.mark.parametrize("adapter", [Flux1Adapter, Flux1KontextAdapter])
def test_native_cp_output_gather_moves_after_flux_sp(adapter):
    module, split, gather = _cp_boundaries()
    boundaries = adapter.compose_context_parallel_sequence_boundaries(module)
    assert len(boundaries) == 1
    assert boundaries[0].module is module.proj_out and boundaries[0].phase == "output"
    assert module.proj_out._forward_hooks[boundaries[0].handle.id] is boundaries[0].callback
    assert module._diffusers_hook.get_hook("cp_input---") is split
    assert module.proj_out._diffusers_hook.get_hook("cp_output---proj_out") is None


@pytest.mark.parametrize("boundary", ["image", "text", "ids", "gather"])
def test_flux_sp_refuses_changed_native_cp_metadata(boundary):
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    module, split, gather = _cp_boundaries()
    if boundary == "gather":
        gather.metadata = [ContextParallelOutput(gather_dim=2, expected_dims=3)]
    else:
        name = {"image": "hidden_states", "text": "encoder_hidden_states", "ids": "img_ids"}[
            boundary
        ]
        split.metadata[name] = ContextParallelInput(split_dim=2, expected_dims=3)
    with pytest.raises(ValueError, match="native root image/text split and proj_out gather"):
        Flux1Adapter.compose_context_parallel_sequence_boundaries(module)


def test_kontext_sp_recipe_preserves_conditioning_and_default_off(monkeypatch, tmp_path):
    import flux_kontext_coupled_fixture as base
    import flux_kontext_sp_coupled_fixture as sequence

    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = {
        "algorithm": "grpo-guard",
        "training_mode": "full",
        "dp_size": 2,
        "tp_size": 2,
        "cp_size": 2,
    }
    baseline = base.make_config(tmp_path, tmp_path, options)
    config = sequence.make_config(tmp_path, tmp_path, options)
    assert not baseline.parallelism_args.sequence_parallel
    assert config.parallelism_args.sequence_parallel is True
    assert config.model_args == baseline.model_args
    assert config.training_args == baseline.training_args
    assert config.eval_args == baseline.eval_args
    assert config.training_args.condition_image_size == [32, 64]
    assert sequence.dataset_record is base.dataset_record
    validate_context_parallel_config(config, Flux1KontextAdapter, get_trainer_class("grpo-guard"))


def test_flux_text_padding_requires_mask_and_native_processor(monkeypatch):
    """A removed key mask must change image output; padding alone is not equivalent."""
    from types import SimpleNamespace

    from flow_factory.models.flux._sequence_parallel import FluxSequenceParallelInput

    # This single-rank negative control isolates native attention masking. Actual
    # output gather and all its gradients are covered by the distributed matrix.
    monkeypatch.setattr(
        "flow_factory.models.flux._sequence_parallel.PrepareModuleOutput._apply",
        lambda _style, module, _mesh: module,
    )
    model = flux_transformer(num_layers=2, num_single_layers=2).double()
    model.set_attention_backend("native")
    inputs = {name: value.double() for name, value in flux_inputs(0, 0, "cpu").items()}
    inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"][:, :5]
    inputs["txt_ids"] = inputs["txt_ids"][:5]
    with torch.no_grad():
        expected = model(**inputs, return_dict=False)[0]
        FluxSequenceParallelInput()._apply(model, SimpleNamespace(size=lambda: 2))
        masked = model(**inputs, return_dict=False)[0]
        torch.testing.assert_close(masked, expected, rtol=1e-8, atol=1e-9)
        model.register_forward_pre_hook(
            lambda _module, args, kwargs: (args, {**kwargs, "joint_attention_kwargs": None}),
            with_kwargs=True,
        )
        unmasked = model(**inputs, return_dict=False)[0]
        assert (unmasked - expected).abs().max().item() > 1e-5

    unsupported = flux_transformer()
    unsupported.single_transformer_blocks[0].attn.set_processor(torch.nn.Identity())
    with pytest.raises(ValueError, match="native FluxAttnProcessor at single_transformer_blocks.0"):
        FluxSequenceParallelInput()._apply(unsupported, SimpleNamespace(size=lambda: 2))


@pytest.mark.parametrize(
    "extra,error",
    [
        (
            {"joint_attention_kwargs": {"ip_adapter_image_embeds": torch.zeros(1)}},
            "extra attention arguments",
        ),
        ({"controlnet_block_samples": [torch.zeros(1)]}, "ControlNet residuals"),
        ({"controlnet_single_block_samples": [torch.zeros(1)]}, "ControlNet residuals"),
    ],
)
def test_flux_sp_rejects_unsupported_native_side_inputs(extra, error):
    from types import SimpleNamespace

    from flow_factory.models.flux._sequence_parallel import FluxSequenceParallelInput

    model = flux_transformer()
    FluxSequenceParallelInput()._apply(model, SimpleNamespace(size=lambda: 2))
    with pytest.raises(ValueError, match=error):
        model(**flux_inputs(0, 0, "cpu"), **extra)


def test_native_flux_sp_preserves_lora_scale_with_odd_text(tmp_path):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        {
            "mode": "expanded_lora",
            "checkpointing": False,
            "sp": True,
            "text_tokens": 5,
            "lora_scale": 0.7,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
