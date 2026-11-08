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

"""Native Flux dual-stream image SP boundaries and numerical controls."""

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
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    result = {"rank": rank, **options}
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        model = flux_transformer().double()
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
                        and name.startswith(
                            ("transformer_blocks.", "single_transformer_blocks.", "norm_out.")
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
            mesh,
            resolve_parallel_plan(Flux1Adapter, sequence_parallel=options.get("sp", True)),
        )
        observed = []
        native = peel_peft_wrapper(model)
        placements = _assert_added_placements(native)
        native.transformer_blocks[0].norm1.register_forward_pre_hook(
            lambda _module, args, kwargs: observed.append(("dual_image", args[0].shape[1])),
            with_kwargs=True,
        )
        native.single_transformer_blocks[0].norm.register_forward_pre_hook(
            lambda _module, args, kwargs: observed.append(("single_joint", args[0].shape[1])),
            with_kwargs=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        for step in range(3):
            inputs = {name: value.double() for name, value in flux_inputs(0, step, "cpu").items()}
            differentiable = ("hidden_states", "encoder_hidden_states", "pooled_projections")
            for name in differentiable:
                inputs[name].requires_grad_(True)
            reference_inputs = {
                name: value.detach().clone().requires_grad_(value.requires_grad)
                for name, value in inputs.items()
            }
            expected = reference(**reference_inputs).sample
            actual = model(**inputs).sample
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
            expected.square().mean().backward()
            actual.square().mean().backward()
            for name in differentiable:
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
        image_tokens = 8 if options.get("sp", True) else 16
        assert set(observed) == {("dual_image", image_tokens), ("single_joint", 24)}, observed
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
            restored(**inputs).sample, model(**inputs).sample, rtol=1e-8, atol=1e-9
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
    module.proj_out._diffusers_hook = SimpleNamespace(get_hook=output_hooks.get)
    return module, split, gather


def test_native_cp_callbacks_enclose_flux_sp_without_relocation():
    module, split, gather = _cp_boundaries()
    assert Flux1Adapter.compose_context_parallel_sequence_boundaries(module) == ()
    assert Flux1KontextAdapter.compose_context_parallel_sequence_boundaries(module) == ()
    assert module._diffusers_hook.get_hook("cp_input---") is split
    assert module.proj_out._diffusers_hook.get_hook("cp_output---proj_out") is gather


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
