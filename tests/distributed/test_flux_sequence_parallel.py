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

from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.tensor_parallel import (
    apply_tensor_parallel,
    peel_peft_wrapper,
    resolve_parallel_plan,
)


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    result = {"rank": rank, **options}
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        model = flux_transformer().double()
        if options["mode"] == "lora":
            model = flux_lora(model).double()
        model.set_attention_backend("native")
        reference = copy.deepcopy(model)
        if options["checkpointing"]:
            model.enable_gradient_checkpointing()
            reference.enable_gradient_checkpointing()
        apply_tensor_parallel(
            peel_peft_wrapper(model),
            mesh,
            resolve_parallel_plan(Flux1Adapter, sequence_parallel=True),
        )
        observed = []
        native = peel_peft_wrapper(model)
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
            inputs["hidden_states"].requires_grad_(True)
            expected = reference(**inputs).sample
            actual = model(**inputs).sample
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
            expected.square().mean().backward()
            actual.square().mean().backward()
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
        assert set(observed) == {("dual_image", 8), ("single_joint", 24)}, observed
        result.update(status="pass", active_parameters=active, updates=3, observed=observed)
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
