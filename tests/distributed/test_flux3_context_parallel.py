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

"""Native Flux3 packed TP/SP/CP references and real conditioning lifecycles."""

import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_model_preparation import (
    PreparationAdapter,
    assert_update_parity,
    full_value,
    initialize_cuda,
    optimizer_for,
    parameter_map,
    runtime_observation,
)


def _flux3_optimizer_for(adapter: PreparationAdapter) -> torch.optim.AdamW:
    """Match the conditioned FLUX3 lifecycle fixture's learning-rate scale."""
    optimizer = optimizer_for(adapter)
    optimizer.defaults["lr"] = 1e-3
    for group in optimizer.param_groups:
        group["lr"] = 1e-3
    return optimizer


def _reference_worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=8)
    result = {"rank": rank, "execution_scope": "model_probe", **options}
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from flux3_cp_fixture import (
            VARIANTS,
            build_transformer,
            install_lora,
        )
        from flux3_cp_fixture import model_inputs as fixture_inputs
        from flux3_cp_fixture import native_velocity

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.flux.flux3 import Flux3Adapter
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import (
            context_parallel_geometry,
            data_parallel_geometry,
            tensor_parallel_geometry,
        )

        variant = options["variant"]
        adapter_class = Flux3Adapter

        def model_inputs(root, replica, step, device):
            return fixture_inputs(root, variant, replica, step, device)

        class Flux3PreparationAdapter(PreparationAdapter):
            context_parallel_capability = adapter_class.context_parallel_capability
            context_parallel_diffusers_revision = adapter_class.context_parallel_diffusers_revision
            install_context_parallel_module = adapter_class.install_context_parallel_module
            is_distilled = adapter_class.is_distilled
            _as_2d_position_ids = staticmethod(adapter_class._as_2d_position_ids)
            _prepare_text_ids = staticmethod(adapter_class._prepare_text_ids)
            validate_context_parallel_module = adapter_class.validate_context_parallel_module
            tensor_parallel_plan = adapter_class.tensor_parallel_plan
            sequence_parallel_plan = adapter_class.sequence_parallel_plan
            sequence_parallel_by_default = False
            compose_context_parallel_sequence_boundaries = (
                adapter_class.compose_context_parallel_sequence_boundaries
            )
            _validate_flux3_context_parallel_inputs = (
                adapter_class._validate_flux3_context_parallel_inputs
            )
            _assert_flux3_fsdp_schedule = adapter_class._assert_flux3_fsdp_schedule
            _forward = adapter_class._forward
            forward = adapter_class.forward

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=[
                    "Flux3ModeTransformerBlock",
                    "Flux3SingleTransformerBlock",
                ],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=True,
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=2, sp_size=1
            ),
        )
        assert data_parallel_geometry(accelerator) == (2, rank // 4)
        assert context_parallel_geometry(accelerator) == (2, rank // 2 % 2)
        assert tensor_parallel_geometry(accelerator) == (2, rank % 2)
        result["runtime"] = {
            **runtime_observation(accelerator),
            "tp_size": tensor_parallel_geometry(accelerator)[0],
            "tp_rank": tensor_parallel_geometry(accelerator)[1],
            "cp_size": context_parallel_geometry(accelerator)[0],
            "cp_rank": context_parallel_geometry(accelerator)[1],
        }
        module = build_transformer(variant).float()
        if options["mode"] == "lora":
            module = install_lora(module, variant).float()
        module.set_attention_backend("native")
        reference_module = copy.deepcopy(module).to(accelerator.device)
        adapter = Flux3PreparationAdapter(module, accelerator, options["mode"])
        adapter.pipeline = SimpleNamespace(
            transformer=module, scheduler=FlowMatchEulerDiscreteSDEScheduler(noise_level=0.35)
        )
        adapter.config = SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=True))
        adapter._has_warned_forward_fallback = False
        adapter.declare_component_variants(("policy",))
        reference_adapter = PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        reference_optimizer = _flux3_optimizer_for(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(cp_size=2, tp_size=2, sequence_parallel=True),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        model, optimizer = owner.prepare(_flux3_optimizer_for(adapter))
        owner.install_routes()
        optimizer_fields = ("lr", "eps", "betas", "weight_decay", "amsgrad", "maximize")
        result["optimizer_hyperparameters"] = {
            "parallel": [
                {key: group[key] for key in optimizer_fields} for group in optimizer.param_groups
            ],
            "reference": [
                {key: group[key] for key in optimizer_fields}
                for group in reference_optimizer.param_groups
            ],
        }
        assert (
            result["optimizer_hyperparameters"]["parallel"]
            == result["optimizer_hyperparameters"]["reference"]
        )
        contract = accelerator.unwrap_model(model).context_parallel_contract
        assert contract.capability == adapter_class.context_parallel_capability
        assert contract.boundaries == ()
        result["cp_preparation_owner"] = "diffusers"
        result["fsdp_mesh"] = list(accelerator.parallelism_config.fsdp_dim_names)
        result["updates"] = []
        from flow_factory.models.tensor_parallel import peel_peft_wrapper

        native = peel_peft_wrapper(module)
        stream_shapes = set()
        for label, block in (
            ("image", native.mode_blocks["image"][0]),
            ("reference", native.mode_blocks["image_cond"][0]),
            ("text", native.txt_mode_blocks[0]),
            ("joint", native.single_transformer_blocks[0]),
        ):
            block.norm.register_forward_pre_hook(
                lambda _m, args, kwargs, name=label: stream_shapes.add((name, args[0].shape[1])),
                with_kwargs=True,
            )
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = []
            for replica in range(2):
                inputs = model_inputs(options["fixture"], replica, step, accelerator.device)
                output = native_velocity(
                    lambda **kw: reference("transformer", **kw), inputs, variant
                )
                expected_outputs.append(output.detach())
                (output.square().mean() / 2).backward()
            inputs = model_inputs(options["fixture"], rank // 4, step, accelerator.device)
            assert inputs["latents"].shape == (1, 16, 128)
            assert inputs["image_latents"].shape == (1, 8, 128)
            assert inputs["latent_ids"].shape == (1, 16, 4)
            assert inputs["image_latent_ids"].shape == (1, 8, 4)
            output = adapter.forward(
                **inputs, return_kwargs=["velocity"], noise_level=0.35
            ).velocity
            torch.testing.assert_close(output, expected_outputs[rank // 4], rtol=3e-4, atol=2e-6)
            if step == 0:
                accelerator.backward(output.square().mean())
                expected, actual = parameter_map(reference), parameter_map(model)
                mismatches = 0
                for name, parameter in expected.items():
                    if parameter.grad is None:
                        assert actual[name].grad is None, name
                        continue
                    gradient = full_value(actual[name].grad)
                    torch.testing.assert_close(
                        gradient, parameter.grad / 2, rtol=3e-4, atol=2e-6, msg=name
                    )
                    mismatches += int(
                        not torch.allclose(gradient, parameter.grad, rtol=3e-4, atol=2e-6)
                    )
                assert mismatches > 0
                result["unscaled_loss_control"] = {
                    "gradient_factor": 0.5,
                    "mismatched_parameters": mismatches,
                }
                optimizer.zero_grad(set_to_none=True)
                output = adapter.forward(
                    **inputs, return_kwargs=["velocity"], noise_level=0.35
                ).velocity
            accelerator.backward(output.square().mean() * 2)
            result["updates"].append(
                assert_update_parity(reference, model, reference_optimizer, optimizer)
            )
        assert stream_shapes == {
            ("image", 4),
            ("reference", 2),
            ("text", 20),
            ("joint", 6),
            ("joint", 20),
        }, stream_shapes
        conditioned_stream_shapes = sorted(stream_shapes)
        stream_shapes.clear()
        # Exercise the absent-reference route with warm Adam state from all three conditioned updates.
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        expected_outputs = []
        for replica in range(2):
            no_reference = {
                **model_inputs(options["fixture"], replica, 3, accelerator.device),
                "image_latents": None,
                "image_latent_ids": None,
            }
            value = native_velocity(
                lambda **kw: reference("transformer", **kw), no_reference, variant
            )
            expected_outputs.append(value.detach())
            (value.square().mean() / 2).backward()
        no_reference = {
            **model_inputs(options["fixture"], rank // 4, 3, accelerator.device),
            "image_latents": None,
            "image_latent_ids": None,
        }
        value = adapter.forward(
            **no_reference, return_kwargs=["velocity"], noise_level=0.35
        ).velocity
        torch.testing.assert_close(value, expected_outputs[rank // 4], rtol=3e-4, atol=2e-6)
        accelerator.backward(value.square().mean() * 2)
        inactive = []
        for name, parameter in parameter_map(reference).items():
            if parameter.requires_grad:
                assert (parameter.grad is None) == (".image_cond." in name), name
                if parameter.grad is None:
                    inactive.append(name)
        assert inactive
        assert stream_shapes == {
            ("image", 4),
            ("text", 20),
            ("joint", 4),
            ("joint", 20),
        }, stream_shapes
        result["t2i_control"] = {
            "stream_shapes": sorted(stream_shapes),
            **assert_update_parity(reference, model, reference_optimizer, optimizer),
            "inactive_reference_parameters": inactive,
            "warm_state_unchanged": True,
        }
        result.update(
            status="pass",
            stream_shapes=conditioned_stream_shapes,
            execution_capability="flux3-packed-tp-sp-cp-fp32-probe-v1",
            target_tokens=16,
            reference_tokens=8,
            sequence_parallel=True,
            native_head_count=24,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="DP2 TP2 CP2 requires eight CUDA ranks")
@pytest.mark.parametrize("variant", ["base", "distilled"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cuda_flux3_tp_sp_cp_reference(tmp_path, mode, variant):
    fixture = tmp_path / "fixture"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("flux3_cp_fixture.py")),
            str(fixture),
            variant,
        ],
        check=True,
    )
    results = run_gloo_ranks(
        _reference_worker,
        tmp_path,
        8,
        {"mode": mode, "variant": variant, "fixture": str(fixture), "tp_size": 2, "cp_size": 2},
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "flux3_tp_sp_cp", "ranks": results}))


def test_cuda_flux3_guard_ordered_full(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "flux3_base_coupled_fixture",
            "variant": "base",
            "model_family": "flux3",
            "algorithm": "grpo-guard",
            "training_mode": "full",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
            "sequence_parallel": True,
        },
    )


def test_cuda_flux3_distilled_dppo_ordered_lora(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "flux3_distilled_coupled_fixture",
            "variant": "distilled",
            "model_family": "flux3",
            "algorithm": "dppo",
            "training_mode": "lora",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
            "sequence_parallel": True,
        },
    )
