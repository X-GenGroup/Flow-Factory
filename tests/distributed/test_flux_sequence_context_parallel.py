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

"""Native Flux1/Kontext dual-stream SP composed with TP, CP and FSDP2."""

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


def _reference_worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=8)
    result = {"rank": rank, "execution_scope": "model_probe", **options}
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from flux_kontext_cp_fixture import (
            flux_transformer,
            install_lora,
        )
        from flux_kontext_cp_fixture import model_inputs as kontext_inputs
        from flux_kontext_cp_fixture import reference_forward as kontext_reference
        from flux_kontext_cp_fixture import scheduler

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.flux.flux1 import Flux1Adapter
        from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import (
            context_parallel_geometry,
            data_parallel_geometry,
            tensor_parallel_geometry,
        )

        adapter_class = Flux1Adapter if options["family"] == "flux1" else Flux1KontextAdapter

        def model_inputs(root, replica, step, device):
            inputs = kontext_inputs(root, replica, step, device)
            if options["family"] == "flux1":
                inputs.pop("image_latents")
                inputs["img_ids"] = inputs.pop("latent_ids")[:16]
            return inputs

        def reference_forward(model, inputs):
            if options["family"] == "kontext":
                return kontext_reference(model, inputs)
            text = inputs["prompt_embeds"]
            return model(
                hidden_states=inputs["latents"],
                encoder_hidden_states=text,
                pooled_projections=inputs["pooled_prompt_embeds"],
                timestep=inputs["t"] / 1000,
                guidance=torch.full((1,), 3.5, device=text.device, dtype=text.dtype),
                txt_ids=torch.zeros(text.shape[1], 3, device=text.device, dtype=text.dtype),
                img_ids=inputs["img_ids"],
                return_dict=False,
            )[0]

        class KontextPreparationAdapter(PreparationAdapter):
            context_parallel_capability = adapter_class.context_parallel_capability
            validate_context_parallel_module = adapter_class.validate_context_parallel_module
            tensor_parallel_plan = adapter_class.tensor_parallel_plan
            sequence_parallel_plan = adapter_class.sequence_parallel_plan
            sequence_parallel_by_default = False
            compose_context_parallel_sequence_boundaries = (
                adapter_class.compose_context_parallel_sequence_boundaries
            )
            _validate_context_parallel_inputs = adapter_class._validate_context_parallel_inputs
            forward = adapter_class.forward

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=[
                    "FluxTransformerBlock",
                    "FluxSingleTransformerBlock",
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
        module = flux_transformer().float()
        if options["mode"] == "lora":
            module = install_lora(module).float()
        module.set_attention_backend("native")
        reference_module = copy.deepcopy(module).to(accelerator.device)
        adapter = KontextPreparationAdapter(module, accelerator, options["mode"])
        adapter.pipeline = SimpleNamespace(transformer=module, scheduler=scheduler())
        adapter.declare_component_variants(("policy",))
        reference_adapter = PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        reference_optimizer = optimizer_for(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(cp_size=2, tp_size=2, sequence_parallel=True),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        model, optimizer = owner.prepare(optimizer_for(adapter))
        owner.install_routes()
        contract = accelerator.unwrap_model(model).context_parallel_contract
        assert contract.capability == adapter_class.context_parallel_capability
        assert contract.boundaries == ()
        result["cp_preparation_owner"] = "diffusers"
        result["fsdp_mesh"] = list(accelerator.parallelism_config.fsdp_dim_names)
        result["updates"] = []
        from flow_factory.models.tensor_parallel import peel_peft_wrapper

        native = peel_peft_wrapper(module)
        stream_shapes = set()
        native.transformer_blocks[0].norm1.register_forward_pre_hook(
            lambda _m, args, kwargs: stream_shapes.add(("dual_image", args[0].shape[1])),
            with_kwargs=True,
        )
        native.single_transformer_blocks[0].norm.register_forward_pre_hook(
            lambda _m, args, kwargs: stream_shapes.add(("single_joint", args[0].shape[1])),
            with_kwargs=True,
        )
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = []
            for replica in range(2):
                inputs = model_inputs(options["fixture"], replica, step, accelerator.device)
                output = reference_forward(lambda **kw: reference("transformer", **kw), inputs)
                expected_outputs.append(output.detach())
                (output.square().mean() / 2).backward()
            inputs = model_inputs(options["fixture"], rank // 4, step, accelerator.device)
            assert inputs["latents"].shape == (1, 16, 16)
            if options["family"] == "kontext":
                assert inputs["image_latents"].shape == (1, 8, 16)
                assert inputs["latent_ids"].shape == (24, 3)
            else:
                assert inputs["img_ids"].shape == (16, 3)
            output = adapter.forward(**inputs, return_kwargs=["velocity"]).velocity
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
                output = adapter.forward(**inputs, return_kwargs=["velocity"]).velocity
            accelerator.backward(output.square().mean() * 2)
            result["updates"].append(
                assert_update_parity(reference, model, reference_optimizer, optimizer)
            )
        image_tokens = 24 if options["family"] == "kontext" else 16
        assert stream_shapes == {
            ("dual_image", image_tokens // 4),
            ("single_joint", (image_tokens + inputs["prompt_embeds"].shape[1]) // 2),
        }, stream_shapes
        result.update(
            status="pass",
            stream_shapes=sorted(stream_shapes),
            execution_capability="flux-dual-sp-cp-fp32-probe-v1",
            target_tokens=16,
            reference_tokens=8 if options["family"] == "kontext" else 0,
            sequence_parallel=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="DP2 TP2 CP2 requires eight CUDA ranks")
@pytest.mark.parametrize("family", ["flux1", "kontext"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cuda_flux_dual_sp_cp_reference(tmp_path, mode, family):
    fixture = tmp_path / "fixture"
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("flux_kontext_cp_fixture.py")), str(fixture)],
        check=True,
    )
    results = run_gloo_ranks(
        _reference_worker,
        tmp_path,
        8,
        {"mode": mode, "family": family, "fixture": str(fixture), "tp_size": 2, "cp_size": 2},
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "flux_dual_sp_cp", "ranks": results}))


def test_cuda_flux_kontext_sp_guard_ordered_full(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "flux_kontext_sp_coupled_fixture",
            "model_family": "flux-kontext",
            "algorithm": "grpo-guard",
            "training_mode": "full",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
            "sequence_parallel": True,
        },
    )
