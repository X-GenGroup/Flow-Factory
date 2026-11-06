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

"""Real Wan blocks through the production CP preparation bridge.

This component probe checks the output gather and gradient-reduction contract;
complete SFT execution is qualified separately. It does not compose TP and CP.
"""

from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path

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
    wan_inputs,
)


def _worker(rank: int, store_path: str, result_dir: str, options: dict) -> None:
    initialize_cuda(rank, store_path)
    result = {"rank": rank, "execution_scope": "model_probe", **options}
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from test_wan_sequence_parallel_parity import _build, _with_lora

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import context_parallel_geometry, data_parallel_geometry

        class WanPreparationAdapter(PreparationAdapter):
            context_parallel_capability = Wan2_T2V_Adapter.context_parallel_capability
            validate_context_parallel_module = Wan2_T2V_Adapter.validate_context_parallel_module

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["WanTransformerBlock"],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=options["checkpointing"],
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=1, sp_size=1
            ),
        )
        result["runtime"] = runtime_observation(accelerator)
        assert data_parallel_geometry(accelerator) == (2, rank // 2)
        assert context_parallel_geometry(accelerator) == (2, rank % 2)
        module = _build().float()
        if options["mode"] == "lora":
            module = _with_lora(module).float()
        reference_module = copy.deepcopy(module).to(accelerator.device)
        # Native SDPA supports autograd through Ulysses; native Ring needs LSE output.
        module.set_attention_backend("native")
        reference_module.set_attention_backend("native")
        adapter = WanPreparationAdapter(module, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        reference_adapter = PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        reference_optimizer = optimizer_for(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator), ParallelismArguments(cp_size=2)
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        model, optimizer = owner.prepare(optimizer_for(adapter))
        owner.install_routes()
        assert getattr(accelerator, "_cp_context", None) is None
        result["cp_preparation_owner"] = "diffusers"
        result["fsdp_mesh"] = list(accelerator.parallelism_config.fsdp_dim_names)
        result["updates"] = []
        # Diffusers gather backward returns only this CP rank's output slice.
        # FSDP averages gradients over DP x CP. Multiplying by CP=2 restores the
        # mean over distinct logical samples rather than averaging token pieces.
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = []
            for replica in range(2):
                output = reference(
                    "transformer",
                    **wan_inputs(replica, step, accelerator.device),
                    return_dict=False,
                )[0]
                expected_outputs.append(output.detach())
                (output.square().mean() / 2).backward()
            output = adapter.forward(
                **wan_inputs(rank // 2, step, accelerator.device), return_dict=False
            )[0]
            torch.testing.assert_close(output, expected_outputs[rank // 2], rtol=3e-4, atol=2e-6)
            if step == 0:
                accelerator.backward(output.square().mean())
                expected_parameters, actual_parameters = parameter_map(reference), parameter_map(
                    model
                )
                mismatches = 0
                for name, parameter in expected_parameters.items():
                    if parameter.grad is None:
                        assert actual_parameters[name].grad is None, name
                        continue
                    gradient = full_value(actual_parameters[name].grad)
                    torch.testing.assert_close(
                        gradient, parameter.grad / 2, rtol=3e-4, atol=2e-6, msg=name
                    )
                    mismatches += int(
                        not torch.allclose(gradient, parameter.grad, rtol=3e-4, atol=2e-6)
                    )
                assert (
                    mismatches > 0
                ), "the unscaled-loss control must detect wrong gradient normalization"
                result["unscaled_loss_control"] = {
                    "gradient_factor": 0.5,
                    "mismatched_parameters": mismatches,
                }
                optimizer.zero_grad(set_to_none=True)
                output = adapter.forward(
                    **wan_inputs(rank // 2, step, accelerator.device), return_dict=False
                )[0]
            accelerator.backward(output.square().mean() * 2)
            result["updates"].append(
                assert_update_parity(reference, model, reference_optimizer, optimizer)
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="DP2 x CP2 probe requires four CUDA ranks"
)
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cuda_wan_ulysses_fsdp2_bridge(tmp_path, checkpointing, mode):
    results = run_gloo_ranks(_worker, tmp_path, 4, {"checkpointing": checkpointing, "mode": mode})
    for result in results:
        assert "error" not in result, result.get("error")
        assert result["status"] == "pass"
    print(json.dumps({"probe": "wan_diffusers_cp_fsdp2", "ranks": results}))
