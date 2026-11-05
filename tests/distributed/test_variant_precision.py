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

"""Original dtype, BF16 execution and role isolation through real FSDP2 + TP."""

import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from accelerate import FullyShardedDataParallelPlugin
from accelerate.parallelism_config import ParallelismConfig
from conftest import run_gloo_ranks
from test_model_preparation import (
    PreparationAdapter,
    initialize_cuda,
    optimizer_for,
    runtime_observation,
    wan_inputs,
)
from test_wan_sequence_parallel_parity import _build, _with_lora
from torch.distributed.tensor import DTensor

from flow_factory.hparams.parallelism_args import ParallelismArguments
from flow_factory.loading.coordinator import ModelLoadCoordinator
from flow_factory.trainers.common.model_preparation import ModelPreparation
from flow_factory.trainers.loader import Accelerator


def local_copy(tensor):
    return (tensor.to_local() if isinstance(tensor, DTensor) else tensor).detach().clone()


def precision_worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=4)
    result = {"rank": rank, **options}
    try:
        accelerator = Accelerator(
            mixed_precision="bf16",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["WanTransformerBlock"],
                activation_checkpointing=True,
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
            ),
            parallelism_config=ParallelismConfig(dp_shard_size=2, tp_size=2),
        )
        result["runtime"] = runtime_observation(accelerator)
        mode = options["mode"]
        module = (_with_lora(_build()) if mode == "lora" else _build()).float()
        adapter = PreparationAdapter(module, accelerator, mode)
        adapter.model_args.trainable_parameters_dtype = torch.bfloat16
        adapter.declare_component_variants(("policy", "critic"))
        registry = adapter.component_variant_registry
        assert all(
            p.dtype == torch.float32 for v in registry.variant_names for p in registry.parameters(v)
        )
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(tp_size=2, sequence_parallel=True),
        )
        owner.parallelize()
        model, optimizer = owner.prepare(optimizer_for(adapter))
        owner.install_routes()
        parameters = {role: list(registry.parameters(role)) for role in registry.variant_names}
        owned = {id(p) for values in parameters.values() for p in values}
        assert owned == {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(
            isinstance(p, DTensor) and p.dtype == torch.float32
            for values in parameters.values()
            for p in values
        )
        compute_dtypes = []

        def observe_compute(module, args):
            compute_dtypes.append((args[0].dtype, module.weight.dtype))

        handles = [
            module.register_forward_pre_hook(observe_compute)
            for name, module in model.named_modules()
            if name.endswith("attn1.to_q")
        ]
        assert handles
        try:
            for step, role in enumerate(("policy", "critic", "policy", "critic")):
                optimizer.zero_grad(set_to_none=True)
                inactive = [
                    p for other, values in parameters.items() if other != role for p in values
                ]
                before = {id(p): local_copy(p) for values in parameters.values() for p in values}
                inactive_state = {
                    id(p): {
                        key: local_copy(value) for key, value in optimizer.state.get(p, {}).items()
                    }
                    for p in inactive
                }
                with adapter.use_component_variant(role):
                    output = adapter.forward(
                        **wan_inputs(rank // 2, step, accelerator.device), return_dict=False
                    )[0]
                    assert torch.isfinite(output).all()
                    accelerator.backward(output.square().mean())
                assert all(p.grad is None for p in inactive)
                assert any(
                    p.grad is not None and torch.count_nonzero(local_copy(p.grad))
                    for p in parameters[role]
                )
                assert all(
                    p.grad is None or torch.isfinite(local_copy(p.grad)).all()
                    for p in parameters[role]
                )
                optimizer.step()
                assert any(not torch.equal(local_copy(p), before[id(p)]) for p in parameters[role])
                for p in inactive:
                    torch.testing.assert_close(local_copy(p), before[id(p)], rtol=0, atol=0)
                    assert optimizer.state.get(p, {}).keys() == inactive_state[id(p)].keys()
                    for key, value in inactive_state[id(p)].items():
                        torch.testing.assert_close(
                            local_copy(optimizer.state[p][key]), value, rtol=0, atol=0
                        )
                assert owned == {id(p) for values in parameters.values() for p in values}
            assert compute_dtypes and set(compute_dtypes) == {(torch.bfloat16, torch.bfloat16)}
        finally:
            for handle in handles:
                handle.remove()
        result.update(
            status="pass",
            original_dtype="float32",
            compute_dtype="bfloat16",
            role_updates=4,
            inactive_state_exact=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cuda_late_roles_preserve_original_dtype_and_bf16_compute(tmp_path, mode):
    if torch.cuda.device_count() < 4:
        pytest.skip("mixed-precision variant qualification requires DP2 x TP2 CUDA ranks")
    rows = run_gloo_ranks(precision_worker, tmp_path, 4, {"mode": mode})
    for row in rows:
        assert "error" not in row, row.get("error")
        assert row["status"] == "pass"
    print(json.dumps({"probe": "variant_original_precision", "ranks": rows}))
