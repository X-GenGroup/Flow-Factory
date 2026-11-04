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

"""Preparation order, live parameter identity and shared variant routes."""

from __future__ import annotations

import importlib.metadata
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from conftest import run_gloo_ranks
from peft import LoraConfig, get_peft_model
from torch.distributed.tensor import DTensor

from flow_factory.hparams.parallelism_args import ParallelismArguments
from flow_factory.loading.coordinator import ModelLoadCoordinator
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.runtime import PseudoPipelineRuntime
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.trainers.common.model_preparation import ModelPreparation


class TinyLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class PreparationAdapter(BaseAdapter):
    """Use production variant/runtime mechanisms around a small supplied model."""

    def __init__(self, module: torch.nn.Module, accelerator: Any, mode: str) -> None:
        self.model_args = SimpleNamespace(
            finetune_type=mode,
            trainable_parameters_dtype=torch.float32,
            target_components=["transformer"],
        )
        self.accelerator = accelerator
        self.target_module_map = {"transformer": "all"}
        self.component_runtime = PseudoPipelineRuntime(SimpleNamespace(), {"transformer": module})
        self._components = self.component_runtime.override_components

    @property
    def trainable_component_names(self) -> list[str]:
        return ["transformer"]

    tensor_parallel_plan = Wan2_T2V_Adapter.tensor_parallel_plan
    sequence_parallel_plan = Wan2_T2V_Adapter.sequence_parallel_plan
    sequence_parallel_by_default = True

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_component("transformer")(*args, **kwargs)


def optimizer_for(adapter: PreparationAdapter) -> torch.optim.AdamW:
    registry = adapter.component_variant_registry
    return torch.optim.AdamW(
        [
            {"params": registry.parameters(name), "role_name": name}
            for name in registry.variant_names
        ],
        lr=0.01,
        eps=1e-4,
    )


@pytest.fixture
def prepared_fixture():
    calls = []

    def prepare(*objects):
        calls.append(objects)
        return objects

    accelerator = SimpleNamespace(
        num_processes=1,
        process_index=0,
        distributed_type=DistributedType.NO,
        state=SimpleNamespace(distributed_type=DistributedType.NO),
        unwrap_model=lambda model: model,
        prepare=prepare,
    )
    adapter = PreparationAdapter(TinyLinear(), accelerator, "full")
    adapter.declare_component_variants(("policy", "critic"))
    owner = ModelPreparation(ModelLoadCoordinator(adapter, accelerator), ParallelismArguments())
    return adapter, owner, calls


def test_order_and_duplicate_preparation_fail_before_backend_mutation(prepared_fixture):
    adapter, owner, calls = prepared_fixture
    optimizer = optimizer_for(adapter)
    for operation in (lambda: owner.prepare(optimizer), owner.install_routes, owner.require_ready):
        with pytest.raises(RuntimeError, match="phase is 'new'"):
            operation()
    assert not calls
    owner.parallelize()
    with pytest.raises(RuntimeError, match="phase is 'parallelized'"):
        owner.parallelize()
    model, received_optimizer, loader = owner.prepare(optimizer, "eval-loader")
    assert received_optimizer is optimizer and loader == "eval-loader"
    assert len(calls) == 1 and calls[0][0] is model
    with pytest.raises(RuntimeError, match="phase is 'prepared'"):
        owner.prepare(optimizer)
    owner.install_routes()
    owner.require_ready()
    with pytest.raises(RuntimeError, match="phase is 'ready'"):
        owner.install_routes()
    assert len(calls) == 1
    for name in adapter.component_variant_registry.variant_names:
        with adapter.use_component_variant(name):
            expected = model.members[
                adapter.component_variant_registry.resolve_route(name, "transformer")
            ]
            value = torch.ones(1, 4)
            torch.testing.assert_close(adapter.forward(value), expected(value))


@pytest.mark.parametrize("corruption", ["stale", "duplicate", "missing"])
def test_optimizer_identity_is_checked_before_backend_mutation(prepared_fixture, corruption):
    adapter, owner, calls = prepared_fixture
    owner.parallelize()
    optimizer = optimizer_for(adapter)
    parameters = optimizer.param_groups[0]["params"]
    if corruption == "stale":
        parameters[0] = torch.nn.Parameter(parameters[0].detach().clone())
    elif corruption == "duplicate":
        parameters.append(parameters[0])
    else:
        parameters.pop()
    with pytest.raises(ValueError, match="exactly once before prepare"):
        owner.prepare(optimizer)
    assert not calls


def test_failed_backend_mutation_cannot_be_retried(prepared_fixture):
    adapter, owner, calls = prepared_fixture
    owner.parallelize()
    optimizer = optimizer_for(adapter)

    def fail(*objects):
        calls.append(objects)
        raise RuntimeError("backend changed a parameter then failed")

    owner.load_coordinator.backend.prepare = fail
    with pytest.raises(RuntimeError, match="backend changed"):
        owner.prepare(optimizer)
    with pytest.raises(RuntimeError, match="phase is 'preparing'"):
        owner.prepare(optimizer)
    assert len(calls) == 1


def test_failed_parallel_mutation_cannot_be_retried(prepared_fixture):
    _, owner, _ = prepared_fixture
    calls = []

    def fail():
        calls.append("transform")
        raise RuntimeError("transform failed after changing a module")

    owner._apply_tensor_parallel = fail
    with pytest.raises(RuntimeError, match="transform failed"):
        owner.parallelize()
    with pytest.raises(RuntimeError, match="phase is 'parallelizing'"):
        owner.parallelize()
    assert calls == ["transform"]


def test_live_registry_rebind_precedes_routes_and_snapshots(prepared_fixture):
    adapter, owner, calls = prepared_fixture
    registry = adapter.component_variant_registry
    old = registry.parameters("policy")[0]
    owner.parallelize()
    optimizer = optimizer_for(adapter)

    def replacing_backend(model, optimizer):
        parameter = model.members["transformer"].proj.weight
        replacement = torch.nn.Parameter(parameter.detach().clone())
        model.members["transformer"].proj.weight = replacement
        for group in optimizer.param_groups:
            group["params"] = [replacement if p is parameter else p for p in group["params"]]
        return model, optimizer

    owner.load_coordinator.backend.prepare = replacing_backend
    owner.prepare(optimizer)
    assert registry.parameters("policy")[0] is not old
    assert registry.parameters("policy")[0] is optimizer.param_groups[0]["params"][0]
    with pytest.raises(RuntimeError, match="phase is 'prepared'"):
        owner.require_ready()
    owner.install_routes()
    owner.require_ready()


def test_lora_variants_share_one_prepared_route(prepared_fixture):
    base, _, calls = prepared_fixture
    module = get_peft_model(TinyLinear(), LoraConfig(r=2, target_modules=["proj"]))
    adapter = PreparationAdapter(module, base.accelerator, "lora")
    adapter.declare_component_variants(("policy", "critic"))
    owner = ModelPreparation(
        ModelLoadCoordinator(adapter, base.accelerator), ParallelismArguments()
    )
    owner.parallelize()
    model, optimizer = owner.prepare(optimizer_for(adapter))
    owner.install_routes()
    registry = adapter.component_variant_registry
    assert tuple(model.members) == ("transformer",)
    assert registry.resolve_route("policy", "transformer") == registry.resolve_route(
        "critic", "transformer"
    )
    assert not (
        {id(p) for p in registry.parameters("policy")}
        & {id(p) for p in registry.parameters("critic")}
    )
    assert len(calls) == 1


@pytest.mark.parametrize("missing", [False, True])
def test_unmaterialized_variants_cannot_be_transformed(prepared_fixture, missing):
    adapter, owner, _ = prepared_fixture
    if missing:
        del adapter.component_variant_registry
    else:
        adapter.component_variant_registry._is_frozen = False
    with pytest.raises(RuntimeError, match="materialized, frozen"):
        owner.parallelize()


def full_value(value: torch.Tensor) -> torch.Tensor:
    return (value.full_tensor() if isinstance(value, DTensor) else value).detach().clone()


def parameter_map(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    from flow_factory.models.variants import ComponentVariantRegistry

    pairs = [
        (ComponentVariantRegistry._canonical_prepared_parameter_name(name), parameter)
        for name, parameter in model.named_parameters()
    ]
    assert len(dict(pairs)) == len(pairs)
    return dict(sorted(pairs))


def assert_update_parity(reference, actual, reference_optimizer, optimizer):
    """Compare each gradient, update and warm optimizer state, including inactive roles."""
    expected, received = parameter_map(reference), parameter_map(actual)
    assert expected.keys() == received.keys()
    before = {}
    inactive_state = {}
    gradients = 0
    nonzero_gradients = 0
    for name, parameter in expected.items():
        current = received[name]
        assert (parameter.grad is None) == (current.grad is None), name
        before[name] = (full_value(parameter), full_value(current))
        if parameter.grad is not None:
            gradients += 1
            gradient = full_value(current.grad)
            assert torch.isfinite(gradient).all() and torch.isfinite(parameter.grad).all(), name
            nonzero_gradients += int(bool(torch.count_nonzero(gradient)))
            torch.testing.assert_close(gradient, parameter.grad, rtol=3e-4, atol=2e-6, msg=name)
            scale = parameter.grad.abs().max()
            assert (gradient - parameter.grad).abs().max() <= 8e-4 * scale + 2e-8, name
        else:
            inactive_state[name] = {
                key: full_value(value) for key, value in optimizer.state.get(current, {}).items()
            }
    assert gradients and nonzero_gradients
    reference_optimizer.step()
    optimizer.step()
    changed = 0
    for name, parameter in expected.items():
        current = received[name]
        ref_before, current_before = before[name]
        ref_update, current_update = (
            full_value(parameter) - ref_before,
            full_value(current) - current_before,
        )
        torch.testing.assert_close(current_update, ref_update, rtol=3e-4, atol=2e-6, msg=name)
        if parameter.grad is None:
            torch.testing.assert_close(
                full_value(current), current_before, rtol=0, atol=0, msg=name
            )
            for key, value in inactive_state[name].items():
                torch.testing.assert_close(
                    full_value(optimizer.state[current][key]), value, rtol=0, atol=0, msg=name
                )
        else:
            changed += int(bool(torch.count_nonzero(ref_update)))
        expected_state, actual_state = reference_optimizer.state.get(
            parameter, {}
        ), optimizer.state.get(current, {})
        assert expected_state.keys() == actual_state.keys(), name
        for key in expected_state:
            torch.testing.assert_close(
                full_value(actual_state[key]),
                full_value(expected_state[key]),
                rtol=8e-4,
                atol=2e-7,
                msg=f"{name}.{key}",
            )
    assert changed
    return {
        "gradient_parameters": gradients,
        "updated_parameters": changed,
        "inactive_parameters": len(inactive_state),
        "elementwise_comparison": True,
        "nonzero_gradients": nonzero_gradients > 0,
    }


def wan_inputs(replica: int, step: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(41 + step * 2 + replica)
    return {
        "hidden_states": torch.randn(1, 4, 2, 4, 4, generator=generator).to(device),
        "encoder_hidden_states": torch.randn(1, 5, 16, generator=generator).to(device),
        "timestep": torch.tensor([450.0 + 100 * replica], device=device),
    }


def initialize_cuda(rank: int, store_path: str, world_size: int = 4) -> None:
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size), ACCELERATE_USE_FSDP="true"
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        store=dist.FileStore(store_path, world_size),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )


def runtime_observation(accelerator) -> dict[str, Any]:
    return {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "device": accelerator.device.type,
        "device_index": accelerator.device.index,
        "distributed_type": accelerator.distributed_type.name,
        "fsdp_version": getattr(
            getattr(accelerator.state, "fsdp_plugin", None), "fsdp_version", None
        ),
        "zero_stage": getattr(
            getattr(accelerator.state, "deepspeed_plugin", None), "zero_stage", None
        ),
        "process_group_backend": dist.get_backend(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "accelerate", "peft")
        },
    }


def _variant_worker(rank: int, store_path: str, result_dir: str, options: dict) -> None:
    initialize_cuda(rank, store_path)
    result = {"rank": rank, **options}
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from test_wan_sequence_parallel_parity import _build, _with_lora

        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.trainers.common import model_preparation as preparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import data_parallel_geometry

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["WanTransformerBlock"],
                activation_checkpointing=True,
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, tp_size=2, cp_size=1, sp_size=1
            ),
        )
        result["runtime"] = runtime_observation(accelerator)
        assert data_parallel_geometry(accelerator) == (2, rank // 2)
        mode = options["mode"]
        module = (_with_lora(_build()) if mode == "lora" else _build()).float()
        adapter = PreparationAdapter(module, accelerator, mode)
        adapter.declare_component_variants(("policy", "critic"))
        reference_module = (_with_lora(_build()) if mode == "lora" else _build()).float()
        reference_adapter = PreparationAdapter(
            reference_module.to(accelerator.device), accelerator, mode
        )
        reference_adapter.declare_component_variants(("policy", "critic"))
        reference_members = reference_adapter.component_variant_registry.bundle_members()
        for route, member in reference_members.items():
            member.load_state_dict(
                adapter.component_variant_registry.bundle_members()[route].state_dict()
            )
        reference = ModelBundle(reference_members)
        reference_optimizer = optimizer_for(reference_adapter)
        sharded_ids = []
        original_apply = preparation.apply_tensor_parallel

        def record(module, mesh, plan):
            sharded_ids.append(id(module))
            return original_apply(module, mesh, plan)

        preparation.apply_tensor_parallel = record
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(tp_size=2, sequence_parallel=True),
        )
        owner.parallelize()
        assert len(sharded_ids) == (1 if mode == "lora" else 2)
        assert len(sharded_ids) == len(set(sharded_ids))
        optimizer = optimizer_for(adapter)
        stale = {id(p) for group in optimizer.param_groups for p in group["params"]}
        model, optimizer = owner.prepare(optimizer)
        owner.install_routes()
        owner.require_ready()
        registry = adapter.component_variant_registry
        owned = {id(p) for name in registry.variant_names for p in registry.parameters(name)}
        assert owned == {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert owned <= {id(p) for p in model.parameters()}
        assert owned != stale
        assert all(isinstance(p, DTensor) for p in model.parameters())
        result["updates"] = []
        for step, role in enumerate(("policy", "critic", "policy", "critic")):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            reference_outputs = []
            with reference_adapter.use_component_variant(role):
                route = reference_adapter.component_variant_registry.resolve_route(
                    role, "transformer"
                )
                for replica in range(2):
                    output = reference(
                        route, **wan_inputs(replica, step, accelerator.device), return_dict=False
                    )[0]
                    reference_outputs.append(output.detach())
                    (output.square().mean() / 2).backward()
            with adapter.use_component_variant(role):
                output = adapter.forward(
                    **wan_inputs(rank // 2, step, accelerator.device), return_dict=False
                )[0]
                torch.testing.assert_close(
                    output, reference_outputs[rank // 2], rtol=3e-4, atol=2e-6
                )
                accelerator.backward(output.square().mean())
            result["updates"].append(
                {
                    "role": role,
                    **assert_update_parity(reference, model, reference_optimizer, optimizer),
                }
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="DP2 x TP2 preparation requires four CUDA ranks"
)
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cuda_variant_preparation_matches_unsharded_updates(tmp_path, mode):
    results = run_gloo_ranks(_variant_worker, tmp_path, 4, {"mode": mode})
    for result in results:
        assert "error" not in result, result.get("error")
        assert result["status"] == "pass"
    print(json.dumps({"probe": "variant_preparation", "ranks": results}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Backend preparation requires CUDA")
def test_cuda_backend_prepares_one_root_and_updates_both_routes(backend):
    from accelerate import DistributedDataParallelKwargs

    from flow_factory.trainers.loader import Accelerator

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    assert accelerator.num_processes == 2
    assert (
        accelerator.distributed_type.name
        == {"ddp": "MULTI_GPU", "fsdp2": "FSDP", "zero2": "DEEPSPEED"}[backend]
    )
    torch.manual_seed(10)
    adapter = PreparationAdapter(TinyLinear(), accelerator, "full")
    adapter.declare_component_variants(("policy", "critic"))
    owner = ModelPreparation(ModelLoadCoordinator(adapter, accelerator), ParallelismArguments())
    owner.parallelize()
    model, optimizer = owner.prepare(optimizer_for(adapter))
    owner.install_routes()
    owner.require_ready()
    registry = adapter.component_variant_registry
    for role in registry.variant_names:
        optimizer.zero_grad(set_to_none=True)
        before = [full_value(p) for p in registry.parameters(role)]
        with adapter.use_component_variant(role):
            with accelerator.autocast():
                output = adapter.forward(torch.ones(2, 4, device=accelerator.device))
                loss = output.square().mean()
            accelerator.backward(loss)
            optimizer.step()
        assert any(
            not torch.equal(old, full_value(p)) for old, p in zip(before, registry.parameters(role))
        )
    with pytest.raises(RuntimeError, match="phase is 'ready'"):
        owner.prepare(optimizer)
    print(
        json.dumps(
            {
                "probe": "backend_preparation",
                "runtime": runtime_observation(accelerator),
                "backend": backend,
            }
        )
    )
