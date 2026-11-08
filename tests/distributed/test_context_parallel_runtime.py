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

"""Native transport and physical-denoiser preparation ownership."""

import json
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from conftest import run_gloo_ranks
from peft import LoraConfig, get_peft_model
from test_model_preparation import PreparationAdapter, TinyLinear, initialize_cuda

from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
from flow_factory.hparams.parallelism_args import ParallelismArguments
from flow_factory.loading.coordinator import ModelLoadCoordinator
from flow_factory.models.context_parallel import (
    PreparedContextParallel,
    validate_prepared_context_parallel,
)
from flow_factory.models.model_bundle import ModelBundle
from flow_factory.models.runtime import PseudoPipelineRuntime
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.trainers.common.model_preparation import ModelPreparation


@pytest.mark.parametrize(
    "transport,degrees,backend", [("ulysses", (1, 2), "native"), ("ring", (2, 1), "_native_flash")]
)
def test_native_transport_identity_requires_matching_prepared_degrees(transport, degrees, backend):
    capability = replace(Wan2_T2V_Adapter.context_parallel_capability, transport=transport)
    assert capability.transport_degrees(2) == degrees
    assert capability.attention_backend == backend
    module = TinyLinear()
    native = SimpleNamespace(ring_degree=degrees[0], ulysses_degree=degrees[1])
    module._parallel_config = SimpleNamespace(context_parallel_config=native)
    bundle = ModelBundle({"transformer": module})
    bundle.context_parallel_contract = PreparedContextParallel(capability, 2, (module,))
    validate_prepared_context_parallel((bundle,), 2)
    native.ring_degree, native.ulysses_degree = degrees[::-1]
    with pytest.raises(ValueError, match="changed Diffusers plan"):
        validate_prepared_context_parallel((bundle,), 2)


@pytest.mark.parametrize("transport", ["anything", "auto", "all_to_all", None])
def test_transport_does_not_silently_select_an_unqualified_algorithm(transport):
    with pytest.raises(ValueError, match="unsupported context-parallel transport"):
        replace(Wan2_T2V_Adapter.context_parallel_capability, transport=transport)


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("train_second", [False, True])
def test_all_variant_and_frozen_denoisers_follow_one_preparation_owner(
    monkeypatch, mode, train_second
):
    events = []

    class CPLinear(TinyLinear):
        def set_attention_backend(self, backend):
            assert backend == "native"
            assert (id(self), "configure") in events
            events.append((id(self), "backend"))

        def enable_parallelism(self, config):
            assert events[-1] == (id(self), "install")
            assert (id(self), "backend") in events
            self._parallel_config = SimpleNamespace(context_parallel_config=config)
            events.append((id(self), "native"))

    class DualAdapter(PreparationAdapter):
        context_parallel_capability = Wan2_T2V_Adapter.context_parallel_capability
        context_parallel_component_names = ("transformer", "transformer_2")

        @property
        def trainable_component_names(self):
            return list(self.model_args.target_components)

        @classmethod
        def validate_context_parallel_module(cls, module, degree, tp_size=1):
            assert isinstance(module, CPLinear) and degree == 2
            events.append((id(module), "validate"))

        @classmethod
        def configure_context_parallel_module(cls, module):
            assert all(event == "validate" for _, event in events[: len(expected_roots)])
            events.append((id(module), "configure"))

        @classmethod
        def install_context_parallel_module(cls, module, config):
            events.append((id(module), "install"))
            super().install_context_parallel_module(module, config)

    accelerator = SimpleNamespace(
        num_processes=4,
        process_index=0,
        is_main_process=True,
        distributed_type=DistributedType.NO,
        state=SimpleNamespace(distributed_type=DistributedType.NO),
        parallelism_config=SimpleNamespace(
            dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=1
        ),
    )
    modules = {name: CPLinear() for name in DualAdapter.context_parallel_component_names}
    trainable = list(modules) if train_second else ["transformer"]
    for name, module in modules.items():
        if name not in trainable:
            module.requires_grad_(False)
        elif mode == "lora":
            modules[name] = get_peft_model(
                module, LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
            )
    adapter = DualAdapter(modules["transformer"], accelerator, mode)
    adapter.model_args.target_components = trainable
    adapter.target_module_map = {name: "all" for name in modules}
    adapter.component_runtime = PseudoPipelineRuntime(SimpleNamespace(), modules)
    adapter._components = adapter.component_runtime.override_components
    adapter.declare_component_variants(("policy", "critic"))
    expected_roots = {
        id(module) for module in adapter.component_variant_registry.bundle_members().values()
    }
    monkeypatch.setattr(
        "flow_factory.trainers.common.model_preparation.context_parallel_mesh", lambda *_: object()
    )
    owner = ModelPreparation(
        ModelLoadCoordinator(adapter, accelerator), ParallelismArguments(cp_size=2, dp_shard_size=2)
    )
    owner._apply_context_parallel(AttentionBackendAccelerator(backend="native"))
    expected_count = (4 if train_second else 3) if mode == "full" else 2
    assert len(owner._context_parallel.modules) == expected_count
    for module in owner._context_parallel.modules:
        assert [event for identity, event in events if identity == id(module)] == [
            "validate",
            "configure",
            "backend",
            "install",
            "native",
        ]
    if not train_second:
        frozen = modules["transformer_2"]
        assert frozen in owner._context_parallel.modules
        assert all(not parameter.requires_grad for parameter in frozen.parameters())
        assert all(
            record.component_name != "transformer_2"
            for variant in ("policy", "critic")
            for record in adapter.component_variant_registry.parameter_records(variant)
        )
    bundle = ModelBundle(adapter.component_variant_registry.bundle_members())
    bundle.context_parallel_contract = owner._context_parallel
    validate_prepared_context_parallel((bundle,), 2)
    before = list(events)
    with pytest.raises(ValueError, match="exactly once"):
        owner._apply_context_parallel(AttentionBackendAccelerator(backend="native"))
    assert events[len(before) :] == [(id(owner._context_parallel.modules[0]), "validate")]


def _mesh_worker(rank, store_path, result_dir, options):
    from torch.distributed.device_mesh import DeviceMesh

    from flow_factory.models.context_parallel import context_parallel_mesh

    tp = options["tp_size"]
    world = 4 * tp
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world
    )
    result = {
        "rank": rank,
        "execution_scope": "topology_only",
        "runtime": {
            "world_size": world,
            "rank": rank,
            "device": "cpu",
            "distributed_type": "PROCESS_GROUP",
            "process_group_backend": dist.get_backend(),
        },
    }
    try:
        dimensions = (2, 2, tp) if tp > 1 else (2, 2)
        names = ("dp_shard", "cp", "tp") if tp > 1 else ("dp_shard", "cp")
        mesh = DeviceMesh("cpu", torch.arange(world).reshape(dimensions), mesh_dim_names=names)
        accelerator = SimpleNamespace(
            num_processes=world,
            process_index=rank,
            torch_device_mesh=mesh,
            parallelism_config=SimpleNamespace(
                dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=tp
            ),
        )
        result["transports"] = {}
        for transport in ("ulysses", "ring"):
            actual = context_parallel_mesh(accelerator, transport)
            expected_peers = [
                rank % tp + rank // (2 * tp) * (2 * tp) + offset * tp for offset in range(2)
            ]
            assert actual[transport].mesh.tolist() == expected_peers
            other = "ring" if transport == "ulysses" else "ulysses"
            assert actual[other].size() == 1
            if tp > 1:
                assert torch.equal(actual["tensor"].mesh, mesh["tp"].mesh)
            result["transports"][transport] = actual[transport].mesh.tolist()
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [1, 2])
def test_native_transport_mesh_keeps_all_logical_peers(tmp_path, tp_size):
    rows = run_gloo_ranks(_mesh_worker, tmp_path, 4 * tp_size, {"tp_size": tp_size})
    assert all(row.get("status") == "pass" for row in rows), rows


def _ring_worker(rank, store_path, result_dir, options):
    from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from torch.distributed.device_mesh import DeviceMesh

    initialize_cuda(rank, store_path, world_size=2)
    result = {
        "rank": rank,
        "execution_scope": "attention_probe",
        "execution_capability": "native-ring-bf16-attention-v1",
        "cases": [],
    }
    try:
        mesh = DeviceMesh(
            "cuda", torch.arange(2).reshape(1, 2, 1), mesh_dim_names=("data", "ring", "ulysses")
        )
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(ring_degree=2, ulysses_degree=1)
        )
        config.setup(rank, 2, torch.device("cuda", rank), mesh=mesh)
        result["runtime"] = {
            "world_size": 2,
            "rank": rank,
            "device": "cuda",
            "distributed_type": "PROCESS_GROUP",
            "process_group_backend": dist.get_backend(),
        }
        for heads in (19, 15):
            for step in range(3):
                torch.manual_seed(421 + heads + step)
                inputs = [
                    torch.randn(1, 16, heads, 64, device="cuda", dtype=torch.bfloat16) / 3
                    for _ in range(3)
                ]
                upstream = torch.randn_like(inputs[0]) / 5
                # Independent FP64 global softmax; no distributed output or derivative
                # enters the reference. This fixture's explicit four-BF16-eps scale
                # budget accounts for low-precision outputs and two KV partial sums;
                # it does not replace the model/FP32 update-reference tolerances.
                references = [value.double().detach().requires_grad_() for value in inputs]
                query, key, value = [tensor.transpose(1, 2) for tensor in references]
                expected = ((query @ key.transpose(-2, -1) / 8).softmax(-1) @ value).transpose(1, 2)
                expected_gradients = torch.autograd.grad(
                    (expected * upstream.double()).sum(), references
                )
                local = [
                    tensor.chunk(2, dim=1)[rank].contiguous().detach().requires_grad_()
                    for tensor in inputs
                ]
                actual = dispatch_attention_fn(
                    *local, backend="_native_flash", parallel_config=config
                )
                with torch.no_grad():
                    replay = dispatch_attention_fn(
                        *local, backend="_native_flash", parallel_config=config
                    )
                assert torch.equal(actual, replay)
                gradients = torch.autograd.grad(
                    (actual.float() * upstream.chunk(2, dim=1)[rank].float()).sum(), local
                )
                comparisons = []
                for name, observed, reference in zip(
                    ("output", "query", "key", "value"),
                    (actual, *gradients),
                    (expected, *expected_gradients),
                ):
                    target = reference.chunk(2, dim=1)[rank].detach()
                    assert torch.isfinite(observed).all() and torch.count_nonzero(target)
                    budget = 4 * torch.finfo(torch.bfloat16).eps * target.abs().max()
                    residual = (observed.double() - target).abs()
                    assert bool((residual <= budget).all()), (
                        name,
                        residual.max().item(),
                        budget.item(),
                    )
                    comparisons.append(
                        {
                            "name": name,
                            "max_residual": residual.max().item(),
                            "budget": budget.item(),
                            "elements": target.numel(),
                        }
                    )
                # Omitting remote KV is a distinguishable error, not a finite-loss pass.
                q, k, v = [tensor.double().transpose(1, 2) for tensor in local]
                missing_remote = ((q @ k.transpose(-2, -1) / 8).softmax(-1) @ v).transpose(1, 2)
                target = expected.chunk(2, dim=1)[rank]
                threshold = 4 * torch.finfo(torch.bfloat16).eps * target.abs().max()
                assert (missing_remote - target).abs().max() > threshold
                result["cases"].append(
                    {
                        "heads": heads,
                        "step": step,
                        "same_layout_replay_exact": True,
                        "missing_remote_rejected": True,
                        "comparisons": comparisons,
                    }
                )
        result["elementwise_comparison"] = True
        result["nonzero_gradients"] = True
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_native_ring_forward_backward_and_replay(tmp_path):
    rows = run_gloo_ranks(_ring_worker, tmp_path, 2, {})
    assert all(row.get("status") == "pass" and len(row["cases"]) == 6 for row in rows), rows
