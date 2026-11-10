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

"""Sharded target materialization with each adapter's context-parallel hooks (Gloo).

Production preparation detaches targets to meta, applies TP, installs the adapter's CP
hooks, then FSDP2-shards over ``dp_shard x cp`` before materializing. This compares that
order against the replicated path for every CP-capable adapter at DP2 x CP2 and
DP2 x CP2 x TP2: local shards, placements, buffers and installed CP configs must be
identical, and where the CP fixture supplies native inputs, so must one forward and backward.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks

CP_SIZE = 2


def _meshes(tp_size: int):
    """The production ``dp_shard x cp x tp`` mesh, its FSDP flattening and Diffusers CP mesh."""
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    mesh = init_device_mesh("cpu", (2, CP_SIZE, tp_size), mesh_dim_names=("dp_shard", "cp", "tp"))
    fsdp = mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
    if tp_size == 1:
        values, names = mesh.mesh.reshape(2, 1, CP_SIZE), ("data", "ring", "ulysses")
    else:
        values = mesh.mesh.reshape(2, CP_SIZE, tp_size).permute(0, 2, 1).unsqueeze(2)
        names = ("data", "tensor", "ring", "ulysses")
    return mesh, fsdp, DeviceMesh("cpu", values, mesh_dim_names=names)


def _first(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return torch.cat([item.flatten() for item in output if isinstance(item, torch.Tensor)])
    return output.sample if hasattr(output, "sample") else output


def _native_forward(case: str, replica: int) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
    """The CP fixture's native call for ``case``, or ``None`` when it supplies no inputs."""
    device = torch.device("cpu")
    if case == "wan_t2v":
        from test_model_preparation import wan_inputs

        def wan_t2v(module: torch.nn.Module) -> torch.Tensor:
            dtype = next(module.parameters()).dtype
            inputs = {key: value.to(dtype) for key, value in wan_inputs(replica, 0, device).items()}
            return _first(module(**inputs, return_dict=False))

        return wan_t2v
    if case == "wan_i2v":
        from wan_i2v_cp_fixture import model_inputs

        return lambda module: _first(module(**model_inputs(replica, 0, device), return_dict=False))
    if case in ("wan_t2v_dual", "wan_t2v_expanded", "wan_i2v_dual", "wan_flf2v"):
        from wan_remaining_cp_fixture import forward_inputs, native_velocity

        variant = case[len("wan_") :]
        return lambda module: native_velocity(module, variant, forward_inputs(variant, replica))
    if case.startswith("flux2_"):
        from flux2_cp_fixture import fixed_inputs, native_velocity

        variant = case[len("flux2_") :]
        return lambda module: native_velocity(module, fixed_inputs(variant, replica), variant)
    if case == "flux3_base":
        from flux3_cp_fixture import fixed_inputs, native_velocity

        return lambda module: native_velocity(module, fixed_inputs("base", replica), "base")
    if case == "qwen_image":
        from qwen_cp_fixture import model_inputs

        return lambda module: _first(module(**model_inputs(replica, 0, device)))
    if case.startswith("qwen_edit_"):
        from qwen_edit_cp_fixture import model_inputs

        return lambda module: _first(module(**model_inputs(replica, 0, device)))
    if case.startswith("ltx2"):
        from ltx2_cp_fixture import model_inputs

        ltx23 = case == "ltx23_i2av"
        return lambda module: _first(module(**model_inputs(ltx23, ltx23, replica, 0, device)))
    if case == "sensenova":
        from sensenova_cp_fixture import prefix_and_inputs

        from flow_factory.models.sensenova.modeling.neo_unify.modeling_qwen3 import (
            set_attn_backend,
        )

        # flash-attn rejects these FP32 probes; SDPA is the CPU-capable native kernel.
        set_attn_backend("sdpa")

        def sensenova(module: torch.nn.Module) -> torch.Tensor:
            inputs, _ = prefix_and_inputs(module, replica=replica)
            return module(**inputs)

        return sensenova
    return None


def _prepare(model, adapter, units, meshes, tp_size: int, sequence_parallel: bool, detach: bool):
    from diffusers.models._modeling_parallel import ContextParallelConfig
    from torch.distributed.fsdp import fully_shard

    from flow_factory.loading.materialization import detach_parameters_to_meta
    from flow_factory.models.tensor_parallel import (
        apply_tensor_parallel,
        peel_peft_wrapper,
        resolve_parallel_plan,
    )

    mesh, fsdp, cp_mesh = meshes
    source = detach_parameters_to_meta(model) if detach else None
    native = peel_peft_wrapper(model)
    if tp_size > 1:
        plan = resolve_parallel_plan(adapter, sequence_parallel=sequence_parallel)
        apply_tensor_parallel(native, mesh["tp"], plan)
    capability = adapter.context_parallel_capability
    adapter.validate_context_parallel_module(native, CP_SIZE, tp_size)
    adapter.configure_context_parallel_module(native)
    if hasattr(native, "set_attention_backend"):
        native.set_attention_backend(capability.attention_backend)
    ring, ulysses = capability.transport_degrees(CP_SIZE)
    adapter.install_context_parallel_module(
        native, ContextParallelConfig(ring_degree=ring, ulysses_degree=ulysses, mesh=cp_mesh)
    )
    if sequence_parallel and tp_size > 1:
        adapter.compose_context_parallel_sequence_boundaries(native)
    for unit in units:
        fully_shard(model.get_submodule(unit), mesh=fsdp)
    fully_shard(model, mesh=fsdp)
    if source is not None:
        source.materialize(model, torch.device("cpu"))
    return model


def _local(tensor: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    tp_size, case = options["tp_size"], options["case"]
    world_size = 2 * CP_SIZE * tp_size
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    result: Dict[str, Any] = {"rank": rank, "case": case, "tp_size": tp_size}
    try:
        from test_target_materialization import _fixture, _sharding_units

        from flow_factory.models.tensor_parallel import peel_peft_wrapper

        sequence_parallel = case.endswith("_sp")
        architecture = case.removesuffix("_sp")
        model, fixture_forward, adapter = _fixture(architecture)
        native = peel_peft_wrapper(model)
        prefix = next(name for name, item in model.named_modules() if item is native)
        units = [f"{prefix}.{unit}" for unit in _sharding_units(native)]
        meshes = _meshes(tp_size)
        replicated = _prepare(
            copy.deepcopy(model), adapter, units, meshes, tp_size, sequence_parallel, False
        )
        candidate = _prepare(
            copy.deepcopy(model), adapter, units, meshes, tp_size, sequence_parallel, True
        )

        expected = dict(replicated.named_parameters())
        result["differing"] = [
            name
            for name, parameter in candidate.named_parameters()
            if getattr(parameter, "placements", None) != getattr(expected[name], "placements", None)
            or not torch.equal(_local(parameter), _local(expected[name]))
        ]
        reference_buffers = dict(replicated.named_buffers())
        result["buffers_equal"] = all(
            torch.equal(buffer, reference_buffers[name])
            for name, buffer in candidate.named_buffers()
        )

        def configured(root):
            return [
                name
                for name, item in root.named_modules()
                if getattr(item, "_parallel_config", None) is not None
            ]

        result["cp_configured"] = configured(candidate)
        result["cp_configured_equal"] = configured(candidate) == configured(replicated)

        # Call through the FSDP root so its pre-forward unshards the root-owned parameters.
        forward = _native_forward(architecture, rank // (CP_SIZE * tp_size)) or fixture_forward
        result["forward_checked"] = forward is not None
        if forward is not None:
            output, reference = forward(candidate), forward(replicated)
            result["output_equal"] = bool(torch.equal(output, reference))
            output.float().square().mean().backward()
            reference.float().square().mean().backward()
            result["gradients_differing"] = [
                name
                for name, parameter in candidate.named_parameters()
                if (parameter.grad is None) != (expected[name].grad is None)
                or (
                    parameter.grad is not None
                    and not torch.equal(_local(parameter.grad), _local(expected[name].grad))
                )
            ]
            result["gradients_present"] = any(
                parameter.grad is not None for parameter in candidate.parameters()
            )
        result["status"] = "pass"
    except Exception as error:  # reported to the parent, which asserts on it
        import traceback

        result["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()[-3000:]}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


#: Fixtures whose head counts divide by CP2 x TP2 also run the TP composition.
_TP1_ONLY = ("flux1", "wan_t2v", "flux1_kontext")
_ARCHITECTURES = (
    "sd3_5",
    "z_image",
    "h3",
    "flux1",
    "qwen_image_21",
    "wan_t2v",
    "wan_i2v",
    "wan_t2v_dual",
    "wan_t2v_expanded",
    "wan_i2v_dual",
    "wan_flf2v",
    "flux1_kontext",
    "flux2_dev",
    "flux2_klein4b",
    "flux2_klein9b",
    "flux3_base",
    "qwen_image",
    "qwen_edit_2509",
    "qwen_edit_2511",
    "ltx20_t2av",
    "ltx23_i2av",
    "sensenova",
    "bagel_mot",
)
_CASES = [
    (case, tp_size)
    for case in _ARCHITECTURES
    for tp_size in (1, 2)
    if tp_size == 1 or case not in _TP1_ONLY
]


@requires_gloo
@pytest.mark.parametrize("case, tp_size", _CASES, ids=[f"{c}-tp{t}" for c, t in _CASES])
def test_context_parallel_meta_preparation_matches_replicated(
    tmp_path: Path, case: str, tp_size: int
) -> None:
    import importlib.util

    import diffusers

    if case == "flux3_base" and not hasattr(diffusers, "Flux3Transformer2DModel"):
        pytest.skip("this diffusers has no Flux3Transformer2DModel")
    if case.startswith("bagel_mot") and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("Bagel's vendored attention imports flash_attn at module scope")

    results = run_gloo_ranks(
        _worker,
        tmp_path,
        2 * CP_SIZE * tp_size,
        {"case": case, "tp_size": tp_size},
        timeout=900,
    )
    for result in results:
        assert "error" not in result, result["error"]
        assert result["differing"] == [], result
        assert result["buffers_equal"] and result["cp_configured_equal"], result
        assert result["cp_configured"], result
        if result["forward_checked"]:
            assert result["output_equal"] and result["gradients_present"], result
            assert result["gradients_differing"] == [], result
