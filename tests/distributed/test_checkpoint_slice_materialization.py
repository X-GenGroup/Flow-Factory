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

"""Checkpoint-backed targets materialize the natively loaded run's exact local shards.

Four gloo ranks form DP2 x TP2. Each case loads a saved tiny transformer twice:
natively, in the adapter's loader mode, and checkpoint-backed. Both receive
LoRA from the same seed, which must give identical adapter weights because
neither construction consumes the global RNG. After TP and per-block FSDP2,
every rank's local shards must be equal, while the checkpoint-backed root
never held its base weights on the host.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks

WORLD_SIZE = 4


def _case(name: str):
    import sys

    loading_tests = str(Path(__file__).resolve().parents[1] / "loading")
    if loading_tests not in sys.path:
        sys.path.insert(0, loading_tests)
    from test_checkpoint_slices import _sd3, _wan

    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    return {
        "sd3_eager": (_sd3, SD3_5Adapter, False),
        "wan_meta": (_wan, Wan2_T2V_Adapter, True),
    }[name]


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from peft import LoraConfig, get_peft_model
        from test_target_materialization import _prepare, _sharding_units
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        from flow_factory.loading.checkpoint_slices import (
            CheckpointParameter,
            construct_from_checkpoint,
        )
        from flow_factory.loading.materialization import detach_parameters_to_meta
        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            peel_peft_wrapper,
            resolve_parallel_plan,
        )

        build, adapter, low_cpu_mem_usage = _case(options["case"])
        folder = Path(options["folder"])
        dtype = getattr(torch, options["dtype"])
        component_cls = type(build())
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        targets = adapter.default_target_modules.fget(object.__new__(adapter))

        def with_lora(module):
            torch.manual_seed(5)
            config = LoraConfig(r=2, lora_alpha=2, target_modules=targets)
            return get_peft_model(module, config)

        def tensor_parallel(module, tp_mesh):
            apply_tensor_parallel(
                peel_peft_wrapper(module),
                tp_mesh,
                resolve_parallel_plan(adapter, sequence_parallel=False),
            )

        native = with_lora(
            component_cls.from_pretrained(
                folder, torch_dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage
            )
        )
        backed, reason = construct_from_checkpoint(
            component_cls, folder, dtype, low_cpu_mem_usage=low_cpu_mem_usage
        )
        assert reason is None, reason
        backed = with_lora(backed)
        lora = {n: p for n, p in native.named_parameters() if "lora_" in n}
        backed_lora = {n: p for n, p in backed.named_parameters() if "lora_" in n}
        lora_equal = lora.keys() == backed_lora.keys() and all(
            torch.equal(backed_lora[n], v) for n, v in lora.items()
        )
        checkpoint_backed = sum(isinstance(p, CheckpointParameter) for p in backed.parameters())
        base_bytes = sum(
            p.untyped_storage().nbytes()
            for p in backed.parameters()
            if isinstance(p, CheckpointParameter)
        )

        units = [f"base_model.model.{unit}" for unit in _sharding_units(peel_peft_wrapper(native))]
        native = _prepare(native, mesh, tensor_parallel, units)
        source = detach_parameters_to_meta(backed)
        backed = _prepare(backed, mesh, tensor_parallel, units)
        report = source.materialize(backed, torch.device("cpu"))

        expected = dict(native.named_parameters())
        differing = [
            name
            for name, parameter in backed.named_parameters()
            if parameter.dtype != expected[name].dtype
            or not torch.equal(
                parameter.to_local() if isinstance(parameter, DTensor) else parameter,
                (
                    expected[name].to_local()
                    if isinstance(expected[name], DTensor)
                    else expected[name]
                ),
            )
        ]
        buffers = dict(native.named_buffers())
        buffers_equal = all(
            b.dtype == buffers[n].dtype and torch.equal(b, buffers[n])
            for n, b in backed.named_buffers()
        )
        result.update(
            lora_equal=lora_equal,
            checkpoint_backed=checkpoint_backed,
            base_bytes=base_bytes,
            differing=differing[:10],
            buffers_equal=buffers_equal,
            local_bytes=report.local_bytes,
        )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


_CASES = [
    ("sd3_eager", "float32", "float32"),
    ("sd3_eager", "bfloat16", "float32"),
    ("wan_meta", "float32", "float32"),
    ("wan_meta", "bfloat16", "bfloat16"),
]


@requires_gloo
@pytest.mark.parametrize("case, stored, dtype", _CASES, ids=[f"{c}-{s}-{d}" for c, s, d in _CASES])
def test_checkpoint_backed_targets_materialize_the_native_local_shards(
    tmp_path: Path, case: str, stored: str, dtype: str
) -> None:
    build = _case(case)[0]
    folder = tmp_path / "component"
    build().to(getattr(torch, stored)).save_pretrained(folder, safe_serialization=True)
    results = run_gloo_ranks(
        _worker,
        tmp_path / "run",
        WORLD_SIZE,
        {"case": case, "folder": str(folder), "dtype": dtype},
    )
    for result in results:
        assert "error" not in result, result["error"]
        assert result["lora_equal"], result
        assert result["checkpoint_backed"] > 0, result
        assert result["base_bytes"] <= 8 * result["checkpoint_backed"], result
        assert result["differing"] == [] and result["buffers_equal"], result
