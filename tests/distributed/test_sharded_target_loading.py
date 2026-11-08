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

"""Accelerate FSDP2 preparation of a meta bundle reproduces replicated training exactly.

Four NCCL ranks form DP2 x TP2. Tiny SD3.5 with default LoRA is prepared through
``FSDPBackendLoadRuntime`` twice: from loaded values, and from meta parameters
materialized after Accelerate's FSDP2 preparation. Local shards must be equal
after preparation and after one optimizer step, which also proves the optimizer
groups were rebuilt onto the materialized parameters. The efficient-loading
case gives only rank 0 the loaded values.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks

WORLD_SIZE = 4


def _local(parameter: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def _prepare(accelerator, module, *, sharded: bool, efficient: bool):
    from test_region_entry_parity import _ARCHITECTURES

    from flow_factory.loading.backend import FSDPBackendLoadRuntime
    from flow_factory.loading.domain import ComponentDescriptor, ComponentRole, LoadPlanner
    from flow_factory.models.model_bundle import ModelBundle
    from flow_factory.models.tensor_parallel import (
        apply_tensor_parallel,
        peel_peft_wrapper,
        resolve_parallel_plan,
    )
    from flow_factory.models.variants import ComponentVariantRegistry

    adapter = _ARCHITECTURES["sd3_5"]()[2]
    plan = LoadPlanner().build(
        [ComponentDescriptor(name="transformer", root="transformer", role=ComponentRole.TARGET)]
    )
    accelerator.state.fsdp_plugin.cpu_ram_efficient_loading = efficient
    runtime = FSDPBackendLoadRuntime(accelerator, plan, adapter)
    bundle = ModelBundle({"transformer": module})
    if sharded:
        runtime.detach_targets(bundle, ComponentVariantRegistry._canonical_prepared_parameter_name)
    apply_tensor_parallel(
        peel_peft_wrapper(module),
        accelerator.torch_device_mesh["tp"],
        resolve_parallel_plan(adapter, sequence_parallel=False),
    )
    optimizer = torch.optim.AdamW(
        [p for p in bundle.parameters() if p.requires_grad], lr=1e-2, foreach=False
    )
    torch.cuda.reset_peak_memory_stats()
    prepared, prepared_optimizer = runtime.prepare(bundle, optimizer)
    accelerator.state.fsdp_plugin.cpu_ram_efficient_loading = False
    return prepared, prepared_optimizer, torch.cuda.max_memory_allocated()


def _step(prepared, optimizer, forward) -> None:
    # Through the bundle root, whose FSDP2 pre-forward hook unshards root-unit parameters.
    output = forward(lambda **inputs: prepared("transformer", **inputs))
    output.float().square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE), ACCELERATE_USE_FSDP="true"
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        store=dist.FileStore(store_path, WORLD_SIZE),
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=300),
    )
    result: Dict[str, Any] = {"rank": rank}
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from test_region_entry_parity import _ARCHITECTURES, _with_lora

        from flow_factory.trainers.loader import Accelerator

        accelerator = Accelerator(
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["JointTransformerBlock"],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
            ),
            parallelism_config=ParallelismConfig(dp_shard_size=2, tp_size=2),
        )

        def model():
            module, _, adapter = _ARCHITECTURES["sd3_5"]()
            return _with_lora(module, adapter).float()

        def forward(module):
            generator = torch.Generator().manual_seed(11)
            return module(
                hidden_states=torch.randn(1, 4, 8, 8, generator=generator).cuda(),
                encoder_hidden_states=torch.randn(1, 5, 16, generator=generator).cuda(),
                pooled_projections=torch.randn(1, 8, generator=generator).cuda(),
                timestep=torch.tensor([500.0]).cuda(),
                return_dict=False,
            )[0]

        efficient = options["efficient"]
        reference, reference_optimizer, reference_peak = _prepare(
            accelerator, model(), sharded=False, efficient=False
        )
        candidate = model()
        if efficient and rank != 0:
            with torch.no_grad():
                for parameter in candidate.parameters():
                    parameter.zero_()
        candidate, candidate_optimizer, candidate_peak = _prepare(
            accelerator, candidate, sharded=True, efficient=efficient
        )

        def compare(stage: str) -> list:
            expected = dict(accelerator.unwrap_model(reference).named_parameters())
            return [
                f"{stage}:{name}"
                for name, parameter in accelerator.unwrap_model(candidate).named_parameters()
                if not torch.equal(_local(parameter), _local(expected[name]))
            ]

        differing = compare("prepared")
        _step(reference, reference_optimizer, forward)
        _step(candidate, candidate_optimizer, forward)
        differing += compare("stepped")
        groups = [len(group["params"]) for group in candidate_optimizer.param_groups]
        unique = len({id(p) for g in candidate_optimizer.param_groups for p in g["params"]})
        result.update(
            differing=differing[:10],
            groups=groups,
            unique=unique,
            reference_peak=reference_peak,
            candidate_peak=candidate_peak,
            status="pass",
        )
    except Exception as error:  # reported to the parent, which asserts on it
        import traceback

        result["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()[-2000:]}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE, reason="sharded target loading requires four GPUs"
)
@pytest.mark.parametrize("efficient", [False, True], ids=["replicated-source", "rank0-source"])
def test_meta_bundle_prepared_by_accelerate_trains_like_the_replicated_bundle(
    tmp_path: Path, efficient: bool
) -> None:
    results = run_gloo_ranks(_worker, tmp_path, WORLD_SIZE, {"efficient": efficient}, timeout=600)
    for result in results:
        assert "error" not in result, result["error"]
        assert result["differing"] == [], result
        assert sum(result["groups"]) == result["unique"], result
