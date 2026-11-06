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

"""Declared Diffusers Ulysses capability and its one-shot prepared-root bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn


@dataclass(frozen=True)
class ContextParallelCapability:
    """Identify the layout and output-gradient semantics verified by an adapter."""

    layout: Literal["wan_dense_video"]
    transport: Literal["ulysses"] = "ulysses"
    output_gradient: Literal["local_slice"] = "local_slice"
    version: int = 1


@dataclass(frozen=True)
class PreparedContextParallel:
    """Bind installed Diffusers hooks to exactly one bundle and CP degree."""

    capability: ContextParallelCapability
    degree: int
    modules: tuple[nn.Module, ...]


def validate_prepared_context_parallel(objects: tuple[Any, ...], degree: int) -> None:
    """Refuse the Accelerator bridge unless every model belongs to the prepared root."""
    from .model_bundle import ModelBundle

    models = [item for item in objects if isinstance(item, nn.Module)]
    if len(models) != 1 or not isinstance(models[0], ModelBundle):
        raise ValueError("context parallel preparation requires one Flow-Factory ModelBundle")
    contract = getattr(models[0], "context_parallel_contract", None)
    if not isinstance(contract, PreparedContextParallel) or contract.degree != degree:
        raise ValueError(
            f"context parallel cp_size={degree} requires hooks installed by ModelPreparation"
        )
    members = set(models[0].modules())
    if not contract.modules or any(module not in members for module in contract.modules):
        raise ValueError("context parallel hooks must belong to the prepared bundle")
    for module in contract.modules:
        config = getattr(module, "_parallel_config", None)
        cp = getattr(config, "context_parallel_config", None)
        if cp is None or cp.ulysses_degree != degree or cp.ring_degree != 1:
            raise ValueError(
                "context parallel preparation found a missing or changed Diffusers plan"
            )


def context_parallel_mesh(accelerator: Any) -> Any:
    """Map the existing CP peers onto Diffusers' explicit singleton Ring dimension."""
    from torch.distributed.device_mesh import DeviceMesh

    from ..utils.dist import parallel_topology

    topology = parallel_topology(accelerator)
    if topology.tp_size != 1 or topology.dp_replicate_size != 1:
        raise ValueError(f"Ulysses requires TP=1 and dp_replicate_size=1: {topology.dimensions}")
    mesh = accelerator.torch_device_mesh
    if mesh is None:
        raise ValueError("context parallel preparation requires an Accelerator device mesh")
    result = DeviceMesh(
        mesh.device_type,
        mesh.mesh.reshape(topology.dp_shard_size, 1, topology.cp_size),
        mesh_dim_names=("data", "ring", "ulysses"),
    )
    if not torch.equal(result["ulysses"].mesh, mesh["cp"].mesh):
        raise ValueError("Diffusers Ulysses peers differ from the configured CP peers")
    return result
