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
from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class ContextParallelCapability:
    """Identify the layout and output-gradient semantics verified by an adapter."""

    layout: Literal["wan_dense_video", "flux_joint_image_text", "qwen_joint_image_text"]
    transport: Literal["ulysses"] = "ulysses"
    output_gradient: Literal["local_slice"] = "local_slice"
    version: int = 1
    tensor_parallel: bool = False
    required_precision_policy: str | None = None


@dataclass(frozen=True)
class ContextParallelBoundary:
    """Retain one native CP callback installed outside a TP/SP boundary."""

    module: nn.Module
    phase: Literal["input", "output"]
    callback: Callable[..., Any]
    handle: RemovableHandle


@dataclass(frozen=True)
class PreparedContextParallel:
    """Bind installed Diffusers hooks to exactly one bundle and CP degree."""

    capability: ContextParallelCapability
    degree: int
    modules: tuple[nn.Module, ...]
    boundaries: tuple[ContextParallelBoundary, ...] = ()


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
    for boundary in contract.boundaries:
        hooks = (
            boundary.module._forward_pre_hooks
            if boundary.phase == "input"
            else boundary.module._forward_hooks
        )
        if boundary.module not in members or hooks.get(boundary.handle.id) is not boundary.callback:
            raise ValueError("context parallel composition lost an owned sequence boundary")
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
    if topology.dp_replicate_size != 1:
        raise ValueError(f"Ulysses requires dp_replicate_size=1: {topology.dimensions}")
    mesh = accelerator.torch_device_mesh
    if mesh is None:
        raise ValueError("context parallel preparation requires an Accelerator device mesh")
    if topology.tp_size == 1:
        values = mesh.mesh.reshape(topology.dp_shard_size, 1, topology.cp_size)
        names = ("data", "ring", "ulysses")
    else:
        values = (
            mesh.mesh.reshape(topology.dp_shard_size, topology.cp_size, topology.tp_size)
            .permute(0, 2, 1)
            .unsqueeze(2)
        )
        names = ("data", "tensor", "ring", "ulysses")
    result = DeviceMesh(mesh.device_type, values, mesh_dim_names=names)
    if not torch.equal(result["ulysses"].mesh, mesh["cp"].mesh):
        raise ValueError("Diffusers Ulysses peers differ from the configured CP peers")
    if topology.tp_size > 1 and not torch.equal(result["tensor"].mesh, mesh["tp"].mesh):
        raise ValueError("Diffusers mesh changed the configured TP peers")
    return result


def compose_wan_sequence_boundaries(module: nn.Module) -> tuple[ContextParallelBoundary, ...]:
    """Run native CP split before TP scatter, and native CP gather after TP gather.

    Diffusers wraps ``forward`` while PyTorch TP/SP uses module hooks. Without
    this ordering bridge, SP scatters before CP and restores tokens after CP,
    permuting video positions relative to the native CP rotary slices. Only the
    two boundary callbacks move; Diffusers still owns slicing, gather autograd,
    rotary positions and attention communication.
    """
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook

    entry, exit_module = module.blocks[0], module.proj_out
    split_registry = getattr(entry, "_diffusers_hook", None)
    gather_registry = getattr(exit_module, "_diffusers_hook", None)
    split = split_registry.get_hook("cp_input---blocks.0") if split_registry else None
    gather = gather_registry.get_hook("cp_output---proj_out") if gather_registry else None
    if not isinstance(split, ContextParallelSplitHook) or not isinstance(
        gather, ContextParallelGatherHook
    ):
        raise ValueError("Wan TP/SP+CP requires the qualified native input/output hooks")

    def split_input(target: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        return split.pre_forward(target, *args, **kwargs)

    def gather_output(target: nn.Module, args: tuple, output: Any) -> Any:
        return gather.post_forward(target, output)

    split_registry.remove_hook("cp_input---blocks.0", recurse=False)
    gather_registry.remove_hook("cp_output---proj_out", recurse=False)
    return (
        ContextParallelBoundary(
            entry,
            "input",
            split_input,
            entry.register_forward_pre_hook(split_input, with_kwargs=True, prepend=True),
        ),
        ContextParallelBoundary(
            exit_module,
            "output",
            gather_output,
            exit_module.register_forward_hook(gather_output),
        ),
    )
