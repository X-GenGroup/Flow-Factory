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

"""FSDP2 preparation for explicitly admitted frozen auxiliary modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method

from .component_execution import OptimizationGradients

if TYPE_CHECKING:
    from ..trainers.abc import BaseTrainer


def data_mesh(device_type: str) -> DeviceMesh:
    """Reuse the world group for the admitted data-only topology."""
    return DeviceMesh.from_group(dist.group.WORLD, device_type, mesh_dim_names=("data",))


def shard_frozen_module(
    module: nn.Module, mesh: DeviceMesh, *, methods: tuple[str, ...] = ()
) -> None:
    """Shard one frozen root and register any native calls that bypass forward."""
    if isinstance(module, FSDPModule):
        raise ValueError(f"auxiliary {type(module).__name__} is already FSDP2 prepared")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"FSDP2 auxiliary {type(module).__name__} must have frozen parameters")
    if any(not callable(getattr(module, name, None)) for name in methods):
        raise ValueError(f"auxiliary {type(module).__name__} lacks configured methods {methods}")
    module.eval()
    fully_shard(module, mesh=mesh, reshard_after_forward=True)
    for name in methods:
        register_fsdp_forward_method(module, name)


def shard_pickscore(model: nn.Module, mesh: DeviceMesh) -> None:
    """Shard CLIP's towers and projections, leaving its scalar score scale replicated."""
    for module in (
        model.text_model,
        model.text_projection,
        model.vision_model,
        model.visual_projection,
    ):
        shard_frozen_module(module, mesh)


def prepare_reward_gradient_auxiliaries(trainer: BaseTrainer) -> tuple[str, ...]:
    """Prepare the image decoder and synchronous scorer after their native load."""
    plan = getattr(trainer.adapter, "component_execution_plan", None)
    if plan is None:
        return ()
    model_roots = tuple(
        name
        for name, execution in plan.model.items()
        if execution.optimization_gradients is OptimizationGradients.INPUT
        and execution.policy is not None
        and execution.policy.placement == "fsdp2"
    )
    scorer_names = (
        tuple(
            name
            for name, config in trainer.reward_loader.get_reward_configs("train").items()
            if plan.rewards[config.get_identity_key()].policy.placement == "fsdp2"
        )
        if trainer.reward_models
        else ()
    )
    if not model_roots and not scorer_names:
        return ()

    mesh = data_mesh(trainer.accelerator.device.type)
    for name in model_roots:
        decoder = trainer.adapter._require_component(name)
        shard_frozen_module(decoder, mesh, methods=("decode",))
        trainer.adapter.set_component(name, decoder)

    prepared = set()
    for name in scorer_names:
        model = trainer.reward_models[name].model
        if id(model) not in prepared:
            shard_pickscore(model, mesh)
            prepared.add(id(model))
    return tuple(f"model.{name}" for name in model_roots) + tuple(
        f"rewards.{name}" for name in scorer_names
    )
