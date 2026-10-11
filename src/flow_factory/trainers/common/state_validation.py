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

"""Validation helpers for structured trainer states and model outputs."""

from typing import Any

import torch

from ...samples import ComponentTimes, LatentState, MultiModalStepOutput


def require_component_sigmas(trainer: Any, times: ComponentTimes) -> dict[str, torch.Tensor]:
    """Return per-component sigmas in authoritative component order."""
    expected_names = trainer.adapter.trajectory_component_order
    sigmas = times.sigma
    if sigmas is None or tuple(sigmas) != expected_names:
        received = None if sigmas is None else tuple(sigmas)
        raise ValueError(
            f"expected component sigmas in order {expected_names} for "
            f"{type(trainer).__name__}, received {received}"
        )
    return dict(sigmas)


def require_latent_state(trainer: Any, state: Any, identifier: str) -> LatentState:
    """Return a latent state validated against the declared component order."""
    expected_names = trainer.adapter.trajectory_component_order
    if not isinstance(state, LatentState):
        raise TypeError(
            f"expected LatentState for {identifier} on {type(trainer).__name__}, "
            f"received {type(state).__name__}"
        )
    if state.component_names != expected_names:
        raise ValueError(
            f"expected {identifier} on {type(trainer).__name__} in component order "
            f"{expected_names}, received {state.component_names}"
        )
    return state


def state_batch_size(trainer: Any, state: LatentState, identifier: str = "latent state") -> int:
    """Return the batch size of a latent state's primary component."""
    primary = trainer.adapter.trajectory_component_order[0]
    component = require_latent_state(trainer, state, identifier).components[primary]
    if component.ndim < 2:
        raise ValueError(
            f"expected {identifier} component {primary!r} on {type(trainer).__name__} to be a "
            f"batched tensor with a leading batch dimension and shape (B, ...), received shape "
            f"{tuple(component.shape)}"
        )
    return component.shape[0]


def require_velocity_state(
    trainer: Any,
    output: MultiModalStepOutput,
    source: str,
    expected_state: LatentState,
) -> LatentState:
    """Return a velocity state validated against the state it was predicted for.

    Shape and device must match the input state exactly. Dtype may differ from
    the stored latent dtype, but every velocity component must use one common
    floating-point dtype.
    """
    expected_names = trainer.adapter.trajectory_component_order
    reference = require_latent_state(trainer, expected_state, f"{source} forward state")
    velocity = output.velocity
    if velocity is None or velocity.component_names != expected_names:
        received = None if velocity is None else velocity.component_names
        raise ValueError(
            f"expected {source} velocity for {type(trainer).__name__} in component order "
            f"{expected_names}, received {received}; request 'velocity' through return_fields"
        )
    primary_name = expected_names[0]
    primary_dtype = velocity.components[primary_name].dtype
    for name in expected_names:
        component = velocity.components[name]
        component_reference = reference.components[name]
        if component.shape != component_reference.shape:
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} to "
                f"match its forward state shape {tuple(component_reference.shape)}, received "
                f"shape {tuple(component.shape)}"
            )
        if component.device != component_reference.device:
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} on "
                f"the forward state device {component_reference.device}, received "
                f"{component.device}"
            )
        if not component.is_floating_point():
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} to "
                f"be a floating point tensor, received {component.dtype}"
            )
        if component.dtype != primary_dtype:
            raise ValueError(
                f"expected {source} velocity component {name!r} for {type(trainer).__name__} to "
                f"share the dtype of component {primary_name!r} ({primary_dtype}), received "
                f"{component.dtype}"
            )
    return velocity
