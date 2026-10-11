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

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from ...samples import (
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    StackedSampleBatch,
)
from ...scheduler import SDESchedulerOutput
from ...utils.base import filter_kwargs

_STORAGE_KEYS = {
    "trajectory",
    "timesteps",
    "all_latents",
    "latent_index_map",
    "log_probs",
    "log_prob_index_map",
}


# Trainer-owned batch fields: written by the feedback stage for the loss, never a
# model conditioning argument, so they must not reach adapters that accept ``**kwargs``.
_TRAINER_METADATA_KEYS = {"advantage"}


_BRIDGE_OWNED_BATCH_KEYS = _STORAGE_KEYS | _TRAINER_METADATA_KEYS


_STATE_OWNED_FORWARD_KEYS = {
    "t",
    "t_next",
    "latents",
    "next_latents",
    "compute_log_prob",
    "return_kwargs",
    "noise_level",
}


def build_forward_state_kwargs(
    adapter: Any,
    batch: StackedSampleBatch,
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve the forward arguments every ``forward_state`` override may pass on.

    Owns the whole argument boundary: the state, times, return fields and noise
    level belong to the bridge, and trajectory storage plus trainer metadata are
    read from the batch rather than forwarded. Batch conditioning is resolved
    first and explicit kwargs are layered on top; trainers drop the keys the batch
    already carries before calling, so batch values take precedence over
    training-argument defaults.
    """
    collisions = tuple(name for name in _STATE_OWNED_FORWARD_KEYS if name in kwargs)
    if collisions:
        raise ValueError(
            f"explicit forward_state kwargs collide with state-owned arguments {collisions}"
        )
    owned = tuple(sorted(name for name in kwargs if name in _BRIDGE_OWNED_BATCH_KEYS))
    if owned:
        raise ValueError(
            f"explicit forward_state kwargs collide with trainer-owned arguments {owned}; the "
            f"{type(adapter).__name__} bridge reads trajectory storage and trainer metadata from "
            "the batch and never forwards them to the model"
        )
    forward_kwargs = {
        key: value
        for key, value in batch.items()
        if key not in _BRIDGE_OWNED_BATCH_KEYS and key not in _STATE_OWNED_FORWARD_KEYS
    }
    forward_kwargs.update(kwargs)
    return forward_kwargs


def default_forward_state(
    adapter: Any,
    *,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
    compute_log_prob: bool,
    return_fields: Tuple[str, ...],
    noise_level: Optional[float],
    forward_kwargs: Mapping[str, Any],
) -> MultiModalStepOutput:
    expected_names = ("latent",)
    received = {
        "state": state.component_names,
        "timestep": tuple(times.timestep),
        "next_timestep": tuple(times.next_timestep),
        "next_state": None if next_state is None else next_state.component_names,
    }
    if (
        state.component_names != expected_names
        or tuple(times.timestep) != expected_names
        or tuple(times.next_timestep) != expected_names
        or (next_state is not None and next_state.component_names != expected_names)
    ):
        raise ValueError(
            f"expected exactly component order {expected_names} for default forward_state, "
            f"received {received}"
        )
    output = adapter.forward(
        t=times.timestep["latent"],
        t_next=times.next_timestep["latent"],
        latents=state.components["latent"],
        next_latents=None if next_state is None else next_state.components["latent"],
        compute_log_prob=compute_log_prob,
        return_kwargs=return_fields,
        noise_level=noise_level,
        **filter_kwargs(adapter.forward, **forward_kwargs),
    )
    if not isinstance(output, SDESchedulerOutput):
        raise TypeError(
            "expected adapter.forward to return SDESchedulerOutput in forward_state, "
            f"received {type(output).__name__}"
        )

    def wrap(value: Optional[torch.Tensor]) -> Optional[LatentState]:
        return None if value is None else LatentState({"latent": value})

    def wrap_statistic(value: Optional[torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        return None if value is None else {"latent": value}

    return MultiModalStepOutput(
        next_state=wrap(output.next_latents),
        next_state_mean=wrap(output.next_latents_mean),
        std_dev_t=wrap_statistic(output.std_dev_t),
        dt=wrap_statistic(output.dt),
        log_prob=output.log_prob,
        component_log_probs=wrap_statistic(output.log_prob),
        velocity=wrap(output.velocity),
    )
