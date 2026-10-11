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

# src/flow_factory/trainers/forward_process.py
"""Forward-process helpers shared by the decoupled trainers.

Decoupled algorithms (e.g. DiffusionNFT, AWM, DPO) train on a freshly noised terminal
state instead of a stored rollout transition, so they share one velocity-only
forward contract rather than the coupled replay contract of ``CoupledReplayRuntimeMixin``.
"""

from collections.abc import Mapping
from typing import Any

from ..samples import ComponentTimes, LatentState
from .common.forward_kwargs import training_forward_kwargs
from .common.state_validation import (
    require_component_sigmas,
    require_latent_state,
    require_velocity_state,
    state_batch_size,
)


def forward_velocity_state(
    trainer: Any,
    batch: Mapping[str, Any],
    state: LatentState,
    times: ComponentTimes,
    *,
    source: str,
    noise_level: float = 0.0,
    **overrides: Any,
) -> LatentState:
    """Predict velocity for a forward-noised state through the active parameters.

    The caller owns the autocast / ``no_grad`` / parameter-swap scope, so this
    helper never opens one; it only bridges the state into ``forward_state`` and
    validates the returned velocity.

    Args:
        trainer: Trainer supplying the adapter and training arguments.
        batch: Collated sample batch supplying conditioning arguments.
        state: Forward-noised latent state to evaluate.
        times: Component times for the noised state.
        source: Pass identifier reported by validation errors.
        noise_level: Scheduler noise-level override; decoupled training uses ``0.0``.
        **overrides: Explicit forward arguments taking precedence over training args.

    Returns:
        Predicted velocity state keyed by component.
    """
    forward_kwargs = training_forward_kwargs(trainer, batch)
    forward_kwargs.update(overrides)
    # Validate the batch rank before the forward: the velocity check below only
    # compares shapes, so an unbatched state would pass unnoticed if the adapter
    # mirrored it.
    state_batch_size(trainer, state, f"{source} forward state")
    output = trainer.adapter.forward_state(
        batch=batch,
        state=state,
        times=times,
        compute_log_prob=False,
        return_fields=("velocity",),
        noise_level=noise_level,
        **forward_kwargs,
    )
    return require_velocity_state(trainer, output, source, state)
