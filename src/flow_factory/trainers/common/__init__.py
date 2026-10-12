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

"""Shared, algorithm-independent trainer primitives."""

from .dpo_objective import dpo_objective
from .forward_kwargs import (
    reference_forward_kwargs,
    replay_forward_kwargs,
    training_forward_kwargs,
)
from .pairwise_activation import pairwise_policy_activation_context
from .replay_batching import move_and_stack_samples
from .sample_prefetch import iter_prefetched_batches
from .state_validation import (
    require_component_sigmas,
    require_latent_state,
    require_velocity_state,
    state_batch_size,
)

__all__ = [
    "dpo_objective",
    "iter_prefetched_batches",
    "move_and_stack_samples",
    "pairwise_policy_activation_context",
    "reference_forward_kwargs",
    "replay_forward_kwargs",
    "require_component_sigmas",
    "require_latent_state",
    "require_velocity_state",
    "state_batch_size",
    "training_forward_kwargs",
]
