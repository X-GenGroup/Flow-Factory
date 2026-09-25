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
