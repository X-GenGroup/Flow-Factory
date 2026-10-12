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

"""Structured transition replay shared by coupled policy trainers."""

from collections.abc import Mapping
from typing import Any

import torch

from ...samples import LatentState, MultiModalStepOutput, ReplayStep, StackedSampleBatch
from ..common.forward_kwargs import reference_forward_kwargs, replay_forward_kwargs

KL_SPACE_FIELDS: dict[str, tuple[str, str]] = {
    "v-based": ("velocity", "velocity"),
    "x-based": ("next_state_mean", "next_latents_mean"),
}
CANONICAL_RETURN_FIELDS: tuple[str, ...] = (
    "log_prob",
    "next_latents",
    "next_latents_mean",
    "std_dev_t",
    "dt",
    "velocity",
)


class CoupledReplayRuntimeMixin:
    """Own replay forwards, output validation, and reference-KL projections."""

    def _replay_forward_kwargs(self, batch: StackedSampleBatch) -> dict[str, Any]:
        """Return replay defaults while preserving batch-key precedence."""
        return replay_forward_kwargs(self, batch)

    def _effective_transition_std(
        self,
        component: str,
        std_dev_t: torch.Tensor,
        dt: torch.Tensor,
        *,
        context: str,
    ) -> torch.Tensor:
        """Return the x-space Gaussian standard deviation for one component."""
        dynamics_type = self.adapter.scheduler_group[component].dynamics_type
        if dynamics_type in ("Flow-SDE", "Dance-SDE"):
            scale = std_dev_t * torch.sqrt(-dt)
        elif dynamics_type == "CPS":
            scale = std_dev_t
        else:
            raise ValueError(
                f"{context} received component {component!r} dynamics_type "
                f"{dynamics_type!r}; expected one of ('Flow-SDE', 'Dance-SDE', 'CPS'). "
                "Coupled algorithms must not use ODE dynamics (see constraints #7)."
            )
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError(
                f"{context} expected component {component!r} to have a finite strictly positive "
                f"stochastic transition scale for dynamics_type={dynamics_type!r}, received "
                f"{scale.detach().cpu().tolist()}"
            )
        return scale

    def _replay_forward(
        self,
        batch: StackedSampleBatch,
        replay: ReplayStep,
        return_fields: tuple[str, ...],
    ) -> MultiModalStepOutput:
        """Replay one stored transition through the current policy."""
        return self.adapter.forward_state(
            batch=batch,
            state=replay.state,
            times=replay.times,
            next_state=replay.next_state,
            compute_log_prob=True,
            return_fields=return_fields,
            noise_level=self.adapter.scheduler.noise_level,
            **self._replay_forward_kwargs(batch),
        )

    def _reference_forward(
        self,
        batch: StackedSampleBatch,
        replay: ReplayStep,
        return_fields: tuple[str, ...],
        **overrides: Any,
    ) -> MultiModalStepOutput:
        """Replay the same transition through frozen reference parameters."""
        forward_kwargs = reference_forward_kwargs(self, batch, **overrides)
        with torch.no_grad(), self.adapter.use_ref_parameters():
            return self.adapter.forward_state(
                batch=batch,
                state=replay.state,
                times=replay.times,
                next_state=replay.next_state,
                compute_log_prob=False,
                return_fields=return_fields,
                noise_level=self.adapter.scheduler.noise_level,
                **forward_kwargs,
            )

    def _require_replay_log_prob(self, replay: ReplayStep, step_index: int) -> torch.Tensor:
        """Return the stored rollout joint log probability for one transition."""
        batch_size = self._replay_batch_size(replay)
        log_prob = replay.log_prob
        if isinstance(log_prob, torch.Tensor) and log_prob.shape == (batch_size,):
            return log_prob
        received = (
            tuple(log_prob.shape) if isinstance(log_prob, torch.Tensor) else type(log_prob).__name__
        )
        raise ValueError(
            f"expected stored rollout log_prob for {type(self).__name__} replay at "
            f"step_index={step_index} to be a tensor of shape (B,) with batch size "
            f"{batch_size}, received {received}; rerun sampling with compute_log_prob=True"
        )

    def _require_policy_log_prob(
        self,
        output: MultiModalStepOutput,
        step_index: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Return the current-policy joint log probability for the PPO ratio."""
        log_prob = output.log_prob
        if isinstance(log_prob, torch.Tensor) and log_prob.shape == (batch_size,):
            return log_prob
        received = (
            tuple(log_prob.shape) if isinstance(log_prob, torch.Tensor) else type(log_prob).__name__
        )
        raise ValueError(
            f"expected policy log_prob for {type(self).__name__} replay at "
            f"step_index={step_index} to be a tensor of shape (B,) with batch size "
            f"{batch_size}, received {received}; request 'log_prob' through return_fields "
            "and keep compute_log_prob=True"
        )

    def _replay_batch_size(self, replay: ReplayStep) -> int:
        """Return the replay batch size from the primary component state."""
        primary = self.adapter.trajectory_component_order[0]
        return replay.state.components[primary].shape[0]

    def _canonical_return_fields(self, fields: Any) -> tuple[str, ...]:
        """Order requested scheduler output fields deterministically."""
        requested = set(fields)
        unknown = tuple(sorted(requested.difference(CANONICAL_RETURN_FIELDS)))
        if unknown:
            raise ValueError(
                f"unknown return field {unknown[0]!r}; expected a subset of "
                f"{CANONICAL_RETURN_FIELDS}"
            )
        return tuple(name for name in CANONICAL_RETURN_FIELDS if name in requested)

    def _kl_space_fields(self, kl_space: str, argument_name: str) -> tuple[str, str]:
        """Resolve the output attribute and legacy request name for a KL space."""
        if kl_space not in KL_SPACE_FIELDS:
            raise ValueError(
                f"expected {argument_name} in {tuple(KL_SPACE_FIELDS)}, received {kl_space!r}"
            )
        return KL_SPACE_FIELDS[kl_space]

    def _require_output_state(
        self,
        output: MultiModalStepOutput,
        field: str,
        source: str,
    ) -> LatentState:
        """Return a required latent-state field in authoritative component order."""
        state = getattr(output, field)
        expected_names = self.adapter.trajectory_component_order
        if state is None:
            raise ValueError(
                f"expected {source} forward output field {field!r} in component order "
                f"{expected_names}, received None; request it through return_fields"
            )
        if state.component_names != expected_names:
            raise ValueError(
                f"expected {source} forward output field {field!r} in component order "
                f"{expected_names}, received {state.component_names}"
            )
        return state

    def _require_component_mapping(
        self,
        values: Mapping[str, torch.Tensor] | None,
        field: str,
        source: str,
    ) -> Mapping[str, torch.Tensor]:
        """Return a required per-component mapping in authoritative component order."""
        expected_names = self.adapter.trajectory_component_order
        if values is None:
            raise ValueError(
                f"expected {source} {field} for {type(self).__name__} replay in component "
                f"order {expected_names}, received None"
            )
        if tuple(values) != expected_names:
            raise ValueError(
                f"expected {source} {field} for {type(self).__name__} replay in component "
                f"order {expected_names}, received {tuple(values)}"
            )
        return values

    def _component_squared_error_elements(
        self,
        new_state: LatentState,
        old_state: LatentState,
        source: str,
    ) -> dict[str, torch.Tensor]:
        """Return raw per-element squared error for each component."""
        expected_names = self.adapter.trajectory_component_order
        if old_state.component_names != expected_names:
            raise ValueError(
                f"expected {source} state in component order {expected_names}, "
                f"received {old_state.component_names}"
            )
        return {
            name: (new_state.components[name] - old_state.components[name]) ** 2
            for name in expected_names
        }

    def _reference_kl_divergence(
        self,
        output: MultiModalStepOutput,
        ref_output: MultiModalStepOutput,
        replay: ReplayStep,
    ) -> torch.Tensor:
        """Return policy-vs-reference squared error in the configured KL space."""
        output_field, _ = self._kl_space_fields(self.training_args.kl_type, "kl_type")
        errors = self._component_squared_error_elements(
            self._require_output_state(output, output_field, "policy"),
            self._require_output_state(ref_output, output_field, "reference"),
            "reference",
        )
        return torch.mean(self.adapter.reduce_latent_values(errors, state=replay.state))
