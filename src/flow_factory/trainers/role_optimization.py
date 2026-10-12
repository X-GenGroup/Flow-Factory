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

"""Coordinate disjoint model-role updates through one physical optimizer."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, cast

import torch
from accelerate import Accelerator, DistributedType

# Roles are the trainer's vocabulary. The model layer only knows component
# variants under caller-chosen names; the mapping from a role to a variant is
# made here, by the algorithm that owns the meaning of those names.
RoleName = str
_STATE_KEYS = {"version", "role_steps", "optimizer_group_roles", "active_phase"}
_STATE_VERSION = 3


def _validate_positive_float(value: object, field_name: str, role_name: RoleName) -> None:
    """Validate one finite positive optimizer value."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"expected numeric {field_name} for role {role_name!r}, "
            f"received {type(value).__name__}: {value!r}"
        )
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(
            f"expected finite {field_name} > 0 for role {role_name!r}, received {value!r}"
        )


@dataclass(frozen=True)
class RoleOptimizerConfig:
    """Store one role's clip norm, update cadence and moment hyperparameters.

    The moment fields describe AdamW directly; a Muon role fills them from its
    fallback settings, which govern the AdamW half of its split.
    """

    role_name: RoleName
    learning_rate: float
    adam_betas: Tuple[float, float]
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: float
    update_frequency: int = 1

    def __post_init__(self) -> None:
        """Validate role-local optimizer configuration."""
        if not isinstance(self.role_name, str) or not self.role_name:
            raise ValueError(
                "expected a non-empty string role name, " f"received {self.role_name!r}"
            )
        _validate_positive_float(self.learning_rate, "learning_rate", self.role_name)
        if (
            not isinstance(self.adam_betas, tuple)
            or len(self.adam_betas) != 2
            or any(
                not isinstance(beta, (int, float))
                or isinstance(beta, bool)
                or not math.isfinite(float(beta))
                or beta < 0
                or beta >= 1
                for beta in self.adam_betas
            )
        ):
            raise ValueError(
                "expected adam_betas as a two-item tuple with values in [0, 1) "
                f"for role {self.role_name!r}, received {self.adam_betas!r}"
            )
        if (
            not isinstance(self.adam_weight_decay, (int, float))
            or isinstance(self.adam_weight_decay, bool)
            or not math.isfinite(float(self.adam_weight_decay))
            or self.adam_weight_decay < 0
        ):
            raise ValueError(
                "expected finite adam_weight_decay >= 0 "
                f"for role {self.role_name!r}, received {self.adam_weight_decay!r}"
            )
        _validate_positive_float(self.adam_epsilon, "adam_epsilon", self.role_name)
        _validate_positive_float(self.max_grad_norm, "max_grad_norm", self.role_name)
        if (
            not isinstance(self.update_frequency, int)
            or isinstance(self.update_frequency, bool)
            or self.update_frequency < 1
        ):
            raise ValueError(
                "expected update_frequency >= 1 as an int "
                f"for role {self.role_name!r}, received {self.update_frequency!r}"
            )


@dataclass
class OptimizationRole:
    """Store one role's optimizer ownership and local update state."""

    config: RoleOptimizerConfig
    parameters: Tuple[torch.Tensor, ...]
    optimizer_group_ids: Tuple[int, ...]
    step: int = 0
    scheduler: Optional[Any] = None
    last_grad_norm: Optional[float] = None


@dataclass(frozen=True)
class RolePhase:
    """Declare one role phase and its repetition count."""

    role_name: RoleName
    repeats: int = 1

    def __post_init__(self) -> None:
        """Validate one update-plan phase."""
        if not isinstance(self.role_name, str) or not self.role_name:
            raise ValueError(
                "expected a non-empty string role name, " f"received {self.role_name!r}"
            )
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool) or self.repeats < 1:
            raise ValueError(
                f"expected repeats >= 1 as an int for role {self.role_name!r}, "
                f"received {self.repeats!r}"
            )


@dataclass(frozen=True)
class RoleUpdatePlan:
    """Store an ordered immutable sequence of role phases."""

    phases: Tuple[RolePhase, ...]

    def __post_init__(self) -> None:
        """Validate the ordered phase sequence."""
        if not isinstance(self.phases, tuple) or not self.phases:
            raise ValueError(
                "expected phases as a non-empty tuple of RolePhase values, "
                f"received {type(self.phases).__name__}: {self.phases!r}"
            )
        invalid = tuple(
            (index, type(phase).__name__, phase)
            for index, phase in enumerate(self.phases)
            if not isinstance(phase, RolePhase)
        )
        if invalid:
            raise TypeError(
                "expected every update-plan phase to be RolePhase, "
                f"received invalid entries {invalid!r}"
            )


class RoleOptimizationCoordinator:
    """Run exclusive sequential role phases on one physical optimizer.

    Each ``phase(role)`` window performs exactly one optimizer step. Inactive
    role gradients must remain ``None``. Gradient accumulation, if used, stays
    inside one role and never mixes roles in the same window.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        model_bundle: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        roles: Mapping[RoleName, OptimizationRole],
    ) -> None:
        self.accelerator = accelerator
        self.model_bundle = model_bundle
        self.optimizer = optimizer
        self.roles = dict(roles)
        self._optimizer_group_roles = self._validate_ownership()
        self._active_role_name: Optional[RoleName] = None
        self._microbatch_open = False
        self._microbatch_finished = False
        self._microbatch_backward_count = 0
        self._phase_step_count = 0

    @property
    def active_role_name(self) -> RoleName:
        """Return the role owning the currently open phase."""
        if self._active_role_name is None:
            raise RuntimeError("expected an open role phase, received no active phase")
        return self._active_role_name

    @contextmanager
    def phase(self, role_name: RoleName) -> Iterator[OptimizationRole]:
        """Open one complete role-local accumulation window.

        Args:
            role_name: Trainable role to update.

        Yields:
            The active role ownership record.
        """
        if role_name not in self.roles:
            raise KeyError(f"expected role_name in {tuple(self.roles)!r}, received {role_name!r}")
        if self._active_role_name is not None:
            raise RuntimeError(
                f"cannot enter role phase {role_name!r}: open phase "
                f"{self._active_role_name!r} must finish first"
            )
        stale_gradients = self._parameters_with_grad()
        if stale_gradients:
            raise RuntimeError(
                "role phase entry expected every optimizer parameter grad to be None, "
                f"received gradients for {stale_gradients!r}"
            )

        self._active_role_name = role_name
        self._phase_step_count = 0
        body_failed = False
        try:
            yield self.roles[role_name]
        except BaseException:
            body_failed = True
            raise
        finally:
            try:
                if not body_failed:
                    if self._microbatch_open:
                        raise RuntimeError(
                            f"role phase exit for {role_name!r} expected no open microbatch"
                        )
                    if self._phase_step_count != 1:
                        raise RuntimeError(
                            f"role phase exit for {role_name!r} expected exactly one "
                            f"optimizer step, received {self._phase_step_count}"
                        )
                    uncleared_gradients = self._parameters_with_grad()
                    if uncleared_gradients:
                        raise RuntimeError(
                            f"role phase exit for {role_name!r} expected all gradients "
                            f"cleared, received gradients for {uncleared_gradients!r}"
                        )
            finally:
                self._active_role_name = None
                self._phase_step_count = 0

    @contextmanager
    def microbatch(self) -> Iterator[OptimizationRole]:
        """Wrap one replay unit in Accelerator's accumulation context.

        Yields:
            The active role ownership record.
        """
        if self._active_role_name is None:
            raise RuntimeError(
                "cannot enter microbatch without an open phase; "
                "expected phase(role_name) to be entered first"
            )
        role_name = self._active_role_name
        if self._microbatch_open:
            raise RuntimeError(f"cannot enter nested microbatch for open role phase {role_name!r}")
        if self._phase_step_count:
            raise RuntimeError(
                f"extra sync/microbatch for role {role_name!r}: phase already stepped"
            )

        role = self.roles[role_name]
        with self.accelerator.accumulate(self.model_bundle):
            self._microbatch_open = True
            self._microbatch_finished = False
            self._microbatch_backward_count = 0
            body_failed = False
            try:
                yield role
            except BaseException:
                body_failed = True
                raise
            finally:
                try:
                    if not body_failed and not self._microbatch_finished:
                        raise RuntimeError(
                            f"microbatch exit for role {role_name!r} expected exactly one "
                            "finish_microbatch() call, received 0"
                        )
                finally:
                    self._microbatch_open = False
                    self._microbatch_finished = False
                    self._microbatch_backward_count = 0

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate one role-local loss inside an open microbatch.

        Args:
            loss: Scalar differentiable loss.
        """
        role_name = self.active_role_name
        if not self._microbatch_open:
            raise RuntimeError(f"backward for role {role_name!r} expected an open microbatch")
        if self._microbatch_finished:
            raise RuntimeError(
                f"backward for role {role_name!r} cannot run after finish_microbatch()"
            )
        if self._microbatch_backward_count:
            raise RuntimeError(
                f"backward already called for role {role_name!r} microbatch; "
                f"expected exactly one call, received {self._microbatch_backward_count + 1}"
            )
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"expected torch.Tensor loss for role {role_name!r}, "
                f"received {type(loss).__name__}: {loss!r}"
            )
        if loss.numel() != 1:
            raise ValueError(
                f"expected scalar loss for role {role_name!r}, "
                f"received shape {tuple(loss.shape)!r}"
            )
        # DeepSpeed performs the optimizer step from inside `backward()` at the GAS
        # boundary (`DeepSpeedOptimizerWrapper.step()` is intentionally a no-op), so
        # role isolation has to cover this call as well as the explicit step below.
        role = self.roles[role_name]
        with self._active_optimizer_groups_only(role):
            self.accelerator.backward(loss)
        self._microbatch_backward_count = 1

    @contextmanager
    def _active_optimizer_groups_only(self, role: OptimizationRole) -> Iterator[None]:
        """Hide inactive parameter groups for one physical optimizer step.

        Native AdamW skips parameters whose gradients are ``None``. DeepSpeed ZeRO-2
        materializes partitioned zero gradients instead, so stepping the shared
        optimizer also advances inactive roles' Adam state and mutates parameters that
        already have momentum or weight decay. Other backends temporarily empty the
        inactive groups; DeepSpeed instead skips them in ``_optimizer_step``. Either way
        every backend gets the role-exclusive step this coordinator promises without
        changing group order, hyperparameters, or checkpoint layout.

        Args:
            role: Active role whose optimizer groups remain populated.

        Yields:
            Control while ``step()`` can update only the active role's parameters.
        """
        active_group_ids = set(role.optimizer_group_ids)
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            # Accelerate wraps DeepSpeedZeroOptimizer, whose step ignores its public
            # param_groups and iterates internal `bit16_groups`. `_optimizer_step(i)` is
            # the narrow seam that advances one group's AdamW parameters and state; all
            # reduction, partition cleanup, and all-gather remain in DeepSpeed.step().
            zero_optimizer = getattr(self.optimizer, "optimizer", None)
            optimizer_step = getattr(zero_optimizer, "_optimizer_step", None)
            if not callable(optimizer_step):
                raise RuntimeError(
                    "DeepSpeed multi-role optimization requires a callable "
                    "DeepSpeedZeroOptimizer._optimizer_step(group_no) to isolate active "
                    f"role {role.config.role_name!r}; received optimizer wrapper "
                    f"{type(self.optimizer).__name__} with inner "
                    f"{type(zero_optimizer).__name__}. This DeepSpeed version cannot "
                    "guarantee inactive roles remain unchanged."
                )

            def active_only_optimizer_step(group_no: int) -> None:
                if group_no in active_group_ids:
                    optimizer_step(group_no)

            zero_optimizer._optimizer_step = active_only_optimizer_step
            try:
                yield
            finally:
                zero_optimizer._optimizer_step = optimizer_step
            return

        hidden: Dict[int, Any] = {}
        for group_id, group in enumerate(self.optimizer.param_groups):
            if group_id in active_group_ids:
                continue
            hidden[group_id] = group["params"]
            group["params"] = []
        try:
            yield
        finally:
            for group_id, parameters in hidden.items():
                self.optimizer.param_groups[group_id]["params"] = parameters

    def finish_microbatch(self) -> bool:
        """Finish one replay unit and step only at a valid sync boundary.

        Returns:
            Whether the prepared optimizer applied a parameter update. Returns
            ``False`` for non-sync microbatches and mixed-precision overflow skips.
        """
        # Both role-gradient assertions below read ``parameter.grad``; see
        # ``_gradients_reach_parameters`` for why that is not universal.
        role_name = self.active_role_name
        if not self._microbatch_open:
            raise RuntimeError(
                f"finish_microbatch for role {role_name!r} expected an open microbatch"
            )
        if self._microbatch_finished:
            raise RuntimeError(
                f"finish_microbatch already called for role {role_name!r} microbatch"
            )
        if self._microbatch_backward_count != 1:
            raise RuntimeError(
                f"finish_microbatch for role {role_name!r} expected exactly one backward "
                f"call, received {self._microbatch_backward_count}"
            )
        self._microbatch_finished = True
        if not self.accelerator.sync_gradients:
            return False
        if self._phase_step_count:
            raise RuntimeError(
                f"extra sync for role {role_name!r}: phase already stepped "
                f"{self._phase_step_count} time(s)"
            )

        role = self.roles[role_name]
        if self._gradients_reach_parameters():
            if not any(parameter.grad is not None for parameter in role.parameters):
                raise RuntimeError(
                    f"active role {role_name!r} expected at least one gradient at the sync "
                    f"boundary, received none. Backend {self.accelerator.distributed_type}; "
                    f"{self._role_ownership_report()}"
                )
            inactive_gradients = tuple(
                other_name
                for other_name, other_role in self.roles.items()
                if other_name != role_name
                and any(parameter.grad is not None for parameter in other_role.parameters)
            )
            if inactive_gradients:
                raise RuntimeError(
                    f"inactive role gradients for {inactive_gradients!r} while role "
                    f"{role_name!r} is active; expected every inactive gradient to be None"
                )

        # Kept so a caller can report it: a role whose gradient norm collapses or spikes
        # is the first sign that its objective went wrong, and the losses alone do not
        # show it.
        clip_parameters = (
            self.model_bundle.parameters() if self._uses_fsdp1() else role.parameters
        )
        grad_norm = self.accelerator.clip_grad_norm_(
            clip_parameters,
            role.config.max_grad_norm,
        )
        role.last_grad_norm = None if grad_norm is None else float(grad_norm)
        with self._active_optimizer_groups_only(role):
            self.optimizer.step()
        step_was_skipped = getattr(self.optimizer, "step_was_skipped", False)
        self.optimizer.zero_grad(set_to_none=True)
        self._phase_step_count = 1
        if not isinstance(step_was_skipped, bool):
            raise TypeError(
                "expected prepared optimizer step_was_skipped to be bool, "
                f"received {type(step_was_skipped).__name__}: {step_was_skipped!r}"
            )
        if step_was_skipped:
            return False
        role.step += 1
        if role.scheduler is not None:
            role.scheduler.step()
        return True

    def _role_ownership_report(self) -> str:
        """Describe what every role owns right now, for a gradient failure to quote.

        A role that reaches its sync boundary without gradients has either been handed
        the wrong parameters or had them frozen since its forward, and telling those two
        apart from the outside is otherwise a guessing game.

        Returns:
            One clause per role, plus which parameter identities the roles share.
        """
        clauses = []
        for name, role in self.roles.items():
            total = len(role.parameters)
            trainable = sum(1 for parameter in role.parameters if parameter.requires_grad)
            with_grad = sum(1 for parameter in role.parameters if parameter.grad is not None)
            clauses.append(
                f"{name}: {total} params, {trainable} require grad, {with_grad} have grad"
            )
        identities = {name: {id(p) for p in role.parameters} for name, role in self.roles.items()}
        names = list(identities)
        overlaps = [
            f"{left}/{right} share {len(identities[left] & identities[right])}"
            for index, left in enumerate(names)
            for right in names[index + 1 :]
            if identities[left] & identities[right]
        ]
        report = "; ".join(clauses)
        if overlaps:
            report = f"{report}; overlapping identities: {', '.join(overlaps)}"
        return report

    def _gradients_reach_parameters(self) -> bool:
        """Report whether this backend leaves gradients on ``parameter.grad``.

        DeepSpeed reduces gradients into the engine's own partitioned buffers and clears
        ``parameter.grad``, so reading it there says nothing about whether a role produced
        gradients -- the answer is always "none", for every role, always. Asserting on it
        would reject every correct DeepSpeed run.

        What the per-microbatch assertions guard against, a role that trains nothing, is
        checked once at setup by the prepared-trainability guard, and DeepSpeed itself
        raises if asked to step without gradients. So the loss here is narrow.

        Returns:
            Whether ``parameter.grad`` is authoritative for this backend.
        """
        return (
            self.accelerator.distributed_type != DistributedType.DEEPSPEED
            and not self._uses_fsdp1()
        )

    def _uses_fsdp1(self) -> bool:
        """Return whether gradients live on FSDP1 flattened/sharded storage.

        With ``use_orig_params=True``, an original parameter can expose no local
        grad on ranks that do not own its shard, while another role's original
        parameter can expose a non-None zero view into the same FlatParameter.
        Those views are not authoritative for role ownership. FSDP1 clipping
        likewise needs the complete prepared root so Accelerate delegates to
        FSDP's collective norm implementation.
        """
        if self.accelerator.distributed_type != DistributedType.FSDP:
            return False
        fsdp_plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
        return getattr(fsdp_plugin, "fsdp_version", 1) < 2

    def state_dict(self) -> Dict[str, Any]:
        """Return closed-phase role counters and optimizer-group ownership."""
        if self._active_role_name is not None or self._microbatch_open:
            raise RuntimeError(
                "state_dict expected a closed phase and microbatch, "
                f"received active_phase={self._active_role_name!r}, "
                f"microbatch_open={self._microbatch_open!r}"
            )
        self._validate_no_pending_gradients("state_dict")
        return {
            "version": _STATE_VERSION,
            "role_steps": {role_name: role.step for role_name, role in self.roles.items()},
            "optimizer_group_roles": self._optimizer_group_roles,
            "active_phase": self._active_role_name,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore closed-phase role counters after strict validation.

        Args:
            state_dict: Coordinator state produced by :meth:`state_dict`.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "expected coordinator state as a mapping, "
                f"received {type(state_dict).__name__}: {state_dict!r}"
            )
        received_version = state_dict.get("version")
        if received_version in (1, 2):
            raise ValueError(
                f"coordinator state version {received_version} used a retired chronology "
                f"and cannot be migrated; expected version {_STATE_VERSION}"
            )
        if (
            not isinstance(received_version, int)
            or isinstance(received_version, bool)
            or received_version != _STATE_VERSION
        ):
            raise ValueError(
                f"expected coordinator state version {_STATE_VERSION}, "
                f"received {received_version!r}"
            )
        received_keys = set(state_dict)
        if received_keys != _STATE_KEYS:
            raise ValueError(
                f"expected state keys {tuple(sorted(_STATE_KEYS))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        if state_dict["active_phase"] is not None:
            raise ValueError(
                "expected coordinator state at a closed phase with active_phase=None, "
                f"received {state_dict['active_phase']!r}"
            )
        raw_group_roles = state_dict["optimizer_group_roles"]
        if not isinstance(raw_group_roles, tuple):
            raise TypeError(
                "expected optimizer_group_roles as a tuple, "
                f"received {type(raw_group_roles).__name__}: {raw_group_roles!r}"
            )
        if raw_group_roles != self._optimizer_group_roles:
            raise ValueError(
                "optimizer group roles mismatch: expected "
                f"{self._optimizer_group_roles!r}, received {raw_group_roles!r}"
            )
        role_steps = state_dict["role_steps"]
        if not isinstance(role_steps, Mapping):
            raise TypeError(
                "expected role_steps as a mapping, "
                f"received {type(role_steps).__name__}: {role_steps!r}"
            )
        expected_role_names = tuple(self.roles)
        received_role_names = tuple(role_steps)
        if received_role_names != expected_role_names:
            raise ValueError(
                f"role_steps roles mismatch: expected {expected_role_names!r}, "
                f"received {received_role_names!r}"
            )
        for role_name, step in role_steps.items():
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(
                    f"expected non-negative int role step for {role_name!r}, received {step!r}"
                )
        if self._active_role_name is not None or self._microbatch_open:
            raise RuntimeError(
                "expected coordinator load at a closed phase and microbatch, "
                f"received active_phase={self._active_role_name!r}, "
                f"microbatch_open={self._microbatch_open!r}"
            )
        self._validate_no_pending_gradients("load_state_dict")
        for role_name, step in role_steps.items():
            self.roles[cast(RoleName, role_name)].step = step

    def _validate_no_pending_gradients(self, operation: str) -> None:
        """Reject state operations before accumulated gradients reach synchronization."""
        pending_counts = {
            role_name: sum(parameter.grad is not None for parameter in role.parameters)
            for role_name, role in self.roles.items()
        }
        pending_counts = {role_name: count for role_name, count in pending_counts.items() if count}
        if pending_counts:
            total_count = sum(pending_counts.values())
            raise RuntimeError(
                f"{operation} cannot run with pending accumulated gradients for roles "
                f"{tuple(pending_counts)!r}: {pending_counts!r}, {total_count} parameter(s); "
                "a required sync boundary must step and clear every role before state save/load"
            )

    def _validate_ownership(self) -> Tuple[RoleName, ...]:
        """Validate disjoint exhaustive role-to-group parameter ownership."""
        if not self.roles:
            raise ValueError("expected at least one optimization role, received none")
        parameter_owners: Dict[int, RoleName] = {}
        owned_group_ids: Dict[int, RoleName] = {}
        for role_name, role in self.roles.items():
            if role_name != role.config.role_name:
                raise ValueError(
                    f"expected role mapping key {role_name!r} to match config role_name, "
                    f"received {role.config.role_name!r}"
                )
            if not role.parameters:
                raise ValueError(
                    f"expected role {role_name!r} to own at least one parameter, received none"
                )
            if not role.optimizer_group_ids:
                raise ValueError(
                    f"expected role {role_name!r} to own optimizer groups, received none"
                )
            for parameter in role.parameters:
                if not isinstance(parameter, torch.Tensor) or not parameter.requires_grad:
                    raise TypeError(
                        f"expected a trainable torch.Tensor owned by role {role_name!r}, "
                        f"received {type(parameter).__name__}: {parameter!r}"
                    )
                existing_owner = parameter_owners.get(id(parameter))
                if existing_owner is not None:
                    raise ValueError(
                        "expected disjoint role parameter ownership, "
                        f"received role {role_name!r} sharing a parameter with "
                        f"role {existing_owner!r}"
                    )
                parameter_owners[id(parameter)] = role_name
            for group_id in role.optimizer_group_ids:
                if not isinstance(group_id, int) or isinstance(group_id, bool):
                    raise TypeError(
                        f"expected int optimizer group id for role {role_name!r}, "
                        f"received {type(group_id).__name__}: {group_id!r}"
                    )
                existing_owner = owned_group_ids.get(group_id)
                if existing_owner is not None:
                    raise ValueError(
                        f"expected disjoint optimizer group ownership, group {group_id} "
                        f"is owned by roles {existing_owner!r} and {role_name!r}"
                    )
                owned_group_ids[group_id] = role_name

        optimizer_groups = self.optimizer.param_groups
        expected_group_ids = tuple(range(len(optimizer_groups)))
        received_group_ids = tuple(sorted(owned_group_ids))
        if received_group_ids != expected_group_ids:
            raise ValueError(
                f"expected exhaustive optimizer group ids {expected_group_ids!r}, "
                f"received {received_group_ids!r}"
            )
        optimizer_parameter_ids = []
        optimizer_group_roles = []
        for group_id, group in enumerate(optimizer_groups):
            expected_role_name = owned_group_ids[group_id]
            received_role_name = group.get("role_name")
            if received_role_name != expected_role_name:
                raise ValueError(
                    f"optimizer group {group_id} expected role_name "
                    f"{expected_role_name!r}, received {received_role_name!r}"
                )
            group_parameter_ids = tuple(id(parameter) for parameter in group["params"])
            role_parameter_ids = {
                id(parameter) for parameter in self.roles[expected_role_name].parameters
            }
            if any(parameter_id not in role_parameter_ids for parameter_id in group_parameter_ids):
                raise ValueError(
                    f"optimizer group {group_id} for role {expected_role_name!r} "
                    "contains parameters not owned by that role"
                )
            optimizer_parameter_ids.extend(group_parameter_ids)
            optimizer_group_roles.append(expected_role_name)

        if len(set(optimizer_parameter_ids)) != len(optimizer_parameter_ids):
            raise ValueError(
                "expected each optimizer parameter in exactly one group, "
                f"received duplicate ids {optimizer_parameter_ids!r}"
            )
        if set(optimizer_parameter_ids) != set(parameter_owners):
            raise ValueError(
                "expected exhaustive optimizer parameter ownership, "
                f"expected ids {tuple(sorted(parameter_owners))!r}, "
                f"received {tuple(sorted(optimizer_parameter_ids))!r}"
            )
        for role_name, role in self.roles.items():
            grouped_ids = {
                id(parameter)
                for group_id in role.optimizer_group_ids
                for parameter in optimizer_groups[group_id]["params"]
            }
            role_ids = {id(parameter) for parameter in role.parameters}
            if grouped_ids != role_ids:
                raise ValueError(
                    f"expected exhaustive optimizer parameters for role {role_name!r}: "
                    f"expected ids {tuple(sorted(role_ids))!r}, "
                    f"received {tuple(sorted(grouped_ids))!r}"
                )
        return cast(Tuple[RoleName, ...], tuple(optimizer_group_roles))

    def _parameters_with_grad(self) -> Tuple[Tuple[RoleName, int], ...]:
        """Return role and local parameter indices whose gradients are present."""
        return tuple(
            (role_name, parameter_index)
            for role_name, role in self.roles.items()
            for parameter_index, parameter in enumerate(role.parameters)
            if parameter.grad is not None
        )
