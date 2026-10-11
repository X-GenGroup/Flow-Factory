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

# src/flow_factory/trainers/distillation/opd/common.py
"""Shared target-space math and teacher loading for DiffusionOPD.

Target projection is shared by the teacher and student passes; the per-sample
loss consumes their projected outputs. Both come in a single-tensor flavour (the
single-latent contract, used by the parity tests) and a structured flavour keyed
by ``trajectory_component_order``. Teacher loading stores each teacher LoRA
checkpoint in a named-parameter snapshot using the adapter primitives in
:mod:`flow_factory.models.abc`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

import torch

from ....samples import ComponentTimes, LatentState, MultiModalStepOutput
from ....utils.base import to_broadcast_tensor
from ....utils.logger_utils import setup_logger
from ....utils.noise_schedule import flow_match_sigma

if TYPE_CHECKING:
    from ...models.abc import BaseAdapter

logger = setup_logger(__name__, rank_zero_only=True)

LOSS_TARGETS: Tuple[str, ...] = ("xt", "v", "x0")
# Scheduler output field each target reads, in the legacy request name (what
# ``return_fields`` asks ``forward`` for) and the structured attribute name.
TARGET_REQUEST_FIELDS: Mapping[str, str] = {
    "xt": "next_latents_mean",
    "v": "velocity",
    "x0": "velocity",
}
_TARGET_OUTPUT_FIELDS: Mapping[str, str] = {
    "xt": "next_state_mean",
    "v": "velocity",
    "x0": "velocity",
}


def _require_known_loss_target(loss_target: str, context: Optional[str] = None) -> None:
    """Validate the configured target space name."""
    if loss_target not in LOSS_TARGETS:
        raise ValueError(
            f"{_context_prefix(context)}DiffusionOPD loss_target must be one of "
            f"{LOSS_TARGETS}, got {loss_target!r}."
        )


def validate_loss_target_for_dynamics(
    loss_target: str,
    dynamics_type: str,
    *,
    component: Optional[str] = None,
) -> None:
    """Validate that the target is defined for the scheduler dynamics.

    Args:
        loss_target: Configured target space (``xt``, ``v``, or ``x0``).
        dynamics_type: Active scheduler dynamics.
        component: Optional trajectory component owning the scheduler, reported
            by the error so a heterogeneous group names the offender.

    Raises:
        ValueError: ``v`` or ``x0`` is requested for non-ODE dynamics.
    """
    if loss_target in ("v", "x0") and dynamics_type != "ODE":
        context = "" if component is None else f" for component {component!r}"
        raise ValueError(
            "DiffusionOPD velocity-derived targets require ODE dynamics: "
            f"received loss_target={loss_target!r} with dynamics_type={dynamics_type!r}"
            f"{context}. Use scheduler.dynamics_type='ODE' or set train.loss_target='xt'."
        )


def resolve_scheduler_group_dynamics(adapter: "BaseAdapter", loss_target: str) -> bool:
    """Validate every component scheduler and report whether the group is stochastic.

    ``v``/``x0`` stay ODE-only for every component, and a group that mixes ODE
    with stochastic components is rejected: the per-step KL denominator is
    ``1`` for the deterministic members and a transition variance for the
    stochastic ones, so combining them needs a mixed normalization that is not
    defined yet.

    Args:
        adapter: Adapter declaring ``trajectory_component_order`` and its
            matching ``scheduler_group``.
        loss_target: Configured target space (``xt``, ``v``, or ``x0``).

    Returns:
        ``True`` when every component runs stochastic dynamics.

    Raises:
        ValueError: A component rejects the target, or the group mixes ODE and
            stochastic dynamics.
    """
    _require_known_loss_target(loss_target)
    expected_names = adapter.trajectory_component_order
    dynamics: Dict[str, str] = {}
    for name in expected_names:
        dynamics_type = adapter.scheduler_group[name].dynamics_type
        validate_loss_target_for_dynamics(loss_target, dynamics_type, component=name)
        dynamics[name] = dynamics_type
    ode_components = tuple(name for name, value in dynamics.items() if value == "ODE")
    if ode_components and len(ode_components) != len(expected_names):
        raise ValueError(
            "DiffusionOPD rejects a mixed ODE/SDE scheduler group until a mathematically "
            f"explicit mixed normalization is defined: received {dynamics} for "
            f"trajectory_component_order {expected_names}."
        )
    return not ode_components


def _require_matching_target(
    value: Optional[torch.Tensor],
    *,
    name: str,
    loss_target: str,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Validate and return one model output used as a target."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Expected `{name}` to be a torch.Tensor for loss_target={loss_target!r}, "
            f"got {type(value).__name__}: {value!r}."
        )
    if value.shape != latents.shape:
        raise ValueError(
            "DiffusionOPD target projection requires the same shape for `latents` "
            f"and `{name}`, got latents={tuple(latents.shape)} and "
            f"{name}={tuple(value.shape)}."
        )
    return value


def project_distillation_target(
    *,
    loss_target: str,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    next_latents_mean: Optional[torch.Tensor],
    velocity: Optional[torch.Tensor],
    sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project a scheduler step output into the configured target space.

    Args:
        loss_target: Target space (``xt``, ``v``, or ``x0``).
        latents: Current noisy latent state.
        timestep: Current scheduler-scale timestep.
        next_latents_mean: Predicted one-step transition mean.
        velocity: Predicted flow velocity.
        sigma: Optional stored noise level used by ``x0``. When omitted the
            flow-matching schedule ``sigma = timestep / 1000`` is derived from
            ``timestep``.

    Returns:
        The prediction represented in the configured target space.

    Raises:
        TypeError: A required input is not a tensor.
        ValueError: The target is unsupported or a model output shape differs
            from ``latents``.
    """
    if not isinstance(latents, torch.Tensor):
        raise TypeError(
            "Expected `latents` to be a torch.Tensor when projecting a DiffusionOPD "
            f"target, got {type(latents).__name__}: {latents!r}."
        )

    if loss_target == "xt":
        return _require_matching_target(
            next_latents_mean,
            name="next_latents_mean",
            loss_target=loss_target,
            latents=latents,
        )
    if loss_target not in ("v", "x0"):
        raise ValueError(
            f"DiffusionOPD loss_target must be one of ('xt', 'v', 'x0'), got {loss_target!r}."
        )

    velocity = _require_matching_target(
        velocity,
        name="velocity",
        loss_target=loss_target,
        latents=latents,
    )
    if loss_target == "v":
        return velocity

    if sigma is None:
        if not isinstance(timestep, torch.Tensor):
            raise TypeError(
                "Expected `timestep` to be a torch.Tensor for loss_target='x0', "
                f"got {type(timestep).__name__}: {timestep!r}."
            )
        sigma = flow_match_sigma(timestep.float())
    elif not isinstance(sigma, torch.Tensor):
        raise TypeError(
            "Expected `sigma` to be a torch.Tensor for loss_target='x0', "
            f"got {type(sigma).__name__}: {sigma!r}."
        )
    latents_float = latents.float()
    broadcast_sigma = to_broadcast_tensor(sigma.float(), latents_float)
    return latents_float - broadcast_sigma * velocity.float()


def compute_per_sample_distillation_loss(
    student_target: torch.Tensor,
    teacher_target: torch.Tensor,
    *,
    self_normalize: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute target-space MSE with optional detached self-normalization.

    Args:
        student_target: Student prediction in the configured target space.
        teacher_target: Detached teacher prediction in the same target space.
        self_normalize: Whether to divide by detached mean absolute error.
        eps: Positive denominator floor added after self-normalization.

    Returns:
        Per-sample loss reduced over all non-batch dimensions.

    Raises:
        TypeError: Targets are not tensors or ``self_normalize`` is not bool.
        ValueError: Target shapes are invalid or ``eps`` is not positive.
    """
    if not isinstance(student_target, torch.Tensor) or not isinstance(teacher_target, torch.Tensor):
        raise TypeError(
            "Expected `student_target` and `teacher_target` to be torch.Tensor values, "
            f"got {type(student_target).__name__} and {type(teacher_target).__name__}."
        )
    if student_target.shape != teacher_target.shape:
        raise ValueError(
            "DiffusionOPD loss requires matching shapes for `student_target` and "
            f"`teacher_target`, got {tuple(student_target.shape)} and "
            f"{tuple(teacher_target.shape)}."
        )
    if student_target.ndim < 2:
        raise ValueError(
            "DiffusionOPD loss requires target tensors with a batch dimension and at "
            f"least one feature dimension, got shape {tuple(student_target.shape)}."
        )
    if not isinstance(self_normalize, bool):
        raise TypeError(
            "Expected `self_normalize` to be a bool, "
            f"got {type(self_normalize).__name__}: {self_normalize!r}."
        )
    if eps <= 0:
        raise ValueError(f"Expected `eps` to be positive, got {eps!r}.")

    error = student_target.float() - teacher_target.float()
    per_sample_mse = error.square().flatten(1).mean(dim=1)
    if not self_normalize:
        return per_sample_mse
    scale = error.abs().flatten(1).mean(dim=1).detach()
    return per_sample_mse / (scale + eps)


def _context_prefix(context: Optional[str]) -> str:
    """Render the caller context (teacher/student pass and step) as a message prefix."""
    if context is None:
        return ""
    if not isinstance(context, str) or not context:
        raise ValueError(
            f"expected a non-empty string or None for the DiffusionOPD projection context, "
            f"received {context!r}"
        )
    return f"{context}: "


def _require_state(
    adapter: "BaseAdapter",
    value: object,
    *,
    role: str,
    context: Optional[str] = None,
) -> LatentState:
    """Validate a structured latent argument against the adapter component order."""
    prefix = _context_prefix(context)
    if not isinstance(value, LatentState):
        raise TypeError(
            f"{prefix}expected LatentState for DiffusionOPD {role}, "
            f"received {type(value).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    if value.component_names != expected_names:
        raise ValueError(
            f"{prefix}expected DiffusionOPD {role} in component order {expected_names}, "
            f"received {value.component_names}"
        )
    batch_size: Optional[int] = None
    for name in expected_names:
        component = value.components[name]
        if component.ndim < 1:
            raise ValueError(
                f"{prefix}expected DiffusionOPD {role} component {name!r} to be a batched "
                f"tensor with a leading batch dimension, received shape "
                f"{tuple(component.shape)}"
            )
        if batch_size is None:
            batch_size = component.shape[0]
        elif component.shape[0] != batch_size:
            raise ValueError(
                f"{prefix}expected DiffusionOPD {role} component {name!r} to share the batch "
                f"size {batch_size} of component {expected_names[0]!r}, received "
                f"{tuple(component.shape)}"
            )
    return value


def _validate_component_times(
    adapter: "BaseAdapter",
    times: ComponentTimes,
    state: LatentState,
    batch_size: int,
    context: Optional[str] = None,
) -> None:
    """Validate the whole replay time contract before any target branch runs.

    Every present coordinate must cover the authoritative component order and
    carry exactly one value per sample on the state device. The single
    documented exception is the legacy terminal ``t_next``: the one-component
    replay stores it as a shared 0-dim scalar zero because adapters keep one
    timestep per denoising step while latents keep one more rollout position.
    """
    prefix = _context_prefix(context)
    expected_names = adapter.trajectory_component_order
    if not isinstance(times, ComponentTimes):
        raise TypeError(
            f"{prefix}expected ComponentTimes for DiffusionOPD replay times, received "
            f"{type(times).__name__}"
        )
    fields: Dict[str, Optional[Mapping[str, torch.Tensor]]] = {
        "timestep": times.timestep,
        "next_timestep": times.next_timestep,
        "sigma": times.sigma,
        "next_sigma": times.next_sigma,
    }
    for field, values in fields.items():
        if values is None:
            continue
        if tuple(values) != expected_names:
            raise ValueError(
                f"{prefix}expected DiffusionOPD replay {field} in component order "
                f"{expected_names}, received {tuple(values)}"
            )
        for name in expected_names:
            value = values[name]
            reference = state.components[name]
            if value.shape != (batch_size,):
                terminal = (
                    field == "next_timestep" and expected_names == ("latent",) and value.ndim == 0
                )
                if not terminal:
                    raise ValueError(
                        f"{prefix}expected DiffusionOPD replay {field} for component {name!r} "
                        f"to hold one value per sample with shape {(batch_size,)}, received "
                        f"{tuple(value.shape)}"
                    )
                if bool(value.ne(0).item()):
                    raise ValueError(
                        f"{prefix}expected the DiffusionOPD replay {field} for component "
                        f"{name!r} terminal fallback to be the scalar 0, received {value}"
                    )
            if value.device != reference.device:
                raise ValueError(
                    f"{prefix}expected DiffusionOPD replay {field} for component {name!r} on "
                    f"the replay state device {reference.device}, received {value.device}"
                )


def _resolve_component_sigmas(
    adapter: "BaseAdapter",
    times: ComponentTimes,
    context: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Resolve the per-component noise level used by the ``x0`` projection."""
    expected_names = adapter.trajectory_component_order
    if times.sigma is None:
        # The flow-matching fallback (sigma = timestep / 1000) is the legacy
        # single-latent contract; a structured trajectory must store sigmas.
        if expected_names != ("latent",):
            raise ValueError(
                f"{_context_prefix(context)}DiffusionOPD loss_target='x0' requires "
                f"per-component sigmas in component order {expected_names}; the replay "
                "carries no stored sigma and the flow-matching fallback is only defined "
                "for the legacy single 'latent' component."
            )
        return {"latent": flow_match_sigma(times.timestep["latent"].float())}
    return {name: times.sigma[name] for name in expected_names}


def project_distillation_target_state(
    adapter: "BaseAdapter",
    *,
    loss_target: str,
    state: LatentState,
    output: MultiModalStepOutput,
    times: ComponentTimes,
    context: Optional[str] = None,
) -> LatentState:
    """Project a structured step output into the configured target space.

    Each component is projected independently with the same math as
    :func:`project_distillation_target`; ``x0`` uses the component's stored
    sigma and falls back to the flow-matching schedule only for the legacy
    single-``latent`` trajectory.

    Args:
        adapter: Adapter declaring ``trajectory_component_order``.
        loss_target: Target space (``xt``, ``v``, or ``x0``).
        state: Replay state the prediction was produced from.
        output: Structured scheduler output carrying the requested field.
        times: Replay coordinates supplying the ``x0`` noise level.
        context: Optional caller description (teacher/student pass and replay
            step) prefixed to every raised message.

    Returns:
        A :class:`LatentState` in ``trajectory_component_order`` holding the
        prediction in the configured target space.

    Raises:
        TypeError: A structured argument has the wrong type.
        ValueError: The target is unsupported, a component is missing, or a
            component shape/device/sigma is inconsistent with the state.
    """
    prefix = _context_prefix(context)
    _require_known_loss_target(loss_target, context=context)
    state = _require_state(adapter, state, role="replay state", context=context)
    if not isinstance(output, MultiModalStepOutput):
        raise TypeError(
            f"{prefix}expected MultiModalStepOutput for DiffusionOPD forward output, received "
            f"{type(output).__name__}"
        )
    expected_names = adapter.trajectory_component_order
    batch_size = state.components[expected_names[0]].shape[0]
    _validate_component_times(adapter, times, state, batch_size, context=context)
    field = _TARGET_OUTPUT_FIELDS[loss_target]
    predicted = getattr(output, field)
    if not isinstance(predicted, LatentState) or predicted.component_names != expected_names:
        received = (
            "None"
            if predicted is None
            else (
                str(predicted.component_names)
                if isinstance(predicted, LatentState)
                else type(predicted).__name__
            )
        )
        raise ValueError(
            f"{prefix}expected DiffusionOPD forward output field {field!r} for "
            f"loss_target={loss_target!r} in component order {expected_names}, received "
            f"{received}; request it through return_fields."
        )
    for name in expected_names:
        reference = state.components[name]
        value = predicted.components[name]
        if value.shape != reference.shape:
            raise ValueError(
                f"{prefix}expected DiffusionOPD forward output field {field!r} component "
                f"{name!r} to match the replay state shape {tuple(reference.shape)}, received "
                f"{tuple(value.shape)}"
            )
        if value.device != reference.device:
            raise ValueError(
                f"{prefix}expected DiffusionOPD forward output field {field!r} component "
                f"{name!r} on the replay state device {reference.device}, received "
                f"{value.device}"
            )

    if loss_target in ("xt", "v"):
        return LatentState({name: predicted.components[name] for name in expected_names})

    sigmas = _resolve_component_sigmas(adapter, times, context=context)
    projection_times = ComponentTimes(
        timestep=times.timestep,
        next_timestep=times.next_timestep,
        sigma={name: sigmas[name].float() for name in expected_names},
        next_sigma=times.next_sigma,
    )
    return adapter.project_velocity_to_clean_state(
        LatentState(
            {name: state.components[name].float() for name in expected_names},
            active_masks=state.active_masks,
        ),
        projection_times,
        LatentState(
            {name: predicted.components[name].float() for name in expected_names},
            active_masks=predicted.active_masks,
        ),
    )


def _validate_component_denominators(
    adapter: "BaseAdapter",
    denominators: Mapping[str, torch.Tensor],
    batch_size: int,
) -> None:
    """Validate the per-component KL denominators used by stochastic dynamics."""
    expected_names = adapter.trajectory_component_order
    if not isinstance(denominators, Mapping):
        raise TypeError(
            f"expected Mapping[str, torch.Tensor] or None for DiffusionOPD per-component KL "
            f"denominators, received {type(denominators).__name__}"
        )
    if tuple(denominators) != expected_names:
        raise ValueError(
            f"expected DiffusionOPD per-component KL denominators in component order "
            f"{expected_names}, received {tuple(denominators)}"
        )
    for name in expected_names:
        denominator = denominators[name]
        if not isinstance(denominator, torch.Tensor):
            raise TypeError(
                f"expected a torch.Tensor DiffusionOPD KL denominator for component {name!r}, "
                f"received {type(denominator).__name__}"
            )
        if denominator.shape != (batch_size,):
            raise ValueError(
                f"expected DiffusionOPD KL denominator for component {name!r} with shape "
                f"{(batch_size,)}, received {tuple(denominator.shape)}"
            )
        if not bool(torch.isfinite(denominator).all()):
            raise ValueError(
                f"expected the DiffusionOPD KL denominator for component {name!r} to be "
                f"finite, received {denominator}"
            )
        if not bool((denominator > 0).all()):
            raise ValueError(
                f"expected the DiffusionOPD KL denominator for component {name!r} to be "
                f"strictly positive, received {denominator}"
            )


def compute_structured_distillation_loss(
    adapter: "BaseAdapter",
    *,
    student_target: LatentState,
    teacher_target: LatentState,
    state: LatentState,
    self_normalize: bool,
    eps: float = 1e-8,
    denominators: Optional[Mapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute the per-sample structured distillation loss.

    Exactly one state-aware reduction runs, always over raw latent-shaped
    values, so a masked model never sees a pre-reduced tensor. Self-normalization
    stays global: one detached scale per sample shared by every component, which
    preserves the relative component weighting.

    The per-component KL denominators of stochastic dynamics are applied around
    that reduction according to how many components share it. A single component
    divides the reduced ``(B,)`` value, reproducing the legacy floating-point
    order exactly for an arbitrary positive denominator (this stays correct
    under a dynamic mask because the denominator is a per-sample scalar).
    Several components may carry different denominators, so each divides the
    raw squared-error elements of its own component before the shared reduction.

    Args:
        adapter: Adapter owning the reduction and component order.
        student_target: Student prediction per component.
        teacher_target: Detached teacher prediction per component.
        state: Replay state, forwarded to the adapter reducers so a masked
            model reduces over its active elements only.
        self_normalize: Whether to divide by the detached mean absolute error.
        eps: Positive denominator floor added after self-normalization.
        denominators: Optional per-component KL denominators with one value per
            sample. ``None`` keeps the deterministic (ODE) reduction.

    Returns:
        Per-sample loss with shape ``(batch_size,)``.

    Raises:
        TypeError: A structured argument or flag has the wrong type.
        ValueError: Component order, shapes, ``eps``, or a denominator value is
            invalid.
    """
    student_target = _require_state(adapter, student_target, role="student target")
    teacher_target = _require_state(adapter, teacher_target, role="teacher target")
    state = _require_state(adapter, state, role="replay state")
    if not isinstance(self_normalize, bool):
        raise TypeError(
            f"expected a bool for DiffusionOPD self_normalize, "
            f"received {type(self_normalize).__name__}: {self_normalize!r}"
        )
    if eps <= 0:
        raise ValueError(f"expected a positive DiffusionOPD loss eps, received {eps!r}")

    expected_names = adapter.trajectory_component_order
    errors: Dict[str, torch.Tensor] = {}
    for name in expected_names:
        student = student_target.components[name]
        teacher = teacher_target.components[name]
        if teacher.shape != student.shape:
            raise ValueError(
                f"expected DiffusionOPD teacher target component {name!r} to match the student "
                f"target shape {tuple(student.shape)}, received {tuple(teacher.shape)}"
            )
        errors[name] = student.float() - teacher.float()
    squared = {name: errors[name].square() for name in expected_names}

    scale: Optional[torch.Tensor] = None
    if self_normalize:
        absolute = {name: errors[name].abs() for name in expected_names}
        scale = adapter.reduce_latent_values(absolute, state=state).detach()

    values = squared
    reduced_denominator: Optional[torch.Tensor] = None
    if denominators is not None:
        batch_size = student_target.components[expected_names[0]].shape[0]
        _validate_component_denominators(adapter, denominators, batch_size)
        if len(expected_names) == 1:
            # One component shares one denominator, so dividing the reduced
            # value reproduces the legacy floating-point order exactly for an
            # arbitrary positive denominator (not only powers of two).
            reduced_denominator = denominators[expected_names[0]]
        else:
            values = {
                name: squared[name] / to_broadcast_tensor(denominators[name], squared[name])
                for name in expected_names
            }

    per_sample_loss = adapter.reduce_latent_values(values, state=state)
    if scale is not None:
        per_sample_loss = per_sample_loss / (scale + eps)
    if reduced_denominator is not None:
        per_sample_loss = per_sample_loss / reduced_denominator
    return per_sample_loss


def load_teachers(
    adapter: "BaseAdapter",
    teacher_paths: List[str],
    teacher_param_device: str,
    teacher_names: Optional[List[Optional[str]]] = None,
) -> List[str]:
    """Load each teacher LoRA checkpoint into a named-parameter snapshot.

    For every teacher the live student LoRA tensors are snapshotted, the
    teacher checkpoint is loaded into the active adapter slot via
    :meth:`BaseAdapter._load_lora` (clobbering the student weights), captured
    into a named snapshot via :meth:`BaseAdapter.add_named_parameters`, and the
    student weights are then restored. Swap a teacher in at run time with
    ``with adapter.use_named_parameters(name): ...``.

    Because ``_load_lora`` loads into the student's active ``"default"`` adapter
    slot, every teacher checkpoint MUST share the student's LoRA architecture
    (same ``target_components`` / target modules and rank-compatible weights).
    Incompatible checkpoints raise a clear error pointing at this constraint.

    Args:
        adapter: Active :class:`BaseAdapter` in LoRA finetune mode with the
            student adapter already attached.
        teacher_paths: Local checkpoint paths or HF Hub repo ids
            (``owner/repo[/subfolder][@revision]``, optional ``hf://`` prefix),
            resolved via :meth:`BaseAdapter._resolve_checkpoint_path`. Must be
            non-empty.
        teacher_param_device: ``'cpu'`` (low VRAM, H2D copy per swap) or
            ``'cuda'`` (on-device, LoRA-sized VRAM per teacher).
        teacher_names: Optional snapshot names (one per path). A ``None`` entry
            (or short list) falls back to ``'opd_teacher_{i}'``.

    Returns:
        Snapshot names in the same order as ``teacher_paths`` -- the lookup
        keys for :meth:`BaseAdapter.use_named_parameters`.

    Raises:
        ValueError: ``teacher_paths`` is empty, the adapter is not in LoRA
            mode, or it exposes no trainable LoRA components.
        RuntimeError: a teacher checkpoint is incompatible with the student's
            LoRA architecture.
    """
    if not teacher_paths:
        raise ValueError(
            f"DiffusionOPD requires at least one teacher LoRA path; got teacher_paths={teacher_paths!r}."
        )
    if adapter.model_args.finetune_type != "lora":
        raise ValueError(
            "load_teachers requires the adapter to be in 'lora' finetune mode "
            f"(teacher LoRAs load into the student's adapter slot), but "
            f"model_args.finetune_type={adapter.model_args.finetune_type!r}."
        )

    target_components: List[str] = [
        comp for comp, mods in adapter.target_module_map.items() if mods
    ]
    if not target_components:
        raise ValueError(
            "Adapter has no trainable LoRA components; expected at least one entry with "
            f"non-empty modules in target_module_map={adapter.target_module_map!r}."
        )

    names: List[str] = []
    for i, path in enumerate(teacher_paths):
        name = (
            teacher_names[i]
            if teacher_names and i < len(teacher_names) and teacher_names[i]
            else f"opd_teacher_{i}"
        )
        _load_one_teacher(adapter, name, path, target_components, teacher_param_device)
        names.append(name)

    logger.info(
        f"Loaded {len(names)} DiffusionOPD teacher(s): {names} (device={teacher_param_device!r})."
    )
    return names


def _load_one_teacher(
    adapter: "BaseAdapter",
    name: str,
    lora_path: str,
    target_components: List[str],
    device: str,
) -> None:
    """Load one teacher LoRA into snapshot ``name``, restoring student weights.

    The student LoRA tensors are the live ``nn.Parameter`` objects; ``_load_lora``
    mutates them in place, so we keep detached clones and copy them back in a
    ``finally`` block (even if loading raised). Loading errors are surfaced with
    the LoRA-architecture constraint that almost always causes them.
    """
    # Resolve HF Hub specs / validate local layout before touching weights.
    lora_path = adapter._resolve_checkpoint_path(lora_path)
    if len(target_components) > 1:
        for comp in target_components:
            sub = os.path.join(lora_path, comp)
            if not os.path.exists(sub):
                raise FileNotFoundError(
                    f"Multi-component LoRA layout requires per-component subdirectories; "
                    f"missing {sub!r} for component {comp!r} under teacher path {lora_path!r}."
                )

    live_params = adapter._get_component_parameters(target_components)
    if not live_params:
        raise ValueError(
            f"No trainable LoRA parameters found on components {target_components!r}; "
            "ensure the student LoRA adapter is attached before loading teachers."
        )
    saved_data = [p.detach().clone() for p in live_params]

    try:
        adapter._load_lora(lora_path)
        adapter.add_named_parameters(
            name=name,
            target_components=target_components,
            device=device,
            overwrite=True,
        )
    except (RuntimeError, ValueError, KeyError, TypeError) as e:
        # Almost always a LoRA-architecture mismatch (rank/alpha/target modules)
        # between the teacher checkpoint and the student adapter slot.
        raise RuntimeError(
            f"Failed to load teacher LoRA {name!r} from {lora_path!r}. Teacher checkpoints "
            f"must share the student's LoRA architecture (target_components={target_components}, "
            "matching target modules and compatible rank/alpha), since they load into the "
            "student's active adapter slot. Verify the teacher was trained with the same "
            f"LoRA config as the student. Original error: {e}"
        ) from e
    finally:
        # Always restore the student weights, even if loading/snapshotting raised.
        with torch.no_grad():
            for live, saved in zip(live_params, saved_data):
                live.data.copy_(saved.to(live.device))
