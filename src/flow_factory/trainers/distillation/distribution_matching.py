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

"""Flow-matching x0 cores for DMD and TDM distillation."""

from __future__ import annotations

import math
from numbers import Real
from typing import Mapping, Tuple

import torch

from ...models.abc import BaseAdapter
from ...samples import ComponentTimes, LatentState, NoisedState
from ...utils.base import to_broadcast_tensor

_EPS = 1e-6


def _require_state(value: LatentState, *, identifier: str) -> LatentState:
    if not isinstance(value, LatentState):
        raise TypeError(
            f"expected LatentState for {identifier}, received {type(value).__name__}: {value!r}"
        )
    return value


def _require_component_order(
    state: LatentState,
    expected_names: tuple[str, ...],
    *,
    identifier: str,
) -> None:
    if state.component_names != expected_names:
        raise ValueError(
            f"expected {identifier} component order {expected_names}, "
            f"received {state.component_names}"
        )


def _validate_state_against(
    state: LatentState,
    reference: LatentState,
    *,
    identifier: str,
    require_detached: bool,
) -> None:
    expected_names = reference.component_names
    _require_component_order(state, expected_names, identifier=identifier)
    for name in expected_names:
        component = state.components[name]
        reference_component = reference.components[name]
        if component.shape != reference_component.shape:
            raise ValueError(
                f"expected {identifier} component {name!r} shape "
                f"{tuple(reference_component.shape)}, received {tuple(component.shape)}"
            )
        if component.device != reference_component.device:
            raise ValueError(
                f"expected {identifier} component {name!r} device "
                f"{reference_component.device}, received {component.device}"
            )
        if not component.is_floating_point():
            raise TypeError(
                f"expected floating-point tensor for {identifier} component {name!r}, "
                f"received dtype {component.dtype}"
            )
        if require_detached and component.requires_grad:
            raise ValueError(
                f"expected {identifier} component {name!r} to have requires_grad=False, "
                "received requires_grad=True"
            )


def _require_adapter(adapter: BaseAdapter) -> BaseAdapter:
    if not isinstance(adapter, BaseAdapter):
        raise TypeError(f"expected BaseAdapter, received {type(adapter).__name__}: {adapter!r}")
    return adapter


def _sigma_components(
    sigma: Mapping[str, torch.Tensor] | LatentState,
    reference: LatentState,
    *,
    identifier: str,
) -> Mapping[str, torch.Tensor]:
    if isinstance(sigma, LatentState):
        components = sigma.components
    elif isinstance(sigma, Mapping):
        components = sigma
    else:
        raise TypeError(
            f"expected Mapping[str, torch.Tensor] or LatentState for {identifier}, "
            f"received {type(sigma).__name__}: {sigma!r}"
        )
    if tuple(components) != reference.component_names:
        raise ValueError(
            f"expected {identifier} component order {reference.component_names}, "
            f"received {tuple(components)}"
        )
    broadcast: dict[str, torch.Tensor] = {}
    for name, component in reference.components.items():
        scale = components[name]
        if not isinstance(scale, torch.Tensor):
            raise TypeError(
                f"expected torch.Tensor for {identifier}[{name!r}], "
                f"received {type(scale).__name__}: {scale!r}"
            )
        if scale.device != component.device:
            raise ValueError(
                f"expected {identifier}[{name!r}] device {component.device}, received {scale.device}"
            )
        if not scale.is_floating_point():
            raise TypeError(
                f"expected floating {identifier}[{name!r}], received dtype {scale.dtype}"
            )
        batch_size = component.shape[0]
        if scale.shape not in (torch.Size([]), torch.Size([batch_size]), component.shape):
            if scale.ndim == 1 and scale.shape[0] == batch_size:
                scale = scale.reshape(batch_size, *([1] * (component.ndim - 1)))
            else:
                raise ValueError(
                    f"expected {identifier}[{name!r}] shape (), {(batch_size,)}, or "
                    f"{tuple(component.shape)}, received {tuple(scale.shape)}"
                )
        elif scale.ndim == 1:
            scale = scale.reshape(batch_size, *([1] * (component.ndim - 1)))
        broadcast[name] = scale.to(dtype=torch.float32)
    return broadcast


def _map_components(
    left: LatentState,
    right: LatentState,
    transform,
    *,
    identifier: str,
) -> LatentState:
    _validate_state_against(right, left, identifier=identifier, require_detached=False)
    return LatentState(
        {
            name: transform(left.components[name], right.components[name], name)
            for name in left.component_names
        },
        active_masks=left.active_masks,
    )


def add_flow_noise(
    x0: LatentState,
    noise: LatentState,
    sigma: Mapping[str, torch.Tensor] | LatentState,
) -> LatentState:
    """Forward flow-matching noising: ``x_t = (1 - σ) x0 + σ ε``."""
    x0 = _require_state(x0, identifier="x0")
    noise = _require_state(noise, identifier="noise")
    scales = _sigma_components(sigma, x0, identifier="sigma")
    return _map_components(
        x0,
        noise,
        lambda clean, eps, name: (1.0 - scales[name]) * clean.to(torch.float32)
        + scales[name] * eps.to(torch.float32),
        identifier="noise",
    )


def flow_velocity_target(x0: LatentState, noise: LatentState) -> LatentState:
    """Flow-matching velocity target: ``v = ε - x0``."""
    x0 = _require_state(x0, identifier="x0")
    noise = _require_state(noise, identifier="noise")
    return _map_components(
        x0,
        noise,
        lambda clean, eps, _name: eps.to(torch.float32) - clean.to(torch.float32),
        identifier="noise",
    )


def velocity_to_x0(
    x_t: LatentState,
    velocity: LatentState,
    sigma: Mapping[str, torch.Tensor] | LatentState,
) -> LatentState:
    """Clean-state estimate under the noise-ward convention: ``x0 = x_t - σ v``.

    Trainers must use ``adapter.project_velocity_to_clean_state()`` so
    data-ward adapters (MiniMax H3) project correctly. This helper is the
    noise-ward math used by unit tests.
    """
    x_t = _require_state(x_t, identifier="x_t")
    velocity = _require_state(velocity, identifier="velocity")
    scales = _sigma_components(sigma, x_t, identifier="sigma")
    return _map_components(
        x_t,
        velocity,
        lambda state, velocity_component, name: state.to(torch.float32)
        - scales[name] * velocity_component.to(torch.float32),
        identifier="velocity",
    )


def flow_matching_loss(
    adapter: BaseAdapter,
    v_pred: LatentState,
    v_target: LatentState,
    *,
    state: LatentState,
) -> torch.Tensor:
    """Mean-squared flow-matching loss between predicted and target velocity."""
    adapter = _require_adapter(adapter)
    v_pred = _require_state(v_pred, identifier="v_pred")
    v_target = _require_state(v_target, identifier="v_target")
    state = _require_state(state, identifier="state")
    expected_names = adapter.trajectory_component_order
    _require_component_order(state, expected_names, identifier="state")
    _require_component_order(v_pred, expected_names, identifier="v_pred")
    _validate_state_against(v_pred, state, identifier="v_pred", require_detached=False)
    _validate_state_against(v_target, state, identifier="v_target", require_detached=True)
    squared = {
        name: (
            v_pred.components[name].to(torch.float32) - v_target.components[name].to(torch.float32)
        ).square()
        for name in expected_names
    }
    return adapter.reduce_latent_values(squared, state=state).to(torch.float32).mean()


def dmd_generator_loss(
    adapter: BaseAdapter,
    x0_gen: LatentState,
    x0_real: LatentState,
    x0_fake: LatentState,
    *,
    eps: float = _EPS,
) -> torch.Tensor:
    """Stop-grad x0 DMD generator loss whose gradient follows ``(x0_fake - x0_real)``."""
    adapter = _require_adapter(adapter)
    x0_gen = _require_state(x0_gen, identifier="x0_gen")
    x0_real = _require_state(x0_real, identifier="x0_real")
    x0_fake = _require_state(x0_fake, identifier="x0_fake")
    if (
        not isinstance(eps, Real)
        or isinstance(eps, bool)
        or not math.isfinite(float(eps))
        or eps <= 0
    ):
        raise ValueError(f"expected finite eps > 0, received {eps!r}")
    expected_names = adapter.trajectory_component_order
    _require_component_order(x0_gen, expected_names, identifier="x0_gen")
    _validate_state_against(x0_real, x0_gen, identifier="x0_real", require_detached=True)
    _validate_state_against(x0_fake, x0_gen, identifier="x0_fake", require_detached=True)

    abs_diff = {
        name: (
            x0_gen.components[name].to(torch.float32) - x0_real.components[name].to(torch.float32)
        ).abs()
        for name in expected_names
    }
    normalizer = (
        adapter.reduce_latent_values(abs_diff, state=x0_gen).to(torch.float32).clamp_min(float(eps))
    )
    squared: dict[str, torch.Tensor] = {}
    for name in expected_names:
        gen = x0_gen.components[name].to(torch.float32)
        real = x0_real.components[name].to(torch.float32)
        fake = x0_fake.components[name].to(torch.float32)
        scale = normalizer
        if gen.ndim > 1:
            scale = normalizer.reshape(normalizer.shape[0], *([1] * (gen.ndim - 1)))
        direction = torch.nan_to_num((fake - real) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        target = (gen - direction).detach()
        squared[name] = 0.5 * (gen - target).square()
    return adapter.reduce_latent_values(squared, state=x0_gen).to(torch.float32).mean()


def tdm_conditional_renoise(
    adapter: BaseAdapter,
    clean_state: LatentState,
    model_noise: LatentState,
    *,
    mid_times: ComponentTimes,
    target_times: ComponentTimes,
    importance_clip: float = 20.0,
    eps: float = _EPS,
    generator: torch.Generator | None = None,
) -> Tuple[NoisedState, torch.Tensor]:
    """Conditionally diffuse one generator clean prediction within its ODE stage.

    This is the linear-flow form of the official TDM transition. ``model_noise``
    carries the noise paired with ``clean_state`` by the selected generator velocity;
    the target state preserves that trajectory noise and adds only the conditional
    variance needed to reach ``target_times``. The returned density ratio corrects
    fake-score training from fresh Gaussian noise to the resulting mixed noise.

    Args:
        adapter: Adapter providing component order and masked forward noising.
        clean_state: Detached/live generator clean prediction by component.
        model_noise: Generator-implied noise paired with the clean prediction.
        mid_times: Lower-noise boundary coordinates ``sigma_mid``.
        target_times: Sampled higher-noise coordinates ``sigma_t``.
        importance_clip: Symmetric cap on the importance ratio.
        eps: Positive numerical floor.

    Returns:
        Conditional noised state and one detached importance ratio per sample.
    """
    adapter = _require_adapter(adapter)
    clean_state = _require_state(clean_state, identifier="tdm clean_state")
    model_noise = _require_state(model_noise, identifier="tdm model_noise")
    expected_names = adapter.trajectory_component_order
    _require_component_order(clean_state, expected_names, identifier="tdm clean_state")
    _validate_state_against(
        model_noise,
        clean_state,
        identifier="tdm model_noise",
        require_detached=True,
    )
    if mid_times.sigma is None or tuple(mid_times.sigma) != expected_names:
        received = None if mid_times.sigma is None else tuple(mid_times.sigma)
        raise ValueError(
            f"expected TDM mid sigma component order {expected_names}, received {received}"
        )
    if target_times.sigma is None or tuple(target_times.sigma) != expected_names:
        received = None if target_times.sigma is None else tuple(target_times.sigma)
        raise ValueError(
            f"expected TDM target sigma component order {expected_names}, received {received}"
        )
    if (
        isinstance(importance_clip, bool)
        or not isinstance(importance_clip, Real)
        or not math.isfinite(float(importance_clip))
        or importance_clip <= 0
    ):
        raise ValueError(f"expected finite TDM importance_clip > 0, received {importance_clip!r}")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, Real)
        or not math.isfinite(float(eps))
        or eps <= 0
    ):
        raise ValueError(f"expected finite TDM eps > 0, received {eps!r}")

    mixed_components: dict[str, torch.Tensor] = {}
    fresh_components: dict[str, torch.Tensor] = {}
    for name in expected_names:
        clean = clean_state.components[name].to(torch.float32)
        implied_noise = model_noise.components[name].to(torch.float32)
        sigma_mid = to_broadcast_tensor(mid_times.sigma[name].to(clean), clean)
        sigma_t = to_broadcast_tensor(target_times.sigma[name].to(clean), clean)
        valid = (
            torch.isfinite(sigma_mid)
            & torch.isfinite(sigma_t)
            & (sigma_mid >= 0)
            & (sigma_mid < sigma_t)
            & (sigma_t < 1)
        )
        if not bool(valid.all().item()):
            raise ValueError(
                f"expected TDM component {name!r} sigmas to satisfy "
                f"0 <= sigma_mid < sigma_t < 1, received "
                f"sigma_mid={mid_times.sigma[name].tolist()}, "
                f"sigma_t={target_times.sigma[name].tolist()}"
            )
        alpha_mid = 1.0 - sigma_mid
        alpha_t = 1.0 - sigma_t
        ratio = alpha_t / alpha_mid.clamp_min(float(eps))
        old_noise_coeff = ratio * sigma_mid
        beta_sq = sigma_t.square() - old_noise_coeff.square()
        if bool((beta_sq < -float(eps)).any().item()):
            raise ValueError(
                f"TDM conditional variance became negative for component {name!r}: "
                f"minimum beta^2={beta_sq.min().item()}"
            )
        beta = beta_sq.clamp_min(0).sqrt()
        if generator is None:
            fresh = torch.randn_like(clean)
        else:
            fresh = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(
                device=clean.device
            )
        mixed = (old_noise_coeff * implied_noise + beta * fresh) / sigma_t.clamp_min(float(eps))
        mixed_components[name] = mixed
        fresh_components[name] = fresh

    # Keep the likelihood calculation in float32, but restore each component's
    # actual latent dtype/device at the strict adapter boundary.
    adapter_noise = LatentState(
        {name: mixed_components[name].to(clean_state.components[name]) for name in expected_names},
        active_masks=clean_state.active_masks,
    )
    noised = adapter.apply_forward_process_noise(clean_state, target_times, adapter_noise)
    mixed_square = {name: mixed_components[name].square() for name in expected_names}
    fresh_square = {name: fresh_components[name].square() for name in expected_names}
    log_ratio = -0.5 * adapter.reduce_latent_values(mixed_square, state=clean_state).to(
        torch.float32
    ) + 0.5 * adapter.reduce_latent_values(fresh_square, state=clean_state).to(torch.float32)
    log_clip = math.log(float(importance_clip))
    importance = torch.exp(log_ratio.clamp(-log_clip, log_clip)).detach()
    return noised, importance


def tdm_fake_loss(
    adapter: BaseAdapter,
    x0_fake: LatentState,
    x0_target: LatentState,
    *,
    sigma: torch.Tensor,
    importance: torch.Tensor,
    snr_gamma: float = 5.0,
) -> torch.Tensor:
    """Importance-weighted, SNR-capped clean-state MSE for the fake score."""
    adapter = _require_adapter(adapter)
    x0_fake = _require_state(x0_fake, identifier="x0_fake")
    x0_target = _require_state(x0_target, identifier="x0_target")
    expected_names = adapter.trajectory_component_order
    _require_component_order(x0_fake, expected_names, identifier="x0_fake")
    _validate_state_against(x0_target, x0_fake, identifier="x0_target", require_detached=True)
    if not isinstance(sigma, torch.Tensor) or not sigma.is_floating_point():
        raise TypeError(
            f"expected floating torch.Tensor sigma, received {type(sigma).__name__}: {sigma!r}"
        )
    if not isinstance(importance, torch.Tensor) or importance.ndim != 1:
        raise TypeError(
            "expected importance as a rank-1 torch.Tensor, "
            f"received {type(importance).__name__} shape="
            f"{None if not isinstance(importance, torch.Tensor) else tuple(importance.shape)}"
        )
    batch_size = next(iter(x0_fake.components.values())).shape[0]
    if importance.shape[0] != batch_size:
        raise ValueError(
            f"expected importance shape ({batch_size},), received {tuple(importance.shape)}"
        )
    if (
        not isinstance(snr_gamma, Real)
        or isinstance(snr_gamma, bool)
        or not math.isfinite(float(snr_gamma))
        or snr_gamma <= 0
    ):
        raise ValueError(f"expected finite snr_gamma > 0, received {snr_gamma!r}")
    sigma_values = sigma.to(dtype=torch.float32)
    if sigma_values.ndim == 0:
        sigma_flat = sigma_values.reshape(1).expand(batch_size)
    else:
        if sigma_values.shape[0] != batch_size:
            raise ValueError(
                f"expected sigma batch size {batch_size}, received shape {tuple(sigma_values.shape)}"
            )
        sigma_flat = sigma_values.reshape(batch_size, -1)
        if not torch.equal(sigma_flat, sigma_flat[:, :1].expand_as(sigma_flat)):
            raise ValueError(
                "expected a single sigma per sample, received varying extra dimensions "
                f"shape={tuple(sigma_values.shape)}"
            )
        sigma_flat = sigma_flat[:, 0]
    snr = ((1.0 - sigma_flat) ** 2) / torch.clamp(sigma_flat.square(), min=_EPS)
    weight = torch.minimum(snr, torch.full_like(snr, float(snr_gamma)))
    squared = {
        name: (
            x0_fake.components[name].to(torch.float32)
            - x0_target.components[name].to(torch.float32)
        ).square()
        for name in expected_names
    }
    per_sample = adapter.reduce_latent_values(squared, state=x0_fake).to(torch.float32)
    return (per_sample * importance.to(per_sample) * weight).mean()


def revised_x0_loss(
    adapter: BaseAdapter,
    x0_student: LatentState,
    correction: LatentState,
    normalizer_reference: LatentState,
    *,
    use_huber: bool,
    huber_c: float,
    eps: float = _EPS,
) -> torch.Tensor:
    """Regress the student's clean state toward ``x0_student + correction``.

    Every TDM-R1 generator term has this shape and differs only in which direction
    revises the target and which reference sets the scale: the distribution-matching
    term follows the teacher against the fake score, the guidance reward follows the
    teacher's conditional-minus-unconditional direction, and the surrogate reward
    follows the surrogate's guided direction against its frozen reference. Sharing one
    implementation keeps the three from drifting apart in normalization or stop-grad
    placement, which is where a silently wrong objective would hide.

    Args:
        adapter: Adapter providing component order and latent reduction.
        x0_student: Live clean prediction carrying the gradient.
        correction: Detached direction added to the student to form the target.
        normalizer_reference: Detached state whose distance to the student scales the loss.
        use_huber: Whether to use the Pseudo-Huber metric instead of squared error.
        huber_c: Finite positive Pseudo-Huber constant.
        eps: Lower clamp on the normalizer.

    Returns:
        Scalar loss whose gradient drives the student along ``correction``.
    """
    adapter = _require_adapter(adapter)
    x0_student = _require_state(x0_student, identifier="x0_student")
    correction = _require_state(correction, identifier="correction")
    normalizer_reference = _require_state(normalizer_reference, identifier="normalizer_reference")
    if not isinstance(use_huber, bool):
        raise TypeError(
            f"expected use_huber as a bool, received {type(use_huber).__name__}: {use_huber!r}"
        )
    if (
        not isinstance(huber_c, Real)
        or isinstance(huber_c, bool)
        or not math.isfinite(float(huber_c))
        or huber_c <= 0
    ):
        raise ValueError(f"expected finite huber_c > 0, received {huber_c!r}")
    if (
        not isinstance(eps, Real)
        or isinstance(eps, bool)
        or not math.isfinite(float(eps))
        or eps <= 0
    ):
        raise ValueError(f"expected finite eps > 0, received {eps!r}")
    expected_names = adapter.trajectory_component_order
    _require_component_order(x0_student, expected_names, identifier="x0_student")
    _validate_state_against(correction, x0_student, identifier="correction", require_detached=True)
    _validate_state_against(
        normalizer_reference,
        x0_student,
        identifier="normalizer_reference",
        require_detached=True,
    )

    abs_diff = {
        name: (
            x0_student.components[name].detach().to(torch.float32)
            - normalizer_reference.components[name].to(torch.float32)
        ).abs()
        for name in expected_names
    }
    normalizer = (
        adapter.reduce_latent_values(abs_diff, state=x0_student)
        .to(torch.float32)
        .clamp_min(float(eps))
    )
    residuals: dict[str, torch.Tensor] = {}
    for name in expected_names:
        student = x0_student.components[name].to(torch.float32)
        target = (student.detach() + correction.components[name].to(torch.float32)).detach()
        delta = student - target
        if use_huber:
            c = float(huber_c)
            residuals[name] = torch.sqrt(delta.square() + c * c) - c
        else:
            residuals[name] = 0.5 * delta.square()
    reduced = adapter.reduce_latent_values(residuals, state=x0_student).to(torch.float32)
    return (reduced / normalizer).mean()


def tdm_generator_loss(
    adapter: BaseAdapter,
    x0_student: LatentState,
    x0_real: LatentState,
    x0_fake: LatentState,
    *,
    use_huber: bool,
    huber_c: float,
    eps: float = _EPS,
) -> torch.Tensor:
    """Stop-grad TDM generator surrogate in clean-prediction space."""
    adapter = _require_adapter(adapter)
    x0_student = _require_state(x0_student, identifier="x0_student")
    x0_real = _require_state(x0_real, identifier="x0_real")
    x0_fake = _require_state(x0_fake, identifier="x0_fake")
    if not isinstance(use_huber, bool):
        raise TypeError(
            f"expected use_huber as a bool, received {type(use_huber).__name__}: {use_huber!r}"
        )
    if (
        not isinstance(huber_c, Real)
        or isinstance(huber_c, bool)
        or not math.isfinite(float(huber_c))
        or huber_c <= 0
    ):
        raise ValueError(f"expected finite huber_c > 0, received {huber_c!r}")
    if (
        not isinstance(eps, Real)
        or isinstance(eps, bool)
        or not math.isfinite(float(eps))
        or eps <= 0
    ):
        raise ValueError(f"expected finite eps > 0, received {eps!r}")
    expected_names = adapter.trajectory_component_order
    _require_component_order(x0_student, expected_names, identifier="x0_student")
    _validate_state_against(x0_real, x0_student, identifier="x0_real", require_detached=True)
    _validate_state_against(x0_fake, x0_student, identifier="x0_fake", require_detached=True)

    correction = LatentState(
        {
            name: (
                x0_real.components[name].to(torch.float32)
                - x0_fake.components[name].to(torch.float32)
            ).detach()
            for name in expected_names
        },
        active_masks=x0_student.active_masks,
    )
    return revised_x0_loss(
        adapter,
        x0_student,
        correction,
        x0_real,
        use_huber=use_huber,
        huber_c=huber_c,
        eps=eps,
    )
