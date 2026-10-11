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

from typing import Any, Dict, Mapping, Optional

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...samples import (
    ComponentTimes,
    LatentState,
    NoisedState,
)
from ...utils.base import to_broadcast_tensor
from ...utils.noise_schedule import flow_match_sigma
from .masks import _expand_active_mask


def build_training_component_times(
    adapter: Any,
    primary_timesteps: torch.Tensor,
    *,
    batch: Optional[Mapping[str, Any]],
) -> ComponentTimes:
    if not isinstance(primary_timesteps, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor primary_timesteps for build_training_component_times, "
            f"received {type(primary_timesteps).__name__}: {primary_timesteps!r}"
        )
    if primary_timesteps.ndim != 1:
        raise ValueError(
            "expected primary_timesteps with one scheduler coordinate per sample, shape "
            f"(B,), received {tuple(primary_timesteps.shape)}"
        )
    expected_names = ("latent",)
    if adapter.trajectory_component_order != expected_names:
        raise ValueError(
            f"expected trajectory_component_order {expected_names} for the default "
            f"build_training_component_times, received {adapter.trajectory_component_order}; "
            "override the hook to map the primary coordinate per component"
        )
    sigma = flow_match_sigma(primary_timesteps)
    return ComponentTimes(
        timestep={"latent": primary_timesteps},
        next_timestep={"latent": torch.zeros_like(primary_timesteps)},
        sigma={"latent": sigma},
        next_sigma={"latent": torch.zeros_like(sigma)},
    )


def resolve_replay_projection_times(
    adapter: Any,
    times: ComponentTimes,
    *,
    batch: Optional[Mapping[str, Any]],
) -> ComponentTimes:
    """Prefer authoritative replay sigmas and map only legacy timestep-only data.

    Args:
        adapter: Adapter that maps a primary scheduler coordinate when needed.
        times: Coordinates read from one replay transition.
        batch: Optional stacked batch required by model-specific time mapping.

    Returns:
        The original stored coordinates when sigmas are present, otherwise mapped
        coordinates reconstructed from the stored primary timestep.
    """
    if times.sigma is not None:
        return times
    primary_name = adapter.trajectory_component_order[0]
    return adapter.build_training_component_times(
        times.timestep[primary_name],
        batch=batch,
    )


def project_velocity_to_clean_state(
    adapter: Any,
    state: LatentState,
    times: ComponentTimes,
    velocity: LatentState,
) -> LatentState:
    """Project a noised state to clean data using the adapter's velocity direction."""
    expected_names = adapter.trajectory_component_order
    for argument, value in (("state", state), ("velocity", velocity)):
        if not isinstance(value, LatentState):
            raise TypeError(
                f"expected LatentState for project_velocity_to_clean_state {argument}, "
                f"received {type(value).__name__}"
            )
        if value.component_names != expected_names:
            raise ValueError(
                f"expected project_velocity_to_clean_state {argument} component order "
                f"{expected_names}, received {value.component_names}"
            )
    if times.sigma is None or tuple(times.sigma) != expected_names:
        received = None if times.sigma is None else tuple(times.sigma)
        raise ValueError(
            f"expected sigma component order {expected_names} for "
            f"project_velocity_to_clean_state, received {received}"
        )
    direction = adapter.flow_velocity_direction
    if direction not in ("noise", "data"):
        raise ValueError(
            "expected flow_velocity_direction to be 'noise' or 'data' for "
            f"project_velocity_to_clean_state, received {direction!r}"
        )

    projected: Dict[str, torch.Tensor] = {}
    for name in expected_names:
        component_state = state.components[name]
        component_velocity = velocity.components[name]
        if component_state.ndim < 2:
            raise ValueError(
                f"expected project_velocity_to_clean_state component {name!r} to be batched "
                f"with shape (B, ...), received {tuple(component_state.shape)}"
            )
        if component_velocity.shape != component_state.shape:
            raise ValueError(
                f"expected velocity component {name!r} to match state shape "
                f"{tuple(component_state.shape)}, received {tuple(component_velocity.shape)}"
            )
        if component_velocity.device != component_state.device:
            raise ValueError(
                f"expected velocity component {name!r} on state device "
                f"{component_state.device}, received {component_velocity.device}"
            )
        sigma = times.sigma[name]
        batch_size = component_state.shape[0]
        if sigma.ndim > 1 or sigma.numel() not in (1, batch_size):
            raise ValueError(
                f"expected sigma for component {name!r} to hold one value per sample with "
                f"shape ({batch_size},), received {tuple(sigma.shape)}"
            )
        compute_dtype = torch.promote_types(component_state.dtype, component_velocity.dtype)
        compute_state = component_state.to(dtype=compute_dtype)
        compute_velocity = component_velocity.to(dtype=compute_dtype)
        sigma = to_broadcast_tensor(sigma, compute_state)
        sign = -1.0 if direction == "noise" else 1.0
        projected[name] = compute_state + sign * sigma * compute_velocity
    return LatentState(projected, active_masks=state.active_masks)


def apply_forward_process_noise(
    adapter: Any,
    clean_state: LatentState,
    times: ComponentTimes,
    noise: LatentState,
) -> NoisedState:
    expected_names = adapter.trajectory_component_order
    for argument, state in (("clean_state", clean_state), ("noise", noise)):
        if not isinstance(state, LatentState):
            raise TypeError(
                f"expected LatentState for apply_forward_process_noise {argument}, "
                f"received {type(state).__name__}"
            )
        if state.component_names != expected_names:
            raise ValueError(
                f"expected apply_forward_process_noise {argument} component order "
                f"{expected_names}, received {state.component_names}"
            )
    if times.sigma is None or tuple(times.sigma) != expected_names:
        received = None if times.sigma is None else tuple(times.sigma)
        raise ValueError(
            f"expected sigma component order {expected_names} for "
            f"apply_forward_process_noise, received {received}"
        )
    direction = adapter.flow_velocity_direction
    if direction not in ("noise", "data"):
        raise ValueError(
            "expected flow_velocity_direction to be 'noise' or 'data' for "
            f"apply_forward_process_noise, received {direction!r}"
        )
    velocity_sign = 1.0 if direction == "noise" else -1.0
    primary_name = expected_names[0]
    primary_clean = clean_state.components[primary_name]
    if primary_clean.ndim < 2:
        raise ValueError(
            f"expected apply_forward_process_noise clean_state component {primary_name!r} to be "
            f"a batched tensor with a leading batch dimension and shape (B, ...), received shape "
            f"{tuple(primary_clean.shape)}"
        )
    batch_size = primary_clean.shape[0]
    noised: Dict[str, torch.Tensor] = {}
    target_velocity: Dict[str, torch.Tensor] = {}
    for name in expected_names:
        clean_latents = clean_state.components[name]
        component_noise = noise.components[name]
        if clean_latents.ndim < 2 or clean_latents.shape[0] != batch_size:
            raise ValueError(
                f"expected batched clean_state component {name!r} with shape (B, ...) and "
                f"batch size {batch_size}, received {tuple(clean_latents.shape)}"
            )
        if component_noise.shape != clean_latents.shape:
            raise ValueError(
                f"expected noise component {name!r} to match the clean shape "
                f"{tuple(clean_latents.shape)}, received {tuple(component_noise.shape)}"
            )
        if (
            component_noise.dtype != clean_latents.dtype
            or component_noise.device != clean_latents.device
        ):
            raise ValueError(
                f"expected noise component {name!r} to match the clean dtype/device "
                f"({clean_latents.dtype}, {clean_latents.device}), received "
                f"({component_noise.dtype}, {component_noise.device})"
            )
        sigma = times.sigma[name]
        if sigma.ndim > 1 or sigma.numel() not in (1, batch_size):
            raise ValueError(
                f"expected sigma for component {name!r} to hold one value per sample with "
                f"shape ({batch_size},), received {tuple(sigma.shape)}"
            )
        sigma = to_broadcast_tensor(sigma, clean_latents)
        component_noised = (1 - sigma) * clean_latents + sigma * component_noise
        component_target = velocity_sign * (component_noise - clean_latents)
        if clean_state.active_masks is not None:
            # The caller drew the noise at full shape, so the RNG stream is
            # mask-independent; masking only decides which elements move, so inactive
            # conditioning stays clean and contributes no training signal.
            mask = _expand_active_mask(
                clean_state.active_masks[name],
                clean_latents,
                component=name,
                operation="apply_forward_process_noise",
            )
            component_noised = torch.where(mask, component_noised, clean_latents)
            component_target = torch.where(
                mask, component_target, torch.zeros_like(component_target)
            )
        noised[name] = component_noised
        target_velocity[name] = component_target
    return NoisedState(
        state=LatentState(noised, active_masks=clean_state.active_masks),
        target_velocity=LatentState(target_velocity, active_masks=clean_state.active_masks),
        noise=noise,
    )


def add_forward_process_noise(
    adapter: Any,
    clean_state: LatentState,
    times: ComponentTimes,
    *,
    generator: Optional[torch.Generator],
) -> NoisedState:
    expected_names = ("latent",)
    if adapter.trajectory_component_order != expected_names:
        raise ValueError(
            f"expected trajectory_component_order {expected_names} for the default "
            f"add_forward_process_noise, received {adapter.trajectory_component_order}; "
            "override the hook to draw noise in component order"
        )
    if clean_state.component_names != expected_names:
        raise ValueError(
            f"expected exactly components {expected_names} for default noising, "
            f"received {clean_state.component_names}"
        )
    if times.sigma is None or tuple(times.sigma) != expected_names:
        received = None if times.sigma is None else tuple(times.sigma)
        raise ValueError(
            f"expected sigma component order {expected_names} before the default noise draw, "
            f"received {received}"
        )
    clean_latents = clean_state.components["latent"]
    # The single draw happens here, in component order, so the RNG stream stays
    # reproducible; the application hook below consumes no randomness.
    noise = randn_tensor(
        clean_latents.shape,
        generator=generator,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    return adapter.apply_forward_process_noise(
        clean_state,
        times,
        LatentState({"latent": noise}),
    )
