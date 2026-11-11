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

"""DiffusionART scalar guidance and Euler reference-transition trust region.

The paper uses denoising-direction velocity. Flow-Factory uses dx/dsigma, so
its target is old_velocity minus endpoint guidance (sigma decreases in rollout).
"""

import torch


def _mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Average only active output coordinates, independently for each sample."""
    if mask is None:
        return values.flatten(1).mean(1)
    mask = torch.broadcast_to(mask, values.shape)
    count = mask.flatten(1).sum(1)
    return torch.where(mask, values, 0.0).flatten(1).sum(1) / count.clamp_min(1)


@torch.no_grad()
def endpoint_guidance(
    endpoints: torch.Tensor,
    rewards: torch.Tensor,
    *,
    epsilon: float,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one RMS-normalized covariance direction for a complete prompt group.

    Rewards are already combined by the framework's configured group reducer.
    Constant rewards and zero covariance produce exactly zero guidance. Frozen
    coordinates are excluded from both covariance and its RMS normalization.
    """
    if endpoints.ndim < 2 or endpoints.shape[0] < 2 or rewards.shape != (endpoints.shape[0],):
        raise ValueError("ART covariance requires K>=2 endpoints and exactly K scalar rewards")
    x, r = endpoints.detach().float(), rewards.detach().to(endpoints.device).float()
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(r).all()):
        raise ValueError("ART endpoints and rewards must be finite")
    mask = None
    if active_mask is not None:
        masks = torch.broadcast_to(active_mask, x.shape)
        if not bool((masks == masks[:1]).all()):
            raise ValueError("ART prompt-group endpoints must share their active-coordinate mask")
        mask = masks[:1]
    if bool((r == r[0]).all()):
        return torch.zeros_like(x[:1])
    centered = r - r.mean()
    covariance = ((x - x.mean(0)) * centered.reshape(-1, *([1] * (x.ndim - 1)))).sum(
        0, keepdim=True
    ) / (len(r) - 1)
    if mask is not None:
        covariance = torch.where(mask, covariance, 0.0)
    # Match the upstream historical direction: reward MAD then RMS, with an
    # explicit epsilon floor. No clipping or per-microbatch statistics enter.
    direction = covariance / centered.abs().mean()
    rms = _mean(direction.square(), mask).sqrt()
    return direction / rms.clamp_min(epsilon).reshape(-1, *([1] * (x.ndim - 1)))


def art_objective(
    velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    guidance: torch.Tensor,
    sigma: torch.Tensor,
    dt: torch.Tensor,
    *,
    guidance_scale: float,
    kl_threshold: float,
    reference_noise_level: float,
    epsilon: float,
    active_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute sigma-weighted MSE and a detached, direction-aware ODE trust mask.

    Only current velocity is differentiable. The virtual Flow-SDE transition
    supplies the KL metric even though the actual rollout is deterministic ODE.
    A large move is masked only while its current descent direction points
    farther away from the rollout anchor; corrective moves remain trainable.
    """
    if velocity.shape != old_velocity.shape or velocity.shape != guidance.shape:
        raise ValueError("ART current/old velocities and guidance must have identical shapes")
    batch_size = velocity.shape[0]
    s, delta = sigma.detach().float().reshape(-1), dt.detach().float().reshape(-1)
    if s.shape != (batch_size,) or delta.shape != (batch_size,):
        raise ValueError("ART requires one sigma and Euler delta per sample")
    if (
        not bool(torch.isfinite(s).all())
        or not bool(torch.isfinite(delta).all())
        or bool(((s <= 0) | (s > 1) | (delta >= 0) | (s + delta < -epsilon)).any())
    ):
        raise ValueError(
            "ART requires 0 < sigma <= 1 and a negative Euler delta ending at sigma >= 0"
        )
    v, old, g = velocity.float(), old_velocity.detach().float(), guidance.detach().float()
    target = old - guidance_scale * g
    residual, movement = target - v, v - old
    energy = _mean(movement.square(), active_mask)
    alignment = _mean(movement * residual.detach(), active_mask)
    initial = torch.isclose(s, torch.ones_like(s), atol=epsilon, rtol=0)
    noise2 = reference_noise_level**2
    variance = torch.where(
        initial, torch.full_like(s, noise2), noise2 * s / (1 - s).clamp_min(epsilon) * (-delta)
    )
    coefficient = torch.where(initial, delta, delta * (1 + 0.5 * noise2))
    kl = coefficient.square() * energy / (2 * variance.clamp_min(epsilon**2))
    active = _mean(g.square(), active_mask) > 0
    keep = (active & ~((kl >= kl_threshold) & (alignment > 0))).detach()
    mse = s * _mean(residual.square(), active_mask)
    return {
        "per_sample_loss": torch.where(keep, mse, 0.0),
        "target": target,
        "keep": keep,
        "kl": kl.detach(),
        "alignment": alignment.detach(),
    }
