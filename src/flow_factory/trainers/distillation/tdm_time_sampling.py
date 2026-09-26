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

"""Conditional interval sampling in actual flow noise coordinates."""

from __future__ import annotations

import inspect
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterator

import torch

from ...hparams.training_args.tdm import TDMQuerySamplingPolicy
from ...utils.noise_schedule import TIMESTEP_MAX, flow_match_sigma


@dataclass(frozen=True)
class TDMGenerationProvenance:
    """Store rollout-only data required to replay one TDM query policy."""

    source_shift: float

    def __post_init__(self) -> None:
        """Require a positive finite source-to-actual shift."""
        if not math.isfinite(self.source_shift) or self.source_shift <= 0:
            raise ValueError(
                "TDM generation source_shift must be positive and finite, "
                f"received {self.source_shift!r}"
            )


@dataclass(frozen=True)
class RationalFlowShift:
    """Map between source and actual sigma for one rollout batch."""

    gamma: torch.Tensor

    def __post_init__(self) -> None:
        """Require positive finite shift factors."""
        if not isinstance(self.gamma, torch.Tensor):
            raise TypeError(
                "TDM source-uniform transform expected a torch.Tensor gamma, "
                f"received {type(self.gamma).__name__}"
            )
        if not bool((torch.isfinite(self.gamma) & (self.gamma > 0)).all()):
            raise ValueError("TDM generation shifts must be positive and finite")

    def to_source(self, actual_sigma: torch.Tensor) -> torch.Tensor:
        """Invert actual flow sigmas into the unshifted source coordinate."""
        gamma = self.gamma.to(device=actual_sigma.device, dtype=actual_sigma.dtype)
        return actual_sigma / (gamma - (gamma - 1) * actual_sigma)

    def to_actual(self, source_sigma: torch.Tensor) -> torch.Tensor:
        """Apply the rollout flow shift to source sigmas."""
        gamma = self.gamma.to(device=source_sigma.device, dtype=source_sigma.dtype)
        return gamma * source_sigma / (1 + (gamma - 1) * source_sigma)


@contextmanager
def capture_generation_shift(scheduler: Any) -> Iterator[list[float]]:
    """Capture one rollout's effective shift while preserving the scheduler method.

    Args:
        scheduler: Primary scheduler used by the current generation call.

    Yields:
        A list containing the captured shift after a successful generation call.

    Raises:
        ValueError: No schedule was built, shifts differ within the rollout, or the
            schedule is not a supported pure flow shift.
    """
    original = scheduler.set_timesteps
    signature = inspect.signature(original)
    absent = object()
    instance_method = vars(scheduler).get("set_timesteps", absent)
    shifts: list[float] = []

    @wraps(original)
    def capture(*args: Any, **kwargs: Any) -> Any:
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        shift = _generation_shift(scheduler, arguments.arguments.get("mu"))
        result = original(*args, **kwargs)
        if shifts and shift != shifts[0]:
            raise ValueError("source_uniform requires one effective shift per rollout")
        if not shifts:
            shifts.append(shift)
        return result

    scheduler.set_timesteps = capture
    try:
        yield shifts
        if not shifts:
            raise ValueError("source_uniform did not capture a set_timesteps() call")
    finally:
        if instance_method is absent:
            del scheduler.set_timesteps
        else:
            scheduler.set_timesteps = instance_method


def _generation_shift(scheduler: Any, mu: float | None) -> float:
    """Resolve the effective rational flow shift from the actual generation call."""
    config = scheduler.config
    modifiers = [
        name
        for name in (
            "shift_terminal",
            "invert_sigmas",
            "use_karras_sigmas",
            "use_exponential_sigmas",
            "use_beta_sigmas",
        )
        if config.get(name, False)
    ]
    if "use_flow_sigmas" in config and not config.use_flow_sigmas:
        modifiers.append("use_flow_sigmas=False")
    if modifiers:
        raise ValueError(
            "source_uniform requires a pure flow-shift schedule; "
            f"unsupported schedule modifiers: {modifiers!r}"
        )
    if config.get("use_dynamic_shifting", False):
        kind = config.get("time_shift_type", "exponential")
        if mu is None or kind not in ("linear", "exponential"):
            raise ValueError(f"Cannot resolve generation shift: type={kind!r}, mu={mu!r}")
        shift = (
            (math.exp(mu) if mu <= math.log(sys.float_info.max) else float("inf"))
            if kind == "exponential"
            else mu
        )
    else:
        shift = getattr(scheduler, "shift", config.get("flow_shift"))
    if shift is None or not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"Generation shift must be positive and finite, received {shift!r}")
    return float(shift)


def sample_interval_sigma(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    distribution: str,
    logit_mean: float,
    logit_std: float,
    source_transform: RationalFlowShift | None = None,
) -> torch.Tensor:
    """Draw one independent sigma per interval using float64 probability arithmetic.

    Args:
        lower: Actual lower noise coordinates in [0, 1].
        upper: Actual upper noise coordinates in [0, 1], above lower.
        distribution: Actual-uniform, conditional logit-normal, or source-uniform.
        logit_mean: Mean of the untruncated normal in logit space.
        logit_std: Positive standard deviation of that normal.
        source_transform: Rollout-owned source/actual transform; source-uniform only.

    Returns:
        Float64 noise coordinates; callers protect representable output interiors.
    """
    lower, upper = lower.double(), upper.double()
    if not bool(((0 <= lower) & (lower < upper) & (upper <= 1)).all()):
        raise ValueError("TDM sigma intervals require 0 <= lower < upper <= 1")
    fraction = torch.rand(lower.shape, device=lower.device, dtype=torch.float64)
    eps = torch.finfo(torch.float64).eps
    fraction = fraction.clamp(eps, 1 - eps)
    if distribution == "actual_uniform":
        return lower + fraction * (upper - lower)
    if distribution == "source_uniform":
        if source_transform is None:
            raise ValueError("source_uniform requires the shift captured during generation")
        u_lower = source_transform.to_source(lower)
        u_upper = source_transform.to_source(upper)
        uniform = u_lower + fraction * (u_upper - u_lower)
        return source_transform.to_actual(uniform)
    if distribution != "conditional_logit_normal":
        raise ValueError(f"Unsupported TDM query distribution: {distribution!r}")

    z_lower = (torch.logit(lower) - logit_mean) / logit_std
    z_upper = (torch.logit(upper) - logit_mean) / logit_std
    # Reflect right-tail intervals before evaluating the CDF, avoiding subtraction
    # of probabilities rounded to one. erfc also preserves far left-tail mass.
    reflect = z_lower > 0
    cdf_lower_z = torch.where(reflect, -z_upper, z_lower)
    cdf_upper_z = torch.where(reflect, -z_lower, z_upper)
    cdf_lower = 0.5 * torch.erfc(-cdf_lower_z / math.sqrt(2))
    cdf_upper = 0.5 * torch.erfc(-cdf_upper_z / math.sqrt(2))
    probability_lower = torch.nextafter(cdf_lower, cdf_upper)
    probability_upper = torch.nextafter(cdf_upper, cdf_lower)
    if not bool(((cdf_lower < cdf_upper) & (probability_lower < probability_upper)).all()):
        raise ValueError(
            "TDM conditional_logit_normal interval has no reliable float64 probability "
            f"interior: lower={lower.tolist()}, upper={upper.tolist()}, "
            f"tdm_query_logit_mean={logit_mean}, tdm_query_logit_std={logit_std}"
        )
    quantile = torch.where(reflect, 1 - fraction, fraction)
    probability = cdf_lower + quantile * (cdf_upper - cdf_lower)
    probability = torch.minimum(torch.maximum(probability, probability_lower), probability_upper)
    z = torch.special.ndtri(probability)
    z = torch.where(reflect, -z, z)
    return torch.sigmoid(logit_mean + logit_std * z)


def sample_query_timestep(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    policy: TDMQuerySamplingPolicy,
    source_transform: RationalFlowShift | None,
) -> torch.Tensor:
    """Sample one query timestep while preserving the legacy uniform path exactly.

    Args:
        lower: Actual lower scheduler timesteps.
        upper: Actual upper scheduler timesteps.
        policy: Immutable TDM query policy.
        source_transform: Captured source-coordinate transform when required.

    Returns:
        Sampled scheduler timesteps in the input dtype and device.
    """
    if policy.distribution == "actual_uniform":
        fraction = torch.rand(lower.shape, device=lower.device, dtype=lower.dtype)
        precision = torch.finfo(fraction.dtype)
        fraction = fraction.clamp(min=precision.eps, max=1.0 - precision.eps)
        return lower + (upper - lower) * fraction
    sigma = sample_interval_sigma(
        flow_match_sigma(lower.double()),
        flow_match_sigma(upper.double()),
        distribution=policy.distribution,
        logit_mean=policy.logit_mean,
        logit_std=policy.logit_std,
        source_transform=source_transform,
    )
    return (sigma * TIMESTEP_MAX).to(lower.dtype)
