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

# src/flow_factory/utils/noise_schedule.py
"""Utility functions for noise schedule and time sampling.

``timestep_range=(frac_lo, frac_hi)`` is a **fraction along the denoising axis**
from scheduler time 1000 (noisy) toward 0 (clean). Mapping:

    t_scheduler = TIMESTEP_MAX * (1 - frac)

So ``(0, 0.99)`` yields ``t ∈ [TIMESTEP_MAX * 0.01, TIMESTEP_MAX]`` (e.g. [10, 1000]
when TIMESTEP_MAX=1000). All samplers return **scheduler-scale** timesteps in
``[0, TIMESTEP_MAX]``; trainers pass them to ``adapter.forward(t=...)`` without
extra scaling. Use ``flow_match_sigma(t) = t / TIMESTEP_MAX`` for linear
flow interpolation ``x_t = (1-σ) x_0 + σ ε``.

All samplers accept an optional ``generator`` argument for reproducible /
cross-rank-deterministic draws:

* ``generator=None`` (default): use the global default RNG on ``device``.
* ``generator is not None``: every internal random op runs on
  ``generator.device``; the final tensor is moved to ``device`` before return.
  This lets callers seed with a CPU generator while the output lives on GPU,
  guaranteeing that two ranks with the same seed produce byte-identical
  timesteps regardless of their device placement.
"""

from typing import Optional, Tuple, Union

import torch

from .precision import within_one_native_ulp

TIMESTEP_MAX = 1000.0


def flow_match_sigma(t_scheduler: torch.Tensor) -> torch.Tensor:
    """Map a valid scheduler timestep to flow-matching sigma.

    Args:
        t_scheduler: Scheduler-scale tensor in ``[0, TIMESTEP_MAX]``.

    Returns:
        Sigma in ``[0, 1]`` with the caller's floating dtype, or the default
        floating dtype for integer inputs.

    Raises:
        TypeError: If ``t_scheduler`` is not a tensor.
        ValueError: If any timestep is non-finite or outside the public range.
    """
    if not isinstance(t_scheduler, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor t_scheduler, received "
            f"{type(t_scheduler).__name__}: {t_scheduler!r}"
        )
    if (
        not bool(torch.isfinite(t_scheduler).all())
        or bool((t_scheduler < 0).any())
        or bool((t_scheduler > TIMESTEP_MAX).any())
    ):
        raise ValueError(
            f"expected t_scheduler in [0, {TIMESTEP_MAX:g}], received " f"{t_scheduler.tolist()}"
        )
    return _flow_match_sigma_unchecked(t_scheduler)


def _flow_match_sigma_unchecked(t_scheduler: torch.Tensor) -> torch.Tensor:
    """Convert an already-validated scheduler tensor without host synchronization."""
    output_dtype = (
        t_scheduler.dtype if t_scheduler.is_floating_point() else torch.get_default_dtype()
    )
    # CUDA float32 division may round the largest representable timestep below
    # TIMESTEP_MAX back to sigma == 1. Compute the tiny coordinate mapping in
    # float64, then restore the caller's dtype so strict open intervals remain
    # strict and CPU/GPU produce the same result.
    return (t_scheduler.to(torch.float64) / TIMESTEP_MAX).clamp(0.0, 1.0).to(output_dtype)


def validate_flow_match_coordinates(
    t_scheduler: torch.Tensor,
    sigma: torch.Tensor,
    *,
    identifier: str = "flow-matching coordinates",
) -> None:
    """Validate redundant flow-matching coordinates within one native ULP.

    ``t_scheduler`` and ``sigma`` may be rounded independently when one is
    materialized as ``sigma * TIMESTEP_MAX``. Comparing in sigma space with the
    larger native unit in the last place (ULP) accepts that representation noise
    without hiding a semantic schedule mismatch.

    Args:
        t_scheduler: Scheduler-scale timesteps in ``[0, TIMESTEP_MAX]``.
        sigma: Flow-matching sigma coordinates in ``[0, 1]``.

    Raises:
        TypeError: If timestep is not a real numeric tensor or sigma is not floating.
        ValueError: If shape, device, domain, or coordinate relation is invalid.
    """
    if (
        not isinstance(t_scheduler, torch.Tensor)
        or t_scheduler.dtype == torch.bool
        or t_scheduler.is_complex()
    ):
        raise TypeError(
            f"expected {identifier} t_scheduler as real numeric torch.Tensor, received "
            f"{type(t_scheduler).__name__}/{getattr(t_scheduler, 'dtype', None)}"
        )
    if not isinstance(sigma, torch.Tensor) or not sigma.is_floating_point():
        raise TypeError(
            f"expected {identifier} sigma as floating torch.Tensor, received "
            f"{type(sigma).__name__}/{getattr(sigma, 'dtype', None)}"
        )
    if t_scheduler.shape != sigma.shape:
        raise ValueError(
            f"expected {identifier} shapes to match, received "
            f"t_scheduler={tuple(t_scheduler.shape)} and sigma={tuple(sigma.shape)}"
        )
    if t_scheduler.device != sigma.device:
        raise ValueError(
            f"expected {identifier} devices to match, received "
            f"t_scheduler={t_scheduler.device} and sigma={sigma.device}"
        )
    valid = (
        torch.isfinite(t_scheduler)
        & torch.isfinite(sigma)
        & (t_scheduler >= 0)
        & (t_scheduler <= TIMESTEP_MAX)
        & (sigma >= 0)
        & (sigma <= 1)
    )
    if not bool(valid.all()):
        raise ValueError(
            f"expected {identifier} timestep and sigma finite with timestep in "
            f"[0, {TIMESTEP_MAX:g}] and sigma in [0, 1], received "
            f"timestep={t_scheduler.tolist()} and sigma={sigma.tolist()}"
        )

    expected_sigma = _flow_match_sigma_unchecked(t_scheduler)
    if not within_one_native_ulp(expected_sigma, sigma):
        raise ValueError(
            f"expected {identifier} timestep == sigma * {TIMESTEP_MAX:g} within one native "
            f"ULP, received timestep={t_scheduler.tolist()} and sigma={sigma.tolist()}"
        )


def fraction_range_to_t_bounds(frac_lo: float, frac_hi: float) -> Tuple[float, float]:
    """Return (t_min, t_max) in scheduler scale for fraction range [frac_lo, frac_hi]."""
    t_min = TIMESTEP_MAX * (1.0 - frac_hi)
    t_max = TIMESTEP_MAX * (1.0 - frac_lo)
    return t_min, t_max


def _rng_device(
    generator: Optional[torch.Generator],
    fallback: torch.device,
) -> torch.device:
    """Choose the device random ops must run on.

    When ``generator`` is supplied, every random op MUST execute on
    ``generator.device`` (PyTorch constraint); otherwise use ``fallback``.
    """
    return generator.device if generator is not None else fallback


def _normalize_timestep_range(
    timestep_range: Union[float, Tuple[float, float]],
) -> Tuple[float, float]:
    """Coerce ``timestep_range`` to a ``(frac_lo, frac_hi)`` pair."""
    if isinstance(timestep_range, (list, tuple)):
        return float(timestep_range[0]), float(timestep_range[1])
    return 0.0, float(timestep_range)


class TimeSampler:
    """Continuous and discrete time sampler for flow matching training."""

    @staticmethod
    def _raw_logit_normal_unit(
        num_rows: int,
        device: torch.device,
        stratified: bool,
        logit_mean: float,
        logit_std: float,
        time_shift: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample one ``raw`` value in (0, 1) per row with logit-normal + optional shift warp.

        Args:
            generator: If given, all ``torch.rand / randn / randperm`` calls are
                routed through it on ``generator.device``; the final tensor is
                then moved to ``device``.
        """
        rng_device = _rng_device(generator, device)

        if stratified:
            u_base = torch.rand(num_rows, generator=generator, device=rng_device)
            base = (torch.arange(num_rows, device=rng_device) + u_base) / num_rows
            normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
            u_standard = normal_dist.icdf(torch.clamp(base, 1e-7, 1 - 1e-7))
            perm = torch.randperm(num_rows, generator=generator, device=rng_device)
            u_standard = u_standard[perm]
        else:
            # ``torch.randn`` accepts ``generator``; stays on ``rng_device``.
            u_standard = torch.randn(num_rows, generator=generator, device=rng_device)

        u = u_standard * logit_std + logit_mean
        raw = torch.sigmoid(u)
        raw = time_shift * raw / (1 + (time_shift - 1) * raw)
        raw = torch.clamp(raw, min=0.01, max=1.0 - 1e-6)
        return raw.to(device)

    @staticmethod
    def logit_normal_shifted(
        batch_size: int,
        num_timesteps: int,
        timestep_range: Union[float, Tuple[float, float]],
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        time_shift: float = 3.0,
        device: torch.device = torch.device("cpu"),
        stratified: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Logit-normal time sampling; returns scheduler-scale timesteps in ``[0, TIMESTEP_MAX]``.

        ``timestep_range`` is interpreted as ``(frac_lo, frac_hi)`` (fraction along 1000→0).
        A unit interval sample ``raw`` is mapped to ``frac = frac_lo + raw * (frac_hi - frac_lo)``,
        then ``t = TIMESTEP_MAX * (1 - frac)``.

        Args:
            generator: Optional ``torch.Generator`` for deterministic draws.
                When supplied, the same ``generator.initial_seed()`` produces
                byte-identical output on any rank.
        """
        frac_lo, frac_hi = _normalize_timestep_range(timestep_range)

        raw = TimeSampler._raw_logit_normal_unit(
            num_timesteps,
            device,
            stratified,
            logit_mean,
            logit_std,
            time_shift,
            generator=generator,
        )
        frac = frac_lo + raw * (frac_hi - frac_lo)
        t = TIMESTEP_MAX * (1.0 - frac)
        return t.unsqueeze(1).expand(num_timesteps, batch_size)

    @staticmethod
    def independent_logit_normal_shifted(
        batch_size: int,
        num_timesteps: int,
        timestep_range: Union[float, Tuple[float, float]],
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        time_shift: float = 1.0,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw an independent logit-normal coordinate per term and sample.

        Unlike :meth:`logit_normal_shifted`, this offline-oriented method
        materializes ``(num_timesteps, batch_size)`` rather than expanding one
        coordinate across each batch row.
        """
        _require_positive_int(batch_size, "batch_size")
        _require_positive_int(num_timesteps, "num_timesteps")
        output_device = torch.device(device)
        rng_device = _rng_device(generator, output_device)
        u_standard = torch.randn(
            (num_timesteps, batch_size),
            generator=generator,
            device=rng_device,
        )
        raw = torch.sigmoid(u_standard * logit_std + logit_mean)
        raw = time_shift * raw / (1 + (time_shift - 1) * raw)
        raw = torch.clamp(raw, min=0.01, max=1.0 - 1e-6)
        frac_lo, frac_hi = _normalize_timestep_range(timestep_range)
        frac = frac_lo + raw * (frac_hi - frac_lo)
        return (TIMESTEP_MAX * (1.0 - frac)).to(output_device)

    @staticmethod
    def uniform(
        batch_size: int,
        num_timesteps: int,
        timestep_range: Union[float, Tuple[float, float]],
        time_shift: float = 1.0,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Uniform sampling over fraction interval, mapped to ``[0, TIMESTEP_MAX]``.

        Optional ``time_shift`` warps the fraction before mapping. ``generator``
        semantics are identical to :meth:`logit_normal_shifted`.
        """
        frac_lo, frac_hi = _normalize_timestep_range(timestep_range)
        rng_device = _rng_device(generator, device)

        rand_u = torch.rand(num_timesteps, generator=generator, device=rng_device)
        normalized = (torch.arange(num_timesteps, device=rng_device) + rand_u) / num_timesteps
        f = frac_lo + normalized * (frac_hi - frac_lo)
        perm = torch.randperm(num_timesteps, generator=generator, device=rng_device)
        f = f[perm]
        if abs(time_shift - 1.0) > 1e-6:
            f = time_shift * f / (1 + (time_shift - 1) * f)
        t = TIMESTEP_MAX * (1.0 - f)
        return t.to(device).unsqueeze(1).expand(-1, batch_size)

    @staticmethod
    def independent_uniform(
        batch_size: int,
        num_timesteps: int,
        timestep_range: Union[float, Tuple[float, float]],
        time_shift: float = 1.0,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw an independent uniform coordinate per term and sample."""
        _require_positive_int(batch_size, "batch_size")
        _require_positive_int(num_timesteps, "num_timesteps")
        output_device = torch.device(device)
        rng_device = _rng_device(generator, output_device)
        frac_lo, frac_hi = _normalize_timestep_range(timestep_range)
        fraction = torch.rand(
            (num_timesteps, batch_size),
            generator=generator,
            device=rng_device,
        )
        fraction = frac_lo + fraction * (frac_hi - frac_lo)
        if abs(time_shift - 1.0) > 1e-6:
            fraction = time_shift * fraction / (1 + (time_shift - 1) * fraction)
        return (TIMESTEP_MAX * (1.0 - fraction)).to(output_device)

    @staticmethod
    def discrete(
        batch_size: int,
        num_train_timesteps: int,
        scheduler_timesteps: torch.Tensor,
        timestep_range: Union[float, Tuple[float, float]] = 1.0,
        include_init: bool = True,
        force_init: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Discrete stratified sampling from ``scheduler_timesteps`` (scheduler scale, e.g. 0–1000).

        ``timestep_range=(frac_lo, frac_hi)`` keeps indices ``i`` whose ``ts[i]``
        lies in ``[TIMESTEP_MAX*(1-frac_hi), TIMESTEP_MAX*(1-frac_lo)]``, then
        stratifies over the contiguous index span ``[min_i, max_i]`` among
        those matches; every index in that span, including ``max_i``, can be drawn.

        Args:
            include_init: Whether the first matching index ``min_i`` may be drawn;
                ignored when ``force_init`` is True.
            force_init: Always draw ``min_i`` first and stratify the remaining
                ``num_train_timesteps - 1`` draws over the later indices.
            generator: Optional ``torch.Generator`` for deterministic draws.
                Index computation stays on ``scheduler_timesteps.device`` since
                ``scheduler_timesteps`` is typically tiny (≤50 elements) and
                the bookkeeping is cheap; only the final stratified ``rand``
                draw uses ``generator``.

        Raises:
            ValueError: If ``timestep_range`` selects no scheduler timestep, or no index remains
                to draw from after excluding ``min_i``.
        """
        device = scheduler_timesteps.device
        ts = scheduler_timesteps.float()
        num_steps = len(ts)

        frac_start, frac_end = _normalize_timestep_range(timestep_range)
        t_min, t_max = fraction_range_to_t_bounds(frac_start, frac_end)
        mask = (ts >= t_min - 1e-3) & (ts <= t_max + 1e-3)
        valid_indices = torch.where(mask)[0]
        if valid_indices.numel() == 0:
            raise ValueError(
                f"expected timestep_range {timestep_range!r} (t in [{t_min:.3f}, {t_max:.3f}]) "
                f"to select at least one scheduler timestep, received {ts.tolist()}"
            )

        min_idx = int(valid_indices.min().item())
        stop_idx = int(valid_indices.max().item()) + 1

        if force_init:
            if num_train_timesteps == 1:
                t_indices = torch.tensor([min_idx], device=device, dtype=torch.long)
            else:
                start_idx = min_idx + 1
                rest = TimeSampler._stratified_sample(
                    num_train_timesteps - 1, start_idx, stop_idx, device, generator=generator
                )
                t_indices = torch.cat(
                    [torch.tensor([min_idx], device=device, dtype=torch.long), rest]
                )
        else:
            start_idx = min_idx if include_init else min_idx + 1
            t_indices = TimeSampler._stratified_sample(
                num_train_timesteps, start_idx, stop_idx, device, generator=generator
            )

        t_indices = t_indices.clamp(min=0, max=num_steps - 1)
        timesteps = ts[t_indices].unsqueeze(1).expand(-1, batch_size)
        return timesteps

    @staticmethod
    def _stratified_sample(
        num_samples: int,
        start_idx: int,
        stop_idx: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Stratified sampling of indices from the half-open range ``[start_idx, stop_idx)``.

        ``boundaries`` is always built on ``device``; only the uniform
        perturbation draw uses ``generator`` (on ``generator.device``) and is
        then moved to ``device`` for the index arithmetic.

        Raises:
            ValueError: If the range is empty.
        """
        if stop_idx <= start_idx:
            raise ValueError(
                f"expected a non-empty index range [{start_idx}, {stop_idx}) for "
                f"{num_samples} stratified timestep draws"
            )
        rng_device = _rng_device(generator, device)
        boundaries = torch.linspace(start_idx, stop_idx, num_samples + 1, device=device)
        lower, upper = boundaries[:-1].long(), boundaries[1:].long()
        rand_u = torch.rand(num_samples, generator=generator, device=rng_device).to(device)
        return lower + (rand_u * (upper - lower)).long()


def _require_positive_int(value: object, field_name: str) -> None:
    """Require a positive exact integer for materialized sampler shapes."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received {type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(f"expected {field_name} >= 1, received {value}")
