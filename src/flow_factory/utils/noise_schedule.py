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
"""
Utility functions for noise schedule and time sampling.

``timestep_range=(frac_lo, frac_hi)`` is a **fraction along the denoising axis**
from scheduler time 1000 (noisy) toward 0 (clean). Mapping:

    t_scheduler = TIMESTEP_MAX * (1 - frac)

So ``(0, 0.99)`` yields ``t ∈ [TIMESTEP_MAX * 0.01, TIMESTEP_MAX]`` (e.g. [10, 1000]
when TIMESTEP_MAX=1000). All samplers return **scheduler-scale** timesteps in
``[0, TIMESTEP_MAX]``; trainers pass them to ``adapter.forward(t=...)`` without
extra scaling. Use ``flow_match_sigma(t) = t / TIMESTEP_MAX`` for linear
flow interpolation ``x_t = (1-σ) x_0 + σ ε``.
"""
import torch
from typing import Tuple, Union

TIMESTEP_MAX = 1000.0


def flow_match_sigma(t_scheduler: torch.Tensor) -> torch.Tensor:
    """Map scheduler timestep in [0, TIMESTEP_MAX] to σ in [0, 1] for x_t = (1-σ)x0 + σ ε."""
    return (t_scheduler / TIMESTEP_MAX).clamp(0.0, 1.0)


def fraction_range_to_t_bounds(frac_lo: float, frac_hi: float) -> Tuple[float, float]:
    """Return (t_min, t_max) in scheduler scale for fraction range [frac_lo, frac_hi]."""
    t_min = TIMESTEP_MAX * (1.0 - frac_hi)
    t_max = TIMESTEP_MAX * (1.0 - frac_lo)
    return t_min, t_max


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
    ) -> torch.Tensor:
        """Samples ``raw`` in (0, 1) with logit-normal + optional shift warp (legacy shape)."""
        if stratified:
            base = (torch.arange(num_rows, device=device) + torch.rand(num_rows, device=device)) / num_rows
            normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
            u_standard = normal_dist.icdf(torch.clamp(base, 1e-7, 1 - 1e-7))
            u_standard = u_standard[torch.randperm(num_rows, device=device)]
        else:
            u_standard = torch.randn(num_rows, device=device)

        u = u_standard * logit_std + logit_mean
        raw = torch.sigmoid(u)
        raw = time_shift * raw / (1 + (time_shift - 1) * raw)
        return torch.clamp(raw, min=0.01, max=1.0 - 1e-6)

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
    ) -> torch.Tensor:
        """
        Logit-normal time sampling; returns scheduler-scale timesteps in ``[0, TIMESTEP_MAX]``.

        ``timestep_range`` is interpreted as ``(frac_lo, frac_hi)`` (fraction along 1000→0).
        A unit interval sample ``raw`` is mapped to ``frac = frac_lo + raw * (frac_hi - frac_lo)``,
        then ``t = TIMESTEP_MAX * (1 - frac)``.
        """
        if isinstance(timestep_range, (list, tuple)):
            frac_lo, frac_hi = float(timestep_range[0]), float(timestep_range[1])
        else:
            frac_lo, frac_hi = 0.0, float(timestep_range)

        raw = TimeSampler._raw_logit_normal_unit(num_timesteps, device, stratified, logit_mean, logit_std, time_shift)
        frac = frac_lo + raw * (frac_hi - frac_lo)
        t = TIMESTEP_MAX * (1.0 - frac)
        return t.unsqueeze(1).expand(num_timesteps, batch_size)

    @staticmethod
    def uniform(
        batch_size: int,
        num_timesteps: int,
        timestep_range: Union[float, Tuple[float, float]],
        time_shift: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Uniform sampling over fraction interval, then map to ``[0, TIMESTEP_MAX]`` scheduler times.
        Optional ``time_shift`` warps the fraction before mapping (same as legacy uniform).
        """
        if isinstance(timestep_range, (list, tuple)):
            frac_lo, frac_hi = float(timestep_range[0]), float(timestep_range[1])
        else:
            frac_lo, frac_hi = 0.0, float(timestep_range)

        rand_u = torch.rand(num_timesteps, device=device)
        normalized = (torch.arange(num_timesteps, device=device) + rand_u) / num_timesteps
        f = frac_lo + normalized * (frac_hi - frac_lo)
        f = f[torch.randperm(num_timesteps, device=device)]
        if abs(time_shift - 1.0) > 1e-6:
            f = time_shift * f / (1 + (time_shift - 1) * f)
        t = TIMESTEP_MAX * (1.0 - f)
        return t.unsqueeze(1).expand(-1, batch_size)

    @staticmethod
    def discrete(
        batch_size: int,
        num_train_timesteps: int,
        scheduler_timesteps: torch.Tensor,
        timestep_range: Union[float, Tuple[float, float]] = 1.0,
        include_init: bool = True,
        force_init: bool = False,
    ) -> torch.Tensor:
        """
        Discrete stratified sampling from ``scheduler_timesteps`` (scheduler scale, e.g. 0–1000).

        ``timestep_range=(frac_lo, frac_hi)`` keeps indices ``i`` whose ``ts[i]`` lies in
        ``[TIMESTEP_MAX*(1-frac_hi), TIMESTEP_MAX*(1-frac_lo)]``, then stratifies over
        the contiguous index span ``[min_i, max_i]`` among those matches.
        """
        device = scheduler_timesteps.device
        ts = scheduler_timesteps.float()
        num_steps = len(ts)

        if isinstance(timestep_range, (list, tuple)):
            frac_start, frac_end = float(timestep_range[0]), float(timestep_range[1])
        else:
            frac_start, frac_end = 0.0, float(timestep_range)

        t_min, t_max = fraction_range_to_t_bounds(frac_start, frac_end)
        mask = (ts >= t_min - 1e-3) & (ts <= t_max + 1e-3)
        valid_indices = torch.where(mask)[0]

        min_idx = int(valid_indices.min().item())
        max_idx = int(valid_indices.max().item())

        if force_init:
            if num_train_timesteps == 1:
                t_indices = torch.tensor([min_idx], device=device, dtype=torch.long)
            else:
                start_idx = min_idx + 1
                t_indices = torch.cat(
                    [
                        torch.tensor([min_idx], device=device, dtype=torch.long),
                        TimeSampler._stratified_sample(num_train_timesteps - 1, start_idx, max_idx, device),
                    ]
                )
        else:
            start_idx = min_idx if include_init else min_idx + 1
            t_indices = TimeSampler._stratified_sample(num_train_timesteps, start_idx, max_idx, device)

        t_indices = t_indices.clamp(min=0, max=num_steps - 1)
        timesteps = ts[t_indices].unsqueeze(1).expand(-1, batch_size)
        return timesteps

    @staticmethod
    def _stratified_sample(
        num_samples: int,
        start_idx: int,
        end_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Stratified sampling of indices from [start_idx, end_idx]."""
        boundaries = torch.linspace(start_idx, end_idx, num_samples + 1, device=device)
        lower, upper = boundaries[:-1].long(), boundaries[1:].long()
        rand_u = torch.rand(num_samples, device=device)
        return lower + (rand_u * (upper - lower)).long()
