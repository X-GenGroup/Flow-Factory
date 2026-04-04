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

# src/flow_factory/utils/dist_metrics.py
"""Scalar distributed reductions for logging when each rank holds local shards.

Uses :meth:`accelerator.reduce` with ``sum`` for (count, sum, sum_sq) triples and
``torch.distributed.all_reduce`` for MIN/MAX when needed. Single-process runs
skip collective ops.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator


def _dist_ready() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def reduce_sum_vector(accelerator: Accelerator, t: torch.Tensor) -> torch.Tensor:
    """All-rank sum of a 1-D float tensor (identical in single-process)."""
    if not _dist_ready():
        return t
    return accelerator.reduce(t.detach().clone(), reduction="sum")  # type: ignore[return-value]


def global_mean_std_numpy(accelerator: Accelerator, x: np.ndarray) -> Tuple[float, float]:
    """Pooled mean and population std over all ranks for a 1-D sample array (local shard)."""
    x = np.asarray(x, dtype=np.float64)
    n = float(len(x))
    if n == 0:
        t = torch.tensor([0.0, 0.0, 0.0], device=accelerator.device, dtype=torch.float64)
    else:
        t = torch.tensor(
            [n, float(np.sum(x)), float(np.sum(x * x))],
            device=accelerator.device,
            dtype=torch.float64,
        )
    t = reduce_sum_vector(accelerator, t)
    n_t, s, ss = t[0].item(), t[1].item(), t[2].item()
    if n_t < 1:
        return 0.0, 1e-6
    mean = s / n_t
    std = max((ss / n_t - mean**2) ** 0.5, 1e-6)
    return mean, std


def global_mean_stds_from_arrays(
    accelerator: Accelerator, arrays: List[np.ndarray]
) -> List[Tuple[float, float]]:
    """Pooled mean/std for each array in *arrays* with one batched sum-reduce."""
    stats: List[float] = []
    for x in arrays:
        x = np.asarray(x, dtype=np.float64)
        n = float(len(x))
        if n == 0:
            stats.extend([0.0, 0.0, 0.0])
        else:
            stats.extend([n, float(np.sum(x)), float(np.sum(x * x))])
    if not stats:
        return []
    t = torch.tensor(stats, device=accelerator.device, dtype=torch.float64)
    t = reduce_sum_vector(accelerator, t)
    out: List[Tuple[float, float]] = []
    n_arr = len(arrays)
    for i in range(n_arr):
        n_t, s, ss = t[3 * i].item(), t[3 * i + 1].item(), t[3 * i + 2].item()
        if n_t < 1:
            out.append((0.0, 1e-6))
        else:
            mean = s / n_t
            std = max((ss / n_t - mean**2) ** 0.5, 1e-6)
            out.append((mean, std))
    return out


def all_reduce_max_float(accelerator: Accelerator, local: float) -> float:
    t = torch.tensor([local], device=accelerator.device, dtype=torch.float64)
    if not _dist_ready():
        return float(t.item())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def all_reduce_min_float(accelerator: Accelerator, local: float) -> float:
    t = torch.tensor([local], device=accelerator.device, dtype=torch.float64)
    if not _dist_ready():
        return float(t.item())
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return float(t.item())


def global_min_max_numpy(accelerator: Accelerator, x: np.ndarray) -> Tuple[float, float]:
    """Global min and max over all elements on all ranks."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        lo = float("inf")
        hi = float("-inf")
    else:
        lo = float(np.min(x))
        hi = float(np.max(x))
    lo = all_reduce_min_float(accelerator, lo)
    hi = all_reduce_max_float(accelerator, hi)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return 0.0, 0.0
    return lo, hi


def global_mean_abs_numpy(accelerator: Accelerator, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = float(len(x))
    s = float(np.sum(np.abs(x))) if n else 0.0
    t = torch.tensor([n, s], device=accelerator.device, dtype=torch.float64)
    t = reduce_sum_vector(accelerator, t)
    n_t, s_t = t[0].item(), t[1].item()
    if n_t < 1:
        return 0.0
    return s_t / n_t


def global_mean_of_scalar_per_group(
    accelerator: Accelerator, g_stds: np.ndarray
) -> float:
    """Mean of per-group values (e.g. group stds) pooled across ranks."""
    g_stds = np.asarray(g_stds, dtype=np.float64)
    local_sum = float(g_stds.sum()) if len(g_stds) else 0.0
    local_count = float(len(g_stds))
    t = torch.tensor([local_sum, local_count], device=accelerator.device, dtype=torch.float64)
    t = reduce_sum_vector(accelerator, t)
    tot = t[1].item()
    if tot < 1:
        return 0.0
    return t[0].item() / tot


def global_max_min_of_scalar_per_group(
    accelerator: Accelerator, g_stds: np.ndarray
) -> Tuple[float, float]:
    """Global max and min of per-group stds (or similar) across all groups on all ranks."""
    g_stds = np.asarray(g_stds, dtype=np.float64)
    if len(g_stds) == 0:
        local_max = float("-inf")
        local_min = float("inf")
    else:
        local_max = float(np.max(g_stds))
        local_min = float(np.min(g_stds))
    mx = all_reduce_max_float(accelerator, local_max)
    mn = all_reduce_min_float(accelerator, local_min)
    if not math.isfinite(mx):
        mx = 0.0
    if not math.isfinite(mn):
        mn = 0.0
    return mx, mn


def global_std_of_group_means(accelerator: Accelerator, g_means: np.ndarray) -> float:
    """Population std of per-group means across all groups on all ranks."""
    g_means = np.asarray(g_means, dtype=np.float64)
    n = float(len(g_means))
    if n == 0:
        t = torch.tensor([0.0, 0.0, 0.0], device=accelerator.device, dtype=torch.float64)
    else:
        t = torch.tensor(
            [n, float(np.sum(g_means)), float(np.sum(g_means * g_means))],
            device=accelerator.device,
            dtype=torch.float64,
        )
    t = reduce_sum_vector(accelerator, t)
    n_g, s, ss = t[0].item(), t[1].item(), t[2].item()
    if n_g < 1:
        return 0.0
    mean = s / n_g
    return max((ss / n_g - mean**2), 0.0) ** 0.5


def global_zero_std_ratio(
    accelerator: Accelerator,
    rewards: np.ndarray,
    group_indices: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fraction of groups with near-zero std, pooled across ranks."""
    rewards = np.asarray(rewards, dtype=np.float64)
    unique_groups = np.unique(group_indices)
    zero_std_count = sum(
        1 for gid in unique_groups if np.std(rewards[group_indices == gid]) < eps
    )
    n_groups = len(unique_groups)
    t = torch.tensor(
        [float(zero_std_count), float(n_groups)],
        device=accelerator.device,
        dtype=torch.float64,
    )
    t = reduce_sum_vector(accelerator, t)
    denom = t[1].item()
    if denom < 1:
        return 0.0
    return t[0].item() / denom
