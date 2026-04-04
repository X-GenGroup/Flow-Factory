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

# src/flow_factory/utils/dist.py
"""Distributed utilities: environment helpers, tensor/sample gathering, and metric reductions.

Sections:
- Environment helpers
- Tensor gathering
- Sample gathering
- Scalar metric reductions (numpy-based, used by AdvantageProcessor)
- Tensor metric reductions (batched global stats, used by Trainers)
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils.operations import gather_object

from ..samples import BaseSample
from .base import is_tensor_list
from .logger_utils import setup_logger

logger = setup_logger(__name__)


def _is_distributed() -> bool:
    """Check whether ``torch.distributed`` is available and initialized."""
    return dist.is_available() and dist.is_initialized()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def get_world_size() -> int:
    """Detect the distributed world size from environment variables.

    Returns:
        int: The number of distributed processes.  Falls back to ``1`` when
            no recognized environment variable is set.

    Note:
        Checks ``WORLD_SIZE`` (PyTorch/Accelerate/DDP), ``OMPI_COMM_WORLD_SIZE``
        (OpenMPI/Horovod), and ``PMI_SIZE`` (Intel MPI/Slurm) in that order.
    """
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])
    if "PMI_SIZE" in os.environ:
        return int(os.environ["PMI_SIZE"])
    return 1


# ---------------------------------------------------------------------------
# Tensor gathering
# ---------------------------------------------------------------------------


def all_gather_tensor_list(
    accelerator: Accelerator,
    tensor_list: List[torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Gather a variable-length list of heterogeneous-shape tensors from all ranks.

    Args:
        accelerator: Accelerator instance for device and process info.
        tensor_list: Local list of tensors.  Each tensor may have a different
            shape but all must share the same number of dimensions.  Lists on
            different ranks may have different lengths.
        dtype: Output dtype.  Defaults to the dtype of the first tensor.
        device: Device for the returned tensors.

    Returns:
        List[torch.Tensor]: Concatenation of all ranks' tensor lists, ordered
            by rank index.

    Note:
        Requires 3 NCCL calls: (1) gather list lengths, (2) gather per-tensor
        shapes, (3) gather flattened data.
    """
    if not tensor_list:
        return []

    assert all(isinstance(t, torch.Tensor) for t in tensor_list), (
        "All elements in tensor_list must be torch.Tensor"
    )
    assert all(t.dim() == tensor_list[0].dim() for t in tensor_list), (
        "All tensors must have the same number of dimensions"
    )

    tensor_dim = tensor_list[0].dim()
    tensor_dtype = tensor_list[0].dtype if dtype is None else dtype
    device = torch.device(device)

    # Step 1: Gather lengths of tensor_list from all ranks
    local_length = torch.tensor(
        [len(tensor_list)], device=accelerator.device, dtype=torch.long
    )
    gathered_lengths = [
        torch.zeros(1, dtype=torch.long, device=accelerator.device)
        for _ in range(accelerator.num_processes)
    ]
    dist.all_gather(gathered_lengths, local_length)
    gathered_lengths = [int(length.item()) for length in gathered_lengths]

    # Step 2: Gather shapes of each tensor from all ranks
    local_shapes = torch.tensor(
        [list(t.shape) for t in tensor_list],
        device=accelerator.device,
        dtype=torch.long,
    )
    gathered_shapes = [
        torch.zeros(
            (length, tensor_dim), dtype=torch.long, device=accelerator.device
        )
        for length in gathered_lengths
    ]
    dist.all_gather(gathered_shapes, local_shapes)
    gathered_shapes = [shapes.cpu() for shapes in gathered_shapes]

    # Compute total flattened length per rank
    flat_lengths = [
        sum(int(shape.prod().item()) for shape in this_rank_shapes)
        for this_rank_shapes in gathered_shapes
    ]

    # Step 3: Gather all tensors via flattened concatenation
    local_flat_tensor = torch.cat(
        [t.flatten() for t in tensor_list], dim=0
    ).to(device=accelerator.device, dtype=tensor_dtype)
    gathered_flat_tensors = [
        torch.zeros(length, dtype=tensor_dtype, device=accelerator.device)
        for length in flat_lengths
    ]
    dist.all_gather(gathered_flat_tensors, local_flat_tensor)
    gathered_flat_tensors = [t.cpu() for t in gathered_flat_tensors]

    # Step 4: Reconstruct tensors from gathered shapes and flattened data
    gathered_tensors = []
    for this_rank_shapes, this_rank_flat_tensor in zip(
        gathered_shapes, gathered_flat_tensors
    ):
        offset = 0
        for shape in this_rank_shapes:
            length = int(shape.prod().item())
            this_tensor = (
                this_rank_flat_tensor[offset : offset + length]
                .reshape(shape.tolist())
                .to(device)
            )
            gathered_tensors.append(this_tensor)
            offset += length

    return gathered_tensors


def all_gather_nested_tensor_list(
    accelerator: Accelerator,
    nested_tensor_list: List[List[torch.Tensor]],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> List[List[torch.Tensor]]:
    """Gather a nested list-of-lists of tensors from all ranks.

    Args:
        accelerator: Accelerator instance.
        nested_tensor_list: Local nested structure, e.g.
            ``[[t1, t2], [t3]]``.  Inner lists may differ in length across
            ranks and tensors may differ in shape.
        dtype: Output dtype.  Defaults to the dtype of the first tensor.
        device: Device for the returned tensors.

    Returns:
        List[List[torch.Tensor]]: Gathered nested structure from all ranks,
            concatenated in rank order.

    Note:
        Requires 5 NCCL calls: 3 inside :func:`all_gather_tensor_list` plus
        2 for the nested structure metadata (list count and inner lengths).
    """
    # Flatten the local nested structure into a single list
    flat_tensor_list = [t for sublist in nested_tensor_list for t in sublist]

    # Gather the flattened tensors
    gathered_flat_tensors = all_gather_tensor_list(
        accelerator, flat_tensor_list, dtype=dtype, device=device
    )

    # Gather structure metadata (inner list lengths) from all ranks
    local_structure = torch.tensor(
        [len(sublist) for sublist in nested_tensor_list],
        dtype=torch.long,
        device=accelerator.device,
    )

    local_list_count = torch.tensor(
        [local_structure.numel()], device=accelerator.device, dtype=torch.long
    )
    gathered_list_counts = [
        torch.zeros_like(local_list_count)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_list_counts, local_list_count)

    gathered_structures = [
        torch.zeros(count.item(), dtype=torch.long, device=accelerator.device)
        for count in gathered_list_counts
    ]
    dist.all_gather(gathered_structures, local_structure)

    # Reconstruct nested structure
    gathered_nested_tensors = []
    flat_tensor_idx = 0
    for rank_structure in gathered_structures:
        for inner_list_len in rank_structure.tolist():
            length = int(inner_list_len)
            inner_list = gathered_flat_tensors[
                flat_tensor_idx : flat_tensor_idx + length
            ]
            gathered_nested_tensors.append(inner_list)
            flat_tensor_idx += length

    assert flat_tensor_idx == len(gathered_flat_tensors), (
        "Mismatch in reconstructed tensor count when rebuilding nested structure."
    )

    return gathered_nested_tensors


# ---------------------------------------------------------------------------
# Sample gathering
# ---------------------------------------------------------------------------


def _gather_field_values(
    accelerator: Accelerator,
    field_values: list,
    device: torch.device,
) -> list:
    """Gather a single field's values across ranks using type-based dispatch.

    Args:
        accelerator: Accelerator instance.
        field_values: Per-sample values for one field on this rank.
        device: Target device for the returned values.

    Returns:
        list: Gathered values from all ranks, concatenated in rank order.

    Note:
        Dispatch order: (1) uniform-shape Tensor -> ``accelerator.gather``,
        (2) heterogeneous Tensor list -> :func:`all_gather_tensor_list`,
        (3) nested Tensor list -> :func:`all_gather_nested_tensor_list`,
        (4) fallback -> CPU pickle via ``gather_object``.
    """
    if not field_values:
        return gather_object(field_values)

    # 1. Single Tensor per sample with uniform shape
    if (
        isinstance(field_values[0], torch.Tensor)
        and all(
            isinstance(v, torch.Tensor) and v.shape == field_values[0].shape
            for v in field_values
        )
    ):
        stacked = torch.stack(field_values).to(accelerator.device)
        gathered = accelerator.gather(stacked)
        return [t.to(device) for t in gathered]

    # 2. List[Tensor] (possibly heterogeneous shapes)
    if is_tensor_list(field_values):
        return all_gather_tensor_list(
            accelerator=accelerator,
            tensor_list=field_values,
            device=device,
        )

    # 3. List[List[Tensor]]
    if isinstance(field_values[0], list) and is_tensor_list(field_values[0]):
        return all_gather_nested_tensor_list(
            accelerator=accelerator,
            nested_tensor_list=field_values,
            device=device,
        )

    # 4. Fallback: pickle serialization
    return gather_object(field_values)


_EXTRA_PREFIX = "__extra__."


def gather_samples(
    accelerator: Accelerator,
    samples: List[BaseSample],
    field_names: List[str],
    device: Union[str, torch.device] = torch.device("cpu"),
) -> List[BaseSample]:
    """Gather a list of BaseSample instances from all ranks.

    Args:
        accelerator: Accelerator instance.
        samples: Local samples on this rank.
        field_names: Fields to gather.  When ``'extra_kwargs'`` is included,
            each key inside the dict is gathered independently and reassembled.
        device: Target device for tensor fields in the returned samples.

    Returns:
        List[BaseSample]: Samples from all ranks, concatenated in rank order.
    """
    if not samples:
        return []

    sample_cls = samples[0].__class__
    device = torch.device(device)

    # Separate extra_kwargs from regular fields
    has_extra_kwargs = "extra_kwargs" in field_names
    regular_fields = sorted(f for f in field_names if f != "extra_kwargs")
    extra_keys: List[str] = []
    if has_extra_kwargs:
        extra_keys = sorted({k for s in samples for k in s.extra_kwargs})

    all_keys = regular_fields + [f"{_EXTRA_PREFIX}{k}" for k in extra_keys]
    d: dict = {key: [] for key in all_keys}

    # Collect and gather each key
    for key in all_keys:
        if key.startswith(_EXTRA_PREFIX):
            extra_k = key[len(_EXTRA_PREFIX) :]
            field_values = [s.extra_kwargs.get(extra_k) for s in samples]
        else:
            field_values = [getattr(sample, key) for sample in samples]
        d[key] = _gather_field_values(accelerator, field_values, device)

    # Reconstruct BaseSample objects
    n_gathered = len(d[all_keys[0]]) if all_keys else 0
    gathered_samples = []
    for i in range(n_gathered):
        kwargs = {f: d[f][i] for f in regular_fields}
        if has_extra_kwargs:
            kwargs["extra_kwargs"] = {
                k: d[f"{_EXTRA_PREFIX}{k}"][i] for k in extra_keys
            }
        gathered_samples.append(sample_cls(**kwargs))
    return gathered_samples


# ---------------------------------------------------------------------------
# Scalar metric reductions (numpy-based, used by AdvantageProcessor)
# ---------------------------------------------------------------------------


def all_reduce_min_float(accelerator: Accelerator, local: float) -> float:
    """All-reduce a single float scalar with MIN across all ranks.

    Args:
        accelerator: Accelerator instance.
        local: Local float value on this rank.

    Returns:
        float: The global minimum across all ranks.
    """
    t = torch.tensor([local], device=accelerator.device, dtype=torch.float64)
    if _is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return float(t.item())


def all_reduce_max_float(accelerator: Accelerator, local: float) -> float:
    """All-reduce a single float scalar with MAX across all ranks.

    Args:
        accelerator: Accelerator instance.
        local: Local float value on this rank.

    Returns:
        float: The global maximum across all ranks.
    """
    t = torch.tensor([local], device=accelerator.device, dtype=torch.float64)
    if _is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def global_mean_std_numpy(
    accelerator: Accelerator, x: np.ndarray
) -> Tuple[float, float]:
    """Compute pooled global mean and population std from a local numpy shard.

    Args:
        accelerator: Accelerator instance.
        x: 1-D numpy array (local shard on this rank).

    Returns:
        Tuple[float, float]: ``(mean, std)`` across all ranks.
            Returns ``(0.0, 1e-6)`` when total count is zero.

    Note:
        Uses a single SUM all-reduce of a packed ``(count, sum, sum_sq)`` triple.
    """
    x = np.asarray(x, dtype=np.float64)
    n = float(len(x))
    if n == 0:
        t = torch.tensor(
            [0.0, 0.0, 0.0], device=accelerator.device, dtype=torch.float64
        )
    else:
        t = torch.tensor(
            [n, float(np.sum(x)), float(np.sum(x * x))],
            device=accelerator.device,
            dtype=torch.float64,
        )
    t = accelerator.reduce(t, reduction="sum")
    n_t, s, ss = t[0].item(), t[1].item(), t[2].item()
    if n_t < 1:
        return 0.0, 1e-6
    mean = s / n_t
    std = max((ss / n_t - mean**2) ** 0.5, 1e-6)
    return mean, std


def global_mean_stds_from_arrays(
    accelerator: Accelerator, arrays: List[np.ndarray]
) -> List[Tuple[float, float]]:
    """Compute pooled global mean/std for multiple arrays with one batched reduce.

    Args:
        accelerator: Accelerator instance.
        arrays: List of 1-D numpy arrays (local shards on this rank).

    Returns:
        List[Tuple[float, float]]: Per-array ``(mean, std)`` across all ranks.
            Returns ``(0.0, 1e-6)`` for arrays with zero total count.

    Note:
        All ``(count, sum, sum_sq)`` triples are packed into a single vector
        for one SUM all-reduce call, regardless of the number of arrays.
    """
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
    t = accelerator.reduce(t, reduction="sum")
    out: List[Tuple[float, float]] = []
    for i in range(len(arrays)):
        n_t, s, ss = t[3 * i].item(), t[3 * i + 1].item(), t[3 * i + 2].item()
        if n_t < 1:
            out.append((0.0, 1e-6))
        else:
            mean = s / n_t
            std = max((ss / n_t - mean**2) ** 0.5, 1e-6)
            out.append((mean, std))
    return out


def global_min_max_numpy(
    accelerator: Accelerator, x: np.ndarray
) -> Tuple[float, float]:
    """Compute global min and max of a 1-D numpy array across all ranks.

    Args:
        accelerator: Accelerator instance.
        x: 1-D numpy array (local shard on this rank).

    Returns:
        Tuple[float, float]: ``(global_min, global_max)``.
            Returns ``(0.0, 0.0)`` when all shards are empty.
    """
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
    """Compute global mean of absolute values across all ranks.

    Args:
        accelerator: Accelerator instance.
        x: 1-D numpy array (local shard on this rank).

    Returns:
        float: Global mean of ``|x|``.  Returns ``0.0`` when total count is zero.
    """
    x = np.asarray(x, dtype=np.float64)
    n = float(len(x))
    s = float(np.sum(np.abs(x))) if n else 0.0
    t = torch.tensor([n, s], device=accelerator.device, dtype=torch.float64)
    t = accelerator.reduce(t, reduction="sum")
    n_t, s_t = t[0].item(), t[1].item()
    if n_t < 1:
        return 0.0
    return s_t / n_t


def global_mean_of_scalar_per_group(
    accelerator: Accelerator, g_stds: np.ndarray
) -> float:
    """Compute the global mean of per-group scalar values pooled across ranks.

    Args:
        accelerator: Accelerator instance.
        g_stds: 1-D numpy array of per-group scalars (e.g. group stds) on
            this rank.

    Returns:
        float: Weighted mean across all groups on all ranks.
    """
    g_stds = np.asarray(g_stds, dtype=np.float64)
    local_sum = float(g_stds.sum()) if len(g_stds) else 0.0
    local_count = float(len(g_stds))
    t = torch.tensor(
        [local_sum, local_count], device=accelerator.device, dtype=torch.float64
    )
    t = accelerator.reduce(t, reduction="sum")
    tot = t[1].item()
    if tot < 1:
        return 0.0
    return t[0].item() / tot


def global_max_min_of_scalar_per_group(
    accelerator: Accelerator, g_stds: np.ndarray
) -> Tuple[float, float]:
    """Compute the global max and min of per-group scalar values across ranks.

    Args:
        accelerator: Accelerator instance.
        g_stds: 1-D numpy array of per-group scalars on this rank.

    Returns:
        Tuple[float, float]: ``(global_max, global_min)`` across all groups
            on all ranks.
    """
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


def global_std_of_group_means(
    accelerator: Accelerator, g_means: np.ndarray
) -> float:
    """Compute the population std of per-group means across all ranks.

    Args:
        accelerator: Accelerator instance.
        g_means: 1-D numpy array of per-group mean values on this rank.

    Returns:
        float: Population std of all group means globally.

    Note:
        Uses a single SUM all-reduce of a packed ``(count, sum, sum_sq)`` triple.
    """
    g_means = np.asarray(g_means, dtype=np.float64)
    n = float(len(g_means))
    if n == 0:
        t = torch.tensor(
            [0.0, 0.0, 0.0], device=accelerator.device, dtype=torch.float64
        )
    else:
        t = torch.tensor(
            [n, float(np.sum(g_means)), float(np.sum(g_means * g_means))],
            device=accelerator.device,
            dtype=torch.float64,
        )
    t = accelerator.reduce(t, reduction="sum")
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
    """Compute the fraction of groups with near-zero std, pooled across ranks.

    Args:
        accelerator: Accelerator instance.
        rewards: 1-D numpy array of reward values (local shard).
        group_indices: Integer array mapping each element to its group.
        eps: Threshold below which a group's std is considered zero.

    Returns:
        float: Ratio of zero-std groups to total groups across all ranks.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    unique_groups = np.unique(group_indices)
    zero_std_count = sum(
        1
        for gid in unique_groups
        if np.std(rewards[group_indices == gid]) < eps
    )
    n_groups = len(unique_groups)
    t = torch.tensor(
        [float(zero_std_count), float(n_groups)],
        device=accelerator.device,
        dtype=torch.float64,
    )
    t = accelerator.reduce(t, reduction="sum")
    denom = t[1].item()
    if denom < 1:
        return 0.0
    return t[0].item() / denom


# ---------------------------------------------------------------------------
# Tensor metric reductions (batched global stats, used by Trainers)
# ---------------------------------------------------------------------------


def global_tensor_stats(
    accelerator: Accelerator,
    x: torch.Tensor,
) -> Dict[str, float]:
    """Compute global min, max, mean, and population std for a 1-D tensor shard.

    Args:
        accelerator: Accelerator instance.
        x: 1-D tensor (local shard on this rank).

    Returns:
        Dict[str, float]: ``{'min': ..., 'max': ..., 'mean': ..., 'std': ...}``
            with global statistics across all ranks.

    Note:
        Uses 3 all-reduce calls: one SUM for a packed ``(count, sum, sum_sq)``
        triple, one MIN, and one MAX.  Single-process runs skip collective ops.
    """
    x = x.detach().float()
    count = float(x.numel())
    if count == 0:
        total, sum_sq = 0.0, 0.0
        local_min = float("inf")
        local_max = float("-inf")
    else:
        total = float(x.sum())
        sum_sq = float((x**2).sum())
        local_min = float(x.min())
        local_max = float(x.max())

    packed = torch.tensor(
        [count, total, sum_sq], device=accelerator.device, dtype=torch.float64
    )
    packed = accelerator.reduce(packed, reduction="sum")
    global_count = packed[0].item()
    global_sum = packed[1].item()
    global_sum_sq = packed[2].item()

    global_min = all_reduce_min_float(accelerator, local_min)
    global_max = all_reduce_max_float(accelerator, local_max)

    if global_count < 1:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    mean = global_sum / global_count
    std = max((global_sum_sq / global_count - mean**2), 0.0) ** 0.5
    if not math.isfinite(global_min):
        global_min = 0.0
    if not math.isfinite(global_max):
        global_max = 0.0
    return {"min": global_min, "max": global_max, "mean": mean, "std": std}


def global_tensor_stats_batch(
    accelerator: Accelerator,
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """Compute global stats for multiple tensors with only 3 all-reduce calls.

    Args:
        accelerator: Accelerator instance.
        tensors: Mapping from metric name to a 1-D tensor (local shard on
            this rank).

    Returns:
        Dict[str, Dict[str, float]]:
            ``{name: {'min': ..., 'max': ..., 'mean': ..., 'std': ...}}``.

    Note:
        All tensors' ``(count, sum, sum_sq)`` are packed into one SUM reduce;
        local mins and maxes are packed into one MIN and one MAX reduce.
        The total communication cost is **3 all-reduce calls** regardless of
        the number of tensors.
    """
    if not tensors:
        return {}

    keys = list(tensors.keys())

    # Build packed vectors: (count, sum, sum_sq) triples, local mins, local maxes
    sum_triples: List[float] = []
    local_mins: List[float] = []
    local_maxes: List[float] = []

    for key in keys:
        x = tensors[key].detach().float()
        count = float(x.numel())
        if count == 0:
            sum_triples.extend([0.0, 0.0, 0.0])
            local_mins.append(float("inf"))
            local_maxes.append(float("-inf"))
        else:
            sum_triples.extend([count, float(x.sum()), float((x**2).sum())])
            local_mins.append(float(x.min()))
            local_maxes.append(float(x.max()))

    device = accelerator.device

    # SUM reduce for packed (count, sum, sum_sq) triples
    packed_sum = torch.tensor(sum_triples, device=device, dtype=torch.float64)
    packed_sum = accelerator.reduce(packed_sum, reduction="sum")

    # MIN reduce for local minimums
    packed_min = torch.tensor(local_mins, device=device, dtype=torch.float64)
    if _is_distributed():
        dist.all_reduce(packed_min, op=dist.ReduceOp.MIN)

    # MAX reduce for local maximums
    packed_max = torch.tensor(local_maxes, device=device, dtype=torch.float64)
    if _is_distributed():
        dist.all_reduce(packed_max, op=dist.ReduceOp.MAX)

    # Unpack global results
    out: Dict[str, Dict[str, float]] = {}
    for i, key in enumerate(keys):
        global_count = packed_sum[3 * i].item()
        global_sum = packed_sum[3 * i + 1].item()
        global_sum_sq = packed_sum[3 * i + 2].item()
        global_min = packed_min[i].item()
        global_max = packed_max[i].item()

        if global_count < 1:
            out[key] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        else:
            mean = global_sum / global_count
            std = max((global_sum_sq / global_count - mean**2), 0.0) ** 0.5
            if not math.isfinite(global_min):
                global_min = 0.0
            if not math.isfinite(global_max):
                global_max = 0.0
            out[key] = {
                "min": global_min,
                "max": global_max,
                "mean": mean,
                "std": std,
            }

    return out


def reduce_loss_info(
    accelerator: Accelerator,
    loss_info: Dict[str, List[torch.Tensor]],
) -> Dict[str, Any]:
    """Reduce a trainer's accumulated loss_info dict across ranks.

    Automatically classifies each key's value list by tensor dimensionality:

    - **Per-sample tensors** (``dim() > 0``): concatenated, then reduced via
      :func:`global_tensor_stats_batch` and flattened to
      ``{key}_{min|max|mean|std}``.
    - **Scalars** (``dim() == 0``): stacked, averaged, and mean-reduced via
      ``accelerator.reduce``, kept as ``{key}``.

    Args:
        accelerator: Accelerator instance.
        loss_info: Mapping from metric name to a list of tensors accumulated
            over gradient-accumulation steps.

    Returns:
        Dict[str, Any]: Flat dict of globally reduced metrics, ready to pass
            to ``log_data``.
    """
    per_sample: Dict[str, torch.Tensor] = {}
    scalars: Dict[str, torch.Tensor] = {}

    # Classify by tensor dimensionality
    for k, v in loss_info.items():
        if v[0].dim() > 0:
            per_sample[k] = torch.cat(v)
        else:
            scalars[k] = torch.stack(v).mean()

    flat: Dict[str, Any] = {}

    # Per-sample tensors: batched global stats (3 all-reduce calls)
    if per_sample:
        stats = global_tensor_stats_batch(accelerator, per_sample)
        for k, s in stats.items():
            for stat_name, val in s.items():
                flat[f"{k}_{stat_name}"] = val

    # Scalar tensors: mean reduction (1 all-reduce call)
    if scalars:
        reduced_scalars = accelerator.reduce(scalars, reduction="mean")
        flat.update(reduced_scalars)

    return flat
