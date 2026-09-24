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
from functools import lru_cache
from types import NoneType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    get_args,
    get_type_hints,
)

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils.operations import gather_object

from ..samples import BaseSample
from .base import is_tensor_list
from .logger_utils import setup_logger

if TYPE_CHECKING:
    from .group_coordinator import GroupCoordinator

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


def gather_aligned_floating_tensors(
    accelerator: Accelerator,
    tensors: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Gather same-shape floating tensors with one packed collective.

    Same-dtype fields are stacked behind one named axis. Heterogeneous dtypes
    fall back to one gather per field: byte packing preserves values but costs
    more than the saved launch on the small CRD vectors this helper serves.
    """
    if not tensors:
        return {}
    names = tuple(sorted(tensors))
    first = tensors[names[0]]
    if not isinstance(first, torch.Tensor):
        raise TypeError(
            f"expected tensor {names[0]!r} to be torch.Tensor for packed gather, "
            f"received {type(first).__name__}: {first!r}"
        )
    if first.ndim < 1:
        raise ValueError(
            f"expected tensor {names[0]!r} to have a leading batch dimension for packed "
            f"gather, received shape {tuple(first.shape)}"
        )
    expected_shape = first.shape
    expected_device = first.device
    dtypes = set()
    for name in names:
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"expected tensor {name!r} to be torch.Tensor for packed gather, "
                f"received {type(tensor).__name__}: {tensor!r}"
            )
        if not tensor.is_floating_point():
            raise TypeError(
                f"expected tensor {name!r} to use a floating dtype for packed gather, "
                f"received dtype={tensor.dtype}"
            )
        if tensor.shape != expected_shape:
            raise ValueError(
                f"expected tensor {name!r} shape {tuple(expected_shape)} for packed gather, "
                f"received shape {tuple(tensor.shape)}"
            )
        if tensor.device != expected_device:
            raise ValueError(
                f"expected tensor {name!r} on device {expected_device} for packed gather, "
                f"received device {tensor.device}"
            )
        if tensor.requires_grad:
            raise ValueError(
                f"expected tensor {name!r} to be detached before packed gather, "
                "received requires_grad=True"
            )
        dtypes.add(tensor.dtype)

    if len(dtypes) > 1:
        return {name: accelerator.gather(tensors[name]) for name in names}

    packed = torch.stack([tensors[name] for name in names], dim=-1)
    gathered = accelerator.gather(packed)
    if not isinstance(gathered, torch.Tensor):
        raise TypeError(
            "expected packed gather to return torch.Tensor, "
            f"received {type(gathered).__name__}: {gathered!r}"
        )
    if gathered.ndim != packed.ndim or gathered.shape[1:] != packed.shape[1:]:
        raise ValueError(
            "expected packed gather to preserve every non-batch dimension "
            f"{tuple(packed.shape[1:])}, received shape {tuple(gathered.shape)}"
        )
    return {name: gathered[..., index] for index, name in enumerate(names)}


def all_gather_tensor_list(
    accelerator: Accelerator,
    tensor_list: List[torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
    process_group: Optional[dist.ProcessGroup] = None,
    world_size: Optional[int] = None,
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
    assert all(
        isinstance(t, torch.Tensor) for t in tensor_list
    ), "All elements in tensor_list must be torch.Tensor"
    assert all(
        t.dim() == tensor_list[0].dim() for t in tensor_list
    ), "All tensors must have the same number of dimensions"

    tensor_dim = tensor_list[0].dim()
    tensor_dtype = tensor_list[0].dtype if dtype is None else dtype
    device = torch.device(device)
    collective_world_size = world_size or accelerator.num_processes

    # Step 1: Gather lengths of tensor_list from all ranks
    local_length = torch.tensor([len(tensor_list)], device=accelerator.device, dtype=torch.long)
    gathered_lengths = [
        torch.zeros(1, dtype=torch.long, device=accelerator.device)
        for _ in range(collective_world_size)
    ]
    dist.all_gather(gathered_lengths, local_length, group=process_group)
    gathered_lengths = [int(length.item()) for length in gathered_lengths]

    # Step 2: Gather shapes of each tensor from all ranks
    local_shapes = torch.tensor(
        [list(t.shape) for t in tensor_list],
        device=accelerator.device,
        dtype=torch.long,
    )
    gathered_shapes = [
        torch.zeros((length, tensor_dim), dtype=torch.long, device=accelerator.device)
        for length in gathered_lengths
    ]
    dist.all_gather(gathered_shapes, local_shapes, group=process_group)
    gathered_shapes = [shapes.cpu() for shapes in gathered_shapes]

    # Compute total flattened length per rank
    flat_lengths = [
        sum(int(shape.prod().item()) for shape in this_rank_shapes)
        for this_rank_shapes in gathered_shapes
    ]

    # Step 3: Gather all tensors via flattened concatenation
    local_flat_tensor = torch.cat(
        [t.to(device=accelerator.device, dtype=tensor_dtype).flatten() for t in tensor_list], dim=0
    )
    gathered_flat_tensors = [
        torch.zeros(length, dtype=tensor_dtype, device=accelerator.device)
        for length in flat_lengths
    ]
    dist.all_gather(gathered_flat_tensors, local_flat_tensor, group=process_group)
    gathered_flat_tensors = [t.cpu() for t in gathered_flat_tensors]

    # Step 4: Reconstruct tensors from gathered shapes and flattened data
    gathered_tensors = []
    for this_rank_shapes, this_rank_flat_tensor in zip(gathered_shapes, gathered_flat_tensors):
        offset = 0
        for shape in this_rank_shapes:
            length = int(shape.prod().item())
            this_tensor = (
                this_rank_flat_tensor[offset : offset + length].reshape(shape.tolist()).to(device)
            )
            gathered_tensors.append(this_tensor)
            offset += length

    # Clean up temporary tensors. The caching allocator reuses these freed
    # blocks automatically; calling torch.cuda.empty_cache() here would force a
    # device synchronization and drop the allocator pool, making subsequent
    # cudaMalloc slower in this gather hot path.
    del gathered_shapes, gathered_flat_tensors

    return gathered_tensors


def all_gather_nested_tensor_list(
    accelerator: Accelerator,
    nested_tensor_list: List[List[torch.Tensor]],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
    process_group: Optional[dist.ProcessGroup] = None,
    world_size: Optional[int] = None,
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
    collective_world_size = world_size or accelerator.num_processes
    gathered_flat_tensors = all_gather_tensor_list(
        accelerator,
        flat_tensor_list,
        dtype=dtype,
        device=device,
        process_group=process_group,
        world_size=collective_world_size,
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
        torch.zeros_like(local_list_count) for _ in range(collective_world_size)
    ]
    dist.all_gather(gathered_list_counts, local_list_count, group=process_group)

    gathered_structures = [
        torch.zeros(count.item(), dtype=torch.long, device=accelerator.device)
        for count in gathered_list_counts
    ]
    dist.all_gather(gathered_structures, local_structure, group=process_group)

    # Reconstruct nested structure
    gathered_nested_tensors = []
    flat_tensor_idx = 0
    for rank_structure in gathered_structures:
        for inner_list_len in rank_structure.tolist():
            length = int(inner_list_len)
            inner_list = gathered_flat_tensors[flat_tensor_idx : flat_tensor_idx + length]
            gathered_nested_tensors.append(inner_list)
            flat_tensor_idx += length

    assert flat_tensor_idx == len(
        gathered_flat_tensors
    ), "Mismatch in reconstructed tensor count when rebuilding nested structure."

    return gathered_nested_tensors


# ---------------------------------------------------------------------------
# Sample gathering
# ---------------------------------------------------------------------------


def _gather_field_values(
    accelerator: Accelerator,
    field_values: list,
    device: torch.device,
    group_coordinator: Optional["GroupCoordinator"] = None,
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
    subgroup_collective = (
        group_coordinator is not None and not group_coordinator.uses_global_collective
    )
    process_group = group_coordinator.process_group if subgroup_collective else None
    world_size = group_coordinator.group_world_size if subgroup_collective else None
    if not field_values:
        return (
            group_coordinator.gather_object(field_values)
            if subgroup_collective
            else gather_object(field_values)
        )

    # 1. Single Tensor per sample with uniform shape
    if isinstance(field_values[0], torch.Tensor) and all(
        isinstance(v, torch.Tensor) and v.shape == field_values[0].shape for v in field_values
    ):
        stacked = torch.stack(field_values).to(accelerator.device)
        gathered = (
            group_coordinator.gather(stacked)
            if subgroup_collective
            else accelerator.gather(stacked)
        )
        return [t.to(device) for t in gathered]

    # 2. List[Tensor] (possibly heterogeneous shapes)
    if is_tensor_list(field_values):
        return all_gather_tensor_list(
            accelerator=accelerator,
            tensor_list=field_values,
            device=device,
            process_group=process_group,
            world_size=world_size,
        )

    # 3. List[List[Tensor]]
    if isinstance(field_values[0], list) and is_tensor_list(field_values[0]):
        return all_gather_nested_tensor_list(
            accelerator=accelerator,
            nested_tensor_list=field_values,
            device=device,
            process_group=process_group,
            world_size=world_size,
        )

    # 4. Fallback: pickle serialization
    return (
        group_coordinator.gather_object(field_values)
        if subgroup_collective
        else gather_object(field_values)
    )


_EXTRA_PREFIX = "__extra__."
_CPU_PACKED_GATHER_MAX_BYTES = 4 * 1024 * 1024


def _scalar_annotation_kind(annotation: Any) -> Optional[type]:
    """Return the primitive scalar represented by a possibly optional annotation."""

    args = tuple(arg for arg in get_args(annotation) if arg is not NoneType)
    if args:
        if len(args) != 1:
            return None
        annotation = args[0]
    return annotation if annotation in {bool, int, float, str} else None


@lru_cache(maxsize=None)
def _sample_type_hints(sample_cls: type) -> Dict[str, Any]:
    """Resolve one concrete sample schema once per process."""

    return get_type_hints(sample_cls)


def _gather_sample_metadata(
    accelerator: Accelerator,
    values_by_key: Mapping[str, List[Any]],
    scalar_kinds: Mapping[str, type],
    target_device: torch.device,
    group_coordinator: Optional["GroupCoordinator"],
) -> Tuple[Dict[str, List[Any]], set[str], Dict[str, torch.Tensor], int]:
    """Gather optional-field presence and primitive scalars in GPU tensors.

    A single int64 payload carries presence bits, every bool/int value, UTF-8
    string lengths, and uniform int64 tensor fields such as token IDs. Float
    scalars, when present, share one float64 payload. This replaces one pickle
    collective per optional identity/source field and folds the common
    ``prompt_ids + group identity`` path into one NCCL gather.
    """

    if not values_by_key:
        return {}, set(), {}, 0
    batch_size = len(next(iter(values_by_key.values())))
    metadata_keys = sorted(
        key
        for key, values in values_by_key.items()
        if key in scalar_kinds or any(value is None for value in values)
    )
    string_keys = sorted(key for key in metadata_keys if scalar_kinds.get(key) is str)
    presence_keys = [key for key in metadata_keys if key not in string_keys]
    int_tensor_keys = sorted(
        key
        for key, values in values_by_key.items()
        if values
        and key not in metadata_keys
        and isinstance(values[0], torch.Tensor)
        and values[0].dtype == torch.int64
        and values[0].numel() > 0
        and not values[0].requires_grad
        and all(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.int64
            and value.shape == values[0].shape
            and not value.requires_grad
            for value in values
        )
    )
    if not metadata_keys and not int_tensor_keys:
        return {}, set(), {}, 0

    int_keys = sorted(key for key in metadata_keys if scalar_kinds.get(key) in {bool, int})
    float_keys = sorted(key for key in metadata_keys if scalar_kinds.get(key) is float)
    presence = torch.tensor(
        [
            [int(values_by_key[key][row] is not None) for key in presence_keys]
            for row in range(batch_size)
        ],
        dtype=torch.int64,
        device=accelerator.device,
    )
    int_values = torch.tensor(
        [
            [
                int(values_by_key[key][row]) if values_by_key[key][row] is not None else 0
                for key in int_keys
            ]
            for row in range(batch_size)
        ],
        dtype=torch.int64,
        device=accelerator.device,
    )
    string_lengths = torch.tensor(
        [
            [
                (
                    len(values_by_key[key][row].encode("utf-8"))
                    if values_by_key[key][row] is not None
                    else -1
                )
                for key in string_keys
            ]
            for row in range(batch_size)
        ],
        dtype=torch.int64,
        device=accelerator.device,
    )
    int_tensor_widths = {key: values_by_key[key][0].numel() for key in int_tensor_keys}
    int_tensor_values = [
        torch.stack(values_by_key[key])
        .to(accelerator.device)
        .reshape(batch_size, int_tensor_widths[key])
        for key in int_tensor_keys
    ]
    packed_int = torch.cat(
        (presence, int_values, string_lengths, *int_tensor_values),
        dim=1,
    )
    if group_coordinator is not None and not group_coordinator.uses_global_collective:
        gathered_int = group_coordinator.gather(packed_int)
    elif accelerator.num_processes > 1:
        gathered_int = accelerator.gather(packed_int)
    else:
        gathered_int = packed_int
    gathered_int_device = gathered_int
    gathered_int = gathered_int_device.cpu()
    gathered_presence = gathered_int[:, : len(presence_keys)].to(torch.bool)
    reconstructed: Dict[str, List[Any]] = {}
    resolved = set()
    offset = len(presence_keys)

    for index, key in enumerate(int_keys):
        presence_index = presence_keys.index(key)
        value_column = gathered_int[:, offset + index]
        kind = scalar_kinds[key]
        reconstructed[key] = [
            kind(value.item()) if present else None
            for present, value in zip(gathered_presence[:, presence_index], value_column)
        ]
        resolved.add(key)
    offset += len(int_keys)

    gathered_string_lengths = {
        key: gathered_int[:, offset + index].clone() for index, key in enumerate(string_keys)
    }
    offset += len(string_keys)

    for key in int_tensor_keys:
        width = int_tensor_widths[key]
        sample_shape = values_by_key[key][0].shape
        field = gathered_int_device[:, offset : offset + width].reshape(
            gathered_int_device.shape[0],
            *sample_shape,
        )
        reconstructed[key] = list(field.to(target_device))
        resolved.add(key)
        offset += width

    if float_keys:
        local_float = torch.tensor(
            [
                [
                    float(values_by_key[key][row]) if values_by_key[key][row] is not None else 0.0
                    for key in float_keys
                ]
                for row in range(batch_size)
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )
        if group_coordinator is not None and not group_coordinator.uses_global_collective:
            gathered_float = group_coordinator.gather(local_float)
        elif accelerator.num_processes > 1:
            gathered_float = accelerator.gather(local_float)
        else:
            gathered_float = local_float
        gathered_float = gathered_float.cpu()
        for index, key in enumerate(float_keys):
            presence_index = presence_keys.index(key)
            reconstructed[key] = [
                float(value.item()) if present else None
                for present, value in zip(
                    gathered_presence[:, presence_index],
                    gathered_float[:, index],
                )
            ]
            resolved.add(key)

    for index, key in enumerate(presence_keys):
        if key not in resolved and not torch.any(gathered_presence[:, index]):
            reconstructed[key] = [None] * int(gathered_presence.shape[0])
            resolved.add(key)
    return reconstructed, resolved, gathered_string_lengths, int(gathered_int.shape[0])


def _gather_packed_string_fields(
    accelerator: Accelerator,
    values_by_key: Mapping[str, List[Any]],
    gathered_lengths: Mapping[str, torch.Tensor],
    group_coordinator: Optional["GroupCoordinator"],
) -> Dict[str, List[Optional[str]]]:
    """Gather all UTF-8 sample fields in one padded uint8 tensor.

    String byte lengths ride the int64 metadata collective. This second
    collective carries only bytes and is skipped when every string is empty or
    ``None``. Rank-local payloads are padded to the largest rank payload so the
    transfer remains a regular NCCL all-gather rather than a pickle/object
    collective.
    """

    keys = sorted(gathered_lengths)
    if not keys:
        return {}
    batch_size = len(values_by_key[keys[0]])
    length_matrix = torch.stack([gathered_lengths[key] for key in keys], dim=1)
    gathered_count = int(length_matrix.shape[0])
    world_size = gathered_count // batch_size
    rank_lengths = length_matrix.reshape(world_size, batch_size, len(keys))
    rank_byte_counts = rank_lengths.clamp_min(0).sum(dim=(1, 2))
    max_rank_bytes = int(rank_byte_counts.max().item())

    local_bytes = bytearray()
    for row in range(batch_size):
        for key in keys:
            value = values_by_key[key][row]
            if value is not None:
                local_bytes.extend(value.encode("utf-8"))

    if max_rank_bytes:
        packed = torch.zeros(
            max_rank_bytes,
            dtype=torch.uint8,
            device=accelerator.device,
        )
        if local_bytes:
            packed[: len(local_bytes)] = torch.tensor(
                list(local_bytes),
                dtype=torch.uint8,
                device=accelerator.device,
            )
        if group_coordinator is not None and not group_coordinator.uses_global_collective:
            gathered_bytes = group_coordinator.gather(packed)
        elif accelerator.num_processes > 1:
            gathered_bytes = accelerator.gather(packed)
        else:
            gathered_bytes = packed
        gathered_bytes = gathered_bytes.reshape(world_size, max_rank_bytes).cpu()
    else:
        gathered_bytes = torch.empty((world_size, 0), dtype=torch.uint8)

    reconstructed: Dict[str, List[Optional[str]]] = {key: [] for key in keys}
    for rank in range(world_size):
        offset = 0
        for row in range(batch_size):
            for key_index, key in enumerate(keys):
                length = int(rank_lengths[rank, row, key_index].item())
                if length < 0:
                    reconstructed[key].append(None)
                    continue
                value = bytes(gathered_bytes[rank, offset : offset + length].tolist()).decode(
                    "utf-8"
                )
                reconstructed[key].append(value)
                offset += length
    return reconstructed


def _is_uniform_tensor_field(field_values: List[Any]) -> bool:
    """Return whether one sample field can participate in numeric packing."""
    if not field_values or not isinstance(field_values[0], torch.Tensor):
        return False
    first = field_values[0]
    if first.numel() == 0 or first.requires_grad:
        return False
    return all(
        isinstance(value, torch.Tensor)
        and value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        and not value.requires_grad
        for value in field_values
    )


def _packed_tensor_field_chunks(
    accelerator: Accelerator,
    values_by_key: Mapping[str, List[Any]],
    target_device: torch.device,
    collective_world_size: Optional[int] = None,
) -> List[List[str]]:
    """Choose deterministic same-dtype/device field chunks worth packing."""
    groups: Dict[Tuple[torch.dtype, torch.device], List[str]] = {}
    for key in sorted(values_by_key):
        values = values_by_key[key]
        if _is_uniform_tensor_field(values):
            first = values[0]
            groups.setdefault((first.dtype, first.device), []).append(key)

    chunks: List[List[str]] = []
    collective_world_size = collective_world_size or accelerator.num_processes
    for names in groups.values():
        if len(names) < 2:
            continue
        if target_device.type != "cpu":
            chunks.append(names)
            continue

        current: List[str] = []
        current_bytes = 0
        for name in names:
            values = values_by_key[name]
            field_bytes = (
                len(values) * values[0].numel() * values[0].element_size() * collective_world_size
            )
            if field_bytes > _CPU_PACKED_GATHER_MAX_BYTES:
                if len(current) >= 2:
                    chunks.append(current)
                current = []
                current_bytes = 0
                continue
            if current and current_bytes + field_bytes > _CPU_PACKED_GATHER_MAX_BYTES:
                if len(current) >= 2:
                    chunks.append(current)
                current = []
                current_bytes = 0
            current.append(name)
            current_bytes += field_bytes
        if len(current) >= 2:
            chunks.append(current)
    return chunks


def _gather_packed_tensor_fields(
    accelerator: Accelerator,
    values_by_key: Mapping[str, List[Any]],
    keys: List[str],
    target_device: torch.device,
    group_coordinator: Optional["GroupCoordinator"] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Gather one same-dtype field chunk and reconstruct its original shapes."""
    batch_size = len(values_by_key[keys[0]])
    widths = {key: values_by_key[key][0].numel() for key in keys}
    total_width = sum(widths.values())
    first = values_by_key[keys[0]][0]
    packed = torch.empty(
        (batch_size, total_width),
        device=accelerator.device,
        dtype=first.dtype,
    )
    offset = 0
    for key in keys:
        width = widths[key]
        stacked = torch.stack(values_by_key[key]).to(accelerator.device)
        packed[:, offset : offset + width].copy_(stacked.reshape(batch_size, width))
        offset += width

    gathered = (
        group_coordinator.gather(packed)
        if group_coordinator is not None and not group_coordinator.uses_global_collective
        else accelerator.gather(packed)
    )
    if not isinstance(gathered, torch.Tensor):
        raise TypeError(
            "expected packed sample-field gather to return torch.Tensor, "
            f"received {type(gathered).__name__}: {gathered!r}"
        )
    if gathered.ndim != 2 or gathered.shape[1] != total_width:
        raise ValueError(
            "expected packed sample-field gather to preserve width "
            f"{total_width}, received shape {tuple(gathered.shape)}"
        )
    gathered = gathered.to(target_device)
    reconstructed: Dict[str, List[torch.Tensor]] = {}
    offset = 0
    for key in keys:
        width = widths[key]
        sample_shape = values_by_key[key][0].shape
        field = gathered[:, offset : offset + width].reshape(
            gathered.shape[0],
            *sample_shape,
        )
        reconstructed[key] = list(field)
        offset += width
    return reconstructed


def gather_samples(
    accelerator: Accelerator,
    samples: List[BaseSample],
    field_names: List[str],
    device: Union[str, torch.device] = torch.device("cpu"),
    group_coordinator: Optional["GroupCoordinator"] = None,
    extra_field_names: Optional[List[str]] = None,
) -> List[BaseSample]:
    """Gather a list of BaseSample instances from all ranks.

    Args:
        accelerator: Accelerator instance.
        samples: Local samples on this rank. All entries must share one concrete sample class.
        field_names: Consumer-requested fields to gather. When ``extra_kwargs`` is included, each
            key is gathered independently. Fields declared by the concrete class in
            ``reconstruction_required_fields`` are always added before reconstruction.
        device: Target device for tensor fields in the returned samples.
        extra_field_names: Optional subset of ``extra_kwargs`` keys to gather.
            ``None`` preserves the existing behavior when ``extra_kwargs`` is
            requested in ``field_names``; an explicit list avoids transporting
            unrelated reward bookkeeping for consumers such as DPO pairing.

    Returns:
        List[BaseSample]: Samples from all ranks, concatenated in rank order.
    """
    if not samples:
        return []

    sample_cls = samples[0].__class__
    device = torch.device(device)

    # Separate extra_kwargs from regular fields
    reconstruction_fields = sample_cls.reconstruction_required_fields
    gathered_fields = set(field_names) | set(reconstruction_fields)
    has_extra_kwargs = "extra_kwargs" in gathered_fields or extra_field_names is not None
    regular_fields = sorted(f for f in gathered_fields if f != "extra_kwargs")
    extra_keys: List[str] = []
    if has_extra_kwargs:
        extra_keys = (
            sorted({k for s in samples for k in s.extra_kwargs})
            if extra_field_names is None
            else sorted(set(extra_field_names))
        )

    all_keys = regular_fields + [f"{_EXTRA_PREFIX}{k}" for k in extra_keys]
    d: dict = {key: [] for key in all_keys}
    values_by_key: Dict[str, List[Any]] = {}
    for key in all_keys:
        if key.startswith(_EXTRA_PREFIX):
            extra_k = key[len(_EXTRA_PREFIX) :]
            field_values = [s.extra_kwargs.get(extra_k) for s in samples]
        else:
            field_values = [getattr(sample, key) for sample in samples]
        values_by_key[key] = field_values

    type_hints = _sample_type_hints(sample_cls)
    scalar_kinds = {
        key: kind
        for key in regular_fields
        if (kind := _scalar_annotation_kind(type_hints.get(key))) is not None
    }
    for key in all_keys:
        if not key.startswith(_EXTRA_PREFIX):
            continue
        non_null = [value for value in values_by_key[key] if value is not None]
        if non_null and all(type(value) is type(non_null[0]) for value in non_null):
            if type(non_null[0]) in {bool, int, float, str}:
                scalar_kinds[key] = type(non_null[0])
    metadata, metadata_keys, string_lengths, n_gathered = _gather_sample_metadata(
        accelerator,
        values_by_key,
        scalar_kinds,
        device,
        group_coordinator,
    )
    d.update(metadata)
    if string_lengths:
        d.update(
            _gather_packed_string_fields(
                accelerator,
                values_by_key,
                string_lengths,
                group_coordinator,
            )
        )
        metadata_keys.update(string_lengths)
    remaining_values = {
        key: values for key, values in values_by_key.items() if key not in metadata_keys
    }

    # Same-dtype numeric fields share one gather. For CPU output, cap the
    # global payload: a large monolithic D2H copy benchmarks substantially
    # slower than field-wise copies even though it launches fewer collectives.
    packed_keys = set()
    collective_world_size = (
        group_coordinator.group_world_size
        if group_coordinator is not None
        else accelerator.num_processes
    )
    for chunk in _packed_tensor_field_chunks(
        accelerator,
        remaining_values,
        device,
        collective_world_size=collective_world_size,
    ):
        d.update(
            _gather_packed_tensor_fields(
                accelerator,
                remaining_values,
                chunk,
                device,
                group_coordinator=group_coordinator,
            )
        )
        packed_keys.update(chunk)

    for key in remaining_values:
        if key not in packed_keys:
            d[key] = _gather_field_values(
                accelerator,
                values_by_key[key],
                device,
                group_coordinator=group_coordinator,
            )

    # Reconstruct BaseSample objects
    if n_gathered == 0 and all_keys:
        n_gathered = len(d[all_keys[0]])
    gathered_samples = []
    for i in range(n_gathered):
        kwargs = {f: d[f][i] for f in regular_fields}
        if has_extra_kwargs:
            kwargs["extra_kwargs"] = {k: d[f"{_EXTRA_PREFIX}{k}"][i] for k in extra_keys}
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


def global_mean_std_numpy(accelerator: Accelerator, x: np.ndarray) -> Tuple[float, float]:
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
        t = torch.tensor([0.0, 0.0, 0.0], device=accelerator.device, dtype=torch.float64)
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


def global_min_max_numpy(accelerator: Accelerator, x: np.ndarray) -> Tuple[float, float]:
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
    # Fuse min & max into one MIN all-reduce over [lo, -hi] (max(h) == -min(-h)).
    packed = torch.tensor([lo, -hi], device=accelerator.device, dtype=torch.float64)
    if _is_distributed():
        dist.all_reduce(packed, op=dist.ReduceOp.MIN)
    lo = float(packed[0].item())
    hi = -float(packed[1].item())
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


def global_mean_of_scalar_per_group(accelerator: Accelerator, g_stds: np.ndarray) -> float:
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
    t = torch.tensor([local_sum, local_count], device=accelerator.device, dtype=torch.float64)
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


def global_std_of_group_means(accelerator: Accelerator, g_means: np.ndarray) -> float:
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
        t = torch.tensor([0.0, 0.0, 0.0], device=accelerator.device, dtype=torch.float64)
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
    zero_std_count = sum(1 for gid in unique_groups if np.std(rewards[group_indices == gid]) < eps)
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
        Uses 2 all-reduce calls: one SUM for a packed ``(count, sum, sum_sq)``
        triple, and one MIN that fuses the global min with the negated global
        max (``max(h) == -min(-h)``).  Single-process runs skip collective ops.
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

    packed = torch.tensor([count, total, sum_sq], device=accelerator.device, dtype=torch.float64)
    packed = accelerator.reduce(packed, reduction="sum")
    global_count = packed[0].item()
    global_sum = packed[1].item()
    global_sum_sq = packed[2].item()

    # Fuse min & max into one MIN all-reduce over [local_min, -local_max].
    extrema = torch.tensor([local_min, -local_max], device=accelerator.device, dtype=torch.float64)
    if _is_distributed():
        dist.all_reduce(extrema, op=dist.ReduceOp.MIN)
    global_min = float(extrema[0].item())
    global_max = -float(extrema[1].item())

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
    """Compute global stats for multiple tensors with only 2 all-reduce calls.

    Args:
        accelerator: Accelerator instance.
        tensors: Mapping from metric name to a 1-D tensor (local shard on
            this rank).

    Returns:
        Dict[str, Dict[str, float]]:
            ``{name: {'min': ..., 'max': ..., 'mean': ..., 'std': ...}}``.

    Note:
        All tensors' ``(count, sum, sum_sq)`` are packed into one SUM reduce;
        local mins and negated local maxes are packed into a single MIN reduce
        (``max(h) == -min(-h)``).  The total communication cost is **2
        all-reduce calls** regardless of the number of tensors. Keys are sorted
        so packed slot order matches across ranks; every rank must pass the
        **same** set of metric names.
    """
    if not tensors:
        return {}

    keys = sorted(tensors)

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

    # Fuse min & max into one MIN all-reduce over [local_mins, -local_maxes]
    # (max(h) == -min(-h)); negate the second half back to recover the maxes.
    n = len(keys)
    packed_extrema = torch.tensor(
        local_mins + [-m for m in local_maxes], device=device, dtype=torch.float64
    )
    if _is_distributed():
        dist.all_reduce(packed_extrema, op=dist.ReduceOp.MIN)
    packed_min = packed_extrema[:n]
    packed_max = -packed_extrema[n:]

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

    # Per-sample tensors: batched global stats (2 all-reduce calls)
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
