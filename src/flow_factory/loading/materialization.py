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

"""Materialize target roots directly in their prepared parameter layout.

A target root's loaded parameter values are retained as a source while the
root's parameters move to the meta device. A ``CheckpointParameter`` instead
records where its value lives, so every rank reads only its own block of the
checkpoint. Tensor/sequence parallel plans and
FSDP2 then build the final layout on meta, so no rank ever holds the unsharded
root and every new shard at the same time. ``ParameterSource.materialize``
finally allocates each rank's local shards and copies the matching region of
the source into them.

The region comes from the final parameter's own mesh and placements. Every
placement that FSDP2 and the tensor-parallel styles produce for parameters
(``Shard``, FSDP2's ``_StridedShard`` and ``Replicate``) selects one
contiguous block of the global tensor, so a rank reads only its own bytes and
the result equals the replicated path, where the unsharded value is
distributed by the same placements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard

from ..utils.logger_utils import setup_logger
from .checkpoint_slices import SliceReader, checkpoint_reader

logger = setup_logger(__name__)

__all__ = [
    "MaterializationReport",
    "ParameterSource",
    "detach_parameters_to_meta",
]


def _identity(name: str) -> str:
    return name


@dataclass
class _SourceEntry:
    names: Tuple[str, ...]
    shape: torch.Size
    dtype: torch.dtype
    value: Optional[torch.Tensor]
    consumed: bool = False
    reader: Optional[SliceReader] = None


@dataclass(frozen=True)
class MaterializationReport:
    """Counts from one materialization, for logging and tests."""

    parameters: int
    local_bytes: int
    source_bytes: int


@dataclass
class ParameterSource:
    """Loaded parameter values of one root, keyed by every canonical alias.

    Args:
        entries: One entry per unique parameter, reachable through each alias.
        source_rank: ``None`` when every rank holds an identical value, or the
            global rank that alone holds them; the others receive each value
            by broadcast during materialization.
    """

    entries: Dict[str, _SourceEntry]
    source_rank: Optional[int] = None
    _released: bool = field(default=False, init=False)

    def materialize(
        self,
        root: nn.Module,
        device: torch.device,
        *,
        canonical_name: Callable[[str], str] = _identity,
    ) -> MaterializationReport:
        """Allocate ``root``'s final local parameters and fill them from the source.

        Args:
            root: The prepared module whose parameters are all on meta.
            device: Device for the local parameter storage.
            canonical_name: Maps a prepared parameter name, which may contain
                distributed wrapper segments, back to the name recorded by
                :func:`detach_parameters_to_meta`.

        Returns:
            Counts of materialized parameters and local/source bytes.

        Raises:
            RuntimeError: If the source was already released.
            ValueError: If a parameter is not on meta, a source is missing,
                shapes disagree, a placement is unsupported, or a recorded
                source was not consumed.
        """
        if self._released:
            raise RuntimeError("parameter source was already released by an earlier materialize")
        resident = [name for name, parameter in root.named_parameters() if not parameter.is_meta]
        if resident:
            raise ValueError(
                "materialization expects every parameter on the meta device; "
                f"resident={resident[:5]} ({len(resident)} total)"
            )
        broadcast = self.source_rank is not None and dist.is_initialized()
        rank = dist.get_rank() if broadcast else None
        buffers: List[Tuple[nn.Module, str, torch.Tensor]] = [
            (module, name, buffer)
            for module in root.modules()
            for name, buffer in module._buffers.items()
            if buffer is not None
        ]
        meta_buffers = [name for _, name, buffer in buffers if buffer.is_meta]
        if meta_buffers:
            raise ValueError(
                f"materialization expects real buffers; meta buffers={meta_buffers[:5]}"
            )
        root.to_empty(device=device)
        with torch.no_grad():
            for module, name, previous in buffers:
                module._buffers[name].copy_(previous)
        del buffers
        # FSDP2 keeps parameter identities through ``to_empty``; plain modules may not.
        parameters = list(root.named_parameters())

        local_bytes = source_bytes = 0
        with torch.no_grad():
            for name, parameter in parameters:
                key = canonical_name(name)
                entry = self.entries.get(key)
                if entry is None:
                    raise ValueError(
                        f"no loaded source for prepared parameter {name!r} "
                        f"(canonical {key!r}); recorded examples={list(self.entries)[:5]}"
                    )
                if tuple(parameter.shape) != tuple(entry.shape):
                    raise ValueError(
                        f"prepared parameter {name!r} has global shape {tuple(parameter.shape)}, "
                        f"but its source {entry.names[0]!r} has {tuple(entry.shape)}"
                    )
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                offset, shape = _local_region(name, parameter)
                if tuple(local.shape) != shape:
                    raise ValueError(
                        f"prepared parameter {name!r} has local shape {tuple(local.shape)}, "
                        f"but its placements select {shape}"
                    )
                value = entry.value
                if entry.reader is not None:
                    value = entry.reader.read(offset, shape).to(entry.dtype)
                    offset = (0,) * len(shape)
                elif broadcast:
                    full = (
                        value.to(device)
                        if rank == self.source_rank
                        else torch.empty(entry.shape, dtype=entry.dtype, device=device)
                    )
                    dist.broadcast(full, src=self.source_rank)
                    value = full
                region = value
                for dim, (start, size) in enumerate(zip(offset, shape)):
                    region = region.narrow(dim, start, size)
                local.copy_(region)
                entry.consumed = True
                local_bytes += local.numel() * local.element_size()
                source_bytes += region.numel() * region.element_size()
        unconsumed = sorted(
            {entry.names[0] for entry in self.entries.values() if not entry.consumed}
        )
        if unconsumed:
            raise ValueError(
                "loaded sources were not consumed by any prepared parameter: "
                f"{unconsumed[:5]} ({len(unconsumed)} total)"
            )
        self.release()
        return MaterializationReport(len(parameters), local_bytes, source_bytes)

    def release(self) -> None:
        """Drop every retained source value."""
        for entry in self.entries.values():
            entry.value = None
        self._released = True


def _local_region(name: str, parameter: torch.Tensor) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return the global offset and shape of this rank's block of ``parameter``."""
    if not isinstance(parameter, DTensor):
        return (0,) * parameter.dim(), tuple(parameter.shape)
    unsupported = [
        placement
        for placement in parameter.placements
        if not isinstance(placement, (Shard, _StridedShard, Replicate))
    ]
    if unsupported:
        raise ValueError(
            f"prepared parameter {name!r} has placements {parameter.placements}; "
            f"materialization supports Shard, strided Shard and Replicate, not {unsupported}"
        )
    shape, offset = compute_local_shape_and_global_offset(
        parameter.shape, parameter.device_mesh, parameter.placements
    )
    return tuple(offset), tuple(shape)


def detach_parameters_to_meta(
    root: nn.Module,
    *,
    canonical_name: Callable[[str], str] = _identity,
    source_rank: Optional[int] = None,
) -> ParameterSource:
    """Record ``root``'s loaded parameter values and replace them with meta parameters.

    Shared parameters keep one meta replacement, so aliases stay tied. Buffers,
    ``requires_grad`` and dtypes are unchanged.

    Args:
        root: Unprepared module whose parameters hold their loaded values.
        canonical_name: Maps each parameter name to the key that prepared names
            will canonicalize to.
        source_rank: ``None`` when every rank loaded identical values. Otherwise
            the global rank that alone holds them; other ranks may hold meta or
            uninitialized values, and only that rank's values are retained.

    Returns:
        The source for :meth:`ParameterSource.materialize`.

    Raises:
        TypeError: If a parameter is already distributed.
        ValueError: If a parameter on the source rank is already on meta, or two
            unique parameters share a canonical name.
    """
    holds_source = source_rank is None or (dist.is_initialized() and dist.get_rank() == source_rank)
    names_by_id: Dict[int, List[str]] = {}
    for name, parameter in root.named_parameters(remove_duplicate=False):
        names_by_id.setdefault(id(parameter), []).append(name)
    entries: Dict[str, _SourceEntry] = {}
    replacements: Dict[int, nn.Parameter] = {}
    seen_modules = set()
    for module in root.modules():
        if id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        for leaf, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            replacement = replacements.get(id(parameter))
            if replacement is None:
                names = tuple(names_by_id[id(parameter)])
                if isinstance(parameter, DTensor):
                    raise TypeError(
                        f"parameter {names[0]!r} is already distributed; detach target "
                        "parameters before tensor parallel or FSDP2 preparation"
                    )
                # Every rank reads a checkpoint-backed parameter from its checkpoint.
                reader = checkpoint_reader(parameter)
                if reader is None and holds_source and parameter.is_meta:
                    raise ValueError(
                        f"parameter {names[0]!r} has no loaded value on the meta device"
                    )
                entry = _SourceEntry(
                    names=names,
                    shape=parameter.shape,
                    dtype=parameter.dtype,
                    value=parameter.detach() if holds_source and reader is None else None,
                    reader=reader,
                )
                for alias in {canonical_name(name) for name in names}:
                    if alias in entries:
                        raise ValueError(
                            f"canonical parameter name {alias!r} is shared by "
                            f"{entries[alias].names[0]!r} and {names[0]!r}"
                        )
                    entries[alias] = entry
                replacement = nn.Parameter(
                    torch.empty_like(parameter, device="meta"),
                    requires_grad=parameter.requires_grad,
                )
                replacements[id(parameter)] = replacement
            module._parameters[leaf] = replacement
    unique = {id(entry): entry for entry in entries.values()}.values()
    logger.info(
        "Moved %d target parameters to meta; retained %.2f GiB of loaded values on this rank",
        len(replacements),
        sum(
            entry.value.numel() * entry.value.element_size()
            for entry in unique
            if entry.value is not None
        )
        / 2**30,
    )
    return ParameterSource(entries=entries, source_rank=source_rank)
