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

"""Packed activation layouts over canonical, ordinary DTensor weight shards.

Weights retain their native two-dimensional shape and ordering. Only activations
move between contiguous projection shards and a slice of every logical block.
The native differentiable all-to-all implements the inverse exchange in backward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_to_all_single
from torch.distributed.tensor import DTensor, Placement
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


def _blocks(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) < 2 or any(type(value) is not int or value <= 0 for value in result):
        raise ValueError(
            f"packed block_sizes must contain at least two positive integers; received {result}"
        )
    return result


def _native_blocks(module: nn.Linear, name: str, width: int) -> tuple[int, ...]:
    values = getattr(module, name, None)
    if values is None:
        raise ValueError(f"packed projection requires block_sizes or native {name} metadata")
    blocks = _blocks(values)
    if sum(blocks) != width:
        raise ValueError(f"native {name}={blocks} must sum to packed width={width}")
    return blocks


@dataclass(frozen=True)
class _PackedExchange:
    """One rank's reversible feature permutation and peer split sizes."""

    send_indices: tuple[int, ...]
    receive_indices: tuple[int, ...]
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    mesh: DeviceMesh

    @classmethod
    def build(
        cls, width: int, blocks: tuple[int, ...], mesh: DeviceMesh, *, inverse: bool
    ) -> _PackedExchange:
        degree = mesh.size()
        if width % sum(blocks) or any((width // sum(blocks)) * block % degree for block in blocks):
            raise ValueError(
                f"packed width={width}, block_sizes={blocks} must define integer blocks "
                f"each divisible by tp_size={degree}"
            )
        unit = width // sum(blocks)
        local_width = width // degree
        contiguous = [
            list(range(rank * local_width, (rank + 1) * local_width)) for rank in range(degree)
        ]
        packed = [[] for _ in range(degree)]
        offset = 0
        for block in blocks:
            size = unit * block // degree
            for rank in range(degree):
                packed[rank].extend(range(offset + rank * size, offset + (rank + 1) * size))
            offset += unit * block
        sources, targets = (packed, contiguous) if inverse else (contiguous, packed)
        rank = mesh.get_local_rank()
        source_positions = {value: index for index, value in enumerate(sources[rank])}
        send = [
            [source_positions[value] for value in target if value in source_positions]
            for target in targets
        ]
        source_sets = [set(source) for source in sources]
        received = [[value for value in targets[rank] if value in source] for source in source_sets]
        received_positions = {
            value: index for index, value in enumerate(value for peer in received for value in peer)
        }
        return cls(
            tuple(index for peer in send for index in peer),
            tuple(received_positions[value] for value in targets[rank]),
            tuple(map(len, send)),
            tuple(map(len, received)),
            mesh,
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if isinstance(value, DTensor) or value.shape[-1] != len(self.send_indices):
            raise ValueError(
                f"packed exchange expects a local activation of width={len(self.send_indices)}; "
                f"received {type(value).__name__} shape={tuple(value.shape)}"
            )
        send_indices = torch.tensor(self.send_indices, device=value.device)
        receive_indices = torch.tensor(self.receive_indices, device=value.device)
        outgoing = value.index_select(-1, send_indices).movedim(-1, 0).contiguous()
        incoming = all_to_all_single(
            torch.empty_like(outgoing),
            outgoing,
            output_split_sizes=list(self.receive_counts),
            input_split_sizes=list(self.send_counts),
            group=self.mesh.get_group(),
        )
        return incoming.index_select(0, receive_indices).movedim(0, -1).contiguous()


class PackedColwiseParallel(ColwiseParallel):
    """Shard canonical output rows and exchange activations into per-block shards.

    Args:
        block_sizes: Relative widths of the native packed outputs, for example
            ``(1, 1, 1, 4, 4)`` for query/key/value and two MLP streams. When
            omitted, require the native Linear's ``_tp_packed_col_blocks`` widths.

    Outputs are local tensors containing this rank's contiguous slice of every
    block. Each forward and backward adds one activation all-to-all. Parameter
    names, dimensions, ordering, and native state serialization remain unchanged.
    """

    def __init__(self, block_sizes: Sequence[int] | None = None) -> None:
        super().__init__()
        self.block_sizes = None if block_sizes is None else _blocks(block_sizes)

    def resolve_block_sizes(self, module: nn.Linear) -> tuple[int, ...]:
        """Resolve native metadata before deriving PEFT projections without it."""
        return (
            self.block_sizes
            if self.block_sizes is not None
            else _native_blocks(module, "_tp_packed_col_blocks", module.out_features)
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"PackedColwiseParallel requires nn.Linear; received {type(module).__name__}"
            )
        exchange = _PackedExchange.build(
            module.out_features, self.resolve_block_sizes(module), device_mesh, inverse=False
        )
        super()._apply(module, device_mesh)
        module.register_forward_hook(lambda _module, _inputs, output: exchange(output))
        return module


class PackedRowwiseParallel(RowwiseParallel):
    """Exchange per-block input shards before a canonical row-parallel linear.

    Args:
        block_sizes: Relative widths of the concatenated logical input blocks.
            When omitted, require the native Linear's ``_tp_packed_row_blocks`` widths.
        output_layouts: Native torch output placement; defaults to replicated.
            A token-axis shard enables the ordinary reduce-scatter SP boundary.

    Input activations must be local per-block shards. The exchange is the exact
    inverse of :class:`PackedColwiseParallel`; weights retain native ordering.
    """

    def __init__(
        self, block_sizes: Sequence[int] | None = None, *, output_layouts: Placement | None = None
    ) -> None:
        super().__init__(output_layouts=output_layouts)
        self.block_sizes = None if block_sizes is None else _blocks(block_sizes)

    def resolve_block_sizes(self, module: nn.Linear) -> tuple[int, ...]:
        """Resolve the base projection's native widths for a derived LoRA A."""
        return (
            self.block_sizes
            if self.block_sizes is not None
            else _native_blocks(module, "_tp_packed_row_blocks", module.in_features)
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"PackedRowwiseParallel requires nn.Linear; received {type(module).__name__}"
            )
        exchange = _PackedExchange.build(
            module.in_features, self.resolve_block_sizes(module), device_mesh, inverse=True
        )
        module.register_forward_pre_hook(lambda _module, inputs: (exchange(inputs[0]), *inputs[1:]))
        return super()._apply(module, device_mesh)
