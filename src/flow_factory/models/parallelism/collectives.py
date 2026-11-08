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


"""Token and partial-gradient collectives with explicit backward semantics."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Placement, Replicate, Shard


class _AllReduceGradient(torch.autograd.Function):
    """Identity forward, summed gradient backward -- Megatron's ``f`` operator."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:  # type: ignore[override]
        ctx.mesh = mesh
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # type: ignore[override]
        partial = DTensor.from_local(gradient, ctx.mesh, [Partial()], run_check=False)
        return partial.redistribute(ctx.mesh, [Replicate()]).to_local(), None


class _RegionAllReduceGradient(_AllReduceGradient):
    """``f`` at a column-parallel region's entry.

    The same arithmetic as :class:`_AllReduceGradient`, as its own node type so
    that :func:`_passed_region_entry` can tell a region entry from the
    conditioning reductions of :class:`SequenceBroadcastOutput`, which sit a few
    autograd nodes upstream of most attention inputs.
    """


def _settled(value: torch.Tensor) -> torch.Tensor:
    """Return ``value`` with any pending functional collective waited on."""
    return value.wait() if isinstance(value, AsyncCollectiveTensor) else value


class _GatherTokens(torch.autograd.Function):
    """All-gather a token-sharded activation; reduce-scatter its gradient.

    The gradient arriving here is the sum of every consumer's partial gradient
    over the whole sequence, so one reduce-scatter both completes the sum and
    hands each rank its own tokens -- where ``PrepareModuleInput``'s local slice
    left each column-parallel consumer to all-reduce the whole sequence first.
    Both directions wait before returning, so nothing leaves in flight.
    """

    @staticmethod
    def forward(ctx, local: torch.Tensor, mesh: DeviceMesh, dim: int) -> torch.Tensor:  # type: ignore[override]
        ctx.mesh, ctx.dim = mesh, dim
        shard = DTensor.from_local(local, mesh, [Shard(dim)], run_check=False)
        return _settled(shard.redistribute(mesh, [Replicate()]).to_local())

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # type: ignore[override]
        partial = DTensor.from_local(gradient, ctx.mesh, [Partial()], run_check=False)
        return _settled(partial.redistribute(ctx.mesh, [Shard(ctx.dim)]).to_local()), None, None


#: Autograd node types a region entry leaves on the tensors it lifts.
_REGION_ENTRY_NODES: Tuple[type, ...] = (
    _RegionAllReduceGradient._backward_cls,
    _GatherTokens._backward_cls,
)


def _passed_region_entry(value: torch.Tensor, depth: int = 4) -> bool:
    """Whether ``value`` reaches a region entry within ``depth`` autograd nodes.

    A consumer reads the lifted tensor directly or through a view, a cast or a
    dropout -- Wan's image context is a slice of the lifted text stream -- so
    the entry is the producer or a few nodes above it. The previous block's
    entries are a dozen or more nodes up, through attention, the output
    projection, the residual and the norm, so a short walk cannot mistake one
    of them for this region's.
    """
    frontier = [value.grad_fn]
    seen = set()
    for _ in range(depth + 1):
        parents = []
        for node in frontier:
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, _REGION_ENTRY_NODES):
                return True
            parents.extend(parent for parent, _ in node.next_functions)
        frontier = parents
    return False


def gather_tokens(local: torch.Tensor, mesh: DeviceMesh, dim: int) -> torch.Tensor:
    """All-gather tokens, reduce-scatter partial input gradients, and settle both."""
    return _GatherTokens.apply(local, mesh, dim)


def reduce_region_gradient(value: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Mark a column-region entry and sum its input's partial backward contributions."""
    return _RegionAllReduceGradient.apply(value, mesh)


def redistribute_sequence(
    value: torch.Tensor,
    mesh: DeviceMesh,
    source: Placement,
    desired: Placement,
    *,
    owner: str,
) -> torch.Tensor:
    """Transfer a local BSH activation with DTensor's placement-derived backward.

    This is not ``gather_tokens``: that operator explicitly reduce-scatters
    consumer-partial gradients. Here DTensor derives the backward from the
    supplied placements. Clone materializes the result and releases storage
    belonging to a complete producer sequence after a scatter.
    """
    if not isinstance(value, torch.Tensor) or isinstance(value, DTensor) or value.ndim != 3:
        raise ValueError(f"{owner} token boundaries require a local rank-three activation")
    if isinstance(desired, Shard) and value.shape[1] % mesh.size():
        raise ValueError(
            f"{owner} tokens={value.shape[1]} must be divisible by tp_size={mesh.size()}"
        )
    return (
        DTensor.from_local(value, mesh, [source], run_check=False)
        .redistribute(placements=[desired])
        .to_local()
        .clone()
    )
