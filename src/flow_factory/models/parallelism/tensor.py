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


"""Parameter placement styles and head-normalization arithmetic."""

from __future__ import annotations

import types
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Placement,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle


def _distribute_own_parameters(
    module: nn.Module,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
    src_data_rank: int | None,
) -> Tuple[str, ...]:
    """Make each of a module's *own* parameters a DTensor, keeping trainability.

    Forwarding ``requires_grad`` is the reason these styles exist at all:
    ``torch``'s own styles construct the replacement ``nn.Parameter`` with the
    default, which unfreezes a frozen base layer and changes the gradient count
    under LoRA. Asserting that in one place rather than four is the point of
    this helper.

    ``distribute_tensor`` broadcasts from ``src_data_rank``, making one rank's
    weights authoritative instead of trusting that every rank initialized or
    loaded identically.

    Returns:
        The names the module now owns as DTensors, in registration order, so a
        forward hook can walk a fixed tuple instead of rebuilding
        ``named_parameters`` on every call.
    """
    names = []
    for name, parameter in list(module.named_parameters(recurse=False)):
        if not isinstance(parameter, DTensor):
            module.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(
                        parameter.data,
                        device_mesh,
                        list(placements),
                        src_data_rank=src_data_rank,
                    ),
                    requires_grad=parameter.requires_grad,
                ),
            )
        names.append(name)
    return tuple(names)


class _OwnParameterStyle(ParallelStyle):
    """Base for the styles that distribute a module's own parameters.

    Subclasses go through :func:`_distribute_own_parameters` and differ in the
    placement they ask for and in what they do afterwards. Their source rank is
    ``ParallelStyle.src_data_rank``, which ``parallelize_module`` assigns before
    it applies the style.
    """


class _LocalViewStyle(_OwnParameterStyle):
    """Distribute direct parameters while exposing plain tensor views during forward.

    A pre-hook shadows each direct parameter in the module's instance dictionary;
    an ``always_call=True`` post-hook removes that view even if forward raises.
    The registered parameter stays a DTensor under its original name, preserving
    optimizer and checkpoint ownership. FSDP2 and state-dict readers must never
    observe a stale plain-tensor shadow between calls.

    Subclasses choose the local view and its backward placement: replicated
    computation uses ``to_local()``, while head- or token-partial contributions
    must mark their gradient ``Partial`` over the style's TP mesh.

    Checkpoint recomputation must re-enter the module call to reinstall the
    view. A checkpointed inner function that reads its owner's parameter after
    the outer post-hook has removed the shadow is unsafe. H3's chunked RMSNorm
    reads its weight inside token chunks and therefore does not checkpoint those
    chunks independently.

    Hooks capture parameter names and the style's view functions at application
    time. Parameter lookup remains live so replacement by FSDP2 is respected;
    conditional styles may retain additional owner state in their view functions.
    """

    #: How the module's own parameters are distributed before ``forward`` sees
    #: their local views.
    _placements: Tuple[Placement, ...] = (Replicate(),)

    def _local_view(self, parameter: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
        """Return the plain tensor ``forward`` should see for this parameter."""
        raise NotImplementedError

    def _local_view_for(self, name: str):
        """Return the view function for the parameter called ``name``."""
        return self._local_view

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        names = _distribute_own_parameters(
            module, device_mesh, self._placements, self.src_data_rank
        )
        views = {name: self._local_view_for(name) for name in names}

        def shadow(owner: nn.Module, _inputs) -> None:
            shadows, parameters = owner.__dict__, owner._parameters
            for name in names:
                parameter = parameters[name]
                if isinstance(parameter, DTensor):
                    shadows[name] = views[name](parameter, device_mesh)

        def drop(owner: nn.Module, _inputs, _outputs) -> None:
            shadows = owner.__dict__
            for name in names:
                shadows.pop(name, None)

        module.register_forward_pre_hook(shadow)
        module.register_forward_hook(drop, always_call=True)
        return module


class ReplicateParallel(_LocalViewStyle):
    """Hold a module's direct parameters replicated and compute on local views.

    This completes parameters unnamed by an adapter's shard plan, keeping all
    gradients on compatible DTensor meshes for clipping. ``_LocalViewStyle``
    preserves their registered names and exposes plain views only during forward.

    Plain activations allow modules with both bare parameters and child modules
    to use ordinary residual arithmetic. They also avoid DTensor convolution
    backward marking an already-complete replicated weight gradient ``Partial``
    and reducing it again. ``to_local()`` restores replicated gradient placement.

    This style assumes every TP peer computes the complete parameter gradient.
    For parameters applied only to local heads or tokens, use
    ``ShardedInputReplicatedWeight`` or ``SequenceBroadcastParameter`` instead.
    """

    def _local_view(self, parameter: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
        """A plain view of a replicated parameter, with ``to_local``'s backward.

        ``to_local`` rewraps the gradient with the forward placements, so it
        arrives ``Replicate`` and nothing reduces it -- which is exactly right
        here, where every rank already computed the whole gradient. The
        sequence-parallel region is where that assertion becomes false; see
        :class:`SequenceBroadcastParameter`.
        """
        return _replicated_view(parameter, device_mesh)


def _replicated_view(parameter: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
    del device_mesh
    return parameter.to_local()


class SharedInputColwiseParallel(_LocalViewStyle):
    """Column-shard a projection whose input gradient its region entry reduces.

    torch's ``ColwiseParallel`` lifts its own input with ``DTensor.from_local``,
    so its backward all-reduces the input gradient -- once per projection. q, k
    and v read one activation, so an attention paid three full-size all-reduces
    per stream where one suffices, and under sequence parallelism three
    all-reduces where one reduce-scatter suffices. This style computes the
    projection on this rank's rows of the weight and leaves its input gradient a
    partial sum; the :class:`ColumnParallelRegion` or
    :class:`SequenceParallelRegion` on the enclosing module sums every
    consumer's partial gradient once. Megatron splits the work the same way,
    with its ``f`` operator at the region entry.

    The weight and bias are ``Shard(0)``, exactly as ``ColwiseParallel`` places
    them, so checkpoints, head localization and FSDP2 composition see no
    difference. ``to_local``'s backward rewraps their gradients ``Shard(0)``,
    which is right: the input is whole on every rank, so each rank computed the
    entire gradient of its own rows.

    Outside a region the input gradient would stay a partial sum -- the loss
    bit-identical and everything upstream training on this rank's share -- so
    :func:`apply_tensor_parallel` refuses a plan that leaves one without an
    enclosing entry, and checks on the first gradient-recording call that the
    input really came through it. ``verify_input`` is off only for LoRA's
    ``lora_B``, whose input is ``lora_A``'s output by construction.
    """

    _placements = (Shard(0),)

    def __init__(self, *, verify_input: bool = True) -> None:
        super().__init__()
        if not isinstance(verify_input, bool):
            raise TypeError(
                f"expected a bool verify_input, received {type(verify_input).__name__}: "
                f"{verify_input!r}"
            )
        self.verify_input = verify_input

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"SharedInputColwiseParallel applies to nn.Linear, received {type(module).__name__}"
            )
        return super()._apply(module, device_mesh)

    def _local_view(self, parameter: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
        return _replicated_view(parameter, device_mesh)


class ShardedAcrossHeadRMSNorm(_OwnParameterStyle):
    """Shard an RMSNorm whose normalized axis spans every head.

    Wan and LTX-2 build qk-norm over ``num_heads * head_dim`` and apply it to
    the flat projection output, *before* heads are split. That makes it a
    different problem from the per-head case
    :class:`ShardedInputReplicatedWeight` solves: there the weight is shared
    across heads and only the gradient is wrong, here the normalized axis is
    itself what column parallelism split, so a replicated weight makes the
    **forward** wrong -- ``F.rms_norm`` raises on the normalized shape rather
    than failing quietly, which is how the case was found.

    The fix is to shard the weight ``Shard(0)``, keeping it aligned with the
    column-parallel projection feeding it, and to complete the reduction the
    normalization needs: each rank sums the squares of its slice and an
    all-reduce adds the slices up. The divisor stays the **global** normalized
    size, because that is what the unsharded module divided by. What follows
    the reduction is ``F.rms_norm``'s arithmetic: a half-precision input is
    scaled and weighted in float32 and rounded once.

    An unpaired norm adds one ``[batch, seq, 1]`` statistic all-reduce in each
    direction. The applier combines compatible Q/K statistics into one paired
    collective; these exchanges remain additional to projection traffic.

    ``forward`` is replaced rather than hooked because the math itself changes:
    ``nn.RMSNorm.forward`` calls ``F.rms_norm`` with the module's
    ``normalized_shape``, which no longer describes this rank's slice.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, nn.RMSNorm):
            raise TypeError(
                "ShardedAcrossHeadRMSNorm applies to nn.RMSNorm, received "
                f"{type(module).__name__}"
            )
        if len(module.normalized_shape) != 1:
            raise ValueError(
                "ShardedAcrossHeadRMSNorm expects a 1-D normalized shape, received "
                f"{tuple(module.normalized_shape)}"
            )
        normalized_size = int(module.normalized_shape[0])
        if normalized_size % device_mesh.size() != 0:
            raise ValueError(
                f"normalized size {normalized_size} is not divisible by the "
                f"tensor-parallel degree {device_mesh.size()}"
            )
        epsilon = module.eps if module.eps is not None else torch.finfo(torch.float32).eps

        # `Shard(0)` keeps the weight aligned with the column-parallel
        # projection feeding it.
        _distribute_own_parameters(module, device_mesh, (Shard(0),), self.src_data_rank)
        # Read by `_pair_across_head_norms`, which computes this norm inside its
        # query partner's call; plain numbers, so a deep copy carries them.
        module._ff_across_head_norm = (normalized_size, epsilon)

        tp_group = device_mesh.get_group()

        def forward(self: nn.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
            local = (
                hidden_states.to_local() if isinstance(hidden_states, DTensor) else hidden_states
            )
            output_dtype = local.dtype
            local = _norm_accumulation_input(local)
            # Every rank consumes the summed statistic, so its backward is a
            # fan-in over the group -- an all-reduce again, not Megatron's
            # identity `g`. Measured against `nn.RMSNorm`, the identity keeps only
            # this rank's share and redistributing a `Partial` term to
            # `Replicate` rescales it; both leave the forward and the weight
            # gradient right and put the input gradient 32% and 98% off.
            square_sum = differentiable_all_reduce(_across_head_square_sum(local), group=tp_group)
            return _across_head_normalize(
                local, square_sum, normalized_size, epsilon, self.weight
            ).to(output_dtype)

        module.forward = types.MethodType(forward, module)
        return module


def _norm_accumulation_input(local: torch.Tensor) -> torch.Tensor:
    """Share one FP32 view across both derivative branches of a low-precision norm.

    Casting each branch separately rounds the direct and statistic gradients
    before they cancel. The shared cast accumulates both in FP32 and rounds the
    resulting input gradient once, as the native RMSNorm backward does. This
    retains one FP32 activation per local query/key slice during backward.
    """
    return local.float() if local.dtype.itemsize < 4 else local


def _across_head_square_sum(local: torch.Tensor) -> torch.Tensor:
    """This rank's share of the sum of squares over the normalized axis.

    ``vector_norm`` avoids materializing a full elementwise square. Low-precision
    inputs already share an FP32 accumulation view with the normalization branch;
    independent branch casts would corrupt cancellation in backward. The dtype
    argument also preserves widening for direct callers and never narrows FP64.
    """
    accumulate_in = torch.float32 if local.dtype.itemsize < 4 else local.dtype
    return torch.linalg.vector_norm(local, 2, dim=-1, keepdim=True, dtype=accumulate_in).pow(2)


def _across_head_normalize(
    local: torch.Tensor,
    square_sum: torch.Tensor,
    normalized_size: int,
    epsilon: float,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """Scale ``local`` by the group's summed statistic and its slice of the weight.

    Products stay in the statistic's dtype and round once, as ``F.rms_norm``'s
    do: casting the scale to bf16 first moved 35% of outputs off it. This
    rank's slice of the weight lines up with its slice of the feature axis, and
    ``to_local``'s backward hands the gradient back ``Shard(0)`` -- each rank
    computed the gradient for its own slice.
    """
    normalized = local * torch.rsqrt(square_sum / normalized_size + epsilon)
    if weight is not None:
        normalized = normalized * weight.to_local()
    return normalized.to(local.dtype)


class ShardedInputReplicatedWeight(_LocalViewStyle):
    """Replicate a module's weight whose input holds only this rank's heads.

    Per-head qk-norm is why this exists. ``norm_q.weight`` is ``[head_dim]`` and
    shared by every head, so the weight itself is replicated -- but
    column-parallel q/k leave this rank holding only ``heads / tp_size`` heads,
    so the weight gradient it accumulates is a partial sum over its own heads.
    Treated as plainly replicated, nothing reconciles those sums and the
    gradient is silently wrong.

    ``forward`` computes on plain local tensors and is handed a view of the
    weight whose backward declares the gradient ``Partial`` -- the view
    :class:`SequenceBroadcastParameter` uses for a token shard. Living in the
    autograd graph rather than in a hook on the parameter is what makes it
    survive ``fully_shard``: FSDP replaces the parameter objects, so a
    ``register_post_accumulate_grad_hook`` installed beforehand is simply lost,
    while the autograd graph is rebuilt every step.

    Lifting the input to a ``Shard`` DTensor instead derives the same gradient
    bit for bit, at the price of sending every op of the norm through DTensor
    dispatch, and it breaks a norm that assembles its output into a fresh
    buffer, as H3's chunked RMSNorm does: ``new_empty`` on a sharded DTensor
    allocates a replicated one.

    The normalized axis is the last one and never the sharded one, so the
    forward needs no collective, and the weight gradient sums over every other
    axis, so it does not depend on which of them holds the heads.
    ``shard_axis`` names that axis for the reader of a plan -- 1 where the
    processor transposes before normalizing, as SD3's does, 2 where it does not.
    """

    def __init__(self, *, shard_axis: int = 2) -> None:
        super().__init__()
        if isinstance(shard_axis, bool) or not isinstance(shard_axis, int):
            raise TypeError(
                f"expected an int shard_axis, received {type(shard_axis).__name__}: "
                f"{shard_axis!r}"
            )
        if shard_axis < 0:
            raise ValueError(f"expected a non-negative shard_axis, received {shard_axis}")
        self.shard_axis = shard_axis

    def _local_view(self, parameter: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
        """A plain view whose backward declares the gradient a partial sum over heads."""
        return _ToLocalWithPartialGradient.apply(parameter, device_mesh)


class _ToLocalWithPartialGradient(torch.autograd.Function):
    """Identity to the local tensor, with the gradient declared a partial sum.

    ``DTensor.to_local()`` rewraps its gradient with the *forward* placements,
    so a ``Replicate`` parameter gets a ``Replicate`` gradient -- an assertion
    that every rank computed the same value. That is false for a parameter
    applied to a shard: broadcast over a sharded token axis, or shared by heads
    of which this rank holds only some. Each rank then holds part of the sum,
    and declaring the gradient ``Partial`` is what makes DTensor add them up.

    The mesh is passed in, not read off the parameter. By the time the forward
    runs, ``fully_shard`` has replaced the parameter with one on the
    *concatenated* ``(dp_shard, tp)`` mesh -- and ``from_local`` accepts a
    one-element placement list there without complaint, putting the ``Partial``
    on mesh dimension 0, which is ``dp_shard``. The missing token-slice sum
    would then be reduced over data-parallel replicas, whose gradients FSDP has
    already reduced, and never over the tensor-parallel peers that hold the
    missing terms. Measured on a 2x2 gloo mesh: the one-placement form summed
    ranks {0, 2} where the correct ``[Replicate(), Partial()]`` sums {0, 1}.
    """

    @staticmethod
    def forward(ctx, parameter: DTensor, mesh: DeviceMesh) -> torch.Tensor:  # type: ignore[override]
        ctx.mesh = mesh
        # Never `_local_tensor` itself: autograd stamps this Function's
        # `grad_fn` on whatever it returns, which would make the parameter's own
        # storage a non-leaf that `deepcopy` refuses.
        local = parameter.to_local()
        return local.view_as(local)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # type: ignore[override]
        return DTensor.from_local(gradient, ctx.mesh, [Partial()], run_check=False), None
