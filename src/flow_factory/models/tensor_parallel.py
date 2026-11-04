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

"""Tensor-parallel sharding of a trainable denoiser.

Flow-Factory applies its own plan rather than delegating to either upstream
implementation, for reasons that are not interchangeable:

* diffusers ships ``_tp_plan`` dicts and an applier, but it **refuses any model
  with PEFT adapters injected** (``diffusers/hooks/tensor_parallel.py``) because
  its plans cover base ``Linear`` layers only, which would leave LoRA unsharded
  and, on a rowwise layer, skip its all-reduce. LoRA is Flow-Factory's primary
  finetune mode. Its plan dicts remain the authoritative reference for which
  projections to shard, and are mirrored by the adapter hooks.
* Accelerate's ``_prepare_tp`` completes a partially-sharded model by importing
  ``transformers.integrations.tensor_parallel.ReplicateParallel``, a symbol the
  ``transformers`` 5.x line removed while ``pyproject.toml`` supports both 4.x
  and 5.x.

Three properties of the completed plan were established on real GPUs and are
what the pieces below exist to preserve:

1. **The plan must be total.** Any parameter the plan does not name stays a
   plain tensor (pure TP) or becomes a dp-only DTensor (TP+FSDP), and gradient
   clipping then raises -- "mixed torch.Tensor and DTensor" without FSDP, "All
   operands in aten.stack must have the same mesh" with it.
2. **A replicated weight whose input is a shard needs its own style.** Per-head
   qk-norm is the case: the weight is ``[head_dim]`` and shared by every head,
   but column-parallel q/k leave this rank holding only its own heads, so the
   weight gradient is a partial sum. Treating it as plainly replicated is
   silently wrong -- and invisible in the total gradient norm, because those
   gradients are ~1e-5 against a norm of ~1.
3. **torch's own styles unfreeze parameters.** ``ColwiseParallel`` rebuilds each
   parameter as ``nn.Parameter(distribute_tensor(...))`` without forwarding
   ``requires_grad``, which defaults to True. On a LoRA run that silently turns
   every sharded ``base_layer`` back into a trainable weight.
"""

from __future__ import annotations

import inspect
import logging
import types
from typing import Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
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
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    parallelize_module,
)

logger = logging.getLogger(__name__)

# Styles that transform activations without distributing any parameter.
_ACTIVATION_ONLY_STYLES: tuple = (
    PrepareModuleInput,
    PrepareModuleOutput,
    PrepareModuleInputOutput,
)

__all__ = [
    "ColumnParallelRegion",
    "ComposedParallelStyle",
    "ReplicateParallel",
    "SequenceParallelRegion",
    "SharedInputColwiseParallel",
    "SequenceBroadcastOutput",
    "SequenceBroadcastParameter",
    "ShardedAcrossHeadRMSNorm",
    "peel_peft_wrapper",
    "ShardedInputReplicatedWeight",
    "apply_tensor_parallel",
    "ConditionalEntry",
    "build_total_plan",
    "resolve_parallel_plan",
    "localize_attention_head_counts",
]


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
    """Distribute a module's own parameters, and let ``forward`` see plain tensors.

    ``forward`` must not meet those DTensors, so a pre-hook writes a local view
    of each into the module's instance dictionary, where attribute lookup finds
    it before ``nn.Module.__getattr__`` reaches ``_parameters``; a post-hook
    removes it again. The parameter itself is untouched, so
    ``named_parameters`` and ``state_dict`` keep reporting the DTensor under its
    original name -- which is why this is not ``torch.nn.utils.parametrize``,
    whose rewrite to ``parametrizations.<name>.original`` would make a
    checkpoint saved under tensor parallelism unloadable without it.

    Subclasses supply :meth:`_local_view`, and that one expression is their
    whole difference: a replicated parameter wants ``to_local()``, while one
    applied to a shard -- of the heads or of the tokens -- needs its gradient
    declared ``Partial``. Sharing the rest is what gives the two lifetime
    invariants below one implementation to stay true of instead of several
    kept in step by hand.

    **The shadow must not outlive the forward.** FSDP2 and the checkpoint APIs
    read parameters through ``getattr``, and a leftover plain tensor there makes
    them fail on a DTensor-only attribute -- observed as ``'Tensor' object has
    no attribute '_local_tensor'`` during a composed (dp, tp) save. The drop
    hook is registered ``always_call=True`` because PyTorch skips an ordinary
    forward hook when the forward raises, and this framework catches
    forward-path exceptions in several places -- including the ``ValueError``
    that :func:`_guard_sequence_divisibility` raises from an inner block's
    pre-hook, by which point the root and the outer blocks have already
    installed their shadows. A leftover plain tensor then poisons every later
    ``getattr`` and masks the original error with that ``_local_tensor``
    failure.

    **The window between the hooks is a trap**, measured rather than predicted.
    A module that reads its *own* parameter inside a function it hands to
    ``torch.utils.checkpoint`` is unsafe: the recompute calls that function
    directly, so this module's pre-hook does not fire again and the shadow is
    already gone. It raises on the first backward -- "aten.mm got mixed
    torch.Tensor and DTensor" -- so it cannot become a wrong gradient, and it
    only shows up when the input requires grad. Whole-block checkpointing is
    safe for the opposite reason: it re-enters ``forward``, so every inner
    pre-hook fires. H3's three chunked wrappers each clear it, but each for a
    different reason, and ``_ChunkedRMSNorm`` clears it only because it does not
    checkpoint -- it does read ``self.weight`` in its chunk body, so adding a
    checkpoint there, which its sibling already has, would make it fail. See
    ``.scratch/tp_probe/probe_chunked_recompute.py``.

    The hooks walk the parameter names captured at application time rather than
    re-deriving them: ``named_parameters`` goes through ``_named_members``,
    which allocates a dedup set and formats every name string on each call, and
    the dict scan it replaced copied the whole instance ``__dict__`` to delete
    the one or two entries it already knew. Measured at 2.68 -> 0.37 us per
    module per forward, over roughly 200 styled modules in a LoRA run, with the
    denoiser forward running once per diffusion step in rollout and twice per
    training micro-batch under activation checkpointing. Nothing large is
    captured: the closures reference the name tuple, the mesh and a bound
    method, never the module, so they add no reference cycle.
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
    """Hold a module's own parameters replicated, and compute on local views.

    This is the completion style for every module the shard plan does not name.
    The parameters become ``Replicate`` DTensors, which is what the plan has to
    be total for: gradient clipping needs every gradient on one mesh.

    ``forward`` must not see those DTensors, so a pre-hook writes each one's
    ``to_local()`` view into the module's instance dictionary, where attribute
    lookup finds it before ``nn.Module.__getattr__`` reaches ``_parameters``.
    The parameter itself is untouched, so ``named_parameters`` and ``state_dict``
    keep reporting the DTensor under its original name -- which is why this is
    not ``torch.nn.utils.parametrize``, whose rewrite to
    ``parametrizations.<name>.original`` would make a checkpoint saved under
    tensor parallelism unloadable without it.

    Computing locally rather than lifting the activations to DTensors, as this
    style used to, is not a simplification but two fixes:

    * **A module may own bare parameters and children.** Wan and LTX-2 keep
      their AdaLN tables as bare ``nn.Parameter`` on the transformer block and
      on the root. Lifting such a module's inputs made its residual mix a
      DTensor with the plain output of an inner replicated projection, which
      ``aten.mul`` rejects.
    * **Convolution weight gradients were scaled by ``tp_size``.**
      ``convolution_backward``'s sharding rule marks ``grad_weight`` ``Partial``
      unconditionally -- correct for a batch-sharded input, wrong for a
      replicated one, where every rank already computed the whole gradient.
      Measured as exactly 2.000000 on two ranks. ``to_local()``'s own backward
      rewraps the gradient with the forward placements, so it arrives
      ``Replicate`` and nothing reduces it. SD3.5's patch embedder and Wan's are
      the convolutions this affects.

    For a module whose input is a *shard*, use
    :class:`ShardedInputReplicatedWeight`; this style computes its weight
    gradient wrong.
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

    The extra traffic is one ``[batch, seq, 1]`` all-reduce per norm --
    ``1 / normalized_size`` of what a row-parallel output projection already
    costs -- so it does not change the communication picture.

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
            # Every rank consumes the summed statistic, so its backward is a
            # fan-in over the group -- an all-reduce again, not Megatron's
            # identity `g`. Measured against `nn.RMSNorm`, the identity keeps only
            # this rank's share and redistributing a `Partial` term to
            # `Replicate` rescales it; both leave the forward and the weight
            # gradient right and put the input gradient 32% and 98% off.
            square_sum = differentiable_all_reduce(_across_head_square_sum(local), group=tp_group)
            return _across_head_normalize(local, square_sum, normalized_size, epsilon, self.weight)

        module.forward = types.MethodType(forward, module)
        return module


def _across_head_square_sum(local: torch.Tensor) -> torch.Tensor:
    """This rank's share of the sum of squares over the normalized axis.

    ``vector_norm`` with ``dtype=`` accumulates in fp32 inside one fused
    reduction, where ``local.float().pow(2).sum(-1)`` materialized two full fp32
    copies of the activation -- 640 MiB above a 160 MiB bf16 input, against
    0.2 MiB. Squaring in bf16 would be cheaper still but moves the result by
    1.4e-4 relative, against 8.4e-7 here, which is the order of this framework's
    gradient-parity tolerances. ``vector_norm`` refuses to narrow, so the dtype
    is only ever widened.
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


class SequenceBroadcastParameter(_LocalViewStyle):
    """Replicate a module's *own* parameters and sum the gradient they accumulate.

    The style for every parameter inside the sequence-parallel region: a
    modulation table held bare on a block, an affine norm weight, the output
    projection. All of them are applied to each token independently, so under a
    sharded token axis each rank computes with its own slice and the gradient it
    accumulates is a partial sum. The forward needs nothing -- a replicated
    weight on a token slice is already right -- so the parameter is distributed
    ``Replicate``, which keeps clipping and the optimizer on one mesh, and
    ``forward`` is handed a local view whose backward declares the gradient
    ``Partial``. Lifting the activations to DTensors instead cannot reach either
    case: a bare ``nn.Parameter`` is read inside the block's own ``forward``,
    where there is no module boundary to hook, and a ``Linear`` fed a DTensor
    sharded on an interior axis fails outright -- ``F.linear`` flattens batch
    into sequence, and "Attempted to flatten multiple dimensions, with dimension
    1 being sharded" is where that ends.

    Getting this wrong is silent in the worst way: the loss is **bit-identical**,
    because the forward genuinely is correct on every rank, and only the
    parameter's gradient comes out at this rank's share of the tokens. Measured
    at exactly 1/2 on two ranks before the fix.

    Only the module's direct parameters are touched, so a block carrying both a
    modulation table and sharded projections keeps both treatments. ``exclude``
    names direct parameters that modulate a stream the overlay leaves whole --
    LTX-2.3's prompt tables act on the replicated text stream -- so every rank
    already holds their full gradient and summing it would multiply it by the
    group size; they take :class:`ReplicateParallel`'s view instead.
    """

    def __init__(self, *, exclude: Tuple[str, ...] = ()) -> None:
        super().__init__()
        self.exclude = frozenset(exclude)

    def _local_view_for(self, name: str):
        return _replicated_view if name in self.exclude else self._local_view

    def _local_view(self, parameter: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
        """A plain view whose backward declares the gradient a partial sum.

        ``device_mesh`` is the style's own application mesh, not one read off
        ``parameter`` inside the hook: by the time the forward runs,
        ``fully_shard`` has moved the parameter onto the concatenated
        ``(dp_shard, tp)`` mesh, and a one-element placement list there lands the
        ``Partial`` on ``dp_shard``.
        """
        return _ToLocalWithPartialGradient.apply(parameter, device_mesh)


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


def _argument_positions(module: nn.Module, names: Sequence[str]) -> Dict[str, Optional[int]]:
    """Map each named argument of ``module.forward`` to its position, if it has one.

    diffusers calls the same module by keyword in one place and positionally in
    another, so an entry is declared by name and found either way.

    Raises:
        ValueError: If ``forward`` takes no argument by that name and accepts no
            ``**kwargs`` that could carry it.
    """
    parameters = list(inspect.signature(module.forward).parameters.values())
    positional = [
        p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    keyword = {p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    variadic = any(p.kind is p.VAR_KEYWORD for p in parameters)
    positions: Dict[str, Optional[int]] = {}
    for name in names:
        if name in positional:
            positions[name] = positional.index(name)
        elif name in keyword or variadic:
            positions[name] = None
        else:
            raise ValueError(
                f"{type(module).__name__}.forward takes no argument named {name!r}, so a "
                f"column-parallel region entry cannot lift it; expected one of "
                f"{[p.name for p in parameters]}"
            )
    return positions


class _RegionEntry(ParallelStyle):
    """Lift the named inputs of a module whose column-parallel consumers share them.

    Subclasses choose the lift per input and per call. The inputs are found by
    keyword or by position; one that is absent or ``None`` passes through, as
    Wan's self-attention passes ``encoder_hidden_states=None`` and reads
    ``hidden_states`` for its keys.
    """

    def _names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def _lift(
        self, name: str, value: torch.Tensor, device_mesh: DeviceMesh, call: Mapping[str, object]
    ) -> torch.Tensor:
        raise NotImplementedError

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        positions = _argument_positions(module, self._names())
        owner = type(module).__name__

        def lift(name: str, value: object, call: Mapping[str, object]) -> object:
            if value is None:
                return None
            if isinstance(value, DTensor) or not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"expected a plain tensor or None for {owner}'s {name!r} at a "
                    f"column-parallel region entry, received {type(value).__name__}"
                )
            return self._lift(name, value, device_mesh, call)

        def enter(_module: nn.Module, args: tuple, kwargs: dict):
            lifted_args, lifted_kwargs = list(args), dict(kwargs)
            for name, position in positions.items():
                if name in kwargs:
                    lifted_kwargs[name] = lift(name, kwargs[name], kwargs)
                elif position is not None and position < len(args):
                    lifted_args[position] = lift(name, args[position], kwargs)
            return tuple(lifted_args), lifted_kwargs

        module.register_forward_pre_hook(enter, with_kwargs=True)
        return module


def _validated_names(names: object, field: str) -> Tuple[str, ...]:
    if not isinstance(names, tuple) or not all(isinstance(n, str) and n for n in names):
        raise TypeError(f"expected a tuple of argument names for {field}, received {names!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"expected distinct argument names for {field}, received {names!r}")
    return names


class ColumnParallelRegion(_RegionEntry):
    """Open a column-parallel region under tensor parallelism alone.

    Every named input is whole on every rank and passes through Megatron's
    ``f``: identity forward, one all-reduce of its gradient backward. Each
    :class:`SharedInputColwiseParallel` consumer inside leaves its share of
    that gradient unreduced, and autograd sums the shares before they reach
    ``f`` -- one collective per input stream instead of one per projection.
    """

    def __init__(self, *, inputs: Tuple[str, ...]) -> None:
        super().__init__()
        self.inputs = _validated_names(inputs, "inputs")
        if not self.inputs:
            raise ValueError("ColumnParallelRegion needs at least one input to lift")

    def _names(self) -> Tuple[str, ...]:
        return self.inputs

    def _lift(
        self, name: str, value: torch.Tensor, device_mesh: DeviceMesh, call: Mapping[str, object]
    ) -> torch.Tensor:
        return _RegionAllReduceGradient.apply(value, device_mesh)


class SequenceParallelRegion(_RegionEntry):
    """Open a column-parallel region whose token-sharded inputs are gathered.

    ``gather`` names the inputs that arrive sharded on ``token_dim``: they are
    all-gathered so attention sees every token, and their gradient comes back
    through one reduce-scatter. ``replicated`` names inputs the overlay leaves
    whole -- a text stream -- which take :class:`ColumnParallelRegion`'s ``f``.
    This replaces ``PrepareModuleInput(Shard -> Replicate, use_local_output=True)``,
    whose forward is the same all-gather but whose local slice left three
    full-sequence all-reduces to the q/k/v consumers.

    ``gather_when``, given the call's keyword arguments, says whether this call
    arrives sharded at all. A module shared by a sharded and an unsharded pass
    -- SenseNova's attention serves both its generation and its understanding
    branch -- gathers on the first and lifts like :class:`ColumnParallelRegion`
    on the second.
    """

    def __init__(
        self,
        *,
        gather: Tuple[str, ...],
        replicated: Tuple[str, ...] = (),
        token_dim: int = 1,
        gather_when: Optional[Callable[[Mapping[str, object]], bool]] = None,
    ) -> None:
        super().__init__()
        self.gather = _validated_names(gather, "gather")
        self.replicated = _validated_names(replicated, "replicated")
        if not self.gather:
            raise ValueError("SequenceParallelRegion needs at least one input to gather")
        overlap = set(self.gather) & set(self.replicated)
        if overlap:
            raise ValueError(
                f"an input cannot be both gathered and replicated, received {sorted(overlap)}"
            )
        if isinstance(token_dim, bool) or not isinstance(token_dim, int) or token_dim < 1:
            raise ValueError(f"expected a positive int token_dim, received {token_dim!r}")
        if gather_when is not None and not callable(gather_when):
            raise TypeError(f"expected a callable gather_when, received {gather_when!r}")
        self.token_dim = token_dim
        self.gather_when = gather_when

    def _names(self) -> Tuple[str, ...]:
        return self.gather + self.replicated

    def _lift(
        self, name: str, value: torch.Tensor, device_mesh: DeviceMesh, call: Mapping[str, object]
    ) -> torch.Tensor:
        if name in self.gather and (self.gather_when is None or self.gather_when(call)):
            return _GatherTokens.apply(value, device_mesh, self.token_dim)
        return _RegionAllReduceGradient.apply(value, device_mesh)


class SequenceBroadcastOutput(ParallelStyle):
    """Sum the gradient of outputs that broadcast over a sharded sequence.

    The counterpart of :class:`SequenceBroadcastParameter` one level upstream.
    A diffusion transformer computes its conditioning once -- a timestep
    embedding, a modulation vector -- and broadcasts it over every token. Under
    sequence parallelism each rank multiplies that value by its own slice, so
    the gradient arriving back at the module that produced it is a partial sum
    over this rank's tokens, and nothing in the tensor-parallel plan reduces it:
    the value itself never was sharded, so no placement records the debt.

    Left unfixed this is silent in the same way the parameter case is. The
    forward is correct on every rank and the loss is bit-identical; only the
    conditioning stack trains at a fraction of its true gradient -- measured at
    73% low for Wan's time embedder on two ranks.

    ``output_indices`` selects which of a tuple-returning module's outputs are
    broadcast. Wan's condition embedder returns four, of which the two timestep
    tensors are; its text states flow into cross-attention, where the ordinary
    column-parallel input reduction already covers them.
    """

    def __init__(self, *, output_indices: tuple = (0,)) -> None:
        super().__init__()
        self.output_indices = output_indices

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        selected = set(self.output_indices)

        def reduce_gradients(_module: nn.Module, _inputs, outputs):
            if isinstance(outputs, torch.Tensor):
                return _AllReduceGradient.apply(outputs, device_mesh)
            if not isinstance(outputs, (tuple, list)):
                return outputs
            rebuilt = [
                (
                    _AllReduceGradient.apply(value, device_mesh)
                    if index in selected and isinstance(value, torch.Tensor)
                    else value
                )
                for index, value in enumerate(outputs)
            ]
            return type(outputs)(rebuilt) if isinstance(outputs, list) else tuple(rebuilt)

        module.register_forward_hook(reduce_gradients)
        return module


class SequenceAlignedOutput(ParallelStyle):
    """Align conditioning outputs with the sharded sequence, whichever layout they have.

    One family can embed one timestep per sample in one checkpoint and one per
    token in another -- Wan2.2-TI2V-5B sets ``expand_timesteps``, LTX-2 I2AV gives
    its conditioning frames their own timestep -- while the plan is chosen
    before any config is read. An output of rank ``broadcast_ndim`` is one value
    per sample, broadcast over the tokens, and gets
    :class:`SequenceBroadcastOutput`'s gradient sum. An output with one more axis
    carries a token axis at ``token_dim`` and is narrowed to this rank's tokens,
    like the hidden states it modulates; the narrowing is a redistribute whose
    backward gathers the gradient, so summing it as well would double it. The
    layout is read per call, from the output itself.
    """

    def __init__(
        self,
        *,
        output_indices: tuple,
        broadcast_ndim: int,
        token_dim: int = 1,
        batch_size_kwarg: Optional[str] = None,
    ) -> None:
        super().__init__()
        if broadcast_ndim < 1 or not 0 < token_dim <= broadcast_ndim:
            raise ValueError(
                "expected broadcast_ndim >= 1 and a token_dim in [1, broadcast_ndim], received "
                f"broadcast_ndim={broadcast_ndim}, token_dim={token_dim}"
            )
        self.output_indices = output_indices
        self.broadcast_ndim = broadcast_ndim
        self.token_dim = token_dim
        self.batch_size_kwarg = batch_size_kwarg

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Install the hook.

        With ``batch_size_kwarg`` the module embeds a flattened ``[B * S]``
        input -- LTX-2's ``time_embed(timestep.flatten(), batch_size=B)`` -- so
        both layouts have the same rank and the row count against the batch
        size passed by that keyword tells them apart.
        """
        selected = set(self.output_indices)
        owner = type(module).__name__

        def narrow(value: torch.Tensor) -> torch.Tensor:
            whole = DTensor.from_local(value, device_mesh, [Replicate()], run_check=False)
            return whole.redistribute(device_mesh, [Shard(self.token_dim)]).to_local()

        def align(index: int, value: torch.Tensor, batch_size: Optional[int]) -> torch.Tensor:
            if batch_size is not None:
                rows = value.shape[0]
                if rows == batch_size:
                    return _AllReduceGradient.apply(value, device_mesh)
                if rows % batch_size:
                    raise ValueError(
                        f"expected output {index} of {owner} to have a multiple of "
                        f"batch_size={batch_size} rows, received shape {tuple(value.shape)}"
                    )
                tokens = value.reshape(batch_size, rows // batch_size, *value.shape[1:])
                return narrow(tokens).reshape(-1, *value.shape[1:])
            if value.ndim == self.broadcast_ndim:
                return _AllReduceGradient.apply(value, device_mesh)
            if value.ndim == self.broadcast_ndim + 1:
                return narrow(value)
            raise ValueError(
                f"expected output {index} of {owner} to have rank {self.broadcast_ndim} (one value "
                f"per sample) or {self.broadcast_ndim + 1} (one per token), received shape "
                f"{tuple(value.shape)}"
            )

        def align_outputs(_module: nn.Module, _args, kwargs, outputs):
            batch_size = None
            if self.batch_size_kwarg is not None:
                batch_size = kwargs.get(self.batch_size_kwarg)
                if (
                    isinstance(batch_size, bool)
                    or not isinstance(batch_size, int)
                    or batch_size < 1
                ):
                    raise TypeError(
                        f"expected {owner} to be called with a positive int "
                        f"{self.batch_size_kwarg}=..., received {batch_size!r}"
                    )
            if isinstance(outputs, torch.Tensor):
                return align(0, outputs, batch_size) if 0 in selected else outputs
            if not isinstance(outputs, (tuple, list)):
                raise TypeError(
                    f"expected {owner} to return a tensor, tuple or list, received "
                    f"{type(outputs).__name__}"
                )
            rebuilt = [
                (
                    align(index, value, batch_size)
                    if index in selected and isinstance(value, torch.Tensor)
                    else value
                )
                for index, value in enumerate(outputs)
            ]
            return type(outputs)(rebuilt) if isinstance(outputs, list) else tuple(rebuilt)

        module.register_forward_hook(align_outputs, with_kwargs=True)
        return module


class SequenceShardedArgument(ParallelStyle):
    """Narrow one argument of a module's call to this rank's tokens.

    For a per-token side input the transformer computes at full length and hands
    every block -- Qwen-Image-Edit-2511's ``modulate_index`` under
    ``zero_cond_t`` -- while the overlay has already narrowed the hidden states
    it is applied to. The argument is found by keyword or by position, because
    diffusers calls a block by keyword in its eager loop and positionally
    through ``_gradient_checkpointing_func``; ``None`` passes through. The slice
    is the redistribute that narrows the hidden states, so both land on the same
    token range.
    """

    def __init__(self, *, name: str, position: int, token_dim: int = 1) -> None:
        super().__init__()
        if position < 0 or token_dim < 1:
            raise ValueError(
                f"expected a non-negative position and a positive token_dim for {name!r}, "
                f"received position={position}, token_dim={token_dim}"
            )
        self.name = name
        self.position = position
        self.token_dim = token_dim

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        owner = type(module).__name__

        def narrow(value: object) -> object:
            if value is None:
                return None
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"expected a tensor or None for {owner}'s {self.name!r}, received "
                    f"{type(value).__name__}"
                )
            whole = DTensor.from_local(value, device_mesh, [Replicate()], run_check=False)
            return whole.redistribute(device_mesh, [Shard(self.token_dim)]).to_local()

        def narrow_argument(_module: nn.Module, args: tuple, kwargs: dict):
            if self.name in kwargs:
                return args, {**kwargs, self.name: narrow(kwargs[self.name])}
            if len(args) > self.position:
                narrowed = narrow(args[self.position])
                return (*args[: self.position], narrowed, *args[self.position + 1 :]), kwargs
            return None

        module.register_forward_pre_hook(narrow_argument, with_kwargs=True)
        return module


class ComposedParallelStyle(ParallelStyle):
    """Apply several styles to one module, in order.

    ``parallelize_module`` takes one style per path, but a module can need two
    unrelated treatments. ``PrepareModuleInput`` and ``PrepareModuleOutput``
    transform activations and distribute nothing, so a module that owns
    parameters *and* sits on a sharding boundary -- which every transformer
    block holding a modulation table does -- would otherwise have those
    parameters left undistributed, and ``apply_tensor_parallel`` rejects the
    plan. Each style installs its own hooks and none reads another's, so the
    order between them does not matter.
    """

    def __init__(self, *styles: ParallelStyle) -> None:
        super().__init__()
        if not styles:
            raise ValueError("ComposedParallelStyle needs at least one style")
        self.styles = styles

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        for style in self.styles:
            module = style._apply(module, device_mesh)
        return module


def _modules_owning_parameters(root: nn.Module) -> Iterable[str]:
    """Yield the name of every module that directly owns a parameter."""
    for module_name, module in root.named_modules():
        if any(True for _ in module.named_parameters(recurse=False)):
            yield module_name


class ConditionalEntry(NamedTuple):
    """A plan entry whose modules only some checkpoints of a family have.

    One adapter serves a whole family, and the plan is a ``classmethod`` chosen
    before any weight is read, so it cannot branch on the loaded config. Wan's
    I2V adapter covers both Wan2.1-I2V, whose attention carries a second
    key/value stream for the CLIP image conditioning
    (``added_kv_proj_dim=5120``), and Wan2.2-I2V, which has none
    (``added_kv_proj_dim=None``) -- naming those three modules unconditionally
    makes the glob expansion reject the plan *after* a 14B load.

    Entries sharing a ``group`` are one condition and must agree: either every
    pattern in the group matches or none does. That keeps a typo loud, which is
    the whole point of rejecting a pattern that matches nothing.
    """

    style: ParallelStyle
    group: str


def _adapter_name(adapter: object) -> str:
    return getattr(adapter, "__name__", None) or type(adapter).__name__


def sequence_parallel_enabled(adapter: object, sequence_parallel: Optional[bool]) -> bool:
    """Resolve the configured ``parallelism.sequence_parallel`` against an adapter.

    ``True`` and ``False`` decide outright. Unset defers to the adapter's
    ``sequence_parallel_by_default``, which an adapter turns on only once its
    overlay has been verified end to end on real weights: where the sharded
    sequence carries a prompt-dependent length, or a per-token tensor the
    overlay has not been checked against, a default-on overlay turns a working
    tensor-parallel run into one that fails after the load.

    Args:
        adapter: The adapter class or instance.
        sequence_parallel: The configured tri-state.

    Returns:
        Whether the adapter's sequence-parallel overlay applies.
    """
    if sequence_parallel is not None:
        if not isinstance(sequence_parallel, bool):
            raise TypeError(
                "expected parallelism.sequence_parallel to be a bool or None, received "
                f"{type(sequence_parallel).__name__}: {sequence_parallel!r}"
            )
        return sequence_parallel
    by_default = getattr(adapter, "sequence_parallel_by_default", None)
    if not isinstance(by_default, bool):
        raise TypeError(
            f"expected {_adapter_name(adapter)}.sequence_parallel_by_default to be a bool, "
            f"received {type(by_default).__name__}: {by_default!r}"
        )
    return by_default


def resolve_parallel_plan(
    adapter: object, *, sequence_parallel: Optional[bool]
) -> Dict[str, ParallelStyle]:
    """Merge an adapter's tensor- and sequence-parallel plans under one policy.

    One implementation for two callers that need different things from the same
    rule: the pre-load validator calls it for its *effect*, to reject a bad
    combination before a 14B download, and the applier calls it for its *value*.

    Override semantics, not union: sequence parallelism *refines* the
    tensor-parallel plan, so an entry here replaces the one it restates -- an
    out-projection that reduce-scatters instead of all-reducing is the same
    module, not a second one.

    Whether the overlay applies is :func:`sequence_parallel_enabled`. Asking for
    it by name on an adapter without one is refused, and so is an adapter that
    enables it by default without declaring one, which is the adapter's bug.

    Args:
        adapter: The adapter class or instance owning the plans.
        sequence_parallel: The configured ``parallelism.sequence_parallel``.

    Returns:
        The merged plan.

    Raises:
        ValueError: If the overlay applies and the adapter declares none.
    """
    plan = dict(adapter.tensor_parallel_plan())
    name = _adapter_name(adapter)
    overlay = adapter.sequence_parallel_plan()
    if not sequence_parallel_enabled(adapter, sequence_parallel):
        if sequence_parallel is None and overlay:
            logger.info(
                "%s declares a sequence-parallel plan but does not enable it by default; "
                "set parallelism.sequence_parallel: true to shard the token axis as well.",
                name,
            )
        return plan

    if not overlay:
        if sequence_parallel:
            raise ValueError(
                f"parallelism.sequence_parallel=true was requested but {name} declares no "
                "sequence-parallel plan. Sharding the token axis needs per-architecture "
                "entries -- which norms live in the sharded region, where attention gathers "
                "back, and which parameters broadcast over tokens -- so there is no safe "
                "default. Unset the flag to run with tensor parallelism alone."
            )
        raise ValueError(
            f"{name}.sequence_parallel_by_default is True but {name} declares no "
            "sequence-parallel plan; declare one or turn the default off."
        )
    plan.update(overlay)
    logger.info(
        "Sequence parallelism overlays %d entries on the tensor-parallel plan", len(overlay)
    )
    return plan


def _resolve_patterns(
    root: nn.Module, styles: Mapping[str, ParallelStyle]
) -> Tuple[Dict[str, ParallelStyle], Dict[str, bool]]:
    """Resolve ``*`` wildcards in plan keys against the live module tree.

    Plans are written as ``transformer_blocks.*.attn.to_q``, mirroring the
    notation diffusers uses; ``parallelize_module`` wants concrete paths.

    Matching runs over every module, not only parameter-owning ones: once PEFT
    has injected LoRA, a planned ``Linear`` becomes a ``lora.Linear`` wrapper
    that owns no parameter directly, so restricting the search to parameter
    owners would make a valid plan look like it matched nothing.

    An exact path overrides a wildcard that also reaches it, whatever order the
    plan listed them in. Sequence parallelism needs that: every block takes the
    same treatment through ``blocks.*`` while the first one additionally
    scatters the token axis, and leaving the winner to dict ordering would make
    a correct plan depend on how it was written.

    Knows nothing about which entries are allowed to match nothing -- it reports
    what each pattern found and lets :func:`_reject_unmatched` decide. That
    split is why neither half has to be read to change the other.

    Returns:
        The resolved path-to-style mapping, and whether each original pattern
        matched at least one module.
    """
    # `named_modules()` names the root `""`, which is dropped here so a wildcard
    # cannot match it -- but `""` is a legitimate exact key: Wan and LTX-2 keep
    # an AdaLN table as a bare parameter on the transformer root, and
    # `build_total_plan` completes it under that name.
    candidates = [name for name, _ in root.named_modules() if name]
    reachable = set(candidates) | {""}
    exact = {path: style for path, style in styles.items() if "*" not in path}
    resolved: Dict[str, ParallelStyle] = {}
    matched: Dict[str, bool] = {}

    for pattern, style in styles.items():
        if "*" not in pattern:
            matched[pattern] = pattern in reachable
            continue
        prefix, _, suffix = pattern.partition("*")
        suffix = suffix.lstrip(".")
        hits = [
            name
            for name in candidates
            if name.startswith(prefix) and name.endswith(suffix)
            # `*` stands for exactly one path segment.
            and (middle := name[len(prefix) : len(name) - len(suffix)].strip("."))
            and "." not in middle
        ]
        matched[pattern] = bool(hits)
        for name in hits:
            # An exact entry for the same path wins, but the wildcard still
            # counts as matched so it is not reported unused.
            if name not in exact:
                resolved[name] = style

    resolved.update(exact)
    return resolved, matched


def _reject_unmatched(
    plan: Mapping[str, ParallelStyle], matched: Mapping[str, bool]
) -> Iterable[str]:
    """Rule on the patterns that found nothing, and yield the paths to drop.

    A pattern matching no module normally means the plan names a layer this
    architecture does not have -- a typo, or a renamed layer -- and is an error.
    A :class:`ConditionalEntry` is the exception: its modules exist only in some
    checkpoints of the family its adapter serves. Its ``group`` carries the
    condition, and the members of a group must agree, so a misspelling inside an
    otherwise-present group is still rejected.

    This holds exact paths to the same standard as wildcards, where the form
    this replaced checked only wildcards. All 16 registered adapters pass
    either way, so the stricter rule costs nothing and closes the gap a
    misspelled exact entry went through -- it previously reached
    ``parallelize_module``, which drops an unmatched path silently, and was
    caught only if the module it should have named owned a parameter that the
    totality assertion then reported as undistributed.

    Raises:
        ValueError: If an unconditional pattern matched nothing, or one
            conditional group matched only partly.
    """
    groups: Dict[str, Dict[str, bool]] = {}
    for path, entry in plan.items():
        if isinstance(entry, ConditionalEntry):
            groups.setdefault(entry.group, {})[path] = matched[path]
        elif not matched[path]:
            raise ValueError(
                f"tensor-parallel plan pattern {path!r} matched no module. The plan "
                f"must name modules this architecture has, or the layer it was written "
                f"for has been renamed."
            )

    for group, outcomes in groups.items():
        if all(outcomes.values()):
            continue
        if any(outcomes.values()):
            # Mixed means a typo, not a different checkpoint: the condition the
            # group names holds for the architecture, so every member of it
            # should have been found.
            raise ValueError(
                f"conditional tensor-parallel group {group!r} matched only partly, which "
                f"means one of its paths is misspelled rather than absent from this "
                f"checkpoint: "
                + ", ".join(
                    f"{path!r}={'matched' if hit else 'missing'}" for path, hit in outcomes.items()
                )
            )
        logger.info(
            "Conditional tensor-parallel group %r names no module in this checkpoint and "
            "was skipped: %s",
            group,
            ", ".join(repr(path) for path in outcomes),
        )
        yield from outcomes


def _expand_globs(root: nn.Module, plan: Mapping[str, ParallelStyle]) -> Dict[str, ParallelStyle]:
    """Turn a written plan into concrete paths, rejecting the ones that name nothing.

    Composition of :func:`_resolve_patterns`, which does not know that an entry
    may be allowed to match nothing, and :func:`_reject_unmatched`, which does
    not know how a pattern is matched.
    """
    styles = {
        path: entry.style if isinstance(entry, ConditionalEntry) else entry
        for path, entry in plan.items()
    }
    resolved, matched = _resolve_patterns(root, styles)
    for path in _reject_unmatched(plan, matched):
        resolved.pop(path, None)
    return resolved


def _lora_adapter_names(module: nn.Module) -> List[str]:
    """Return the adapter names PEFT injected into this module, if any."""
    lora_a = getattr(module, "lora_A", None)
    if lora_a is None or not hasattr(lora_a, "keys"):
        return []
    return [name for name in lora_a.keys() if any(True for _ in lora_a[name].parameters())]


def derive_lora_plan(
    root: nn.Module, resolved_plan: Mapping[str, ParallelStyle]
) -> Dict[str, ParallelStyle]:
    """Rewrite a plan so it reaches through PEFT's LoRA wrappers.

    ``ColwiseParallel`` accepts only ``nn.Linear`` and ``nn.Embedding``, so an
    entry naming a module PEFT has replaced with ``lora.Linear`` has to target
    its ``base_layer`` instead. The adapter matrices then need styles chosen so
    the LoRA branch lands in the same placement as the base branch it is added
    to:

    ``colwise`` base (weight sharded on out_features, output ``Shard(-1)``):
        ``lora_A`` sees the full input and emits the small rank dimension, so it
        is replicated; ``lora_B`` maps rank -> out_features and is sharded like
        the base, so both branches end with the same local width.

    ``rowwise`` base (weight sharded on in_features, output all-reduced):
        the incoming activation is already a shard, so ``lora_A`` must be
        sharded on in_features too -- replicating it would shape-mismatch
        against the local activation width. Its output is reduced back to the
        full rank dimension and ``lora_B`` is replicated.

    ``lora_A`` copies the base style's ``input_layouts`` rather than taking the
    ``RowwiseParallel`` default. The two coincide for a row-parallel layer fed a
    shard, which is every style shipped today, but a row-parallel layer fed a
    *replicated* input is a legitimate way to shard a weight packed on its
    output axis -- and there the default ``Shard(-1)`` would declare a full-width
    activation to be a local shard, making the global contraction ``tp_size``
    times too wide. That surfaces as "a and b must have same reduction dim"
    inside the adapter, far from the plan that caused it.
    """
    derived: Dict[str, ParallelStyle] = {}
    for path, style in resolved_plan.items():
        module = root.get_submodule(path)
        adapters = _lora_adapter_names(module)
        if not adapters:
            derived[path] = style
            continue
        derived[f"{path}.base_layer"] = style
        for adapter in adapters:
            if isinstance(style, RowwiseParallel):
                # `input_layouts` and `output_layouts` are stored as 1-tuples by
                # `__init__`, which would be wrapped again if handed back whole.
                inputs = getattr(style, "input_layouts", None)
                outputs = getattr(style, "output_layouts", None)
                derived[f"{path}.lora_A.{adapter}"] = RowwiseParallel(
                    input_layouts=inputs[0] if inputs else None,
                    output_layouts=outputs[0] if outputs else None,
                )
                # Under sequence parallelism the base branch reduce-scatters its
                # output onto the token axis, so the adapter branch has to land
                # there too -- otherwise the two are added at different lengths
                # ("size of tensor a (160) must match tensor b (320)"). Carrying
                # the layout through `lora_A` does that, and then `lora_B` is
                # applied per token on a slice, so its gradient is a partial sum.
                token_sharded = bool(outputs) and any(isinstance(p, Shard) for p in outputs)
                derived[f"{path}.lora_B.{adapter}"] = (
                    SequenceBroadcastParameter() if token_sharded else ReplicateParallel()
                )
            elif isinstance(style, SharedInputColwiseParallel):
                # The base arm leaves its input gradient a partial sum for the
                # region entry to reduce, so the adapter arm has to as well:
                # `lora_B` does not reduce the gradient of its input, which
                # makes `lora_A`'s weight gradient a partial sum too -- the case
                # `SequenceBroadcastParameter`'s view declares.
                derived[f"{path}.lora_A.{adapter}"] = SequenceBroadcastParameter()
                derived[f"{path}.lora_B.{adapter}"] = SharedInputColwiseParallel(verify_input=False)
            else:
                if not isinstance(style, ColwiseParallel):
                    # Only the weight-sharding styles have a known-correct
                    # adapter pairing. Everything else a plan names is either an
                    # activation transform (`PrepareModuleInput/Output`, or a
                    # `ComposedParallelStyle` carrying one) or a sequence-parallel
                    # parameter style, and both go wrong here in ways that are
                    # hard to read at the point of failure:
                    #
                    # * an activation transform relocated to `base_layer` hooks
                    #   the base branch only, so the gather runs on one arm of
                    #   `base_layer(x) + lora_B(lora_A(x))` and the add sees
                    #   `[B, S, out]` against `[B, S/tp, out]`;
                    # * `lora_B -> ColwiseParallel` halves the adapter arm's width
                    #   against a full-width base arm ("the size of tensor a (128)
                    #   must match the size of tensor b (64)");
                    # * and `lora_A -> ReplicateParallel` inside the
                    #   sequence-parallel region accumulates only this rank's
                    #   token share into a replicated weight, which is the silent
                    #   1/tp_size gradient `SequenceBroadcastParameter` exists to
                    #   prevent.
                    #
                    # The adapters document this as a limit; refuse it here so it
                    # names the module instead of surfacing mid-forward.
                    raise ValueError(
                        f"LoRA targets {path!r}, which the tensor-parallel plan covers with "
                        f"{type(style).__name__} -- a style whose adapter pairing is not "
                        f"derivable. Only ColwiseParallel, SharedInputColwiseParallel and "
                        f"RowwiseParallel are: the rest are activation transforms or "
                        f"sequence-parallel parameter styles, "
                        f"which would hook the base branch alone or train the adapter on this "
                        f"rank's token share. Remove {path.rsplit('.', 1)[-1]!r} from "
                        f"model.target_modules; or set parallelism.sequence_parallel: false, "
                        f"which keeps tensor parallelism and drops the sequence-parallel "
                        f"overlay that puts such styles on a projection; or set "
                        f"parallelism.tp_size=1."
                    )
                derived[f"{path}.lora_A.{adapter}"] = ReplicateParallel()
                derived[f"{path}.lora_B.{adapter}"] = ColwiseParallel()
    return derived


def build_total_plan(
    root: nn.Module, shard_plan: Mapping[str, ParallelStyle]
) -> Dict[str, ParallelStyle]:
    """Complete a sharding plan so that it covers every parameter-owning module.

    Args:
        root: The module that will be parallelized.
        shard_plan: Architecture-specific styles, optionally using ``*`` to stand
            for one path segment.

    Returns:
        A plan naming every parameter-owning module in ``root``, with
        :class:`ReplicateParallel` filled in wherever ``shard_plan`` is silent.
    """
    resolved = _expand_globs(root, shard_plan)
    total = derive_lora_plan(root, resolved)
    for module_name in _modules_owning_parameters(root):
        existing = total.get(module_name)
        if existing is None:
            # Including `""`, the root: Wan and LTX-2 keep an AdaLN table there
            # as a bare parameter, and skipping it left that parameter a plain
            # tensor, which `apply_tensor_parallel` then rejects.
            total[module_name] = ReplicateParallel()
        elif isinstance(
            existing, (*_ACTIVATION_ONLY_STYLES, SequenceBroadcastOutput, _RegionEntry)
        ):
            # An activation transform distributes nothing, so a boundary module
            # that owns parameters still needs a parameter style beside it.
            total[module_name] = ComposedParallelStyle(ReplicateParallel(), existing)
    return total


#: Set on an attention module once its head count is localized, which makes
#: :func:`localize_attention_head_counts` idempotent. :func:`apply_tensor_parallel`
#: never localizes a module twice -- ``distribute_module`` refuses a second
#: application first -- but the function is public, and it rewrites a count in
#: place: halving ``heads`` twice mis-splits every projection, silently for a
#: cross-attention with no rotary width to trip on.
_LOCALIZED_HEADS_MARKER = "_ff_tp_heads_localized"


def _owns_a_column_shard(module: nn.Module) -> bool:
    """Report whether one of ``module``'s own projections holds a dim-0 shard.

    Read off the sharded parameters rather than off the plan, so LoRA's extra
    nesting (``to_q.base_layer``, ``to_q.lora_B.default``) needs no special
    case. Within an attention module only q/k/v are column-parallel: the output
    projection is row-parallel (``Shard(1)``) and qk-norm keeps a replicated
    weight. So a dim-0 shard under one of its projections means its heads were
    split, and its absence means they were not.

    Scoped to direct children so a feed-forward sharded elsewhere in the same
    block does not count: ``attn1`` and ``ffn`` are siblings, and a plan that
    shards only the latter must leave the former's head count global.
    """
    return any(
        isinstance(parameter, DTensor)
        and any(isinstance(p, Shard) and p.dim == 0 for p in parameter.placements)
        for child in module.children()
        for parameter in child.parameters()
    )


def localize_attention_head_counts(root: nn.Module, tp_size: int) -> int:
    """Rewrite each attention module's head geometry to the share it now owns.

    Column-parallel q/k/v leave each rank with ``heads / tp_size`` heads. A
    processor that reshapes with ``unflatten(-1, (attn.heads, -1))`` would read
    the global count and mis-split the local projection, so the attribute is set
    to this rank's share -- which after sharding is what it genuinely is.
    Processors written as ``unflatten(-1, (-1, head_dim))`` infer the count and
    are unaffected either way.

    Only attention modules the plan actually column-shards are rewritten. A plan
    that shards the feed-forward and leaves attention replicated -- which is the
    only option for an architecture whose qk-norm spans all heads -- keeps
    full-width projections, so halving the count would split each head into
    ``tp_size`` times too many channels. That fails loudly where a rotary
    embedding checks the head width and **silently** where there is none, which
    is the case for Wan's cross-attention.

    ``inner_dim`` is localized with it, for processors that derive the head width
    as ``attn.inner_dim // attn.heads`` -- diffusers' Qwen-Image processor does,
    and leaving ``inner_dim`` global there makes the derived ``head_dim``
    ``tp_size`` times too large. The two attributes describe the same projection,
    so they are either both global or both local; mixing them is what breaks.
    Behind a per-head qk-norm that is loud rather than silent: the derived width
    no longer matches the norm's ``[head_dim]`` weight, so ``attn.norm_q``
    raises a shape mismatch instead of training on a mis-split tensor.

    Args:
        root: The parallelized module tree.
        tp_size: The tensor-parallel degree.

    Returns:
        The number of attention modules adjusted.

    Raises:
        ValueError: If a column-sharded head count or inner width is not
            divisible by ``tp_size``.
    """

    def _local_share(value: object, module: nn.Module, attribute: str) -> Optional[int]:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return None
        if value % tp_size != 0:
            raise ValueError(
                f"tensor parallelism requires every attention {attribute} to be divisible "
                f"by the tensor-parallel degree, but {type(module).__name__} has "
                f"{attribute}={value} and tp_size={tp_size}."
            )
        return value // tp_size

    adjusted = 0
    for module in root.modules():
        if getattr(module, "heads", None) is None or not _owns_a_column_shard(module):
            continue
        if getattr(module, _LOCALIZED_HEADS_MARKER, False):
            continue
        heads = _local_share(module.heads, module, "head count")
        if heads is None:
            continue
        module.heads = heads
        inner_dim = _local_share(getattr(module, "inner_dim", None), module, "inner width")
        if inner_dim is not None:
            module.inner_dim = inner_dim
        setattr(module, _LOCALIZED_HEADS_MARKER, True)
        adjusted += 1
    return adjusted


def peel_peft_wrapper(module: nn.Module) -> nn.Module:
    """Return the base transformer under a PEFT wrapper, else ``module``.

    A plan names paths in the architecture's own tree
    (``transformer_blocks.*.attn.to_q``). Once PEFT has injected LoRA the
    component is a ``PeftModel`` whose children sit under
    ``base_model.model.``, so every pattern would match nothing. The base model
    keeps its injected ``lora.Linear`` submodules -- they are what
    :func:`derive_lora_plan` then targets -- while being the tree the plan was
    written against.

    ``CompileAccelerator._peel_peft`` does the same thing for the same reason;
    the two are kept separate only because the acceleration layer runs after
    ``prepare`` and is a different review surface.
    """
    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return module


class _Output(NamedTuple):
    """Addresses a module's i-th return value, where a scatter may also live."""

    index: int


def _leaf_styles(style: ParallelStyle) -> Iterable[ParallelStyle]:
    """Yield the styles a plan entry applies, opening both kinds of composite.

    ``PrepareModuleInputOutput`` keeps its layouts on the ``PrepareModuleInput``
    and ``PrepareModuleOutput`` it builds, not on itself.
    """
    if isinstance(style, ComposedParallelStyle):
        for inner in style.styles:
            yield from _leaf_styles(inner)
    elif isinstance(style, PrepareModuleInputOutput):
        yield style.prepare_module_input
        yield style.prepare_module_output
    else:
        yield style


def _scatter_entries(plan: Mapping[str, ParallelStyle]):
    """Yield ``(path, argument, axis)`` for every token-axis scatter in a plan.

    A scatter is a layout transform that *wants* a ``Shard`` it did not
    already have: it receives the whole sequence and hands back this rank's
    slice.

    Four things make this narrower than it looks, each of them a false positive
    found by running it:

    * **The input must not already be sharded on that axis.** A style whose
      ``input_layouts`` and ``desired_input_layouts`` agree is a *pass-through*
      -- it declares the layout the argument already arrives in so the rest of
      the signature can be described. H3 writes ``transformer_blocks.*`` that
      way, with only ``transformer_blocks.0`` doing the real
      ``Replicate() -> Shard(1)``. Counting the pass-through makes the guard's
      hook on blocks 1..N measure the local slice instead of the global
      sequence, which demands ``(S / tp) % tp == 0`` -- effectively
      ``tp**2 | S`` -- and rejects a geometry that divides. Measured at
      ``S=6, tp=2``: ``6 % 2 == 0``, and the run died with "must be divisible
      by tp_size=2; received 3".
    * **It must be read before the plan is applied.** ``RowwiseParallel._apply``
      assigns itself ``desired_input_layouts = (Shard(-1),)``, because a
      row-parallel matmul needs its input sharded on the contraction axis. Scan
      an applied plan and every ``to_out`` looks like a scatter.
    * **The style must be a ``PrepareModuleInput`` or ``PrepareModuleOutput``**
      -- alone, composed, or as a half of a ``PrepareModuleInputOutput``.
      Nothing else can scatter.
    * **A negative axis is never the token axis.** ``Shard(-1)`` is the feature
      contraction; the token axis is named positively, 0 or 1.

    ``argument`` names where to look: an int for a positional parameter, a str
    for a keyword one, and ``_Output(index)`` for the module's i-th return value.
    All three occur. An *output* scatter is in fact the better form, and the one
    SD3.5 and Qwen-Image use: diffusers calls a transformer block by keyword in
    the eager forward and positionally through ``_gradient_checkpointing_func``,
    so a block-input transform can satisfy only one of the two conventions,
    while a forward hook is convention-free.
    """

    def scatters(source: object, desired: object) -> bool:
        """Whether this pair moves a whole axis into shards for the first time."""
        if not (isinstance(desired, Shard) and desired.dim >= 0):
            return False
        return not (isinstance(source, Shard) and source.dim == desired.dim)

    def moved(desired, sources, label):
        """Yield ``(argument, axis)`` for each pair this style actually scatters.

        Positional arguments, keyword arguments and outputs are the same
        question asked three ways, so they share the pairing and the
        bounds-guard rather than carrying a copy each -- ``label`` is the only
        difference. ``desired`` and ``sources`` are indexed alike: both are
        sequences, or both are mappings over the same keys.
        """
        keys = desired.keys() if isinstance(desired, Mapping) else range(len(desired))
        for key in keys:
            layout = desired[key]
            source = (
                sources.get(key)
                if isinstance(sources, Mapping)
                else (sources[key] if key < len(sources) else None)
            )
            if scatters(source, layout):
                yield label(key), layout.dim

    for path, style in plan.items():
        for candidate in _leaf_styles(style):
            if isinstance(candidate, PrepareModuleInput):
                for argument, axis in moved(
                    getattr(candidate, "desired_input_layouts", None) or (),
                    getattr(candidate, "input_layouts", None) or (),
                    lambda index: index,
                ):
                    yield path, argument, axis
                for argument, axis in moved(
                    getattr(candidate, "desired_input_kwarg_layouts", None) or {},
                    getattr(candidate, "input_kwarg_layouts", None) or {},
                    lambda name: name,
                ):
                    yield path, argument, axis
            elif isinstance(candidate, PrepareModuleOutput):
                # Addressed by position, not "every output": a module returning a
                # hidden state beside a mask would otherwise have the mask
                # measured on an axis the plan never sharded.
                for argument, axis in moved(
                    getattr(candidate, "desired_output_layouts", None) or (),
                    getattr(candidate, "output_layouts", None) or (),
                    _Output,
                ):
                    yield path, argument, axis


def _settle_collective_outputs(root: nn.Module, plan: Mapping[str, ParallelStyle]) -> int:
    """Make every activation leaving the plan a settled, self-owned tensor.

    A style that computes one placement and hands back another ends in a
    collective, and torch issues it with ``async_op=True``
    (``RowwiseParallel._prepare_output_fn``). ``DTensor.to_local()`` then returns
    ``local_tensor.view_as(local_tensor)`` *without* unwrapping the
    ``AsyncCollectiveTensor``, so what leaves the module is a **promise**: its
    buffer is owned by the functional-collectives work registry until someone
    calls ``wait_tensor`` on that exact tensor, not by the tensor's own
    refcount.

    Torch's default is right for its own usage -- the next op in the same forward
    consumes the result, which triggers the wait and clears the registry. It
    stops being right the moment a forward can end *without* that consumption,
    and tensor parallelism puts us there: ``torch.utils.checkpoint``'s
    non-reentrant recompute runs with ``early_stop=True`` by default and aborts
    through ``_StopRecomputationError`` as soon as it has recomputed what
    backward asked for. Any collective issued before that point and not yet
    consumed stays in flight forever. The same hole opens for a raised forward
    and for a branch whose output is discarded.

    Measured consequence before this existed: GRPO on SD3.5-Medium at
    ``tp_size: 2`` grew ~4.5 GB per epoch and died in the third evaluation, with
    288 live 15.6 MiB buffers added per epoch -- 24 blocks x 12 micro-batches --
    allocated by ``_functional_collectives.all_reduce`` under
    ``Redistribute.forward``. Every one of them had *no Python referrer at all*,
    which is what named the registry as the holder; torch's own exit warning
    counted "unwaited collective calls" in exact multiples of the layer count.
    Zero growth with ``fsdp_activation_checkpointing: false``, and zero at
    ``tp_size: 1`` where there is no collective to leave in flight.

    The predicate is the invariant, so this costs nothing when there is nothing
    pending: a settled output is not an ``AsyncCollectiveTensor`` and the hook
    does nothing at all. Nor does it cost throughput -- ``async_op`` buys overlap
    only when there is independent work between issue and consumption, and at a
    module boundary the consumer is the very next op.

    Returns:
        The number of module boundaries now settled.
    """

    def settle(_module: nn.Module, _inputs, outputs) -> None:
        """Discharge the collective. Deliberately changes nothing that flows.

        ``trigger_wait`` is used for its side effect and its return value is
        dropped: it waits, marks the tensor ``completed`` so a later real op does
        not wait twice, and clears the registry entry -- while the output object
        stays exactly what the module produced. Returning ``None`` tells PyTorch
        to leave the output alone.

        Returning ``wait()``'s value instead would be wrong, and a regression
        test caught it: both ``wait`` and ``trigger_wait`` hand back the
        *unwrapped* ``elem``, a different object without the autograd metadata
        the module's output carries, so backward died with "element 0 of tensors
        does not require grad and does not have a grad_fn".
        """
        for value in outputs if isinstance(outputs, tuple) else (outputs,):
            if isinstance(value, AsyncCollectiveTensor):
                value.trigger_wait()

    def may_be_in_flight(style: ParallelStyle) -> bool:
        """Whether this style can hand back an output it has not waited for.

        The question is not *which class* the style is but whether the placement
        it returns differs from the one it computed -- that difference is the
        redistribute, and the redistribute is the collective. A class allowlist
        answers it wrongly in both directions: a style outside the list silently
        loses its discharge (``SequenceParallel`` is the first entry
        :meth:`BaseAdapter.sequence_parallel_plan` recommends and would have
        dropped out), while a default ``ColwiseParallel`` is inside it and can
        never be in flight -- its ``output_layouts`` default of ``Shard(-1)`` is
        exactly what the column-parallel matmul computes, so
        ``_prepare_output_fn`` skips the redistribute. Measured on a real
        SD3.5-medium plan: 230 of 337 registrations were that dead case.

        An unrecognized style is assumed to be in flight, so a new one is
        protected by default rather than quietly excluded.
        """
        if isinstance(style, ColwiseParallel):
            # The one placement a colwise matmul produces by itself.
            return style.output_layouts != (Shard(-1),)
        # Rowwise always computes `Partial`, which is never a legal output, and a
        # `PrepareModule*` exists to redistribute. Styles defined in this module
        # keep activations plain or are explicit transitions that settle their
        # own collectives.
        return not isinstance(style, (_OwnParameterStyle, SequenceBroadcastOutput, _RegionEntry))

    paths = {
        path
        for path, style in plan.items()
        if any(may_be_in_flight(candidate) for candidate in _leaf_styles(style))
    }
    for path in sorted(paths):
        root.get_submodule(path).register_forward_hook(settle)
    return len(paths)


def _guard_sequence_divisibility(root: nn.Module, scatters: List, tp_size: int) -> int:
    """Refuse a token count the scatter cannot split evenly, before it deadlocks.

    ``PrepareModuleInput`` lifts a plain tensor with ``DTensor.from_local``,
    which has no global length to read and infers ``local * tp_size``. A
    125-token sequence scattered 63/62 makes the two ranks infer 126 and 124, so
    the gather at the next attention entry posts all-gathers of *different
    sizes* -- and that does not raise. Measured on LTX-2: both GPUs at 100%, not
    one line of output, four minutes until killed. A silent deadlock is worse to
    debug than a silent wrong answer, and the condition is not always under the
    user's control -- Qwen-Image-2.1's sharded sequence includes the
    vision-language encoder's output, whose length comes from the prompt.

    The check costs no collective, because the scatter is the one place that
    still sees the whole sequence: a local ``%`` on a shape, once per step.

    What an unguarded scatter costs is architecture-dependent, which is why the
    guard covers both input and output forms rather than only the loud one:
    LTX-2 deadlocks, while SD3.5 and Qwen-Image happen to have something
    position-indexed downstream of the gather -- an unpatchify, a rotary
    multiply -- that trips on the inflated length first and reports a shape
    error naming neither parallelism nor the sequence. Measured at 1,089 image
    tokens on two ranks.

    Returns:
        The number of scatter points guarded.
    """

    def measure(value, path: str, argument, axis: int) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim <= axis:
            return
        length = value.shape[axis]
        if not length % tp_size:
            return
        where = (
            f"its output {argument.index}"
            if isinstance(argument, _Output)
            else f"argument {argument!r}"
        )
        raise ValueError(
            f"sequence parallelism splits axis {axis} of {where} at {path!r}, so that axis "
            f"must be divisible by tp_size={tp_size}; received {length}. An uneven split "
            f"makes each rank infer a different global length "
            f"({length // tp_size + 1} and {length // tp_size} here), and the next gather "
            f"then posts mismatched all-gather sizes -- which deadlocks rather than "
            f"raising, or surfaces far downstream as a shape error naming neither. Choose "
            f"a geometry whose token count divides; where the count depends on the "
            f"prompt, set parallelism.sequence_parallel: false, which keeps tensor "
            f"parallelism and drops only the sequence split."
        )

    guarded = 0
    for path, argument, axis in scatters:
        module = root.get_submodule(path)

        # Both halves are prepended for the same reason: the scatter's own hook
        # is already registered and replaces the whole sequence with this rank's
        # slice, so a hook running after it would measure the slice -- 1,089
        # tokens would pass as 545.
        if isinstance(argument, _Output):

            def on_output(_module, _inputs, outputs, _argument=argument, _axis=axis, _path=path):
                values = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                if _argument.index < len(values):
                    measure(values[_argument.index], _path, _argument, _axis)

            module.register_forward_hook(on_output, prepend=True)
        else:

            def on_input(
                _module, inputs, kwargs, _argument=argument, _axis: int = axis, _path: str = path
            ):
                if isinstance(_argument, int):
                    value = inputs[_argument] if len(inputs) > _argument else None
                else:
                    value = kwargs.get(_argument)
                measure(value, _path, _argument, _axis)
                return None

            module.register_forward_pre_hook(on_input, prepend=True, with_kwargs=True)
        guarded += 1
    return guarded


def _shared_input_projections(
    plan: Mapping[str, ParallelStyle],
) -> Dict[str, SharedInputColwiseParallel]:
    """Map every path the plan gives :class:`SharedInputColwiseParallel` to its style."""
    return {
        path: leaf
        for path, style in plan.items()
        for leaf in _leaf_styles(style)
        if isinstance(leaf, SharedInputColwiseParallel)
    }


def _require_region_entries(plan: Mapping[str, ParallelStyle]) -> None:
    """Refuse a shared-input projection that no region entry encloses.

    Its input gradient is a partial sum that only an enclosing entry reduces;
    without one the loss stays bit-identical while everything upstream trains on
    this rank's share of its gradient.

    Raises:
        ValueError: Naming the projections without an enclosing entry.
    """
    regions = [
        path
        for path, style in plan.items()
        if any(isinstance(leaf, _RegionEntry) for leaf in _leaf_styles(style))
    ]
    orphans = sorted(
        path
        for path in _shared_input_projections(plan)
        if not any(region == "" or path.startswith(f"{region}.") for region in regions)
    )
    if orphans:
        raise ValueError(
            f"{len(orphans)} SharedInputColwiseParallel projection(s) have no enclosing "
            f"ColumnParallelRegion or SequenceParallelRegion, so their input gradient would "
            f"stay a partial sum over the tensor-parallel group: {orphans[:5]}. Put a region "
            f"entry on the module that receives their input, or use ColwiseParallel."
        )


def _install_region_input_checks(root: nn.Module, plan: Mapping[str, ParallelStyle]) -> int:
    """Check, on each projection's first gradient-recording call, that its input was lifted.

    :func:`_require_region_entries` proves an entry encloses the projection, not
    that the entry names the argument the projection reads: a processor can
    feed a projection from an input the entry does not lift, and that is as
    silent as having no entry. The walk runs once per projection and the hook
    then removes itself, so it costs nothing after the first training step.

    Returns:
        The number of projections checked.
    """
    hooks = {}
    for path, style in _shared_input_projections(plan).items():
        if not style.verify_input:
            continue

        def check(_module: nn.Module, args: tuple, _path: str = path) -> None:
            if not torch.is_grad_enabled() or not args:
                return None
            value = args[0]
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                return None
            if not _passed_region_entry(value):
                raise RuntimeError(
                    f"{_path} leaves its input gradient a partial sum for its region entry to "
                    f"reduce, but its input did not pass through a ColumnParallelRegion or "
                    f"SequenceParallelRegion: the entry on its enclosing module does not lift "
                    f"the argument this projection reads. Name that argument in the entry's "
                    f"inputs, or use ColwiseParallel for this projection."
                )
            hooks[_path].remove()
            return None

        hooks[path] = root.get_submodule(path).register_forward_pre_hook(check)
    return len(hooks)


class _NormPair:
    """What one attention's paired across-head norms hand each other within a forward.

    ``key`` is the key projection's latest output, until ``norm_q`` consumes
    it; ``result`` is that key with its version and its summed statistic, until
    ``norm_k`` claims it. Only tensors and ints are held, so a deep copy of the
    module is unaffected, and the attention's ``always_call`` forward hook
    empties both on every exit -- including a raised forward and checkpoint
    recompute stopping early, either of which would otherwise leave a layer's
    activations alive until its next forward.
    """

    __slots__ = ("key", "result")

    def __init__(self) -> None:
        self.key: Optional[torch.Tensor] = None
        self.result: Optional[Tuple[torch.Tensor, int, torch.Tensor]] = None

    def clear(self) -> None:
        self.key = None
        self.result = None


def _same_key(recorded: torch.Tensor, version: int, received: torch.Tensor) -> bool:
    """Whether ``received`` is the recorded key, unmodified.

    By storage, not by object: a module with a full backward hook rewraps its
    output in an alias after its forward hooks have run, so the processor hands
    ``norm_k`` a different object over the same memory. The recorded key keeps
    that memory alive, so equal addresses mean the same tensor, and an alias
    shares the version counter, so an in-place write in between is caught.
    """
    if received is recorded:
        return received._version == version
    return (
        received.data_ptr() == recorded.data_ptr()
        and received.storage_offset() == recorded.storage_offset()
        and received.shape == recorded.shape
        and received.stride() == recorded.stride()
        and received.dtype == recorded.dtype
        and received.device == recorded.device
        and received._version == version
    )


def _pair_across_head_norms(
    root: nn.Module, plan: Mapping[str, ParallelStyle], tp_mesh: DeviceMesh
) -> int:
    """Give each attention's q and k across-head norms one all-reduce between them.

    Each :class:`ShardedAcrossHeadRMSNorm` all-reduces its own ``[batch, seq,
    1]`` statistic, forward and backward -- two small, latency-bound
    collectives per attention where one would carry both. The norms are
    separate module calls, so the pairing reads the key off ``to_k``'s output:
    Wan's and LTX-2's processors project q, k and v before normalizing either,
    so the key exists when ``norm_q`` runs, and ``norm_q`` computes both
    statistics, packs them flat -- a cross-attention's lengths differ -- and
    reduces them together. ``norm_k`` normalizes its own input with the stored
    statistic when that input is the same key, unmodified, and computes on its
    own otherwise, so a processor that orders the calls differently or
    transforms the key in between stays correct, only unpaired.

    Returns:
        The number of attention modules paired.
    """
    styled = {
        path
        for path, style in plan.items()
        if any(isinstance(leaf, ShardedAcrossHeadRMSNorm) for leaf in _leaf_styles(style))
    }
    group = tp_mesh.get_group()
    paired = 0
    for query_path in sorted(path for path in styled if path.endswith(".norm_q")):
        prefix = query_path[: -len(".norm_q")]
        if f"{prefix}.norm_k" not in styled:
            continue
        attention = root.get_submodule(prefix)
        key_projection = getattr(attention, "to_k", None)
        if not isinstance(key_projection, nn.Module):
            continue
        _install_norm_pair(attention, key_projection, group)
        paired += 1
    return paired


def _install_norm_pair(attention: nn.Module, key_projection: nn.Module, group) -> None:
    norm_q, norm_k = attention.norm_q, attention.norm_k
    unfused_q, unfused_k = norm_q.forward, norm_k.forward
    q_size, q_epsilon = norm_q._ff_across_head_norm
    k_size, k_epsilon = norm_k._ff_across_head_norm
    pair = _NormPair()
    attention._ff_norm_pair = pair

    def record_key(_module: nn.Module, _inputs, output) -> None:
        pair.key = output if isinstance(output, torch.Tensor) else None
        pair.result = None

    def paired_query(self: nn.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
        key = pair.key
        if key is None or isinstance(key, DTensor) or isinstance(hidden_states, DTensor):
            return unfused_q(hidden_states)
        pair.key = None
        query_square_sum = _across_head_square_sum(hidden_states)
        key_square_sum = _across_head_square_sum(key)
        dtype = torch.promote_types(query_square_sum.dtype, key_square_sum.dtype)
        packed = torch.cat(
            [query_square_sum.reshape(-1).to(dtype), key_square_sum.reshape(-1).to(dtype)]
        )
        summed = differentiable_all_reduce(packed, group=group)
        split = query_square_sum.numel()
        query_total = summed[:split].reshape(query_square_sum.shape).to(query_square_sum.dtype)
        key_total = summed[split:].reshape(key_square_sum.shape).to(key_square_sum.dtype)
        pair.result = (key, key._version, key_total)
        return _across_head_normalize(hidden_states, query_total, q_size, q_epsilon, self.weight)

    def paired_key(self: nn.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
        result = pair.result
        if result is not None and _same_key(result[0], result[1], hidden_states):
            pair.result = None
            return _across_head_normalize(hidden_states, result[2], k_size, k_epsilon, self.weight)
        return unfused_k(hidden_states)

    def release(_module: nn.Module, _inputs, _outputs) -> None:
        pair.clear()

    norm_q.forward = types.MethodType(paired_query, norm_q)
    norm_k.forward = types.MethodType(paired_key, norm_k)
    key_projection.register_forward_hook(record_key)
    attention.register_forward_hook(release, always_call=True)


def _tensor_leaves(value: object, path: str) -> Iterable[Tuple[str, torch.Tensor]]:
    """Yield the plain tensors of a forward's arguments in a rank-independent order."""
    if isinstance(value, DTensor):
        return
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _tensor_leaves(value[key], f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, f"{path}[{index}]")


def _comparable_bits(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Flatten ``tensor`` to int64 whose elementwise equality is value identity.

    Floating values widen to float64 first, which holds every float16, bfloat16
    and float32 value exactly; NCCL reduces no 16-bit integer type, so a direct
    bit view of a half-precision tensor could not be compared.
    """
    detached = tensor.detach().to(device)
    if detached.is_complex():
        detached = torch.view_as_real(detached)
    if detached.is_floating_point():
        return detached.to(torch.float64).contiguous().view(torch.int64).reshape(-1)
    return detached.to(torch.int64).reshape(-1)


def install_replicated_input_check(root: nn.Module, tp_mesh: DeviceMesh, name: str):
    """Raise before any forward of ``root`` whose inputs differ across the group.

    The plan assumes every peer enters ``root`` with the same activations: its
    column-parallel layers take slices of them and its row-parallel all-reduces
    sum the slices back. Inputs that differ give an output matching neither
    peer's unsharded result, and nothing raises. This compares every plain
    tensor among the positional and keyword arguments by value, with a MAX and a
    MIN all-reduce of their packed int64 form, after a two-element signature
    exchange that keeps the packed sizes equal on every peer. All peers see the
    same reductions, so they raise together rather than leaving one waiting.

    Args:
        root: The sharded component, as passed to :func:`apply_tensor_parallel`.
        tp_mesh: The tensor-parallel dimension of the device mesh.
        name: Component label for the error message.

    Returns:
        The hook handle.
    """
    group = tp_mesh.get_group()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if tp_mesh.device_type == "cuda"
        else torch.device(tp_mesh.device_type)
    )

    def agree(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        high, low = values.clone(), values.clone()
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
        return high, low

    def check(module: nn.Module, args: tuple, kwargs: dict) -> None:
        del module
        leaves = list(_tensor_leaves(args, "args")) + list(_tensor_leaves(kwargs, "kwargs"))
        bits = [_comparable_bits(tensor, device) for _, tensor in leaves]
        total = sum(b.numel() for b in bits)
        high, low = agree(torch.tensor([len(bits), total], dtype=torch.int64, device=device))
        if not torch.equal(high, low):
            raise RuntimeError(
                f"tensor-parallel peers entered {name} with differently shaped inputs: this "
                f"rank passes {len(bits)} tensor(s) of {total} element(s) in all, while the "
                f"group ranges from {low.tolist()} to {high.tolist()}"
            )
        if not bits:
            return None
        high, low = agree(torch.cat(bits))
        if torch.equal(high, low):
            return None
        differs = high != low
        offset = 0
        for (path, tensor), packed in zip(leaves, bits):
            count = int(differs[offset : offset + packed.numel()].sum())
            if count:
                raise RuntimeError(
                    f"tensor-parallel peers entered {name} with different {path}: {count} of "
                    f"{packed.numel()} element(s) differ (shape {tuple(tensor.shape)}, dtype "
                    f"{tensor.dtype}). The plan combines its peers' slices of one input, so its "
                    "output matches neither peer; look upstream for a draw from an unshared "
                    "RNG or a rank-dependent branch."
                )
            offset += packed.numel()
        return None

    return root.register_forward_pre_hook(check, with_kwargs=True)


def apply_tensor_parallel(
    root: nn.Module,
    tp_mesh: DeviceMesh,
    shard_plan: Mapping[str, ParallelStyle],
) -> Dict[str, ParallelStyle]:
    """Shard ``root`` over ``tp_mesh``, completing and validating the plan.

    Args:
        root: The trainable module to shard, after LoRA injection and variant
            materialization so every parameter that will exist already does.
        tp_mesh: The tensor-parallel dimension of the device mesh.
        shard_plan: The architecture's colwise/rowwise entries.

    Returns:
        The completed plan that was applied, for logging and tests.

    Raises:
        RuntimeError: If any parameter is left undistributed, which would make
            gradient clipping fail later with an opaque dispatch error.
        ValueError: If a :class:`SharedInputColwiseParallel` projection has no
            enclosing region entry to reduce its input gradient.
    """
    total = build_total_plan(root, shard_plan)
    _require_region_entries(total)

    # torch's built-in styles rebuild each parameter as `nn.Parameter(...)`
    # without forwarding `requires_grad`, which defaults to True. Left alone
    # that unfreezes every sharded `base_layer` of a LoRA run, turning it into a
    # partial full finetune and invalidating the frozen-ness that the LoRA-only
    # checkpoint gather captures at `fully_shard` time.
    trainability = {name: p.requires_grad for name, p in root.named_parameters()}

    # Read before applying: `RowwiseParallel._apply` assigns itself
    # `desired_input_layouts = (Shard(-1),)`, so an applied plan makes every
    # row-parallel layer look like a token-axis scatter.
    scatters = list(_scatter_entries(total))

    parallelize_module(root, tp_mesh, total, src_data_rank=0)

    restored = 0
    for name, parameter in root.named_parameters():
        expected = trainability.get(name)
        if expected is not None and parameter.requires_grad != expected:
            parameter.requires_grad_(expected)
            restored += 1

    adjusted = localize_attention_head_counts(root, tp_mesh.size())
    guarded = _guard_sequence_divisibility(root, scatters, tp_mesh.size())
    settled = _settle_collective_outputs(root, total)
    _install_region_input_checks(root, total)
    _pair_across_head_norms(root, total, tp_mesh)

    # Buffers must stay replicated. `BackendLoadRuntime.bootstrap_targets`
    # broadcasts every target buffer from rank 0 under cpu-efficient loading, so
    # a sharded buffer would have its peers' shards silently overwritten with
    # rank 0's. That call runs before this one today; asserting here keeps the
    # invariant true rather than resting on the ordering never changing.
    sharded_buffers = [
        name
        for name, buffer in root.named_buffers()
        if isinstance(buffer, DTensor) and any(not p.is_replicate() for p in buffer.placements)
    ]
    if sharded_buffers:
        raise RuntimeError(
            f"the tensor-parallel plan sharded {len(sharded_buffers)} buffer(s), which a "
            f"rank-0 buffer broadcast would overwrite with rank 0's shard: "
            f"{sharded_buffers[:5]}"
        )

    undistributed = [name for name, p in root.named_parameters() if not isinstance(p, DTensor)]
    if undistributed:
        raise RuntimeError(
            f"the tensor-parallel plan left {len(undistributed)} parameter(s) "
            f"undistributed, which makes gradient clipping fail later with a mesh "
            f"dispatch error: {undistributed[:5]}"
        )

    logger.info(
        "Applied tensor parallelism over %d ranks: %d plan entries, %d attention "
        "modules localized, %d parameter(s) refrozen, %d sequence scatter(s) guarded, "
        "%d collective output(s) settled.",
        tp_mesh.size(),
        len(total),
        adjusted,
        restored,
        guarded,
        settled,
    )
    return total
