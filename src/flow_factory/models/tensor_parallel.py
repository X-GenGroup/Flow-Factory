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

import logging
import types
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
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
    "ComposedParallelStyle",
    "ReplicateParallel",
    "SequenceBroadcastOutput",
    "SequenceBroadcastParameter",
    "ShardedAcrossHeadRMSNorm",
    "peel_peft_wrapper",
    "ShardedInputReplicatedWeight",
    "apply_tensor_parallel",
    "build_total_plan",
    "localize_attention_head_counts",
]


class ReplicateParallel(ParallelStyle):
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

    def __init__(self, *, src_data_rank: int | None = 0) -> None:
        super().__init__()
        self.src_data_rank = src_data_rank

    def _partition(self, module: nn.Module, device_mesh: DeviceMesh) -> None:
        for name, parameter in list(module.named_parameters(recurse=False)):
            if isinstance(parameter, DTensor):
                continue
            # `distribute_tensor` broadcasts from `src_data_rank`, making one
            # rank's weights authoritative instead of trusting that every rank
            # initialized or loaded identically.
            module.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(
                        parameter.data,
                        device_mesh,
                        [Replicate()],
                        src_data_rank=self.src_data_rank,
                    ),
                    requires_grad=parameter.requires_grad,
                ),
            )

    @staticmethod
    def _shadow_with_local_views(module: nn.Module, _inputs) -> None:
        """Expose each replicated parameter to ``forward`` as a plain tensor.

        Written into the instance dictionary, which attribute lookup consults
        before ``nn.Module.__getattr__`` reaches ``_parameters``. Refreshed every
        forward rather than installed once: ``to_local()`` has to run inside the
        autograd graph for the gradient to find its way back to the parameter,
        and an activation-checkpointed recompute needs its own node. Under FSDP2
        the parameter is unsharded by this point, so the local view is the full
        tensor.
        """
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(parameter, DTensor):
                module.__dict__[name] = parameter.to_local()

    @staticmethod
    def _drop_local_views(module: nn.Module, _inputs, _outputs) -> None:
        """Restore plain attribute access to the parameter itself.

        The shadow must not outlive the forward. FSDP2 and the checkpoint APIs
        read parameters through ``getattr``, and a leftover plain tensor there
        makes them fail on a DTensor-only attribute -- observed as
        ``'Tensor' object has no attribute '_local_tensor'`` during a composed
        (dp, tp) save.

        That window is also a trap, measured rather than predicted. A module
        that reads its *own* parameter inside a function it hands to
        ``torch.utils.checkpoint`` is unsafe: the recompute calls that function
        directly, so this module's pre-hook does not fire again and the shadow
        is already gone. It raises on the first backward -- "aten.mm got mixed
        torch.Tensor and DTensor" -- so it cannot become a wrong gradient, and
        it only shows up when the input requires grad. Whole-block checkpointing
        is safe for the opposite reason: it re-enters ``forward``, so every inner
        pre-hook fires. H3's three chunked wrappers each clear it, but each for
        a different reason, and ``_ChunkedRMSNorm`` clears it only because it
        does not checkpoint -- it does read ``self.weight`` in its chunk body,
        so adding a checkpoint there, which its sibling already has, would make
        it fail. See ``.scratch/tp_probe/probe_chunked_recompute.py``.
        """
        for name in list(module.__dict__):
            if name in module._parameters:
                del module.__dict__[name]

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self._partition(module, device_mesh)
        module.register_forward_pre_hook(self._shadow_with_local_views)
        module.register_forward_hook(self._drop_local_views)
        return module


class ShardedAcrossHeadRMSNorm(ParallelStyle):
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
    normalization needs: summing squares over a sharded axis yields a
    ``Partial``, and redistributing it to ``Replicate`` is the all-reduce.
    DTensor owns that collective's backward. The divisor stays the **global**
    normalized size, because that is what the unsharded module divided by.

    The extra traffic is one ``[batch, seq, 1]`` all-reduce per norm --
    ``1 / normalized_size`` of what a row-parallel output projection already
    costs -- so it does not change the communication picture.

    ``forward`` is replaced rather than hooked because the math itself changes:
    ``nn.RMSNorm.forward`` calls ``F.rms_norm`` with the module's
    ``normalized_shape``, which no longer describes this rank's slice.
    """

    def __init__(self, *, src_data_rank: int | None = 0) -> None:
        super().__init__()
        self.src_data_rank = src_data_rank

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

        if module.weight is not None and not isinstance(module.weight, DTensor):
            module.register_parameter(
                "weight",
                nn.Parameter(
                    distribute_tensor(
                        module.weight.data,
                        device_mesh,
                        [Shard(0)],
                        src_data_rank=self.src_data_rank,
                    ),
                    requires_grad=module.weight.requires_grad,
                ),
            )

        def forward(self: nn.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
            local = (
                hidden_states.to_local() if isinstance(hidden_states, DTensor) else hidden_states
            )
            sharded = DTensor.from_local(local, device_mesh, [Shard(-1)], run_check=False)
            square_sum = sharded.float().pow(2).sum(-1, keepdim=True)
            if any(p.is_partial() for p in square_sum.placements):
                square_sum = square_sum.redistribute(device_mesh, [Replicate()])
            # Only the [batch, seq, 1] statistic is held in fp32; upcasting the
            # activation would cost more memory than the sharding saves.
            scale = torch.rsqrt(square_sum / normalized_size + epsilon).to(local.dtype)
            normalized = sharded * scale
            if self.weight is not None:
                normalized = normalized * self.weight
            return normalized.to_local()

        module.forward = types.MethodType(forward, module)
        return module


class ShardedInputReplicatedWeight(ParallelStyle):
    """Replicate a module's weight while declaring its input a shard.

    Per-head qk-norm is why this exists. ``norm_q.weight`` is ``[head_dim]`` and
    shared by every head, so the weight itself is replicated -- but
    column-parallel q/k leave this rank holding only ``heads / tp_size`` heads,
    so the *input* is a shard along the head axis.

    Declaring that input ``Replicate`` is what makes the weight gradient
    silently wrong: each rank then accumulates a partial sum over its own heads
    and nothing reconciles them. Declaring it ``Shard(shard_axis)`` puts the truth
    in the autograd graph, so DTensor derives a ``Partial`` weight gradient and
    reduces it when the gradient is materialized.

    Expressing it in the graph, rather than as a hook on the parameter, is what
    makes it survive ``fully_shard``: FSDP replaces the parameter objects, so a
    ``register_post_accumulate_grad_hook`` installed beforehand is simply lost,
    while the autograd graph is rebuilt every step.

    The normalized axis is the last one and is never the sharded axis, so the
    forward pass needs no collective.

    The axis is a parameter because architectures disagree on where the head
    axis lands after the reshape -- SD3.5 and FLUX put it at 1, Qwen and H3 at
    2. It is *not* a way to express the sequence axis: a ``Linear`` handed a
    DTensor sharded on an interior dimension fails in ``F.linear``, which
    flattens batch into sequence. The sequence-parallel region uses
    :class:`SequenceBroadcastParameter` instead, which leaves activations plain
    and corrects only the gradient.
    """

    def __init__(self, *, shard_axis: int = 2, src_data_rank: int | None = 0) -> None:
        super().__init__()
        self.shard_axis = shard_axis
        self.src_data_rank = src_data_rank

    def _partition(self, module: nn.Module, device_mesh: DeviceMesh) -> None:
        for name, parameter in list(module.named_parameters(recurse=False)):
            if isinstance(parameter, DTensor):
                continue
            module.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(
                        parameter.data,
                        device_mesh,
                        [Replicate()],
                        src_data_rank=self.src_data_rank,
                    ),
                    requires_grad=parameter.requires_grad,
                ),
            )

    def _lift(self, device_mesh: DeviceMesh, inputs):
        return tuple(
            (
                DTensor.from_local(value, device_mesh, [Shard(self.shard_axis)], run_check=False)
                if isinstance(value, torch.Tensor) and not isinstance(value, DTensor)
                else value
            )
            for value in inputs
        )

    @staticmethod
    def _lower(outputs):
        if isinstance(outputs, DTensor):
            return outputs.to_local()
        if isinstance(outputs, tuple):
            return tuple(o.to_local() if isinstance(o, DTensor) else o for o in outputs)
        return outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self._partition(module, device_mesh)
        module.register_forward_pre_hook(lambda _module, inputs: self._lift(device_mesh, inputs))
        module.register_forward_hook(lambda _module, _inputs, outputs: self._lower(outputs))
        return module


class _ToLocalWithPartialGradient(torch.autograd.Function):
    """Identity to the local tensor, with the gradient declared a partial sum.

    ``DTensor.to_local()`` rewraps its gradient with the *forward* placements,
    so a ``Replicate`` parameter gets a ``Replicate`` gradient -- an assertion
    that every rank computed the same value. Under sequence parallelism that is
    false for a parameter broadcast over the token axis: each rank multiplied it
    by its own slice and holds part of the sum. Declaring the gradient
    ``Partial`` is what makes DTensor add them up.
    """

    @staticmethod
    def forward(ctx, parameter: DTensor) -> torch.Tensor:  # type: ignore[override]
        ctx.mesh = parameter.device_mesh
        return parameter.to_local()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # type: ignore[override]
        return DTensor.from_local(gradient, ctx.mesh, [Partial()], run_check=False)


class SequenceBroadcastParameter(ParallelStyle):
    """Replicate a module's *own* parameters and sum the gradient they accumulate.

    The style for every parameter inside the sequence-parallel region: a
    modulation table held bare on a block, an affine norm weight, the output
    projection. All of them are applied to each token independently, so under a
    sharded token axis each rank computes with its own slice and the gradient it
    accumulates is a partial sum. The forward needs nothing -- a replicated
    weight on a token slice is already right -- so the parameter is distributed
    ``Replicate``, which keeps clipping and the optimizer on one mesh, and
    ``forward`` is handed a local view whose backward declares the gradient
    ``Partial``.

    Why not :class:`ShardedInputReplicatedWeight` with the sequence axis, which
    would let DTensor derive the same thing: it cannot reach either case. A bare
    ``nn.Parameter`` is read inside the block's own ``forward``, where there is
    no module boundary to hook. And a ``Linear`` fed a DTensor sharded on an
    interior axis fails outright -- ``F.linear`` flattens batch into sequence,
    and "Attempted to flatten multiple dimensions, with dimension 1 being
    sharded" is where that ends. Keeping activations plain sidesteps both.

    Getting this wrong is silent in the worst way: the loss is **bit-identical**,
    because the forward genuinely is correct on every rank, and only the
    parameter's gradient comes out at this rank's share of the tokens. Measured
    at exactly 1/2 on two ranks before the fix.

    Only the module's direct parameters are touched, so a block carrying both a
    modulation table and sharded projections keeps both treatments.
    """

    def __init__(self, *, src_data_rank: int | None = 0) -> None:
        super().__init__()
        self.src_data_rank = src_data_rank

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        for name, parameter in list(module.named_parameters(recurse=False)):
            if isinstance(parameter, DTensor):
                continue
            module.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(
                        parameter.data,
                        device_mesh,
                        [Replicate()],
                        src_data_rank=self.src_data_rank,
                    ),
                    requires_grad=parameter.requires_grad,
                ),
            )

        def shadow(owner: nn.Module, _inputs) -> None:
            for name, parameter in owner.named_parameters(recurse=False):
                if isinstance(parameter, DTensor):
                    owner.__dict__[name] = _ToLocalWithPartialGradient.apply(parameter)

        def drop(owner: nn.Module, _inputs, _outputs) -> None:
            for name in list(owner.__dict__):
                if name in owner._parameters:
                    del owner.__dict__[name]

        module.register_forward_pre_hook(shadow)
        module.register_forward_hook(drop)
        return module


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


def _expand_globs(root: nn.Module, plan: Mapping[str, ParallelStyle]) -> Dict[str, ParallelStyle]:
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

    Raises:
        ValueError: If a pattern matches no module, which means the plan names a
            layer this architecture does not have.
    """
    candidates = [name for name, _ in root.named_modules() if name]
    resolved: Dict[str, ParallelStyle] = {}
    exact = {path: style for path, style in plan.items() if "*" not in path}
    for pattern, style in plan.items():
        if "*" not in pattern:
            continue
        prefix, _, suffix = pattern.partition("*")
        suffix = suffix.lstrip(".")
        matched = False
        for name in candidates:
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            middle = name[len(prefix) : len(name) - len(suffix)].strip(".")
            # `*` stands for exactly one path segment.
            if middle and "." not in middle:
                resolved[name] = style
                matched = True
                if name in exact:
                    # Recorded as matched so the pattern is not reported unused,
                    # but the exact entry below is the one that applies.
                    continue
        if not matched:
            raise ValueError(
                f"tensor-parallel plan pattern {pattern!r} matched no module. The plan "
                f"must name modules this architecture has, or the layer it was written "
                f"for has been renamed."
            )
    resolved.update(exact)
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
            else:
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
        elif isinstance(existing, (*_ACTIVATION_ONLY_STYLES, SequenceBroadcastOutput)):
            # An activation transform distributes nothing, so a boundary module
            # that owns parameters still needs a parameter style beside it.
            total[module_name] = ComposedParallelStyle(ReplicateParallel(), existing)
    return total


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
    This one is caught rather than silent: per-head qk-norm puts the head axis in
    the DTensor graph, so the mismatch raises a sharding-propagation error at
    ``attn.norm_q`` instead of training on a mis-split tensor.

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
        heads = _local_share(module.heads, module, "head count")
        if heads is None:
            continue
        module.heads = heads
        inner_dim = _local_share(getattr(module, "inner_dim", None), module, "inner width")
        if inner_dim is not None:
            module.inner_dim = inner_dim
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


def _scatter_entries(plan: Mapping[str, ParallelStyle]):
    """Yield ``(path, argument, axis)`` for every token-axis scatter in a plan.

    A scatter is a ``PrepareModuleInput`` that *wants* a ``Shard``: it receives
    the whole sequence and hands back this rank's slice.

    Three things make this narrower than it looks, each of them a false positive
    found by running it:

    * **It must be read before the plan is applied.** ``RowwiseParallel._apply``
      assigns itself ``desired_input_layouts = (Shard(-1),)``, because a
      row-parallel matmul needs its input sharded on the contraction axis. Scan
      an applied plan and every ``to_out`` looks like a scatter.
    * **The style must be a ``PrepareModuleInput``.** Nothing else can scatter.
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
    for path, style in plan.items():
        candidates = style.styles if isinstance(style, ComposedParallelStyle) else (style,)
        for candidate in candidates:
            if isinstance(candidate, (PrepareModuleInput, PrepareModuleInputOutput)):
                positional = getattr(candidate, "desired_input_layouts", None) or ()
                for index, layout in enumerate(positional):
                    if isinstance(layout, Shard) and layout.dim >= 0:
                        yield path, index, layout.dim
                keyword = getattr(candidate, "desired_input_kwarg_layouts", None) or {}
                for name, layout in keyword.items():
                    if isinstance(layout, Shard) and layout.dim >= 0:
                        yield path, name, layout.dim
            if isinstance(candidate, (PrepareModuleOutput, PrepareModuleInputOutput)):
                outputs = getattr(candidate, "desired_output_layouts", None) or ()
                for index, layout in enumerate(outputs):
                    if isinstance(layout, Shard) and layout.dim >= 0:
                        # By position, not "every output": a module returning a
                        # hidden state beside a mask would otherwise have the
                        # mask measured on an axis the plan never sharded.
                        yield path, _Output(index), layout.dim


def _guard_sequence_divisibility(root: nn.Module, scatters: List, tp_size: int) -> int:
    """Refuse a token count the scatter cannot split evenly, before it deadlocks.

    ``PrepareModuleInput`` lifts a plain tensor with ``DTensor.from_local``,
    which has no global length to read and infers ``local * tp_size``. A
    125-token sequence scattered 63/62 makes the two ranks infer 126 and 124, so
    the gather at the next attention entry posts all-gathers of *different
    sizes* -- and that does not raise. Measured on LTX-2: both GPUs at 100%, not
    one line of output, four minutes until killed. A silent deadlock is worse to
    debug than a silent wrong answer, and the condition is not always under the
    user's control -- SenseNova's sharded sequence includes the text prefix,
    whose length comes from the prompt.

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
            f"raising, or surfaces far downstream as a shape error naming neither. Pad the "
            f"sequence, or choose a geometry whose token count divides."
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
    """
    total = build_total_plan(root, shard_plan)

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
        "modules localized, %d parameter(s) refrozen, %d sequence scatter(s) guarded.",
        tp_mesh.size(),
        len(total),
        adjusted,
        restored,
        guarded,
    )
    return total
