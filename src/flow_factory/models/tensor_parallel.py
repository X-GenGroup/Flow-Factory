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
from typing import Dict, Iterable, List, Mapping

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    parallelize_module,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ReplicateParallel",
    "ShardedInputReplicatedWeight",
    "apply_tensor_parallel",
    "build_total_plan",
    "localize_attention_head_counts",
]


class ReplicateParallel(ParallelStyle):
    """Hold a module's own parameters replicated across the tensor-parallel mesh.

    This is the completion style for modules whose input really is replicated.
    It makes every gradient in the module share one mesh, which is what gradient
    clipping requires, and lets DTensor perform the Partial -> Replicate
    reduction itself.

    The activation hooks are not optional: a module holding a DTensor weight
    cannot be applied to a plain activation (``aten.mm`` rejects the mix), so the
    input is lifted to ``Replicate`` and the output lowered back to a local
    tensor. Both conversions are metadata-only -- ``Replicate`` to ``Replicate``
    needs no collective.

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
    def _lift(device_mesh: DeviceMesh, inputs):
        return tuple(
            (
                DTensor.from_local(value, device_mesh, [Replicate()], run_check=False)
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


class ShardedInputReplicatedWeight(ParallelStyle):
    """Replicate a module's weight while declaring its input a shard.

    Per-head qk-norm is why this exists. ``norm_q.weight`` is ``[head_dim]`` and
    shared by every head, so the weight itself is replicated -- but
    column-parallel q/k leave this rank holding only ``heads / tp_size`` heads,
    so the *input* is a shard along the head axis.

    Declaring that input ``Replicate`` is what makes the weight gradient
    silently wrong: each rank then accumulates a partial sum over its own heads
    and nothing reconciles them. Declaring it ``Shard(head_axis)`` puts the truth
    in the autograd graph, so DTensor derives a ``Partial`` weight gradient and
    reduces it when the gradient is materialized.

    Expressing it in the graph, rather than as a hook on the parameter, is what
    makes it survive ``fully_shard``: FSDP replaces the parameter objects, so a
    ``register_post_accumulate_grad_hook`` installed beforehand is simply lost,
    while the autograd graph is rebuilt every step.

    The normalized axis is the last one and is never the sharded axis, so the
    forward pass needs no collective.
    """

    def __init__(self, *, head_axis: int = 2, src_data_rank: int | None = 0) -> None:
        super().__init__()
        self.head_axis = head_axis
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
                DTensor.from_local(value, device_mesh, [Shard(self.head_axis)], run_check=False)
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

    Raises:
        ValueError: If a pattern matches no module, which means the plan names a
            layer this architecture does not have.
    """
    candidates = [name for name, _ in root.named_modules() if name]
    resolved: Dict[str, ParallelStyle] = {}
    for pattern, style in plan.items():
        if "*" not in pattern:
            resolved[pattern] = style
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
        if not matched:
            raise ValueError(
                f"tensor-parallel plan pattern {pattern!r} matched no module. The plan "
                f"must name modules this architecture has, or the layer it was written "
                f"for has been renamed."
            )
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
                derived[f"{path}.lora_A.{adapter}"] = RowwiseParallel()
                derived[f"{path}.lora_B.{adapter}"] = ReplicateParallel()
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
        if module_name and module_name not in total:
            total[module_name] = ReplicateParallel()
    return total


def localize_attention_head_counts(root: nn.Module, tp_size: int) -> int:
    """Rewrite each attention module's head count to the count it now owns.

    Column-parallel q/k/v leave each rank with ``heads / tp_size`` heads. A
    processor that reshapes with ``unflatten(-1, (attn.heads, -1))`` would read
    the global count and mis-split the local projection, so the attribute is set
    to this rank's share -- which after sharding is what it genuinely is.
    Processors written as ``unflatten(-1, (-1, head_dim))`` infer the count and
    are unaffected either way.

    Args:
        root: The parallelized module tree.
        tp_size: The tensor-parallel degree.

    Returns:
        The number of attention modules adjusted.

    Raises:
        ValueError: If any head count is not divisible by ``tp_size``.
    """
    adjusted = 0
    for module in root.modules():
        heads = getattr(module, "heads", None)
        if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0:
            continue
        if heads % tp_size != 0:
            raise ValueError(
                f"tensor parallelism requires every attention head count to be divisible "
                f"by the tensor-parallel degree, but {type(module).__name__} has "
                f"heads={heads} and tp_size={tp_size}."
            )
        module.heads = heads // tp_size
        adjusted += 1
    return adjusted


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

    parallelize_module(root, tp_mesh, total, src_data_rank=0)

    restored = 0
    for name, parameter in root.named_parameters():
        expected = trainability.get(name)
        if expected is not None and parameter.requires_grad != expected:
            parameter.requires_grad_(expected)
            restored += 1

    adjusted = localize_attention_head_counts(root, tp_mesh.size())

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
        "modules localized, %d parameter(s) refrozen.",
        tp_mesh.size(),
        len(total),
        adjusted,
        restored,
    )
    return total
