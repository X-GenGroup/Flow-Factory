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
from typing import Dict, Iterable, List, Mapping, Optional

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
        """
        for name in list(module.__dict__):
            if name in module._parameters:
                del module.__dict__[name]

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self._partition(module, device_mesh)
        module.register_forward_pre_hook(self._shadow_with_local_views)
        module.register_forward_hook(self._drop_local_views)
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
        if module_name not in total:
            # Including `""`, the root: Wan and LTX-2 keep an AdaLN table there
            # as a bare parameter, and skipping it left that parameter a plain
            # tensor, which `apply_tensor_parallel` then rejects.
            total[module_name] = ReplicateParallel()
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
