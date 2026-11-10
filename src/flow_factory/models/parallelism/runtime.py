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


"""Apply a total TP/SP plan while preserving trainability and native head geometry."""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module

from .guards import (
    _guard_sequence_divisibility,
    _install_region_input_checks,
    _leaf_styles,
    _pair_across_head_norms,
    _require_region_entries,
    _scatter_entries,
    _settle_collective_outputs,
)
from .plan import build_total_plan
from .sequence import SequencePadding

logger = logging.getLogger(__name__)


#: Set on an attention module once its head count is localized, which makes
#: :func:`localize_attention_head_counts` idempotent. :func:`apply_tensor_parallel`
#: never localizes a module twice -- ``distribute_module`` refuses a second
#: application first -- but the function is public, and it rewrites a count in
#: place: halving ``heads`` twice mis-splits every projection, silently for a
#: cross-attention with no rotary width to trip on.
_LOCALIZED_HEADS_MARKER = "_ff_tp_heads_localized"


def _owns_a_column_shard(module: nn.Module) -> bool:
    """Detect a dimension-zero parameter shard below one of a module's direct children.

    Inspecting live parameters also reaches PEFT's nested ``base_layer`` and
    ``lora_B`` projections. Column projections and across-head norm weights can
    carry this layout; a row-parallel output weight alone carries ``Shard(1)``.

    For an attention module, sibling feed-forward parameters are outside this
    search. Their sharding must not cause an unsharded attention's head metadata
    to be localized.
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

    Only attention modules with detected column shards are rewritten. A plan
    that leaves attention replicated keeps full-width projections, so halving
    the count would split each head into
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


def _install_sequence_padding(
    root: nn.Module, plan: Mapping[str, ParallelStyle], tp_mesh: DeviceMesh
) -> int:
    """Install every :class:`SequencePadding` after the scatters and guards it precedes."""
    installed = 0
    for path, style in plan.items():
        for leaf in _leaf_styles(style):
            if not isinstance(leaf, SequencePadding):
                continue
            if "*" in path:
                raise ValueError(
                    f"{type(leaf).__name__} pads one module's streams and needs a concrete plan "
                    f"path, received {path!r}"
                )
            leaf.install(root.get_submodule(path), tp_mesh)
            installed += 1
    return installed


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
    padded = _install_sequence_padding(root, total, tp_mesh)
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
        "%d sequence padding owner(s) installed, %d collective output(s) settled.",
        tp_mesh.size(),
        len(total),
        adjusted,
        restored,
        guarded,
        padded,
        settled,
    )
    return total
