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


"""Validate and settle installed parallel regions and collective boundaries."""

from __future__ import annotations

import types
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
)

from .autocast import _native_norm_output_dtype
from .collectives import _passed_region_entry
from .plan import ComposedParallelStyle
from .sequence import SequenceBroadcastOutput, _RegionEntry
from .tensor import (
    ShardedAcrossHeadRMSNorm,
    SharedInputColwiseParallel,
    _across_head_normalize,
    _across_head_square_sum,
    _norm_accumulation_input,
    _OwnParameterStyle,
)


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
    """Find standard token-axis scatters before their plan entries are applied.

    Yield ``(path, argument, axis)`` for ``PrepareModuleInput`` and
    ``PrepareModuleOutput``, including their composed forms. A scatter requests
    a nonnegative-axis ``Shard`` absent from the source layout; an existing shard
    is a pass-through and must not be checked as a full global sequence.
    ``argument`` is a positional index, keyword name, or ``_Output(index)``.

    Read the plan before application because built-in row-parallel styles mutate
    their desired input layouts. Negative-axis feature/head placements are
    excluded by this helper's token-axis convention. Model-owned custom
    boundaries must validate their own conditional or padded token layouts.
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
    """Wait for pending activation collectives before outputs leave planned modules.

    A layout redistribution can expose an ``AsyncCollectiveTensor`` whose work
    registry retains its buffer until a consumer waits. Checkpoint recomputation
    may stop before that consumer; a discarded branch or raised forward can
    likewise leave the collective unconsumed.

    The hook checks a tensor output or each direct tuple member and triggers
    pending waits in place, returning ``None`` to retain their autograd
    metadata. Already-settled outputs require no wait. Recognized styles that
    cannot leave a pending redistribution are skipped; unknown styles receive
    the hook so their output boundary is protected.

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
    """Reject uneven token lengths at the standard scatter boundaries in a plan.

    Subsequent gathers reconstruct global shape from plain local tensors as
    ``local_length * tp_size``. Uneven shards would give peers different global
    lengths and incompatible collective sizes. Checking the complete sequence
    before scattering needs only a local shape modulo, with no collective.

    Both input and output boundaries are checked before their installed scatter
    hooks run. Model-owned custom boundaries are validated by their own owners.

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
    it; ``result`` retains that key, its version, summed statistic and shared
    accumulation view until ``norm_k`` claims it. Only tensors and ints are held,
    so a deep copy is unaffected, and the attention's ``always_call`` forward hook
    empties both on every exit -- including a raised forward and checkpoint
    recompute stopping early, either of which would otherwise leave a layer's
    activations alive until its next forward.
    """

    __slots__ = ("key", "result")

    def __init__(self) -> None:
        self.key: Optional[torch.Tensor] = None
        self.result: Optional[Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]] = None

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
        query_work = _norm_accumulation_input(hidden_states)
        key_work = _norm_accumulation_input(key)
        query_square_sum = _across_head_square_sum(query_work)
        key_square_sum = _across_head_square_sum(key_work)
        dtype = torch.promote_types(query_square_sum.dtype, key_square_sum.dtype)
        packed = torch.cat(
            [query_square_sum.reshape(-1).to(dtype), key_square_sum.reshape(-1).to(dtype)]
        )
        summed = differentiable_all_reduce(packed, group=group)
        split = query_square_sum.numel()
        query_total = summed[:split].reshape(query_square_sum.shape).to(query_square_sum.dtype)
        key_total = summed[split:].reshape(key_square_sum.shape).to(key_square_sum.dtype)
        pair.result = (key, key._version, key_total, key_work)
        return _across_head_normalize(query_work, query_total, q_size, q_epsilon, self.weight).to(
            _native_norm_output_dtype(hidden_states)
        )

    def paired_key(self: nn.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
        result = pair.result
        if result is not None and _same_key(result[0], result[1], hidden_states):
            pair.result = None
            return _across_head_normalize(result[3], result[2], k_size, k_epsilon, self.weight).to(
                _native_norm_output_dtype(hidden_states)
            )
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
    """Check plain tensor input values across TP peers before each root forward.

    The plan assumes every peer enters ``root`` with the same activations: its
    column-parallel layers take slices of them and its row-parallel all-reduces
    sum the slices back. Inputs that differ give an output matching neither
    peer's unsharded result, and nothing raises. This compares every plain
    tensor among the positional and keyword arguments by value, with a MAX and a
    MIN all-reduce of their packed int64 form, after a two-element signature
    exchange that keeps the packed sizes equal on every peer. All peers see the
    same reductions, so they raise together rather than leaving one waiting.

    The signature compares tensor count and total flattened element count;
    per-leaf shapes, paths and non-tensor arguments are not compared.

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
