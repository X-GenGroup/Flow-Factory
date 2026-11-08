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


"""Sequence regions, conditioning derivatives and activation placement styles."""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle

from .collectives import _AllReduceGradient, _GatherTokens, _RegionAllReduceGradient
from .tensor import _LocalViewStyle, _replicated_view, _ToLocalWithPartialGradient


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


class SequenceBroadcastSubmodule(ParallelStyle):
    """Sum token-partial derivatives after a complete child projection/PEFT branch.

    The containing module owns this activation boundary, leaving the child's
    parameter style independently derivable for LoRA. In particular, AdaLN
    scale/shift paths must see this reduction before splitting the linear output.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        SequenceBroadcastOutput()._apply(module.get_submodule(self.path), device_mesh)
        return module
