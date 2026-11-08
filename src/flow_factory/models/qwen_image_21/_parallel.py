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

"""Qwen-2.1 joint-token and differentiable prefix-cache parallel boundaries."""

from __future__ import annotations

import inspect
import math
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle

from ..context_parallel import ContextParallelBoundary
from ..tensor_parallel import (
    ComposedParallelStyle,
    ReplicateParallel,
    SequenceParallelRegion,
    _AllReduceGradient,
)


@dataclass(frozen=True)
class _Call:
    mode: str | None
    length: int


class _State:
    """Keep call metadata on one prepared model, including checkpoint re-entry."""

    def __init__(self) -> None:
        self.call: ContextVar[_Call | None] = ContextVar("qwen21_parallel_call", default=None)
        self.tokens: ContextVar[tuple] = ContextVar("qwen21_parallel_tokens", default=())
        self.tp_mesh = None
        self.cp_config = None

    @property
    def active(self) -> bool:
        call = self.call.get()
        if call is None:
            raise RuntimeError("Qwen-2.1 parallel operation has no owning model/block call")
        return call.mode != "extract"

    @property
    def degree(self) -> int:
        cp = 1 if self.cp_config is None else self.cp_config.ulysses_degree
        tp = 1 if self.tp_mesh is None else self.tp_mesh.size()
        return cp * tp


def _pad(value: torch.Tensor, dim: int, length: int, fill: int = 0) -> torch.Tensor:
    amount = length - value.shape[dim]
    if amount < 0:
        raise ValueError(f"cannot pad Qwen-2.1 axis {dim} from {value.shape[dim]} to {length}")
    if not amount:
        return value
    shape = list(value.shape)
    shape[dim] = amount
    return torch.cat([value, value.new_full(shape, fill)], dim=dim)


def _arguments(module: nn.Module, args: tuple, kwargs: dict) -> dict:
    return dict(
        inspect.signature(type(module).forward).bind_partial(module, *args, **kwargs).arguments
    )


def _replace(module: nn.Module, args: tuple, kwargs: dict, changes: dict) -> tuple[tuple, dict]:
    names = list(inspect.signature(type(module).forward).parameters)[1:]
    args, kwargs = list(args), dict(kwargs)
    for name, value in changes.items():
        index = names.index(name)
        if name in kwargs or index >= len(args):
            kwargs[name] = value
        else:
            args[index] = value
    return tuple(args), kwargs


def _install_scope(module: nn.Module, state: _State) -> None:
    if hasattr(module, "_ff_qwen21_parallel_state"):
        if module._ff_qwen21_parallel_state is not state:
            raise ValueError("Qwen-2.1 parallel scopes must share one model owner")
        return
    for child in module.modules():
        child._ff_qwen21_parallel_state = state
    original_forward = module.forward

    @wraps(original_forward)
    def forward(*args, **kwargs):
        call = _arguments(module, args, kwargs)
        mode = call.get("kv_cache_mode")
        if mode == "cached":
            length = math.prod(call["img_shapes"][0][-1])
        else:
            mask = call["img_mask"][0]
            length = int((mask.to(torch.int64) * 3 + 1).sum())
        token = state.call.set(_Call(mode, length))
        try:
            output = original_forward(*args, **kwargs)
            if mode != "extract":
                if isinstance(output, tuple):
                    output = (output[0][:, :length], *output[1:])
                else:
                    output.sample = output.sample[:, :length]
            return output
        finally:
            state.call.reset(token)

    module.forward = forward
    for block in module.transformer_blocks:

        def enter(owner, args, kwargs):
            call = _arguments(owner, args, kwargs)
            mask = call.get("target_token_mask")
            if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
                raise ValueError(
                    "Qwen-2.1 parallel execution requires its native causal token mask"
                )
            token = state.call.set(_Call(call.get("kv_cache_mode"), mask.numel()))
            state.tokens.set((*state.tokens.get(), token))

        def leave(owner, args, output):
            tokens = state.tokens.get()
            if tokens:
                state.call.reset(tokens[-1])
                state.tokens.set(tokens[:-1])

        block._ff_qwen21_scope_enter = block.register_forward_pre_hook(
            enter, with_kwargs=True, prepend=True
        )
        block.register_forward_hook(leave, always_call=True)


def _prepare_metadata(module: nn.Module, args: tuple, kwargs: dict, state: _State, *, hidden: bool):
    call = _arguments(module, args, kwargs)
    if call.get("kv_cache_mode") == "extract":
        return args, kwargs
    mask = call["target_token_mask"]
    length = math.ceil(mask.numel() / state.degree) * state.degree
    changes = {"target_token_mask": _pad(mask, 0, length)}
    if hidden:
        changes["hidden_states"] = _pad(call["hidden_states"], 1, length)
    if "rotary_emb" in call:
        changes["rotary_emb"] = _pad(call["rotary_emb"], 0, length, 1)
    return _replace(module, args, kwargs, changes)


class _Root(ParallelStyle):
    def _apply(self, module, device_mesh):
        state = _State()
        state.tp_mesh = device_mesh
        configure_processors(module, state=state)
        return module


class _Region(SequenceParallelRegion):
    def __init__(self):
        super().__init__(gather=("hidden_states",))

    def _apply(self, module, device_mesh):
        return SequenceParallelRegion(
            gather=("hidden_states",),
            gather_when=lambda call: module._ff_qwen21_parallel_state.active,
        )._apply(module, device_mesh)


class _Input(ParallelStyle):
    def __init__(self, *, first: bool = False, final: bool = False) -> None:
        self.first, self.final = first, final

    def _apply(self, module, device_mesh):
        def move(owner, args, kwargs):
            state = module._ff_qwen21_parallel_state
            if not state.active:
                return None
            if not self.final and state.cp_config is None:
                args, kwargs = _prepare_metadata(owner, args, kwargs, state, hidden=self.first)
            call = _arguments(owner, args, kwargs)
            mask = call["target_token_mask"]
            if self.final and state.cp_config is None:
                mask = _pad(mask, 0, math.ceil(mask.numel() / state.degree) * state.degree)
            changes = {
                "target_token_mask": DTensor.from_local(
                    mask, device_mesh, [Replicate()], run_check=False
                )
                .redistribute(placements=[Shard(0)])
                .to_local()
            }
            if self.first:
                value = call["hidden_states"]
                changes["hidden_states"] = (
                    DTensor.from_local(value, device_mesh, [Replicate()], run_check=False)
                    .redistribute(placements=[Shard(1)])
                    .to_local()
                )
            return _replace(owner, args, kwargs, changes)

        module.register_forward_pre_hook(move, with_kwargs=True)
        return module


class _Output(ParallelStyle):
    def __init__(self, *, gather: bool = False) -> None:
        self.gather = gather

    def _apply(self, module, device_mesh):
        def move(value):
            source, target = (Shard(1), Replicate()) if self.gather else (Replicate(), Shard(1))
            return (
                DTensor.from_local(value, device_mesh, [source], run_check=False)
                .redistribute(placements=[target])
                .to_local()
            )

        if self.gather:

            def before(owner, args):
                return (
                    (move(args[0]), *args[1:]) if module._ff_qwen21_parallel_state.active else None
                )

            module.register_forward_pre_hook(before)
        else:

            def after(owner, args, output):
                return move(output) if module._ff_qwen21_parallel_state.active else output

            module.register_forward_hook(after)
        return module


class _Broadcast(ParallelStyle):
    def _apply(self, module, device_mesh):
        def after(owner, args, output):
            return (
                _AllReduceGradient.apply(output, device_mesh)
                if module._ff_qwen21_parallel_state.active
                else output
            )

        module.register_forward_hook(after)
        return module


def sequence_parallel_plan() -> dict[str, ParallelStyle]:
    """Shard residuals while retaining ordinary TP projection and LoRA ownership."""
    region = lambda: ComposedParallelStyle(_Region(), _Output())
    return {
        "": _Root(),
        "modulation": _Broadcast(),
        "transformer_blocks.*": _Input(),
        "transformer_blocks.0": _Input(first=True),
        "transformer_blocks.*.attn": region(),
        "transformer_blocks.*.img_mlp": region(),
        "norm_out": _Input(final=True),
        "norm_out.linear": ComposedParallelStyle(ReplicateParallel(), _Broadcast()),
        "proj_out": ComposedParallelStyle(ReplicateParallel(), _Output(gather=True)),
    }


def joint_attention_mask(
    segments, key_valid, *, batch: int, length: int, actual_length: int, device
):
    """Materialize the native text-causal/image-block mask, excluding internal padding."""
    mask = torch.zeros(length, length, dtype=torch.bool, device=device)
    prefix = 0
    for start, end, is_text in segments:
        if start != prefix or not start < end <= actual_length:
            raise ValueError(f"invalid Qwen-2.1 prefix segment {(start, end, is_text)}")
        mask[start:end, :end] = True
        if is_text:
            mask[start:end, start:end] = torch.ones(
                end - start, end - start, dtype=torch.bool, device=device
            ).tril()
        prefix = end
    mask[prefix:actual_length, :actual_length] = True
    # Padding queries have no loss; give them one valid key to avoid empty softmax rows.
    mask[actual_length:, 0] = True
    mask = mask[None, None].expand(batch, 1, length, length)
    if key_valid is not None:
        mask = mask & _pad(key_valid.bool(), 1, length)[:, None, None, :]
    return mask


def configure_processors(module: nn.Module, *, state: _State | None = None) -> None:
    """Install the model-local processor without replacing native projection or cache owners."""
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

    for block in module.transformer_blocks:
        if type(block.attn.processor) not in (QwenImage21AttnProcessor, _ParallelProcessor):
            raise ValueError(
                "Qwen-2.1 parallelism requires its native segmented attention processor"
            )
    state = state or getattr(module, "_ff_qwen21_parallel_state", None) or _State()
    _install_scope(module, state)
    for block in module.transformer_blocks:
        if isinstance(block.attn.processor, _ParallelProcessor):
            if block.attn.processor.state is not state:
                raise ValueError("Qwen-2.1 processor belongs to a different prepared model")
            continue
        block.attn.set_processor(_ParallelProcessor(state))


class _ParallelProcessor:
    _attention_backend = "native"
    _parallel_config = None

    def __init__(self, state: _State) -> None:
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

        self.state = state
        self.serial = QwenImage21AttnProcessor()

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ):
        from diffusers.hooks.context_parallel import EquipartitionSharder
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage21 import _qwenimage21_prepare_qkv

        if kv_cache_mode == "extract":
            self.serial._attention_backend = self._attention_backend
            return self.serial(
                attn,
                hidden_states,
                attention_mask,
                rotary_emb,
                layer_cache,
                kv_cache_mode,
                cache_write_slice,
                segments,
                key_valid,
            )
        query, key, value, _ = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, None, None, None
        )
        cp = self.state.cp_config
        degree = 1 if cp is None else cp.ulysses_degree
        length = query.shape[1] * degree
        call = self.state.call.get()
        if call is None:
            raise RuntimeError("Qwen-2.1 attention lost its native call metadata")
        if kv_cache_mode == "cached":
            prefix_k, prefix_v = layer_cache.get()
            prefix_length = prefix_k.shape[1]
            prefix_padded = math.ceil(prefix_length / degree) * degree
            prefix_k, prefix_v = (_pad(x, 1, prefix_padded) for x in (prefix_k, prefix_v))
            if cp is not None:
                prefix_k, prefix_v = (
                    EquipartitionSharder.shard(x, 1, cp._flattened_mesh)
                    for x in (prefix_k, prefix_v)
                )
            key, value = torch.cat([prefix_k, key], dim=1), torch.cat([prefix_v, value], dim=1)
            valid = torch.ones(
                query.shape[0], prefix_length + call.length, dtype=torch.bool, device=query.device
            )
            if attention_mask is not None:
                valid = attention_mask[:, 0, 0].bool()
            prefix_valid = _pad(valid[:, :prefix_length], 1, prefix_padded)
            target_valid = _pad(valid[:, prefix_length:], 1, length)
            mask = torch.cat(
                [
                    torch.cat(pair, dim=1)
                    for pair in zip(
                        prefix_valid.chunk(degree, dim=1), target_valid.chunk(degree, dim=1)
                    )
                ],
                dim=1,
            )[:, None, None]
            if (
                cp is None
                and attention_mask is None
                and prefix_padded == prefix_length
                and length == call.length
            ):
                mask = None
        else:
            mask = joint_attention_mask(
                segments,
                key_valid,
                batch=query.shape[0],
                length=length,
                actual_length=call.length,
                device=query.device,
            )
        query_length = query.shape[1]
        if cp is not None and query_length != key.shape[1]:
            # Native Ulysses backward uses one sequence length for Q, K and V.
            # These extra queries have no output/loss; cached keys and their
            # gradients retain the exact native prefix semantics.
            query = _pad(query, 1, key.shape[1])
        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=mask,
            # Native segmented prefix attention owns its default backend; only
            # cached decode selects the configured backend outside CP.
            backend=(
                self._attention_backend if cp is not None or kv_cache_mode == "cached" else None
            ),
            parallel_config=self._parallel_config,
        )
        output = output[:, :query_length]
        return attn.to_out[1](attn.to_out[0](output.flatten(2, 3).type_as(query)))


def install_context_parallel(module: nn.Module, config: Any) -> None:
    """Retain native Ulysses, split/gather autograd and model-owned prefix boundaries."""
    from diffusers.hooks.context_parallel import ContextParallelSplitHook
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    state = module._ff_qwen21_parallel_state
    if config.ring_degree != 1:
        raise ValueError("Qwen-2.1 prefix/block-mask attention requires native Ulysses transport")
    state.cp_config = config
    plan = {
        f"transformer_blocks.{index}": {
            **(
                {"hidden_states": ContextParallelInput(split_dim=1, expected_dims=3)}
                if index == 0
                else {}
            ),
            "rotary_emb": ContextParallelInput(split_dim=0, expected_dims=2),
            "target_token_mask": ContextParallelInput(split_dim=0, expected_dims=1),
        }
        for index in range(len(module.transformer_blocks))
    }
    plan["norm_out"] = {"target_token_mask": ContextParallelInput(split_dim=0, expected_dims=1)}
    plan["proj_out"] = ContextParallelOutput(gather_dim=1, expected_dims=3)
    module.enable_parallelism(config=config, cp_plan=plan)
    boundaries = []
    for index, block in enumerate(module.transformer_blocks):
        registry = block._diffusers_hook
        name = f"cp_input---transformer_blocks.{index}"
        native = registry.get_hook(name)
        if not isinstance(native, ContextParallelSplitHook):
            raise ValueError("Qwen-2.1 lost its native CP block boundary")
        registry.remove_hook(name, recurse=False)

        def split(owner, args, kwargs, native=native, first=index == 0):
            if _arguments(owner, args, kwargs).get("kv_cache_mode") == "extract":
                return args, kwargs
            args, kwargs = _prepare_metadata(owner, args, kwargs, state, hidden=first)
            return native.pre_forward(owner, *args, **kwargs)

        handle = block.register_forward_pre_hook(split, with_kwargs=True, prepend=True)
        boundaries.append(ContextParallelBoundary(block, "input", split, handle))
        # Scope entry must inspect the untouched native arguments, before CP and SP.
        scope_handle = block._ff_qwen21_scope_enter
        callback = block._forward_pre_hooks[scope_handle.id]
        scope_handle.remove()
        block._ff_qwen21_scope_enter = block.register_forward_pre_hook(
            callback, with_kwargs=True, prepend=True
        )

    norm = module.norm_out
    registry = norm._diffusers_hook
    native_norm = registry.get_hook("cp_input---norm_out")
    registry.remove_hook("cp_input---norm_out", recurse=False)

    def split_norm(owner, args, kwargs):
        if not state.active:
            return args, kwargs
        call = _arguments(owner, args, kwargs)
        mask = call["target_token_mask"]
        length = math.ceil(mask.numel() / state.degree) * state.degree
        args, kwargs = _replace(owner, args, kwargs, {"target_token_mask": _pad(mask, 0, length)})
        return native_norm.pre_forward(owner, *args, **kwargs)

    handle = norm.register_forward_pre_hook(split_norm, with_kwargs=True, prepend=True)
    boundaries.append(ContextParallelBoundary(norm, "input", split_norm, handle))
    output = module.proj_out
    registry = output._diffusers_hook
    native_gather = registry.get_hook("cp_output---proj_out")
    registry.remove_hook("cp_output---proj_out", recurse=False)

    def gather(owner, args, value):
        return native_gather.post_forward(owner, value) if state.active else value

    handle = output.register_forward_hook(gather)
    boundaries.append(ContextParallelBoundary(output, "output", gather, handle))
    module._ff_qwen21_cp_boundaries = tuple(boundaries)
