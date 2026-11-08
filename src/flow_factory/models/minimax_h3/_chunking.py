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

"""Bound peak memory for token-local MiniMax H3 operations."""

from __future__ import annotations

from types import MethodType
from typing import Callable, Iterable

import torch
from peft.tuners.lora.layer import Linear as LoraLinear
from torch import nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint

from .dependency import require_minimax_h3_support

H3_MAX_FEED_FORWARD_TOKENS = 1024
H3_MAX_ATTENTION_NORM_TOKENS = 1024
H3_MAX_LORA_PROJECTION_TOKENS = 1024
H3_MAX_ROTARY_TOKENS = 1024


def _checkpoint_h3_block_forward(self: nn.Module, *args, **kwargs):
    """Checkpoint one block after its FSDP cast and offload saved inputs to CPU."""
    original_forward = self.__dict__["flow_factory_uncheckpointed_forward"]
    if not torch.is_grad_enabled():
        return original_forward(self, *args, **kwargs)
    valid_rows = getattr(self, "_ff_h3_valid_rows", None)
    if valid_rows is not None:
        native_forward = original_forward

        def original_forward(owner, *inputs, **call_kwargs):
            # CP pre-hooks run before this checkpoint. Replay must use this call's
            # mask even when a later forward installed another packed length.
            previous = owner._ff_h3_valid_rows
            owner._ff_h3_valid_rows = valid_rows
            try:
                return native_forward(owner, *inputs, **call_kwargs)
            finally:
                owner._ff_h3_valid_rows = previous

    input_device_type = next(
        (argument.device.type for argument in args if isinstance(argument, torch.Tensor)),
        "cuda",
    )
    with save_on_cpu(
        pin_memory=input_device_type == "cuda" and torch.cuda.is_available(),
        device_type=input_device_type,
    ):
        return checkpoint(
            original_forward,
            self,
            *args,
            use_reentrant=False,
            preserve_rng_state=False,
            **kwargs,
        )


class _ChunkedFeedForward(nn.Module):
    """Run one existing feed-forward network over remainder-safe token chunks.

    The upstream Diffusers H3 blocks expose their feed-forward layers as a sole
    ``net`` ModuleList. Registering that same ModuleList directly keeps parameter
    identities and ``ff.net.*`` state-dict/LoRA paths unchanged.
    """

    def __init__(self, net: nn.ModuleList, *, max_tokens: int) -> None:
        super().__init__()
        self.net = net
        self.max_tokens = _positive_int(max_tokens, "max_tokens")

    def _forward_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the token-local network without requiring an even split."""
        if hidden_states.ndim < 3:
            raise ValueError(
                "MiniMax H3 feed-forward chunking expected [batch, tokens, hidden] "
                f"input, received shape={tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] <= self.max_tokens:
            return self._forward_chunk(hidden_states)
        if torch.is_grad_enabled():
            return _assemble_sequence_chunks(
                hidden_states,
                self.max_tokens,
                lambda chunk: checkpoint(
                    self._forward_chunk,
                    chunk,
                    use_reentrant=False,
                    preserve_rng_state=True,
                ),
                operation="feed-forward",
            )
        return _assemble_sequence_chunks(
            hidden_states,
            self.max_tokens,
            self._forward_chunk,
            operation="feed-forward",
        )


class _ChunkedRMSNorm(nn.RMSNorm):
    """Bound row-wise RMSNorm temporaries while preserving its parameter path."""

    def __init__(self, norm: nn.RMSNorm, *, max_tokens: int) -> None:
        nn.Module.__init__(self)
        self.normalized_shape = norm.normalized_shape
        self.eps = norm.eps
        self.elementwise_affine = norm.elementwise_affine
        self.register_parameter("weight", norm.weight)
        self.max_tokens = _positive_int(max_tokens, "max_tokens")

    def _forward_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.rms_norm(
            hidden_states,
            self.normalized_shape,
            self.weight,
            self.eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize independent token rows without materializing a full FP32 copy."""
        if hidden_states.ndim < 3:
            raise ValueError(
                "MiniMax H3 RMSNorm chunking expected [batch, tokens, ..., hidden] "
                f"input, received shape={tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] <= self.max_tokens:
            return self._forward_chunk(hidden_states)
        return _assemble_sequence_chunks(
            hidden_states,
            self.max_tokens,
            self._forward_chunk,
            operation="RMSNorm",
        )


class _ChunkedLoraLinear(LoraLinear):
    """Bound PEFT projection temporaries without replacing its module or state tree."""

    flow_factory_max_tokens: int

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Run the complete PEFT linear contract over sequence chunks."""
        if kwargs.get("alora_offsets") is not None:
            raise ValueError("MiniMax H3 LoRA projection chunking does not support aLoRA offsets")
        max_tokens = self.flow_factory_max_tokens
        if hidden_states.ndim < 3 or hidden_states.shape[1] <= max_tokens:
            return super().forward(hidden_states, *args, **kwargs)

        forward_chunk = super().forward
        return _assemble_sequence_chunks(
            hidden_states,
            max_tokens,
            lambda chunk: forward_chunk(chunk, *args, **kwargs),
            operation="LoRA projection",
        )


class _ChunkedH3AttnProcessor:
    """Preserve the upstream H3 attention contract with bounded rotary work."""

    _attention_backend = None
    _parallel_config = None
    flow_factory_max_tokens: int
    flow_factory_dispatch_attention_fn: Callable[..., torch.Tensor]

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_h3_rotary_chunks(
                query,
                *rotary_emb,
                max_tokens=self.flow_factory_max_tokens,
            )
            key = _apply_h3_rotary_chunks(
                key,
                *rotary_emb,
                max_tokens=self.flow_factory_max_tokens,
            )

        output_dtype = query.dtype
        hidden_states = self.flow_factory_dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        del query, key, value
        hidden_states = hidden_states.flatten(2, 3).to(dtype=output_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _apply_h3_rotary_chunks(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    max_tokens: int,
) -> torch.Tensor:
    """Apply row-local H3 rotary embedding without full-sequence temporaries."""
    max_tokens = _positive_int(max_tokens, "max_tokens")
    if hidden_states.ndim != 4:
        raise ValueError(
            "MiniMax H3 rotary embedding expected [batch, tokens, heads, head_dim], "
            f"received shape={tuple(hidden_states.shape)}"
        )
    if cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError(
            "MiniMax H3 rotary embedding expected matching [tokens, rotary_dim] cos/sin, "
            f"received cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )
    if cos.shape[0] != hidden_states.shape[1]:
        raise ValueError(
            "MiniMax H3 rotary embedding sequence mismatch, received "
            f"hidden_tokens={hidden_states.shape[1]}, rotary_tokens={cos.shape[0]}"
        )
    rotary_dim = cos.shape[-1]
    if rotary_dim < 2 or rotary_dim % 2 or rotary_dim > hidden_states.shape[-1]:
        raise ValueError(
            "MiniMax H3 rotary_dim must be positive, even, and no larger than head_dim, "
            f"received rotary_dim={rotary_dim}, head_dim={hidden_states.shape[-1]}"
        )

    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)
    if hidden_states.shape[1] <= max_tokens:
        return _apply_h3_rotary_chunk(hidden_states, cos, sin)

    cos_chunks = iter(cos.split(max_tokens, dim=0))
    sin_chunks = iter(sin.split(max_tokens, dim=0))
    return _assemble_sequence_chunks(
        hidden_states,
        max_tokens,
        lambda chunk: _apply_h3_rotary_chunk(chunk, next(cos_chunks), next(sin_chunks)),
        operation="rotary embedding",
    )


def _apply_h3_rotary_chunk(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the upstream rotate-half convention to one aligned token chunk."""
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    first, second = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-second, first), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


def _assemble_sequence_chunks(
    hidden_states: torch.Tensor,
    max_tokens: int,
    forward_chunk: Callable[[torch.Tensor], torch.Tensor],
    *,
    operation: str,
) -> torch.Tensor:
    """Write token-local chunk results into one final allocation.

    Under tensor parallelism a qk-norm receives a ``DTensor`` sharded on the head
    axis, and ``new_empty`` on one allocates a *replicated* tensor, so copying a
    chunk into it all-gathered every head. A ``DTensor`` chunk is therefore
    copied as its local shard into a local allocation, which is wrapped back
    with the chunks' placements once full; the token axis being chunked must
    not be one of them.
    """
    output = None
    layout = None
    offset = 0
    for chunk in hidden_states.split(max_tokens, dim=1):
        chunk_output = forward_chunk(chunk)
        if chunk_output.shape[:-1] != chunk.shape[:-1]:
            raise RuntimeError(
                f"MiniMax H3 {operation} must preserve input prefix dimensions, "
                f"received input={tuple(chunk.shape)}, output={tuple(chunk_output.shape)}"
            )
        if isinstance(chunk_output, DTensor):
            if layout is None:
                layout = _chunked_dtensor_layout(chunk_output, hidden_states.shape[1], operation)
            chunk_output = chunk_output.to_local()
        if output is None:
            output = chunk_output.new_empty(
                (chunk_output.shape[0], hidden_states.shape[1], *chunk_output.shape[2:])
            )
        output.narrow(1, offset, chunk_output.shape[1]).copy_(chunk_output)
        offset += chunk_output.shape[1]

    if output is None or offset != hidden_states.shape[1]:
        raise RuntimeError(
            f"MiniMax H3 {operation} failed to assemble every input token, "
            f"received input_tokens={hidden_states.shape[1]}, output_tokens={offset}"
        )
    if layout is None:
        return output
    mesh, placements, shape = layout
    return DTensor.from_local(
        output,
        mesh,
        placements,
        run_check=False,
        shape=shape,
        stride=torch.empty(shape, device="meta").stride(),
    )


def _chunked_dtensor_layout(chunk_output: DTensor, tokens: int, operation: str):
    """Return the mesh, placements and global shape of the assembled output."""
    for placement in chunk_output.placements:
        if isinstance(placement, Shard) and placement.dim % chunk_output.ndim == 1:
            raise ValueError(
                f"MiniMax H3 {operation} chunks the token axis, which this tensor is "
                f"sharded on: placements={chunk_output.placements}"
            )
    shape = torch.Size((chunk_output.shape[0], tokens, *chunk_output.shape[2:]))
    return chunk_output.device_mesh, chunk_output.placements, shape


def install_h3_feed_forward_chunking(
    transformer: nn.Module,
    *,
    max_tokens: int = H3_MAX_FEED_FORWARD_TOKENS,
) -> int:
    """Install bounded feed-forward execution on both H3 repeated stacks.

    Installation happens immediately after pretrained component materialization,
    before Flow-Factory resume loading, LoRA injection, gradient checkpointing, or
    distributed wrapping. The mutation is idempotent and preserves every parameter
    object.

    Args:
        transformer: Materialized H3 transformer containing both repeated block stacks.
        max_tokens: Positive maximum token count processed by one feed-forward invocation.

    Returns:
        Number of feed-forward layers configured across both stacks.

    Raises:
        TypeError: If the materialized upstream feed-forward structure or hooks are unsupported.
        ValueError: If ``max_tokens`` is invalid or conflicts with an earlier installation.
    """
    max_tokens = _positive_int(max_tokens, "max_tokens")
    blocks = tuple(_h3_repeated_blocks(transformer))
    configured = 0
    for name, block in blocks:
        feed_forward = getattr(block, "ff", None)
        if isinstance(feed_forward, _ChunkedFeedForward):
            if feed_forward.max_tokens != max_tokens:
                raise ValueError(
                    f"MiniMax H3 {name}.ff already uses max_tokens="
                    f"{feed_forward.max_tokens}, received conflicting {max_tokens}"
                )
            configured += 1
            continue
        net = getattr(feed_forward, "net", None)
        if not isinstance(net, nn.ModuleList):
            raise TypeError(
                f"MiniMax H3 {name}.ff expected a sole net ModuleList, received "
                f"{type(feed_forward).__name__} with net={type(net).__name__}"
            )
        if tuple(feed_forward._modules) != ("net",):
            raise TypeError(
                f"MiniMax H3 {name}.ff expected only the net child module, received "
                f"children={tuple(feed_forward._modules)}"
            )
        if not (
            len(net) == 3
            and type(net[0]).__name__ == "SwiGLU"
            and isinstance(net[1], nn.Dropout)
            and net[1].p == 0.0
            and isinstance(net[2], nn.Linear)
        ):
            raise TypeError(
                f"MiniMax H3 {name}.ff expected SwiGLU, Dropout(0), Linear; "
                f"received={[type(module).__name__ for module in net]}"
            )
        if tuple(feed_forward.named_parameters(recurse=False)) or tuple(
            feed_forward.named_buffers(recurse=False)
        ):
            raise TypeError(f"MiniMax H3 {name}.ff expected no direct parameters or buffers")
        _reject_execution_hooks(feed_forward, f"{name}.ff")
        replacement = _ChunkedFeedForward(net, max_tokens=max_tokens)
        replacement.train(feed_forward.training)
        block.ff = replacement
        configured += 1
    return configured


def install_h3_in_forward_block_checkpointing(transformer: nn.Module) -> int:
    """Place one checkpoint inside every repeated block's FSDP execution boundary.

    Diffusers model-level checkpointing surrounds the block call and therefore saves
    inputs before FSDP2 casts them for mixed-precision compute. Accelerate's generic
    FSDP2 policy checkpoints each direct child separately. This instance-local
    forward delegates the complete original block body to one non-reentrant
    checkpoint after the block's FSDP pre-forward hook has already run. Saved tensor inputs are
    offloaded to CPU until backward replay, using pinned host memory for CUDA inputs. In the
    intended BF16 FSDP2 path these are the post-cast BF16 block inputs.

    Args:
        transformer: Materialized H3 transformer containing both repeated block stacks.

    Returns:
        Number of repeated H3 blocks configured across both stacks.

    Raises:
        TypeError: If a block has an unsupported or conflicting forward installation.
    """
    blocks = tuple(_h3_repeated_blocks(transformer))
    pending = []
    for block_name, block in blocks:
        installed = block.__dict__.get("flow_factory_uncheckpointed_forward")
        shadowed_forward = block.__dict__.get("forward")
        if installed is not None:
            if not (
                installed is type(block).forward
                and isinstance(shadowed_forward, MethodType)
                and shadowed_forward.__self__ is block
                and shadowed_forward.__func__ is _checkpoint_h3_block_forward
            ):
                raise TypeError(
                    f"MiniMax H3 {block_name} has an inconsistent in-forward "
                    "checkpoint installation"
                )
            continue
        original_forward = type(block).forward
        # Removing the last Diffusers hook restores this exact native bound method
        # in the instance dictionary. Foreign wrappers still have a separate owner.
        if shadowed_forward is not None and not (
            isinstance(shadowed_forward, MethodType)
            and shadowed_forward.__self__ is block
            and shadowed_forward.__func__ is original_forward
        ):
            raise TypeError(
                f"MiniMax H3 {block_name} must not shadow forward before in-forward "
                "checkpoint installation"
            )
        if not callable(original_forward):
            raise TypeError(
                f"MiniMax H3 {block_name} expected a callable class forward, received "
                f"{original_forward!r}"
            )
        pending.append((block, original_forward))
    # Validate every repeated block before changing any checkpoint owner.
    for block, original_forward in pending:
        block.flow_factory_uncheckpointed_forward = original_forward
        block.forward = MethodType(_checkpoint_h3_block_forward, block)
    return len(blocks)


def install_h3_attention_norm_chunking(
    transformer: nn.Module,
    *,
    max_tokens: int = H3_MAX_ATTENTION_NORM_TOKENS,
) -> int:
    """Install bounded Q/K head normalization on both H3 repeated stacks.

    Args:
        transformer: Materialized H3 transformer containing both repeated block stacks.
        max_tokens: Positive maximum token count normalized by one local operation.

    Returns:
        Number of Q/K RMSNorm modules configured across both stacks.

    Raises:
        TypeError: If the materialized upstream attention or RMSNorm structure is unsupported.
        ValueError: If ``max_tokens`` is invalid or conflicts with an earlier installation.
    """
    max_tokens = _positive_int(max_tokens, "max_tokens")
    configured = 0
    for block_name, block in _h3_repeated_blocks(transformer):
        attention = getattr(block, "attn", None)
        if not isinstance(attention, nn.Module):
            raise TypeError(
                f"MiniMax H3 {block_name}.attn expected nn.Module, received "
                f"{type(attention).__name__}"
            )
        for norm_name in ("norm_q", "norm_k"):
            path = f"{block_name}.attn.{norm_name}"
            norm = getattr(attention, norm_name, None)
            if isinstance(norm, _ChunkedRMSNorm):
                if norm.max_tokens != max_tokens:
                    raise ValueError(
                        f"MiniMax H3 {path} already uses max_tokens={norm.max_tokens}, "
                        f"received conflicting {max_tokens}"
                    )
                configured += 1
                continue
            if not isinstance(norm, nn.RMSNorm):
                raise TypeError(
                    f"MiniMax H3 {path} expected nn.RMSNorm, received " f"{type(norm).__name__}"
                )
            if tuple(norm.named_children()) or tuple(norm.named_buffers(recurse=False)):
                raise TypeError(f"MiniMax H3 {path} expected no child modules or direct buffers")
            expected_parameters = ("weight",) if norm.elementwise_affine else ()
            if (
                tuple(name for name, _ in norm.named_parameters(recurse=False))
                != expected_parameters
            ):
                raise TypeError(
                    f"MiniMax H3 {path} expected direct parameters={expected_parameters}"
                )
            _reject_execution_hooks(norm, path)
            replacement = _ChunkedRMSNorm(norm, max_tokens=max_tokens)
            replacement.train(norm.training)
            setattr(attention, norm_name, replacement)
            configured += 1
    return configured


def install_h3_lora_projection_chunking(
    transformer: nn.Module,
    *,
    max_tokens: int = H3_MAX_LORA_PROJECTION_TOKENS,
    chunk_output_projection: bool = True,
) -> int:
    """Bound adapted H3 attention projections while preserving PEFT ownership.

    The installer runs after PEFT injection and before distributed preparation. It
    changes only the Python forward implementation on each existing PEFT Linear;
    parameters, children, hooks, and state-dict paths remain owned by that same
    module object.

    Args:
        transformer: Materialized H3 transformer after PEFT injection.
        max_tokens: Positive maximum token count projected by one adapted invocation.
        chunk_output_projection: Whether ``to_out.0`` is chunked too. Under
            sequence parallelism it reduce-scatters its output along the token
            axis, so a chunk would be scattered on its own and the pieces would
            interleave; it then has to run whole.

    Returns:
        Number of adapted Q/K/V/output projections configured across both stacks.

    Raises:
        TypeError: If the materialized attention or PEFT projection structure is unsupported.
        ValueError: If ``max_tokens`` is invalid or conflicts with an earlier installation.
    """
    max_tokens = _positive_int(max_tokens, "max_tokens")
    configured = 0
    for block_name, block in _h3_repeated_blocks(transformer):
        attention = getattr(block, "attn", None)
        if not isinstance(attention, nn.Module):
            raise TypeError(
                f"MiniMax H3 {block_name}.attn expected nn.Module, received "
                f"{type(attention).__name__}"
            )
        if getattr(attention, "fused_projections", False):
            raise TypeError(
                f"MiniMax H3 {block_name}.attn must install LoRA chunking before "
                "fusing its projections"
            )
        to_out = getattr(attention, "to_out", None)
        if not isinstance(to_out, nn.ModuleList) or len(to_out) != 2:
            raise TypeError(
                f"MiniMax H3 {block_name}.attn.to_out expected two-entry ModuleList, "
                f"received {type(to_out).__name__}"
            )
        projections = (
            ("to_q", getattr(attention, "to_q", None)),
            ("to_k", getattr(attention, "to_k", None)),
            ("to_v", getattr(attention, "to_v", None)),
            *((("to_out.0", to_out[0]),) if chunk_output_projection else ()),
        )
        for projection_name, projection in projections:
            path = f"{block_name}.attn.{projection_name}"
            if isinstance(projection, _ChunkedLoraLinear):
                if projection.flow_factory_max_tokens != max_tokens:
                    raise ValueError(
                        f"MiniMax H3 {path} already uses max_tokens="
                        f"{projection.flow_factory_max_tokens}, received conflicting "
                        f"{max_tokens}"
                    )
                configured += 1
                continue
            if not isinstance(projection, LoraLinear):
                continue
            if type(projection) is not LoraLinear:
                raise TypeError(
                    f"MiniMax H3 {path} expected the standard PEFT Linear before "
                    f"chunking, received {type(projection).__name__}"
                )
            if "forward" in projection.__dict__:
                raise TypeError(
                    f"MiniMax H3 {path} must not shadow PEFT Linear.forward on the instance"
                )
            if getattr(projection, "_compiled_call_impl", None) is not None:
                raise TypeError(
                    f"MiniMax H3 {path} must install LoRA chunking before torch.compile"
                )
            if type(getattr(projection, "base_layer", None)) is not nn.Linear:
                raise TypeError(
                    f"MiniMax H3 {path} expected an exact nn.Linear base layer, received "
                    f"{type(getattr(projection, 'base_layer', None)).__name__}"
                )
            if any(getattr(projection, "use_dora", {}).values()):
                raise TypeError(f"MiniMax H3 {path} LoRA chunking does not support DoRA")
            if getattr(projection, "lora_variant", {}):
                raise TypeError(f"MiniMax H3 {path} LoRA chunking supports only vanilla LoRA")
            for adapter_name, dropout in projection.lora_dropout.items():
                if isinstance(dropout, nn.Identity) or (
                    isinstance(dropout, nn.Dropout) and dropout.p == 0.0
                ):
                    continue
                raise TypeError(
                    f"MiniMax H3 {path} LoRA chunking requires zero dropout, received "
                    f"adapter={adapter_name!r}, dropout={dropout!r}"
                )
            _reject_execution_hooks(projection, path)
            projection.__class__ = _ChunkedLoraLinear
            projection.flow_factory_max_tokens = max_tokens
            configured += 1
    return configured


def install_h3_rotary_chunking(
    transformer: nn.Module,
    *,
    max_tokens: int = H3_MAX_ROTARY_TOKENS,
) -> int:
    """Install the bounded processor on every standard H3 attention instance.

    Args:
        transformer: Materialized H3 transformer containing both repeated block stacks.
        max_tokens: Positive maximum token count rotated by one local operation.

    Returns:
        Number of H3 attention processors configured across both stacks.

    Raises:
        ImportError: If the pinned MiniMax H3 attention processor is unavailable.
        TypeError: If the materialized attention processor structure is unsupported.
        ValueError: If ``max_tokens`` is invalid or conflicts with an earlier installation.
    """
    max_tokens = _positive_int(max_tokens, "max_tokens")
    symbols = require_minimax_h3_support()
    processor_class = symbols.MiniMaxH3AttnProcessor
    configured = 0
    for block_name, block in _h3_repeated_blocks(transformer):
        attention = getattr(block, "attn", None)
        if not isinstance(attention, nn.Module):
            raise TypeError(
                f"MiniMax H3 {block_name}.attn expected nn.Module, received "
                f"{type(attention).__name__}"
            )
        processor = getattr(attention, "processor", None)
        if isinstance(processor, _ChunkedH3AttnProcessor):
            if processor.flow_factory_max_tokens != max_tokens:
                raise ValueError(
                    f"MiniMax H3 {block_name}.attn.processor already uses max_tokens="
                    f"{processor.flow_factory_max_tokens}, received conflicting {max_tokens}"
                )
            configured += 1
            continue
        if type(processor) is not processor_class:
            raise TypeError(
                f"MiniMax H3 {block_name}.attn expected the standard attention processor, "
                f"received {type(processor).__name__}"
            )
        if "__call__" in processor.__dict__:
            raise TypeError(
                f"MiniMax H3 {block_name}.attn processor must not shadow __call__ on the instance"
            )
        attention_backend = getattr(processor, "_attention_backend", None)
        parallel_config = getattr(processor, "_parallel_config", None)
        processor.__class__ = _ChunkedH3AttnProcessor
        processor.flow_factory_max_tokens = max_tokens
        processor.flow_factory_dispatch_attention_fn = symbols.dispatch_attention_fn
        processor._attention_backend = attention_backend
        processor._parallel_config = parallel_config
        configured += 1
    return configured


def _h3_repeated_blocks(transformer: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    token_refiner = getattr(transformer, "token_refiner", None)
    stacks = (
        ("token_refiner.refiner_blocks", getattr(token_refiner, "refiner_blocks", None)),
        ("transformer_blocks", getattr(transformer, "transformer_blocks", None)),
    )
    for stack_name, stack in stacks:
        if not isinstance(stack, nn.ModuleList) or not stack:
            raise TypeError(
                f"MiniMax H3 expected non-empty {stack_name} ModuleList, received "
                f"{type(stack).__name__}"
            )
        for index, block in enumerate(stack):
            yield f"{stack_name}.{index}", block


def _reject_execution_hooks(module: nn.Module, path: str) -> None:
    hook_fields = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    active_hooks = tuple(field for field in hook_fields if getattr(module, field, None))
    if active_hooks or getattr(module, "_hf_hook", None) is not None:
        raise TypeError(
            f"MiniMax H3 {path} must be configured before execution hooks; "
            f"received hooks={active_hooks}, hf_hook="
            f"{type(getattr(module, '_hf_hook', None)).__name__}"
        )


def _positive_int(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"MiniMax H3 {field} expected a positive int, received {value!r}")
    return value
