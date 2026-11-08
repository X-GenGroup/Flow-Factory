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

"""H3-owned native packed row cuts around sequence-parallel boundaries."""

from collections.abc import Mapping
from inspect import signature
from typing import Any

import torch
from diffusers.hooks import HookRegistry, ModelHook
from torch import nn

from ..parallelism.context import ContextParallelBoundary, ContextParallelLayout
from ..parallelism.hooks import register_context_input, register_context_output, require_native_hook


class H3ContextParallelLayout(ContextParallelLayout):
    """Stable identity for native packed video/text/audio rows."""

    PACKED_AUDIO_VIDEO_TEXT = "minimax_h3_packed_audio_video_text"


def validate_h3_context_parallel_config(
    config: Mapping[str, Any], degree: int, tp_size: int = 1
) -> None:
    """Reject incompatible native head, patch and packed-FFN layouts before weights."""
    heads = config.get("num_attention_heads")
    if type(heads) is not int or heads < 1 or heads % (degree * tp_size):
        raise ValueError(
            f"MiniMax H3 heads={heads} must be divisible by cp_size={degree} * tp_size={tp_size}"
        )
    width = config.get("ffn_dim")
    if type(width) is not int or width < 1 or width % tp_size:
        raise ValueError(
            f"MiniMax H3 SwiGLU ffn_dim={width} must be divisible by tp_size={tp_size}"
        )
    time_width = config.get("time_embed_dim")
    if type(time_width) is not int or time_width < 1 or time_width % tp_size:
        raise ValueError(
            f"MiniMax H3 time_embed_dim={time_width} must be divisible by tp_size={tp_size}"
        )
    if config.get("num_layers", 0) < 1 or config.get("num_refiner_layers", 0) < 1:
        raise ValueError("MiniMax H3 CP requires native main and text-refiner blocks")
    if (
        config.get("in_channels"),
        config.get("audio_in_channels"),
        tuple(config.get("patch_size", ())),
    ) != (24, 32, (1, 2, 2)):
        raise ValueError(
            "MiniMax H3 CP requires native video24/audio32 channels and patch_size=(1,2,2)"
        )
    head_dim, rope_dim = config.get("attention_head_dim"), config.get("rope_freq_dim")
    if (
        type(head_dim) is not int
        or type(rope_dim) is not int
        or rope_dim < 1
        or head_dim < 6 * rope_dim
    ):
        raise ValueError(
            f"MiniMax H3 rotary width={rope_dim} requires attention_head_dim={head_dim} >= six rotary frequencies"
        )


def validate_h3_context_parallel_module(module: nn.Module, degree: int, tp_size: int = 1) -> None:
    from diffusers import MiniMaxH3Transformer3DModel

    if not isinstance(module, MiniMaxH3Transformer3DModel):
        raise ValueError(
            f"MiniMax H3 CP requires MiniMaxH3Transformer3DModel, received {type(module).__name__}"
        )
    validate_h3_context_parallel_config(module.config, degree, tp_size)
    _validate_native_processors(module)


def _validate_native_processors(module: nn.Module) -> None:
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Attention,
        MiniMaxH3AttnProcessor,
    )

    from ._chunking import _ChunkedH3AttnProcessor
    from .dependency import require_minimax_h3_support

    native_dispatch = require_minimax_h3_support().dispatch_attention_fn
    for name, attention in module.named_modules():
        if not isinstance(attention, MiniMaxH3Attention):
            continue
        processor = attention.processor
        if type(processor) not in (MiniMaxH3AttnProcessor, _ChunkedH3AttnProcessor) or getattr(
            attention, "fused_projections", False
        ):
            raise ValueError(
                f"MiniMax H3 CP requires native unfused attention at {name}, received {type(processor).__name__}"
            )
        if (
            type(processor) is _ChunkedH3AttnProcessor
            and processor.flow_factory_dispatch_attention_fn is not native_dispatch
        ):
            raise ValueError(f"MiniMax H3 CP requires the native chunked dispatch at {name}")


def validate_h3_context_parallel_inputs(values: Mapping[str, Any], divisor: int) -> None:
    """Validate the full packed layout before any native CP/SP communication."""
    if type(divisor) is not int or divisor < 1:
        raise ValueError(f"MiniMax H3 CP padding divisor must be positive, received {divisor!r}")
    positions = values["position_ids"]
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("MiniMax H3 CP requires position_ids shaped (packed_rows,3)")
    count = positions.shape[0]
    if count < 1:
        raise ValueError(f"MiniMax H3 packed_rows={count} must be positive")
    indices = []
    for name, field, width, tag in (
        ("video", "hidden_states", 96, 0),
        ("audio", "audio_hidden_states", 32, 2),
        ("text", "encoder_hidden_states", None, 1),
    ):
        rows = values[name + "_indices"]
        tensor = values[field]
        if not isinstance(rows, torch.Tensor) or rows.dtype != torch.long or rows.ndim != 1:
            raise ValueError(f"MiniMax H3 CP {name}_indices must be a one-dimensional int64 tensor")
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or tensor.shape[:2] != (1, rows.numel())
            or (width is not None and tensor.shape[-1] != width)
        ):
            raise ValueError(
                f"MiniMax H3 CP {name} state must have B1 and {rows.numel()} rows, received {getattr(tensor,'shape',None)}"
            )
        indices.append(rows)
    all_indices = torch.cat(indices)
    if all_indices.numel() != count or not torch.equal(
        all_indices.sort().values, torch.arange(count, device=positions.device)
    ):
        raise ValueError(
            "MiniMax H3 CP modality indices must partition the full packed sequence exactly once"
        )
    tags = values["token_tags"]
    time_indices = values["timestep_indices"]
    times = values["timestep"]
    if (
        tags.dtype != torch.long
        or tags.shape != (count,)
        or time_indices.dtype != torch.long
        or time_indices.shape != (count,)
    ):
        raise ValueError(
            "MiniMax H3 CP requires one int64 modality and timestep index per packed row"
        )
    for rows, tag in zip(indices[:2], (0, 2)):
        if not bool((tags.index_select(0, rows) == tag).all()):
            raise ValueError("MiniMax H3 CP modality tags disagree with their packed indices")
    # Native VL presentation tags vision rows as video even though the text
    # projection owns those embeddings; audio references contribute text labels.
    text_tags = tags.index_select(0, indices[2])
    if not bool(((text_tags == 0) | (text_tags == 1)).all()):
        raise ValueError("MiniMax H3 CP conditioner rows must carry native text/vision tags")
    if (
        times.ndim != 1
        or not times.is_floating_point()
        or not bool(torch.isfinite(times).all())
        or not bool(((times >= 0) & (times <= 1)).all())
        or not bool(((time_indices >= 0) & (time_indices < times.numel())).all())
    ):
        raise ValueError(
            "MiniMax H3 CP row timestep indices must address finite native times in [0,1]"
        )


def _pad_rows(value: torch.Tensor, divisor: int, axis: int, fill: int = 0) -> torch.Tensor:
    extra = -value.shape[axis] % divisor
    if not extra:
        return value
    shape = list(value.shape)
    shape[axis] = extra
    return torch.cat((value, value.new_full(shape, fill)), dim=axis)


def _argument(args, kwargs, name, index):
    return args[index] if len(args) > index else kwargs[name]


def _replace_argument(args, kwargs, name, index, value):
    if len(args) > index:
        args = list(args)
        args[index] = value
        return tuple(args), kwargs
    return args, {**kwargs, name: value}


class _H3RotaryPadding(ModelHook):
    def __init__(self, owner):
        self.owner = owner

    def post_forward(self, module, output):
        cos, sin = output
        return (
            _pad_rows(cos, self.owner._ff_h3_cp_divisor, 0, 1),
            _pad_rows(sin, self.owner._ff_h3_cp_divisor, 0),
        )


class _H3InputPadding(ModelHook):
    def __init__(self, owner, *, first=False, norm=False):
        self.owner, self.first, self.norm = owner, first, norm

    def pre_forward(self, module, *args, **kwargs):
        name = "timestep_indices" if self.norm else "adaln_indices"
        original = _argument(args, kwargs, name, 2)
        # This hook receives the original full per-row argument. In-forward
        # checkpointing snapshots its mask metadata before later root forwards.
        module._ff_h3_valid_rows = original.shape[0]
        args, kwargs = _replace_argument(
            args, kwargs, name, 2, _pad_rows(original, self.owner._ff_h3_cp_divisor, 0)
        )
        if self.first:
            hidden = _argument(args, kwargs, "hidden_states", 0)
            args, kwargs = _replace_argument(
                args, kwargs, "hidden_states", 0, _pad_rows(hidden, self.owner._ff_h3_cp_divisor, 1)
            )
        return args, kwargs


def configure_h3_context_parallel(module: nn.Module) -> None:
    """Pad rotary output before native CP slicing, without altering coordinates."""
    _validate_native_processors(module)
    HookRegistry.check_if_exists_or_initialize(module.rope).register_hook(
        _H3RotaryPadding(module), "ff_h3_rotary_padding"
    )


def install_h3_context_parallel(module: nn.Module, config: Any) -> None:
    """Keep native CP/autograd and mask padded rows after the text refiner."""
    if hasattr(module, "_ff_h3_cp_boundaries"):
        raise ValueError("MiniMax H3 CP hooks must be installed exactly once")
    _validate_native_processors(module)
    tp_size = module.config.num_attention_heads // module.transformer_blocks[0].attn.heads
    module._ff_h3_cp_divisor = config.ring_degree * config.ulysses_degree * tp_size
    module.enable_parallelism(config=config)
    for block in module.token_refiner.refiner_blocks:
        block.attn.processor._parallel_config = None
    call_signature = signature(module.forward)

    def validate(target, args, kwargs):
        values = call_signature.bind(*args, **kwargs).arguments
        validate_h3_context_parallel_inputs(values, target._ff_h3_cp_divisor)

    module.register_forward_pre_hook(validate, with_kwargs=True, prepend=True)
    module._ff_h3_padding_mask_handles = []
    for index, block in enumerate(module.transformer_blocks):
        HookRegistry.check_if_exists_or_initialize(block).register_hook(
            _H3InputPadding(module, first=index == 0), "ff_h3_input_padding"
        )

        def key_mask(attention, args, kwargs, block=block):
            hidden = _argument(args, kwargs, "hidden_states", 0)
            local_rows = hidden.shape[1]
            offset = config._ulysses_local_rank * local_rows
            mask = (
                torch.arange(local_rows, device=hidden.device) + offset < block._ff_h3_valid_rows
            )[None, None, None, :]
            if len(args) > 2:
                if args[2] is not None:
                    raise ValueError("MiniMax H3 CP owns the native padding key mask")
                return (*args[:2], mask, *args[3:]), kwargs
            if kwargs.get("attention_mask") is not None:
                raise ValueError("MiniMax H3 CP owns the native padding key mask")
            return args, {**kwargs, "attention_mask": mask}

        # TP's attention input gather was installed first. Native and the
        # existing Ref2VA chunked processor both forward this mask unchanged.
        module._ff_h3_padding_mask_handles.append(
            block.attn.register_forward_pre_hook(key_mask, with_kwargs=True)
        )
    HookRegistry.check_if_exists_or_initialize(module.norm_out).register_hook(
        _H3InputPadding(module, norm=True), "ff_h3_input_padding"
    )
    # Keep native CP outside the block body for both SP and in-forward checkpointing.
    compose_h3_sequence_boundaries(module)


def compose_h3_sequence_boundaries(module: nn.Module) -> tuple[ContextParallelBoundary, ...]:
    """Pad and cut native rows before SP; gather both heads after SP."""
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    installed = getattr(module, "_ff_h3_cp_boundaries", None)
    if installed is not None:
        for boundary in installed:
            hooks = (
                boundary.module._forward_pre_hooks
                if boundary.phase == "input"
                else boundary.module._forward_hooks
            )
            if hooks.get(boundary.handle.id) is not boundary.callback:
                raise ValueError("MiniMax H3 CP boundary installation was modified")
        return installed
    moves = []

    def checked(target, name, kind, metadata):
        binding = require_native_hook(
            target,
            name,
            kind,
            metadata=metadata,
            error=f"MiniMax H3 TP/SP+CP requires qualified native hook {name}",
        )
        return binding.registry, binding.hook

    for index, block in enumerate(module.transformer_blocks):
        registry, indices = checked(
            block,
            "cp_input---transformer_blocks.*",
            ContextParallelSplitHook,
            {"adaln_indices": ContextParallelInput(split_dim=0, expected_dims=1)},
        )
        hooks = []
        if index == 0:
            _, hidden = checked(
                block,
                "cp_input---transformer_blocks.0",
                ContextParallelSplitHook,
                {"hidden_states": ContextParallelInput(split_dim=1, expected_dims=3)},
            )
            hooks.append(("cp_input---transformer_blocks.0", hidden))
        hooks.append(("cp_input---transformer_blocks.*", indices))
        padding = registry.get_hook("ff_h3_input_padding")
        if not isinstance(padding, _H3InputPadding):
            raise ValueError("MiniMax H3 CP requires its native input padding owner")
        moves.append((block, registry, padding, hooks))
    registry, indices = checked(
        module.norm_out,
        "cp_input---norm_out",
        ContextParallelSplitHook,
        {"timestep_indices": ContextParallelInput(split_dim=0, expected_dims=1)},
    )
    padding = registry.get_hook("ff_h3_input_padding")
    if not isinstance(padding, _H3InputPadding):
        raise ValueError("MiniMax H3 CP requires its output-norm padding owner")
    moves.append((module.norm_out, registry, padding, [("cp_input---norm_out", indices)]))
    gathers = []
    for name in ("proj_out", "audio_proj_out"):
        target = getattr(module, name)
        registry, hook = checked(
            target,
            "cp_output---" + name,
            ContextParallelGatherHook,
            [ContextParallelOutput(gather_dim=1, expected_dims=3)],
        )
        gathers.append((target, registry, "cp_output---" + name, hook))
    boundaries = []
    # Validate the entire native layout before removing any hook.
    for target, registry, padding, hooks in moves:
        registry.remove_hook("ff_h3_input_padding", recurse=False)
        for name, hook in hooks:
            registry.remove_hook(name, recurse=False)

        boundaries.append(register_context_input(target, (padding, *(hook for _, hook in hooks))))
    for target, registry, name, hook in gathers:
        registry.remove_hook(name, recurse=False)

        boundaries.append(register_context_output(target, hook))
    module._ff_h3_cp_boundaries = tuple(boundaries)
    return module._ff_h3_cp_boundaries
