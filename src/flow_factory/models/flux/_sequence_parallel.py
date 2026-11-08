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

"""Flux-owned continuous image/text token shards and native attention ordering."""

import inspect
import math

import torch
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle, PrepareModuleOutput

from ..parallelism.context import ContextParallelBoundary
from ..parallelism.hooks import compose_context_output
from ..parallelism.sequence import SequenceBroadcastSubmodule


def _arguments(module: nn.Module, args: tuple, kwargs: dict) -> dict:
    return dict(inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments)


def _replace(module: nn.Module, args: tuple, kwargs: dict, changes: dict) -> tuple[tuple, dict]:
    positions = {
        name: index for index, name in enumerate(inspect.signature(module.forward).parameters)
    }
    args, kwargs = list(args), dict(kwargs)
    for name, value in changes.items():
        index = positions[name]
        if name in kwargs or index >= len(args):
            kwargs[name] = value
        else:
            args[index] = value
    return tuple(args), kwargs


def _interleave_streams(value: Tensor, text: int, degree: int, dim: int) -> Tensor:
    """Match the rank-major gather of independently sharded text/image streams."""
    text_values, image_values = value.split((text, value.shape[dim] - text), dim=dim)
    return torch.cat(
        [
            part
            for pair in zip(text_values.chunk(degree, dim), image_values.chunk(degree, dim))
            for part in pair
        ],
        dim=dim,
    )


class FluxSequenceParallelInput(ParallelStyle):
    """Pad text before native CP splitting and exclude padded keys from every attention.

    Padding changes only internal text rows. Original image/reference rows and
    their output order remain unchanged. Native attention consumes the boolean
    key mask, including the CP rank-major order of joint text/image keys.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

        for name, child in module.named_modules():
            if hasattr(child, "processor") and not isinstance(child.processor, FluxAttnProcessor):
                raise ValueError(
                    f"Flux SP requires native FluxAttnProcessor at {name}; "
                    f"received {type(child.processor).__name__}"
                )

        def enter(owner: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            call = _arguments(owner, args, kwargs)
            attention = call.get("joint_attention_kwargs") or {}
            unsupported = set(attention) - {"scale"}
            if unsupported:
                raise ValueError(
                    "Flux SP requires native dense attention without extra attention arguments; "
                    f"received {sorted(unsupported)}"
                )
            if (
                call.get("controlnet_block_samples") is not None
                or call.get("controlnet_single_block_samples") is not None
            ):
                raise ValueError("Flux SP does not support full-length ControlNet residuals")
            context = getattr(
                getattr(owner, "_parallel_config", None), "context_parallel_config", None
            )
            cp_size = 1 if context is None else context.ulysses_degree * context.ring_degree
            degree = cp_size * device_mesh.size()
            image, text = call["hidden_states"], call["encoder_hidden_states"]
            if image.shape[1] <= 0 or image.shape[1] % degree:
                raise ValueError(
                    f"Flux SP image tokens={image.shape[1]} must be positive and divisible by CP/SP degree={degree}"
                )
            length = text.shape[1]
            if length <= 0:
                raise ValueError("Flux SP requires a nonempty text stream")
            padded = math.ceil(length / degree) * degree
            if padded == length:
                return args, kwargs
            ids = call["txt_ids"]
            if ids.ndim != 2 or ids.shape != (length, 3):
                raise ValueError(
                    f"Flux SP txt_ids must have shape=({length}, 3), received {tuple(ids.shape)}"
                )
            text = torch.cat(
                [text, text.new_zeros(text.shape[0], padded - length, text.shape[2])], dim=1
            )
            ids = torch.cat([ids, ids.new_zeros(padded - length, 3)], dim=0)
            text_valid = torch.arange(padded, device=text.device) < length
            image_valid = torch.ones(image.shape[1], device=text.device, dtype=torch.bool)
            mask = torch.cat(
                [
                    part
                    for pair in zip(text_valid.chunk(cp_size), image_valid.chunk(cp_size))
                    for part in pair
                ]
            )[None, None, None]
            return _replace(
                owner,
                args,
                kwargs,
                {
                    "encoder_hidden_states": text,
                    "txt_ids": ids,
                    "joint_attention_kwargs": {**attention, "attention_mask": mask},
                },
            )

        module.register_forward_pre_hook(enter, with_kwargs=True)
        # Keep the activation boundary on the complete PEFT projection. The plan
        # independently assigns token-partial parameter gradients to its base/factors.
        PrepareModuleOutput(
            output_layouts=Shard(1),
            desired_output_layouts=Replicate(),
            use_local_output=True,
        )._apply(module.proj_out, device_mesh)
        return module


class FluxSingleStreamLayout(ParallelStyle):
    """Keep native local concatenation aligned with rotary rows and padded-key masks.

    The residual text/image streams already arrive sharded. Native concatenation
    builds [local text, local image]; gathering that input for attention or the
    MLP produces rank-major pairs. Their shared packed output reduce-scatters
    back to precisely that local order before the native split. No block output
    gather is needed, so residual shards persist through the complete stack.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        def gather(value: Tensor) -> Tensor:
            return (
                DTensor.from_local(value, device_mesh, [Shard(1)], run_check=False)
                .redistribute(placements=[Replicate()])
                .to_local()
                .clone()
            )

        def enter(owner: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            call = _arguments(owner, args, kwargs)
            degree = device_mesh.size()
            text = call["encoder_hidden_states"].shape[1] * degree
            image = call["hidden_states"].shape[1] * degree
            rotary = call.get("image_rotary_emb")
            changes = {}
            if rotary is not None:
                if not isinstance(rotary, tuple) or any(
                    table.shape[0] != text + image for table in rotary
                ):
                    raise ValueError(
                        "Flux SP rotary rows must match the CP-local text/image sequence"
                    )
                changes["image_rotary_emb"] = tuple(
                    _interleave_streams(table, text, degree, 0) for table in rotary
                )
            attention = call.get("joint_attention_kwargs") or {}
            mask = attention.get("attention_mask")
            if mask is not None:
                if mask.shape[-1] % (text + image):
                    raise ValueError(
                        "Flux SP key mask must contain complete CP-local text/image rows"
                    )
                parts = mask.split(text + image, dim=-1)
                changes["joint_attention_kwargs"] = {
                    **attention,
                    "attention_mask": torch.cat(
                        [_interleave_streams(part, text, degree, -1) for part in parts], dim=-1
                    ),
                }
            return _replace(owner, args, kwargs, changes)

        module.register_forward_pre_hook(enter, with_kwargs=True)
        # Hook the complete PEFT wrapper: both base and adapter must consume the
        # same gathered input. ColwiseParallel owns this arm's input-gradient sum.
        module.proj_mlp.register_forward_pre_hook(
            lambda _owner, args: (gather(args[0]), *args[1:]), prepend=True
        )
        return module


def validate_flux_sequence_boundaries(module: nn.Module) -> tuple[ContextParallelBoundary, ...]:
    """Keep native image/text splitting before SP and output gathering after SP."""
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    return compose_context_output(
        module,
        input_path="",
        output_path="proj_out",
        expected_inputs={
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(
                split_dim=1, expected_dims=3, split_output=False
            ),
            "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        },
        expected_output=ContextParallelOutput(gather_dim=1, expected_dims=3),
        state_attribute="_ff_flux1_cp_sequence_boundary",
        error="Flux TP/SP+CP requires native root image/text split and proj_out gather",
    )


class FluxModulationOutput(SequenceBroadcastSubmodule):
    """Apply the shared conditioning boundary to this family's AdaLN linear."""

    def __init__(self) -> None:
        super().__init__("linear")
