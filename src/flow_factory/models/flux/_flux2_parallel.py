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

"""Flux2-owned packed TP and continuous token shards within each native stack."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
)

from ...utils.dist import context_parallel_geometry, tensor_parallel_geometry
from ..parallelism.collectives import redistribute_sequence
from ..parallelism.context import (
    ContextParallelBoundary,
    ContextParallelCapability,
    ContextParallelLayout,
)
from ..parallelism.hooks import compose_context_output
from ..parallelism.packed import PackedColwiseParallel, PackedRowwiseParallel
from ..parallelism.plan import ComposedParallelStyle, sequence_parallel_enabled
from ..parallelism.sequence import (
    ColumnParallelRegion,
    SequenceBroadcastOutput,
    SequenceBroadcastParameter,
    SequenceParallelRegion,
)
from ..parallelism.tensor import ShardedInputReplicatedWeight, SharedInputColwiseParallel


class Flux2PackedAttentionMetadata(ParallelStyle):
    """Keep native QKV/MLP split ratios coherent with localized attention heads."""

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        degree = device_mesh.size()
        previous = getattr(module, "_ff_flux2_mlp_tp_degree", None)
        if previous is not None:
            if previous != degree:
                raise ValueError(
                    f"Flux2 MLP was localized with tp_size={previous}, received {degree}"
                )
            return module
        width = getattr(module, "mlp_hidden_dim", None)
        if type(width) is not int or width <= 0 or width % degree:
            raise ValueError(
                f"Flux2 mlp_hidden_dim={width} must be positive and divisible by tp_size={degree}"
            )
        module.mlp_hidden_dim = width // degree
        module._ff_flux2_mlp_tp_degree = degree
        return module


class Flux2TokenBoundary(ParallelStyle):
    """Transfer named token inputs and optionally gather the module's outputs."""

    def __init__(
        self, names: tuple[str, ...], *, scatter: bool = False, gather_output: bool = False
    ) -> None:
        self.names = names
        self.scatter = scatter
        self.gather_output = gather_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        positions = {
            name: index for index, name in enumerate(inspect.signature(module.forward).parameters)
        }
        if any(name not in positions for name in self.names):
            raise ValueError(
                f"Flux2 token boundary cannot resolve {self.names} on {type(module).__name__}"
            )

        def transfer(value, source, desired):
            return redistribute_sequence(value, device_mesh, source, desired, owner="Flux2")

        def enter(_module, args, kwargs):
            args, kwargs = list(args), dict(kwargs)
            source, desired = (Replicate(), Shard(1)) if self.scatter else (Shard(1), Replicate())
            for name in self.names:
                if name in kwargs:
                    kwargs[name] = transfer(kwargs[name], source, desired)
                elif positions[name] < len(args):
                    args[positions[name]] = transfer(args[positions[name]], source, desired)
                else:
                    raise ValueError(f"Flux2 token boundary missing required input {name}")
            return tuple(args), kwargs

        def leave(_module, _args, output):
            if isinstance(output, tuple):
                return tuple(transfer(value, Shard(1), Replicate()) for value in output)
            return transfer(output, Shard(1), Replicate())

        if self.names:
            module.register_forward_pre_hook(enter, with_kwargs=True)
        if self.gather_output:
            module.register_forward_hook(leave)
        return module


class Flux2StackBoundary(ParallelStyle):
    """Retain token shards between blocks; restore canonical order at stack exit.

    Native root code joins the dual streams and later crops the joint stream.
    Those operations consume complete CP-local sequences, so each homogeneous
    stack has one scatter at entry and one gather at exit.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, nn.ModuleList) or not len(module):
            raise ValueError("Flux2 SP requires a nonempty native transformer stack")
        Flux2TokenBoundary(self.names, scatter=True)._apply(module[0], device_mesh)
        Flux2TokenBoundary((), gather_output=True)._apply(module[-1], device_mesh)
        return module


class Flux2OutputBoundary(ParallelStyle):
    """Install output activation transfers around complete native or PEFT modules."""

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        Flux2TokenBoundary(("x",), scatter=True)._apply(module.norm_out, device_mesh)
        SequenceBroadcastOutput()._apply(module.norm_out.linear, device_mesh)
        Flux2TokenBoundary((), gather_output=True)._apply(module.proj_out, device_mesh)
        return module


def flux2_tensor_parallel_plan() -> dict[str, ParallelStyle]:
    """Shard both native stacks while retaining canonical packed parameter state."""
    plan = {
        "transformer_blocks.*.attn": ColumnParallelRegion(
            inputs=("hidden_states", "encoder_hidden_states")
        ),
        "transformer_blocks.*.attn.to_out.0": RowwiseParallel(),
        "transformer_blocks.*.attn.to_add_out": RowwiseParallel(),
        "transformer_blocks.*.ff.linear_in": PackedColwiseParallel((1, 1)),
        "transformer_blocks.*.ff.linear_out": RowwiseParallel(),
        "transformer_blocks.*.ff_context.linear_in": PackedColwiseParallel((1, 1)),
        "transformer_blocks.*.ff_context.linear_out": RowwiseParallel(),
        "single_transformer_blocks.*.attn": Flux2PackedAttentionMetadata(),
        "single_transformer_blocks.*.attn.to_qkv_mlp_proj": PackedColwiseParallel(),
        "single_transformer_blocks.*.attn.to_out": PackedRowwiseParallel(),
    }
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        plan[f"transformer_blocks.*.attn.{name}"] = SharedInputColwiseParallel()
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        plan[f"transformer_blocks.*.attn.{name}"] = ShardedInputReplicatedWeight()
    for name in ("norm_q", "norm_k"):
        plan[f"single_transformer_blocks.*.attn.{name}"] = ShardedInputReplicatedWeight()
    return plan


def flux2_sequence_parallel_plan() -> dict[str, ParallelStyle]:
    """Keep residuals sharded within each stack and the final output region.

    Attention and FF projections gather their inputs. Native dual-to-joint
    concatenation and the final text crop retain canonical CP-local order.
    Per-sample modulation stays replicated with summed token gradients.
    """
    plan = {
        "transformer_blocks": Flux2StackBoundary(("hidden_states", "encoder_hidden_states")),
        "transformer_blocks.*.attn": SequenceParallelRegion(
            gather=("hidden_states", "encoder_hidden_states")
        ),
        "transformer_blocks.*.attn.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
        "transformer_blocks.*.attn.to_add_out": RowwiseParallel(output_layouts=Shard(1)),
        "single_transformer_blocks": Flux2StackBoundary(("hidden_states",)),
        "single_transformer_blocks.*.attn": ComposedParallelStyle(
            Flux2PackedAttentionMetadata(), Flux2TokenBoundary(("hidden_states",))
        ),
        "single_transformer_blocks.*.attn.to_out": PackedRowwiseParallel(output_layouts=Shard(1)),
        "": Flux2OutputBoundary(),
        "proj_out": SequenceBroadcastParameter(),
    }
    for stream in ("ff", "ff_context"):
        plan[f"transformer_blocks.*.{stream}"] = PrepareModuleInput(
            input_layouts=Shard(1), desired_input_layouts=Replicate(), use_local_output=True
        )
        plan[f"transformer_blocks.*.{stream}.linear_out"] = RowwiseParallel(output_layouts=Shard(1))
    for name in (
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "single_stream_modulation",
    ):
        plan[name] = SequenceBroadcastOutput()
    return plan


class Flux2ContextParallelLayout(ContextParallelLayout):
    """Native batched four-axis IDs over target/reference and text streams."""

    TARGET_REFERENCE_TEXT = "flux2_target_reference_text"


def validate_flux2_config(
    config: Mapping[str, Any], degree: int, tp_size: int, *, guidance_embeds: bool
) -> None:
    """Validate native metadata before weights and before installing CP hooks."""
    heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
    if heads % (degree * tp_size):
        raise ValueError(
            f"Flux2 num_attention_heads={heads} must be divisible by cp_size={degree} * tp_size={tp_size}"
        )
    if config.get("guidance_embeds", True) is not guidance_embeds:
        raise ValueError(
            f"Flux2 adapter requires guidance_embeds={guidance_embeds}, received {config.get('guidance_embeds')}"
        )
    axes = config["axes_dims_rope"]
    if (
        len(axes) != 4
        or sum(axes) != head_dim
        or any(type(size) is not int or size <= 0 or size % 2 for size in axes)
    ):
        raise ValueError(
            f"Flux2 requires four positive even rotary axes summing to head_dim={head_dim}, received {axes}"
        )
    channels = config["in_channels"]
    if (
        channels <= 0
        or channels % 4
        or (config.get("out_channels") or channels) != channels
        or config.get("patch_size", 1) != 1
    ):
        raise ValueError(
            "Flux2 requires patch_size=1 and equal positive packed input/output channels divisible by four"
        )
    if config.get("num_layers", 0) < 1 or config.get("num_single_layers", 0) < 1:
        raise ValueError("Flux2 requires both native dual and single transformer stacks")
    ratio = config.get("mlp_ratio")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio <= 0
    ):
        raise ValueError(f"Flux2 requires positive finite mlp_ratio, received {ratio}")
    width = int(heads * head_dim * ratio)
    if width % tp_size:
        raise ValueError(f"Flux2 MLP width={width} must be divisible by tp_size={tp_size}")


class Flux2ParallelAdapterMixin:
    """Shared native denoiser layout; adapters retain their own conditioning policy."""

    supports_sharded_target_materialization = True
    supports_checkpoint_slice_loading = True

    context_parallel_capability = ContextParallelCapability(
        layout=Flux2ContextParallelLayout.TARGET_REFERENCE_TEXT, tensor_parallel=True
    )
    tensor_parallel_head_config = ("transformer", ("num_attention_heads",))
    sequence_parallel_by_default = False
    _flux2_guidance_embeds: bool

    @classmethod
    def tensor_parallel_plan(cls) -> dict[str, ParallelStyle]:
        return flux2_tensor_parallel_plan()

    @classmethod
    def sequence_parallel_plan(cls) -> dict[str, ParallelStyle]:
        return flux2_sequence_parallel_plan()

    def _preflight_flux2_context_parallel(self) -> None:
        if self.config.parallelism_args.effective_context_parallel_size == 1:
            return
        from diffusers import Flux2Transformer2DModel

        config = Flux2Transformer2DModel.load_config(
            self.model_args.model_name_or_path, subfolder="transformer"
        )
        self._validate_context_parallel_config(
            config,
            self.config.parallelism_args.effective_context_parallel_size,
            self.config.parallelism_args.effective_tensor_parallel_size,
        )

    @classmethod
    def _validate_context_parallel_config(
        cls, config: Mapping[str, Any], degree: int, tp_size: int = 1
    ) -> None:
        validate_flux2_config(config, degree, tp_size, guidance_embeds=cls._flux2_guidance_embeds)

    @classmethod
    def validate_context_parallel_module(
        cls, module: nn.Module, degree: int, tp_size: int = 1
    ) -> None:
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2AttnProcessor,
            Flux2ParallelSelfAttnProcessor,
        )

        from diffusers import Flux2Transformer2DModel

        if not isinstance(module, Flux2Transformer2DModel):
            raise ValueError(
                f"Flux2 CP requires Flux2Transformer2DModel, received {type(module).__name__}"
            )
        cls._validate_context_parallel_config(module.config, degree, tp_size)
        for block in module.transformer_blocks:
            if type(block.attn.processor) is not Flux2AttnProcessor:
                raise ValueError(
                    "Flux2 CP requires native dense attention without reference KV caches"
                )
        for block in module.single_transformer_blocks:
            if type(block.attn.processor) is not Flux2ParallelSelfAttnProcessor:
                raise ValueError(
                    "Flux2 CP requires native dense single attention without reference KV caches"
                )

    @classmethod
    def compose_context_parallel_sequence_boundaries(
        cls, module: nn.Module
    ) -> tuple[ContextParallelBoundary, ...]:
        """Keep native stream splits before SP and the image gather after it."""
        from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

        return compose_context_output(
            module,
            input_path="",
            output_path="proj_out",
            expected_inputs={
                name: ContextParallelInput(split_dim=1, expected_dims=3, split_output=False)
                for name in ("hidden_states", "encoder_hidden_states", "img_ids", "txt_ids")
            },
            expected_output=ContextParallelOutput(gather_dim=1, expected_dims=3),
            state_attribute="_ff_flux2_cp_sequence_boundary",
            error="Flux2 SP requires native root stream/ID splits and final proj_out gather",
        )

    def _validate_flux2_context_parallel_inputs(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_latents: torch.Tensor | None,
        image_latent_ids: torch.Tensor | None,
        joint_attention_kwargs: dict | None,
        *,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_text_ids: torch.Tensor | None = None,
    ) -> None:
        degree, _ = context_parallel_geometry(self.accelerator)
        if degree == 1:
            return
        if joint_attention_kwargs:
            raise ValueError(
                "Flux2 CP does not admit attention masks, reference KV caches or extra attention arguments"
            )
        if t.numel() != 1:
            raise ValueError(f"Flux2 CP requires one sample timestep, received {tuple(t.shape)}")
        config = self.pipeline.transformer.config
        tp_size, _ = tensor_parallel_geometry(self.accelerator)
        token_degree = degree * (
            tp_size
            if sequence_parallel_enabled(self, self.config.parallelism_args.sequence_parallel)
            else 1
        )

        def stream(name, values, ids, width, *, divisible):
            if (
                not isinstance(values, torch.Tensor)
                or values.ndim != 3
                or values.shape[0] != 1
                or values.shape[-1] != width
                or values.shape[1] <= 0
            ):
                raise ValueError(
                    f"Flux2 CP {name} must have shape (1, positive_tokens, {width}), received {getattr(values, 'shape', None)}"
                )
            if not isinstance(ids, torch.Tensor) or ids.shape != (1, values.shape[1], 4):
                raise ValueError(
                    f"Flux2 CP {name} IDs must have shape (1, {values.shape[1]}, 4), received {getattr(ids, 'shape', None)}"
                )
            if divisible and values.shape[1] % token_degree:
                raise ValueError(
                    f"Flux2 CP {name} tokens={values.shape[1]} must be divisible by CP/SP degree={token_degree}"
                )

        stream("target", latents, latent_ids, config.in_channels, divisible=False)
        stream("text", prompt_embeds, text_ids, config.joint_attention_dim, divisible=True)
        tokens = latents.shape[1]
        if image_latents is not None:
            stream(
                "reference", image_latents, image_latent_ids, config.in_channels, divisible=False
            )
            tokens += image_latents.shape[1]
        elif image_latent_ids is not None:
            raise ValueError("Flux2 CP reference IDs require reference latents")
        if tokens % token_degree:
            raise ValueError(
                f"Flux2 CP combined target/reference tokens={tokens} must be divisible by CP/SP degree={token_degree}"
            )
        if negative_prompt_embeds is not None:
            stream(
                "negative text",
                negative_prompt_embeds,
                negative_text_ids,
                config.joint_attention_dim,
                divisible=True,
            )
        elif negative_text_ids is not None:
            raise ValueError("Flux2 CP negative text IDs require negative prompt embeddings")
