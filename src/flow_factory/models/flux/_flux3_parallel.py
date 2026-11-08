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

"""Flux3-owned packed projections and native per-token sequence boundaries."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

from ...utils.dist import context_parallel_geometry, tensor_parallel_geometry
from ..context_parallel import (
    ContextParallelBoundary,
    ContextParallelCapability,
    ContextParallelLayout,
)
from ..tensor_parallel import (
    ColumnParallelRegion,
    PackedColwiseParallel,
    SequenceBroadcastParameter,
    SequenceParallelRegion,
    ShardedInputReplicatedWeight,
    SharedInputColwiseParallel,
    sequence_parallel_enabled,
)

STACKS = (
    "mode_blocks.image",
    "mode_blocks.image_cond",
    "txt_mode_blocks",
    "single_transformer_blocks",
)


class Flux3TokenBoundary(ParallelStyle):
    """Align one block with the continuous stack's residual and modulation shards.

    Joint blocks concatenate their local text and content before attention. Their
    full rotary table follows that rank-major gather order. Early modulation is
    already sharded at its producer. Joint modulation stays complete at the
    native producer/concatenation boundary and is sliced at each consumer.
    """

    def __init__(self, *, joint: bool, scatter: bool, gather: bool) -> None:
        self.joint = joint
        self.scatter = scatter
        self.gather = gather

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        names = ("hidden_states", "temb_mod")
        if self.joint:
            names += ("encoder_hidden_states", "temb_mod_context")
        positions = {name: i for i, name in enumerate(inspect.signature(module.forward).parameters)}
        required = (*names, "image_rotary_emb")
        if any(name not in positions for name in required):
            raise ValueError(
                f"Flux3 token boundary cannot resolve {required} on {type(module).__name__}"
            )
        degree = device_mesh.size()

        def transfer(value, source, desired):
            if not isinstance(value, torch.Tensor) or isinstance(value, DTensor) or value.ndim != 3:
                raise ValueError("Flux3 SP requires local rank-three residual/modulation tensors")
            if isinstance(desired, Shard) and value.shape[1] % degree:
                raise ValueError(
                    f"Flux3 tokens={value.shape[1]} must be divisible by tp_size={degree}"
                )
            return (
                DTensor.from_local(value, device_mesh, [source], run_check=False)
                .redistribute(placements=[desired])
                .to_local()
                .clone()
            )

        def enter(_module, args, kwargs):
            args, kwargs = list(args), dict(kwargs)

            def get(name):
                if name in kwargs:
                    return kwargs[name]
                if positions[name] < len(args):
                    return args[positions[name]]
                raise ValueError(f"Flux3 SP missing native input {name}")

            def put(name, value):
                if name in kwargs:
                    kwargs[name] = value
                else:
                    args[positions[name]] = value

            if self.joint:
                text, content = get("encoder_hidden_states").shape[1], get("hidden_states").shape[1]
                if not self.scatter:
                    text, content = text * degree, content * degree
                if text % degree or content % degree:
                    raise ValueError(
                        f"Flux3 joint SP text={text}, content={content} must be divisible by tp_size={degree}"
                    )
                rotary = get("image_rotary_emb")
                if not isinstance(rotary, tuple) or len(rotary) != 2:
                    raise ValueError("Flux3 SP requires the native cosine/sine rotary pair")
                reordered = []
                for table in rotary:
                    if table.ndim != 2 or table.shape[0] != text + content:
                        raise ValueError(
                            "Flux3 rotary rows must match the complete text/content sequence"
                        )
                    text_parts, content_parts = table[:text].chunk(degree), table[text:].chunk(
                        degree
                    )
                    reordered.append(
                        torch.cat(
                            [value for pair in zip(text_parts, content_parts) for value in pair],
                            dim=0,
                        )
                    )
                put("image_rotary_emb", tuple(reordered))
            for name in names:
                residual = name in ("hidden_states", "encoder_hidden_states")
                if (residual and self.scatter) or (not residual and self.joint):
                    put(name, transfer(get(name), Replicate(), Shard(1)))
            return tuple(args), kwargs

        def leave(_module, _args, output):
            if isinstance(output, tuple):
                return tuple(transfer(value, Shard(1), Replicate()) for value in output)
            return transfer(output, Shard(1), Replicate())

        module.register_forward_pre_hook(enter, with_kwargs=True)
        if self.gather:
            module.register_forward_hook(leave)
        return module


class Flux3StackBoundary(ParallelStyle):
    """Scatter once and gather once around a native mode or joint block stack."""

    def __init__(self, *, joint: bool) -> None:
        self.joint = joint

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, nn.ModuleList) or not len(module):
            raise ValueError("Flux3 SP requires a nonempty native transformer stack")
        for index, block in enumerate(module):
            Flux3TokenBoundary(
                joint=self.joint, scatter=index == 0, gather=index == len(module) - 1
            )._apply(block, device_mesh)
        return module


class Flux3ModulationBoundary(PrepareModuleOutput):
    """Keep only local early-modulation storage throughout the repeated stack."""

    def __init__(self) -> None:
        super().__init__(
            output_layouts=Replicate(), desired_output_layouts=Shard(1), use_local_output=True
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        super()._apply(module, device_mesh)
        # Replicate -> Shard is a view; retaining it would pin the full producer
        # allocation even though every block consumes only this rank's rows.
        module.register_forward_hook(lambda _module, _args, output: output.clone())
        return module


class Flux3OutputBoundary(ParallelStyle):
    """Wrap output activation transfers around the complete native or PEFT layers.

    The target length was already checked by its early-modulation scatter.
    Native output normalization receives the same complete CP-local target rows
    and per-token timestep embedding after the root's reference crop.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        PrepareModuleInput(
            input_layouts=(Replicate(), Replicate()),
            desired_input_layouts=(Shard(1), Shard(1)),
            use_local_output=True,
        )._apply(module.norm_out, device_mesh)
        PrepareModuleOutput(
            output_layouts=Shard(1), desired_output_layouts=Replicate(), use_local_output=True
        )._apply(module.proj_out, device_mesh)
        module.proj_out.register_forward_hook(lambda _module, _args, output: output.clone())
        return module


def flux3_tensor_parallel_plan() -> dict[str, ParallelStyle]:
    plan = {}
    for stack in STACKS:
        plan[f"{stack}.*.attn"] = ColumnParallelRegion(inputs=("hidden_states",))
        for name in ("to_q", "to_k", "to_v"):
            plan[f"{stack}.*.attn.{name}"] = SharedInputColwiseParallel()
        plan[f"{stack}.*.attn.to_out.0"] = RowwiseParallel()
        for name in ("norm_q", "norm_k"):
            plan[f"{stack}.*.attn.{name}"] = ShardedInputReplicatedWeight()
        plan[f"{stack}.*.ff.linear_in"] = PackedColwiseParallel((1, 1))
        plan[f"{stack}.*.ff.linear_out"] = RowwiseParallel()
    return plan


def flux3_sequence_parallel_plan() -> dict[str, ParallelStyle]:
    """Keep residuals local through each repeated stack and after the target crop.

    Native mode-to-joint concatenation consumes complete CP-local streams.
    Early per-token modulation is sharded once at its producer; joint modulation
    remains a shared full side input because the native root concatenates target
    and reference modulation. Each joint block takes its matching local slice.
    No residual gather/scatter occurs between blocks in the same stack.
    """
    plan = {}
    for stack in STACKS:
        plan[stack] = Flux3StackBoundary(joint=stack == "single_transformer_blocks")
        plan[f"{stack}.*.attn"] = SequenceParallelRegion(gather=("hidden_states",))
        plan[f"{stack}.*.attn.to_out.0"] = RowwiseParallel(output_layouts=Shard(1))
        plan[f"{stack}.*.ff"] = PrepareModuleInput(
            input_layouts=Shard(1), desired_input_layouts=Replicate(), use_local_output=True
        )
        plan[f"{stack}.*.ff.linear_out"] = RowwiseParallel(output_layouts=Shard(1))
    plan["early_stream_modulations.*"] = Flux3ModulationBoundary()
    plan[""] = Flux3OutputBoundary()
    plan["norm_out.linear"] = SequenceBroadcastParameter()
    plan["proj_out"] = SequenceBroadcastParameter()
    return plan


def flux3_context_parallel_plan() -> dict[str, Any]:
    """Split native mode streams before per-token modulation and gather target output."""
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    inputs = {
        name: ContextParallelInput(split_dim=1, expected_dims=3, split_output=False)
        for name in ("hidden_states", "encoder_hidden_states", "image_cond_hidden_states")
    }
    inputs.update(
        {
            name: ContextParallelInput(split_dim=0, expected_dims=2, split_output=False)
            for name in ("img_ids", "txt_ids", "image_cond_ids")
        }
    )
    return {"": inputs, "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3)}


class Flux3ContextParallelLayout(ContextParallelLayout):
    THREE_MODE_IMAGE_TEXT_REFERENCE = "flux3_image_text_reference_modes"


class Flux3ParallelAdapterMixin:
    """Own native mode-stream CP and packed TP/SP without changing stored weights."""

    context_parallel_capability = ContextParallelCapability(
        layout=Flux3ContextParallelLayout.THREE_MODE_IMAGE_TEXT_REFERENCE, tensor_parallel=True
    )
    context_parallel_diffusers_revision = (
        "0.41.0.dev0",
        "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941",
    )
    tensor_parallel_head_config = ("transformer", ("num_attention_heads",))
    sequence_parallel_by_default = False

    def _assert_flux3_fsdp_schedule(self, values: tuple[int, ...], phase: str) -> None:
        """Reject different native call schedules across the prepared FSDP2 mesh."""
        accelerator = getattr(self, "accelerator", None)
        if not getattr(accelerator, "is_fsdp2", False) or accelerator.num_processes < 2:
            return
        import torch.distributed as dist

        mesh = getattr(accelerator, "torch_device_mesh", None)
        if mesh is None:
            groups = [None]
        else:
            dimensions = tuple(accelerator.parallelism_config.fsdp_dim_names)
            if not dimensions:
                raise ValueError("Flux3 FSDP2 schedule validation requires its prepared mesh axes")
            groups = [mesh[name].get_group() for name in dimensions]
        lower = torch.tensor(values, device=accelerator.device, dtype=torch.int64)
        upper = lower.clone()
        # Propagate over every FSDP axis before raising: a partial mismatch in one
        # group must not strand another group at its next collective. TP is excluded.
        for group in groups:
            dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(lower, upper):
            raise ValueError(
                f"Flux3 FSDP2 {phase} requires identical native module-call schedules "
                f"across FSDP peers; received bounds {lower.tolist()} to {upper.tolist()}. "
                "Align per-sample call counts, reference presence and CFG branches. "
                "Reference token counts may differ across logical DP peers."
            )

    @classmethod
    def _validate_context_parallel_config(
        cls, config: Mapping[str, Any], degree: int, tp_size: int = 1
    ) -> None:
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        if any(type(value) is not int or value <= 0 for value in (heads, head_dim)):
            raise ValueError(f"Flux3 heads={heads}, head_dim={head_dim} must be positive integers")
        if heads % (degree * tp_size):
            raise ValueError(
                f"Flux3 num_attention_heads={heads} must be divisible by cp_size={degree} * tp_size={tp_size}"
            )
        axes = config["axes_dims_rope"]
        if (
            len(axes) != 4
            or sum(axes) != head_dim
            or any(type(x) is not int or x <= 0 or x % 2 for x in axes)
        ):
            raise ValueError(
                f"Flux3 requires four positive even rotary axes summing to head_dim={head_dim}, received {axes}"
            )
        channels = config["in_channels"]
        if (
            type(channels) is not int
            or channels <= 0
            or channels % 4
            or config.get("cond_in_channels", 128) != channels
        ):
            raise ValueError(
                "Flux3 CP requires equal positive target/reference packed widths divisible by four"
            )
        if config.get("num_layers", 0) < 1 or config.get("num_single_layers", 0) < 1:
            raise ValueError("Flux3 CP requires all native per-mode and joint block stacks")
        if type(config.get("flow_map", False)) is not bool:
            raise ValueError(f"Flux3 flow_map must be boolean, received {config.get('flow_map')}")
        ratio = config.get("mlp_ratio", 3.0)
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or ratio <= 0
        ):
            raise ValueError(f"Flux3 mlp_ratio must be positive and finite, received {ratio}")
        if int(heads * head_dim * ratio) % tp_size:
            raise ValueError(
                f"Flux3 MLP width={int(heads*head_dim*ratio)} must be divisible by tp_size={tp_size}"
            )

    def _preflight_flux3_context_parallel(self) -> None:
        degree = self.config.parallelism_args.effective_context_parallel_size
        if degree == 1:
            return
        from diffusers import Flux3Transformer2DModel

        config = Flux3Transformer2DModel.load_config(
            self.model_args.model_name_or_path, subfolder="transformer"
        )
        self._validate_context_parallel_config(
            config, degree, self.config.parallelism_args.effective_tensor_parallel_size
        )

    @classmethod
    def validate_context_parallel_module(
        cls, module: nn.Module, degree: int, tp_size: int = 1
    ) -> None:
        """Reject unsupported native mode geometry and processors before installation."""
        from diffusers.models.transformers.transformer_flux3 import Flux3AttnProcessor

        from diffusers import Flux3Transformer2DModel

        if not isinstance(module, Flux3Transformer2DModel):
            raise ValueError(
                f"Flux3 CP requires native Flux3Transformer2DModel, received {type(module).__name__}"
            )
        cls._validate_context_parallel_config(module.config, degree, tp_size)
        for stack in STACKS:
            for block in module.get_submodule(stack):
                if type(block.attn.processor) is not Flux3AttnProcessor:
                    raise ValueError(
                        "Flux3 CP requires native dense Flux3AttnProcessor without caches or masks"
                    )

    @classmethod
    def install_context_parallel_module(cls, module: nn.Module, config: Any) -> None:
        """Install native CP using the model-owned stream and final-output plan."""
        module.enable_parallelism(config=config, cp_plan=flux3_context_parallel_plan())

    @classmethod
    def compose_context_parallel_sequence_boundaries(
        cls, module: nn.Module
    ) -> tuple[ContextParallelBoundary, ...]:
        """Retain native root splits and gather SP rows before the native CP output."""
        from diffusers.hooks.context_parallel import (
            ContextParallelGatherHook,
            ContextParallelSplitHook,
        )

        root = getattr(module, "_diffusers_hook", None)
        out = getattr(module.proj_out, "_diffusers_hook", None)
        split = root.get_hook("cp_input---") if root is not None else None
        installed = getattr(module, "_ff_flux3_cp_sequence_boundary", None)
        gather = out.get_hook("cp_output---proj_out") if out is not None else None
        if installed is not None:
            gather = installed[0]
        expected = flux3_context_parallel_plan()
        if (
            not isinstance(split, ContextParallelSplitHook)
            or split.metadata != expected[""]
            or not isinstance(gather, ContextParallelGatherHook)
            or gather.metadata != [expected["proj_out"]]
        ):
            raise ValueError(
                "Flux3 SP requires owned native root mode/ID splits and final target gather"
            )
        if installed is not None:
            boundary = installed[1]
            if (
                boundary.module is not module.proj_out
                or module.proj_out._forward_hooks.get(boundary.handle.id) is not boundary.callback
                or out is None
                or out.get_hook("cp_output---proj_out") is not None
            ):
                raise ValueError("Flux3 CP/SP lost its owned output gather boundary")
            return (boundary,)

        # Diffusers post_forward is inside forward; torch SP output hooks run
        # afterward. CP must gather only after those hooks restore its local rows.
        out.remove_hook("cp_output---proj_out", recurse=False)

        def gather_output(owner, args, output):
            return gather.post_forward(owner, output)

        boundary = ContextParallelBoundary(
            module.proj_out,
            "output",
            gather_output,
            module.proj_out.register_forward_hook(gather_output),
        )
        module._ff_flux3_cp_sequence_boundary = (gather, boundary)
        return (boundary,)

    def _validate_flux3_context_parallel_inputs(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_latents: torch.Tensor | None,
        image_latent_ids: torch.Tensor | None,
        attention_kwargs: Mapping[str, Any] | None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_text_ids: torch.Tensor | None = None,
    ) -> None:
        degree, _ = context_parallel_geometry(self.accelerator)
        if degree == 1:
            return
        if attention_kwargs:
            raise ValueError("Flux3 CP does not admit masks, caches or extra attention arguments")
        if t.numel() != 1:
            raise ValueError(f"Flux3 CP requires one sample timestep, received {tuple(t.shape)}")
        config = self.transformer_config
        tp_size, _ = tensor_parallel_geometry(self.accelerator)
        token_degree = degree * (
            tp_size
            if sequence_parallel_enabled(self, self.config.parallelism_args.sequence_parallel)
            else 1
        )

        def stream(name, value, ids, width):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or value.shape[0] != 1
                or value.shape[-1] != width
                or value.shape[1] <= 0
            ):
                raise ValueError(
                    f"Flux3 CP {name} must have shape (1, positive_tokens, {width}), received {getattr(value,'shape',None)}"
                )
            if value.shape[1] % token_degree:
                raise ValueError(
                    f"Flux3 CP {name} tokens={value.shape[1]} must be divisible by CP/SP degree={token_degree}"
                )
            if (
                not isinstance(ids, torch.Tensor)
                or ids.shape != (value.shape[1], 4)
                or not torch.isfinite(ids).all()
            ):
                raise ValueError(
                    f"Flux3 CP {name} IDs must be finite with shape ({value.shape[1]}, 4), received {getattr(ids,'shape',None)}"
                )

        stream("target", latents, latent_ids, config.in_channels)
        stream("text", prompt_embeds, text_ids, config.joint_attention_dim)
        if image_latents is not None:
            stream("reference", image_latents, image_latent_ids, config.cond_in_channels)
        elif image_latent_ids is not None:
            raise ValueError("Flux3 CP reference IDs require reference latents")
        if negative_prompt_embeds is not None:
            ids = negative_text_ids
            if ids is None:
                ids = self._prepare_text_ids(negative_prompt_embeds.shape[1], device=latents.device)
            else:
                ids = self._as_2d_position_ids(ids, "negative_text_ids")
            stream("negative text", negative_prompt_embeds, ids, config.joint_attention_dim)
        elif negative_text_ids is not None:
            raise ValueError("Flux3 CP negative text IDs require negative embeddings")
