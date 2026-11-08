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

"""SD3-owned masked text padding and complete-wrapper SP boundaries."""

import inspect

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle, PrepareModuleOutput

from ..tensor_parallel import SequenceBroadcastOutput
from ._context_parallel import SD35ContextAttnProcessor


class SD35ModulationOutput(ParallelStyle):
    """Complete token-partial gradients after the native/PEFT AdaLN linear merges."""

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        SequenceBroadcastOutput()._apply(module.linear, device_mesh)
        return module


class SD35SequenceParallelInput(ParallelStyle):
    """Pad text before CP/SP and preserve its original length for masked attention."""

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from diffusers.models.attention_processor import JointAttnProcessor2_0

        attentions = [
            attn
            for block in module.transformer_blocks
            for attn in (block.attn, block.attn2)
            if attn is not None
        ]
        if any(
            type(attn.processor) not in (JointAttnProcessor2_0, SD35ContextAttnProcessor)
            for attn in attentions
        ):
            raise ValueError("SD3 SP requires the native joint attention processor")
        for attn in attentions:
            previous = attn.processor
            processor = SD35ContextAttnProcessor()
            processor._attention_backend = getattr(previous, "_attention_backend", None)
            processor._parallel_config = getattr(previous, "_parallel_config", None)
            attn.set_processor(processor)
        module._ff_sd35_sp_degree = device_mesh.size()
        signature = inspect.signature(module.forward)

        def enter(owner, args, kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            values = bound.arguments
            text = values.get("encoder_hidden_states")
            image = values.get("hidden_states")
            if not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[1] < 1:
                raise ValueError("SD3 SP requires nonempty rank-3 text embeddings")
            attention = values.get("joint_attention_kwargs") or {}
            if (
                set(attention) - {"scale"}
                or values.get("block_controlnet_hidden_states") is not None
                or values.get("skip_layers")
            ):
                raise ValueError(
                    "SD3 SP does not support custom attention, ControlNet residuals or skipped blocks"
                )
            cp = getattr(getattr(owner, "_parallel_config", None), "context_parallel_config", None)
            degree = device_mesh.size() * (cp.ulysses_degree if cp is not None else 1)
            patch = owner.config.patch_size
            if (
                not isinstance(image, torch.Tensor)
                or image.ndim != 4
                or image.shape[0] != text.shape[0]
                or any(size % patch for size in image.shape[-2:])
                or (image.shape[-2] // patch) * (image.shape[-1] // patch) % degree
            ):
                raise ValueError(
                    f"SD3 SP requires patch-aligned image tokens divisible by CP/SP degree={degree}"
                )
            length = text.shape[1]
            values["encoder_hidden_states"] = F.pad(text, (0, 0, 0, (-length) % degree))
            values["joint_attention_kwargs"] = {**attention, "cp_text_tokens": length}
            return bound.args, bound.kwargs

        module.register_forward_pre_hook(enter, with_kwargs=True)
        PrepareModuleOutput(
            output_layouts=Shard(1), desired_output_layouts=Replicate(), use_local_output=True
        )._apply(module.proj_out, device_mesh)
        return module
