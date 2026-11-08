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

"""Qwen-owned masked text padding and residual-stream SP entry."""

import inspect

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle, PrepareModuleOutput

from ..parallelism.sequence import SequenceBroadcastSubmodule


class QwenSequenceParallelInput(ParallelStyle):
    """Pad native text/masks, then scatter text only after native rotary construction.

    The first block's text boundary follows the optional native CP input split.
    Attention gathers both residual streams to their CP-local native order;
    masks and rotary tables therefore stay CP-local and replicated across TP.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenDoubleStreamAttnProcessor2_0,
        )

        for block in module.transformer_blocks:
            if not isinstance(block.attn.processor, QwenDoubleStreamAttnProcessor2_0):
                raise ValueError("Qwen SP requires native masked joint attention")
        signature = inspect.signature(module.forward)

        def enter(owner, args, kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            values = bound.arguments
            text, image = values.get("encoder_hidden_states"), values.get("hidden_states")
            if not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[1] < 1:
                raise ValueError("Qwen SP requires nonempty rank-3 text embeddings")
            mask = values.get("encoder_hidden_states_mask")
            if mask is not None and mask.shape != text.shape[:2]:
                raise ValueError("Qwen SP text mask must match original text embeddings")
            if (
                set(values.get("attention_kwargs") or {}) - {"scale"}
                or values.get("controlnet_block_samples") is not None
            ):
                raise ValueError(
                    "Qwen SP does not support extra attention arguments or ControlNet residuals"
                )
            cp = getattr(getattr(owner, "_parallel_config", None), "context_parallel_config", None)
            degree = device_mesh.size() * (cp.ulysses_degree if cp is not None else 1)
            if (
                not isinstance(image, torch.Tensor)
                or image.ndim != 3
                or image.shape[0] != text.shape[0]
                or image.shape[1] < 1
                or image.shape[1] % degree
            ):
                raise ValueError(
                    f"Qwen SP image token count must be positive and divisible by CP/SP degree={degree}"
                )
            padding = (-text.shape[1]) % degree
            if padding:
                if mask is None:
                    mask = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
                values["encoder_hidden_states"] = F.pad(text, (0, 0, 0, padding))
                values["encoder_hidden_states_mask"] = F.pad(mask, (0, padding), value=0)
            return bound.args, bound.kwargs

        first = module.transformer_blocks[0]
        block_signature = inspect.signature(first.forward)

        def scatter_text(owner, args, kwargs):
            bound = block_signature.bind_partial(*args, **kwargs)
            text = bound.arguments["encoder_hidden_states"]
            if text.shape[1] % device_mesh.size():
                raise ValueError("Qwen SP text must be padded before the native CP/SP cuts")
            bound.arguments["encoder_hidden_states"] = (
                DTensor.from_local(text, device_mesh, [Replicate()], run_check=False)
                .redistribute(placements=[Shard(1)])
                .to_local()
            )
            return bound.args, bound.kwargs

        module.register_forward_pre_hook(enter, with_kwargs=True)
        first.register_forward_pre_hook(scatter_text, with_kwargs=True)
        PrepareModuleOutput(
            output_layouts=Shard(1), desired_output_layouts=Replicate(), use_local_output=True
        )._apply(module.proj_out, device_mesh)
        return module


class QwenModulationOutput(SequenceBroadcastSubmodule):
    """Apply the shared conditioning boundary to this family's AdaLN linear."""

    def __init__(self) -> None:
        super().__init__("linear")
