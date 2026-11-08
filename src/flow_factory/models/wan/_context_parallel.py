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

"""Wan-owned native context-parallel layout and TP/SP boundary composition."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..context_parallel import ContextParallelBoundary, ContextParallelLayout


class WanContextParallelLayout(ContextParallelLayout):
    """Stable identities for qualified Wan context-parallel layouts."""

    DENSE_VIDEO = "wan_dense_video"


def compose_wan_sequence_boundaries(module: nn.Module) -> tuple[ContextParallelBoundary, ...]:
    """Run native CP split before TP scatter, and native CP gather after TP gather.

    Diffusers wraps ``forward`` while PyTorch TP/SP uses module hooks. Without
    this ordering bridge, SP scatters before CP and restores tokens after CP,
    permuting video positions relative to the native CP rotary slices. Only the
    two boundary callbacks move; Diffusers still owns slicing, gather autograd,
    rotary positions and attention communication.
    """
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook

    entry, exit_module = module.blocks[0], module.proj_out
    split_registry = getattr(entry, "_diffusers_hook", None)
    gather_registry = getattr(exit_module, "_diffusers_hook", None)
    split = split_registry.get_hook("cp_input---blocks.0") if split_registry else None
    gather = gather_registry.get_hook("cp_output---proj_out") if gather_registry else None
    if not isinstance(split, ContextParallelSplitHook) or not isinstance(
        gather, ContextParallelGatherHook
    ):
        raise ValueError("Wan TP/SP+CP requires the qualified native input/output hooks")

    def split_input(target: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        return split.pre_forward(target, *args, **kwargs)

    def gather_output(target: nn.Module, args: tuple, output: Any) -> Any:
        return gather.post_forward(target, output)

    split_registry.remove_hook("cp_input---blocks.0", recurse=False)
    gather_registry.remove_hook("cp_output---proj_out", recurse=False)
    return (
        ContextParallelBoundary(
            entry,
            "input",
            split_input,
            entry.register_forward_pre_hook(split_input, with_kwargs=True, prepend=True),
        ),
        ContextParallelBoundary(
            exit_module,
            "output",
            gather_output,
            exit_module.register_forward_hook(gather_output),
        ),
    )
