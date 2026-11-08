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

"""Qwen-owned ordering of native CP callbacks around image/text SP."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from torch import nn

from ..parallelism.context import ContextParallelBoundary
from ..parallelism.hooks import register_context_input, register_context_output


def compose_qwen_sequence_boundaries(module: nn.Module) -> tuple[ContextParallelBoundary, ...]:
    """Split image tokens by CP before SP and restore them in reverse order.

    Qwen scatters image tokens at ``img_in`` and text at the first block after
    native rotary construction. Partition native block-entry metadata between
    those two boundaries. Per-block masks and edit modulation indices enter native CP
    before SP narrows the image index. Diffusers retains all slicing, gather
    autograd, rotary and attention communication; no gradient factor is added.
    """
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    entry, exit_module = module.transformer_blocks[0], module.proj_out
    split_name = "cp_input---transformer_blocks.0"
    arguments_name = "cp_input---transformer_blocks.*"
    gather_name = "cp_output---proj_out"
    split_registry = getattr(entry, "_diffusers_hook", None)
    gather_registry = getattr(exit_module, "_diffusers_hook", None)
    split = split_registry.get_hook(split_name) if split_registry else None
    gather = gather_registry.get_hook(gather_name) if gather_registry else None
    expected_split = {
        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
    }
    expected_arguments = {
        "modulate_index": ContextParallelInput(split_dim=1, expected_dims=2),
        "encoder_hidden_states_mask": ContextParallelInput(split_dim=1, expected_dims=2),
    }
    if (
        not isinstance(split, ContextParallelSplitHook)
        or split.metadata != expected_split
        or not isinstance(gather, ContextParallelGatherHook)
        or gather.metadata != [ContextParallelOutput(gather_dim=1, expected_dims=3)]
    ):
        raise ValueError(
            "Qwen TP/SP+CP requires the qualified native image/text input and output hooks"
        )
    argument_hooks = []
    for block in module.transformer_blocks:
        registry = getattr(block, "_diffusers_hook", None)
        hook = registry.get_hook(arguments_name) if registry else None
        if not isinstance(hook, ContextParallelSplitHook) or hook.metadata != expected_arguments:
            raise ValueError(
                "Qwen TP/SP+CP requires native per-block mask and modulation-index hooks"
            )
        argument_hooks.append((block, registry, hook))

    image_split = ContextParallelSplitHook(
        {0: replace(split.metadata["hidden_states"], split_output=True)}, split.parallel_config
    )
    text_split = ContextParallelSplitHook(
        {"encoder_hidden_states": split.metadata["encoder_hidden_states"]}, split.parallel_config
    )
    text_split.initialize_hook(entry)

    def split_image(target: nn.Module, args: tuple, output: Any) -> Any:
        return image_split.post_forward(target, output)

    # Validate every native owner above before changing any installed callback.
    split_registry.remove_hook(split_name, recurse=False)
    gather_registry.remove_hook(gather_name, recurse=False)
    boundaries = [
        ContextParallelBoundary(
            module.img_in,
            "output",
            split_image,
            module.img_in.register_forward_hook(split_image, prepend=True),
        ),
        register_context_input(entry, (text_split,)),
    ]
    for block, registry, hook in argument_hooks:
        registry.remove_hook(arguments_name, recurse=False)
        boundaries.append(register_context_input(block, (hook,)))
    boundaries.append(register_context_output(exit_module, gather))
    return tuple(boundaries)
