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


"""Validate native CP hooks and move their callbacks around Torch TP/SP hooks.

Families choose boundaries and metadata. These helpers only own callback
registration: validate all bindings before moving any, keep CP input before SP,
and restore SP outputs before CP. Native hooks retain communication/autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
from diffusers.hooks.hooks import HookRegistry, ModelHook
from torch import nn

from .context import ContextParallelBoundary


@dataclass(frozen=True)
class NativeContextHook:
    """A validated native hook at its original module and registry name."""

    module: nn.Module
    name: str
    registry: HookRegistry
    hook: ModelHook

    def remove(self) -> None:
        """Detach exactly the callback validated by this binding."""
        if self.registry.get_hook(self.name) is not self.hook:
            raise ValueError(f"native CP hook changed before relocation: {self.name}")
        self.registry.remove_hook(self.name, recurse=False)

    def move_input(self) -> ContextParallelBoundary:
        """Move native splitting before existing Torch input hooks."""
        self.remove()
        return register_context_input(self.module, (self.hook,))

    def move_output(self) -> ContextParallelBoundary:
        """Move native gathering after existing Torch output hooks."""
        self.remove()
        return register_context_output(self.module, self.hook)


def require_native_hook(
    module: nn.Module,
    name: str,
    kind: type,
    *,
    metadata: Any = None,
    error: str,
) -> NativeContextHook:
    """Validate a native owner without modifying any installed callback.

    ``metadata=None`` preserves type-only validation for families which do not
    declare exact metadata. Layout-owning callers should supply their native plan.
    """
    registry = getattr(module, "_diffusers_hook", None)
    hook = registry.get_hook(name) if registry is not None else None
    if not isinstance(hook, kind) or (metadata is not None and hook.metadata != metadata):
        raise ValueError(f"{error}; hook={name}")
    return NativeContextHook(module, name, registry, hook)


def register_context_input(module: nn.Module, hooks: Sequence[Any]) -> ContextParallelBoundary:
    """Run an ordered native padding/split chain before Torch's SP entry.

    Callers first validate and detach every original binding. The returned handle
    is retained by PreparedContextParallel, not hidden in a process-global registry.
    """
    hooks = tuple(hooks)
    if not hooks:
        raise ValueError("a context input boundary requires at least one native hook")

    def split(owner: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        for hook in hooks:
            args, kwargs = hook.pre_forward(owner, *args, **kwargs)
        return args, kwargs

    handle = module.register_forward_pre_hook(split, with_kwargs=True, prepend=True)
    return ContextParallelBoundary(module, "input", split, handle)


def register_context_output(module: nn.Module, hook: Any) -> ContextParallelBoundary:
    """Run the native CP output gather after Torch's SP output restoration."""

    def gather(owner: nn.Module, args: tuple, output: Any) -> Any:
        return hook.post_forward(owner, output)

    handle = module.register_forward_hook(gather)
    return ContextParallelBoundary(module, "output", gather, handle)


def compose_context_output(
    module: nn.Module,
    *,
    input_path: str,
    output_path: str,
    expected_inputs: Any,
    expected_output: Any,
    state_attribute: str,
    error: str,
) -> tuple[ContextParallelBoundary, ...]:
    """Retain a native split and move its output gather after SP exactly once.

    The family supplies both module paths and metadata. Other layouts can compose
    individual bindings when input padding or multiple outputs need distinct owners.
    """
    require_native_hook(
        module.get_submodule(input_path),
        f"cp_input---{input_path}",
        ContextParallelSplitHook,
        metadata=expected_inputs,
        error=error,
    )
    output = module.get_submodule(output_path)
    name = f"cp_output---{output_path}"
    installed = getattr(module, state_attribute, None)
    if installed is not None:
        gather, boundary = installed
        registry = getattr(output, "_diffusers_hook", None)
        if (
            not isinstance(gather, ContextParallelGatherHook)
            or gather.metadata != [expected_output]
            or boundary.module is not output
            or output._forward_hooks.get(boundary.handle.id) is not boundary.callback
            or registry is None
            or registry.get_hook(name) is not None
        ):
            raise ValueError(f"{error}; lost owned output gather boundary")
        return (boundary,)
    binding = require_native_hook(
        output, name, ContextParallelGatherHook, metadata=[expected_output], error=error
    )
    boundary = binding.move_output()
    setattr(module, state_attribute, (binding.hook, boundary))
    return (boundary,)
