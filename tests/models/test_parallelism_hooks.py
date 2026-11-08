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


"""Native/Torch boundary ordering, preflight and retained-handle ownership."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from diffusers.hooks.hooks import HookRegistry, ModelHook
from torch import nn

from flow_factory.models.parallelism.hooks import (
    register_context_input,
    require_native_hook,
)


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return x * self.weight


class NativeHook(ModelHook):
    def __init__(self, events, phase):
        self.events, self.phase = events, phase
        self.metadata = {"axis": 1}

    def pre_forward(self, module, *args, **kwargs):
        if self.phase == "input":
            self.events.append("cp-in")
            if args:
                return (args[0] + 1, *args[1:]), kwargs
            return args, {**kwargs, "x": kwargs["x"] + 1}
        return args, kwargs

    def post_forward(self, module, output):
        if self.phase == "output":
            self.events.append("cp-out")
            return output + 5
        return output


@pytest.mark.parametrize("keyword", [False, True])
def test_native_boundaries_surround_torch_hooks_and_preserve_autograd(keyword):
    events = []
    module = Scale()
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(NativeHook(events, "input"), "native-in")
    registry.register_hook(NativeHook(events, "output"), "native-out")

    def sp_input(owner, args, kwargs):
        events.append("sp-in")
        return ((args[0] * 2,), kwargs) if args else (args, {"x": kwargs["x"] * 2})

    def sp_output(owner, args, output):
        events.append("sp-out")
        return output * 3

    module.register_forward_pre_hook(sp_input, with_kwargs=True)
    module.register_forward_hook(sp_output)
    bindings = [
        require_native_hook(module, name, NativeHook, metadata={"axis": 1}, error="layout")
        for name in ("native-in", "native-out")
    ]
    boundaries = (bindings[0].move_input(), bindings[1].move_output())
    x = torch.tensor(4.0, requires_grad=True)
    result = module(x=x) if keyword else module(x)
    assert events == ["cp-in", "sp-in", "sp-out", "cp-out"]
    assert result.item() == 65
    result.backward()
    assert x.grad.item() == 12
    assert module.weight.grad.item() == 30
    for boundary in boundaries:
        hooks = module._forward_pre_hooks if boundary.phase == "input" else module._forward_hooks
        assert hooks[boundary.handle.id] is boundary.callback
    assert registry.get_hook("native-in") is None
    assert registry.get_hook("native-out") is None


@pytest.mark.parametrize("failure", ["missing", "type", "metadata"])
def test_preflight_rejects_bad_second_boundary_before_removing_first(failure):
    module = Scale()
    registry = HookRegistry.check_if_exists_or_initialize(module)
    original = NativeHook([], "input")
    registry.register_hook(original, "first")
    if failure != "missing":
        second = NativeHook([], "output") if failure == "metadata" else ModelHook()
        registry.register_hook(second, "second")
    with pytest.raises(ValueError, match="qualified layout"):
        [
            require_native_hook(
                module,
                name,
                NativeHook,
                metadata={"axis": 0 if name == "second" else 1},
                error="qualified layout",
            )
            for name in ("first", "second")
        ]
    assert registry.get_hook("first") is original
    assert not module._forward_pre_hooks


def test_a_stale_binding_cannot_remove_a_replacement_hook():
    module = Scale()
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(NativeHook([], "input"), "entry")
    binding = require_native_hook(module, "entry", NativeHook, error="layout")
    registry.remove_hook("entry", recurse=False)
    replacement = NativeHook([], "input")
    registry.register_hook(replacement, "entry")
    with pytest.raises(ValueError, match="changed before relocation"):
        binding.move_input()
    assert registry.get_hook("entry") is replacement


def test_composed_padding_and_split_retain_the_model_declared_order():
    module = Scale()
    events = []

    def padding(owner, *args, **kwargs):
        events.append("pad")
        return (torch.cat((args[0], args[0].new_zeros(1))),), kwargs

    def split(owner, *args, **kwargs):
        events.append("split")
        assert args[0].shape == (4,)
        return (args[0][-2:],), kwargs

    register_context_input(
        module, (SimpleNamespace(pre_forward=padding), SimpleNamespace(pre_forward=split))
    )
    assert torch.equal(module(torch.arange(3.0)), torch.tensor([4.0, 0.0]))
    assert events == ["pad", "split"]


@pytest.mark.parametrize(
    "legacy,owner,names",
    [
        (
            "tensor_parallel",
            "parallelism.sequence",
            ("SequenceParallelRegion", "SequenceBroadcastParameter"),
        ),
        ("tensor_parallel", "parallelism.runtime", ("apply_tensor_parallel",)),
        ("tensor_parallel", "parallelism.plan", ("resolve_parallel_plan", "build_total_plan")),
        (
            "context_parallel",
            "parallelism.context",
            ("PreparedContextParallel", "ContextParallelCapability"),
        ),
        (
            "packed_tensor_parallel",
            "parallelism.packed",
            ("PackedColwiseParallel", "PackedRowwiseParallel"),
        ),
    ],
)
def test_legacy_imports_resolve_to_the_same_types_and_functions(legacy, owner, names):
    old = importlib.import_module("flow_factory.models." + legacy)
    new = importlib.import_module("flow_factory.models." + owner)
    for name in names:
        assert getattr(old, name) is getattr(new, name)
