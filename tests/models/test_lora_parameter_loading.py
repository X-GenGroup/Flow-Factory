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

"""Strict prepared LoRA value imports preserve parameter and teacher ownership."""

import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from flow_factory.models.abc import BaseAdapter, CheckpointEntry
from flow_factory.trainers.distillation.opd.common import _load_one_teacher


class TinyComponent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))

    def forward(self, value):
        return self.block(value)


def _component():
    return get_peft_model(TinyComponent(), LoraConfig(r=2, lora_alpha=2, target_modules=["0", "1"]))


def _owner(component):
    owner = SimpleNamespace(
        _named_parameters={},
        target_module_map={"transformer": ["0", "1"]},
        _require_component=lambda name: component,
        _unwrap=lambda module: module,
        _get_component_parameters=lambda names: [
            p for p in component.parameters() if p.requires_grad
        ],
        _checkpoint_base_role=lambda: "base",
        _checkpoint_entries=lambda path: [CheckpointEntry("transformer", "base", ".")],
        _resolve_checkpoint_path=lambda path: path,
        _prepare_parameter_write=lambda: None,
    )
    owner.add_named_parameters = MethodType(BaseAdapter.add_named_parameters, owner)
    owner.load_lora_parameter_values = MethodType(BaseAdapter.load_lora_parameter_values, owner)
    return owner


def _write_teacher(path, format):
    teacher = _component()
    with torch.no_grad():
        for index, parameter in enumerate(p for p in teacher.parameters() if p.requires_grad):
            parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) / 17 + index)
    expected = [p.detach().clone() for p in teacher.parameters() if p.requires_grad]
    if format == "peft":
        teacher.save_pretrained(path)
    else:
        path.mkdir()
        state = get_peft_model_state_dict(teacher)
        # Diffusers exports are component-prefixed and have no PEFT model prefix.
        state = {
            "transformer." + name.removeprefix("base_model.model."): value
            for name, value in state.items()
        }
        save_file(state, str(path / "pytorch_lora_weights.safetensors"))
    return expected


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("format", ["peft", "diffusers"])
def test_teacher_import_uses_existing_parameters_and_restores_student(tmp_path, wrapped, format):
    path = tmp_path / "teacher"
    expected = _write_teacher(path, format)
    student = _component()
    if wrapped:
        student.base_model.model.block = checkpoint_wrapper(student.base_model.model.block)
    owner = _owner(student)
    parameters = owner._get_component_parameters(["transformer"])
    before = [p.detach().clone() for p in parameters]
    identities = [id(p) for p in student.parameters()]
    _load_one_teacher(owner, "teacher", str(path), ["transformer"], "cpu")
    actual = owner._named_parameters["teacher"].ema_wrapper.ema_parameters
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert identities == [id(p) for p in student.parameters()]
    for value, reference in zip(parameters, before, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    with BaseAdapter.use_named_parameters(owner, "teacher"):
        for value, reference in zip(parameters, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for value, reference in zip(parameters, before, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "corruption", ["missing", "extra", "shape", "collision", "nonfinite", "alpha"]
)
def test_teacher_import_rejects_corruption_before_any_parameter_copy(tmp_path, corruption):
    path = tmp_path / "teacher"
    _write_teacher(path, "peft")
    weights = path / "adapter_model.safetensors"
    state = load_file(str(weights))
    name = next(iter(state))
    if corruption == "missing":
        del state[name]
    elif corruption == "extra":
        state["base_model.model.unknown.lora_A.weight"] = state[name].clone()
    elif corruption == "shape":
        state[name] = state[name][:-1]
    elif corruption == "collision":
        state[name.replace(".block.", ".block._checkpoint_wrapped_module.")] = state[name].clone()
    elif corruption == "nonfinite":
        state[name].fill_(float("nan"))
    else:
        config_path = path / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["lora_alpha"] *= 2
        config_path.write_text(json.dumps(config))
    save_file(state, str(weights))
    student = _component()
    student.base_model.model.block = checkpoint_wrapper(student.base_model.model.block)
    owner = _owner(student)
    before = [p.detach().clone() for p in student.parameters()]
    with pytest.raises((ValueError, TypeError)):
        owner.load_lora_parameter_values(str(path), ["transformer"])
    for value, reference in zip(student.parameters(), before, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert not owner._named_parameters


def test_conversion_overflow_is_rejected_before_any_copy(tmp_path):
    path = tmp_path / "teacher"
    _write_teacher(path, "peft")
    weights = path / "adapter_model.safetensors"
    values = load_file(str(weights))
    values[next(iter(values))].fill_(1e10)
    save_file(values, str(weights))
    student = _component().half()
    owner = _owner(student)
    before = [p.detach().clone() for p in student.parameters()]
    with pytest.raises(ValueError, match="overflows destination dtype=torch.float16"):
        owner.load_lora_parameter_values(str(path), ["transformer"])
    for value, reference in zip(student.parameters(), before, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_real_registry_and_routed_component_preserve_prepared_ownership(tmp_path):
    from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
    from flow_factory.models.variants import ComponentVariantRegistry, ComponentVariantSpec

    path = tmp_path / "teacher"
    expected = _write_teacher(path, "peft")
    student = _component()
    owner = _owner(student)
    owner.trainable_component_names = ("transformer",)
    owner.get_component = lambda name: student
    registry = ComponentVariantRegistry(owner)
    registry.declare(
        ComponentVariantSpec(
            name="base",
            storage_mode="lora",
            component_routes={"transformer": "transformer"},
            adapter_name="default",
        )
    )
    registry.materialize(("base",))
    student.base_model.model.block = checkpoint_wrapper(student.base_model.model.block)
    registry.rebind_parameters({"transformer": student})
    owner.component_variant_registry = registry
    bundle = ModelBundle({"transformer": student})
    proxy = RoutedComponentProxy(bundle, "transformer", registry, {"transformer": student})
    owner._require_component = lambda name: proxy
    owner.accelerator = SimpleNamespace(unwrap_model=lambda module: module)
    owner._unwrap = MethodType(BaseAdapter._unwrap, owner)
    ids = [id(p) for p in registry.parameters("base")]
    _load_one_teacher(owner, "teacher", str(path), ["transformer"], "cpu")
    assert ids == [id(p) for p in registry.parameters("base")]
    assert proxy.inner is student and owner._require_component("transformer") is proxy
    for value, reference in zip(
        owner._named_parameters["teacher"].ema_wrapper.ema_parameters, expected, strict=True
    ):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_second_component_corruption_leaves_first_component_unchanged(tmp_path):
    _write_teacher(tmp_path / "first", "peft")
    _write_teacher(tmp_path / "second", "peft")
    bad = tmp_path / "second/adapter_model.safetensors"
    values = load_file(str(bad))
    del values[next(iter(values))]
    save_file(values, str(bad))
    modules = {name: _component() for name in ("first", "second")}
    owner = _owner(modules["first"])
    owner.target_module_map = {name: ["0", "1"] for name in modules}
    owner._require_component = modules.__getitem__
    owner._checkpoint_entries = lambda path: [
        CheckpointEntry(name, "base", name) for name in modules
    ]
    owner._get_component_parameters = lambda names: [
        p for name in names for p in modules[name].parameters() if p.requires_grad
    ]
    before = {
        name: [p.detach().clone() for p in module.parameters()] for name, module in modules.items()
    }
    with pytest.raises(ValueError, match="incompatible LoRA keys.*second"):
        owner.load_lora_parameter_values(str(tmp_path), ["first", "second"])
    for name, module in modules.items():
        for value, reference in zip(module.parameters(), before[name], strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
