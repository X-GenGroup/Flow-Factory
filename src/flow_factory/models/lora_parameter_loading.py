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

"""Strict LoRA value imports into existing prepared parameter objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor, distribute_tensor

from ..utils.checkpoint import mapping_lora_state_dict

if TYPE_CHECKING:
    from .abc import BaseAdapter


def _read_values(directory: Path) -> Dict[str, torch.Tensor]:
    """Read one component's PEFT or legacy tensor file without constructing a model."""
    preferred = directory / "adapter_model.safetensors"
    if preferred.is_file():
        return load_file(str(preferred))
    preferred = directory / "adapter_model.bin"
    if preferred.is_file():
        return torch.load(preferred, map_location="cpu", weights_only=True)
    candidates = sorted(directory.glob("*.safetensors")) or sorted(directory.glob("*.bin"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one LoRA tensor file under {str(directory)!r}, received "
            f"{[path.name for path in candidates]!r}"
        )
    return (
        load_file(str(candidates[0]))
        if candidates[0].suffix == ".safetensors"
        else torch.load(candidates[0], map_location="cpu", weights_only=True)
    )


def load_lora_parameter_values(
    adapter: "BaseAdapter", path: str, components: Sequence[str]
) -> None:
    """Validate and copy compatible LoRA values without replacing prepared modules.

    The existing variant registry owns live identities and DTensor placements.
    Checkpoint entries own component paths. PEFT configuration and the complete
    canonical key/shape set are checked before any parameter write. Raw legacy
    files retain the student's scaling and must supply its exact LoRA weight set.

    Args:
        adapter: Adapter with its student LoRA already attached.
        path: Resolved local component checkpoint root.
        components: Components whose existing trainable LoRA values are replaced.

    Raises:
        ValueError: Checkpoint layout, config, names, shapes or values are incompatible.
        TypeError: A checkpoint entry is not a floating tensor or a parameter is not LoRA.
    """
    from .abc import _unwrapped_parameter_name

    registry = getattr(adapter, "component_variant_registry", None)
    role = registry.base_variant if registry is not None else adapter._checkpoint_base_role()
    if registry is not None and registry.active_variant != role:
        raise ValueError(
            f"LoRA value import requires active base role {role!r}, received {registry.active_variant!r}"
        )
    entries = adapter._checkpoint_entries(path)
    if (
        len(entries) != len(components)
        or {entry.component for entry in entries} != set(components)
        or any(entry.role != role for entry in entries)
    ):
        raise ValueError(
            f"expected base-role LoRA entries for {tuple(components)!r}, received {entries!r}"
        )
    entries_by_component = {entry.component: entry for entry in entries}
    plan: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    for component in components:
        module = adapter._unwrap(adapter._require_component(component))
        adapter_name = registry.get_spec(role).adapter_name if registry is not None else "default"
        config = module.peft_config[adapter_name]
        directory = Path(entries_by_component[component].directory(path))
        config_path = directory / "adapter_config.json"
        if config_path.is_file():
            declared = json.loads(config_path.read_text())
            for field in (
                "r",
                "lora_alpha",
                "use_rslora",
                "use_dora",
                "fan_in_fan_out",
                "bias",
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
            ):
                expected = getattr(config, field)
                received = declared.get(field, expected)
                if received != expected:
                    raise ValueError(
                        f"LoRA component {component!r} requires {field}={expected!r}, received {received!r}"
                    )
        records = (
            [
                (record.parameter_name, record.parameter)
                for record in registry.parameter_records(role)
                if record.component_name == component
            ]
            if registry is not None
            else [(name, value) for name, value in module.named_parameters() if value.requires_grad]
        )
        expected_parameters: Dict[str, torch.nn.Parameter] = {}
        for name, parameter in records:
            canonical = _unwrapped_parameter_name(name)
            if ".lora_A." not in canonical and ".lora_B." not in canonical:
                raise TypeError(
                    f"strict LoRA value import requires A/B parameters, received {component}.{canonical}"
                )
            if canonical in expected_parameters:
                raise ValueError(f"duplicate canonical prepared LoRA key {component}.{canonical}")
            expected_parameters[canonical] = parameter
        if not expected_parameters:
            raise ValueError(f"component {component!r} has no live LoRA parameters")
        values: Dict[str, torch.Tensor] = {}
        for name, value in _read_values(directory).items():
            # A Diffusers single-component export prefixes the component, while
            # Flow-Factory/PEFT component exports start at the component module.
            source_name = name[len(component) + 1 :] if name.startswith(component + ".") else name
            mapped = mapping_lora_state_dict({source_name: value}, adapter_name=adapter_name)
            canonical = _unwrapped_parameter_name(next(iter(mapped)))
            if canonical in values:
                raise ValueError(f"duplicate canonical checkpoint LoRA key {component}.{canonical}")
            values[canonical] = value
        missing = sorted(set(expected_parameters) - set(values))
        unexpected = sorted(set(values) - set(expected_parameters))
        if missing or unexpected:
            raise ValueError(
                f"incompatible LoRA keys for component {component!r}: missing={missing!r}, unexpected={unexpected!r}"
            )
        for name, parameter in expected_parameters.items():
            value = values[name]
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(
                    f"expected floating LoRA tensor for {component}.{name}, received {type(value).__name__}"
                )
            if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"invalid LoRA value for {component}.{name}: expected finite shape={tuple(parameter.shape)}, received shape={tuple(value.shape)}"
                )
            plan.append((parameter, value))
    snapshot_parameters = adapter._get_component_parameters(list(components))
    if [id(parameter) for parameter, _ in plan] != [
        id(parameter) for parameter in snapshot_parameters
    ]:
        raise ValueError(
            "prepared LoRA parameter identities/order differ from named-snapshot ownership"
        )
    adapter._prepare_parameter_write()
    prepared = []
    for parameter, value in plan:
        value = value.to(device=parameter.device, dtype=parameter.dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"LoRA value overflows destination dtype={parameter.dtype}, shape={tuple(parameter.shape)}"
            )
        if isinstance(parameter, DTensor):
            value = distribute_tensor(
                value, parameter.device_mesh, parameter.placements, src_data_rank=None
            )
        prepared.append((parameter, value))
    with torch.no_grad():
        for parameter, value in prepared:
            parameter.copy_(value)
