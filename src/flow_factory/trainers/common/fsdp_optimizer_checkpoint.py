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

"""Restore native FSDP2 AdamW checkpoints with legitimately uninitialized parameters.

PyTorch initializes every trainable parameter when constructing an empty optimizer's
DCP load template. An existing checkpoint can instead have no state for parameters
that never received a gradient. Native DCP metadata owns that distinction; missing
fields of an initialized parameter remain errors. The on-disk format is unchanged.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from accelerate.optimizer import AcceleratedOptimizer
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


def _native_adamw(optimizer: Any) -> torch.optim.AdamW | None:
    if isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer
    return optimizer if type(optimizer) is torch.optim.AdamW else None


def _parameter_groups(model: torch.nn.Module, optimizer: torch.optim.AdamW) -> list[dict]:
    """Resolve native canonical names without triggering optimizer initialization."""
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    sentinel = next((parameter for parameter in parameters if parameter.requires_grad), None)
    gradient = None if sentinel is None else sentinel.grad
    try:
        # Native get skips its dummy optimizer step when any gradient exists.
        if not optimizer.state and sentinel is not None and gradient is None:
            sentinel.grad = torch.zeros_like(sentinel)
        return get_optimizer_state_dict(model, optimizer)["param_groups"]
    finally:
        if sentinel is not None:
            sentinel.grad = gradient


def _dense_template(model: torch.nn.Module, optimizer: torch.optim.AdamW) -> dict:
    """Allocate the native template without retaining synthetic state or gradients."""
    original_state, original_groups = optimizer.state, optimizer.param_groups
    gradients = [
        (parameter, parameter.grad) for group in original_groups for parameter in group["params"]
    ]
    try:
        optimizer.state = defaultdict(dict)
        optimizer.param_groups = [dict(group) for group in original_groups]
        for parameter, _ in gradients:
            parameter.grad = None
        return get_optimizer_state_dict(model, optimizer)
    finally:
        optimizer.state, optimizer.param_groups = original_state, original_groups
        for parameter, gradient in gradients:
            parameter.grad = gradient


@dataclass(frozen=True)
class SparseAdamWCheckpoint:
    """A metadata-validated sparse optimizer owned by one prepared model root."""

    model: torch.nn.Module
    optimizer: torch.optim.AdamW
    directory: Path
    fields: dict[str, set[str]]
    groups: list[dict]
    template: dict

    def load(self) -> None:
        """Load every declared tensor strictly, then apply exactly saved membership."""
        template = self.template
        if [group["params"] for group in _parameter_groups(self.model, self.optimizer)] != [
            group["params"] for group in self.groups
        ]:
            raise ValueError("Sparse AdamW checkpoint optimizer ownership changed during resume")
        state = {"optimizer": template}
        dcp.load(
            state, storage_reader=FileSystemReader(self.directory), planner=DefaultLoadPlanner()
        )
        loaded = state["optimizer"]
        if loaded["param_groups"] != self.groups:
            raise ValueError("Sparse AdamW checkpoint parameter groups changed while loading")
        # strict=False applies only after exact ownership, complete state fields,
        # tensor schemas and strict DCP reads have passed. Native optimizer loading
        # replaces temporary state and preserves true absence (not a synthetic step).
        original_state = self.optimizer.state
        original_groups = self.optimizer.param_groups
        gradients = [
            (parameter, parameter.grad)
            for group in original_groups
            for parameter in group["params"]
        ]
        try:
            self.optimizer.state = defaultdict(dict, original_state)
            self.optimizer.param_groups = [dict(group) for group in original_groups]
            set_optimizer_state_dict(
                self.model, self.optimizer, loaded, options=StateDictOptions(strict=False)
            )
        except BaseException:
            self.optimizer.state, self.optimizer.param_groups = original_state, original_groups
            raise
        finally:
            for parameter, gradient in gradients:
                parameter.grad = gradient


def inspect_sparse_adamw_checkpoint(
    model: torch.nn.Module, optimizer: Any, input_dir: str | Path, optimizer_index: int
) -> SparseAdamWCheckpoint | None:
    """Preflight native metadata; return a loader only when state is actually sparse."""
    native = _native_adamw(optimizer)
    if native is None:
        return None
    directory = Path(input_dir) / f"optimizer_{optimizer_index}"
    reader = FileSystemReader(directory)
    metadata = reader.read_metadata()
    if set(metadata.planner_data) != set(metadata.state_dict_metadata):
        raise ValueError("AdamW checkpoint metadata and planner paths disagree")
    fields: dict[str, set[str]] = {}
    group_fields: dict[int, set[str]] = {}
    for name, entry in metadata.state_dict_metadata.items():
        path = metadata.planner_data[name]
        if len(path) != 4 or path[0] != "optimizer":
            raise ValueError(f"Unsupported AdamW checkpoint metadata path: {path!r}")
        if path[1] == "state" and isinstance(path[2], str) and isinstance(path[3], str):
            if not isinstance(entry, TensorStorageMetadata):
                raise ValueError(f"AdamW checkpoint state must contain tensors: {name}")
            fields.setdefault(path[2], set()).add(path[3])
        elif path[1] == "param_groups" and type(path[2]) is int and isinstance(path[3], str):
            if not isinstance(entry, BytesStorageMetadata):
                raise ValueError(f"AdamW checkpoint parameter group must contain metadata: {name}")
            group_fields.setdefault(path[2], set()).add(path[3])
        else:
            raise ValueError(f"Unsupported AdamW checkpoint metadata path: {path!r}")
        if name != ".".join(str(part) for part in path):
            raise ValueError(f"AdamW checkpoint metadata path does not match key: {name}")

    expected_groups = _parameter_groups(model, native)
    if set(group_fields) != set(range(len(expected_groups))):
        raise ValueError("AdamW checkpoint parameter-group count mismatch")
    for index, group in enumerate(expected_groups):
        if group_fields[index] != set(group):
            raise ValueError(f"AdamW checkpoint parameter-group fields mismatch at group {index}")
    group_state = {
        "optimizer": {
            "param_groups": [
                {field: None for field in group_fields[index]}
                for index in range(len(expected_groups))
            ]
        }
    }
    # These are replicated byte metadata only; reading without collectives allows
    # ownership validation before Accelerate mutates models or optimizer tensors.
    dcp.load(group_state, storage_reader=reader, planner=DefaultLoadPlanner(), no_dist=True)
    saved_groups = group_state["optimizer"]["param_groups"]
    if [group["params"] for group in saved_groups] != [
        group["params"] for group in expected_groups
    ]:
        raise ValueError("AdamW checkpoint parameter-group ownership mismatch")
    all_names = {name for group in expected_groups for name in group["params"]}
    if fields.keys() - all_names:
        raise ValueError(
            f"AdamW checkpoint has unknown parameter states: {sorted(fields.keys() - all_names)}"
        )
    for group in expected_groups:
        expected_fields = {"step", "exp_avg", "exp_avg_sq"}
        if group["amsgrad"]:
            expected_fields.add("max_exp_avg_sq")
        for name in group["params"]:
            if name in fields and fields[name] != expected_fields:
                raise ValueError(
                    f"AdamW checkpoint incomplete or unexpected state fields for {name!r}: {sorted(fields[name])}"
                )
    if set(fields) == all_names:
        return None
    template = _dense_template(model, native)
    for name, expected_fields in fields.items():
        current = template["state"].get(name)
        if current is None or set(current) != expected_fields:
            raise ValueError(f"Sparse AdamW checkpoint state fields mismatch for {name!r}")
        for field, tensor in current.items():
            entry = metadata.state_dict_metadata[f"optimizer.state.{name}.{field}"]
            if tensor.shape != entry.size or tensor.dtype != entry.properties.dtype:
                raise ValueError(
                    f"Sparse AdamW checkpoint tensor schema mismatch for {name!r}.{field}"
                )
    template["state"] = {name: template["state"][name] for name in fields}
    return SparseAdamWCheckpoint(model, native, directory, fields, saved_groups, template)


__all__ = ["inspect_sparse_adamw_checkpoint"]
