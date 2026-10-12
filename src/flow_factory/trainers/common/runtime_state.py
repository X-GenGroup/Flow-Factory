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

"""Safe checkpoint storage for trainer progress and late-bound child state."""

import hashlib
import json
import math
import os
import random
import re
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Protocol

import numpy as np
import torch
from accelerate.utils import SCALER_NAME
from accelerate.utils import load as accelerate_load
from safetensors.torch import load_file, save_file

from ..execution import TrainingProgress

TRAINER_RUNTIME_STATE_VERSION = 1
TRAINER_RUNTIME_FORMAT = "flow_factory.trainer_runtime"
TRAINER_RUNTIME_METADATA_FILENAME = "flow_factory_trainer_runtime.json"
TRAINER_RUNTIME_TENSOR_PREFIX = "flow_factory_trainer_runtime"
_RUNTIME_STATE_KEYS = frozenset({"version", "progress", "children"})
_PROGRESS_KEYS = frozenset({"optimizer_step", "rollout_iteration", "data_epoch"})
_METADATA_KEYS = frozenset(
    {
        "format",
        "version",
        "identity",
        "child_names",
        "state_files",
        "tensor_file",
        "state",
    }
)
_STATE_FILE_KEYS = frozenset({"path", "size", "sha256"})
_IDENTITY_KEYS = frozenset(
    {
        "trainer",
        "adapter",
        "algorithm",
        "model",
        "finetune_type",
        "optimizer_roles",
        "parameter_schema_digest",
        "optimizer_schema_digest",
        "execution_contract_digest",
        "data_contract_digest",
        "distributed_type",
        "backend_schema_digest",
        "mixed_precision",
        "gradient_scaler",
        "world_size",
    }
)
_LEGACY_CUSTOM_STATE_PATTERN = re.compile(r"^custom_checkpoint_\d+\.pkl$")
_TENSOR_FILENAME_PATTERN = re.compile(
    rf"^{TRAINER_RUNTIME_TENSOR_PREFIX}\.[0-9a-f]{{32}}\.safetensors$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NODE_TYPES = frozenset(
    {"mapping", "list", "tuple", "none", "bool", "int", "float", "str", "tensor"}
)


class CheckpointableChild(Protocol):
    """Structural interface required from a named runtime child."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return serializable child state."""
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore child state."""
        ...

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Validate child state without mutating the live child."""
        ...


class TrainerRuntimeState:
    """Own unified progress and late-bound child checkpoint state (e.g. EMA, reference).

    Runtime payloads intentionally do not use Accelerate's generic custom checkpoint
    objects because those objects are serialized with pickle. The framework writes a
    strictly tagged JSON tree and plain tensors in safetensors instead. A load is first
    decoded and validated without mutation, then committed only after Accelerate has
    successfully restored the policy, optimizer, scheduler, and RNG state.
    """

    checkpoint_id = "flow_factory.trainer_runtime.v1"

    def __init__(
        self,
        progress: TrainingProgress | None = None,
        *,
        child_names: Iterable[str] = (),
        identity: Mapping[str, Any] | None = None,
    ) -> None:
        self._progress = _require_progress(TrainingProgress() if progress is None else progress)
        self._child_names = _normalize_child_names(child_names)
        self._identity = _normalize_identity({} if identity is None else identity)
        self._children: dict[str, CheckpointableChild] = {}
        self._pending_child_states: dict[str, Mapping[str, Any]] = {}
        self._validated_load: tuple[TrainingProgress, dict[str, Mapping[str, Any]]] | None = None
        self._load_received = False

    @property
    def progress(self) -> TrainingProgress:
        """Return the current immutable progress value."""
        return self._progress

    @progress.setter
    def progress(self, progress: TrainingProgress) -> None:
        """Replace the current progress value after strict type validation."""
        self._progress = _require_progress(progress)

    @property
    def child_names(self) -> tuple[str, ...]:
        """Return the immutable child declaration order."""
        return self._child_names

    @property
    def identity(self) -> dict[str, Any]:
        """Return the immutable-compatible state-resume identity fields."""
        return {**self._identity, "optimizer_roles": list(self._identity["optimizer_roles"])}

    @property
    def pending_child_names(self) -> tuple[str, ...]:
        """Return restored children that have not yet consumed their payload."""
        return tuple(name for name in self._child_names if name in self._pending_child_states)

    @property
    def load_received(self) -> bool:
        """Return whether a validated runtime payload has been committed."""
        return self._load_received

    @property
    def validated_load_pending(self) -> bool:
        """Return whether a preflighted payload is waiting for policy-state restore."""
        return self._validated_load is not None

    def configure_identity(self, identity: Mapping[str, Any]) -> None:
        """Bind the realized trainer/model/optimizer layout exactly once."""
        normalized = _normalize_identity(identity)
        if self._identity["trainer"] != "unspecified":
            if self._identity != normalized:
                raise RuntimeError(
                    "trainer runtime identity changed after configuration: expected "
                    f"{self._identity!r}, received {normalized!r}"
                )
            return
        self._identity = normalized

    def prepare_save(self, output_dir: str | os.PathLike[str]) -> None:
        """Atomically publish a JSON manifest and safetensors runtime payload.

        The tensor generation is written first and the JSON manifest is the commit
        marker. Its generated basename prevents an interrupted overwrite from pairing
        old JSON state with new tensors.
        """
        state = self.state_dict()
        tensors: dict[str, torch.Tensor] = {}
        encoded_state = _encode_node(state, tensors=tensors, path="runtime")
        output_path = os.fspath(output_dir)
        os.makedirs(output_path, exist_ok=True)

        metadata_path = os.path.join(output_path, TRAINER_RUNTIME_METADATA_FILENAME)
        if os.path.exists(metadata_path):
            raise FileExistsError(
                "trainer runtime checkpoints are immutable and cannot overwrite an "
                f"existing manifest: {metadata_path!r}"
            )

        tensor_filename = f"{TRAINER_RUNTIME_TENSOR_PREFIX}.{uuid.uuid4().hex}.safetensors"
        tensor_path = os.path.join(output_path, tensor_filename)
        tensor_temp_path = f"{tensor_path}.tmp"
        metadata_temp_path = f"{metadata_path}.tmp"
        state_files = _collect_accelerate_state_files(output_path)

        try:
            save_file(tensors, tensor_temp_path)
            os.replace(tensor_temp_path, tensor_path)
            metadata = {
                "format": TRAINER_RUNTIME_FORMAT,
                "version": TRAINER_RUNTIME_STATE_VERSION,
                "identity": self.identity,
                "child_names": list(self._child_names),
                "state_files": state_files,
                "tensor_file": _describe_file(tensor_path, tensor_filename),
                "state": encoded_state,
            }
            with open(metadata_temp_path, "w", encoding="utf-8") as metadata_file:
                json.dump(
                    metadata,
                    metadata_file,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                metadata_file.write("\n")
            os.replace(metadata_temp_path, metadata_path)
        finally:
            for temporary_path in (tensor_temp_path, metadata_temp_path):
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def validate_load(
        self,
        input_dir: str | os.PathLike[str],
        *,
        children: Mapping[str, CheckpointableChild] | None = None,
        invariant_validator: (
            Callable[[TrainingProgress, Mapping[str, Mapping[str, Any]]], None] | None
        ) = None,
        expected_process_index: int | None = None,
        expected_device_type: str | None = None,
    ) -> None:
        """Decode and stage compatible state before policy/optimizer mutation."""
        if self._load_received:
            raise RuntimeError("trainer runtime state has already received a checkpoint load")
        if self._validated_load is not None:
            raise RuntimeError("trainer runtime state already has a validated pending load")
        if self._children:
            raise RuntimeError(
                "trainer runtime state must validate before attaching children; already "
                f"attached children: {tuple(self._children)!r}"
            )

        input_path = os.fspath(input_dir)
        legacy_files = tuple(
            sorted(
                filename
                for filename in os.listdir(input_path)
                if _LEGACY_CUSTOM_STATE_PATTERN.fullmatch(filename) is not None
            )
        )
        if legacy_files:
            raise RuntimeError(
                "state checkpoint contains legacy or foreign pickle custom state "
                f"files {legacy_files!r}; this runtime accepts only JSON + safetensors. "
                "Resume model weights instead or regenerate a trusted state checkpoint."
            )

        metadata_path = os.path.join(input_path, TRAINER_RUNTIME_METADATA_FILENAME)
        if os.path.islink(metadata_path) or not os.path.isfile(metadata_path):
            raise RuntimeError(
                "state checkpoint is incompatible with trainer runtime-state v1: expected "
                f"metadata file {metadata_path!r}, received missing file. Checkpoints created "
                "before safe runtime-state v1 did not serialize TrainingProgress and "
                "EMA/reference state safely; resume their model weights instead of using "
                "resume_type='state'."
            )
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(
                metadata_file,
                object_pairs_hook=_reject_duplicate_object_pairs,
                parse_constant=_reject_json_constant,
            )
        metadata = _require_mapping(metadata, "trainer runtime metadata")
        _require_exact_keys(metadata, _METADATA_KEYS, "trainer runtime metadata")
        _validate_metadata_header(metadata, self._child_names, self._identity)
        _validate_accelerate_state_files(
            input_path,
            metadata["state_files"],
            require_complete=self._identity["trainer"] != "unspecified",
            expected_process_index=expected_process_index,
            expected_device_type=expected_device_type,
            world_size=self._identity["world_size"],
            require_scaler=self._identity["gradient_scaler"] not in ("none", "unspecified"),
        )

        tensor_filename = _validate_checkpoint_file(
            input_path,
            metadata["tensor_file"],
            context="trainer runtime tensor_file",
        )
        if not _TENSOR_FILENAME_PATTERN.fullmatch(tensor_filename):
            raise ValueError(
                "trainer runtime metadata tensor_file path must be a generated runtime "
                f"safetensors basename, received {tensor_filename!r}"
            )
        _validate_manifest_file_set(
            input_path,
            state_files=metadata["state_files"],
            tensor_filename=tensor_filename,
        )
        tensor_path = os.path.join(input_path, tensor_filename)
        tensors = load_file(tensor_path, device="cpu")
        consumed_tensors: set[str] = set()
        state = _decode_node(
            metadata["state"],
            tensors=tensors,
            consumed_tensors=consumed_tensors,
            path="runtime",
        )
        extra_tensors = frozenset(tensors).difference(consumed_tensors)
        if extra_tensors:
            raise ValueError(
                "trainer runtime safetensors contains unreferenced tensors: "
                f"{tuple(sorted(extra_tensors))!r}"
            )

        progress, child_payloads = _decode_runtime_state(state, self._child_names)
        if children is None:
            if self._child_names:
                raise RuntimeError(
                    "trainer runtime resume preflight requires realized child validators for "
                    f"{self._child_names!r}"
                )
        else:
            children = _require_mapping(children, "trainer runtime preflight children")
            _require_exact_keys(
                children,
                frozenset(self._child_names),
                "trainer runtime preflight children",
            )
            for name in self._child_names:
                child = children[name]
                _require_checkpointable_child(child, name, require_validator=True)
                child.validate_state_dict(child_payloads[name])
        if invariant_validator is not None:
            if not callable(invariant_validator):
                raise TypeError(
                    "trainer runtime invariant_validator must be callable, received "
                    f"{type(invariant_validator).__name__}: {invariant_validator!r}"
                )
            invariant_validator(progress, child_payloads)
        self._validated_load = (progress, child_payloads)

    def commit_validated_load(self) -> None:
        """Commit the preflighted runtime payload after Accelerate succeeds."""
        if self._validated_load is None:
            raise RuntimeError("trainer runtime state has no validated load to commit")
        progress, children = self._validated_load
        self._install_loaded_state(progress, children)
        self._validated_load = None

    def attach_child(self, name: str, child: CheckpointableChild) -> None:
        """Attach one declared child and consume its pending payload if present."""
        _require_child_name(name)
        if name not in self._child_names:
            raise KeyError(
                f"runtime child {name!r} was not declared; expected one of "
                f"{self._child_names!r}"
            )
        if name in self._children:
            raise RuntimeError(f"runtime child {name!r} is already attached")
        _require_checkpointable_child(child, name)

        if name in self._pending_child_states:
            payload = self._pending_child_states[name]
            child.load_state_dict(dict(payload))
            del self._pending_child_states[name]
        self._children[name] = child

    def state_dict(self) -> dict[str, Any]:
        """Return progress and every declared child for safe checkpointing."""
        children: dict[str, dict[str, Any]] = {}
        for name in self._child_names:
            if name in self._children:
                payload = self._children[name].state_dict()
                payload = _require_child_payload(payload, name)
            elif name in self._pending_child_states:
                payload = self._pending_child_states[name]
            else:
                raise RuntimeError(
                    f"cannot serialize runtime child {name!r}: it is neither attached "
                    "nor backed by a pending restored payload"
                )
            children[name] = dict(payload)

        progress = self._progress
        return {
            "version": TRAINER_RUNTIME_STATE_VERSION,
            "progress": {
                "optimizer_step": progress.optimizer_step,
                "rollout_iteration": progress.rollout_iteration,
                "data_epoch": progress.data_epoch,
            },
            "children": children,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load an already-decoded payload for narrow in-process integrations."""
        progress, children = _decode_runtime_state(state_dict, self._child_names)
        self._install_loaded_state(progress, children)

    def _install_loaded_state(
        self,
        progress: TrainingProgress,
        children: dict[str, Mapping[str, Any]],
    ) -> None:
        """Install one decoded payload exactly once before child attachment."""
        if self._load_received:
            raise RuntimeError("trainer runtime state has already received a checkpoint load")
        if self._children:
            raise RuntimeError(
                "trainer runtime state must load before attaching children; already attached "
                f"children: {tuple(self._children)!r}"
            )
        self._progress = progress
        self._pending_child_states = children
        self._load_received = True


def _encode_node(
    value: Any,
    *,
    tensors: dict[str, torch.Tensor],
    path: str,
) -> dict[str, Any]:
    """Encode one strictly typed tree node and extract every tensor leaf."""
    if type(value) is torch.Tensor:
        if value.layout is not torch.strided or value.is_sparse or value.is_complex():
            raise TypeError(
                f"runtime tensor at {path} must be a dense real strided tensor, "
                f"received layout={value.layout}, dtype={value.dtype}"
            )
        if value.device.type == "meta":
            raise TypeError(f"runtime tensor at {path} cannot reside on the meta device")
        name = f"tensor_{len(tensors):08d}"
        tensors[name] = value.detach().to(device="cpu").contiguous().clone()
        return {"type": "tensor", "name": name}
    if isinstance(value, torch.Tensor):
        raise TypeError(
            f"runtime tensor at {path} must be a plain torch.Tensor, "
            f"received {type(value).__name__}"
        )
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"runtime mapping key at {path} must be str, "
                    f"received {type(key).__name__}: {key!r}"
                )
            items.append([key, _encode_node(item, tensors=tensors, path=f"{path}.{key}")])
        return {"type": "mapping", "items": items}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _encode_node(item, tensors=tensors, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _encode_node(item, tensors=tensors, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"runtime float at {path} must be finite, received {value!r}")
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    raise TypeError(f"unsupported runtime state value at {path}: {type(value).__name__}: {value!r}")


def _decode_node(
    node: Any,
    *,
    tensors: Mapping[str, torch.Tensor],
    consumed_tensors: set[str],
    path: str,
) -> Any:
    """Decode one tagged JSON node without constructing executable objects."""
    node = _require_mapping(node, f"encoded runtime node at {path}")
    node_type = node.get("type")
    if type(node_type) is not str or node_type not in _NODE_TYPES:
        raise ValueError(f"encoded runtime node at {path} has invalid type tag {node_type!r}")
    if node_type in ("mapping", "list", "tuple"):
        _require_exact_keys(node, frozenset({"type", "items"}), f"{node_type} node at {path}")
        items = node["items"]
        if not isinstance(items, list):
            raise TypeError(
                f"encoded runtime {node_type} items at {path} must be list, "
                f"received {type(items).__name__}: {items!r}"
            )
        if node_type == "mapping":
            result: dict[str, Any] = {}
            for index, pair in enumerate(items):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise TypeError(
                        f"encoded runtime mapping item {index} at {path} must be a "
                        f"two-item list, received {pair!r}"
                    )
                key, child_node = pair
                if type(key) is not str:
                    raise TypeError(
                        f"encoded runtime mapping key at {path} must be str, "
                        f"received {type(key).__name__}: {key!r}"
                    )
                if key in result:
                    raise ValueError(
                        f"encoded runtime mapping at {path} contains duplicate key {key!r}"
                    )
                result[key] = _decode_node(
                    child_node,
                    tensors=tensors,
                    consumed_tensors=consumed_tensors,
                    path=f"{path}.{key}",
                )
            return result
        decoded_items = [
            _decode_node(
                item,
                tensors=tensors,
                consumed_tensors=consumed_tensors,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(items)
        ]
        return decoded_items if node_type == "list" else tuple(decoded_items)
    if node_type == "none":
        _require_exact_keys(node, frozenset({"type"}), f"none node at {path}")
        return None
    if node_type in ("bool", "int", "float", "str"):
        _require_exact_keys(node, frozenset({"type", "value"}), f"{node_type} node at {path}")
        value = node["value"]
        expected_type = {"bool": bool, "int": int, "float": float, "str": str}[node_type]
        if type(value) is not expected_type:
            raise TypeError(
                f"encoded runtime {node_type} at {path} must contain "
                f"{expected_type.__name__}, received {type(value).__name__}: {value!r}"
            )
        if node_type == "float" and not math.isfinite(value):
            raise ValueError(f"encoded runtime float at {path} must be finite")
        return value

    _require_exact_keys(node, frozenset({"type", "name"}), f"tensor node at {path}")
    tensor_name = node["name"]
    if type(tensor_name) is not str or not tensor_name:
        raise TypeError(
            f"encoded runtime tensor name at {path} must be a non-empty str, "
            f"received {type(tensor_name).__name__}: {tensor_name!r}"
        )
    if tensor_name in consumed_tensors:
        raise ValueError(f"encoded runtime state references tensor {tensor_name!r} more than once")
    if tensor_name not in tensors:
        raise ValueError(f"encoded runtime state references missing tensor {tensor_name!r}")
    tensor = tensors[tensor_name]
    if type(tensor) is not torch.Tensor:
        raise TypeError(
            f"runtime safetensors value {tensor_name!r} must be a plain torch.Tensor, "
            f"received {type(tensor).__name__}"
        )
    consumed_tensors.add(tensor_name)
    return tensor


def _decode_runtime_state(
    state_dict: Mapping[str, Any], expected_child_names: tuple[str, ...]
) -> tuple[TrainingProgress, dict[str, Mapping[str, Any]]]:
    """Validate one serialized runtime state without mutating its receiver."""
    state_dict = _require_mapping(state_dict, "trainer runtime state")
    _require_exact_keys(state_dict, _RUNTIME_STATE_KEYS, "trainer runtime state")

    version = state_dict["version"]
    if type(version) is not int:
        raise TypeError(
            "expected trainer runtime state version to be int, "
            f"received {type(version).__name__}: {version!r}"
        )
    if version != TRAINER_RUNTIME_STATE_VERSION:
        raise ValueError(
            "trainer runtime state version mismatch: expected "
            f"{TRAINER_RUNTIME_STATE_VERSION}, received {version}"
        )

    raw_progress = _require_mapping(state_dict["progress"], "trainer runtime progress")
    _require_exact_keys(raw_progress, _PROGRESS_KEYS, "trainer runtime progress")
    progress = TrainingProgress(
        optimizer_step=_require_counter(raw_progress["optimizer_step"], "optimizer_step"),
        rollout_iteration=_require_counter(raw_progress["rollout_iteration"], "rollout_iteration"),
        data_epoch=_require_counter(raw_progress["data_epoch"], "data_epoch"),
    )

    raw_children = _require_mapping(state_dict["children"], "trainer runtime children")
    _require_exact_keys(raw_children, frozenset(expected_child_names), "trainer runtime children")
    children = {
        name: dict(_require_child_payload(raw_children[name], name))
        for name in expected_child_names
    }
    return progress, children


def _normalize_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact state-resume identity without accepting lookalike runs."""
    identity = _require_mapping(identity, "trainer runtime identity")
    if not identity:
        return {
            "trainer": "unspecified",
            "adapter": "unspecified",
            "algorithm": "unspecified",
            "model": "unspecified",
            "finetune_type": "unspecified",
            "optimizer_roles": (),
            "parameter_schema_digest": "unspecified",
            "optimizer_schema_digest": "unspecified",
            "execution_contract_digest": "unspecified",
            "data_contract_digest": "unspecified",
            "distributed_type": "unspecified",
            "backend_schema_digest": "unspecified",
            "mixed_precision": "unspecified",
            "gradient_scaler": "unspecified",
            "world_size": 1,
        }
    _require_exact_keys(identity, _IDENTITY_KEYS, "trainer runtime identity")
    normalized: dict[str, Any] = {}
    for field_name in (
        "trainer",
        "adapter",
        "algorithm",
        "model",
        "finetune_type",
        "parameter_schema_digest",
        "optimizer_schema_digest",
        "execution_contract_digest",
        "data_contract_digest",
        "distributed_type",
        "backend_schema_digest",
        "mixed_precision",
        "gradient_scaler",
    ):
        value = identity[field_name]
        if type(value) is not str or not value:
            raise TypeError(
                f"trainer runtime identity {field_name} must be a non-empty str, "
                f"received {type(value).__name__}: {value!r}"
            )
        normalized[field_name] = value
    optimizer_roles = identity["optimizer_roles"]
    if isinstance(optimizer_roles, (str, bytes)):
        raise TypeError("trainer runtime identity optimizer_roles must be a sequence")
    try:
        optimizer_roles = tuple(optimizer_roles)
    except TypeError as error:
        raise TypeError("trainer runtime identity optimizer_roles must be a sequence") from error
    for role in optimizer_roles:
        if type(role) is not str or not role:
            raise TypeError(
                "trainer runtime identity optimizer role must be a non-empty str, "
                f"received {type(role).__name__}: {role!r}"
            )
    if len(set(optimizer_roles)) != len(optimizer_roles):
        raise ValueError(
            "trainer runtime identity optimizer_roles must be unique, received "
            f"{optimizer_roles!r}"
        )
    normalized["optimizer_roles"] = optimizer_roles
    world_size = identity["world_size"]
    if type(world_size) is not int or world_size < 1:
        raise TypeError(
            "trainer runtime identity world_size must be a positive int, "
            f"received {type(world_size).__name__}: {world_size!r}"
        )
    normalized["world_size"] = world_size
    return normalized


def _collect_accelerate_state_files(output_path: str) -> list[dict[str, Any]]:
    """Record every already-written Accelerate state artifact with integrity data."""
    entries = []
    for directory, directory_names, filenames in os.walk(output_path):
        for directory_name in directory_names:
            directory_path = os.path.join(directory, directory_name)
            if os.path.islink(directory_path):
                raise RuntimeError(
                    "state checkpoint staging cannot contain symlinked "
                    f"directories: {directory_path!r}"
                )
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            if os.path.islink(file_path) or not os.path.isfile(file_path):
                raise RuntimeError(
                    "state checkpoint staging requires regular files, " f"received {file_path!r}"
                )
            relative_path = os.path.relpath(file_path, output_path).replace(os.sep, "/")
            entries.append(_describe_file(file_path, relative_path))
    return sorted(entries, key=lambda entry: entry["path"])


def _describe_file(file_path: str, relative_path: str) -> dict[str, Any]:
    """Describe one immutable checkpoint artifact without loading it into memory."""
    return {
        "path": relative_path,
        "size": os.path.getsize(file_path),
        "sha256": _file_sha256(file_path),
    }


def _validate_manifest_file_set(
    input_path: str,
    *,
    state_files: Any,
    tensor_filename: str,
) -> None:
    """Reject unmanifested files that a backend loader could otherwise consume."""
    if not isinstance(state_files, list):
        raise TypeError("trainer runtime metadata state_files must be a list")
    expected_paths = {
        TRAINER_RUNTIME_METADATA_FILENAME,
        tensor_filename,
    }
    for index, entry in enumerate(state_files):
        entry = _require_mapping(entry, f"trainer runtime state_files[{index}]")
        relative_path = entry.get("path")
        if type(relative_path) is not str:
            raise TypeError(
                f"trainer runtime state_files[{index}] path must be str, "
                f"received {type(relative_path).__name__}: {relative_path!r}"
            )
        expected_paths.add(relative_path)

    actual_paths = set()
    for directory, directory_names, filenames in os.walk(input_path):
        for directory_name in directory_names:
            directory_path = os.path.join(directory, directory_name)
            if os.path.islink(directory_path):
                raise RuntimeError(
                    "state checkpoint cannot contain symlinked directories during resume: "
                    f"{directory_path!r}"
                )
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            if os.path.islink(file_path) or not os.path.isfile(file_path):
                raise RuntimeError(
                    "state checkpoint resume requires regular files, " f"received {file_path!r}"
                )
            actual_paths.add(os.path.relpath(file_path, input_path).replace(os.sep, "/"))
    unmanifested_state_paths = tuple(
        sorted(
            path
            for path in actual_paths.difference(expected_paths)
            if _is_backend_state_candidate(path)
        )
    )
    if unmanifested_state_paths:
        raise RuntimeError(
            "state checkpoint contains unmanifested backend artifacts that could be "
            f"consumed during resume: unmanifested={unmanifested_state_paths!r}"
        )


def _is_backend_state_candidate(path: str) -> bool:
    """Identify unmanifested names that Accelerate or a backend may consume."""
    top_level = path.split("/", 1)[0]
    prefixes = (
        "model",
        "pytorch_model",
        "optimizer",
        "scheduler",
        "sampler",
        "scaler",
        "random_states",
        "custom_checkpoint",
        "dl_state_dict",
        TRAINER_RUNTIME_TENSOR_PREFIX,
        "flow_factory_",
    )
    return top_level.startswith(prefixes)


def _file_sha256(file_path: str) -> str:
    """Return the streaming SHA-256 digest for one checkpoint artifact."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_file(input_path: str, entry: Any, *, context: str) -> str:
    """Validate one manifest entry and its size/digest before state mutation."""
    entry = _require_mapping(entry, context)
    _require_exact_keys(entry, _STATE_FILE_KEYS, context)
    relative_path = entry["path"]
    if (
        type(relative_path) is not str
        or not relative_path
        or relative_path.startswith("/")
        or "\\" in relative_path
        or any(part in ("", ".", "..") for part in relative_path.split("/"))
    ):
        raise ValueError(
            "trainer runtime state file path must be a normalized relative POSIX path, "
            f"received {relative_path!r}"
        )
    expected_size = entry["size"]
    if type(expected_size) is not int or expected_size < 0:
        raise TypeError(
            "trainer runtime state file size must be a non-negative int, "
            f"received {type(expected_size).__name__}: {expected_size!r}"
        )
    expected_sha256 = entry["sha256"]
    if type(expected_sha256) is not str or not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError(
            "trainer runtime state file sha256 must be a lowercase hexadecimal digest, "
            f"received {expected_sha256!r}"
        )

    file_path = os.path.join(input_path, *relative_path.split("/"))
    if os.path.islink(file_path) or not os.path.isfile(file_path):
        raise RuntimeError(f"trainer runtime state artifact is missing: expected {file_path!r}")
    received_size = os.path.getsize(file_path)
    if received_size != expected_size:
        raise RuntimeError(
            "trainer runtime state artifact size mismatch before resume: expected "
            f"{expected_size} bytes for {relative_path!r}, received {received_size}"
        )
    received_sha256 = _file_sha256(file_path)
    if received_sha256 != expected_sha256:
        raise RuntimeError(
            "trainer runtime state artifact SHA-256 mismatch before resume: expected "
            f"{expected_sha256} for {relative_path!r}, received {received_sha256}"
        )
    return relative_path


def _validate_accelerate_state_files(
    input_path: str,
    entries: Any,
    *,
    require_complete: bool,
    expected_process_index: int | None,
    expected_device_type: str | None,
    world_size: int,
    require_scaler: bool,
) -> None:
    """Reject missing/truncated core state and parse RNG before model mutation."""
    if not isinstance(entries, list):
        raise TypeError(
            "trainer runtime metadata state_files must be a list, "
            f"received {type(entries).__name__}: {entries!r}"
        )
    received_paths = []
    for index, entry in enumerate(entries):
        relative_path = _validate_checkpoint_file(
            input_path,
            entry,
            context=f"trainer runtime state_files[{index}]",
        )
        received_paths.append(relative_path)

    if len(set(received_paths)) != len(received_paths):
        raise ValueError(
            "trainer runtime metadata state_files contains duplicate paths: "
            f"{tuple(received_paths)!r}"
        )
    if received_paths != sorted(received_paths):
        raise ValueError("trainer runtime metadata state_files must be sorted by path")
    if not require_complete:
        return
    if type(expected_process_index) is not int:
        raise TypeError(
            "exact state preflight requires accelerator.process_index as an int, "
            f"received {type(expected_process_index).__name__}: "
            f"{expected_process_index!r}"
        )
    if expected_process_index < 0 or expected_process_index >= world_size:
        raise ValueError(
            "exact state preflight process index is outside the checkpoint world: "
            f"process_index={expected_process_index}, world_size={world_size}"
        )
    if type(expected_device_type) is not str or not expected_device_type:
        raise TypeError(
            "exact state preflight requires accelerator.device.type as a non-empty "
            f"str, received {type(expected_device_type).__name__}: "
            f"{expected_device_type!r}"
        )

    paths = frozenset(received_paths)
    if not any(_is_prepared_model_artifact(path) for path in paths):
        raise RuntimeError("exact state checkpoint is missing its prepared model artifact")
    if not any(_is_optimizer_artifact(path) for path in paths):
        raise RuntimeError("exact state checkpoint is missing its optimizer artifact")
    if require_scaler and SCALER_NAME not in paths:
        raise RuntimeError(
            "exact state checkpoint is missing the gradient scaler artifact required "
            "by its runtime identity"
        )
    rng_paths = tuple(
        sorted(
            path for path in paths if re.fullmatch(r"random_states_[0-9]+\.pkl", path) is not None
        )
    )
    if not rng_paths:
        raise RuntimeError("exact state checkpoint is missing its per-rank RNG artifact")
    expected_rng_path = f"random_states_{expected_process_index}.pkl"
    if expected_rng_path not in rng_paths:
        raise RuntimeError(
            "exact state checkpoint is missing the current rank RNG artifact: "
            f"expected {expected_rng_path!r}, available={rng_paths!r}"
        )
    _validate_rng_state(
        input_path,
        expected_rng_path,
        expected_device_type=expected_device_type,
    )


def _validate_rng_state(
    input_path: str,
    relative_path: str,
    *,
    expected_device_type: str,
) -> None:
    """Validate the exact RNG payload Accelerate will consume for this rank."""
    rng_path = os.path.join(input_path, relative_path)
    rng_state = accelerate_load(rng_path, map_location="cpu", weights_only=True)
    rng_state = _require_mapping(
        rng_state,
        f"exact-resume RNG state {relative_path!r}",
    )
    required_rng_keys = {
        "random_state",
        "numpy_random_seed",
        "torch_manual_seed",
    }
    device_rng_keys = {
        "cuda": "torch_cuda_manual_seed",
        "xpu": "torch_xpu_manual_seed",
        "mlu": "torch_mlu_manual_seed",
        "sdaa": "torch_sdaa_manual_seed",
        "musa": "torch_musa_manual_seed",
        "hpu": "torch_hpu_manual_seed",
        "neuron": "torch_neuron_manual_seed",
        "xla": "xm_seed",
    }
    if expected_device_type == "mps":
        raise RuntimeError(
            "Accelerate does not serialize the MPS RNG state required for exact "
            "training resume; use a model-weight resume on MPS"
        )
    if expected_device_type != "cpu":
        device_key = device_rng_keys.get(expected_device_type)
        if device_key is None:
            raise RuntimeError(
                "exact training resume does not recognize the accelerator RNG "
                f"device type {expected_device_type!r}"
            )
        required_rng_keys.add(device_key)
    missing_rng_keys = required_rng_keys.difference(rng_state)
    if missing_rng_keys:
        raise ValueError(
            f"exact-resume RNG state {relative_path!r} is missing required keys: "
            f"{tuple(sorted(missing_rng_keys))!r}"
        )
    if "step" in rng_state and (type(rng_state["step"]) is not int or rng_state["step"] < 0):
        raise ValueError(
            f"exact-resume RNG state {relative_path!r} step must be a non-negative "
            f"int, received {rng_state['step']!r}"
        )

    try:
        random.Random().setstate(rng_state["random_state"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"exact-resume RNG state {relative_path!r} has invalid random_state"
        ) from error
    try:
        np.random.RandomState().set_state(rng_state["numpy_random_seed"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"exact-resume RNG state {relative_path!r} has invalid numpy_random_seed"
        ) from error
    torch_state = rng_state["torch_manual_seed"]
    if type(torch_state) is not torch.Tensor:
        raise TypeError(
            f"exact-resume RNG {relative_path!r} torch_manual_seed must be a "
            f"plain torch.Tensor, received {type(torch_state).__name__}"
        )
    try:
        torch.Generator(device="cpu").set_state(torch_state)
    except RuntimeError as error:
        raise ValueError(
            f"exact-resume RNG state {relative_path!r} has invalid torch_manual_seed"
        ) from error

    if expected_device_type in device_rng_keys and expected_device_type != "xla":
        device_states = rng_state[device_rng_keys[expected_device_type]]
        if not isinstance(device_states, (list, tuple)) or not device_states:
            raise TypeError(
                f"exact-resume RNG {relative_path!r} device state must be a non-empty "
                f"sequence, received {type(device_states).__name__}: {device_states!r}"
            )
        device_module = getattr(torch, expected_device_type, None)
        device_count = getattr(device_module, "device_count", None)
        if not callable(device_count):
            raise RuntimeError(
                "exact training resume cannot validate the visible RNG topology for "
                f"device type {expected_device_type!r}; use a model-weight resume"
            )
        visible_device_count = device_count()
        if type(visible_device_count) is not int or visible_device_count < 1:
            raise RuntimeError(
                "exact training resume expected at least one visible device for RNG "
                f"type {expected_device_type!r}, received {visible_device_count!r}"
            )
        if len(device_states) != visible_device_count:
            raise ValueError(
                f"exact-resume RNG {relative_path!r} device-state topology mismatch: "
                f"expected {visible_device_count} visible {expected_device_type} device "
                f"states, received {len(device_states)}"
            )
        for index, device_state in enumerate(device_states):
            if type(device_state) is not torch.Tensor:
                raise TypeError(
                    f"exact-resume RNG {relative_path!r} device state {index} must be "
                    f"a plain torch.Tensor, received {type(device_state).__name__}"
                )
            if (
                device_state.device.type != "cpu"
                or device_state.dtype is not torch.uint8
                or device_state.ndim != 1
                or device_state.numel() == 0
            ):
                raise ValueError(
                    f"exact-resume RNG {relative_path!r} device state {index} must "
                    "be a non-empty one-dimensional CPU uint8 tensor, received "
                    f"device={device_state.device}, dtype={device_state.dtype}, "
                    f"shape={tuple(device_state.shape)!r}"
                )
            try:
                generator = torch.Generator(device=torch.device(expected_device_type, index))
                generator.set_state(device_state)
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"exact-resume RNG {relative_path!r} device state {index} cannot "
                    f"be installed on {expected_device_type}:{index}"
                ) from error
    elif expected_device_type == "xla":
        xm_seed = rng_state[device_rng_keys[expected_device_type]]
        if type(xm_seed) is not int and type(xm_seed) is not torch.Tensor:
            raise TypeError(
                f"exact-resume RNG {relative_path!r} xm_seed must be an int or plain "
                f"torch.Tensor, received {type(xm_seed).__name__}"
            )


def _is_prepared_model_artifact(path: str) -> bool:
    """Recognize Accelerate model artifacts across plain, FSDP, and DeepSpeed."""
    basename = path.rsplit("/", 1)[-1]
    return (
        basename in {"model.safetensors", "pytorch_model.bin"}
        or (basename.startswith("pytorch_model_fsdp") and basename.endswith(".bin"))
        or path.startswith("pytorch_model_fsdp_")
        or (
            path.startswith("pytorch_model/")
            and "model_states" in basename.lower()
            and "optim_states" not in basename.lower()
        )
    )


def _is_optimizer_artifact(path: str) -> bool:
    """Recognize standalone or backend-integrated Accelerate optimizer state."""
    basename = path.rsplit("/", 1)[-1]
    return (
        basename == "optimizer.bin"
        or (basename.startswith("optimizer_") and basename.endswith(".bin"))
        or path.startswith("optimizer_")
        # DeepSpeed stores optimizer shards inside its model checkpoint directory.
        or (path.startswith("pytorch_model/") and "optim" in basename.lower())
    )


def _validate_metadata_header(
    metadata: Mapping[str, Any],
    expected_child_names: tuple[str, ...],
    expected_identity: Mapping[str, Any],
) -> None:
    """Validate the non-executable runtime manifest header."""
    format_name = metadata["format"]
    if type(format_name) is not str or format_name != TRAINER_RUNTIME_FORMAT:
        raise ValueError(
            "trainer runtime metadata format mismatch: expected "
            f"{TRAINER_RUNTIME_FORMAT!r}, received {format_name!r}"
        )
    version = metadata["version"]
    if type(version) is not int:
        raise TypeError(
            "expected trainer runtime metadata version to be int, "
            f"received {type(version).__name__}: {version!r}"
        )
    if version != TRAINER_RUNTIME_STATE_VERSION:
        raise ValueError(
            "trainer runtime metadata version mismatch: expected "
            f"{TRAINER_RUNTIME_STATE_VERSION}, received {version}"
        )

    received_identity = _normalize_identity(metadata["identity"])
    if received_identity != expected_identity:
        raise ValueError(
            "trainer runtime metadata identity mismatch: expected "
            f"{dict(expected_identity)!r}, received {received_identity!r}"
        )

    child_names = metadata["child_names"]
    if not isinstance(child_names, list):
        raise TypeError(
            "expected trainer runtime metadata child_names as a list, "
            f"received {type(child_names).__name__}: {child_names!r}"
        )
    for name in child_names:
        _require_child_name(name)
    if tuple(child_names) != expected_child_names:
        raise ValueError(
            "trainer runtime metadata child_names mismatch: expected "
            f"{expected_child_names!r}, received {tuple(child_names)!r}"
        )


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently accepting the last one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"trainer runtime JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject NaN and infinity tokens accepted by Python's JSON decoder."""
    raise ValueError(f"trainer runtime JSON contains non-finite constant {value!r}")


def _require_progress(progress: Any) -> TrainingProgress:
    """Require the concrete immutable progress value used by execution drivers."""
    if type(progress) is not TrainingProgress:
        raise TypeError(
            "expected progress to be TrainingProgress, "
            f"received {type(progress).__name__}: {progress!r}"
        )
    return progress


def _normalize_child_names(child_names: Iterable[str]) -> tuple[str, ...]:
    """Validate and freeze declared child names."""
    if isinstance(child_names, (str, bytes)):
        raise TypeError(
            "expected child_names to be an iterable of names, "
            f"received {type(child_names).__name__}: {child_names!r}"
        )
    try:
        names = tuple(child_names)
    except TypeError as error:
        raise TypeError(
            "expected child_names to be an iterable of names, "
            f"received {type(child_names).__name__}: {child_names!r}"
        ) from error
    for name in names:
        _require_child_name(name)
    if len(set(names)) != len(names):
        raise ValueError(f"runtime child names must be unique, received {names!r}")
    return names


def _require_child_name(name: Any) -> None:
    """Require a non-empty concrete string child name."""
    if type(name) is not str or not name:
        raise TypeError(
            "expected runtime child name to be a non-empty str, "
            f"received {type(name).__name__}: {name!r}"
        )


def _require_checkpointable_child(
    child: Any,
    name: str,
    *,
    require_validator: bool = True,
) -> None:
    """Require checkpoint save, preflight-validation, and restore methods."""
    missing = tuple(
        method_name
        for method_name in (
            "state_dict",
            "load_state_dict",
            *(("validate_state_dict",) if require_validator else ()),
        )
        if not callable(getattr(child, method_name, None))
    )
    if missing:
        raise TypeError(
            f"runtime child {name!r} must provide callable state_dict/load_state_dict/"
            "validate_state_dict; "
            f"missing or non-callable methods: {missing!r}"
        )


def _require_child_payload(payload: Any, name: str) -> Mapping[str, Any]:
    """Require a mapping payload from or for one child state object."""
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"expected runtime child {name!r} state as a mapping, "
            f"received {type(payload).__name__}: {payload!r}"
        )
    return payload


def _require_mapping(value: Any, identifier: str) -> Mapping[str, Any]:
    """Require a mapping for a serialized state layer."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"expected {identifier} as a mapping, received {type(value).__name__}: {value!r}"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected_keys: frozenset[str], identifier: str
) -> None:
    """Require concrete string keys and an exact state schema."""
    non_string_keys = tuple(key for key in value if type(key) is not str)
    if non_string_keys:
        raise TypeError(f"expected {identifier} keys to be str, received {non_string_keys!r}")
    received_keys = frozenset(value)
    if received_keys != expected_keys:
        raise ValueError(
            f"{identifier} keys mismatch: expected {tuple(sorted(expected_keys))!r}, "
            f"received {tuple(sorted(received_keys))!r}"
        )


def _require_counter(value: Any, name: str) -> int:
    """Require a non-negative integer progress counter without bool coercion."""
    if type(value) is not int:
        raise TypeError(
            f"expected trainer runtime progress {name} to be int, "
            f"received {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"expected trainer runtime progress {name} >= 0, received {value}")
    return value


__all__ = [
    "CheckpointableChild",
    "TRAINER_RUNTIME_FORMAT",
    "TRAINER_RUNTIME_METADATA_FILENAME",
    "TRAINER_RUNTIME_STATE_VERSION",
    "TRAINER_RUNTIME_TENSOR_PREFIX",
    "TrainerRuntimeState",
]
