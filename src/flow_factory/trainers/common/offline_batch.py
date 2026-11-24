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

"""Offline batch utilities that keep dataset, model, and algorithm fields separate."""

from collections.abc import Mapping, Set
from dataclasses import is_dataclass
from typing import Any, Dict, Union

import torch

from ...contracts import NON_MODEL_CONDITION_KEYS
from ...data_utils.offline_conditions import encode_raw_offline_conditions
from ...data_utils.offline_train_data import offline_preprocess_kwargs
from ...models.condition_state import PreparedConditionState


def resolve_offline_condition(condition: Mapping[str, Any], trainer: Any) -> Dict[str, Any]:
    """Encode raw input once per record and move the resulting batch to the policy device."""
    if "_offline_raw" in condition:
        preprocess = trainer.adapter.preprocess_func
        kwargs = offline_preprocess_kwargs(trainer.config, preprocess, None)
        with torch.no_grad():
            condition = encode_raw_offline_conditions(condition["_offline_raw"], preprocess, kwargs)
    return move_condition_to_device(condition, trainer.accelerator.device)


def move_condition_to_device(
    condition: Mapping[str, Any],
    device: Union[torch.device, str],
    *,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    """Copy a cached condition tree while moving only its tensor leaves.

    Args:
        condition: Input-only model condition mapping from an offline batch.
        device: Destination device for tensor leaves.
        non_blocking: Forwarded to :meth:`torch.Tensor.to`.

    Returns:
        A new mutable mapping whose container tree is detached from the cache row.
    """
    _require_string_key_mapping(condition, "offline condition")
    if not isinstance(non_blocking, bool):
        raise TypeError(f"non_blocking must be a bool, got {non_blocking!r}")
    target_device = torch.device(device)
    return {
        key: _move_condition_value(value, target_device, non_blocking=non_blocking)
        for key, value in condition.items()
    }


def bind_output_forward_context(
    condition: Mapping[str, Any],
    forward_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind input conditions and output-derived model fields without mutation.

    Args:
        condition: Cached input-only model fields.
        forward_context: Adapter-owned fields derived from output geometry.

    Returns:
        A new model-conditioning mapping.

    Raises:
        TypeError: If either argument is not a string-keyed mapping.
        ValueError: If either side contains non-model fields or both own one key.
    """
    _require_string_key_mapping(condition, "offline condition")
    _require_string_key_mapping(forward_context, "output forward context")
    _reject_non_model_keys(condition, "offline condition")
    _reject_non_model_keys(forward_context, "output forward context")
    collisions = tuple(sorted(set(condition).intersection(forward_context)))
    if collisions:
        raise ValueError(
            "output forward context collides with cached condition keys "
            f"{collisions}; input and output fields must have one owner"
        )
    return {**condition, **forward_context}


def bind_prepared_condition_output(
    prepared: PreparedConditionState,
    output_forward_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind one realized input condition to candidate-specific output fields.

    Args:
        prepared: Input-owned realization shared by every candidate for a request.
        output_forward_context: Candidate-specific output-derived model fields.

    Returns:
        A new complete model-forward mapping.
    """
    if not isinstance(prepared, PreparedConditionState):
        raise TypeError(
            "expected PreparedConditionState for prepared offline condition, "
            f"received {type(prepared).__name__}: {prepared!r}"
        )
    return bind_output_forward_context(
        prepared.model_forward_condition(),
        output_forward_context,
    )


def _move_condition_value(
    value: Any,
    device: torch.device,
    *,
    non_blocking: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        _require_string_key_mapping(value, "nested offline condition")
        return {
            key: _move_condition_value(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_condition_value(item, device, non_blocking=non_blocking) for item in value]
    if type(value) is tuple:
        return tuple(
            _move_condition_value(item, device, non_blocking=non_blocking) for item in value
        )
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        moved = (_move_condition_value(item, device, non_blocking=non_blocking) for item in value)
        return type(value)(*moved)
    if isinstance(value, torch.Size):
        return torch.Size(value)
    if isinstance(value, Set) or is_dataclass(value):
        raise TypeError(
            "offline condition trees support Mapping, list, tuple, namedtuple, and tensor "
            f"containers only; received unsupported {type(value).__name__}"
        )
    return value


def _require_string_key_mapping(value: Any, identifier: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"expected Mapping[str, Any] for {identifier}, received "
            f"{type(value).__name__}: {value!r}"
        )
    non_string_keys = tuple(key for key in value if not isinstance(key, str))
    if non_string_keys:
        raise TypeError(f"expected string keys for {identifier}, received {non_string_keys!r}")
    empty_keys = tuple(key for key in value if not key)
    if empty_keys:
        raise ValueError(
            f"expected non-empty string keys for {identifier}, received {empty_keys!r}"
        )


def _reject_non_model_keys(value: Mapping[str, Any], identifier: str) -> None:
    rejected = tuple(sorted(set(value).intersection(NON_MODEL_CONDITION_KEYS)))
    if rejected:
        raise ValueError(
            f"{identifier} contains fields that cannot enter model forward: {rejected}"
        )


__all__ = [
    "bind_output_forward_context",
    "bind_prepared_condition_output",
    "move_condition_to_device",
]
