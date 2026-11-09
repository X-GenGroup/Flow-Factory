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

"""Model-agnostic pretrained-load and parameter-storage dtype policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import torch

from ..loading.checkpoint_slices import checkpoint_reader, retype_checkpoint_parameter

DTypePolicy = Optional[Union[torch.dtype, Mapping[str, Optional[torch.dtype]]]]


@dataclass(frozen=True)
class PrecisionCastResult:
    """Counts produced by one contract-aware parameter dtype application."""

    trainable: int = 0
    frozen: int = 0
    protected: int = 0


def parameter_dtype_inventory(
    module: torch.nn.Module,
) -> Dict[str, Dict[torch.dtype, int]]:
    """Count trainable and frozen floating-point parameter elements by dtype."""
    inventory: Dict[str, Dict[torch.dtype, int]] = {
        "trainable": {},
        "frozen": {},
    }
    for parameter in module.parameters():
        if not parameter.is_floating_point():
            continue
        role = "trainable" if parameter.requires_grad else "frozen"
        role_inventory = inventory[role]
        role_inventory[parameter.dtype] = role_inventory.get(parameter.dtype, 0) + parameter.numel()
    return inventory


def _resolve_policy_layer(
    policy: DTypePolicy,
    component_name: str,
    *,
    transformer_names: Sequence[str],
    text_encoder_names: Sequence[str],
) -> tuple[bool, Optional[torch.dtype]]:
    """Resolve one policy layer while preserving a matched null result."""
    if policy is None:
        return False, None
    if isinstance(policy, torch.dtype):
        return True, policy
    if component_name in policy:
        return True, policy[component_name]
    if component_name in transformer_names and "transformers" in policy:
        return True, policy["transformers"]
    if component_name in text_encoder_names and "text_encoders" in policy:
        return True, policy["text_encoders"]
    if "default" in policy:
        return True, policy["default"]
    return False, None


def validate_dtype_policy_selectors(
    policy: DTypePolicy,
    *,
    declared_names: Sequence[str],
) -> None:
    """Reject unknown concrete selectors; role-group selectors may be empty."""
    if not isinstance(policy, Mapping):
        return
    allowed = {*declared_names, "default", "transformers", "text_encoders"}
    unknown = sorted(set(policy) - allowed)
    if unknown:
        raise ValueError(
            "expected dtype policy selectors to name declared components, supported groups, "
            f"or 'default'; received unknown={unknown}, declared={sorted(declared_names)}"
        )


def resolve_component_dtype(
    component_name: str,
    *,
    user_policy: DTypePolicy,
    manifest_policy: DTypePolicy,
    transformer_names: Sequence[str],
    text_encoder_names: Sequence[str],
) -> Optional[torch.dtype]:
    """Resolve user concrete/group/default before the adapter manifest layer."""
    matched, dtype = _resolve_policy_layer(
        user_policy,
        component_name,
        transformer_names=transformer_names,
        text_encoder_names=text_encoder_names,
    )
    if matched:
        return dtype
    _, dtype = _resolve_policy_layer(
        manifest_policy,
        component_name,
        transformer_names=transformer_names,
        text_encoder_names=text_encoder_names,
    )
    return dtype


def component_dtype_mapping(
    *,
    user_policy: DTypePolicy,
    manifest_policy: DTypePolicy,
    component_names: Sequence[str],
    transformer_names: Sequence[str],
    text_encoder_names: Sequence[str],
    manifest_declared_names: Sequence[str] | None = None,
) -> dict[str, torch.dtype]:
    """Resolve user and adapter dtype policies for concrete components.

    Args:
        user_policy: User-selected dtype policy, which takes precedence.
        manifest_policy: Adapter-declared fallback dtype policy.
        component_names: Concrete component names to resolve into the returned mapping.
        transformer_names: Concrete names matched by the ``transformers`` group selector.
        text_encoder_names: Concrete names matched by the ``text_encoders`` group selector.
        manifest_declared_names: Optional superset used only to validate adapter-manifest
            selectors, including class-declared optional components. Resolution still iterates
            ``component_names``. Defaults to ``component_names``.

    Returns:
        Concrete component names mapped to their resolved non-null dtypes.

    Raises:
        ValueError: If either policy contains a selector outside its declared namespace.
    """
    validate_dtype_policy_selectors(user_policy, declared_names=component_names)
    validate_dtype_policy_selectors(
        manifest_policy,
        declared_names=(
            component_names if manifest_declared_names is None else manifest_declared_names
        ),
    )
    return {
        name: dtype
        for name in component_names
        if (
            dtype := resolve_component_dtype(
                name,
                user_policy=user_policy,
                manifest_policy=manifest_policy,
                transformer_names=transformer_names,
                text_encoder_names=text_encoder_names,
            )
        )
        is not None
    }


def build_component_load_dtype_kwargs(
    *,
    user_policy: DTypePolicy,
    manifest_policy: DTypePolicy,
    component_names: Sequence[str],
    transformer_names: Sequence[str],
    text_encoder_names: Sequence[str],
    manifest_declared_names: Sequence[str] | None = None,
    requested_names: Sequence[str] | None = None,
    preserve_unselected: bool = False,
) -> Dict[str, object]:
    """Build the native-loader dtype argument for eager or selective loading.

    Args:
        user_policy: User-selected dtype policy, which takes precedence.
        manifest_policy: Adapter-declared fallback dtype policy.
        component_names: Concrete component names available to the loader.
        transformer_names: Concrete names matched by the ``transformers`` group selector.
        text_encoder_names: Concrete names matched by the ``text_encoders`` group selector.
        manifest_declared_names: Optional superset used only to validate adapter-manifest
            selectors, including class-declared optional components. Resolution still iterates
            ``component_names``. Defaults to ``component_names``.
        requested_names: Optional subset being loaded by this request.
        preserve_unselected: Whether the loader mapping should retain an explicit null default so
            unselected components keep their checkpoint dtype.

    Returns:
        Empty kwargs when no override applies, or one ``dtype`` keyword accepted by the native
        component loader.

    Raises:
        ValueError: If either policy contains a selector outside its declared namespace.
    """
    if isinstance(user_policy, torch.dtype):
        return {"dtype": user_policy}
    if user_policy is None and isinstance(manifest_policy, torch.dtype):
        return {"dtype": manifest_policy}

    concrete = component_dtype_mapping(
        user_policy=user_policy,
        manifest_policy=manifest_policy,
        component_names=component_names,
        transformer_names=transformer_names,
        text_encoder_names=text_encoder_names,
        manifest_declared_names=manifest_declared_names,
    )
    if requested_names is not None:
        requested = set(requested_names)
        concrete = {name: dtype for name, dtype in concrete.items() if name in requested}
    if not concrete:
        return {}
    if preserve_unselected:
        return {"dtype": {"default": None, **concrete}}
    return {"dtype": concrete}


def _matches_parameter_path(name: str, patterns: Sequence[str]) -> bool:
    segments = name.split(".")
    return any(pattern in segments for pattern in patterns)


def _protected_fp32_patterns(module: torch.nn.Module, target_dtype: torch.dtype) -> tuple[str, ...]:
    """Return the loader-declared FP32 paths that apply to one target dtype."""
    if target_dtype not in (torch.float16, torch.bfloat16):
        return ()

    get_base_model = getattr(module, "get_base_model", None)
    contract_module = get_base_model() if callable(get_base_model) else module
    if not isinstance(contract_module, torch.nn.Module):
        raise TypeError(
            f"expected get_base_model() from {type(module).__name__} to return nn.Module, "
            f"received {type(contract_module).__name__}"
        )
    strict = tuple(getattr(contract_module, "_keep_in_fp32_modules_strict", None) or ())
    legacy = tuple(getattr(contract_module, "_keep_in_fp32_modules", None) or ())
    module_namespace = type(contract_module).__module__
    if module_namespace.startswith("diffusers."):
        return tuple(dict.fromkeys((*legacy, *strict)))
    if target_dtype is torch.float16:
        return tuple(dict.fromkeys((*legacy, *strict)))
    return strict


def _is_quantized(module: torch.nn.Module) -> bool:
    candidates = [module]
    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if isinstance(base_model, torch.nn.Module):
            candidates.append(base_model)
    return any(
        getattr(candidate, "is_quantized", False)
        or getattr(candidate, "hf_quantizer", None) is not None
        or getattr(candidate, "quantization_method", None) is not None
        for candidate in candidates
    )


def cast_module_role_dtypes(
    module: torch.nn.Module,
    *,
    component_name: str,
    trainable_dtype: torch.dtype,
    frozen_dtype: Optional[torch.dtype],
    force_uniform_dtype: Optional[torch.dtype] = None,
    is_adapter_parameter: Optional[Callable[[str], bool]] = None,
) -> PrecisionCastResult:
    """Apply role storage dtypes without violating model-declared FP32 islands."""
    if not isinstance(trainable_dtype, torch.dtype):
        raise TypeError(
            f"expected torch.dtype for trainable component={component_name!r}, "
            f"received {type(trainable_dtype).__name__}: {trainable_dtype!r}"
        )
    if frozen_dtype is not None and not isinstance(frozen_dtype, torch.dtype):
        raise TypeError(
            f"expected torch.dtype or None for frozen component={component_name!r}, "
            f"received {type(frozen_dtype).__name__}: {frozen_dtype!r}"
        )
    if force_uniform_dtype is not None and not isinstance(force_uniform_dtype, torch.dtype):
        raise TypeError(
            f"expected torch.dtype or None for uniform component={component_name!r}, "
            f"received {type(force_uniform_dtype).__name__}: {force_uniform_dtype!r}"
        )

    desired_dtypes = {trainable_dtype}
    if frozen_dtype is not None:
        desired_dtypes.add(frozen_dtype)
    if force_uniform_dtype is not None:
        desired_dtypes = {force_uniform_dtype}
    adapter_predicate = is_adapter_parameter or (lambda _: False)
    quantized = _is_quantized(module)
    if quantized:
        conflicts = [
            (
                name,
                parameter.dtype,
                force_uniform_dtype
                or (trainable_dtype if parameter.requires_grad else frozen_dtype),
            )
            for name, parameter in module.named_parameters()
            if parameter.is_floating_point()
            and not adapter_predicate(name)
            and (
                force_uniform_dtype
                or (trainable_dtype if parameter.requires_grad else frozen_dtype)
            )
            not in (None, parameter.dtype)
        ]
        if conflicts:
            name, received, expected = conflicts[0]
            raise ValueError(
                f"cannot cast quantized component={component_name!r} parameter={name!r}; "
                f"expected existing dtype {received}, received requested dtype {expected}"
            )

    protected_by_dtype = {
        dtype: _protected_fp32_patterns(module, dtype) for dtype in desired_dtypes
    }
    trainable_count = 0
    frozen_count = 0
    protected_count = 0

    for name, parameter in module.named_parameters():
        if not parameter.is_floating_point():
            continue
        target_dtype = force_uniform_dtype or (
            trainable_dtype if parameter.requires_grad else frozen_dtype
        )
        if target_dtype is None:
            continue
        protected = not adapter_predicate(name) and _matches_parameter_path(
            name, protected_by_dtype[target_dtype]
        )
        if protected:
            target_dtype = torch.float32
            protected_count += 1
        if parameter.dtype != target_dtype:
            if checkpoint_reader(parameter) is not None:
                retype_checkpoint_parameter(parameter, target_dtype)
            else:
                parameter.data = parameter.data.to(dtype=target_dtype)
        if parameter.requires_grad:
            trainable_count += 1
        else:
            frozen_count += 1

    buffer_dtype = force_uniform_dtype or frozen_dtype
    if buffer_dtype is not None and not quantized:
        patterns = protected_by_dtype[buffer_dtype]
        for name, buffer in module.named_buffers():
            if not buffer.is_floating_point():
                continue
            target_dtype = (
                torch.float32 if _matches_parameter_path(name, patterns) else buffer_dtype
            )
            if target_dtype is torch.float32:
                protected_count += 1
            if buffer.dtype != target_dtype:
                buffer.data = buffer.data.to(dtype=target_dtype)

    return PrecisionCastResult(
        trainable=trainable_count,
        frozen=frozen_count,
        protected=protected_count,
    )
