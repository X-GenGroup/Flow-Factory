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

"""Model-declared operation precision, independent of parameter storage dtype."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from inspect import Parameter, signature
from types import MethodType
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.overrides import TorchFunctionMode

_SDPA = F.scaled_dot_product_attention


@dataclass(frozen=True)
class ComputePrecisionRule:
    """Select an operation inside model-relative module paths.

    Only native SDPA in FP32 is currently qualified. Additional operations must
    supply their own argument, output and backward contract before opting in.
    """

    module_pattern: str
    operation: Literal["sdpa"] = "sdpa"
    dtype: Literal["float32"] = "float32"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.module_pattern, str)
            or not self.module_pattern
            or self.operation != "sdpa"
            or self.dtype != "float32"
        ):
            raise ValueError(f"unsupported compute precision rule: {self!r}")


@dataclass(frozen=True)
class ComputePrecisionPolicy:
    """Versioned model-owned rules applied to each declared transformer."""

    name: str
    rules: tuple[ComputePrecisionRule, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.rules, tuple)
            or not self.rules
            or any(not isinstance(rule, ComputePrecisionRule) for rule in self.rules)
        ):
            raise ValueError(f"invalid compute precision policy: {self!r}")


def resolve_compute_precision_policy(
    adapter_cls: type, config: Any
) -> ComputePrecisionPolicy | None:
    """Resolve an explicit model policy and reject incompatible accelerators."""
    name = getattr(config.model_args, "compute_precision_policy", None)
    if name is None:
        return None
    policies = adapter_cls.compute_precision_policies
    matches = [policy for policy in policies if policy.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"{adapter_cls.__name__} does not uniquely declare compute_precision_policy={name!r}; "
            f"available={tuple(policy.name for policy in policies)}"
        )
    acceleration = config.acceleration_args
    if acceleration.rollout or any(
        spec.name != "attention_backend" or spec.params != {"backend": "native"}
        for spec in acceleration.shared
    ):
        raise ValueError(
            f"compute_precision_policy={name!r} supports only native attention and no other accelerators"
        )
    return matches[0]


def _sdpa_float32(
    operation: Callable,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32) or any(
        tensor.dtype != query.dtype or tensor.device != query.device for tensor in (key, value)
    ):
        raise ValueError(
            "FP32 SDPA requires colocated FP16/BF16/FP32 operands with identical dtypes; "
            f"received {tuple((tensor.dtype, tensor.device) for tensor in (query, key, value))}"
        )
    with torch.autocast(device_type=query.device.type, enabled=False):
        output = operation(
            query.float(),
            key.float(),
            value.float(),
            attn_mask=(
                attn_mask.float()
                if attn_mask is not None and attn_mask.is_floating_point()
                else attn_mask
            ),
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    return output.to(query.dtype)


class _OperationPrecisionMode(TorchFunctionMode):
    """A per-forward scope; neither global functions nor tensor types change."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is _SDPA:
            self.calls += 1
            return _sdpa_float32(func, *args, **kwargs)
        return func(*args, **kwargs)


def _precision_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    # Store the original bound method on the module, not in a closure: deepcopy
    # of a full-weight variant must rebind it to the copied module's parameters.
    with _OperationPrecisionMode() as mode:
        output = module._ff_precision_original_forward(*args, **kwargs)
    if mode.calls == 0:
        raise RuntimeError(
            f"compute precision rule {module._ff_precision_rule!r} matched no native SDPA operation; "
            "the attention processor/backend must preserve the declared operation"
        )
    return output


def _bind_precision_forward(module: nn.Module) -> Callable:
    def forward(owner: nn.Module, *args: Any, **kwargs: Any) -> Any:
        return _precision_forward(owner, *args, **kwargs)

    # TP region entries resolve positional arguments from this signature. Keep
    # it per instance and avoid __wrapped__ retaining another variant's module.
    forward.__signature__ = signature(module.forward).replace(
        parameters=(
            Parameter("_precision_owner", Parameter.POSITIONAL_ONLY),
            *signature(module.forward).parameters.values(),
        )
    )
    return MethodType(forward, module)


def apply_compute_precision_policy(
    module: nn.Module, policy: ComputePrecisionPolicy
) -> tuple[str, ...]:
    """Install model rules without changing parameters, module paths or state keys.

    All patterns are validated before mutation. The returned paths are suitable
    for exact-resume identity and diagnostics. A PEFT caller passes its base model.
    Reapplying the same rule is idempotent, including after full-variant cloning.
    """
    selected: dict[str, tuple[nn.Module, ComputePrecisionRule]] = {}
    for rule in policy.rules:
        matches = [
            (name, child)
            for name, child in module.named_modules()
            if fnmatchcase(name, rule.module_pattern)
        ]
        if not matches:
            raise ValueError(
                f"compute precision policy {policy.name!r} matched no modules for {rule.module_pattern!r}"
            )
        for name, child in matches:
            if any(
                not name
                or not other
                or name == other
                or name.startswith(other + ".")
                or other.startswith(name + ".")
                for other in selected
            ):
                raise ValueError(
                    f"compute precision policy {policy.name!r} has overlapping module scopes at {name!r}"
                )
            existing = getattr(child, "_ff_precision_rule", None)
            if existing is not None and existing != rule:
                raise ValueError(
                    f"module {name!r} already owns a different compute precision rule: {existing!r}"
                )
            selected[name] = (child, rule)
    for child, rule in selected.values():
        if getattr(child, "_ff_precision_rule", None) is None:
            child._ff_precision_original_forward = child.forward
            child._ff_precision_rule = rule
            child.forward = _bind_precision_forward(child)
        # Pin selected Diffusers modules to native SDPA even when the process
        # default or a surrounding backend context requests another algorithm.
        if callable(getattr(child, "set_attention_backend", None)):
            child.set_attention_backend("native")
    return tuple(selected)
