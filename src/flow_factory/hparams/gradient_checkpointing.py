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

"""Gradient-checkpointing policy configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Union

GradientCheckpointingMode = Literal["full", "none", "fraction", "every_n", "layers"]


@dataclass(frozen=True)
class GradientCheckpointingSpec:
    """Select model blocks whose activations are recomputed during backward."""

    mode: GradientCheckpointingMode
    fraction: float | None = None
    every_n: int | None = None
    layers: tuple[int, ...] | None = None


GradientCheckpointingPolicy = Union[bool, GradientCheckpointingSpec, Mapping[str, Any]]


def normalize_gradient_checkpointing_policy(
    value: GradientCheckpointingPolicy,
) -> bool | GradientCheckpointingSpec:
    """Normalize a full/disabled bool or one selective policy mapping."""
    if isinstance(value, bool):
        return value
    if isinstance(value, GradientCheckpointingSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "expected train.enable_gradient_checkpointing as bool, mapping, or "
            f"GradientCheckpointingSpec, received {type(value).__name__}: {value!r}"
        )

    allowed = {"mode", "fraction", "every_n", "layers"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "expected train.enable_gradient_checkpointing keys in "
            f"{sorted(allowed)}, received unknown={unknown}"
        )

    configured_modes = [
        name for name in ("fraction", "every_n", "layers") if value.get(name) is not None
    ]
    mode = value.get("mode")
    if mode is None:
        if len(configured_modes) != 1:
            raise ValueError(
                "expected exactly one of fraction, every_n, or layers when checkpoint "
                f"mode is omitted, received configured={configured_modes}"
            )
        mode = configured_modes[0]
    if mode not in {"full", "none", "fraction", "every_n", "layers"}:
        raise ValueError(
            "expected checkpoint mode in ('full', 'none', 'fraction', 'every_n', "
            f"'layers'), received {mode!r}"
        )

    fraction = value.get("fraction")
    every_n = value.get("every_n")
    layers = value.get("layers")
    expected_field = {
        "fraction": "fraction",
        "every_n": "every_n",
        "layers": "layers",
    }.get(mode)
    extraneous = [
        name
        for name, configured in (
            ("fraction", fraction),
            ("every_n", every_n),
            ("layers", layers),
        )
        if configured is not None and name != expected_field
    ]
    if extraneous:
        raise ValueError(f"checkpoint mode={mode!r} does not accept fields {sorted(extraneous)}")

    if mode == "fraction":
        if not isinstance(fraction, Real) or isinstance(fraction, bool):
            raise TypeError(
                f"expected checkpoint fraction as a real number, received "
                f"{type(fraction).__name__}: {fraction!r}"
            )
        if not math.isfinite(float(fraction)) or not 0 < float(fraction) <= 1:
            raise ValueError(f"expected checkpoint fraction in (0, 1], received {fraction!r}")
        return GradientCheckpointingSpec(mode=mode, fraction=float(fraction))

    if mode == "every_n":
        if not isinstance(every_n, int) or isinstance(every_n, bool):
            raise TypeError(
                f"expected checkpoint every_n as int, received "
                f"{type(every_n).__name__}: {every_n!r}"
            )
        if every_n < 1:
            raise ValueError(f"expected checkpoint every_n as positive int, received {every_n!r}")
        return GradientCheckpointingSpec(mode=mode, every_n=every_n)

    if mode == "layers":
        if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
            raise TypeError(
                "expected checkpoint layers as a sequence of non-negative ints, "
                f"received {type(layers).__name__}: {layers!r}"
            )
        normalized_layers = []
        for index in layers:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError(
                    "expected checkpoint layer index as int, "
                    f"received {type(index).__name__}: {index!r}"
                )
            if index < 0:
                raise ValueError(f"expected checkpoint layer index >= 0, received {index!r}")
            if index not in normalized_layers:
                normalized_layers.append(index)
        if not normalized_layers:
            raise ValueError("expected checkpoint layers to contain at least one index")
        return GradientCheckpointingSpec(mode=mode, layers=tuple(normalized_layers))

    if configured_modes:
        raise ValueError(
            f"checkpoint mode={mode!r} does not accept selector fields {configured_modes}"
        )
    return GradientCheckpointingSpec(mode=mode)


def gradient_checkpointing_enabled(
    value: bool | GradientCheckpointingSpec,
) -> bool:
    """Return whether a normalized policy activates any checkpoint boundary."""
    return value if isinstance(value, bool) else value.mode != "none"


def serialize_gradient_checkpointing_policy(
    value: bool | GradientCheckpointingSpec,
) -> bool | dict[str, Any]:
    """Serialize one normalized policy for YAML/config logging."""
    if isinstance(value, bool):
        return value
    serialized: dict[str, Any] = {"mode": value.mode}
    if value.fraction is not None:
        serialized["fraction"] = value.fraction
    if value.every_n is not None:
        serialized["every_n"] = value.every_n
    if value.layers is not None:
        serialized["layers"] = list(value.layers)
    return serialized
