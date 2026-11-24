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

"""Shared training arguments for finite offline flow-matching objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, Tuple, Union

from ._base import TrainingArguments


@dataclass
class OfflineFlowMatchingTrainingArguments(TrainingArguments):
    """Configure model-agnostic flow matching over a finite offline loader."""

    max_epochs: int = field(
        default=1,
        metadata={
            "help": (
                "Number of complete offline dataloader traversals. One successful traversal "
                "is one data epoch; partial traversals do not advance this counter."
            )
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Explicit number of offline dataloader microbatches per optimizer step. "
                "Offline training does not derive this value from grouped rollout geometry."
            )
        },
    )
    weighting_scheme: Literal["logit_normal", "uniform"] = field(
        default="logit_normal",
        metadata={"help": "Distribution used to sample independent flow-matching timesteps."},
    )
    num_train_timesteps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of independently sampled Monte Carlo timestep terms averaged per "
                "offline example. This value does not multiply gradient accumulation."
            )
        },
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": (
                "Fraction range along the denoising axis from scheduler time 1000 to 0. "
                "A scalar upper bound is normalized to (0, upper)."
            )
        },
    )
    time_shift: float = field(
        default=1.0,
        metadata={
            "help": "Noise-odds multiplier. 1.0 is neutral; larger values favor higher noise."
        },
    )
    logit_mean: float = field(
        default=0.0,
        metadata={"help": "Finite mean of the logit-normal timestep distribution."},
    )
    logit_std: float = field(
        default=1.0,
        metadata={"help": "Positive standard deviation of the logit-normal distribution."},
    )

    def __post_init__(self) -> None:
        """Normalize and validate finite offline optimization controls."""
        super().__post_init__()
        self.max_epochs = _positive_int(self.max_epochs, "train.max_epochs")
        self.num_train_timesteps = _positive_int(
            self.num_train_timesteps,
            "train.num_train_timesteps",
        )
        if not isinstance(self.weighting_scheme, str):
            raise TypeError(
                "expected train.weighting_scheme as a string, received "
                f"{type(self.weighting_scheme).__name__}: {self.weighting_scheme!r}"
            )
        if self.weighting_scheme not in ("logit_normal", "uniform"):
            raise ValueError(
                "expected train.weighting_scheme to be 'logit_normal' or 'uniform', "
                f"received {self.weighting_scheme!r}"
            )
        self.timestep_range = _timestep_range(self.timestep_range)
        self.time_shift = _finite_float(
            self.time_shift,
            "train.time_shift",
            strictly_positive=True,
        )
        self.logit_mean = _finite_float(
            self.logit_mean,
            "train.logit_mean",
            strictly_positive=False,
        )
        self.logit_std = _finite_float(
            self.logit_std,
            "train.logit_std",
            strictly_positive=True,
        )


def _positive_int(value: object, field_name: str) -> int:
    """Require one positive integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} as an int >= 1, received " f"{type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(f"expected {field_name} >= 1, received {value}")
    return value


def _finite_float(value: object, field_name: str, *, strictly_positive: bool) -> float:
    """Require one finite real scalar and optionally require positivity."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"expected {field_name} as a finite real number, received "
            f"{type(value).__name__}: {value!r}"
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"expected finite {field_name}, received {value!r}")
    if strictly_positive and converted <= 0:
        raise ValueError(f"expected {field_name} > 0, received {value!r}")
    return converted


def _timestep_range(value: object) -> Tuple[float, float]:
    """Normalize one strict denoising-axis fraction range."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                "expected train.timestep_range as a scalar or two-item sequence, "
                f"received {value!r}"
            )
        lower = _finite_float(
            value[0],
            "train.timestep_range[0]",
            strictly_positive=False,
        )
        upper = _finite_float(
            value[1],
            "train.timestep_range[1]",
            strictly_positive=False,
        )
    else:
        lower = 0.0
        upper = _finite_float(
            value,
            "train.timestep_range",
            strictly_positive=False,
        )
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            "expected train.timestep_range to satisfy 0 <= lower < upper <= 1, "
            f"received {(lower, upper)!r}"
        )
    return lower, upper


__all__ = ["OfflineFlowMatchingTrainingArguments"]
