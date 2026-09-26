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

"""Training arguments for deterministic trajectory distribution matching."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

from .dmd2 import DMD2TrainingArguments, _finite_float


@dataclass(frozen=True)
class TDMQuerySamplingPolicy:
    """Describe the coordinate-independent TDM score-query policy."""

    interval: Literal["trajectory", "reverse"]
    distribution: Literal["actual_uniform", "conditional_logit_normal", "source_uniform"]
    max_sigma: float
    logit_mean: float
    logit_std: float

    def __post_init__(self) -> None:
        """Validate and canonicalize the query policy."""
        if self.interval not in ("trajectory", "reverse"):
            raise ValueError(
                "train.tdm_query_interval must be 'trajectory' or 'reverse', "
                f"received {self.interval!r}"
            )
        if self.distribution not in (
            "actual_uniform",
            "conditional_logit_normal",
            "source_uniform",
        ):
            raise ValueError(
                "train.tdm_query_distribution must be 'actual_uniform', "
                "'conditional_logit_normal', or 'source_uniform', "
                f"received {self.distribution!r}"
            )
        max_sigma = _finite_float(
            self.max_sigma,
            "train.tdm_query_max_sigma",
            allow_zero=False,
        )
        if max_sigma > 1:
            raise ValueError("train.tdm_query_max_sigma must be in (0, 1]")
        if isinstance(self.logit_mean, bool):
            raise TypeError("train.tdm_query_logit_mean must be a finite number, not bool")
        try:
            logit_mean = float(self.logit_mean)
        except (TypeError, ValueError) as error:
            raise TypeError("train.tdm_query_logit_mean must be a finite number") from error
        if not math.isfinite(logit_mean):
            raise ValueError("train.tdm_query_logit_mean must be finite")
        logit_std = _finite_float(
            self.logit_std,
            "train.tdm_query_logit_std",
            allow_zero=False,
        )
        object.__setattr__(self, "max_sigma", max_sigma)
        object.__setattr__(self, "logit_mean", logit_mean)
        object.__setattr__(self, "logit_std", logit_std)

    @property
    def requires_generation_shift(self) -> bool:
        """Return whether sampling requires rollout-owned shift provenance."""
        return self.distribution == "source_uniform"

    @classmethod
    def from_training_args(cls, training_args: Any) -> "TDMQuerySamplingPolicy":
        """Build a policy from the public TDM training fields.

        Args:
            training_args: TDM argument object or a contract-compatible test host.

        Returns:
            Validated immutable query policy.
        """
        return cls(
            interval=training_args.tdm_query_interval,
            distribution=training_args.tdm_query_distribution,
            max_sigma=training_args.tdm_query_max_sigma,
            logit_mean=training_args.tdm_query_logit_mean,
            logit_std=training_args.tdm_query_logit_std,
        )

    def resume_identity(self) -> Dict[str, Any]:
        """Return only policy fields that affect the selected objective.

        Returns:
            Canonical exact-resume identity for the active query policy.
        """
        identity: Dict[str, Any] = {
            "version": 2,
            "interval": self.interval,
            "distribution": self.distribution,
        }
        if self.interval == "reverse":
            identity["max_sigma"] = self.max_sigma
        if self.distribution == "conditional_logit_normal":
            identity["logit_mean"] = self.logit_mean
            identity["logit_std"] = self.logit_std
        return identity


@dataclass
class TDMTrainingArguments(DMD2TrainingArguments):
    """Configure deterministic few-step trajectory distribution matching."""

    gradient_step_per_epoch: int = field(
        default=1,
        metadata={"help": "TDM requires one generator optimizer step per rollout."},
    )
    num_inference_steps: int = 4
    use_huber: bool = True
    huber_c: float = 1e-3
    tdm_snr_gamma: float = 5.0
    tdm_importance_clip: float = 20.0
    tdm_query_interval: Literal["trajectory", "reverse"] = "trajectory"
    tdm_query_distribution: Literal[
        "actual_uniform", "conditional_logit_normal", "source_uniform"
    ] = "actual_uniform"
    tdm_query_max_sigma: float = 1.0
    tdm_query_logit_mean: float = 0.0
    tdm_query_logit_std: float = 1.0

    def __post_init__(self) -> None:
        """Validate trajectory count, replay tolerances, and Huber controls."""
        super().__post_init__()
        policy = self.query_sampling_policy()
        self.tdm_query_max_sigma = policy.max_sigma
        self.tdm_query_logit_mean = policy.logit_mean
        self.tdm_query_logit_std = policy.logit_std
        if not isinstance(self.use_huber, bool):
            raise TypeError(
                f"expected train.use_huber as a bool, received {type(self.use_huber).__name__}: "
                f"{self.use_huber!r}"
            )
        self.huber_c = _finite_float(self.huber_c, "train.huber_c", allow_zero=False)
        self.tdm_snr_gamma = _finite_float(
            self.tdm_snr_gamma,
            "train.tdm_snr_gamma",
            allow_zero=False,
        )
        self.tdm_importance_clip = _finite_float(
            self.tdm_importance_clip,
            "train.tdm_importance_clip",
            allow_zero=False,
        )

    def query_sampling_policy(self) -> TDMQuerySamplingPolicy:
        """Build the immutable score-query policy.

        Returns:
            Validated policy used by trajectory replay and resume identity.
        """
        return TDMQuerySamplingPolicy.from_training_args(self)

    def get_num_train_timesteps(self, args: Any) -> int:
        """Count one backend accumulation unit per trajectory boundary.

        Args:
            args: Parent arguments object; unused because TDM owns the unit count.

        Returns:
            Number of independently backpropagated trajectory boundaries.
        """
        del args
        return self.num_inference_steps

    @staticmethod
    def _validate_replay_tolerance(value: object, field_name: str) -> float:
        """Convert and validate one non-negative finite replay tolerance."""
        if isinstance(value, bool):
            raise TypeError(
                f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
            )
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
            ) from error
        if not math.isfinite(converted) or converted < 0:
            raise ValueError(f"expected finite {field_name} >= 0, received {value!r}")
        return converted
