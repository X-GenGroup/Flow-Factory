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

"""Training arguments shared by distribution-matching distillation trainers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Tuple, cast

from ...contracts.execution import (
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    ExecutionContract,
)
from ..optimizer_args import AdamWOptimizerArguments
from ._base import TrainingArguments

if TYPE_CHECKING:
    from ...models.roles import RoleName


def _finite_float(value: object, field_name: str, *, allow_zero: bool) -> float:
    """Convert and validate one finite optimizer scalar."""
    if isinstance(value, bool):
        raise TypeError(
            f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
        )
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"expected numeric {field_name}, received {type(value).__name__}: {value!r}"
        ) from error
    if not math.isfinite(converted) or converted < 0 or (converted == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"expected finite {field_name} {comparator}, received {value!r}")
    return converted


@dataclass
class DMD2TrainingArguments(TrainingArguments):
    """Configure data-free DMD2 distribution matching."""

    execution_contract: ClassVar[ExecutionContract] = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    gradient_step_per_epoch: int = field(
        default=1,
        metadata={"help": "DMD2 requires one generator optimizer step per rollout."},
    )
    ttur_fake_updates: int = 5
    perturbation_timestep_range: Tuple[float, float] = (0.02, 0.98)
    # The real score is the only role that wants classifier-free guidance: it supplies
    # the target distribution, and a guided teacher is what makes the match worth
    # anything. The generator rolls out CFG-free and the fake score must model exactly
    # what the generator produces, so both keep `train.guidance_scale` (normally 1.0).
    # Left unset, the real score follows `train.guidance_scale`.
    real_guidance_scale: Optional[float] = None
    # The boundary replay must reproduce the stored rollout point. Scheduler dtype
    # round-trips make the supported ODE path bit-exact; configurable tolerances remain
    # for adapters with model-specific numerical seams, and the replay error reports the
    # discrepancy in each original dtype's ULPs rather than encouraging a blind increase.
    replay_rtol: float = 1e-4
    replay_atol: float = 1e-4

    def get_preprocess_guidance_scale(self) -> float:
        """Account for the real score's guidance: the generator samples CFG-free.

        Preprocessing decides from this whether to encode negative prompts. Reading only
        `guidance_scale` would skip them for exactly the configuration this algorithm
        wants -- a CFG-free generator beside a guided real score -- and the guided query
        would then find no negatives and silently fall back to no guidance.

        Returns:
            The largest guidance scale any stage of this run may ask for.
        """
        if self.real_guidance_scale is None:
            return self.guidance_scale
        return max(self.guidance_scale, self.real_guidance_scale)

    def get_reference_guidance_scale(self) -> float:
        """Return the guidance scale the real score is actually queried at.

        Returns:
            ``real_guidance_scale`` when set, otherwise the shared scale.
        """
        if self.real_guidance_scale is None:
            return self.guidance_scale
        return self.real_guidance_scale

    def __post_init__(self) -> None:
        """Validate DMD2 controls."""
        super().__post_init__()
        self.replay_rtol = self._validate_replay_tolerance(self.replay_rtol, "train.replay_rtol")
        self.replay_atol = self._validate_replay_tolerance(self.replay_atol, "train.replay_atol")
        if self.real_guidance_scale is not None:
            self.real_guidance_scale = _finite_float(
                self.real_guidance_scale, "train.real_guidance_scale", allow_zero=True
            )
            if self.real_guidance_scale < 1.0:
                raise ValueError(
                    "expected train.real_guidance_scale >= 1.0, "
                    f"received {self.real_guidance_scale!r}"
                )
        if (
            not isinstance(self.ttur_fake_updates, int)
            or isinstance(self.ttur_fake_updates, bool)
            or self.ttur_fake_updates < 1
        ):
            raise ValueError(
                "expected train.ttur_fake_updates as an int >= 1, "
                f"received {self.ttur_fake_updates!r}"
            )
        if (
            not isinstance(self.perturbation_timestep_range, (tuple, list))
            or len(self.perturbation_timestep_range) != 2
        ):
            raise TypeError(
                "expected train.perturbation_timestep_range as a two-item tuple or list, "
                f"received {type(self.perturbation_timestep_range).__name__}: "
                f"{self.perturbation_timestep_range!r}"
            )
        lower, upper = (
            _finite_float(
                self.perturbation_timestep_range[0],
                "train.perturbation_timestep_range[0]",
                allow_zero=True,
            ),
            _finite_float(
                self.perturbation_timestep_range[1],
                "train.perturbation_timestep_range[1]",
                allow_zero=True,
            ),
        )
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                "expected train.perturbation_timestep_range to satisfy "
                f"0 <= lower < upper <= 1, received {(lower, upper)!r}"
            )
        self.perturbation_timestep_range = (lower, upper)

    def role_update_plan(self) -> "RoleUpdatePlan":
        """Return fake TTUR phases followed by one generator phase."""
        # Constraint #22(c): trainers.role_optimization imports training args.
        from ...trainers.role_optimization import RolePhase, RoleUpdatePlan

        return RoleUpdatePlan(
            phases=(
                RolePhase("fake", repeats=self.ttur_fake_updates),
                RolePhase("generator"),
            )
        )

    @property
    def required_trainable_roles(self) -> Tuple[RoleName, ...]:
        """Return trainable roles in canonical materialization order."""
        return ("generator", "fake")

    @property
    def requires_ref_model(self) -> bool:
        """Require the frozen pretrained score reference."""
        return True

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


DMD2_DEFAULT_OPTIMIZERS: Tuple[AdamWOptimizerArguments, ...] = (
    AdamWOptimizerArguments(name="generator", learning_rate=1e-5),
    AdamWOptimizerArguments(name="fake", learning_rate=1e-5),
)
"""Per-role defaults for roles that have no ``optimizers`` entry in the config file.

These are the algorithm's own numbers, so they live beside its training arguments
rather than in the framework. A role named in ``optimizers`` uses that entry instead,
which is also how a run selects Muon for a role.
"""
