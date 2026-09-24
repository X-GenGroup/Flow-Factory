# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training arguments for TDM-R1 fake-surrogate-generator distillation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Tuple

from ...contracts.execution import ONLINE_EXECUTION_CONTRACT, ExecutionContract
from ...contracts.sampler import (
    WHOLE_GROUP_BATCH_SAMPLER_SELECTION,
    SamplerSelectionContract,
)
from ..optimizer_args import AdamWOptimizerArguments
from .dmd2 import _finite_float
from .tdm import TDMTrainingArguments

if TYPE_CHECKING:
    from ...models.roles import RoleName
    from ...trainers.role_optimization import RoleUpdatePlan


@dataclass
class TDMR1TrainingArguments(TDMTrainingArguments):
    """Configure TDM-R1 with a learned surrogate and frozen reference."""

    execution_contract: ClassVar[ExecutionContract] = ONLINE_EXECUTION_CONTRACT
    sampler_selection_contract: ClassVar[SamplerSelectionContract] = (
        WHOLE_GROUP_BATCH_SAMPLER_SELECTION
    )

    global_std: bool = True
    advantage_aggregation: Literal["sum", "gdpo"] = "gdpo"
    tdm_weight: float = 0.3
    surrogate_preference_beta: float = 1.0
    advantage_clip_range: float = 5.0
    use_huber: bool = False
    surrogate_reference_beta: float = 0.001
    surrogate_reference_threshold: float = 0.05
    surrogate_clip_range: float = 1e-3
    surrogate_slow_decay_min: float = 0.001
    surrogate_slow_decay_max: float = 0.3
    cfg_reward_scale: float = 3.5
    use_time_weighting: bool = True

    def __post_init__(self) -> None:
        """Validate reward and generator objective controls."""
        super().__post_init__()
        if self.advantage_aggregation not in ("sum", "gdpo"):
            raise ValueError(
                "expected train.advantage_aggregation in ('sum', 'gdpo'), "
                f"received {self.advantage_aggregation!r}"
            )
        self.tdm_weight = _finite_float(self.tdm_weight, "train.tdm_weight", allow_zero=False)
        self.surrogate_preference_beta = _finite_float(
            self.surrogate_preference_beta,
            "train.surrogate_preference_beta",
            allow_zero=False,
        )
        self.advantage_clip_range = _finite_float(
            self.advantage_clip_range,
            "train.advantage_clip_range",
            allow_zero=False,
        )
        if not 0.0 < self.tdm_weight < 1.0:
            raise ValueError(
                "expected train.tdm_weight in (0, 1): it mixes the guidance reward against "
                "the surrogate reward as tdm_weight * cfg_reward + (1 - tdm_weight) * "
                f"surrogate_reward, received {self.tdm_weight!r}"
            )
        self.surrogate_reference_beta = _finite_float(
            self.surrogate_reference_beta,
            "train.surrogate_reference_beta",
            allow_zero=True,
        )
        self.surrogate_reference_threshold = _finite_float(
            self.surrogate_reference_threshold,
            "train.surrogate_reference_threshold",
            allow_zero=False,
        )
        self.cfg_reward_scale = _finite_float(
            self.cfg_reward_scale,
            "train.cfg_reward_scale",
            allow_zero=True,
        )
        self.surrogate_clip_range = _finite_float(
            self.surrogate_clip_range,
            "train.surrogate_clip_range",
            allow_zero=True,
        )
        for name in ("surrogate_slow_decay_min", "surrogate_slow_decay_max"):
            decay = _finite_float(getattr(self, name), f"train.{name}", allow_zero=True)
            if decay > 1.0:
                raise ValueError(f"expected train.{name} in [0, 1], received {decay!r}")
            setattr(self, name, decay)
        if self.surrogate_slow_decay_min > self.surrogate_slow_decay_max:
            raise ValueError(
                "expected train.surrogate_slow_decay_min <= train.surrogate_slow_decay_max, "
                f"received {self.surrogate_slow_decay_min} and {self.surrogate_slow_decay_max}"
            )
        if not isinstance(self.use_time_weighting, bool):
            raise TypeError(
                "expected train.use_time_weighting to be a bool, received "
                f"{type(self.use_time_weighting).__name__}: {self.use_time_weighting!r}"
            )

    def role_update_plan(self) -> RoleUpdatePlan:
        """Return fake TTUR phases, then one surrogate and one generator phase."""
        # Constraint #22(c): trainers.role_optimization imports training args.
        from ...trainers.role_optimization import RolePhase, RoleUpdatePlan

        return RoleUpdatePlan(
            phases=(
                RolePhase("fake", repeats=self.ttur_fake_updates),
                RolePhase("surrogate"),
                RolePhase("generator"),
            )
        )

    @property
    def required_trainable_roles(self) -> Tuple[RoleName, ...]:
        """Return trainable roles in canonical materialization order."""
        return ("generator", "fake", "surrogate")


TDM_R1_DEFAULT_OPTIMIZERS: Tuple[AdamWOptimizerArguments, ...] = (
    AdamWOptimizerArguments(name="generator", learning_rate=7.5e-5, betas=(0.0, 0.999)),
    AdamWOptimizerArguments(name="fake", learning_rate=3e-4, betas=(0.0, 0.999)),
    AdamWOptimizerArguments(name="surrogate", learning_rate=3e-4, betas=(0.9, 0.999)),
)
"""Per-role defaults used when a config file declares no ``optimizers`` list.

The generator and fake score run with a zeroed first moment, following TDM-R1;
the surrogate keeps the usual AdamW momentum.
"""
