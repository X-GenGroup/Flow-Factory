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

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

# src/flow_factory/hparams/scheduler_args.py
import yaml

from ..contracts.replay import ReplaySteps, resolve_replay_steps, validate_replay_steps
from .abc import ArgABC


@dataclass
class SchedulerArguments(ArgABC):
    r"""Arguments pertaining to scheduler configuration."""

    dynamics_type: Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"] = field(
        default="Flow-SDE",
        metadata={"help": "Type of SDE dynamics to use."},
    )
    noise_level: float = field(
        default=0.7,
        metadata={"help": "Noise level for SDE sampling."},
    )
    num_sde_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of SDE steps to sample per rollout. "
                "YAML `null` means use every index in `sde_steps` (after `sde_steps` is resolved)."
            )
        },
    )
    sde_steps: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "Rollout transition indices (0-based) eligible for SDE noise; "
                "`num_sde_steps` indices are drawn from this list each rollout. "
                "YAML `null` means indices `0 .. num_inference_steps-2` (all denoising steps except the last), "
                "matching the default SDE scheduler behavior."
            )
        },
    )
    replay_steps: ReplaySteps = field(
        default="auto",
        metadata={
            "help": "Actual rollout transitions to retain and replay: 'auto' uses the selected "
            "SDE steps or every ODE step; 'all' uses every step; a list selects explicit "
            "0-based transitions; null or [] retains no transitions. Endpoint-only objectives "
            "keep their required final latent and sample training times through train.* instead."
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for selecting train steps."},
    )

    def __post_init__(self):
        validate_replay_steps(self.replay_steps)
        available_dynamics = ["Flow-SDE", "Dance-SDE", "CPS", "ODE"]
        assert (
            self.dynamics_type in available_dynamics
        ), f"Invalid dynamics type {self.dynamics_type}. Must be one of {available_dynamics}."

        # ODE has no stochastic steps — zero out SDE-related fields
        if self.dynamics_type == "ODE":
            self.sde_steps = []
            self.num_sde_steps = 0
            self.noise_level = 0.0

    def get_num_replay_steps(self, num_steps: int) -> int:
        """Count configured replay terms without selecting positions or drawing RNG."""
        if self.replay_steps == "auto":
            if self.dynamics_type == "ODE":
                return num_steps
            if self.num_sde_steps is not None:
                return self.num_sde_steps
            return len(self.sde_steps) if self.sde_steps is not None else num_steps - 1
        if self.replay_steps == "all":
            return num_steps
        return len(self.replay_steps or [])

    def validate_replay_selection(
        self, num_steps: int, *, require_stochastic: bool = False
    ) -> None:
        """Validate the payload required by a transition-replay objective.

        A fixed replay subset must be stochastic on every rollout. With random
        SDE selection, use ``auto``; explicit positions require every candidate
        to receive noise. Validation never draws a stochastic selection.
        """
        validate_replay_steps(self.replay_steps, num_steps)
        if self.get_num_replay_steps(num_steps) < 1:
            raise ValueError(
                "This objective requires at least one scheduler.replay_steps transition; "
                f"received replay_steps={self.replay_steps!r}"
            )
        if not require_stochastic:
            return
        if self.dynamics_type == "ODE" or self.noise_level <= 0:
            raise ValueError(
                "Transition likelihood replay requires stochastic dynamics and positive "
                f"noise_level; received dynamics_type={self.dynamics_type!r}, "
                f"noise_level={self.noise_level!r}"
            )
        pool = self.sde_steps if self.sde_steps is not None else list(range(num_steps - 1))
        count = len(pool) if self.num_sde_steps is None else self.num_sde_steps
        if self.replay_steps == "auto":
            return
        selected = resolve_replay_steps(self.replay_steps, num_steps, pool)
        if count != len(pool) or not set(selected).issubset(pool):
            raise ValueError(
                "Explicit scheduler.replay_steps for likelihood replay must be stochastic "
                "on every rollout: use replay_steps='auto' with random SDE selection, or "
                "set num_sde_steps=null and choose a subset of sde_steps; "
                f"received replay_steps={self.replay_steps!r}, sde_steps={pool!r}, "
                f"num_sde_steps={count!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        # Automatic selection preserves the existing execution identity/default.
        if self.replay_steps == "auto":
            result.pop("replay_steps")
        return result

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
