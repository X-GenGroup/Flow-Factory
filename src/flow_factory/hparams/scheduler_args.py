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

    def get_num_replay_steps(self, num_steps: int, *, require_stochastic: bool = False) -> int:
        """Count loss terms without drawing RNG; validate explicit likelihood selections.

        A fixed replay subset must be stochastic on every rollout. With random
        SDE selection, use ``auto`` or retain the complete SDE pool; silently
        intersecting two selections would change the accumulation geometry.
        """
        pool = self.sde_steps if self.sde_steps is not None else list(range(num_steps - 1))
        count = len(pool) if self.num_sde_steps is None else self.num_sde_steps
        if require_stochastic and self.replay_steps == "auto":
            # Preserve legacy config resolution. Coupled trainer initialization
            # already owns dynamics admission, including ODE-only test configs.
            return count
        default = list(range(num_steps)) if self.dynamics_type == "ODE" else pool[:count]
        selected = resolve_replay_steps(self.replay_steps, num_steps, default)
        if not selected:
            raise ValueError(
                "This objective requires at least one scheduler.replay_steps transition"
            )
        if require_stochastic:
            if self.dynamics_type == "ODE" or self.noise_level <= 0:
                raise ValueError(
                    "Transition likelihood replay requires stochastic dynamics and positive noise_level"
                )
            if self.replay_steps != "auto" and (
                count != len(pool) or not set(selected).issubset(pool)
            ):
                raise ValueError(
                    "Explicit scheduler.replay_steps for likelihood replay must be stochastic "
                    "on every rollout: use replay_steps='auto' with random SDE selection, or "
                    "set num_sde_steps=null and choose a subset of sde_steps"
                )
        return len(selected)

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
