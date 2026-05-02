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

# src/flow_factory/hparams/scheduler_args.py
import yaml
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, List

from .abc import ArgABC


@dataclass
class SchedulerArguments(ArgABC):
    r"""Arguments pertaining to scheduler configuration."""

    dynamics_type: Literal["Flow-SDE", 'Dance-SDE', 'CPS', 'ODE'] = field(
        default="Flow-SDE",
        metadata={"help": "Type of SDE dynamics to use."},
    )
    noise_level: float = field(
        default=0.7,
        metadata={"help": "Noise level for SDE sampling."},
    )
    num_sde_steps: Optional[int] = field(
        default=1,
        metadata={"help": (
            "Number of SDE steps to sample per rollout. "
            "YAML `null` means use every index in `sde_steps` (after `sde_steps` is resolved)."
        )},
    )
    sde_steps: Optional[List[int]] = field(
        default=None,
        metadata={"help": (
            "Training trajectory indices (0-based) eligible for SDE noise; "
            "`num_sde_steps` indices are drawn from this list each rollout. "
            "YAML `null` means indices `0 .. num_inference_steps-2` (all denoising steps except the last), "
            "matching the default SDE scheduler behavior."
        )},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for selecting train steps."},
    )

    def __post_init__(self):
        available_dynamics = ["Flow-SDE", 'Dance-SDE', 'CPS', 'ODE']
        assert self.dynamics_type in available_dynamics, f"Invalid dynamics type {self.dynamics_type}. Must be one of {available_dynamics}."

        # ODE has no stochastic steps — zero out SDE-related fields
        if self.dynamics_type == 'ODE':
            self.sde_steps = []
            self.num_sde_steps = 0
            self.noise_level = 0.0

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()