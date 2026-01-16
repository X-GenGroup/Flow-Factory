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
    num_train_steps: int = field(
        default=1,
        metadata={"help": "Number of train steps to sample per rollout."},
    )
    train_steps: Optional[List[int]] = field(
        default=None,
        metadata={"help": (
            "Training step indices for optimization. "
            "`num_train_steps` will be randomly sampled from this list. "
            "If None, uses all the timesteps."
        )},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for selecting train steps."},
    )

    def __post_init__(self):
        pass

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()