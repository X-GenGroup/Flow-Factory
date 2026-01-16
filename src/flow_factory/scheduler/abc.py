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

# src/flow_factory/scheduler/abc.py
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Literal, Any, Dict
from dataclasses import dataclass, fields

import torch
from diffusers.utils.outputs import BaseOutput


@dataclass
class SDESchedulerOutput(BaseOutput):
    """Single SDE step output with latents, statistics, and log probability."""
    next_latents: Optional[torch.FloatTensor] = None
    next_latents_mean: Optional[torch.FloatTensor] = None
    std_dev_t: Optional[torch.FloatTensor] = None
    dt: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None
    noise_pred: Optional[torch.FloatTensor] = None

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDESchedulerOutput":
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})


class SDESchedulerMixin(ABC):
    """
    Abstract mixin for SDE-capable schedulers in RL fine-tuning.
    
    Extends `diffusers` schedulers with stochastic sampling, noise injection control,
    and log probability computation for policy gradient methods.
    
    Usage:
        class MySDEScheduler(DiffusersScheduler, SDESchedulerMixin):
            ...
    
    Attributes:
        sigmas: Noise schedule sigma values (from `diffusers`).
        timesteps: Discrete timesteps (from `diffusers`).
        noise_level: Noise injection scale for SDE sampling.
        train_steps: Indices of steps eligible for SDE noise.
        seed: Random seed for stochastic step selection.
        dynamics_type: SDE variant ("Flow-SDE", "Dance-SDE", "CPS", "ODE").
    """
    
    # From diffusers schedulers
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    config: Any
    
    # SDE-specific
    noise_level: float
    train_steps: torch.Tensor
    num_train_steps: int
    seed: int
    dynamics_type: Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"]
    _is_eval: bool

    # ==================== Mode Management ====================
    @property
    @abstractmethod
    def is_eval(self) -> bool:
        """Whether scheduler is in deterministic eval mode."""
        ...

    @abstractmethod
    def eval(self) -> None:
        """Switch to deterministic ODE sampling (no noise injection)."""
        ...

    @abstractmethod
    def train(self, mode: bool = True) -> None:
        """Switch to stochastic SDE sampling."""
        ...

    @abstractmethod
    def rollout(self, mode: bool = True) -> None:
        """Switch to rollout mode (alias for train)."""
        ...

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set random seed for stochastic step selection."""
        ...

    # ==================== Step Selection ====================
    @property
    @abstractmethod
    def current_sde_steps(self) -> torch.Tensor:
        """Step indices where SDE noise is applied under current seed."""
        ...

    @property
    @abstractmethod
    def train_timesteps(self) -> torch.Tensor:
        """Step indices for training timesteps."""
        ...

    @abstractmethod
    def get_train_timesteps(self) -> torch.Tensor:
        """Timestep values [0, 1000] for training steps."""
        ...

    @abstractmethod
    def get_train_sigmas(self) -> torch.Tensor:
        """Sigma values for training steps."""
        ...

    # ==================== Noise Level ====================
    @abstractmethod
    def get_noise_levels(self) -> torch.Tensor:
        """Noise level for each timestep (0 if not in `current_sde_steps`)."""
        ...

    @abstractmethod
    def get_noise_level_for_timestep(
        self, timestep: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Get noise level for specific timestep(s)."""
        ...

    @abstractmethod
    def get_noise_level_for_sigma(self, sigma: float) -> float:
        """Get noise level for specific sigma value."""
        ...