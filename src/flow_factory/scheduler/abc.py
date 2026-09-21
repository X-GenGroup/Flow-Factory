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
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from diffusers.utils.outputs import BaseOutput


def stable_mean_except_batch(values: torch.Tensor) -> torch.Tensor:
    """Reduce transition statistics without FP32 tree-order drift.

    Rollout and replay can present bit-identical tensors at different memory
    addresses. CUDA's parallel FP32 reduction may then differ by one ULP, which
    makes an exactly on-policy PPO ratio differ from one. Accumulating in FP64
    makes the final cast stable while preserving the autograd path.
    """
    dimensions = tuple(range(1, values.ndim))
    return values.mean(dim=dimensions, dtype=torch.float64).to(values.dtype)


@dataclass
class SDESchedulerOutput(BaseOutput):
    """Single SDE step output with latents, statistics, and log probability."""

    next_latents: Optional[torch.FloatTensor] = None
    next_latents_mean: Optional[torch.FloatTensor] = None
    std_dev_t: Optional[torch.FloatTensor] = None
    dt: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None
    velocity: Optional[torch.FloatTensor] = None

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
        sde_steps: Indices of steps eligible for SDE noise.
        seed: Random seed for stochastic step selection.
        dynamics_type: SDE variant ("Flow-SDE", "Dance-SDE", "CPS", "ODE").
    """

    # From diffusers schedulers
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    config: Any

    # SDE-specific
    noise_level: float
    _sde_steps: Optional[torch.Tensor]
    _num_sde_steps: Optional[int]
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
    def select_random_step_indices(
        self,
        eligible_steps: torch.Tensor,
        num_steps: int,
        *,
        seed_offset: int = 0,
    ) -> torch.Tensor:
        """Select scheduler steps reproducibly under ``set_seed``.

        This is shared by stochastic SDE-step selection and ODE boundary replay.
        Reconstructing a generator from ``seed + seed_offset`` makes the result
        independent of Python's global RNG, identical on every rank, and reproducible
        after resume.

        Args:
            eligible_steps: Non-empty rank-1 integer step indices.
            num_steps: Number of distinct indices to draw.
            seed_offset: Deterministic draw counter within the current scheduler seed.

        Returns:
            Selected indices in seeded random order.
        """
        if (
            not isinstance(eligible_steps, torch.Tensor)
            or eligible_steps.ndim != 1
            or eligible_steps.numel() == 0
            or eligible_steps.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise ValueError(
                "expected non-empty rank-1 integer eligible_steps, received "
                f"{type(eligible_steps).__name__} shape="
                f"{getattr(eligible_steps, 'shape', None)} dtype="
                f"{getattr(eligible_steps, 'dtype', None)}"
            )
        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or not 1 <= num_steps <= eligible_steps.numel()
        ):
            raise ValueError(
                f"expected num_steps in [1, {eligible_steps.numel()}], received {num_steps!r}"
            )
        if not isinstance(seed_offset, int) or isinstance(seed_offset, bool) or seed_offset < 0:
            raise ValueError(f"expected non-negative int seed_offset, received {seed_offset!r}")
        generator = torch.Generator().manual_seed(int(self.seed) + seed_offset)
        selected = torch.randperm(eligible_steps.numel(), generator=generator)[:num_steps]
        return eligible_steps[selected]

    def sample_ode_step_index(self, draw_index: int = 0) -> int:
        """Draw one ODE transition index from the current inference schedule.

        Args:
            draw_index: Deterministic draw counter under the current scheduler seed.

        Returns:
            A zero-based transition index.
        """
        num_steps = len(self.timesteps)
        if num_steps < 1:
            raise ValueError(
                "expected scheduler.timesteps to contain at least one ODE transition "
                f"before sampling a replay boundary, received {num_steps}"
            )
        eligible = torch.arange(num_steps, dtype=torch.int64)
        return int(self.select_random_step_indices(eligible, 1, seed_offset=draw_index)[0].item())

    @property
    @abstractmethod
    def sde_steps(self) -> torch.Tensor:
        """Step indices eligible for SDE noise injection."""
        ...

    @property
    @abstractmethod
    def num_sde_steps(self) -> int:
        """Number of training steps with SDE noise."""
        ...

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

    # ==================== Distillation / KL ====================
    def get_kl_divergence_denominator(
        self,
        std_dev_t: Optional[torch.Tensor],
        dt: Optional[torch.Tensor],
        eps: float = 1e-8,
    ) -> Union[float, torch.Tensor]:
        """Transition variance ``sigma_bar^2`` for the pathwise per-step KL.

        Distillation trainers (e.g. DiffusionOPD) match the student mean
        ``mu_S`` to a teacher mean ``mu_T`` along the rollout via
        ``loss = 0.5 * ||mu_S - mu_T||^2 / denom`` where ``denom`` is the
        variance of the Gaussian transition ``N(mu, sigma_bar^2)`` that
        this scheduler's :meth:`step` defines for the active
        ``dynamics_type``:

        =============  ========================  ================================
        dynamics_type  denom                     transition sampled in ``step``
        =============  ========================  ================================
        ODE            ``1.0``                   deterministic (mean matching)
        Flow-SDE       ``std_dev_t**2 * (-dt)``  ``mu + std_dev_t*sqrt(-dt)*eps``
        Dance-SDE      ``std_dev_t**2 * (-dt)``  ``mu + std_dev_t*sqrt(-dt)*eps``
        CPS            ``std_dev_t**2``          ``mu + std_dev_t*eps``
        =============  ========================  ================================

        For ``ODE`` it returns the scalar ``1.0`` without touching
        ``std_dev_t``/``dt`` (which are zero/``None`` under ODE), so the
        same loss expression is dynamics-agnostic. Non-ODE results are
        clamped to ``>= eps`` to avoid div-by-zero at near-deterministic
        steps.

        Args:
            std_dev_t: Per-sample std from the student ``step`` output.
            dt: Per-sample ``sigma_next - sigma`` (negative) from ``step``.
            eps: Lower clamp for the (non-ODE) denominator.

        Returns:
            ``1.0`` for ODE, otherwise a tensor broadcastable against the
            per-element squared error.
        """
        if self.dynamics_type == "ODE":
            return 1.0
        if not isinstance(std_dev_t, torch.Tensor):
            raise ValueError(
                f"get_kl_divergence_denominator requires a `std_dev_t` tensor for "
                f"dynamics_type={self.dynamics_type!r}, got {type(std_dev_t).__name__}. "
                "Ensure the student step() returns 'std_dev_t' (and 'dt')."
            )
        if self.dynamics_type == "CPS":
            return (std_dev_t.float() ** 2).clamp_min(eps)
        # Flow-SDE / Dance-SDE: Euler-Maruyama transition variance std_dev_t^2 * (-dt).
        if not isinstance(dt, torch.Tensor):
            raise ValueError(
                f"get_kl_divergence_denominator requires a `dt` tensor for "
                f"dynamics_type={self.dynamics_type!r}, got {type(dt).__name__}."
            )
        return (std_dev_t.float() ** 2 * (-dt.float())).clamp_min(eps)
