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

"""MiniMax H3 scheduler with Flow-Factory SDE dynamics."""

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor

from ..utils.base import to_broadcast_tensor
from ..utils.noise_schedule import flow_match_sigma
from .abc import SDESchedulerMixin, SDESchedulerOutput


@dataclass
class MiniMaxH3SDESchedulerOutput(SDESchedulerOutput):
    """Store one MiniMax H3 scheduler transition.

    Attributes:
        next_latents: Sampled next state.
        next_latents_mean: Transition mean.
        std_dev_t: Scheduler diffusion coefficient.
        dt: Sigma step size.
        log_prob: Per-sample mean transition log probability.
        velocity: Original H3 data-ward model velocity.
    """


class MiniMaxH3SDEScheduler(SchedulerMixin, ConfigMixin, SDESchedulerMixin):
    """Apply MiniMax H3 data-ward velocity with Flow-Factory SDE dynamics."""

    _compatibles: List[str] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        shift: float = 12.0,
        noise_level: float = 0.7,
        sde_steps: Optional[Union[int, List[int], torch.Tensor]] = None,
        num_sde_steps: Optional[int] = None,
        seed: int = 42,
        dynamics_type: Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"] = "Flow-SDE",
    ) -> None:
        """Initialize scheduler configuration and stochastic lifecycle state.

        Args:
            shift: Positive exponential sigma shift.
            noise_level: Non-negative stochastic sampling scale.
            sde_steps: Eligible stochastic transition indices.
            num_sde_steps: Number of eligible indices selected per rollout.
            seed: Seed for stochastic transition selection.
            dynamics_type: Configured ODE or SDE dynamics.
        """
        self._validate_positive_number(shift, "shift")
        if not isinstance(noise_level, (int, float)) or isinstance(noise_level, bool):
            raise TypeError(
                "expected numeric noise_level for MiniMaxH3SDEScheduler, "
                f"received {type(noise_level).__name__}: {noise_level!r}"
            )
        if not math.isfinite(float(noise_level)) or noise_level < 0:
            raise ValueError(f"expected finite non-negative noise_level, received {noise_level}")
        if dynamics_type not in ("Flow-SDE", "Dance-SDE", "CPS", "ODE"):
            raise ValueError(
                "expected dynamics_type in ('Flow-SDE', 'Dance-SDE', 'CPS', 'ODE'), "
                f"received {dynamics_type!r}"
            )
        self._shift = float(shift)
        self.noise_level = float(noise_level)
        self._sde_steps = (
            None
            if sde_steps is None
            else torch.as_tensor(sde_steps, dtype=torch.int64).flatten().cpu()
        )
        self._num_sde_steps = num_sde_steps
        self.seed = seed
        self.dynamics_type = dynamics_type
        self._is_eval = False
        self.num_inference_steps: Optional[int] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.model_timesteps: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

    @property
    def shift(self) -> float:
        """Return the active exponential sigma shift."""
        return self._shift

    @property
    def step_index(self) -> Optional[int]:
        """Return the next implicit schedule index."""
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        """Return the configured initial schedule index."""
        return self._begin_index

    @property
    def is_eval(self) -> bool:
        """Return whether deterministic evaluation mode is active."""
        return self._is_eval

    def eval(self) -> None:
        """Switch to deterministic evaluation mode."""
        self._is_eval = True

    def train(self, mode: bool = True) -> None:
        """Switch stochastic training mode on or off."""
        if not isinstance(mode, bool):
            raise TypeError(
                f"expected bool mode for MiniMaxH3SDEScheduler.train, received {type(mode).__name__}"
            )
        self._is_eval = not mode

    def rollout(self, mode: bool = True) -> None:
        """Switch stochastic rollout mode on or off."""
        self.train(mode)

    def set_seed(self, seed: int) -> None:
        """Set the seed used to select stochastic transition indices.

        Args:
            seed: Integer selection seed.

        Returns:
            None.
        """
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(
                f"expected int seed for MiniMaxH3SDEScheduler, received {type(seed).__name__}: {seed!r}"
            )
        self.seed = seed

    def set_shift(self, shift: float) -> None:
        """Set the exponential shift used by the next schedule.

        Args:
            shift: Positive finite exponential shift.

        Returns:
            None.
        """
        self._validate_positive_number(shift, "shift")
        self._shift = float(shift)

    def set_begin_index(self, begin_index: int = 0) -> None:
        """Set the first implicit schedule index.

        Args:
            begin_index: First scalar rollout transition index.

        Returns:
            None.
        """
        if not isinstance(begin_index, int) or isinstance(begin_index, bool):
            raise TypeError(
                "expected int begin_index for MiniMaxH3SDEScheduler, "
                f"received {type(begin_index).__name__}: {begin_index!r}"
            )
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[List[float], torch.Tensor]] = None,
    ) -> None:
        """Build an N-transition schedule from N+1 upstream sigma grid points.

        Args:
            num_inference_steps: Number of denoising transitions.
            device: Device for exposed schedule tensors.
            sigmas: Optional explicit shifted schedule including terminal zero.

        Returns:
            None.
        """
        if sigmas is None:
            if (
                not isinstance(num_inference_steps, int)
                or isinstance(num_inference_steps, bool)
                or num_inference_steps <= 0
            ):
                raise ValueError(
                    "expected num_inference_steps as a positive int when sigmas are omitted, "
                    f"received {num_inference_steps!r}"
                )
            base = torch.linspace(
                1.0, 0.0, num_inference_steps + 1, dtype=torch.float32, device="cpu"
            )
            schedule = self._shift * base / (1 + (self._shift - 1) * base)
            schedule = torch.unique_consecutive(schedule)
            transitions = schedule.numel() - 1
            if transitions != num_inference_steps:
                raise ValueError(
                    f"expected {num_inference_steps} transitions after MiniMax H3 sigma "
                    f"deduplication, received {transitions} from {schedule.numel()} grid points"
                )
        else:
            schedule = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            if num_inference_steps is not None and num_inference_steps != schedule.numel() - 1:
                raise ValueError(
                    "expected num_inference_steps to match explicit sigmas length minus one, "
                    f"received num_inference_steps={num_inference_steps} and "
                    f"len(sigmas)={schedule.numel()}"
                )
            self._validate_sigma_schedule(schedule)
            num_inference_steps = schedule.numel() - 1
        self._validate_sigma_schedule(schedule)
        self.sigmas = schedule.to(device=device)
        self.timesteps = self.sigmas[:-1] * 1000
        self.model_timesteps = 1 - self.sigmas[:-1]
        self.num_inference_steps = int(num_inference_steps)
        self._step_index = None
        self._begin_index = None
        self._validate_sde_configuration()

    @property
    def sde_steps(self) -> torch.Tensor:
        """Return transition indices eligible for stochastic sampling."""
        self._require_schedule()
        if self._sde_steps is None:
            return torch.arange(max(0, len(self.timesteps) - 1), dtype=torch.int64)
        return self._sde_steps

    @property
    def num_sde_steps(self) -> int:
        """Return the number of selected stochastic transitions."""
        return len(self.sde_steps) if self._num_sde_steps is None else self._num_sde_steps

    @property
    def current_sde_steps(self) -> torch.Tensor:
        """Return stochastic transition indices selected by the current seed."""
        if self.num_sde_steps >= len(self.sde_steps):
            return self.sde_steps
        generator = torch.Generator().manual_seed(self.seed)
        selected = torch.randperm(len(self.sde_steps), generator=generator)[: self.num_sde_steps]
        return self.sde_steps[selected]

    @property
    def train_timesteps(self) -> torch.Tensor:
        """Return schedule indices used during policy optimization."""
        return self.current_sde_steps

    def get_train_timesteps(self) -> torch.Tensor:
        """Return framework scheduler times selected for training."""
        self._require_schedule()
        return self.timesteps[self.train_timesteps]

    def get_train_sigmas(self) -> torch.Tensor:
        """Return sigma values selected for training."""
        self._require_schedule()
        return self.sigmas[self.train_timesteps]

    def get_noise_levels(self) -> torch.Tensor:
        """Return per-transition noise levels, non-zero only at ``current_sde_steps``."""
        self._require_schedule()
        levels = torch.zeros_like(self.timesteps, dtype=torch.float32)
        levels[self.current_sde_steps.to(levels.device)] = self.noise_level
        return levels

    def get_noise_level_for_timestep(
        self, timestep: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Return stochastic noise levels for exact stored framework times.

        Args:
            timestep: Scalar or batched stored framework coordinates.

        Returns:
            Scalar or batched effective noise levels.
        """
        if isinstance(timestep, torch.Tensor) and timestep.ndim == 1:
            indices = torch.tensor(
                [self.index_for_timestep(value) for value in timestep],
                device=timestep.device,
            )
            active = torch.isin(indices.cpu(), self.current_sde_steps).to(timestep.device)
            return torch.where(
                active,
                torch.as_tensor(self.noise_level, dtype=timestep.dtype, device=timestep.device),
                torch.zeros((), dtype=timestep.dtype, device=timestep.device),
            )
        index = self.index_for_timestep(timestep)
        return self.noise_level if index in self.current_sde_steps.tolist() else 0.0

    def get_noise_level_for_sigma(
        self, sigma: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Return stochastic noise levels for exact stored sigma values.

        Args:
            sigma: Scalar or batched stored sigma values.

        Returns:
            Scalar or batched effective noise levels.
        """
        sigma_tensor = torch.as_tensor(sigma)
        scalar = sigma_tensor.ndim == 0
        flattened = sigma_tensor.reshape(-1)
        values = []
        for value in flattened:
            index = self._index_for_sigma(value)
            values.append(self.noise_level if index in self.current_sde_steps.tolist() else 0.0)
        result = torch.tensor(values, dtype=sigma_tensor.dtype, device=sigma_tensor.device)
        return result.item() if scalar else result.reshape(sigma_tensor.shape)

    def index_for_timestep(self, timestep: Union[float, torch.Tensor]) -> int:
        """Return the index of an exact stored framework timestep.

        Args:
            timestep: Scalar value from ``scheduler.timesteps``.

        Returns:
            Exact schedule index.
        """
        self._require_schedule()
        value = torch.as_tensor(timestep, device=self.timesteps.device)
        matches = (self.timesteps == value).nonzero()
        if matches.numel() == 0:
            raise ValueError(
                f"expected timestep from scheduler.timesteps, received {value.item()!r}"
            )
        return int(matches[0].item())

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Apply H3 clean-time condition augmentation.

        Args:
            sample: Clean condition sample.
            timestep: H3 clean-time coordinate in ``[0, 1]``.
            noise: Noise tensor matching ``sample``.

        Returns:
            Forward-noised condition sample.
        """
        if sample.shape != noise.shape:
            raise ValueError(
                f"expected sample and noise with equal shape, received {tuple(sample.shape)} "
                f"and {tuple(noise.shape)}"
            )
        clean_time = torch.as_tensor(timestep, dtype=sample.dtype, device=sample.device)
        if bool((clean_time < 0).any()) or bool((clean_time > 1).any()):
            raise ValueError(
                f"expected H3 clean timestep in [0, 1], received {clean_time.tolist()}"
            )
        while clean_time.ndim < sample.ndim:
            clean_time = clean_time.unsqueeze(-1)
        return clean_time * sample + (1 - clean_time) * noise

    def step(
        self,
        velocity: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        latents: torch.Tensor,
        next_latents: Optional[torch.Tensor] = None,
        timestep_next: Optional[Union[float, torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise_level: Optional[Union[int, float, torch.Tensor]] = None,
        compute_log_prob: bool = True,
        return_dict: bool = True,
        return_kwargs: Optional[Sequence[str]] = None,
        dynamics_type: Optional[Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"]] = None,
        sigma_max: Optional[float] = None,
        sigma: Optional[Union[float, torch.Tensor]] = None,
        sigma_next: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[MiniMaxH3SDESchedulerOutput, Tuple[Optional[torch.Tensor], ...]]:
        """Take one H3 data-ward ODE or SDE transition.

        Args:
            velocity: H3 data-ward model velocity.
            timestep: Scalar or batched framework scheduler coordinate.
            latents: Current latent state.
            next_latents: Optional stored next state for replay.
            timestep_next: Optional explicit next framework coordinate.
            generator: SDE-step noise only. Rollout leaves this unset (constraint #12b);
                ``None`` uses the process RNG. Do not pass the ``inference`` generator.
            noise_level: Optional non-negative stochastic scale override.
            compute_log_prob: Whether to compute transition log probability.
            return_dict: Whether to return a scheduler output instead of a tuple.
            return_kwargs: Requested scheduler output fields.
            dynamics_type: Compatibility argument; must match configured dynamics.
            sigma_max: Optional Flow-SDE singularity guard.
            sigma: Optional explicit current sigma.
            sigma_next: Optional explicit next sigma.

        Returns:
            Scheduler output or its legacy tuple representation.
        """
        self._validate_step_inputs(velocity, latents, next_latents)
        requested_fields = (
            (
                "next_latents",
                "next_latents_mean",
                "std_dev_t",
                "dt",
                "log_prob",
                "velocity",
            )
            if return_kwargs is None
            else tuple(return_kwargs)
        )
        supported_fields = {
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
            "velocity",
        }
        unknown_fields = tuple(field for field in requested_fields if field not in supported_fields)
        if unknown_fields:
            raise ValueError(
                f"unknown return fields {unknown_fields}; expected a subset of "
                f"{tuple(sorted(supported_fields))}"
            )
        if dynamics_type is not None and dynamics_type != self.dynamics_type:
            raise ValueError(
                f"configured dynamics_type={self.dynamics_type!r} conflicts with "
                f"per-call dynamics_type={dynamics_type!r}"
            )
        implicit_schedule_step = sigma is None and timestep_next is None
        resolved_step_index: Optional[int] = None
        if implicit_schedule_step and isinstance(timestep, torch.Tensor) and timestep.ndim == 1:
            indices = torch.tensor(
                [self.index_for_timestep(value) for value in timestep],
                dtype=torch.long,
                device=self.sigmas.device,
            )
            sigma_value = self.sigmas[indices]
            sigma_next_value = self.sigmas[indices + 1]
        elif implicit_schedule_step:
            if self._step_index is None:
                resolved_step_index = (
                    self.index_for_timestep(timestep)
                    if self._begin_index is None
                    else self._begin_index
                )
            else:
                resolved_step_index = self._step_index
            if resolved_step_index < 0 or resolved_step_index >= len(self.timesteps):
                raise ValueError(
                    f"implicit step_index {resolved_step_index} is outside "
                    f"[0, {len(self.timesteps) - 1}]"
                )
            sigma_value = self.sigmas[resolved_step_index]
            sigma_next_value = self.sigmas[resolved_step_index + 1]
        else:
            sigma_value, sigma_next_value = self._resolve_sigmas(
                timestep, timestep_next, sigma, sigma_next
            )
        for field, coordinate in (
            ("timestep", timestep if implicit_schedule_step else torch.as_tensor(0.0)),
            ("sigma", sigma_value),
            ("sigma_next", sigma_next_value),
        ):
            coordinate_tensor = torch.as_tensor(coordinate)
            if not bool(torch.isfinite(coordinate_tensor).all()):
                raise ValueError(
                    f"expected finite resolved {field}, received {coordinate_tensor.tolist()}"
                )
            if coordinate_tensor.ndim == 1 and coordinate_tensor.numel() != latents.shape[0]:
                raise ValueError(
                    f"{field} cardinality must match latents batch size {latents.shape[0]}, "
                    f"received {coordinate_tensor.numel()}"
                )
        storage_dtype = latents.dtype
        velocity_float = velocity.float()
        latents_float = latents.float()
        replay_latents = None if next_latents is None else next_latents.float()
        sigma_broadcast = to_broadcast_tensor(sigma_value.float(), latents_float)
        sigma_next_broadcast = to_broadcast_tensor(sigma_next_value.float(), latents_float)
        dt = sigma_next_broadcast - sigma_broadcast
        selected_dynamics = self.dynamics_type
        if selected_dynamics not in ("Flow-SDE", "Dance-SDE", "CPS", "ODE"):
            raise ValueError(f"expected supported dynamics_type, received {selected_dynamics!r}")
        selected_noise = self._resolve_noise_level(sigma_value, noise_level, selected_dynamics)
        noise_broadcast = to_broadcast_tensor(selected_noise, latents_float)

        model_t = 1 - sigma_value.float()
        sigma_from_model_t = 1 - model_t
        sigma_from_model_t = to_broadcast_tensor(sigma_from_model_t.to(storage_dtype), latents)
        denoised = (latents + sigma_from_model_t * velocity).float()
        standard_velocity = -velocity_float

        next_mean, std_dev_t = self._transition_mean(
            selected_dynamics,
            latents_float,
            standard_velocity,
            denoised,
            sigma_broadcast,
            sigma_next_broadcast,
            dt,
            noise_broadcast,
            sigma_max,
        )
        deterministic_samples = None
        if compute_log_prob and selected_dynamics != "ODE":
            transition_scale = (
                std_dev_t if selected_dynamics == "CPS" else std_dev_t * torch.sqrt(-dt)
            )
            deterministic_samples = (
                (transition_scale == 0).reshape(transition_scale.shape[0], -1).all(dim=1)
            )
        sampled = replay_latents
        if sampled is None:
            sampled = self._sample_transition(
                selected_dynamics,
                next_mean,
                std_dev_t,
                dt,
                velocity_float,
                generator,
            )
            sampled = sampled.to(storage_dtype).float()
        if not compute_log_prob:
            log_prob = None
        else:
            log_prob = self._transition_log_prob(
                selected_dynamics, sampled, next_mean, std_dev_t, dt
            )
            if deterministic_samples is not None:
                log_prob = torch.where(deterministic_samples, torch.zeros_like(log_prob), log_prob)
        if selected_dynamics == "ODE":
            sampled = sampled.to(storage_dtype)
            next_mean = next_mean.to(storage_dtype)

        output = MiniMaxH3SDESchedulerOutput(
            **{
                field: {
                    "next_latents": sampled,
                    "next_latents_mean": next_mean,
                    "std_dev_t": std_dev_t,
                    "dt": dt,
                    "log_prob": log_prob,
                    "velocity": velocity,
                }[field]
                for field in requested_fields
            }
        )
        if resolved_step_index is not None:
            self._step_index = resolved_step_index + 1
        if not return_dict:
            return (
                output.next_latents,
                output.next_latents_mean,
                output.velocity,
                output.log_prob,
                output.std_dev_t,
                output.dt,
            )
        return output

    def _transition_mean(
        self,
        dynamics_type: str,
        latents: torch.Tensor,
        standard_velocity: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        dt: torch.Tensor,
        noise_level: torch.Tensor,
        sigma_max: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute one transition mean in float32."""
        if dynamics_type == "ODE":
            ratio = sigma_next / sigma
            return ratio * latents + (1 - ratio) * denoised, torch.zeros_like(sigma)
        if dynamics_type == "Flow-SDE":
            maximum = self.sigmas[1].item() if sigma_max is None else float(sigma_max)
            denominator = 1 - torch.where(sigma == 1.0, maximum, sigma)
            std_dev_t = torch.sqrt(sigma / denominator) * noise_level
            mean = (
                latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + standard_velocity * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )
            return mean, std_dev_t
        if dynamics_type == "Dance-SDE":
            std_dev_t = noise_level
            log_term = 0.5 * noise_level**2 * (latents - denoised * (1 - sigma)) / sigma**2
            return latents + (standard_velocity + log_term) * dt, std_dev_t
        std_dev_t = sigma_next * torch.sin(noise_level * torch.pi / 2)
        noise_endpoint = latents + standard_velocity * (1 - sigma)
        mean = denoised * (1 - sigma_next) + noise_endpoint * torch.sqrt(
            sigma_next**2 - std_dev_t**2
        )
        return mean, std_dev_t

    @staticmethod
    def _sample_transition(
        dynamics_type: str,
        mean: torch.Tensor,
        std_dev_t: torch.Tensor,
        dt: torch.Tensor,
        reference: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        """Draw one transition from its scheduler Gaussian."""
        if dynamics_type == "ODE":
            return mean
        variance_noise = randn_tensor(
            reference.shape,
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        scale = std_dev_t if dynamics_type == "CPS" else std_dev_t * torch.sqrt(-dt)
        return mean + scale * variance_noise

    @staticmethod
    def _transition_log_prob(
        dynamics_type: str,
        sample: torch.Tensor,
        mean: torch.Tensor,
        std_dev_t: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean scalar transition log probability per sample."""
        if dynamics_type == "ODE":
            return torch.zeros(sample.shape[0], dtype=torch.float32, device=sample.device)
        if dynamics_type == "CPS":
            values = -((sample.detach() - mean) ** 2)
        else:
            variance = std_dev_t * torch.sqrt(-dt)
            safe_variance = torch.where(
                variance == 0,
                torch.ones_like(variance),
                variance,
            )
            values = (
                -((sample.detach() - mean) ** 2) / (2 * safe_variance**2)
                - torch.log(safe_variance)
                - math.log(math.sqrt(2 * math.pi))
            )
        return values.mean(dim=tuple(range(1, values.ndim)))

    def _resolve_sigmas(
        self,
        timestep: Union[float, torch.Tensor],
        timestep_next: Optional[Union[float, torch.Tensor]],
        sigma: Optional[Union[float, torch.Tensor]],
        sigma_next: Optional[Union[float, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve exact stored or explicit replay sigma coordinates."""
        if (sigma is None) != (sigma_next is None):
            raise ValueError(
                f"expected sigma and sigma_next together, received sigma={sigma!r}, "
                f"sigma_next={sigma_next!r}"
            )
        if sigma is not None:
            for field, value in (("sigma", sigma), ("sigma_next", sigma_next)):
                value_tensor = torch.as_tensor(value)
                if not bool(torch.isfinite(value_tensor).all()):
                    raise ValueError(f"expected finite {field}, received {value_tensor.tolist()}")
            current = torch.as_tensor(sigma, dtype=torch.float32)
            following = torch.as_tensor(sigma_next, dtype=torch.float32)
        elif timestep_next is not None:
            for field, value in (("timestep", timestep), ("timestep_next", timestep_next)):
                value_tensor = torch.as_tensor(value)
                if not bool(torch.isfinite(value_tensor).all()):
                    raise ValueError(f"expected finite {field}, received {value_tensor.tolist()}")
            current = flow_match_sigma(torch.as_tensor(timestep, dtype=torch.float32))
            following = flow_match_sigma(torch.as_tensor(timestep_next, dtype=torch.float32))
        else:
            index = self.index_for_timestep(timestep)
            current = self.sigmas[index]
            following = self.sigmas[index + 1]
        if current.ndim > 1 or following.ndim > 1 or current.shape != following.shape:
            raise ValueError(
                "expected sigma and sigma_next as matching scalar or per-batch tensors, "
                f"received shapes {tuple(current.shape)} and {tuple(following.shape)}"
            )
        if bool((current <= 0).any()) or bool((current > 1).any()):
            raise ValueError(f"expected sigma in (0, 1], received {current.tolist()}")
        if bool((following < 0).any()) or bool((following >= current).any()):
            raise ValueError(
                f"expected sigma_next in [0, sigma), received sigma={current.tolist()} "
                f"and sigma_next={following.tolist()}"
            )
        return current, following

    def _resolve_noise_level(
        self,
        sigma: torch.Tensor,
        noise_level: Optional[Union[int, float, torch.Tensor]],
        dynamics_type: str,
    ) -> torch.Tensor:
        """Resolve the effective stochastic scale for this transition."""
        if self.is_eval or dynamics_type == "ODE":
            return torch.zeros_like(sigma)
        if noise_level is None:
            resolved = self.get_noise_level_for_sigma(sigma)
            return torch.as_tensor(resolved, dtype=torch.float32, device=sigma.device)
        resolved = torch.as_tensor(noise_level, dtype=torch.float32, device=sigma.device)
        if (
            resolved.ndim > 1
            or not bool(torch.isfinite(resolved).all())
            or bool((resolved < 0).any())
        ):
            raise ValueError(
                "expected finite non-negative scalar or per-batch noise_level, "
                f"received {resolved.tolist()}"
            )
        return resolved

    def _index_for_sigma(self, sigma: torch.Tensor) -> int:
        """Return the exact stored index for one sigma."""
        self._require_schedule()
        value = sigma.to(device=self.sigmas.device, dtype=self.sigmas.dtype)
        matches = (self.sigmas == value).nonzero()
        if matches.numel() == 0:
            raise ValueError(f"expected sigma from scheduler.sigmas, received {value.item()!r}")
        return int(matches[0].item())

    def _validate_sde_configuration(self) -> None:
        """Validate stochastic index selection against the active schedule."""
        if self._sde_steps is not None:
            if self._sde_steps.numel() and (
                int(self._sde_steps.min()) < 0 or int(self._sde_steps.max()) >= len(self.timesteps)
            ):
                raise ValueError(
                    f"expected sde_steps in [0, {len(self.timesteps) - 1}], "
                    f"received {self._sde_steps.tolist()}"
                )
            if self._sde_steps.unique().numel() != self._sde_steps.numel():
                raise ValueError(f"expected unique sde_steps, received {self._sde_steps.tolist()}")
        if self._num_sde_steps is not None and (
            not isinstance(self._num_sde_steps, int)
            or isinstance(self._num_sde_steps, bool)
            or self._num_sde_steps < 0
            or self._num_sde_steps > len(self.sde_steps)
        ):
            raise ValueError(
                f"expected num_sde_steps in [0, {len(self.sde_steps)}], "
                f"received {self._num_sde_steps!r}"
            )

    @staticmethod
    def _validate_step_inputs(
        velocity: torch.Tensor,
        latents: torch.Tensor,
        next_latents: Optional[torch.Tensor],
    ) -> None:
        """Validate scheduler latent tensors before numerical work."""
        if not isinstance(velocity, torch.Tensor) or not isinstance(latents, torch.Tensor):
            raise TypeError(
                "expected torch.Tensor velocity and latents, received "
                f"{type(velocity).__name__} and {type(latents).__name__}"
            )
        if velocity.shape != latents.shape:
            raise ValueError(
                f"expected velocity shape {tuple(latents.shape)}, received {tuple(velocity.shape)}"
            )
        if latents.ndim < 1 or latents.shape[0] == 0:
            raise ValueError(
                f"expected latents with a non-empty batch dimension, received {tuple(latents.shape)}"
            )
        if velocity.device != latents.device or velocity.dtype != latents.dtype:
            raise ValueError(
                "expected velocity dtype/device to match latents, received "
                f"velocity {velocity.dtype}/{velocity.device} and "
                f"latents {latents.dtype}/{latents.device}"
            )
        if next_latents is not None and (
            next_latents.shape != latents.shape
            or next_latents.device != latents.device
            or next_latents.dtype not in (latents.dtype, torch.float32)
        ):
            raise ValueError(
                "expected next_latents shape/device to match latents and dtype to be the input "
                "storage dtype or float32 storage round-trip, received "
                f"{tuple(next_latents.shape)}/{next_latents.dtype}/{next_latents.device} "
                f"against {tuple(latents.shape)}/{latents.dtype}/{latents.device}"
            )

    @staticmethod
    def _validate_sigma_schedule(sigmas: torch.Tensor) -> None:
        """Require a finite strictly decreasing schedule ending at exact zero."""
        if sigmas.numel() < 2 or not bool(torch.isfinite(sigmas).all()):
            raise ValueError(
                "expected sigmas with at least two finite values, " f"received {sigmas.tolist()}"
            )
        if not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
            raise ValueError(
                "expected sigmas to be strictly decreasing and ending at exact 0, "
                f"received {sigmas.tolist()}"
            )
        if sigmas[0].item() > 1.0 or sigmas[0].item() <= 0.0:
            raise ValueError(f"expected first sigma in (0, 1], received {sigmas[0].item()}")

    def _require_schedule(self) -> None:
        """Require set_timesteps before schedule-dependent operations."""
        if self.sigmas is None or self.timesteps is None:
            raise ValueError(
                "expected set_timesteps() before MiniMaxH3SDEScheduler schedule access"
            )

    @staticmethod
    def _validate_positive_number(value: object, field: str) -> None:
        """Require one positive numeric configuration value."""
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f"expected positive finite {field}, received {type(value).__name__}: {value!r}"
            )
