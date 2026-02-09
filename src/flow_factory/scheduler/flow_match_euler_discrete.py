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

# src/flow_factory/scheduler/flow_match_euler_discrete.py

from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Literal
from argparse import Namespace
import logging
from dataclasses import dataclass, field, fields, asdict
import math

import torch
import numpy as np
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from ..utils.base import to_broadcast_tensor
from ..utils.logger_utils import setup_logger

from .abc import SDESchedulerOutput, SDESchedulerMixin

logger = setup_logger(__name__)

def calculate_shift(
    image_seq_len : int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def set_scheduler_timesteps(
    scheduler,
    num_inference_steps: int,
    seq_len: Optional[int] = None,
    sigmas: Optional[Union[List[float], np.ndarray]] = None,
    device: Optional[Union[str, torch.device]] = None,
    mu : Optional[float] = None,
):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    # 5. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
    if mu is None:
        assert seq_len is not None, "`seq_len` must be provided if `mu` is not given."
        mu = calculate_shift(
            seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps

@dataclass
class FlowMatchEulerDiscreteSDESchedulerOutput(SDESchedulerOutput):
    """
    Output class for a single SDE step in Flow Matching.
    """
    pass

class FlowMatchEulerDiscreteSDEScheduler(FlowMatchEulerDiscreteScheduler, SDESchedulerMixin):
    """
        A scheduler with noise level provided within the given steps
    """
    def __init__(
        self,
        noise_level : float = 0.7,
        sde_steps : Optional[Union[int, list, torch.Tensor]] = None,
        num_sde_steps : Optional[int] = None,
        seed : int = 42,
        dynamics_type : Literal["Flow-SDE", 'Dance-SDE', 'CPS', 'ODE'] = "Flow-SDE",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_level = noise_level

        assert self.noise_level >= 0, "Noise level must be non-negative."

        self._sde_steps = torch.tensor(sde_steps, dtype=torch.int64) if sde_steps is not None else None
        self._num_sde_steps = num_sde_steps
        self.seed = seed
        self.dynamics_type = dynamics_type
        self._is_eval = False

    @property
    def is_eval(self):
        return self._is_eval

    def eval(self):
        """Apply ODE Sampling with noise_level = 0"""
        self._is_eval = True

    def train(self, mode: bool = True):
        """Apply SDE Sampling"""
        self._is_eval = not mode

    def rollout(self, mode: bool = True):
        """Apply SDE rollout sampling"""
        self.train(mode=mode)

    @property
    def sde_steps(self) -> torch.Tensor:
        """
            Returns the step indices eligible for SDE noise injection.
        """
        if self._sde_steps is not None:
            if not isinstance(self._sde_steps, torch.Tensor):
                self._sde_steps = torch.tensor(self._sde_steps, dtype=torch.int64)
            return self._sde_steps

        # Default: all steps except the last one
        return torch.arange(0, len(self.timesteps) - 1, dtype=torch.int64)
    
    @property
    def num_sde_steps(self) -> int:
        """
            Returns the number of training steps with SDE noise.
        """
        if self._num_sde_steps is not None:
            return self._num_sde_steps

        # Default: all train steps
        return len(self.sde_steps)

    @property
    def current_sde_steps(self) -> torch.Tensor:
        """
            Returns the current SDE step indices under the self.seed.
            Randomly select self.num_train_steps from self.train_steps.
        """
        if self.num_sde_steps >= len(self.sde_steps):
            return self.sde_steps
        generator = torch.Generator().manual_seed(self.seed)
        selected_indices = torch.randperm(len(self.sde_steps), generator=generator)[:self.num_sde_steps]
        return self.sde_steps[selected_indices]

    @property
    def train_timesteps(self) -> torch.Tensor:
        """
            Returns timestep **indices** that to train on.
        """
        return self.current_sde_steps

    def get_train_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps [0, 1000] within the current window.
        """
        return self.timesteps[self.train_timesteps]

    def get_train_sigmas(self) -> torch.Tensor:
        """
            Returns sigmas within the current window.
        """
        return self.sigmas[self.train_timesteps]

    def get_noise_levels(self) -> torch.Tensor:
        """ Returns noise levels on all timesteps, where noise level is non-zero only within the current window. """
        noise_levels = torch.zeros_like(self.timesteps, dtype=torch.float32)
        noise_levels[self.current_sde_steps] = self.noise_level
        return noise_levels

    def get_noise_level_for_timestep(self, timestep : Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
            Return the noise level for a specific timestep.
        """
        if not isinstance(timestep, torch.Tensor) or timestep.ndim == 0:
            t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
            timestep_index = self.index_for_timestep(t)
            return self.noise_level if timestep_index in self.current_sde_steps else 0.0

        indices = torch.tensor([self.index_for_timestep(t.item()) for t in timestep])
        mask = torch.isin(indices, self.current_sde_steps)
        return torch.where(mask, self.noise_level, 0.0).to(timestep.dtype)


    def get_noise_level_for_sigma(self, sigma : float) -> float:
        """
            Return the noise level for a specific sigma.
        """
        # Find the index that corresponds to the given sigma, no tolerance
        indices = (self.sigmas == sigma).nonzero()
        if len(indices) == 0:
            raise ValueError(f"Sigma {sigma} not found in scheduler sigmas {self.sigmas}")
        pos = 1 if len(indices) > 1 else 0
        index = indices[pos].item()
        if index in self.current_sde_steps:
            return self.noise_level
        return 0.0
    
    def set_seed(self, seed: int):
        """
            Set the random seed for noise steps.
        """
        self.seed = seed

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        latents: torch.Tensor,
        next_latents: Optional[torch.Tensor] = None,
        timestep_next: Optional[Union[float, torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise_level : Optional[Union[int, float, torch.Tensor]] = None,
        compute_log_prob: bool = True,
        return_dict: bool = True,
        return_kwargs : List[str] = ['next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob', 'noise_pred'],
        dynamics_type : Optional[Literal['Flow-SDE', 'Dance-SDE', 'CPS', 'ODE']] = None,
        sigma_max: Optional[float] = None,
    ) -> Union[FlowMatchEulerDiscreteSDESchedulerOutput, Tuple]:
        if timestep_next is None:
            # Get step index and the `timestep_next`
            if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
            ):
                logger.warning(
                    (
                        "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to `FlowMatchEulerDiscreteSDEScheduler.step()`"
                        ", rather than one of the `scheduler.timesteps` as a timestep."
                    ),
                )
                step_index = [int(timestep)] # (1,)
            elif isinstance(timestep, torch.Tensor):
                # Find step_index
                if timestep.ndim == 0:
                    # Scalar tensor
                    step_index = [self.index_for_timestep(timestep)] # (1,)
                elif timestep.ndim == 1:
                    # Batched 1D tensor (B,)
                    step_index = [self.index_for_timestep(t) for t in timestep]
                else:
                    raise ValueError(
                        f"`timestep` must be a scalar or 1D tensor, got shape {tuple(timestep.shape)}. "
                        f"If using expanded timesteps (e.g. for Wan models), pass the original scalar timestep `t` instead."
                    )
            elif isinstance(timestep, float):
                step_index = [self.index_for_timestep(timestep)] # (1, )
            else:
                raise TypeError(f"`timestep` must be float, or torch.Tensor, got {type(timestep).__name__}.")
            
            # Update `timestep` and `timestep_next`
            timestep = self.timesteps[step_index]
            timestep_next = torch.as_tensor([
                self.timesteps[i + 1] if i + 1 < len(self.timesteps)
                else torch.tensor(0, device=timestep.device)
                for i in step_index
            ], device=timestep.device)
            # Update sigma
            sigma = self.sigmas[step_index]
            sigma_prev = self.sigmas[[i + 1 for i in step_index]]
        else:
            # `timestep_next` provided
            sigma = timestep / 1000
            sigma_prev = timestep_next / 1000

        # 1. Numerical Preparation
        noise_pred = noise_pred.float()
        latents = latents.float()
        if next_latents is not None:
            next_latents = next_latents.float()

        # 2. Prepare variables
        dynamics_type = dynamics_type or self.dynamics_type
        if (self.is_eval or dynamics_type == 'ODE'):
            noise_level = 0.0
        elif noise_level is None:
            # Auto-infer the noise_level
            noise_level = self.get_noise_level_for_sigma(sigma)

        noise_level = to_broadcast_tensor(noise_level, latents) # To (B, 1, 1)
        sigma = to_broadcast_tensor(sigma, latents)
        sigma_prev = to_broadcast_tensor(sigma_prev, latents)
        dt = sigma_prev - sigma # dt is negative, (batch_size, 1, 1)

        # 3. Compute next sample
        if dynamics_type == 'ODE':
            # ODE Sampling
            next_latents_mean = latents + noise_pred * dt
            std_dev_t = torch.zeros_like(sigma)

            if next_latents is None:
                next_latents = next_latents_mean

            if compute_log_prob:
                # ODE doesn't support log_prob computation, provide zero
                logger.warning(f"`log_prob` is meaningless when `dynamics_type` is set `ODE`, setting to zero.")
                log_prob = torch.zeros((next_latents.shape[0]), dtype=next_latents.dtype, device=next_latents.device)

        elif dynamics_type == "Flow-SDE":
            # FlowGRPO sde
            sigma_max = sigma_max or self.sigmas[1].item() # To avoid dividing by zero
            sigma_max = to_broadcast_tensor(sigma_max, latents)
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1.0, sigma_max, sigma))) * noise_level # (batch_size, 1, 1)

            next_latents_mean = latents * (1 + std_dev_t**2 / (2 * sigma) * dt) + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            
            if next_latents is None:
                # Non-deterministic step, add noise to it
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                # Last term of Equation (9)
                next_latents = next_latents_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

            if compute_log_prob:
                std_variance = (std_dev_t * torch.sqrt(-1 * dt))
                log_prob = (
                    -((next_latents.detach() - next_latents_mean) ** 2) / (2 * std_variance ** 2)
                    - torch.log(std_variance)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif dynamics_type == "Dance-SDE":
            pred_original_sample = latents - sigma * noise_pred
            std_dev_t = noise_level * torch.sqrt(-1 * dt)
            log_term = 0.5 * noise_level**2 * (latents - pred_original_sample * (1 - sigma)) / sigma**2
            next_latents_mean = latents + (noise_pred + log_term) * dt
            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * variance_noise
            
            if compute_log_prob:
                log_prob = (
                    (-((next_latents.detach() - next_latents_mean) ** 2) / (2 * (std_dev_t**2)))
                    - math.log(std_dev_t)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )

                # mean along all but batch dimension
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif dynamics_type == "CPS":
            # FlowCPS
            std_dev_t = sigma_prev * torch.sin(noise_level * torch.pi / 2)
            x0 = latents - sigma * noise_pred
            x1 = latents + noise_pred * (1 - sigma)
            next_latents_mean = x0 * (1 - sigma_prev) + x1 * torch.sqrt(sigma_prev**2 - std_dev_t**2)
        
            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * variance_noise

            if compute_log_prob:
                log_prob = -((next_latents.detach() - next_latents_mean) ** 2)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


        if not compute_log_prob:
            # # Empty tensor as placeholder
            # log_prob = torch.empty((latents.shape[0]), dtype=torch.float32, device=noise_pred.device)
            log_prob = None # Use None to save memory

        if not return_dict:
            return (next_latents, next_latents_mean, noise_pred, log_prob, std_dev_t, dt)

        d = {}        
        for k in return_kwargs:
            if k in locals():
                d[k] = locals()[k]
            else:
                logger.warning(f"Requested return keyword '{k}' is not available in the step output.")

        return SDESchedulerOutput.from_dict(d)