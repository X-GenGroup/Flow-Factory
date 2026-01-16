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

# src/flow_factory/scheduler/unipc_multistep.py
import math
from dataclasses import dataclass, fields, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import numpy as np
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from ..utils.base import to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from .abc import SDESchedulerOutput, SDESchedulerMixin

logger = setup_logger(__name__)

@dataclass
class UniPCMultistepSDESchedulerOutput(SDESchedulerOutput):
    pass

class UniPCMultistepSDEScheduler(UniPCMultistepScheduler, SDESchedulerMixin):
    """
    UniPC scheduler with SDE sampling support for RL fine-tuning.
    
    Extends UniPCMultistepScheduler with:
    - Stochastic sampling via configurable noise injection
    - Log probability computation for policy gradient methods
    - Train/eval mode switching
    
    Args (additional to UniPCMultistepScheduler):
        noise_level: Noise scaling factor for SDE sampling. Default 0.7.
        train_steps: Indices of steps to apply SDE noise. Default all steps.
        num_train_steps: Number of train steps to sample per rollout.
        seed: Random seed for selecting train steps.
        dynamics_type: "SDE" or "ODE". SDE adds stochastic noise.
    """

    def __init__(
        self,
        noise_level : float = 0.7,
        train_steps : Optional[Union[int, list, torch.Tensor]] = None,
        num_train_steps : Optional[int] = None,
        seed : int = 42,
        dynamics_type : Literal["Flow-SDE", 'Dance-SDE', 'CPS', 'ODE'] = "Flow-SDE",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if train_steps is None:
            # Default to all noise steps
            train_steps = list(range(len(self.timesteps)))

        self.noise_level = noise_level

        assert self.noise_level >= 0, "Noise level must be non-negative."

        self.train_steps = torch.tensor(train_steps, dtype=torch.int64)
        self.num_train_steps = num_train_steps if num_train_steps is not None else len(train_steps) # Default to all noise steps
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
    def current_sde_steps(self) -> torch.Tensor:
        """
            Returns the current SDE step indices under the self.seed.
            Randomly select self.num_train_steps from self.train_steps.
        """
        if self.num_train_steps >= len(self.train_steps):
            return self.train_steps
        generator = torch.Generator().manual_seed(self.seed)
        selected_indices = torch.randperm(len(self.train_steps), generator=generator)[:self.num_train_steps]
        return self.train_steps[selected_indices]

    @property
    def train_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps that to train on.
        """
        return self.current_sde_steps

    def get_train_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps within the current window.
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


    def get_noise_level_for_sigma(self, sigma) -> float:
        """
            Return the noise level for a specific sigma.
        """
        sigma_index = (self.sigmas - sigma).abs().argmin().item()
        if sigma_index in self.train_steps:
            return self.noise_level

        return 0.0
    
    def set_seed(self, seed: int):
        """
            Set the random seed for noise steps.
        """
        self.seed = seed

    def step(
        self,
        noise_pred: torch.FloatTensor,
        timestep: Union[int, float, torch.Tensor],
        latents: torch.FloatTensor,
        next_latents: Optional[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise_level : Optional[Union[int, float, torch.Tensor]] = None,
        compute_log_prob: bool = True,
        return_dict: bool = True,
        return_kwargs : List[str] = ['next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
        dynamics_type : Optional[Literal['Flow-SDE', 'Dance-SDE', 'CPS', 'ODE']] = None,
        sigma_max: Optional[float] = None,
    ) -> Union[UniPCMultistepSDESchedulerOutput, Tuple]:
        """
        SDE step for UniPC scheduler.
        
        Args:
            noise_pred (torch.FloatTensor): The predicted noise residual.
            timestep (int, float, or torch.Tensor): The current timestep.
            latents (torch.FloatTensor): The current latent tensor.
            next_latents (torch.FloatTensor, optional): If provided, use this as the next latents instead of sampling.
            generator (torch.Generator or list of torch.Generator, optional): Random generator(s) for noise sampling.
            noise_level (int, float, or torch.Tensor, optional): Noise scaling factor for SDE sampling. If None, uses scheduler's noise level.
            compute_log_prob (bool): Whether to compute log probability of the step.
            return_dict (bool): Whether to return a UniPCMultistepSDESchedulerOutput or a tuple.
            return_kwargs (list of str): List of output fields to return in the output dataclass.
            dynamics_type (str, optional): "Flow-SDE", "Dance-SDE", "CPS", or "ODE". If None, uses scheduler's dynamics type.
            sigma_max (float, optional): Maximum sigma value for Flow-SDE dynamics.
        """
        if self.is_eval:
            # Calcuate ODE step using `super()`
            next_latents_mean = super().step(noise_pred, timestep, latents, return_dict=False)[0]
            return UniPCMultistepSDESchedulerOutput(next_latents=next_latents_mean)

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
            step_index = int(timestep)
            timestep = self.timesteps[step_index]
            sigma = self.sigmas[step_index] # (1)
            sigma_prev = self.sigmas[step_index + 1] # (1)
        elif isinstance(timestep, torch.Tensor):
            if timestep.ndim == 0:
                # Scalar tensor
                step_index = [self.index_for_timestep(timestep)]
            elif timestep.ndim == 1:
                # Batched 1D tensor (B,)
                step_index = [self.index_for_timestep(t) for t in timestep]
            else:
                raise ValueError(
                    f"`timestep` must be a scalar or 1D tensor, got shape {tuple(timestep.shape)}. "
                    f"If using expanded timesteps (e.g. for Wan models), pass the original scalar timestep `t` instead."
                )
            sigma = self.sigmas[step_index]
            sigma_prev = self.sigmas[[i + 1 for i in step_index]]
        elif isinstance(timestep, float):
            step_index = [self.index_for_timestep(timestep)]
            sigma = self.sigmas[step_index]
            sigma_prev = self.sigmas[[i + 1 for i in step_index]]
        else:
            raise TypeError(f"`timestep` must be float, or torch.Tensor, got {type(timestep).__name__}.")

        # 1. Numerical Preparation
        noise_pred = noise_pred.float()
        latents = latents.float()
        if next_latents is not None:
            next_latents = next_latents.float()

        # 2. Prepare variables
        noise_level = noise_level or (
            0.0 if self.is_eval else self.get_noise_level_for_timestep(timestep)
        )
        noise_level = to_broadcast_tensor(noise_level, latents) # To (B, 1, 1)
        sigma = to_broadcast_tensor(sigma, latents)
        sigma_prev = to_broadcast_tensor(sigma_prev, latents)
        dt = sigma_prev - sigma # dt is negative, (batch_size, 1, 1)

        dynamics_type = dynamics_type or self.dynamics_type
        # 3. Compute next sample
        if dynamics_type == 'ODE':
            # ODE Sampling
            next_latents_mean = latents + noise_pred * dt
            std_dev_t = torch.zeros_like(sigma)

            if next_latents is None:
                next_latents = next_latents_mean

            if compute_log_prob:
                log_prob = -((next_latents.detach() - next_latents_mean) ** 2)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

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
            # Empty tensor as placeholder
            log_prob = torch.empty((latents.shape[0]), dtype=torch.float32, device=noise_pred.device)

        if not return_dict:
            return (next_latents, log_prob, next_latents_mean, std_dev_t, dt)

        d = {}        
        for k in return_kwargs:
            if k in locals():
                d[k] = locals()[k]
            else:
                logger.warning(f"Requested return keyword '{k}' is not available in the step output.")

        return UniPCMultistepSDESchedulerOutput.from_dict(d)