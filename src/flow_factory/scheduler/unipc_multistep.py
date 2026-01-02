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

logger = setup_logger(__name__)

@dataclass
class UniPCMultistepSDESchedulerOutput(BaseOutput):
    next_latents: Optional[torch.FloatTensor] = None
    next_latents_mean: Optional[torch.FloatTensor] = None
    std_dev_t: Optional[torch.FloatTensor] = None
    dt: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None
    noise_pred: Optional[torch.FloatTensor] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniPCMultistepSDESchedulerOutput":
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

class UniPCMultistepSDEScheduler(UniPCMultistepScheduler):
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

    def train(self, *args, **kwargs):
        """Apply SDE Sampling"""
        self._is_eval = False

    def rollout(self, *args, **kwargs):
        """Apply SDE rollout sampling"""
        self.train(*args, **kwargs)

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
        return self.timesteps[self.train_steps]

    def get_train_sigmas(self) -> torch.Tensor:
        """
            Returns sigmas within the current window.
        """
        return self.sigmas[self.train_steps]

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
            return self.noise_level if timestep_index in self.train_steps else 0.0

        indices = torch.tensor([self.index_for_timestep(t.item()) for t in timestep])
        mask = torch.isin(indices, self.train_steps)
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

        TODO: The following code is adapted by Gemini. To be verified and tested.
        """
        noise_pred = noise_pred.float()
        latents = latents.float()
        if next_latents is not None:
            next_latents = next_latents.float()

        # 1. Calcuate ODE step using `super()`
        next_latents_ode = super().step(noise_pred, timestep, latents, return_dict=False)[0]

        # If evaluating as pure ODE, return immediately
        dynamics_type = dynamics_type or self.dynamics_type
        if self.is_eval or dynamics_type == "ODE":
            if not return_dict:
                return (next_latents_ode,)
            return UniPCMultistepSDESchedulerOutput(next_latents=next_latents_ode)

        # 2. Prepare SDE noise parameters
        step_index = self.step_index - 1 # super().step() increments index
        sigma = self.sigmas[step_index]
        sigma_prev = self.sigmas[step_index + 1]
        dt = sigma_prev - sigma

        noise_level = noise_level or self.get_noise_level_for_timestep(timestep)
        noise_level = to_broadcast_tensor(noise_level, latents)
        sigma = to_broadcast_tensor(sigma, latents)
        sigma_prev = to_broadcast_tensor(sigma_prev, latents)
        dt = to_broadcast_tensor(dt, latents)

        v_t = noise_pred
        x_t = latents

        drift_correction = torch.zeros_like(latents)
        std_dev_t = torch.zeros_like(sigma)

        if dynamics_type == "Flow-SDE":            
            # Default sigma_max to 1.0 (standard for flow matching) or 2nd sigma
            s_max = sigma_max or 1.0 
            s_max = to_broadcast_tensor(s_max, latents)

            # Calculation of variance scale
            # std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level (simplified from source)
            # Note: Source uses: sqrt(sigma / (1 - where(sigma==1, max, sigma)))
            denom = 1 - torch.where(sigma == 1.0, s_max, sigma)
            # Avoid division by zero issues
            denom = torch.clamp(denom, min=1e-5)
            std_dev_t = torch.sqrt(sigma / denom) * noise_level

            # SDE Mean Formula:
            # mean = x * (1 + std^2/(2*sigma)*dt) + v * (1 + std^2*(1-sigma)/(2*sigma))*dt
            
            term_x = (std_dev_t**2 / (2 * sigma)) * dt
            term_v = (std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            
            # The SDE implies this specific shift in mean:
            sde_mean = x_t * (1 + term_x) + v_t * dt + v_t * term_v
            
            # The ODE (Euler) implies: x_t + v_t * dt
            euler_mean = x_t + v_t * dt
            
            # The "Drift Correction" to apply to UniPC
            drift_correction = sde_mean - euler_mean

        elif dynamics_type == "Dance-SDE":
            # Dance-SDE Logic
            # std_dev = noise_level * sqrt(-dt)
            std_dev_t = noise_level * torch.sqrt(-1 * dt)
            
            # Log term correction
            # log_term = 0.5 * noise^2 * (x - (x - sigma*v)*(1-sigma)) / sigma^2
            # Simplified: pred_x0 = x - sigma*v
            pred_x0 = x_t - sigma * v_t
            log_term = 0.5 * noise_level**2 * (x_t - pred_x0 * (1 - sigma)) / (sigma**2)
            
            # Correction is simply log_term * dt
            drift_correction = log_term * dt

        elif dynamics_type == "CPS":
            # CPS Logic
            # CPS is structural (geometric mixing), hard to map as a drift correction to UniPC.
            # We will approximate CPS by using UniPC trajectory + CPS Noise variance.
            
            # std_dev_t = sigma_prev * sin(noise_level * pi / 2)
            std_dev_t = sigma_prev * torch.sin(noise_level * math.pi / 2)
            
            # No specific drift correction applied to UniPC mean for CPS in this approximation
            drift_correction = 0.0

        # 4. Apply Correction and Noise
        
        # Apply drift correction to UniPC's high-order mean
        next_latents_mean = next_latents_ode + drift_correction
        
        # Generate Noise
        variance_noise = randn_tensor(
            noise_pred.shape,
            generator=generator,
            device=noise_pred.device,
            dtype=noise_pred.dtype,
        )

        if dynamics_type == "Flow-SDE":
            # noise_term = std_dev * sqrt(-dt) * eps
            noise_val = std_dev_t * torch.sqrt(-1 * dt) * variance_noise
        elif dynamics_type == "Dance-SDE":
             # noise_term = std_dev * eps (std_dev already includes sqrt(-dt))
             noise_val = std_dev_t * variance_noise
        elif dynamics_type == "CPS":
             noise_val = std_dev_t * variance_noise
        else:
             noise_val = 0.0

        next_latents = next_latents_mean + noise_val

        # 5. Compute Log Prob (Optional)
        log_prob = torch.empty((latents.shape[0]), dtype=torch.float32, device=latents.device)
        if compute_log_prob and dynamics_type in ["Flow-SDE", "Dance-SDE"]:
            if dynamics_type == "Flow-SDE":
                 std_variance = (std_dev_t * torch.sqrt(-1 * dt))
            else: 
                 std_variance = std_dev_t
            
            log_prob = (
                -((next_latents.detach() - next_latents_mean) ** 2) / (2 * std_variance ** 2)
                - torch.log(std_variance)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        if not return_dict:
            return (next_latents, next_latents_mean, std_dev_t, log_prob)

        d = {}        
        for k in return_kwargs:
            if k in locals():
                d[k] = locals()[k]
            else:
                logger.warning(f"Requested return keyword '{k}' is not available in the step output.")

        return UniPCMultistepSDESchedulerOutput.from_dict(d)