from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Literal
from argparse import Namespace
import logging
from dataclasses import dataclass
import math

import torch
import numpy as np
from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ..utils.base import to_broadcast_tensor


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

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
    seq_len: int,
    sigmas: Optional[Union[List[float], np.ndarray]] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    # 5. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
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
class FlowMatchEulerDiscreteSDESchedulerOutput(BaseOutput):
    """
    Output class for a single SDE step in Flow Matching.
    """

    prev_sample: torch.FloatTensor
    prev_sample_mean: torch.FloatTensor
    std_dev_t: torch.FloatTensor
    dt: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None

class FlowMatchEulerDiscreteSDEScheduler(FlowMatchEulerDiscreteScheduler):
    """
        A scheduler with noise level provided within the given steps
    """
    def __init__(
        self,
        noise_level : float = 0.7,
        noise_steps : Optional[Union[int, list, torch.Tensor]] = None,
        num_noise_steps : Optional[int] = None,
        seed : int = 42,
        sde_type : Literal["Flow-SDE", 'Dance-SDE', 'CPS'] = "Flow-SDE",
        **kwargs
    ):
        super().__init__(**kwargs)
        if noise_steps is None:
            # Default to all noise steps
            noise_steps = list(range(len(self.timesteps)))

        self.noise_level = noise_level

        assert self.noise_level >= 0, "Noise level must be non-negative."

        self.noise_steps = torch.tensor(noise_steps, dtype=torch.int64)
        self.num_noise_steps = num_noise_steps if num_noise_steps is not None else len(noise_steps) # Default to all noise steps
        self.seed = seed
        self.sde_type = sde_type
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
    def current_noise_steps(self) -> torch.Tensor:
        """
            Returns the current noise steps under the self.seed.
            Randomly select self.num_noise_steps from self.noise_steps.
        """
        if self.num_noise_steps >= len(self.noise_steps):
            return self.noise_steps
        generator = torch.Generator().manual_seed(self.seed)
        selected_indices = torch.randperm(len(self.noise_steps), generator=generator)[:self.num_noise_steps]
        return self.noise_steps[selected_indices]

    @property
    def train_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps that to train on.
        """
        return self.current_noise_steps

    def get_noise_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps within the current window.
        """
        return self.timesteps[self.noise_steps]

    def get_noise_sigmas(self) -> torch.Tensor:
        """
            Returns sigmas within the current window.
        """
        return self.sigmas[self.noise_steps]

    def get_noise_levels(self) -> torch.Tensor:
        """ Returns noise levels on all timesteps, where noise level is non-zero only within the current window. """
        noise_levels = torch.zeros_like(self.timesteps, dtype=torch.float32)
        noise_levels[self.current_noise_steps] = self.noise_level
        return noise_levels

    def get_noise_level_for_timestep(self, time_step) -> float:
        """
            Return the noise level for a specific timestep.
        """
        time_step_index = self.index_for_timestep(time_step)
        if time_step_index in self.noise_steps:
            return self.noise_level

        return 0.0

    def get_noise_level_for_sigma(self, sigma) -> float:
        """
            Return the noise level for a specific sigma.
        """
        sigma_index = (self.sigmas - sigma).abs().argmin().item()
        if sigma_index in self.noise_steps:
            return self.noise_level

        return 0.0
    
    def set_seed(self, seed: int):
        """
            Set the random seed for noise steps.
        """
        self.seed = seed

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[int, float, torch.Tensor],
        sample: torch.FloatTensor,
        prev_sample: Optional[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise_level : Optional[Union[int, float, torch.Tensor]] = None,
        compute_log_prob: bool = True,
        return_dict: bool = True,
        sde_type : Optional[Literal['Flow-SDE', 'Dance-SDE', 'CPS']] = None,
        sigma_max: Optional[float] = None,
    ):
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
        elif isinstance(timestep, (float, torch.Tensor)):
            is_batched = isinstance(timestep, torch.Tensor) and timestep.ndim > 0
            if is_batched:
                step_index = [self.index_for_timestep(t) for t in timestep] # (B,)
            else:
                step_index = [self.index_for_timestep(timestep)]

            sigma = self.sigmas[step_index] # (B, ) or (1, )
            sigma_prev = self.sigmas[[i + 1 for i in step_index]] # (B, ) or (1, )

        # 1. Numerical Preparation
        model_output = model_output.float()
        sample = sample.float()
        if prev_sample is not None:
            prev_sample = prev_sample.float()

        # 2. Prepare variables
        noise_level = noise_level or (
            0.0 if self.is_eval else self.get_noise_level_for_timestep(timestep)
        )
        noise_level = to_broadcast_tensor(noise_level, sample) # To (B, 1, 1)
        sigma = to_broadcast_tensor(sigma, sample)
        sigma_prev = to_broadcast_tensor(sigma_prev, sample)
        dt = sigma_prev - sigma # dt is negative, (batch_size, 1, 1)

        sde_type = sde_type or self.sde_type
        # 3. Compute next sample
        if sde_type == "Flow-SDE":
            # FlowGRPO sde
            sigma_max = sigma_max or self.sigmas[1].item() # To avoid dividing by zero
            sigma_max = to_broadcast_tensor(sigma_max, sample)
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1.0, sigma_max, sigma))) * noise_level # (batch_size, 1, 1)

            prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            
            if prev_sample is None:
                # Non-deterministic step, add noise to it
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                # Last term of Equation (9)
                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

            if compute_log_prob:
                std_variance = (std_dev_t * torch.sqrt(-1 * dt))
                log_prob = (
                    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_variance ** 2)
                    - torch.log(std_variance)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif sde_type == "Dance-SDE":
            pred_original_sample = sample - sigma * model_output
            std_dev_t = noise_level * torch.sqrt(-1 * dt)
            log_term = 0.5 * noise_level**2 * (sample - pred_original_sample * (1 - sigma)) / sigma**2
            prev_sample_mean = sample + (model_output + log_term) * dt
            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise
            
            if compute_log_prob:
                log_prob = (
                    (-((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2)))
                    - math.log(std_dev_t)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )

                # mean along all but batch dimension
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif sde_type == "CPS":
            # FlowCPS
            std_dev_t = sigma_prev * torch.sin(noise_level * torch.pi / 2)
            x0 = sample - sigma * model_output
            x1 = sample + model_output * (1 - sigma)
            prev_sample_mean = x0 * (1 - sigma_prev) + x1 * torch.sqrt(sigma_prev**2 - std_dev_t**2)
        
            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            if compute_log_prob:
                log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


        if not compute_log_prob:
            # Empty tensor as placeholder
            log_prob = torch.empty((sample.shape[0]), dtype=torch.float32, device=model_output.device)

        if not return_dict:
            return (prev_sample, log_prob, prev_sample_mean, std_dev_t, dt)

        return FlowMatchEulerDiscreteSDESchedulerOutput(
            prev_sample=prev_sample,
            prev_sample_mean=prev_sample_mean,
            std_dev_t=std_dev_t,
            log_prob=log_prob,
            dt=dt,
        )