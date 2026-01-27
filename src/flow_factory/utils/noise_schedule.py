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

# src/flow_factory/utils/noise_schedule.py
"""
Utility functions for noise schedule and time sampling.
"""
import torch
from typing import Union

# ============================ Time Samplers ============================
class TimeSampler:
    """Continuous and discrete time sampler for flow matching training."""
    
    @staticmethod
    def logit_normal_shifted(
        batch_size: int,
        num_timesteps: int,
        m: float = 0.0,
        s: float = 1.0,
        shift: float = 3.0,
        device: torch.device = torch.device('cpu'),
        stratified: bool = True,
    ) -> torch.Tensor:
        """
        Logit-normal shifted time sampling.
        
        Returns:
            Tensor of shape (num_timesteps, batch_size) with t in (0, 1).
        """
        if stratified:
            base = (torch.arange(num_timesteps, device=device) + torch.rand(num_timesteps, device=device)) / num_timesteps
            normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
            u_standard = normal_dist.icdf(torch.clamp(base, 1e-7, 1 - 1e-7))
            u_standard = u_standard[torch.randperm(num_timesteps, device=device)]
        else:
            u_standard = torch.randn(num_timesteps, device=device)
        
        u = u_standard * s + m
        t = torch.sigmoid(u)
        t = shift * t / (1 + (shift - 1) * t)
        t = torch.clamp(t, min=0.01)
        
        return t.unsqueeze(1).expand(num_timesteps, batch_size)
    
    @staticmethod
    def uniform(
        batch_size: int,
        num_timesteps: int,
        lower: float = 0.2,
        upper: float = 1.0,
        shift: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Uniform time sampling with optional shift.

        Args:
            batch_size: Number of samples per timestep.
            num_timesteps: Number of timesteps to sample.
            lower: Lower bound of uniform distribution.
            upper: Upper bound of uniform distribution.
            shift: Shift factor for time warping.
            device: Target device.

        Returns:
            Tensor of shape (num_timesteps, batch_size).
        """
        # Stratified uniform sampling
        rand_u = torch.rand(num_timesteps, device=device)
        normalized = (torch.arange(num_timesteps, device=device) + rand_u) / num_timesteps
        t = lower + normalized * (upper - lower)
        t = t[torch.randperm(num_timesteps, device=device)]
        t = shift * t / (1 + (shift - 1) * t)

        return t.unsqueeze(1).expand(-1, batch_size)
    
    # ======================== Discrete Time Samplers ========================
    @staticmethod
    def discrete(
        batch_size: int,
        num_train_timesteps: int,
        scheduler_timesteps: torch.Tensor,
        timestep_fraction: float = 1.0,
        normalize: bool = True,
        include_init: bool = True,
        force_init: bool = False,
    ) -> torch.Tensor:
        """
        Discrete stratified time sampling from scheduler timesteps.
        
        Args:
            batch_size: Number of samples per timestep.
            num_train_timesteps: Number of training timesteps to sample.
            scheduler_timesteps: Actual timesteps from scheduler, shape (num_inference_steps,).
            timestep_fraction: Fraction of trajectory to use (0, 1].
            normalize: If True, normalize timesteps to (0, 1) by dividing by 1000.
            include_init: If True, index 0 is included in sampling range.
            force_init: If True, first sampled timestep is always index 0.
                        (implies include_init=False for remaining samples)
        
        Returns:
            Tensor of shape (num_train_timesteps, batch_size) with sampled timesteps.
        """
        device = scheduler_timesteps.device
        max_idx = int(len(scheduler_timesteps) * timestep_fraction)
        
        if force_init:
            # First is always 0, sample remaining from [1, max_idx]
            if num_train_timesteps == 1:
                t_indices = torch.zeros(1, device=device, dtype=torch.long)
            else:
                start_idx, num_samples = 1, num_train_timesteps - 1
                t_indices = torch.cat([
                    torch.zeros(1, device=device, dtype=torch.long),
                    TimeSampler._stratified_sample(num_samples, start_idx, max_idx, device),
                ])
        else:
            start_idx = 0 if include_init else 1
            t_indices = TimeSampler._stratified_sample(
                num_train_timesteps, start_idx, max_idx, device
            )
        
        t_indices = t_indices.clamp(max=len(scheduler_timesteps) - 1)
        timesteps = scheduler_timesteps[t_indices].unsqueeze(1).expand(-1, batch_size)
        
        return timesteps.float() / 1000.0 if normalize else timesteps

    @staticmethod
    def _stratified_sample(
        num_samples: int,
        start_idx: int,
        end_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Stratified sampling of indices from [start_idx, end_idx]."""
        boundaries = torch.linspace(start_idx, end_idx, num_samples + 1, device=device)
        lower, upper = boundaries[:-1].long(), boundaries[1:].long()
        rand_u = torch.rand(num_samples, device=device)
        return lower + (rand_u * (upper - lower)).long()