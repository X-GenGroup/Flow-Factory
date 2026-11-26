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

"""Deterministic DPM-Solver++ 2M for SD3.5 flow prediction."""

import torch

from diffusers import DPMSolverMultistepScheduler

from .abc import SDESchedulerMixin, SDESchedulerOutput


class FlowDPMpp2MScheduler(DPMSolverMultistepScheduler, SDESchedulerMixin):
    """Use the iTDM rollout schedule and preserve history across velocity-only queries."""

    dynamics_type = "ODE"
    noise_level = 0.0
    num_sde_steps = 0

    def __init__(self, seed=42, replay_steps="auto"):
        super().__init__(
            algorithm_type="dpmsolver++",
            prediction_type="flow_prediction",
            solver_order=2,
            solver_type="midpoint",
            use_flow_sigmas=True,
            flow_shift=3.0,
            # Below every lambda on the finite training grid, so clipping is inactive.
            # Use a finite value for the exact-resume configuration digest.
            lambda_min_clipped=-1e30,
            lower_order_final=True,
            final_sigmas_type="zero",
        )
        self.seed = seed
        self._is_eval = False
        self.set_replay_steps(replay_steps)

    def set_timesteps(self, num_inference_steps, device=None, mu=None):
        # SD3.5 supplies a resolution-dependent mu. The upstream iTDM schedule
        # uses fixed flow_shift=3 at all resolutions.
        super().set_timesteps(num_inference_steps, device=device)
        self._require_finite_schedule()

    @property
    def is_eval(self):
        return self._is_eval

    def train(self, mode=True):
        self._is_eval = not mode

    def eval(self):
        self.train(False)

    def rollout(self, mode=True):
        self.train(mode)

    def set_seed(self, seed):
        self.seed = seed

    @property
    def sde_steps(self):
        return torch.empty(0, dtype=torch.long)

    current_sde_steps = sde_steps

    def get_train_timesteps(self):
        return self.timesteps[self.train_timesteps]

    def get_train_sigmas(self):
        return self.sigmas[self.train_timesteps]

    def get_noise_levels(self):
        return torch.zeros_like(self.timesteps, dtype=torch.float32)

    def get_noise_level_for_timestep(self, timestep):
        if isinstance(timestep, torch.Tensor) and timestep.ndim:
            return torch.zeros_like(timestep, dtype=torch.float32)
        return 0.0

    def get_noise_level_for_sigma(self, sigma):
        return self.get_noise_level_for_timestep(sigma)

    def step(
        self,
        velocity,
        timestep,
        latents,
        timestep_next=None,
        next_latents=None,
        compute_log_prob=False,
        return_dict=True,
        return_kwargs=("next_latents",),
        noise_level=None,
    ):
        if compute_log_prob or next_latents is not None:
            raise ValueError("DPM++ 2M supports deterministic generation and velocity queries")
        fields = set(return_kwargs)
        unsupported = fields - {"velocity", "next_latents", "next_latents_mean", "log_prob"}
        if unsupported:
            raise ValueError(f"Unsupported DPM++ 2M return fields: {sorted(unsupported)}")
        result = SDESchedulerOutput(velocity=velocity if "velocity" in fields else None)
        if fields.intersection(("next_latents", "next_latents_mean")):
            updated = super().step(velocity, timestep, latents).prev_sample
            result = SDESchedulerOutput(
                velocity=result.velocity,
                next_latents=updated if "next_latents" in fields else None,
                next_latents_mean=updated if "next_latents_mean" in fields else None,
            )
        return result if return_dict else result.to_tuple()
