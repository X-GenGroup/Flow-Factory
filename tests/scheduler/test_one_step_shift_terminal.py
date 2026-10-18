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

import warnings

import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from flow_factory.scheduler.flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSDEScheduler,
    set_scheduler_timesteps,
)

_CONFIG = {
    "shift_terminal": 0.02,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 0.9,
}


def test_one_step_schedule_with_shift_terminal_steps_from_noise_to_data() -> None:
    scheduler = FlowMatchEulerDiscreteSDEScheduler(**_CONFIG)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timesteps = set_scheduler_timesteps(scheduler, 1, seq_len=4096, device="cpu")

    assert timesteps.tolist() == [1000.0]
    assert scheduler.sigmas.tolist() == [1.0, 0.0]


@pytest.mark.parametrize("num_inference_steps", [2, 4, 28])
def test_multi_step_shift_terminal_schedule_matches_diffusers(num_inference_steps: int) -> None:
    scheduler = FlowMatchEulerDiscreteSDEScheduler(**_CONFIG)
    reference = FlowMatchEulerDiscreteScheduler(**_CONFIG)

    set_scheduler_timesteps(scheduler, num_inference_steps, seq_len=4096, device="cpu")
    set_scheduler_timesteps(reference, num_inference_steps, seq_len=4096, device="cpu")

    assert torch.equal(scheduler.timesteps, reference.timesteps)
    assert torch.equal(scheduler.sigmas, reference.sigmas)
    assert scheduler.sigmas[-2].item() == pytest.approx(0.02)


def test_non_finite_timesteps_raise_before_rollout() -> None:
    """The unpatched Diffusers stretch divides by zero for one step; the helper rejects it."""
    reference = FlowMatchEulerDiscreteScheduler(**_CONFIG)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="non-finite timesteps"):
            set_scheduler_timesteps(reference, 1, seq_len=4096, device="cpu")
