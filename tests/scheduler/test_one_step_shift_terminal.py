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
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from flow_factory.scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    UniPCMultistepSDEScheduler,
)
from flow_factory.scheduler.flow_match_euler_discrete import set_scheduler_timesteps

_FLOW_MATCH_CONFIG: Dict[str, Any] = {
    "shift_terminal": 0.02,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 0.9,
}
_UNIPC_CONFIG: Dict[str, Any] = {
    "use_flow_sigmas": True,
    "prediction_type": "flow_prediction",
    "flow_shift": 3.0,
    "shift_terminal": 0.02,
}
_SCHEDULERS: Tuple[Tuple[type, type, Dict[str, Any], Dict[str, Any]], ...] = (
    (
        FlowMatchEulerDiscreteSDEScheduler,
        FlowMatchEulerDiscreteScheduler,
        _FLOW_MATCH_CONFIG,
        {"mu": 0.7},
    ),
    (UniPCMultistepSDEScheduler, UniPCMultistepScheduler, _UNIPC_CONFIG, {}),
)


@pytest.mark.parametrize(("scheduler_cls", "_reference_cls", "config", "kwargs"), _SCHEDULERS)
def test_one_step_schedule_with_shift_terminal_steps_from_noise_to_data(
    scheduler_cls: type, _reference_cls: type, config: Dict[str, Any], kwargs: Dict[str, Any]
) -> None:
    scheduler = scheduler_cls(**config)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scheduler.set_timesteps(1, device="cpu", **kwargs)

    assert torch.isfinite(scheduler.timesteps).all()
    assert scheduler.timesteps.numel() == 1
    assert scheduler.sigmas.tolist()[-1] == 0.0


@pytest.mark.parametrize(("scheduler_cls", "reference_cls", "config", "kwargs"), _SCHEDULERS)
@pytest.mark.parametrize("num_inference_steps", [2, 4, 28])
def test_multi_step_shift_terminal_schedule_matches_diffusers(
    scheduler_cls: type,
    reference_cls: type,
    config: Dict[str, Any],
    kwargs: Dict[str, Any],
    num_inference_steps: int,
) -> None:
    scheduler = scheduler_cls(**config)
    reference = reference_cls(**config)

    scheduler.set_timesteps(num_inference_steps, device="cpu", **kwargs)
    reference.set_timesteps(num_inference_steps, device="cpu", **kwargs)

    assert torch.equal(scheduler.timesteps, reference.timesteps)
    assert torch.equal(scheduler.sigmas, reference.sigmas)
    assert scheduler.sigmas[-2].item() == pytest.approx(0.02)


def test_one_step_flow_match_schedule_through_the_shared_helper() -> None:
    scheduler = FlowMatchEulerDiscreteSDEScheduler(**_FLOW_MATCH_CONFIG)

    timesteps = set_scheduler_timesteps(scheduler, 1, seq_len=4096, device="cpu")

    assert timesteps.tolist() == [1000.0]
    assert scheduler.sigmas.tolist() == [1.0, 0.0]


@pytest.mark.parametrize(("scheduler_cls", "_reference_cls", "config", "kwargs"), _SCHEDULERS)
def test_sde_schedulers_reject_non_finite_schedules(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_cls: type,
    _reference_cls: type,
    config: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> None:
    """Every caller of set_timesteps, not only the shared helper, gets the guard."""
    monkeypatch.setattr(
        scheduler_cls,
        "stretch_shift_to_terminal",
        lambda self, t: np.full_like(t, np.nan),
    )
    scheduler = scheduler_cls(**config)

    with pytest.raises(ValueError, match="non-finite"):
        scheduler.set_timesteps(4, device="cpu", **kwargs)


def test_shared_helper_rejects_non_finite_timesteps_from_any_scheduler() -> None:
    """The unpatched Diffusers stretch divides by zero for one step; the helper rejects it."""
    reference = FlowMatchEulerDiscreteScheduler(**_FLOW_MATCH_CONFIG)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="non-finite timesteps"):
            set_scheduler_timesteps(reference, 1, seq_len=4096, device="cpu")
