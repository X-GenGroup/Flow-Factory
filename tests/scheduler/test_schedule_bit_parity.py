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

"""Freeze touched schedule-coordinate bits from main before validation refactoring."""

from __future__ import annotations

from typing import Any, Type

import pytest
import torch

from flow_factory.models.ltx2._common import build_ltx2_full_component_schedule
from flow_factory.models.minimax_h3._common import build_training_component_times
from flow_factory.scheduler.flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSDEScheduler,
    _stable_mean_except_batch,
)
from flow_factory.scheduler.minimax_h3 import MiniMaxH3SDEScheduler
from flow_factory.scheduler.unipc_multistep import UniPCMultistepSDEScheduler
from flow_factory.utils.noise_schedule import flow_match_sigma

_MAIN_BASE_COMMIT = "8560d46649cc2927963fd7e5cdbcc33bd067171b"
_MAIN_TIMESTEP_MAX = 1000.0
_MAIN_COMPONENTS = ("video", "audio")
_MAIN_H3_SHIFTS = {"video": 12.0, "audio": 3.0}


def test_transition_log_prob_mean_uses_stable_fp64_accumulation() -> None:
    values = torch.linspace(-3.0, 5.0, 2 * 257, dtype=torch.float32).reshape(2, 257)
    values.requires_grad_()

    actual = _stable_mean_except_batch(values)
    expected = values.detach().double().mean(dim=1).float()

    assert actual.dtype is torch.float32
    assert torch.equal(actual.detach(), expected)
    actual.sum().backward()
    assert values.grad is not None
    torch.testing.assert_close(
        values.grad,
        torch.full_like(values, 1.0 / values.shape[1]),
        rtol=0,
        atol=torch.finfo(torch.float32).eps,
    )


def _assert_bit_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    identifier: str,
) -> None:
    """Require identical tensor metadata and underlying value bits."""
    assert actual.dtype == expected.dtype, (
        f"{identifier} dtype changed from the main baseline {_MAIN_BASE_COMMIT}: "
        f"expected {expected.dtype}, received {actual.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{identifier} shape changed from the main baseline {_MAIN_BASE_COMMIT}: "
        f"expected {tuple(expected.shape)}, received {tuple(actual.shape)}"
    )
    actual_bits = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bits = expected.detach().cpu().contiguous().view(torch.uint8)
    assert torch.equal(
        actual_bits, expected_bits
    ), f"{identifier} is not bit-exact with main baseline {_MAIN_BASE_COMMIT}"


def _main_flow_match_sigma(timesteps: torch.Tensor) -> torch.Tensor:
    """Reproduce main's pre-refactor flow-match conversion independently."""
    output_dtype = timesteps.dtype if timesteps.is_floating_point() else torch.get_default_dtype()
    return (timesteps.to(torch.float64) / _MAIN_TIMESTEP_MAX).clamp(0.0, 1.0).to(output_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_flow_match_conversion_matches_main_bits(dtype: torch.dtype) -> None:
    timesteps = torch.linspace(0.0, _MAIN_TIMESTEP_MAX, 4097, dtype=dtype)

    _assert_bit_exact(
        flow_match_sigma(timesteps),
        _main_flow_match_sigma(timesteps),
        identifier=f"flow_match_sigma/{dtype}",
    )


def test_integer_flow_match_conversion_matches_main_bits() -> None:
    timesteps = torch.arange(0, 1001, dtype=torch.int64)

    _assert_bit_exact(
        flow_match_sigma(timesteps),
        _main_flow_match_sigma(timesteps),
        identifier="flow_match_sigma/torch.int64",
    )


@pytest.mark.parametrize("num_steps", [1, 2, 4, 10, 50, 1000])
def test_h3_component_schedules_and_training_mapping_match_main_bits(
    num_steps: int,
) -> None:
    schedulers = {
        component: MiniMaxH3SDEScheduler(shift=_MAIN_H3_SHIFTS[component])
        for component in _MAIN_COMPONENTS
    }
    for component in _MAIN_COMPONENTS:
        scheduler = schedulers[component]
        scheduler.set_timesteps(num_steps, device="cpu")
        shift = _MAIN_H3_SHIFTS[component]
        base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
        expected_sigmas = shift * base / (1 + (shift - 1) * base)
        _assert_bit_exact(
            scheduler.sigmas,
            expected_sigmas,
            identifier=f"H3/{num_steps}/{component}/sigmas",
        )
        _assert_bit_exact(
            scheduler.timesteps,
            expected_sigmas[:-1] * _MAIN_TIMESTEP_MAX,
            identifier=f"H3/{num_steps}/{component}/timesteps",
        )
        _assert_bit_exact(
            scheduler.model_timesteps,
            1 - expected_sigmas[:-1],
            identifier=f"H3/{num_steps}/{component}/model_timesteps",
        )

    primary_timesteps = schedulers["video"].timesteps
    actual = build_training_component_times(
        primary_timesteps,
        video_shift=_MAIN_H3_SHIFTS["video"],
        audio_shift=_MAIN_H3_SHIFTS["audio"],
    )
    expected_video_sigma = primary_timesteps / _MAIN_TIMESTEP_MAX
    video_shift = _MAIN_H3_SHIFTS["video"]
    audio_shift = _MAIN_H3_SHIFTS["audio"]
    expected_base_quantile = expected_video_sigma / (
        video_shift - (video_shift - 1) * expected_video_sigma
    )
    expected_audio_sigma = (
        audio_shift * expected_base_quantile / (1 + (audio_shift - 1) * expected_base_quantile)
    )
    expected = {
        "timestep": {
            "video": primary_timesteps,
            "audio": expected_audio_sigma * _MAIN_TIMESTEP_MAX,
        },
        "sigma": {
            "video": expected_video_sigma,
            "audio": expected_audio_sigma,
        },
    }
    for field, components in expected.items():
        actual_components = getattr(actual, field)
        assert tuple(actual_components) == _MAIN_COMPONENTS
        for component, values in components.items():
            _assert_bit_exact(
                actual_components[component],
                values,
                identifier=f"H3/{num_steps}/mapped/{field}/{component}",
            )


@pytest.mark.parametrize(
    "scheduler_type",
    [FlowMatchEulerDiscreteSDEScheduler, UniPCMultistepSDEScheduler],
)
@pytest.mark.parametrize("num_steps", [1, 4, 10, 50, 1000])
def test_generic_scheduler_explicit_grid_conversion_matches_main_bits(
    scheduler_type: Type[Any],
    num_steps: int,
) -> None:
    """Freeze only the explicit replay conversion changed by this refactor."""
    scheduler = scheduler_type()
    scheduler.set_timesteps(num_steps, device="cpu")

    _assert_bit_exact(
        flow_match_sigma(scheduler.timesteps),
        scheduler.timesteps / _MAIN_TIMESTEP_MAX,
        identifier=f"{scheduler_type.__name__}/{num_steps}/explicit-replay-sigmas",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ltx2_full_component_schedule_matches_main_bits(dtype: torch.dtype) -> None:
    rollout_timesteps = torch.tensor(
        [990.4219970703125, 899.9999389648438, 500.0, 0.125],
        dtype=dtype,
    )
    expected_timesteps = torch.cat([rollout_timesteps, torch.zeros(1, dtype=dtype)])
    expected_sigmas = _main_flow_match_sigma(expected_timesteps)

    schedule = build_ltx2_full_component_schedule(object(), rollout_timesteps)

    assert tuple(schedule) == _MAIN_COMPONENTS
    for component in _MAIN_COMPONENTS:
        timesteps, sigmas = schedule[component]
        _assert_bit_exact(
            timesteps,
            expected_timesteps,
            identifier=f"LTX2/{dtype}/{component}/timesteps",
        )
        _assert_bit_exact(
            sigmas,
            expected_sigmas,
            identifier=f"LTX2/{dtype}/{component}/sigmas",
        )
