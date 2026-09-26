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

"""Generation is the sole owner of effective static and dynamic flow shifts."""

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from flow_factory.scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    MiniMaxH3SDEScheduler,
    UniPCMultistepSDEScheduler,
)
from flow_factory.trainers.distillation.tdm_time_sampling import capture_generation_shift


@pytest.mark.parametrize("kind", ["static", "exponential", "linear"])
@pytest.mark.parametrize("family", ["euler", "unipc"])
def test_sampling_shift_matches_upstream_schedule_bitwise(kind, family):
    dynamic = kind != "static"
    config = dict(use_dynamic_shifting=dynamic, time_shift_type=kind if dynamic else "exponential")
    if family == "euler":
        cls, upstream = FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteScheduler
        config["shift"] = 3
    else:
        cls, upstream = UniPCMultistepSDEScheduler, UniPCMultistepScheduler
        config.update(use_flow_sigmas=True, flow_shift=3)
    scheduler, reference = cls(**config), upstream(**config)
    original_signature = inspect.signature(scheduler.set_timesteps)
    for mu in (0.4, 0.9):
        kwargs = dict(
            sigmas=np.array([1.0, 0.75, 0.5, 0.25]),
            num_inference_steps=4,
            mu=mu if dynamic else None,
        )
        with capture_generation_shift(scheduler) as shifts:
            assert inspect.signature(scheduler.set_timesteps) == original_signature
            scheduler.set_timesteps(**kwargs)
        assert "set_timesteps" not in vars(scheduler)
        reference.set_timesteps(**kwargs)
        assert torch.equal(scheduler.sigmas, reference.sigmas)
        assert torch.equal(scheduler.timesteps, reference.timesteps)
        gamma = math.exp(mu) if kind == "exponential" else mu if kind == "linear" else 3
        assert shifts == pytest.approx([gamma])
        raw = np.array([1.0, 0.75, 0.5, 0.25])
        expected = gamma * raw / (1 + (gamma - 1) * raw)
        np.testing.assert_allclose(scheduler.sigmas[:-1].numpy(), expected, atol=2e-6, rtol=0)


def test_h3_captures_static_shift_before_mutation():
    scheduler = MiniMaxH3SDEScheduler(shift=12)
    with capture_generation_shift(scheduler) as first:
        scheduler.set_timesteps(4)
        scheduler.set_shift(8)
    with capture_generation_shift(scheduler) as second:
        scheduler.set_timesteps(4)
    assert first == [12]
    assert second == [8]


@pytest.mark.parametrize(
    "config", [{"shift_terminal": 0.02}, {"invert_sigmas": True}, {"use_karras_sigmas": True}]
)
def test_non_shift_schedule_rejected_without_changing_generation(config):
    scheduler = FlowMatchEulerDiscreteSDEScheduler(**config)
    scheduler.set_timesteps(sigmas=[1, 0.75, 0.5, 0.25])
    with pytest.raises(ValueError, match="pure flow-shift schedule"):
        with capture_generation_shift(scheduler):
            scheduler.set_timesteps(sigmas=[1, 0.75, 0.5, 0.25])
    assert "set_timesteps" not in vars(scheduler)


def test_capture_preserves_positional_arguments_defaults_and_return_value():
    received = []
    returned = object()

    def original(steps, device=None, sigmas=None, mu=2.0, *, marker=None):
        received.append((steps, device, sigmas, mu, marker))
        return returned

    scheduler = SimpleNamespace(
        config={"use_dynamic_shifting": True, "time_shift_type": "linear"},
        set_timesteps=original,
    )
    grid, marker = [1, 0.5], object()
    with capture_generation_shift(scheduler) as shifts:
        assert inspect.signature(scheduler.set_timesteps) == inspect.signature(original)
        assert scheduler.set_timesteps(2, "cpu", grid, 2.0, marker=marker) is returned
        assert scheduler.set_timesteps(2, "cpu", grid, marker=marker) is returned
    assert received == [(2, "cpu", grid, 2.0, marker)] * 2
    assert shifts == [2.0]
    assert scheduler.set_timesteps is original


@pytest.mark.parametrize("stage", ["before", "setup", "after"])
def test_capture_restores_method_without_masking_generation_errors(stage):
    error = RuntimeError("generation failed")

    def original():
        if stage == "setup":
            raise error

    scheduler = SimpleNamespace(config={}, shift=3, set_timesteps=original)
    with pytest.raises(RuntimeError) as caught:
        with capture_generation_shift(scheduler):
            if stage == "before":
                raise error
            scheduler.set_timesteps()
            if stage == "after":
                raise error
    assert caught.value is error
    assert scheduler.set_timesteps is original


def test_capture_rejects_missing_schedule_call():
    scheduler = FlowMatchEulerDiscreteSDEScheduler(shift=3)
    with pytest.raises(ValueError, match="did not capture"):
        with capture_generation_shift(scheduler):
            pass
    assert "set_timesteps" not in vars(scheduler)


@pytest.mark.parametrize("dynamic", [False, True])
def test_capture_rejects_different_shifts_in_one_rollout(dynamic):
    scheduler = FlowMatchEulerDiscreteSDEScheduler(shift=3, use_dynamic_shifting=dynamic)
    with pytest.raises(ValueError, match="one effective shift per rollout"):
        with capture_generation_shift(scheduler):
            scheduler.set_timesteps(4, mu=math.log(3) if dynamic else None)
            scheduler.set_shift(7)
            scheduler.set_timesteps(4, mu=math.log(7) if dynamic else None)
    assert "set_timesteps" not in vars(scheduler)


def test_capture_restores_nested_instance_wrapper():
    scheduler = FlowMatchEulerDiscreteSDEScheduler(shift=3)
    other = FlowMatchEulerDiscreteSDEScheduler(shift=7)
    with capture_generation_shift(scheduler) as outer:
        wrapper = scheduler.set_timesteps
        with capture_generation_shift(scheduler) as inner:
            scheduler.set_timesteps(4)
            assert "set_timesteps" not in vars(other)
        assert scheduler.set_timesteps is wrapper
    assert outer == inner == [3]
    assert "set_timesteps" not in vars(scheduler)
