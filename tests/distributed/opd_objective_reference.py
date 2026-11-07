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

"""Independent OPD target, loss and analytic endpoint derivative checks."""

import pytest
import torch


def _round(value, dtype=torch.float32):
    return value.to(dtype).double()


def assert_ode_projection(row):
    """Reconstruct the scheduler's public ODE mean with explicit FP32 operands."""
    state, velocity = row["state"], row["velocity"]
    timestep, next_timestep = row["timestep"], row["next_timestep"]
    sigma = _round(timestep.double() / 1000)
    next_sigma = _round(next_timestep.double() / 1000)
    dt = _round(next_sigma - sigma).reshape((-1,) + (1,) * (state.ndim - 1))
    mean = _round(state.double() + _round(velocity.double() * dt))
    mean = mean.to(state.dtype).float()
    torch.testing.assert_close(row["target"], mean, rtol=0, atol=0)
    return dt


def assert_objective(row, cp_size):
    """Use independently validated public reductions, never observed derivatives."""
    student, teacher = row["student"], row["teacher"]
    assert student.shape == teacher.shape
    assert row["teacher_detached"]
    error = _round(student.double() - teacher.double())
    count = error[0].numel()
    squared = _round(error.square()).flatten(1).sum(1) / count
    absolute = error.abs().flatten(1).sum(1) / count
    endpoints = row["reductions"]
    expected_reductions = [absolute, squared] if row["self_normalize"] else [squared]
    assert len(endpoints) == len(expected_reductions)
    # Positive reduction rounding is bounded before its public endpoint feeds
    # the division. This is independent of the measured loss/gradient residual.
    unit = torch.finfo(torch.float32).eps / 2
    gamma = (count + 2) * unit / (1 - (count + 2) * unit)
    for actual, expected in zip(endpoints, expected_reductions, strict=True):
        assert bool(((actual.double() - expected).abs() <= gamma * expected.abs() + 1e-30).all())
    scale = (
        _round(endpoints[0].double() + 1e-8) if row["self_normalize"] else torch.ones_like(squared)
    )
    losses = _round(endpoints[-1].double() / scale)
    torch.testing.assert_close(row["loss"].double(), _round(losses.mean()), rtol=2e-6, atol=1e-8)
    root_gradient = _round(torch.tensor(cp_size / row["gas"], dtype=torch.float64))
    upstream = _round(root_gradient / error.shape[0])
    upstream = _round(upstream / scale)
    upstream = _round(upstream / count)
    derivative = _round(2 * error * upstream.reshape((-1,) + (1,) * (error.ndim - 1)))

    def check_gradient(actual, expected):
        torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-30)

    check_gradient(row["target_gradient"], derivative)
    assert torch.count_nonzero(derivative), "teacher/student gap must carry a gradient"
    for factor in (0.0, 0.5, 2.0, -1.0):
        with pytest.raises(AssertionError):
            check_gradient(row["target_gradient"] * factor, derivative)
    if "projection" in row:
        projection = row["projection"]
        dt = assert_ode_projection(projection)
        upstream = derivative.to(projection["target"].dtype).to(projection["state"].dtype).float()
        expected_velocity_gradient = (
            _round(upstream.double() * dt).to(projection["velocity"].dtype).double()
        )
        check_gradient(projection["velocity_gradient"], expected_velocity_gradient)
    return derivative
