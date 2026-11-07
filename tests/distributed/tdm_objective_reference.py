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

"""Independent public-operand TDM objectives and analytic latent derivatives."""

import torch


def _weights(values, masks):
    masks = masks or {}
    active = {
        name: torch.broadcast_to(
            masks.get(name, torch.ones_like(value, dtype=torch.bool)), value.shape
        ).double()
        for name, value in values.items()
    }
    count = sum(mask.flatten(1).sum(1) for mask in active.values())
    assert torch.all(count > 0)
    return {
        name: mask / count.reshape(-1, *([1] * (mask.ndim - 1))) for name, mask in active.items()
    }


def objective_reference(row):
    """Derive scalar and derivative without production helpers or autograd.

    Cast/subtraction points are the public FP32 operands used by the objective.
    Reductions and derivatives after those points are evaluated in FP64. The
    caller compares against bounded FP32 reduction rounding, not exact sums.
    """
    live = row["live"]
    weights = _weights(live, row.get("masks"))
    batch = next(iter(live.values())).shape[0]
    if row["kind"] == "fake":
        sigma = row["sigma"].float().reshape(batch, -1)[:, 0].double()
        importance = row["importance"].float().double()
        snr = (1 - sigma).square() / sigma.square().clamp_min(1e-6)
        scale = snr.clamp_max(row["snr_gamma"]) * importance
        differences = {
            name: (value.float() - row["target"][name].float()).double()
            for name, value in live.items()
        }
        loss = sum(
            (delta.square() * weights[name]).flatten(1).sum(1)
            for name, delta in differences.items()
        )
        gradients = {
            name: 2 * delta * weights[name] * scale.reshape(-1, *([1] * (delta.ndim - 1))) / batch
            for name, delta in differences.items()
        }
        return (loss * scale).mean(), gradients
    assert row["kind"] == "generator"
    normalizer = sum(
        ((value.float() - row["real"][name].float()).abs().double() * weights[name])
        .flatten(1)
        .sum(1)
        for name, value in live.items()
    ).clamp_min(1e-6)
    values, gradients = {}, {}
    for name, value in live.items():
        correction = row["real"][name].float() - row["fake"][name].float()
        target = value.float() + correction
        delta = (value.float() - target).double()
        if row["use_huber"]:
            # Preserve public FP32 square/add/sqrt/subtract endpoints. Computing
            # the residual entirely in FP64 erases real cancellation near zero.
            squared = delta.square().float().double()
            c = torch.tensor(row["huber_c"], dtype=torch.float32).double()
            c_squared = torch.tensor(row["huber_c"] ** 2, dtype=torch.float32).double()
            radicand = (squared + c_squared).float().double()
            root = radicand.sqrt().float().double()
            values[name] = (root - c).float().double()
            derivative = delta / root
        else:
            values[name] = 0.5 * delta.square()
            derivative = delta
        gradients[name] = (
            derivative * weights[name] / normalizer.reshape(-1, *([1] * (value.ndim - 1))) / batch
        )
    loss = sum((values[name] * weights[name]).flatten(1).sum(1) for name in live)
    return (loss / normalizer).mean(), gradients


def assert_objective(row, cp_size):
    expected, gradients = objective_reference(row)
    torch.testing.assert_close(row["loss"].double(), expected, rtol=8e-6, atol=1e-7)
    scale = cp_size / row["gas"]
    for name, gradient in gradients.items():
        expected_gradient = (gradient * scale).to(row["live"][name].dtype).double()
        torch.testing.assert_close(
            row["gradients"][name].double(), expected_gradient, rtol=8e-6, atol=1e-8
        )


def assert_projection(row):
    """Check Wan's noise-ward projection at explicit public FP32 rounding points."""
    for name, state in row["state"].items():
        velocity = row["velocity"][name]
        sigma = row["sigma"][name].float().reshape(-1, *([1] * (state.ndim - 1)))
        product = (sigma.double() * velocity.double()).float().double()
        expected = (state.double() - product).float()
        torch.testing.assert_close(row["clean"][name], expected, rtol=0, atol=0)


def assert_conditional_renoise(row):
    """Use cloned-RNG fresh draws, never reconstruct fresh noise from the result."""
    mixed, fresh = {}, row["fresh"]
    for name, clean in row["clean"].items():
        shape = (-1, *([1] * (clean.ndim - 1)))
        mid, target = row["mid"][name].double().reshape(shape), row["target_sigma"][
            name
        ].double().reshape(shape)
        coefficient = (1 - target) / (1 - mid) * mid
        beta = (target.square() - coefficient.square()).sqrt()
        expected = (
            coefficient * row["implied"][name].double() + beta * fresh[name].double()
        ) / target
        actual = row["noise"][name].double()
        # Absolute error scales with the operands before cancellation.
        magnitude = (
            coefficient.abs() * row["implied"][name].double().abs()
            + beta * fresh[name].double().abs()
        ) / target
        assert torch.all(
            (actual - expected).abs() <= 16 * torch.finfo(torch.float32).eps * magnitude + 1e-8
        )
        mixed[name] = actual
        expected_state = (1 - target) * clean.double() + target * actual
        scale = (1 - target).abs() * clean.double().abs() + target.abs() * actual.abs()
        assert torch.all(
            (row["state"][name].double() - expected_state).abs()
            <= 4 * torch.finfo(torch.float32).eps * scale + 1e-8
        )
        torch.testing.assert_close(
            row["velocity_target"][name], (actual - clean.double()).float(), rtol=0, atol=0
        )
    weights = _weights(row["clean"], row.get("masks"))
    log_ratio = sum(
        ((fresh[name].double().square() - mixed[name].square()) * weights[name]).flatten(1).sum(1)
        * 0.5
        for name in mixed
    )
    clip = torch.tensor(row["importance_clip"], dtype=torch.float64).log()
    expected_importance = log_ratio.clamp(-clip, clip).exp()
    torch.testing.assert_close(
        row["importance"].double(), expected_importance, rtol=8e-6, atol=1e-7
    )
