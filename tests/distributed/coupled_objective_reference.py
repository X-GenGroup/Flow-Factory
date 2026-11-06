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


"""Independent FP64 objective and upstream oracle for dense native Flow-SDE replay.

The fixture supplies one complete latent component, frozen rollout quantities and
constant scheduler coefficients. This is not a generic scheduler or model oracle.
"""

import torch


def objective_reference(row):
    """Differentiate coupled losses without calling their ratio, mask or KL helpers."""
    mean, velocity, following = [row[key].double() for key in ("mean", "velocity", "following")]
    batch = mean.shape[0]
    elements = mean[0].numel()
    shape = (batch,) + (1,) * (mean.ndim - 1)
    std, dt, sigma = [row[key].double().reshape(shape) for key in ("std", "dt", "sigma")]
    scale = std * (-dt).sqrt()
    assert bool((scale > 0).all() and (sigma > 0).all())
    advantage = row["advantage"].double().clamp(*row["adv_clip_range"])
    delta = row["log_prob"].double() - row["old_log_prob"].double()
    plain_ratio = delta.exp()
    if row["algorithm"] == "grpo-guard":
        old_mean = row["old_mean"].double()
        mse = (mean - old_mean).square().flatten(1).mean(1)
        scalar_scale = scale.reshape(batch)
        ratio = (delta * scalar_scale + mse / (2 * scalar_scale)).exp()
        lower, upper = [1 + value for value in row["clip_range"]]
        clipped = (advantage > 0) & (ratio > upper) | (advantage < 0) & (ratio < lower)
        keep = ~clipped
        policy = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(lower, upper))
        derivative = (following - old_mean) / (elements * scale)
        # Bound the two terms before their cancellation, not the observed residual.
        magnitude = ((following - mean).abs() + (mean - old_mean).abs()) / (elements * scale)
        divergence = mse
    elif row["algorithm"] == "dppo":
        ratio = plain_ratio
        if row["kl_mask_type"] == "x-based":
            old_mean = row["old_mean"].double()
            divergence = ((mean - old_mean).square() / (2 * scale.square())).flatten(1).mean(1)
        else:
            divergence = (velocity - row["old_velocity"].double()).square().flatten(1).mean(1)
        removed = (divergence >= row["kl_mask_threshold"]) & (
            (ratio > 1) & (advantage > 0) | (ratio < 1) & (advantage < 0)
        )
        keep = ~removed
        policy = -advantage * ratio * keep
        derivative = (following - mean) / (elements * scale.square())
        magnitude = derivative.abs()
    else:
        assert row["algorithm"] == "grpo"
        ratio = plain_ratio
        lower, upper = [1 + value for value in row["clip_range"]]
        keep = ~((advantage > 0) & (ratio > upper) | (advantage < 0) & (ratio < lower))
        policy = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(lower, upper))
        derivative = (following - mean) / (elements * scale.square())
        magnitude = derivative.abs()
        divergence = torch.zeros_like(ratio)
    coefficient = (-advantage * ratio * keep / batch).reshape(shape)
    mean_gradient = coefficient * derivative
    mean_magnitude = coefficient.abs() * magnitude
    beta = row["kl_beta"]
    reference_loss = torch.tensor(0.0, dtype=torch.float64)
    velocity_reference = torch.zeros_like(velocity)
    if beta:
        if row["kl_type"] == "x-based":
            difference = mean - row["reference"].double()
            term = 2 * beta * difference / (batch * elements)
            mean_gradient = mean_gradient + term
            mean_magnitude = mean_magnitude + term.abs()
        else:
            difference = velocity - row["reference"].double()
            velocity_reference = 2 * beta * difference / (batch * elements)
        reference_loss = beta * difference.square().mean()
    # Flow-SDE mean is affine in velocity; coefficients depend only on stored time.
    jacobian = (1 + std.square() * (1 - sigma) / (2 * sigma)) * dt
    velocity_gradient = mean_gradient * jacobian + velocity_reference
    velocity_magnitude = mean_magnitude * jacobian.abs() + velocity_reference.abs()
    normalization = row["cp_size"] / row["gas"]
    unit = torch.finfo(torch.float32).eps / 2
    operations = 128 + 8 * elements
    assert operations * unit < 1
    gamma = operations * unit / (1 - operations * unit)
    return {
        "ratio": ratio,
        "plain_ratio": plain_ratio,
        "keep": keep,
        "divergence": divergence,
        "loss": policy.mean() + reference_loss,
        "mean_gradient": mean_gradient * normalization,
        "velocity_gradient": velocity_gradient * normalization,
        "mean_budget": gamma * mean_magnitude * normalization + torch.finfo(torch.float32).tiny,
        "velocity_budget": gamma * velocity_magnitude * normalization
        + torch.finfo(torch.float32).tiny,
    }


def assert_objective(row):
    """Check the observed loss/upstreams and reject zero/half/double/sign corruption."""
    reference = objective_reference(row)
    torch.testing.assert_close(row["loss"].double(), reference["loss"], rtol=2e-5, atol=1e-7)
    for field in ("mean", "velocity"):
        actual = row[field + "_gradient"].double()
        expected, budget = reference[field + "_gradient"], reference[field + "_budget"]
        assert actual.shape == expected.shape
        assert bool(((actual - expected).abs() <= budget).all()), (
            field,
            (actual - expected).abs().max(),
            budget.max(),
        )
        if bool((expected.abs() > 4 * budget).any()):
            for factor in (0.0, 0.5, 2.0, -1.0):
                assert not bool(((actual * factor - expected).abs() <= budget).all()), (
                    field,
                    factor,
                )
    return reference
