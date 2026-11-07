# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Independent TDM-R1 scalar preference and raw-velocity derivatives."""

import math
from collections import Counter, defaultdict

import torch
from dgpo_objective_reference import _dsm, _round


def surrogate_reference(row):
    """Evaluate legacy detached weights, clips and anchor without autograd.

    The unchanged DGPO test reference supplies only the raw masked DSM reduction
    proof. TDM-R1's weights, time factor, reference anchor and both derivative
    masks are derived here independently of every production objective helper.
    """
    grouped, counts = defaultdict(list), Counter()
    for peer in row.get("logical_rows", [row]):
        live, _, dtype = _dsm(peer, "new")
        reference, _, _ = _dsm(peer, "reference")
        adv = peer["advantage"].double()
        term = _round(
            _round(
                _round(adv * row["preference_beta"], dtype) * _round(live - reference, dtype), dtype
            )
            / row["group_size"],
            dtype,
        )
        for key, value in zip(peer["group_keys"], term, strict=True):
            grouped[tuple(key)].append(float(value))
            counts[tuple(key)] += 1
    assert all(count == row["group_size"] for count in counts.values())
    sums = {key: math.fsum(values) for key, values in grouped.items()}
    live, count, dtype = _dsm(row, "new")
    reference, _, _ = _dsm(row, "reference")
    adv = row["advantage"].double()
    weights = torch.tensor(
        [1 / (1 + math.exp(-sums[tuple(key)])) for key in row["group_keys"]], dtype=dtype
    ).double()
    ratio = torch.ones_like(live)
    trust = torch.zeros_like(live, dtype=torch.bool)
    if row["clip_range"]:
        slow, _, _ = _dsm(row, "slow")
        ratio = _round(torch.exp(_round(slow - live, dtype)), dtype)
        trust = torch.where(adv > 0, ratio > 1 + row["clip_range"], ratio < 1 - row["clip_range"])
    delta = _round(live - reference, dtype)
    runaway = torch.where(adv > 0, live < reference, live > reference) & (
        _round(delta.square(), dtype) > row["reference_threshold"]
    )
    active = ~(trust | runaway)
    time_weight = (row["boundary_index"] + 1) / row["num_steps"] if row["time_weighting"] else 1.0
    coefficient = _round(weights * adv, dtype)
    preference = _round(_round(coefficient * live, dtype).mean(), dtype)
    penalty = _round(_round(delta.square(), dtype).mean(), dtype)
    scalar = _round(
        time_weight * _round(preference + _round(row["reference_beta"] * penalty, dtype), dtype),
        dtype,
    )
    batch = len(live)
    upstream = time_weight / batch * (coefficient * active + 2 * row["reference_beta"] * delta)
    upstream *= row["cp_size"] / row["gas"]
    gradients = {}
    for name, value in row["new"].items():
        shape = (batch,) + (1,) * (value.ndim - 1)
        target = row["target"][name]
        residual_dtype = torch.promote_types(target.dtype, value.dtype)
        residual = _round(value.double() - target.double(), residual_dtype)
        mask = row["masks"].get(name, torch.ones_like(value, dtype=torch.bool)).expand_as(value)
        gradients[name] = (
            _round(2 * residual * (upstream / count).reshape(shape), value.dtype) * mask
        )
    return {
        "loss": scalar,
        "upstream": upstream,
        "gradients": gradients,
        "trust": trust,
        "runaway": runaway,
        "weights": weights,
        "ratio": ratio,
        "time_weight": time_weight,
    }


def assert_surrogate(row, *, gradients=True):
    expected = surrogate_reference(row)
    torch.testing.assert_close(row["loss"].double(), expected["loss"], rtol=2e-5, atol=2e-6)
    if gradients:
        torch.testing.assert_close(
            row["upstream"].double(), expected["upstream"], rtol=3e-5, atol=2e-6
        )
        for name, value in row["gradients"].items():
            # The test reference rounds only the final raw derivative. Small
            # intermediate FP32 edge rounding is bounded by operation count.
            torch.testing.assert_close(
                value.double(), expected["gradients"][name], rtol=4e-5, atol=2e-6
            )
    return expected


def direction_reference(row):
    """Analytic clean-state reward derivative with frozen normalizer and target."""
    from tdm_objective_reference import _weights

    live = row["live"]
    weights = _weights(live, row.get("masks"))
    batch = next(iter(live.values())).shape[0]
    normalizer = sum(
        ((value.float() - row["reference"][name].float()).abs().double() * weights[name])
        .flatten(1)
        .sum(1)
        for name, value in live.items()
    ).clamp_min(1e-6)
    losses, gradients = {}, {}
    for name, value in live.items():
        target = value.float() + row["direction"][name].float()
        delta = (value.float() - target).double()
        if row["use_huber"]:
            squared = delta.square().float().double()
            c = torch.tensor(row["huber_c"], dtype=torch.float32).double()
            c_squared = torch.tensor(row["huber_c"] ** 2, dtype=torch.float32).double()
            root = (squared + c_squared).float().double().sqrt().float().double()
            losses[name], derivative = (root - c).float().double(), delta / root
        else:
            losses[name], derivative = 0.5 * delta.square(), delta
        gradients[name] = (
            derivative * weights[name] / normalizer.reshape(-1, *([1] * (value.ndim - 1))) / batch
        )
    loss = sum((losses[name] * weights[name]).flatten(1).sum(1) for name in live)
    return (loss / normalizer).mean(), gradients
