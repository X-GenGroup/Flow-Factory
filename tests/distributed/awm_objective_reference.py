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


"""Independent AWM weighted-matching and clipped-ratio analytical derivatives."""

from itertools import permutations

import torch


def rounded(value, dtype):
    return torch.as_tensor(value, dtype=torch.float64).to(dtype).double()


def matching(values, row):
    """Derive each component's matching value and velocity derivative in FP64."""
    counts, masks, squared, errors = {}, {}, {}, {}
    for name, value in values.items():
        error = value.double() - row["target"][name].double()
        mask = row["masks"].get(name)
        mask = torch.ones_like(error) if mask is None else mask.double().expand_as(error)
        masks[name], errors[name] = mask, error
        counts[name] = mask.flatten(1).sum(1)
        squared[name] = (error.square() * mask).flatten(1).sum(1) / counts[name]
    total = sum(counts.values())
    output, slopes = torch.zeros_like(total), {}
    for name, mse in squared.items():
        sigma, scheme = row["sigma"][name].double().reshape(-1), row["weighting"]
        if scheme in ("Uniform", "t", "t**2"):
            weight = (
                torch.ones_like(sigma)
                if scheme == "Uniform"
                else sigma ** (1 if scheme == "t" else 2)
            )
            value, derivative = -mse * weight, weight
        else:
            power = 0.5 if scheme == "huber" else row["ghuber_power"]
            scale = sigma if scheme == "huber" else sigma / power
            value = -((mse + 1e-10).pow(power) - 1e-10**power) * scale
            derivative = power * (mse + 1e-10).pow(power - 1) * scale
        output += value * counts[name] / total
        shape = (len(total),) + (1,) * (errors[name].ndim - 1)
        slopes[name] = -2 * errors[name] * (derivative / total).reshape(shape) * masks[name]
    return rounded(output, torch.float32), slopes, total, masks


def objective_reference(row):
    """No trainer, adapter reduction or autograd is used for expected derivatives."""
    new, slopes, total, masks = matching(row["new"], row)
    old, _, _, _ = matching(row["old"], row)
    batch = len(total)
    adv_dtype = row["advantage"].dtype
    adv = row["advantage"].double().clamp(*row["adv_clip_range"])
    dtype = torch.promote_types(adv_dtype, torch.float32)
    ratio = rounded(rounded(new - old, torch.float32).exp(), torch.float32)
    lower, upper = (1 + bound for bound in row["clip_range"])
    clipped = ratio.clamp(lower, upper)
    direct = rounded(-adv * ratio, dtype)
    constrained = rounded(-adv * clipped, dtype)
    loss = rounded(torch.maximum(direct, constrained).mean(), dtype)
    active = ((adv > 0) & (ratio <= upper)) | ((adv < 0) & (ratio >= lower))
    scale = row["cp_size"] / row["gas"]
    # The ratio and exp edges are FP32 even if synthetic advantages are FP64.
    upstream = rounded(rounded(-adv * scale / batch, dtype), torch.float32)
    upstream = rounded(upstream * ratio * active, torch.float32)
    branches, ranges, budgets, gradients = {}, {}, {}, {}
    for name, slope in slopes.items():
        shape = (batch,) + (1,) * (slope.ndim - 1)
        public_dtype = row["new"][name].dtype
        unit = torch.finfo(torch.float32).eps / 2
        operations = 128 + 8 * slope[0].numel()
        assert operations * unit < 1
        gamma = operations * unit / (1 - operations * unit)
        policy = upstream.reshape(shape) * slope
        bound = gamma * policy.abs() + torch.finfo(torch.float32).tiny
        values = [rounded(policy, public_dtype)]
        intervals = [(rounded(policy - bound, public_dtype), rounded(policy + bound, public_dtype))]
        for other, beta in (("reference", row["kl_beta"]), ("ema", row["ema_kl_beta"])):
            if not beta:
                continue
            diff_dtype = torch.promote_types(public_dtype, row[other][name].dtype)
            diff = rounded(row["new"][name].double() - row[other][name].double(), diff_dtype)
            kl_dtype = row["kl_dtypes"][other]
            coefficient = rounded(rounded(scale, kl_dtype) * beta, kl_dtype)
            coefficient = rounded(coefficient / batch, kl_dtype)
            coefficient = rounded(coefficient / total.reshape(shape), kl_dtype)
            raw = rounded(coefficient * rounded(2 * diff, kl_dtype), kl_dtype) * masks[name]
            values.append(rounded(raw, public_dtype))
            kl_bound = gamma * raw.abs() + torch.finfo(torch.float32).tiny
            intervals.append(
                (rounded(raw - kl_bound, public_dtype), rounded(raw + kl_bound, public_dtype))
            )
        candidates, limits = [], []
        for order in permutations(range(len(values))):
            value = values[order[0]]
            low, high = intervals[order[0]]
            for index in order[1:]:
                value = rounded(value + values[index], public_dtype)
                low = rounded(low + intervals[index][0], public_dtype)
                high = rounded(high + intervals[index][1], public_dtype)
            candidates.append(value)
            limits.append((low, high))
        gradients[name], branches[name], ranges[name] = candidates[0], candidates, limits
        budgets[name] = (
            gamma * sum(value.abs() for value in values) + torch.finfo(torch.float32).tiny
        )
    for other, beta in (("reference", row["kl_beta"]), ("ema", row["ema_kl_beta"])):
        if not beta:
            continue
        kl_dtype = row["kl_dtypes"][other]
        sums = []
        for name, value in row["new"].items():
            difference = rounded(
                value.double() - row[other][name].double(),
                torch.promote_types(value.dtype, row[other][name].dtype),
            )
            sums.append(
                rounded(
                    (rounded(difference.square(), kl_dtype) * masks[name]).flatten(1).sum(1),
                    kl_dtype,
                )
            )
        combined = sums[0]
        for value in sums[1:]:
            combined = rounded(combined + value, kl_dtype)
        penalty = rounded(
            rounded(rounded(combined / total, kl_dtype).mean(), kl_dtype) * beta, kl_dtype
        )
        loss = rounded(loss + penalty, torch.promote_types(dtype, kl_dtype))
    return {
        "loss": loss,
        "gradients": gradients,
        "gradient_candidates": branches,
        "gradient_intervals": ranges,
        "budgets": budgets,
        "ratio": ratio,
        "active": active,
        "old_log_prob": old,
        "log_prob": new,
    }


def assert_objective(row):
    expected = objective_reference(row)
    for field in ("loss", "log_prob", "old_log_prob"):
        torch.testing.assert_close(row[field].double(), expected[field], rtol=2e-5, atol=1e-7)
    for name, limits in expected["gradient_intervals"].items():
        actual, budget = row["gradients"][name].double(), expected["budgets"][name]

        def matches(value):
            return any(
                bool(((value >= low - budget) & (value <= high + budget)).all())
                for low, high in limits
            )

        assert matches(actual), (
            name,
            float((actual - expected["gradients"][name]).abs().max()),
            float(budget.max()),
        )
        if bool((expected["gradients"][name].abs() > 4 * budget).any()):
            for factor in (0.0, 0.5, 2.0, -1.0):
                assert not matches(actual * factor), (name, factor)
    return expected
