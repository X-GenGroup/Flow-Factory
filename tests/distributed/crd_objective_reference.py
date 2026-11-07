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

"""Independent CRD scalar objectives and analytic local velocity derivatives."""

import math

import torch


def rounded(value, dtype):
    return torch.as_tensor(value, dtype=torch.float64).to(dtype).double()


def _geometry(row):
    assert row["new"].keys() == row["old"].keys() == row["target"].keys()
    masks = {
        name: row["masks"].get(name, torch.ones_like(value, dtype=torch.bool)).expand_as(value)
        for name, value in row["new"].items()
    }
    counts = {name: mask.flatten(1).sum(1).double() for name, mask in masks.items()}
    total = sum(counts.values())
    assert bool((total > 0).all())
    return masks, counts, total


def implicit_reference(row):
    """Prove raw-to-public implicit rewards using an absolute cancellation budget."""
    masks, counts, total = _geometry(row)
    sums, magnitudes, errors, normalizers = [], [], {}, {}
    for name, target in row["target"].items():
        terms = {}
        for role in ("new", "old"):
            value = row[role][name]
            dtype = torch.promote_types(value.dtype, target.dtype)
            error = rounded(value.double() - target.double(), dtype)
            squared = rounded(error.square(), dtype)
            normalizer = torch.ones(len(total), dtype=torch.float64)
            if row["adaptive_logp"]:
                normalizer = (
                    ((value.double() - target.double()).abs() * masks[name]).flatten(1).sum(1)
                    / counts[name]
                ).clamp_min(1e-5)
                squared = squared / normalizer.reshape((-1,) + (1,) * (value.ndim - 1))
            terms[role] = squared
            if role == "new":
                errors[name], normalizers[name] = error, normalizer
        dtype = torch.float64 if row["adaptive_logp"] else dtype
        signed = rounded(terms["old"] - terms["new"], dtype)
        sums.append((signed * masks[name]).flatten(1).sum(1))
        magnitudes.append(
            ((terms["old"].abs() + terms["new"].abs()) * masks[name]).flatten(1).sum(1)
        )
    expected = rounded(sum(sums) / total, dtype)
    operations = 8 + sum(value[0].numel() for value in row["new"].values())
    unit = torch.finfo(dtype).eps / 2
    assert operations * unit < 1
    bound = operations * unit / (1 - operations * unit) * sum(magnitudes) / total
    bound += torch.finfo(dtype).tiny
    if "implicit_reward" in row:
        actual = row["implicit_reward"]
        assert actual.dtype == dtype and actual.shape == expected.shape
        assert bool(((actual.double() - expected).abs() <= bound).all()), (
            "implicit reward",
            (actual.double() - expected).abs().max(),
            bound.max(),
        )
        expected = actual.double()
    return expected, errors, normalizers, masks, total, dtype


def _weights(advantages, temperature):
    values = [float(value) for value in advantages]
    if temperature < 0:
        return [torch.full_like(advantages, 1 / len(values))]
    pools = []
    for direction in (1, -1):
        indices = (
            [i for i, value in enumerate(values) if direction * value > 0]
            if temperature == 0
            else list(range(len(values)))
        )
        if not indices:
            pools.append(torch.full_like(advantages, 1 / len(values)))
            continue
        logits = [
            values[i] if temperature == 0 else direction * values[i] / temperature for i in indices
        ]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        result = torch.zeros_like(advantages)
        for index, weight in zip(indices, weights, strict=True):
            result[index] = weight / math.fsum(weights)
        pools.append(result)
    return pools


def objective_reference(row):
    """Derive detached-center/MAD slopes without production helpers or autograd."""
    local, errors, normalizers, masks, total, reward_dtype = implicit_reference(row)
    logical = row.get("logical_rows", [row])
    global_rewards = torch.cat([implicit_reference(peer)[0] for peer in logical])
    global_advantages = torch.cat([peer["advantage"].double() for peer in logical])
    advantage = row["advantage"].double()
    adv_dtype = row["advantage"].dtype
    beta, batch = row["crd_beta"], len(advantage)
    policy_dtype = torch.promote_types(reward_dtype, adv_dtype)
    policy, upstream = torch.zeros((), dtype=torch.float64), torch.zeros_like(advantage)
    directions = _weights(global_advantages, row["weight_temp"])
    upstream_branches = []
    for weights in directions:
        weights = rounded(weights, adv_dtype)
        adv_center = rounded(rounded(global_advantages * weights, adv_dtype).sum(), adv_dtype)
        reward_center = rounded(rounded(global_rewards * weights, policy_dtype).sum(), policy_dtype)
        desired = rounded(advantage - adv_center, adv_dtype)
        centered = rounded(local - reward_center, policy_dtype)
        logits = rounded(beta * centered, policy_dtype)
        if row["crd_loss_type"] == "mse":
            delta = rounded(logits - desired, policy_dtype)
            loss = rounded(rounded(delta.square(), policy_dtype).mean(), policy_dtype)
            slope = 2 * beta * delta / batch
        else:
            target = rounded(desired.sigmoid(), adv_dtype)
            loss = rounded(
                (torch.nn.functional.softplus(logits) - target * logits).mean(), policy_dtype
            )
            slope = beta * (logits.sigmoid() - target) / batch
        policy += loss / len(directions)
        upstream += slope / len(directions)
        upstream_branches.append(slope / len(directions))
    rescale = row["adv_clip_max"] / max(beta, 1e-8)
    policy = rounded(policy * rescale, policy_dtype)
    upstream *= rescale * row["cp_size"] / row["gas"]
    branch_values = [
        rounded(branch * rescale * row["cp_size"] / row["gas"], reward_dtype)
        for branch in upstream_branches
    ]
    cast_upstream = branch_values[0]
    for branch in branch_values[1:]:
        cast_upstream = rounded(cast_upstream + branch, reward_dtype)
    kl_dtype = row["kl_dtype"]
    if row["reward_adaptive_kl"]:
        minimum = 1e-4 / max(row["kl_beta"], 1e-8)
        kl_weights = rounded(minimum + rounded(advantage * (1 - minimum), adv_dtype), adv_dtype)
        kl_coefficients = kl_weights * row["kl_beta"]
    else:
        kl_coefficients = torch.full_like(advantage, row["kl_beta"])
    kl_sums = torch.zeros_like(advantage)
    gradients, intervals, budgets, policy_gradients = {}, {}, {}, {}
    for name, value in row["new"].items():
        shape = (batch,) + (1,) * (value.ndim - 1)
        native = torch.promote_types(value.dtype, row["target"][name].dtype)
        # Division by the detached FP64 MAD casts its derivative back through
        # the native squared residual before the pow/subtraction edges.
        coefficient = rounded(-cast_upstream / total, reward_dtype) / normalizers[name]
        coefficient = rounded(coefficient, native)
        raw = (
            rounded(coefficient.reshape(shape) * rounded(2 * errors[name], native), native)
            * masks[name]
        )
        policy_gradient = rounded(raw, value.dtype)
        reference = row["reference"][name]
        difference_dtype = torch.promote_types(value.dtype, reference.dtype)
        difference = rounded(value.double() - reference.double(), difference_dtype)
        kl_sums += (rounded(difference.square(), kl_dtype) * masks[name]).flatten(1).sum(1)
        kl_scale = rounded(
            rounded(row["cp_size"] / row["gas"], kl_dtype) * row["kl_beta"], kl_dtype
        )
        kl_scale = rounded(kl_scale / batch, kl_dtype)
        if row["reward_adaptive_kl"]:
            kl_scale = rounded(kl_scale * kl_weights, kl_dtype)
        kl_scale = rounded(kl_scale / total, kl_dtype)
        kl_raw = (
            rounded(
                rounded(kl_scale, kl_dtype).reshape(shape) * rounded(2 * difference, kl_dtype),
                kl_dtype,
            )
            * masks[name]
        )
        kl_gradient = rounded(kl_raw, value.dtype)
        gradient = rounded(policy_gradient + kl_gradient, value.dtype)
        # Bound upstream floating reduction error before each native cast; add
        # branch magnitudes so cancellation cannot make the allowance vanish.
        precision = (
            torch.float32
            if native != torch.float64 or adv_dtype != torch.float64
            else torch.float64
        )
        unit = torch.finfo(precision).eps / 2
        operations = 256 + 8 * value[0].numel() + 16 * len(global_rewards)
        assert operations * unit < 1
        gamma = operations * unit / (1 - operations * unit)
        policy_bound = gamma * raw.abs() + torch.finfo(precision).tiny
        kl_bound = gamma * kl_raw.abs() + torch.finfo(precision).tiny
        low = rounded(
            rounded(raw - policy_bound, value.dtype) + rounded(kl_raw - kl_bound, value.dtype),
            value.dtype,
        )
        high = rounded(
            rounded(raw + policy_bound, value.dtype) + rounded(kl_raw + kl_bound, value.dtype),
            value.dtype,
        )
        gradients[name], intervals[name] = gradient, (low, high)
        budgets[name] = (
            gamma * (policy_gradient.abs() + kl_gradient.abs()) + torch.finfo(precision).tiny
        )
        policy_gradients[name] = policy_gradient
    kl = rounded(kl_sums / total, kl_dtype)
    if "kl_div" in row:
        unit = torch.finfo(kl_dtype).eps / 2
        operations = 8 + sum(value[0].numel() for value in row["new"].values())
        bound = operations * unit / (1 - operations * unit) * kl.abs() + torch.finfo(kl_dtype).tiny
        assert bool(((row["kl_div"].double() - kl).abs() <= bound).all())
        kl = row["kl_div"].double()
    kl_loss = (kl_coefficients * kl).mean()
    return dict(
        loss=policy + kl_loss,
        policy=policy,
        kl_loss=kl_loss,
        gradients=gradients,
        intervals=intervals,
        budgets=budgets,
        policy_gradients=policy_gradients,
        upstream=upstream,
        implicit_reward=local,
    )


def assert_objective(row):
    expected = objective_reference(row)
    torch.testing.assert_close(row["loss"].double(), expected["loss"], rtol=3e-5, atol=1e-9)
    for name, target in expected["gradients"].items():
        actual = row["gradients"][name].double()
        low, high = expected["intervals"][name]
        budget = expected["budgets"][name]

        def matches(value):
            return bool(((value >= low - budget) & (value <= high + budget)).all())

        assert matches(actual), (name, float((actual - target).abs().max()), float(budget.max()))
        if bool((target.abs() > 4 * budget).any()):
            for factor in (0.0, 0.5, 2.0, -1.0):
                assert not matches(actual * factor), (name, factor)
    return expected
