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


"""Independent DGPO group objective and analytic full-output derivatives."""

import math
from collections import Counter, defaultdict

import torch


def _round(value, dtype):
    return torch.as_tensor(value, dtype=torch.float64).to(dtype).double()


def _dsm(row, prediction):
    """Independently validate the public DSM operand before downstream algebra.

    Raw residual/subtraction/square casts are specified independently. The
    production CPU/CUDA reduction tree may round positive squared terms before
    division, so validate its endpoint using gamma_n * absolute_sum/count. Never
    apply a delta-relative tolerance to two nearly cancelling DSM operands.
    """
    total, count, terms = None, None, 0
    for name, target in row["target"].items():
        value = row[prediction][name]
        dtype = torch.promote_types(value.dtype, target.dtype)
        residual = _round(target.double() - value.double(), dtype)
        mask = row["masks"].get(name, torch.ones_like(target, dtype=torch.bool))
        mask = mask.expand_as(target)
        contribution = (_round(residual.square(), dtype) * mask).flatten(1).sum(1)
        active = mask.flatten(1).sum(1)
        terms += value[0].numel()
        total = contribution if total is None else total + contribution
        count = active if count is None else count + active
    assert bool((count > 0).all())
    expected = _round(total / count, dtype)
    if "dsm_endpoints" not in row:
        return expected, count, dtype
    actual = row["dsm_endpoints"][prediction]
    assert actual.dtype == dtype and actual.shape == expected.shape
    # N positive additions, component joins, division and rounding of this
    # FP64 reference. This bound also covers the all-zero/masked case.
    operations = terms + len(row["target"]) + 2
    unit = torch.finfo(dtype).eps / 2
    assert operations * unit < 1
    gamma = operations * unit / (1 - operations * unit)
    bound = gamma * (total / count).abs() + torch.finfo(dtype).tiny
    error = (actual.double() - expected).abs()
    assert bool((error <= bound).all()), (prediction, float(error.max()), float(bound.max()))
    # Subsequent expected operations consume this separately proved public
    # rounded operand. Observed preferences, weights or gradients are never used.
    return actual.double(), count, dtype


def local_preference(row):
    """Compute each member's independent group contribution from raw velocities."""
    dsm, _, dtype = _dsm(row, "new")
    reference, _, _ = _dsm(row, "old" if row["use_ema_ref"] else "reference")
    advantage = row["advantage"].double()
    preference_dtype = torch.promote_types(dtype, row["advantage"].dtype)
    contribution = _round(
        _round(
            _round(advantage * row["dpo_beta"], row["advantage"].dtype)
            * _round(dsm - reference, dtype),
            preference_dtype,
        )
        / row["group_size"],
        preference_dtype,
    )
    return contribution


def assert_local_preference(row, actual, group_keys):
    """Prove every dense slot, including exact zeros for locally absent groups."""
    assert actual.numel() == len(group_keys)
    expected = torch.zeros(len(group_keys), dtype=torch.float64)
    occupied = set()
    for index, key, value in zip(
        row["local_group_indices"], row["group_keys"], local_preference(row), strict=True
    ):
        index = int(index)
        assert 0 <= index < len(group_keys) and tuple(key) == tuple(group_keys[index])
        expected[index] += value
        occupied.add(index)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)
    assert all(float(actual[index]) == 0 for index in set(range(len(group_keys))) - occupied)


def objective_reference(row):
    """No production loss/reducer/autograd helper participates in this calculation."""
    logical = row.get("logical_rows", [row])
    grouped, members = defaultdict(list), Counter()
    for peer in logical:
        contribution = local_preference(peer)
        for identity, value in zip(peer["group_keys"], contribution, strict=True):
            grouped[tuple(identity)].append(float(value))
            members[tuple(identity)] += 1
    assert all(count == row["group_size"] for count in members.values())
    global_sums = {key: math.fsum(values) for key, values in grouped.items()}
    dsm, count, dtype = _dsm(row, "new")
    preference_dtype = torch.promote_types(dtype, row["advantage"].dtype)
    weights = torch.tensor(
        [1 / (1 + math.exp(-global_sums[tuple(key)])) for key in row["group_keys"]],
        dtype=preference_dtype,
    ).double()
    adv = row["advantage"].double()
    if row["clip_dsm"] or row["clip_kl"]:
        old, _, _ = _dsm(row, "old")
        ratio = _round(torch.exp(_round(-dsm + old, dtype)), dtype)
        low, high = row["clip_range"]
        clipped = torch.where(adv > 0, ratio > 1 + high, ratio < 1 + low)
    else:
        clipped = torch.zeros_like(adv, dtype=torch.bool)
    batch = adv.numel()
    coefficient = _round(weights * adv, preference_dtype)
    policy = _round(_round(coefficient * dsm, preference_dtype).mean(), preference_dtype)
    kl_sums = torch.zeros(batch, dtype=torch.float64)
    gradients, intervals, budgets = {}, {}, {}
    normalization = row["cp_size"] / row["gas"]
    for name, value in row["new"].items():
        shape = (batch,) + (1,) * (value.ndim - 1)
        target = row["target"][name]
        residual = _round(value.double() - target.double(), dtype)
        mask = row["masks"].get(name, torch.ones_like(value, dtype=torch.bool)).expand_as(value)
        scale = _round(
            _round(normalization / batch, preference_dtype) * coefficient, preference_dtype
        )
        # weights*adv*dsm casts its derivative back to the DSM tensor before
        # the FP32 division/square chain, then the velocity receives its own cast.
        scale = _round(_round(scale, dtype) / count, dtype)
        if row["clip_dsm"]:
            scale = torch.where(clipped, 0.0, scale)
        raw = _round(scale.reshape(shape) * _round(2 * residual, dtype), dtype) * mask
        branches = [_round(raw, value.dtype)]
        # FP32 reduction error is bounded before the low-precision gradient cast.
        unit = torch.finfo(dtype).eps / 2
        operations = 128 + 8 * value[0].numel() + 8 * len(logical)
        assert operations * unit < 1
        gamma = operations * unit / (1 - operations * unit)
        bound = gamma * raw.abs() + torch.finfo(dtype).tiny
        lows = [_round(raw - bound, value.dtype)]
        highs = [_round(raw + bound, value.dtype)]
        if row["kl_beta"]:
            reference = row["reference"][name]
            difference_dtype = torch.promote_types(value.dtype, reference.dtype)
            difference = _round(value.double() - reference.double(), difference_dtype)
            kl_dtype = row["kl_dtype"]
            kl_sums += (_round(difference.square(), kl_dtype) * mask).flatten(1).sum(1)
            scalar = _round(_round(normalization, kl_dtype) * row["kl_beta"], kl_dtype)
            scalar = _round(_round(scalar / batch, kl_dtype) / count, kl_dtype)
            if row["clip_kl"]:
                scalar = torch.where(clipped, 0.0, scalar)
            raw_kl = (
                _round(scalar.reshape(shape) * _round(2 * difference, kl_dtype), kl_dtype) * mask
            )
            branches.append(_round(raw_kl, value.dtype))
            kl_bound = gamma * raw_kl.abs() + torch.finfo(kl_dtype).tiny
            lows.append(_round(raw_kl - kl_bound, value.dtype))
            highs.append(_round(raw_kl + kl_bound, value.dtype))
        gradient, low, high = branches[0], lows[0], highs[0]
        for term, lower, upper in zip(branches[1:], lows[1:], highs[1:], strict=True):
            gradient = _round(gradient + term, value.dtype)
            low, high = _round(low + lower, value.dtype), _round(high + upper, value.dtype)
        gradients[name], intervals[name] = gradient, (low, high)
        budgets[name] = gamma * sum(term.abs() for term in branches) + torch.finfo(dtype).tiny
    kl_loss = torch.tensor(0.0, dtype=torch.float64)
    if row["kl_beta"]:
        kl_loss = _round(
            _round(_round(kl_sums / count, row["kl_dtype"]).mean(), row["kl_dtype"])
            * row["kl_beta"],
            row["kl_dtype"],
        )
    return {
        "loss": policy + kl_loss,
        "gradients": gradients,
        "intervals": intervals,
        "budgets": budgets,
        "group_sums": global_sums,
        "weights": weights,
        "clipped": clipped,
    }


def assert_objective(row):
    expected = objective_reference(row)
    torch.testing.assert_close(row["loss"].double(), expected["loss"], rtol=2e-5, atol=1e-7)
    if "clipped" in row:
        assert torch.equal(row["clipped"], expected["clipped"])
    if "global_sums" in row:
        for key, value in zip(row["dense_group_keys"], row["global_sums"], strict=True):
            torch.testing.assert_close(
                value.double(),
                torch.tensor(expected["group_sums"][tuple(key)], dtype=torch.float64),
                rtol=2e-5,
                atol=1e-7,
            )
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
