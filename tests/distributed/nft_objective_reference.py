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


"""Independent NFT objective with explicit public-operand precision boundaries."""

from itertools import permutations

import torch


def _rounded(value, dtype):
    """Evaluate a primitive in FP64, then apply its declared tensor rounding."""
    return torch.as_tensor(value, dtype=torch.float64).to(dtype).double()


def _quantized_affine_interval(value, bound, coefficient, mix_dtype, output_dtype):
    """Propagate an analytic input interval through a cast and affine derivative."""
    endpoints = [
        _rounded(_rounded(value + direction * bound, mix_dtype) * coefficient, output_dtype)
        for direction in (-1, 1)
    ]
    return torch.minimum(*endpoints), torch.maximum(*endpoints)


def objective_reference(row):
    """Derive detached-scale gradients without adapter, trainer or autograd helpers.

    Wan's velocity-only path returns BF16, while packed Flux returns FP32. The
    positive/negative affine combinations therefore round before clean-state
    projection. This oracle models those casts explicitly instead of comparing a
    different FP64 objective or increasing a loss tolerance.
    """
    names = tuple(row["clean"])
    batch = row["advantage"].numel()
    lower, upper = row["adv_clip_range"]
    adv_dtype = row["advantage"].dtype
    advantage = row["advantage"].double().clamp(lower, upper)
    reward = _rounded(
        _rounded(_rounded(advantage / max(lower, upper), adv_dtype) / 2, adv_dtype) + 0.5, adv_dtype
    ).clamp(0, 1)
    negative_weight = _rounded(1 - reward, adv_dtype)
    beta, normalization = row["nft_beta"], row["cp_size"] / row["gas"]
    sign = 1 if row["velocity_direction"] == "data" else -1
    counts, masks = {}, {}
    for name in names:
        clean = row["clean"][name].double()
        mask = row["masks"].get(name)
        masks[name] = torch.ones_like(clean) if mask is None else mask.double().expand_as(clean)
        counts[name] = masks[name].flatten(1).sum(1)
    total = sum(counts.values())
    assert bool((total > 0).all())
    positive, negative = [torch.zeros(batch, dtype=torch.float64) for _ in range(2)]
    gradients, budgets, scales, candidates, intervals = {}, {}, {}, {}, {}
    reference_sums = []
    for name in names:
        clean, state, new, old = [
            row[key][name].double() for key in ("clean", "state", "new", "old")
        ]
        new_dtype, old_dtype = row["new"][name].dtype, row["old"][name].dtype
        mix_dtype = torch.promote_types(new_dtype, old_dtype)
        compute_dtype = torch.promote_types(row["state"][name].dtype, mix_dtype)
        error_dtype = torch.promote_types(compute_dtype, row["clean"][name].dtype)
        shape = (batch,) + (1,) * (clean.ndim - 1)
        sigma = _rounded(row["sigma"][name], compute_dtype).reshape(shape)
        mask = masks[name]
        scaled_new = _rounded(beta * new, new_dtype)
        pos_v = _rounded(scaled_new + _rounded((1 - beta) * old, old_dtype), mix_dtype)
        neg_v = _rounded(_rounded((1 + beta) * old, old_dtype) - scaled_new, mix_dtype)
        predictions = [
            _rounded(state + _rounded(sign * sigma * velocity, compute_dtype), compute_dtype)
            for velocity in (pos_v, neg_v)
        ]
        errors = [_rounded(prediction - clean, error_dtype) for prediction in predictions]
        component_scales = [
            ((prediction - clean).abs() * mask)
            .flatten(1)
            .sum(1)
            .div(counts[name])
            .clamp_min(1e-5)
            .reshape(shape)
            for prediction in predictions
        ]
        scales[name] = tuple(component_scales)
        positive += (
            _rounded(errors[0].square(), error_dtype) / component_scales[0] * mask
        ).flatten(1).sum(1) / total
        negative += (
            _rounded(errors[1].square(), error_dtype) / component_scales[1] * mask
        ).flatten(1).sum(1) / total
        unit = torch.finfo(torch.float32).eps / 2
        operations = 128 + 8 * clean[0].numel()
        assert operations * unit < 1
        gamma = operations * unit / (1 - operations * unit)
        branches, branch_intervals = [], []
        for error, scale, weight, coefficient in zip(
            errors, component_scales, (reward, negative_weight), (beta, -beta)
        ):
            # Cast from the FP64 detached normalization, differentiate the square,
            # project through sigma, then cross the velocity dtype boundary and
            # differentiate the affine coefficient. Each cast is independent of
            # the production backward implementation.
            upstream = (
                normalization
                * upper
                / beta
                / batch
                * weight.reshape(shape)
                / total.reshape(shape)
                / scale
                * mask
            )
            squared_gradient = _rounded(upstream, error_dtype)
            projected_gradient = _rounded(
                squared_gradient * _rounded(2 * error, error_dtype), error_dtype
            )
            velocity_gradient = _rounded(projected_gradient * sign * sigma, compute_dtype)
            affine_gradient = _rounded(velocity_gradient, mix_dtype)
            branches.append(_rounded(affine_gradient * coefficient, new_dtype))
            # An FP32 reduction/division error can cross a BF16 rounding midpoint.
            # Bound that arithmetic before the lossy cast, then propagate both
            # endpoints monotonically. A post-cast FP32 tolerance misses this case.
            bound = gamma * velocity_gradient.abs() + torch.finfo(torch.float32).tiny
            branch_intervals.append(
                _quantized_affine_interval(
                    velocity_gradient, bound, coefficient, mix_dtype, new_dtype
                )
            )
        reference_gradient = torch.zeros_like(new)
        if row["kl_beta"]:
            difference_dtype = torch.promote_types(new_dtype, row["reference"][name].dtype)
            difference = _rounded(new - row["reference"][name].double(), difference_dtype)
            # The enclosing CUDA autocast promotes pow to FP32; subtraction is
            # still in the public velocity dtype. Observe the actual KL output
            # precision rather than infer it from the operand or model family.
            reference_dtype = row["kl_dtype"]
            reference_sums.append(
                _rounded(
                    (_rounded(difference.square(), reference_dtype) * mask).flatten(1).sum(1),
                    reference_dtype,
                )
            )
            scalar = _rounded(
                _rounded(normalization, reference_dtype) * row["kl_beta"], reference_dtype
            )
            scalar = _rounded(scalar / batch, reference_dtype)
            scalar = _rounded(scalar / total.reshape(shape), reference_dtype)
            reference_gradient = _rounded(
                _rounded(scalar * _rounded(2 * difference, reference_dtype), reference_dtype)
                * mask,
                new_dtype,
            )
        branches.append(reference_gradient)
        branch_intervals.append((reference_gradient, reference_gradient))
        # Autograd may accumulate independent graph branches in different orders.
        # Propagate each finite branch order; no blanket BF16 tolerance is added.
        possibilities, ranges = [], []
        for order in permutations(range(len(branches))):
            accumulated = branches[order[0]]
            low, high = branch_intervals[order[0]]
            for index in order[1:]:
                accumulated = _rounded(accumulated + branches[index], new_dtype)
                low = _rounded(low + branch_intervals[index][0], new_dtype)
                high = _rounded(high + branch_intervals[index][1], new_dtype)
            possibilities.append(accumulated)
            ranges.append((low, high))
        candidates[name], intervals[name] = tuple(possibilities), tuple(ranges)
        gradients[name] = possibilities[0]
        budgets[name] = (
            gamma * sum(value.abs() for value in branches) + torch.finfo(torch.float32).tiny
        )
    policy = upper / beta * (reward * positive + negative_weight * negative)
    reference_loss = torch.tensor(0.0, dtype=torch.float64)
    if reference_sums:
        combined = reference_sums[0]
        for value in reference_sums[1:]:
            combined = _rounded(combined + value, reference_dtype)
        reduced = _rounded(combined / total, reference_dtype)
        reference_loss = _rounded(
            _rounded(reduced.mean(), reference_dtype) * row["kl_beta"], reference_dtype
        )
    return {
        "loss": policy.mean() + reference_loss,
        "gradients": gradients,
        "gradient_candidates": candidates,
        "gradient_intervals": intervals,
        "budgets": budgets,
        "scales": scales,
        "reward": reward,
    }


def assert_objective(row):
    """Keep strict loss/upstream checks and reject missing-CP and sign corruption."""
    expected = objective_reference(row)
    torch.testing.assert_close(row["loss"].double(), expected["loss"], rtol=2e-5, atol=1e-7)
    for name, targets in expected["gradient_candidates"].items():
        actual, budget = row["gradients"][name].double(), expected["budgets"][name]
        assert actual.shape == targets[0].shape

        def matches(value):
            return any(
                bool(((value >= low - budget) & (value <= high + budget)).all())
                for low, high in expected["gradient_intervals"][name]
            )

        assert matches(actual), (
            name,
            min(float((actual - target).abs().max()) for target in targets),
            float(budget.max()),
        )
        if bool((targets[0].abs() > 4 * budget).any()):
            for factor in (0.0, 0.5, 2.0, -1.0):
                assert not matches(actual * factor), (name, factor)
    return expected
