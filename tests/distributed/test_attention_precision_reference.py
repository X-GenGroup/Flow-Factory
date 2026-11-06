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

"""Analytical and corruption controls for the independent attention VJP oracle."""

import math

import pytest
import torch
import torch.nn.functional as F
from attention_precision_reference import (
    assert_attention_derivative,
    attention_vjp_reference,
    fast_f32_gemm_budget,
)


def _row(*, cancellation=False):
    torch.manual_seed(872)
    values = [torch.randn(1, 2, length, 8).bfloat16().float() for length in (4, 8, 8, 4)]
    query, key, value, upstream = values
    if cancellation:
        value[..., 1::2, :] = -value[..., ::2, :]
        key[..., 1::2, :] = key[..., ::2, :]
    operands = [tensor.detach().requires_grad_() for tensor in (query, key, value)]
    output = F.scaled_dot_product_attention(*operands)
    gradients = torch.autograd.grad(output, operands, upstream)
    row = dict(zip(("q", "k", "v", "dy"), values))
    row["output"] = output.detach()
    row.update(zip(("dq", "dk", "dv"), gradients))
    lse = (query.double() @ key.double().transpose(-1, -2) / math.sqrt(8)).logsumexp(-1).float()
    return row, lse


@pytest.mark.parametrize("cancellation", [False, True])
def test_reference_matches_fp64_autograd_and_encloses_fp32_derivatives(cancellation):
    row, lse = _row(cancellation=cancellation)
    reference, budgets = attention_vjp_reference(row, lse)
    values = [row[key].double().requires_grad_() for key in ("q", "k", "v")]
    output = F.scaled_dot_product_attention(*values)
    gradients = torch.autograd.grad(output, values, row["dy"].double())
    torch.testing.assert_close(reference["output"], output, atol=2e-15, rtol=2e-15)
    for key, gradient in zip(("dq", "dk", "dv"), gradients):
        torch.testing.assert_close(reference[key], gradient, atol=2e-15, rtol=2e-15)
        assert_attention_derivative(row[key], reference[key], budgets[key])


def test_budget_is_independent_of_observed_derivatives_and_rejects_local_corruption():
    row, lse = _row()
    reference, budgets = attention_vjp_reference(row, lse)
    corrupted = dict(row)
    for key in ("dq", "dk", "dv"):
        corrupted[key] = torch.full_like(row[key], math.nan)
    again, same_budgets = attention_vjp_reference(corrupted, lse)
    for key in ("dq", "dk", "dv"):
        assert torch.equal(reference[key], again[key])
        assert torch.equal(budgets[key], same_budgets[key])
        for factor in (0.0, 0.5, 2.0):
            with pytest.raises(AssertionError):
                assert_attention_derivative(row[key] * factor, reference[key], budgets[key])
        altered = row[key].contiguous().clone()
        altered.flatten()[0] = reference[key].flatten()[0] + 4 * budgets[key].flatten()[0]
        assert (altered != row[key]).count_nonzero() == 1
        assert (altered - row[key]).norm() / row[key].norm() < 1e-3
        with pytest.raises(AssertionError):
            assert_attention_derivative(altered, reference[key], budgets[key])


def test_fast_f32_one_exact_operand_and_underflow_terms():
    left = torch.tensor([[1.0009764, -0.62531, 1.141234]], dtype=torch.float32)
    right = torch.tensor([[0.75], [-1.25], [2.5]], dtype=torch.bfloat16).float()

    def truncate_tf32(value):
        return (value.view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)

    high = truncate_tf32(left)
    low = truncate_tf32(left - high)
    assert torch.equal(truncate_tf32(right), right)
    reconstructed = high.double() @ right.double() + low.double() @ right.double()
    exact = left.double() @ right.double()
    budget = fast_f32_gemm_budget(left.double().abs(), right.double().abs(), exact_right=True)
    assert bool(((reconstructed - exact).abs() <= budget).all())
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    underflow_budget = fast_f32_gemm_budget(zeros, zeros.T, exact_left=True, exact_right=True)
    assert underflow_budget.item() >= (7 * 3 + 1) * torch.finfo(torch.float32).tiny
    tiny = torch.full_like(zeros, torch.finfo(torch.float32).tiny)
    # Flushing every tiny product to zero must remain within the absolute bound.
    assert bool((tiny @ tiny.T <= fast_f32_gemm_budget(tiny, tiny.T)).all())


@pytest.mark.parametrize("mutation", ["nan", "non_bf16", "subnormal", "invalid_lse"])
def test_reference_refuses_inputs_outside_its_arithmetic_contract(mutation):
    row, lse = _row()
    if mutation == "invalid_lse":
        lse.fill_(math.inf)
    elif mutation == "nan":
        row["q"].flatten()[0] = math.nan
    elif mutation == "subnormal":
        row["q"].flatten()[0] = torch.finfo(torch.float32).tiny / 2
    else:
        row["q"].flatten()[0] = 1.0001
    with pytest.raises(AssertionError):
        attention_vjp_reference(row, lse)
