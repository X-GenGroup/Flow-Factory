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

"""Forward-only error budgets for the qualified native FP32 attention VJP.

This is a fixture-specific reference for native Sm80+ efficient attention,
without masks, dropout or GQA, at supported FP32 operation boundaries. It models CUTLASS
FastF32's three TF32 products and order-dependent FP32 partial reductions. No
observed derivative enters a reference or budget. Other kernels need their own
arithmetic contract; this helper is not a general SDPA tolerance.
"""

import math

import torch

_U = torch.finfo(torch.float32).eps / 2
_TINY = torch.finfo(torch.float32).tiny
_TF32 = 2**-10


def gamma(count: int, unit: float = _U) -> float:
    assert 0 <= count * unit < 1
    return count * unit / (1 - count * unit)


def fast_f32_gemm_budget(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    exact_left: bool = False,
    exact_right: bool = False,
) -> torch.Tensor:
    """Bound a FastF32 product from operand magnitude upper bounds.

    BF16-exact normal operands need no TF32 conversion. Otherwise high-part
    truncation and low-part rounding leave at most t^2 relative reconstruction
    error. The omitted low*low product and FTZ allowances are explicit.
    """
    assert left.dtype == right.dtype == torch.float64
    assert bool((left >= 0).all() and (right >= 0).all())
    inner = left.shape[-1]
    assert inner == right.shape[-2]

    def conversion(value, exact):
        if exact:
            return torch.zeros_like(value), torch.zeros_like(value)
        rho = _TF32**2 * value + (2 + 2 * _TF32) * _TINY
        low = _TF32 * (1 + _TF32) * value + (3 + 2 * _TF32) * _TINY
        return rho, low

    rho_left, low_left = conversion(left, exact_left)
    rho_right, low_right = conversion(right, exact_right)
    conversion_error = (
        left @ rho_right + rho_left @ right + rho_left @ rho_right + low_left @ low_right
    )
    magnitude = left @ right + left @ low_right + low_left @ right
    assert bool(torch.isfinite(magnitude).all())
    assert bool((magnitude < torch.finfo(torch.float32).max).all())
    # <=3N products/accumulations, <=N partial merges, one output rounding.
    # TensorCore accumulation need not round to nearest, hence eps32 (2u).
    count = 4 * inner + 1
    underflow = (7 * inner + 1) * _TINY / (1 - count * 2 * _U)
    return conversion_error + gamma(count, 2 * _U) * magnitude + underflow


def attention_vjp_reference(
    row: dict[str, torch.Tensor], lse: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return FP64 VJP values and forward-derived per-element error budgets.

    ``row`` supplies only q/k/v/dy and the independently replayed native output.
    The caller verifies actual native backend and exact forward/LSE replay before
    passing the native FP32 LSE. Derivative keys are deliberately never read.
    """
    values = [row[key] for key in ("q", "k", "v", "dy")]
    for value in values:
        assert value.dtype == torch.float32 and value.ndim == 4
        assert bool(torch.isfinite(value).all())
        assert bool(((value == 0) | (value.abs() >= _TINY)).all())
    # Native CUDA RMSNorm can produce genuine FP32 Q/K. Determine conversion
    # exactness from forward operands, never from observed derivative residuals.
    q_exact, k_exact, v_exact, upstream_exact = [
        torch.equal(value, value.bfloat16().float()) for value in values
    ]
    q, k, v, upstream = [value.double().cpu() for value in values]
    assert row["output"].dtype == torch.float32
    actual_output = row["output"].double().cpu()
    assert bool(torch.isfinite(actual_output).all())
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2] == upstream.shape[:-2]
    assert q.shape[-1] == k.shape[-1] and k.shape[-2] == v.shape[-2]
    assert upstream.shape == actual_output.shape == (*q.shape[:-1], v.shape[-1])
    query_length, depth = q.shape[-2:]
    key_length, value_depth = v.shape[-2:]
    assert lse.dtype == torch.float32 and lse.shape[:-1] == q.shape[:-2]
    assert lse.shape[-1] >= query_length
    native_lse = lse.double().cpu()[..., :query_length, None]
    assert bool(torch.isfinite(native_lse).all())

    scale = 1 / math.sqrt(depth)
    scale32 = float(torch.tensor(scale, dtype=torch.float32))
    scale_error = abs(scale32 - scale)
    product = q @ k.transpose(-1, -2)
    scores = scale * product
    probability = scores.softmax(-1)
    output = probability @ v
    pre_softmax_gradient = upstream @ v.transpose(-1, -2)
    delta = (upstream * output).sum(-1, keepdim=True)
    centered = pre_softmax_gradient - delta
    scaled = scale * probability * centered
    reference = {
        "output": output,
        "dq": scaled @ k,
        "dk": scaled.transpose(-1, -2) @ q,
        "dv": probability.transpose(-1, -2) @ upstream,
    }

    product_error = fast_f32_gemm_budget(
        q.abs(), k.abs().transpose(-1, -2), exact_left=q_exact, exact_right=k_exact
    )
    score_error = (
        abs(scale32) * product_error
        + scale_error * product.abs()
        + _U * abs(scale32) * (product.abs() + product_error)
        + _TINY
    )
    exponent = scores - native_lse
    subtraction_error = score_error + _U * (scores.abs() + score_error + native_lse.abs()) + _TINY
    exponent_error = subtraction_error + (native_lse - scores.logsumexp(-1, keepdim=True)).abs()
    assert bool(torch.isfinite(exponent_error).all())
    assert bool((exponent.abs() + subtraction_error < 80).all())
    # CUDA expf: <=2 ULP; __expf: <=2+floor(1.16*abs(x)) ULP.
    # Use the larger interval bound, plus ideal-result rounding, for either.
    # The continuous upper envelope avoids a discontinuous floor being rounded
    # down across an integer boundary by the FP64 reference evaluation.
    ulps = 2 + 1.16 * (exponent.abs() + subtraction_error)
    exponential_relative_error = _U + 2 * _U * (1 + _U) * ulps
    probability_error = probability * (
        torch.expm1(exponent_error) + torch.exp(exponent_error) * exponential_relative_error
    )
    assert bool((probability - probability_error > _TINY).all())
    attention_gradient_error = fast_f32_gemm_budget(
        upstream.abs(), v.abs().transpose(-1, -2), exact_left=upstream_exact, exact_right=v_exact
    )
    delta_error = (
        (upstream * (actual_output - output)).sum(-1, keepdim=True).abs()
        + gamma(2 * value_depth) * (upstream.abs() * actual_output.abs()).sum(-1, keepdim=True)
        + 2 * value_depth * _TINY / (1 - 2 * value_depth * _U)
    )
    centered_error = (
        attention_gradient_error
        + delta_error
        + _U * (pre_softmax_gradient.abs() + delta.abs() + attention_gradient_error + delta_error)
        + _TINY
    )
    weighted_error = (
        probability * centered_error
        + centered.abs() * probability_error
        + probability_error * centered_error
        + _U * (probability + probability_error) * (centered.abs() + centered_error)
        + _TINY
    )
    scaled_error = (
        abs(scale32) * weighted_error
        + scale_error * (probability * centered).abs()
        + _U * abs(scale32) * ((probability * centered).abs() + weighted_error)
        + _TINY
    )
    budgets = {
        "dq": scaled_error @ k.abs()
        + fast_f32_gemm_budget(scaled.abs() + scaled_error, k.abs(), exact_right=k_exact),
        "dk": scaled_error.transpose(-1, -2) @ q.abs()
        + fast_f32_gemm_budget(
            (scaled.abs() + scaled_error).transpose(-1, -2), q.abs(), exact_right=q_exact
        ),
        "dv": probability_error.transpose(-1, -2) @ upstream.abs()
        + fast_f32_gemm_budget(
            (probability + probability_error).transpose(-1, -2),
            upstream.abs(),
            exact_right=upstream_exact,
        ),
    }

    # Bound FP64 evaluation separately: each explicit graph path has <=8N
    # arithmetic operations, with a factor 16 for propagation through branches.
    # The qualified Torch CPU implementation uses SLEEF u10 exp/log (<=1 ULP),
    # covered by eps64 rather than half-eps64. Non-cancelling magnitudes protect
    # small derivatives. The score factor covers the at-most-two occurrences of
    # P in a VJP term; the <1e-5 guard bounds higher-order perturbations.
    evaluation_error = gamma(
        128 * (query_length + key_length + depth + value_depth), torch.finfo(torch.float64).eps
    )
    score_magnitude = (scale * (q.abs() @ k.abs().transpose(-1, -2))).amax()
    reference_factor = evaluation_error * (1 + 4 * score_magnitude)
    assert reference_factor < 1e-5
    absolute_centered = upstream.abs() @ v.abs().transpose(-1, -2) + (
        upstream.abs() * (probability @ v.abs() + actual_output.abs())
    ).sum(-1, keepdim=True)
    absolute_scaled = scale * probability * absolute_centered
    absolute_vjp = {
        "dq": absolute_scaled @ k.abs(),
        "dk": absolute_scaled.transpose(-1, -2) @ q.abs(),
        "dv": probability.transpose(-1, -2) @ upstream.abs(),
    }
    for key, budget in budgets.items():
        budget = (budget + reference_factor * absolute_vjp[key]) / (1 - reference_factor)
        budgets[key] = torch.nextafter(budget, torch.full_like(budget, math.inf))
        assert bool(torch.isfinite(reference[key]).all() and torch.isfinite(budgets[key]).all())
        assert bool((budgets[key] > 0).all())
    return reference, budgets


def assert_attention_derivative(
    actual: torch.Tensor, reference: torch.Tensor, budget: torch.Tensor
) -> None:
    """Check every element without a measured-residual-derived tolerance."""
    assert actual.dtype == torch.float32 and actual.shape == reference.shape == budget.shape
    error = (actual.double().cpu() - reference).abs()
    assert bool((error <= budget).all()), {
        "maximum_error": float(error.max()),
        "maximum_error_over_budget": float((error / budget).max()),
    }
