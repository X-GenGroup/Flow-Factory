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

"""Online feedback-sensitive trajectory checks, separate from SFT's strict oracle."""

import pytest
import torch
from test_context_parallel_training import (
    _adam_direction_difference_bound,
    _assert_error_bound,
    _bf16_tensors,
    _check_adam_trajectory,
    _compare,
    _relative_l2,
)


def _diagnostic_relative_l2(left, right, *, denominator_floor=1e-30):
    assert left.keys() == right.keys()
    a = torch.cat([left[name].double().flatten() for name in sorted(left)])
    b = torch.cat([right[name].double().flatten() for name in sorted(right)])
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a - b).norm() / b.norm().clamp_min(denominator_floor))


def predict_updates_at_same_states(actual, native_reference):
    """Rebuild Adam from independent native gradients along actual model states.

    Only the reference's raw gradients, pre-parameters and optimizer settings are
    used. Its optimizer moments/deltas belong to a forced-state leg and are not
    a reference. Local gradient gates must pass before residuals enter a budget.
    """
    contexts = _check_adam_trajectory(actual)
    assert len(actual["updates"]) == len(native_reference["updates"])
    moments, input_errors, predictions = {}, {}, []
    for step, (update, reference, context) in enumerate(
        zip(actual["updates"], native_reference["updates"], contexts), 1
    ):
        assert update["optimizer"] == reference["optimizer"]
        _compare(update["before"], reference["before"], exact=True)
        gradient_residual = _bf16_tensors(update["gradients"], reference["gradients"], atol=1e-3)
        settings = reference["optimizer"]
        beta1, beta2 = settings["betas"]
        gradients = {name: value.double() for name, value in reference["gradients"].items()}
        flat = torch.cat([gradients[name].flatten() for name in sorted(gradients)])
        coefficient = min(1.0, settings["max_grad_norm"] / (float(flat.norm()) + 1e-6))
        predicted, budgets = {}, {}
        count = 2 * flat.numel() + 64 * step + 64
        u = torch.finfo(torch.float64).eps / 2
        assert count * u < 1
        gamma64 = count * u / (1 - count * u)
        for name, gradient in gradients.items():
            k, h = gradient * coefficient, update["clipped_gradients"][name].double()
            old_m, old_v, old_abs = moments.get(
                name, (torch.zeros_like(k), torch.zeros_like(k), torch.zeros_like(k))
            )
            m = beta1 * old_m + (1 - beta1) * k
            v = beta2 * old_v + (1 - beta2) * k.square()
            abs_m = beta1 * old_abs + (1 - beta1) * k.abs()
            assert torch.isfinite(m).all() and torch.isfinite(v).all() and (v >= 0).all()
            moments[name] = m, v, abs_m
            dm, dv = input_errors.get(name, (torch.zeros_like(k), torch.zeros_like(k)))
            dm = beta1 * dm + (1 - beta1) * (h - k).abs()
            dv = beta2 * dv + (1 - beta2) * (h.square() - k.square()).abs()
            input_errors[name] = dm, dv
            before = reference["before"][name].double()
            direction = (m / (1 - beta1**step)) / ((v / (1 - beta2**step)).sqrt() + settings["eps"])
            predicted[name] = -settings["lr"] * (direction + settings["weight_decay"] * before)
            errors = tuple(
                error + rounding for error, rounding in zip((dm, dv), context["rounding"][name])
            )
            bound = (
                settings["lr"]
                * _adam_direction_difference_bound(
                    update["moments"][name],
                    {"exp_avg": m, "exp_avg_sq": v},
                    errors,
                    settings,
                    step,
                )
                + context["delta_rounding"][name]
            )
            # Conservative FP64 operation-count allowance, using positive paths
            # instead of the measured delta residual. FP32 budgets dominate it.
            absolute_path = settings["lr"] * (
                abs_m / (1 - beta1**step) / settings["eps"]
                + settings["weight_decay"] * before.abs()
            )
            budgets[name] = torch.nextafter(
                (bound + gamma64 * (absolute_path + bound)) * (1 + gamma64),
                torch.full_like(bound, float("inf")),
            )
            _assert_error_bound(
                update["deltas"][name], predicted[name], budgets[name], f"native-state Adam: {name}"
            )
        predictions.append(
            {
                "deltas": predicted,
                "budgets": budgets,
                "gradient_relative_l2": gradient_residual,
                "update_relative_l2": _relative_l2(update["deltas"], predicted, limit=4 / 128),
            }
        )
    return predictions


def compare_common_state(actual, reference):
    """Keep the original local gradient gates on identical states and operands."""
    _compare(actual["initial_parameters"], reference["initial_parameters"], exact=True)
    _compare(actual["samples"], reference["samples"], exact=True)
    _compare(actual["feedback"], reference["feedback"], exact=True)
    _compare(actual["replay_inputs"], reference["replay_inputs"], exact=True)
    assert len(actual["objectives"]) == len(reference["objectives"])
    for a, b in zip(actual["objectives"], reference["objectives"]):
        for key in (
            "role",
            "sample_ids",
            "boundary_index",
            "replay_step",
            "perturbation_times",
            "noised",
        ):
            _compare(a.get(key), b.get(key), exact=True)
        assert len(a["modules"]) == len(b["modules"])
        for am, bm in zip(a["modules"], b["modules"]):
            for key in ("name", "stage", "args", "kwargs"):
                _compare(am[key], bm[key], exact=True)
            _compare(am["output"], bm["output"], exact=True)
    assert len(actual["updates"]) == len(reference["updates"])
    metrics = []
    for index, (a, b) in enumerate(zip(actual["updates"], reference["updates"])):
        assert a["role"] == b["role"]
        _compare(a["before"], b["before"], exact=True)
        metric = _bf16_tensors(a["gradients"], b["gradients"], atol=1e-3)
        for factor in (0.0, 0.5, 2.0):
            with pytest.raises(AssertionError):
                _bf16_tensors(
                    {name: value * factor for name, value in a["gradients"].items()},
                    b["gradients"],
                    atol=1e-3,
                )
        metrics.append({"step": index + 1, "role": a["role"], "gradient_relative_l2": metric})
    return metrics


def compare_feedback_updates(actual, reference, *, algorithm, same_state_reference=None):
    """Validate actual trajectories after their independent common-state control.

    DMD2 feedback and guided GRPO may amplify small BF16 errors between evolved
    states. Their cross-trajectory gradient/moment norms are diagnostics. Every
    run still passes independent clipping/Adam equations and parameter caps.
    DMD2 generator updates additionally use CP1 native gradients at the actual
    CP2 states to predict feedback drift, retaining the original update cap on
    actual-to-same-state predictions. Its cross-trajectory update cap failure is
    explicit diagnostic evidence. Other updates retain that cross-trajectory cap.
    Never use this for SFT, unguided GRPO or fixed-topology continuation.
    """
    assert algorithm in ("dmd2", "grpo_cfg")
    ac, bc = _check_adam_trajectory(actual), _check_adam_trajectory(reference)
    assert len(actual["updates"]) == len(reference["updates"])
    predictions = None
    if algorithm == "dmd2":
        assert same_state_reference is not None, "DMD2 requires reverse common-state evidence"
        predictions = (
            predict_updates_at_same_states(actual, same_state_reference),
            predict_updates_at_same_states(reference, reference),
        )
    propagated = {}
    metrics = []
    for step, (a, b, aa, bb) in enumerate(zip(actual["updates"], reference["updates"], ac, bc), 1):
        assert a["optimizer"] == b["optimizer"]
        measured = {
            "gradient_relative_l2_diagnostic": _diagnostic_relative_l2(
                a["gradients"], b["gradients"]
            ),
            "update_relative_l2": _diagnostic_relative_l2(
                a["deltas"], b["deltas"], denominator_floor=1e-12
            ),
            "parameter_relative_l2": _bf16_tensors(a["parameters"], b["parameters"], atol=1e-3),
            "loss_relative_l2_diagnostic": _diagnostic_relative_l2(
                {"loss": a["loss"]}, {"loss": b["loss"]}
            ),
        }
        if algorithm != "dmd2" or a["role"] != "generator":
            measured["update_relative_l2"] = _relative_l2(a["deltas"], b["deltas"], limit=4 / 128)
        measured["cross_trajectory_update_cap_pass"] = measured["update_relative_l2"] <= 4 / 128
        measured["cross_trajectory_update_cap"] = 4 / 128
        if predictions is not None:
            pa, pb = (values[step - 1] for values in predictions)
            for name in a["deltas"]:
                _assert_error_bound(
                    a["deltas"][name].double() - b["deltas"][name].double(),
                    pa["deltas"][name] - pb["deltas"][name],
                    pa["budgets"][name] + pb["budgets"][name],
                    f"native-state feedback prediction: {name}",
                )
            measured["actual_to_native_state_update_relative_l2"] = pa["update_relative_l2"]
            measured["predicted_cross_trajectory_update_relative_l2"] = _diagnostic_relative_l2(
                pa["deltas"], pb["deltas"]
            )
        settings = b["optimizer"]
        beta1, beta2 = settings["betas"]
        for name, state in b["moments"].items():
            h, k = a["clipped_gradients"][name].double(), b["clipped_gradients"][name].double()
            dm, dv = propagated.get(name, (torch.zeros_like(h), torch.zeros_like(h)))
            dm = beta1 * dm + (1 - beta1) * (h - k).abs()
            dv = beta2 * dv + (1 - beta2) * (h.square() - k.square()).abs()
            propagated[name] = dm, dv
            bounds = tuple(
                error + aa["rounding"][name][i] + bb["rounding"][name][i]
                for i, error in enumerate((dm, dv))
            )
            for key, bound in zip(("exp_avg", "exp_avg_sq"), bounds):
                _assert_error_bound(
                    a["moments"][name][key], state[key].double(), bound, f"online CP {key}: {name}"
                )
            direction = _adam_direction_difference_bound(
                a["moments"][name], state, bounds, settings, step
            )
            bound = settings["lr"] * (
                direction
                + settings["weight_decay"] * (aa["before"][name] - bb["before"][name]).abs()
            )
            bound += aa["delta_rounding"][name] + bb["delta_rounding"][name]
            _assert_error_bound(
                a["deltas"][name], b["deltas"][name].double(), bound, f"online CP delta: {name}"
            )
            _compare(a["moments"][name]["step"], state["step"], exact=True)
        for key in ("exp_avg", "exp_avg_sq"):
            measured[key + "_relative_l2_diagnostic"] = _diagnostic_relative_l2(
                {n: s[key] for n, s in a["moments"].items()},
                {n: s[key] for n, s in b["moments"].items()},
            )
        metrics.append(measured)
    return metrics
