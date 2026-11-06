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

"""Negative controls for the narrowly scoped online trajectory oracle."""

import copy

import pytest
import torch
from online_update_reference import (
    compare_common_state,
    compare_feedback_updates,
    predict_updates_at_same_states,
)
from test_context_parallel_training import _optimizer_fixture


def histories():
    gradients = [torch.linspace(-0.7, 1.0, 64) * value for value in (1.0, 0.8)]
    values = (
        _optimizer_fixture([value * 1.02 for value in gradients]),
        _optimizer_fixture(gradients),
    )
    for history in values:
        before = history["initial_parameters"]
        for update in history["updates"]:
            update["role"] = "generator"
            update["before"] = before
            before = update["parameters"]
    return values


def test_feedback_exception_preserves_optimizer_and_update_checks():
    actual, reference = histories()
    metrics = compare_feedback_updates(
        actual, reference, algorithm="dmd2", same_state_reference=actual
    )
    assert metrics[0]["gradient_relative_l2_diagnostic"] > 2 / 128
    assert all(row["update_relative_l2"] < 4 / 128 for row in metrics)
    for kind in ("moment", "update", "clipping"):
        corrupted = copy.deepcopy(actual)
        update = corrupted["updates"][1]
        if kind == "moment":
            update["moments"]["p"]["exp_avg"][0] += 1e-3
        elif kind == "update":
            update["deltas"]["p"][0] += 1e-5
        else:
            update["clipped_gradients"]["p"] *= 0.5
        with pytest.raises(AssertionError):
            compare_feedback_updates(
                corrupted, reference, algorithm="dmd2", same_state_reference=actual
            )


def test_feedback_prediction_requires_a_native_reference_at_actual_states():
    actual, reference = histories()
    with pytest.raises(AssertionError, match="reverse common-state"):
        compare_feedback_updates(actual, reference, algorithm="dmd2")
    wrong_state = copy.deepcopy(actual)
    wrong_state["updates"][0]["before"]["p"][0] += 0.01
    with pytest.raises(AssertionError):
        predict_updates_at_same_states(actual, wrong_state)


def test_prediction_uses_raw_reference_gradients_without_shadow_optimizer_state():
    actual, _ = histories()
    native = copy.deepcopy(actual)
    for update in native["updates"]:
        for key in ("moments", "deltas", "clipped_gradients", "parameters"):
            del update[key]
    predictions = predict_updates_at_same_states(actual, native)
    assert all(row["update_relative_l2"] < 1e-3 for row in predictions)
    for factor in (0.0, 0.5, 2.0):
        wrong = copy.deepcopy(native)
        wrong["updates"][0]["gradients"]["p"] *= factor
        with pytest.raises(AssertionError):
            predict_updates_at_same_states(actual, wrong)
    wrong = copy.deepcopy(native)
    wrong["updates"][0]["gradients"]["p"][0] += 1.0
    with pytest.raises(AssertionError):
        predict_updates_at_same_states(actual, wrong)


def test_dmd2_feedback_drift_is_reported_without_exempting_local_update_errors():
    # One step isolates feedback/update drift while the separate absolute
    # parameter cap still passes; persistent parameter drift is not exempted.
    gradients = [torch.linspace(-0.7, 1.0, 64)]
    reference = _optimizer_fixture(gradients)
    actual = _optimizer_fixture([gradient + 0.05 for gradient in gradients])
    for history in (actual, reference):
        before = history["initial_parameters"]
        for update in history["updates"]:
            update.update(role="generator", before=before)
            before = update["parameters"]
    native = copy.deepcopy(actual)
    metrics = compare_feedback_updates(
        actual, reference, algorithm="dmd2", same_state_reference=native
    )
    assert any(not row["cross_trajectory_update_cap_pass"] for row in metrics)
    assert all(row["actual_to_native_state_update_relative_l2"] < 4 / 128 for row in metrics)
    with pytest.raises(AssertionError):
        compare_feedback_updates(actual, reference, algorithm="grpo_cfg")
    fake_actual, fake_reference, fake_native = copy.deepcopy((actual, reference, native))
    for history in (fake_actual, fake_reference, fake_native):
        for update in history["updates"]:
            update["role"] = "fake"
    with pytest.raises(AssertionError):
        compare_feedback_updates(
            fake_actual, fake_reference, algorithm="dmd2", same_state_reference=fake_native
        )
    corrupted = copy.deepcopy(actual)
    corrupted["updates"][-1]["parameters"]["p"][0] += 1e-5
    corrupted["updates"][-1]["deltas"]["p"][0] += 1e-5
    with pytest.raises(AssertionError):
        compare_feedback_updates(
            corrupted, reference, algorithm="dmd2", same_state_reference=native
        )


def test_feedback_exception_cannot_be_used_for_sft_or_unguided_grpo():
    actual, reference = histories()
    for algorithm in ("sft", "grpo"):
        with pytest.raises(AssertionError):
            compare_feedback_updates(actual, reference, algorithm=algorithm)


@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0])
def test_common_state_control_rejects_misscaled_gradients(factor):
    _, reference = histories()
    for key in ("samples", "feedback", "replay_inputs", "objectives"):
        reference[key] = []
    for index, update in enumerate(reference["updates"]):
        update["role"] = "generator"
        update["before"] = (
            reference["initial_parameters"]
            if index == 0
            else reference["updates"][index - 1]["parameters"]
        )
    compare_common_state(reference, reference)
    corrupted = copy.deepcopy(reference)
    corrupted["updates"][1]["gradients"]["p"] *= factor
    with pytest.raises(AssertionError):
        compare_common_state(corrupted, reference)
