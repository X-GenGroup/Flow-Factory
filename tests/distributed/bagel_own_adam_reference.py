# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Bagel same-state checks and actual-only warm Adam auditing."""

import torch
from test_context_parallel_training import _bf16_tensors, _check_adam_trajectory
from test_model_preparation import full_value

WINDOW = 262144


def synchronize_shadow(actual, shadow, optimizer):
    """Copy current canonical T values to S; never modify T or its optimizer."""
    assert actual.keys() == shadow.keys()
    versions = {name: p._version for name, p in actual.items()}
    state_versions = {
        (id(p), key): (id(value), value._version)
        for p, state in optimizer.state.items()
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    with torch.no_grad():
        for name, parameter in actual.items():
            assert parameter is not shadow[name]
            expected = full_value(parameter)
            assert expected.dtype == shadow[name].dtype == torch.float32
            shadow[name].copy_(expected)
            torch.testing.assert_close(shadow[name], expected, rtol=0, atol=0)
    assert versions == {name: p._version for name, p in actual.items()}
    assert state_versions == {
        (id(p), key): (id(value), value._version)
        for p, state in optimizer.state.items()
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    return {
        "copied_parameters": len(actual),
        "actual_versions_unchanged": True,
        "actual_optimizer_state_unchanged": True,
    }


def same_state_gradients(actual, shadow, *, diagnostic_window_size=WINDOW):
    """Apply the shared all-element and whole-gradient-dictionary limits once.

    Parameter and memory-window relative errors are diagnostics, not additional
    acceptance gates. Their signal energy can vary with an arbitrary partition.
    Own-gradient FP64 Adam recurrence windows remain independently checked.
    """
    assert actual.keys() == shadow.keys() and actual
    assert type(diagnostic_window_size) is int and diagnostic_window_size > 0
    assert all(actual[name].dtype == shadow[name].dtype == torch.float32 for name in actual)
    global_relative_l2 = _bf16_tensors(actual, shadow, atol=1e-3)
    rows = {}
    for name in sorted(actual):
        left, right = actual[name].reshape(-1), shadow[name].reshape(-1)
        full = float((left - right).norm() / right.norm().clamp_min(1e-12))
        maxima = 0.0
        for start in range(0, left.numel(), diagnostic_window_size):
            a, b = (
                left[start : start + diagnostic_window_size],
                right[start : start + diagnostic_window_size],
            )
            residual = float((a - b).norm() / b.norm().clamp_min(1e-12))
            maxima = max(maxima, residual)
        rows[name] = {
            "elements": left.numel(),
            "diagnostic_full_relative_l2": full,
            "diagnostic_max_window_relative_l2": maxima,
            "all_elements_checked": True,
        }
    return {
        "global_relative_l2": global_relative_l2,
        "relative_l2_scope": "all_trainable_gradients",
        "parameters": rows,
    }


def own_adam(records, *, world_size):
    """Audit every actual element through all three warm updates independently."""
    assert world_size in (1, 8)
    assert len(records["updates"]) == 3
    active = set(records["updates"][0]["gradients"])
    assert active
    for update in records["updates"]:
        assert set(update["gradients"]) == active
        assert update["clipping_world_size"] == world_size
        assert update["unclipped_global_gradient_norm"] < update["optimizer"]["max_grad_norm"]
    rows = {}
    for name in sorted(active):
        count = records["initial_parameters"][name].numel()
        windows = 0
        for start in range(0, count, WINDOW):

            def window(value):
                return value.reshape(-1)[start : start + WINDOW]

            reduced = {
                "initial_parameters": {name: window(records["initial_parameters"][name])},
                "updates": [],
            }
            for update in records["updates"]:
                reduced["updates"].append(
                    {
                        **{
                            key: {name: window(update[key][name])}
                            for key in ("gradients", "clipped_gradients", "parameters", "deltas")
                        },
                        "moments": {
                            name: {
                                key: value if key == "step" else window(value)
                                for key, value in update["moments"][name].items()
                            }
                        },
                        "optimizer": update["optimizer"],
                        "loss": update["loss"],
                        "clipping_world_size": update["clipping_world_size"],
                    }
                )
            contexts = _check_adam_trajectory(reduced)
            assert len(contexts) == 3 and all(float(row["coefficient"]) == 1.0 for row in contexts)
            windows += 1
        rows[name] = {
            "elements": count,
            "windows": windows,
            "adam_steps": 3,
            "own_gradient_fp64_recurrence": True,
            "clipping_world_size": update["clipping_world_size"],
        }
    return rows
