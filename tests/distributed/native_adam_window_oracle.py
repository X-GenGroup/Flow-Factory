# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Independent raw-gradient Adam references with bounded all-element windows."""

import torch
from test_model_preparation import full_value


def _records(parameters):
    return {
        "initial_parameters": {
            name: full_value(parameter).cpu() for name, parameter in parameters.items()
        },
        "updates": [],
    }


def _capture_update(parameters, optimizer, records, loss, *, clipping_world_size):
    gradients = {
        name: full_value(parameter.grad).cpu()
        for name, parameter in parameters.items()
        if parameter.grad is not None
    }
    before = (
        records["updates"][-1]["parameters"]
        if records["updates"]
        else records["initial_parameters"]
    )
    # No clipping is needed for this model-only mathematical control. An explicit
    # large threshold retains the common independent Adam oracle's exact schema.
    settings = {
        key: optimizer.param_groups[0][key] for key in ("lr", "betas", "eps", "weight_decay")
    }
    settings["max_grad_norm"] = 1e9
    full_gradient_norm = (
        sum(value.double().square().sum() for value in gradients.values()).sqrt().item()
    )
    assert full_gradient_norm < settings["max_grad_norm"]
    optimizer.step()
    after = {name: full_value(parameter).cpu() for name, parameter in parameters.items()}
    records["updates"].append(
        dict(
            gradients=gradients,
            clipped_gradients=gradients,
            clipping_world_size=clipping_world_size,
            parameters=after,
            deltas={name: after[name] - before[name] for name in gradients},
            moments={
                name: {
                    key: full_value(value).cpu()
                    for key, value in optimizer.state[parameters[name]].items()
                }
                for name in gradients
            },
            optimizer=settings,
            unclipped_global_gradient_norm=full_gradient_norm,
            loss=loss.detach().cpu(),
        )
    )


def _audit_updates(actual, expected):
    """Reuse strict common BF16/FP64 Adam checks on bounded tensor windows.

    Audit independent elementwise Adam recurrences in bounded deterministic
    windows, preserving the actual source process count in every reduced record.
    There is no clipping in this control; every window has coefficient one.
    Every element and all three warm Adam steps remain checked.
    """
    from test_context_parallel_training import _compare_bf16_updates

    rows = {}
    active = expected["updates"][0]["gradients"]
    for name in active:
        count = active[name].numel()
        maxima = {"gradient_relative_l2": 0.0, "update_relative_l2": 0.0}
        for start in range(0, count, 262144):

            def window(value):
                return value.reshape(-1)[start : start + 262144]

            pair = []
            for source in (actual, expected):
                reduced = {
                    "initial_parameters": {name: window(source["initial_parameters"][name])},
                    "updates": [],
                }
                for update in source["updates"]:
                    reduced["updates"].append(
                        {
                            **{
                                key: {name: window(update[key][name])}
                                for key in (
                                    "gradients",
                                    "clipped_gradients",
                                    "parameters",
                                    "deltas",
                                )
                            },
                            "moments": {
                                name: {
                                    key: value if key == "step" else window(value)
                                    for key, value in update["moments"][name].items()
                                }
                            },
                            "optimizer": update["optimizer"],
                            "clipping_world_size": update["clipping_world_size"],
                            "loss": update["loss"],
                        }
                    )
                pair.append(reduced)
            for step in _compare_bf16_updates(*pair):
                for key in maxima:
                    maxima[key] = max(maxima[key], step[key])
        rows[name] = {**maxima, "elements": count, "adam_steps": 3, "elementwise_comparison": True}
    return rows
