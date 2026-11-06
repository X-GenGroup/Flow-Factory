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

"""Observe actual preference operands; verify pair calculus independently in FP64."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from native_observation_utils import cpu

from flow_factory.trainers.offline import offline_dpo as implementation


@contextmanager
def observe_preferences(trainer):
    """Capture real noise, loss operands and upstream derivatives without replacing math."""
    rows, noising, errors = [], [], []
    build = implementation.build_noised_output_state
    mse = implementation.flow_matching_per_sample_loss
    objective = implementation.dpo_objective

    def noised(*args, **kwargs):
        times, value = build(*args, **kwargs)
        noising.append({"times": cpu(times), "noise": cpu(value.noise), "state": cpu(value.state)})
        return times, value

    def loss(adapter, prediction, value):
        result = mse(adapter, prediction, value)
        assert len(prediction.components) == 1 and prediction.active_masks is None
        tensor = next(iter(prediction.components.values()))
        row = {
            "prediction": cpu(tensor),
            "target": cpu(next(iter(value.target_velocity.components.values()))),
            "mse": cpu(result),
            "requires_grad": tensor.requires_grad,
        }
        if tensor.requires_grad:
            tensor.register_hook(lambda gradient: row.update(gradient=cpu(gradient)))
        errors.append(row)
        return result

    def preference(**kwargs):
        result, metrics = objective(**kwargs)
        assert len(errors) == 4 and len(noising) == 2
        rows.append(
            {
                "terms": list(errors),
                "noising": list(noising),
                "loss": cpu(result),
                "beta": kwargs["beta"],
            }
        )
        errors.clear()
        noising.clear()
        return result, metrics

    with (
        patch.object(implementation, "build_noised_output_state", noised),
        patch.object(implementation, "flow_matching_per_sample_loss", loss),
        patch.object(implementation, "dpo_objective", preference),
    ):
        yield rows
    assert not noising and not errors


def assert_preference_rows(rows, *, cp_size, gas=2, timesteps=2):
    """Check full-output CP/GAS exactly once and reject wrong arms or detached policy."""
    from test_context_parallel_training import _compare

    assert rows
    for row in rows:
        terms = row["terms"]
        assert [term["requires_grad"] for term in terms] == [True, True, False, False]
        left, right = row["noising"]
        _compare(left["noise"], right["noise"], exact=True)
        _compare(left["times"], right["times"], exact=True)
        assert any(
            not torch.equal(left["state"]["components"][name], right["state"]["components"][name])
            for name in left["state"]["components"]
        ), "preference arms must be distinct"
        expected = []
        for term in terms:
            error = term["prediction"].double() - term["target"].double()
            value = error.square().flatten(1).mean(1)
            torch.testing.assert_close(term["mse"].double(), value, rtol=2e-6, atol=2e-7)
            expected.append(value)
        pc, pr, rc, rr = expected
        z = 0.5 * row["beta"] * (pc - pr - rc + rr)
        loss = torch.logaddexp(torch.zeros_like(z), z).mean()
        torch.testing.assert_close(row["loss"].double(), loss, rtol=3e-6, atol=5e-7)
        for sign, term in zip((1, -1), terms[:2]):
            value = term["prediction"].double()
            dimensions = (value.shape[0],) + (1,) * (value.ndim - 1)
            derivative = (
                sign
                * row["beta"]
                * torch.sigmoid(z).reshape(dimensions)
                * (value - term["target"].double())
                / value.numel()
                * cp_size
                / gas
                / timesteps
            )
            # The upstream gradient crosses a BF16 velocity on native adapters.
            tolerance = 2 / 128 if term["prediction"].dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(
                term["gradient"].double(), derivative, rtol=tolerance, atol=1e-8
            )
        assert not any("gradient" in term for term in terms[2:])
