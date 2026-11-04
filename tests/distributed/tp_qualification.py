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

"""Numerical oracles shared by TP/SP architecture qualification tests."""

from __future__ import annotations

from typing import Dict

import torch
from torch.distributed.tensor import DTensor


def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a detached value without retaining the training graph."""
    value = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
    return value.detach().clone()


def parameters_by_name(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    """Canonicalize wrapper names and fix collective order across processes."""
    from flow_factory.models.variants import ComponentVariantRegistry

    pairs = [
        (ComponentVariantRegistry._canonical_prepared_parameter_name(name), parameter)
        for name, parameter in model.named_parameters()
    ]
    values = dict(pairs)
    assert len(values) == len(pairs), "transparent wrappers produced duplicate parameter names"
    # Never iterate a set before full_tensor(): Python hash seeds differ across
    # spawned workers, while distributed gathers must have identical order.
    return dict(sorted(values.items()))


def assert_optimizer_step_parity(
    reference: torch.nn.Module,
    sharded: torch.nn.Module,
    *,
    rtol: float,
    atol: float,
) -> Dict[str, int]:
    """Check every gradient, a real update, and unchanged inactive parameters.

    Call after backward on both models. Gradient presence is part of the oracle;
    expected zero gradients are checked too. Comparing update deltas prevents a
    large unchanged parameter value from hiding a missing optimizer update.
    """
    expected = parameters_by_name(reference)
    received = parameters_by_name(sharded)
    assert expected.keys() == received.keys(), "parameter membership changed"
    before = {}
    gradients = 0
    for name, parameter in expected.items():
        actual = received[name]
        assert parameter.requires_grad == actual.requires_grad, f"trainability changed: {name}"
        assert (parameter.grad is None) == (
            actual.grad is None
        ), f"gradient membership changed: {name}"
        before[name] = (full_tensor(parameter), full_tensor(actual))
        torch.testing.assert_close(before[name][1], before[name][0], rtol=rtol, atol=atol, msg=name)
        if parameter.grad is not None:
            gradients += 1
            torch.testing.assert_close(
                full_tensor(actual.grad),
                full_tensor(parameter.grad),
                rtol=rtol,
                atol=atol,
                msg=name,
            )
        if not parameter.requires_grad:
            assert actual.grad is None, f"frozen parameter received a gradient: {name}"
    assert gradients, "the qualification fixture produced no gradients"
    learning_rate = 0.01
    for model in (reference, sharded):
        torch.optim.SGD(model.parameters(), lr=learning_rate).step()
    changed = 0
    for name, parameter in expected.items():
        expected_before, received_before = before[name]
        expected_after = full_tensor(parameter)
        received_after = full_tensor(received[name])
        expected_update = expected_after - expected_before
        received_update = received_after - received_before
        if parameter.grad is None:
            torch.testing.assert_close(expected_after, expected_before, rtol=0, atol=0, msg=name)
            torch.testing.assert_close(received_after, received_before, rtol=0, atol=0, msg=name)
        else:
            changed += int(bool(torch.count_nonzero(expected_update)))
        torch.testing.assert_close(
            received_update, expected_update, rtol=rtol, atol=atol * learning_rate, msg=name
        )
    assert changed, "the qualification fixture did not update any parameter"
    return {"gradients": gradients, "updated_parameters": changed, "parameters": len(expected)}
