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

"""Negative checks for numerical qualification evidence."""

import copy

import pytest
import torch
from tp_qualification import assert_optimizer_step_parity, parameters_by_name


def test_expected_zero_gradient_is_not_omitted_from_the_oracle() -> None:
    reference = torch.nn.Linear(2, 2)
    actual = copy.deepcopy(reference)
    reference.weight.grad = torch.zeros_like(reference.weight)
    actual.weight.grad = torch.ones_like(actual.weight)
    with pytest.raises(AssertionError):
        assert_optimizer_step_parity(reference, actual, rtol=1e-6, atol=1e-8)


def test_missing_gradient_is_an_error_before_stepping() -> None:
    reference = torch.nn.Linear(2, 2)
    actual = copy.deepcopy(reference)
    reference.weight.grad = torch.ones_like(reference.weight)
    with pytest.raises(AssertionError, match="gradient membership"):
        assert_optimizer_step_parity(reference, actual, rtol=1e-6, atol=1e-8)


def test_collective_oracle_order_is_independent_of_registration_order() -> None:
    first = torch.nn.ModuleDict({"z": torch.nn.Linear(2, 2), "a": torch.nn.Linear(2, 2)})
    second = torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2), "z": torch.nn.Linear(2, 2)})
    assert list(parameters_by_name(first)) == list(parameters_by_name(second))
    assert list(parameters_by_name(first)) == ["a.bias", "a.weight", "z.bias", "z.weight"]


def test_checkpoint_wrapper_keeps_canonical_parameter_membership() -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    expected = parameters_by_name(model)
    model[0] = checkpoint_wrapper(model[0])
    assert parameters_by_name(model) == expected
