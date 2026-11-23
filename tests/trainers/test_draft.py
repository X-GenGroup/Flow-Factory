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

from pathlib import Path

import pytest
import torch
import yaml

from flow_factory.hparams import Arguments
from flow_factory.trainers.registry import get_trainer_class
from flow_factory.trainers.rl.draft import unroll_sampling_suffix


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_suffix_gradients_against_independent_euler_unroll(steps):
    times = torch.tensor([1.0, 0.7, 0.46, 0.19])
    initial = torch.tensor([0.6, -0.4], requires_grad=True)
    weight = torch.tensor(0.29, requires_grad=True)
    modes = []
    suffix_entry = None

    def advance(state, current, following):
        nonlocal suffix_entry
        modes.append(torch.is_grad_enabled())
        if modes[-1] and suffix_entry is None:
            suffix_entry = state.detach().requires_grad_()
            state = suffix_entry
        return state + (following - current) * (weight * state.sin() + weight.square())

    endpoint = unroll_sampling_suffix(initial, times, steps, advance)
    gradient, entry_grad = torch.autograd.grad(endpoint.square().sum(), (weight, suffix_entry))
    assert modes == [False] * (4 - steps) + [True] * steps

    x = initial.detach()
    reference_weight = weight.detach().requires_grad_()
    deltas = (-0.3, -0.24, -0.27, -0.19)
    for index, delta in enumerate(deltas):
        if index == 4 - steps:
            reference_entry = x.detach().requires_grad_()
            x = reference_entry
        x = x + delta * (reference_weight * x.sin() + reference_weight.square())
    reference_grad, reference_entry_grad = torch.autograd.grad(
        x.square().sum(), (reference_weight, reference_entry)
    )
    torch.testing.assert_close(endpoint, x)
    torch.testing.assert_close(gradient, reference_grad)
    torch.testing.assert_close(entry_grad, reference_entry_grad)
    assert initial.grad is None
    assert gradient.abs() > 0

    if steps > 1:
        isolated = unroll_sampling_suffix(
            initial, times, steps, lambda x, t, n: advance(x.detach(), t, n)
        )
        wrong_grad = torch.autograd.grad(isolated.square().sum(), weight)[0]
        assert not torch.isclose(wrong_grad, gradient, rtol=1e-3)


def test_draft_example_and_registry():
    config = Arguments.load_from_yaml("examples/draft/lora/sd3_5/pickscore.yaml")
    trainer = get_trainer_class("draft")
    assert trainer.execution_contract == config.training_args.execution_contract
    assert config.training_args.draft_backprop_steps == 2
    assert config.training_args.get_num_train_timesteps(config) == 1


@pytest.mark.parametrize("invalid", [0, True, 41])
def test_draft_admission_rejects_invalid_suffix_length(invalid):
    raw = yaml.safe_load(Path("examples/draft/lora/sd3_5/pickscore.yaml").read_text())
    raw["train"]["draft_backprop_steps"] = invalid
    with pytest.raises(ValueError, match="draft_backprop_steps"):
        Arguments.from_dict(raw)
