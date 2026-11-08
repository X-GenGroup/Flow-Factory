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

"""CPU controls for the independent objective reference, independent of model kernels."""

import copy

import pytest
import torch
from dmd2_objective_reference import (
    direction_difference,
    objective_negative_controls,
    objective_reference,
)


def latent(x):
    return {"components": {"latent": x.detach().clone()}, "active_masks": None}


def generator_event(cp, mode="ordinary", defect=None):
    x = torch.linspace(-0.02, 0.03, 128).reshape(1, 4, 2, 4, 4)
    velocity = torch.nn.Parameter(x.bfloat16())
    sigma = 0.5
    gen = 0.1 - sigma * velocity.float()
    if mode == "near_clamp":
        real = gen.detach() + 2e-7
        fake = real + 2e-5
    elif mode == "cancellation":
        real = gen.detach() + 0.02
        fake = real + torch.linspace(-1e-4, 1e-4, 128).reshape_as(gen)
    else:
        real = gen.detach() + 0.04
        fake = real + torch.linspace(-0.03, 0.05, 128).reshape_as(gen)
    normalizer = (gen - real).abs().mean().clamp_min(1e-6)
    if defect == "normalizer":
        normalizer = normalizer * 2
    direction = (fake - real) / normalizer
    if defect == "score_sign":
        direction = -direction
    target = gen - direction
    if defect != "leaked_target":
        target = target.detach()
    gen.retain_grad()
    loss = 0.5 * (gen - target).square().mean()
    (loss * cp / 2).backward()
    return {
        "role": "generator",
        "objective": {
            "gen": latent(gen),
            "real": latent(real),
            "fake": latent(fake),
            "upstream": {"latent": gen.grad.detach().clone()},
        },
        "replay_step": {"times": {"sigma": {"latent": torch.tensor([sigma])}}},
        "modules": [
            {
                "name": "members.transformer",
                "stage": "generator_clean_replay",
                "dy": velocity.grad.detach().clone(),
            }
        ],
    }


@pytest.mark.parametrize("cp", [1, 2])
@pytest.mark.parametrize("mode", ["ordinary", "near_clamp", "cancellation"])
def test_independent_upstream_and_corruption(cp, mode):
    event = generator_event(cp, mode)
    ref = objective_reference(event, cp)
    assert ref["latent_pass"] and ref["root_pass"]
    assert all(objective_negative_controls(event, cp).values())


@pytest.mark.parametrize("defect", ["normalizer", "score_sign", "leaked_target"])
def test_wrong_algorithm_is_rejected(defect):
    ref = objective_reference(generator_event(2, defect=defect), 2)
    assert not ref["latent_pass"]


def test_forward_only_direction_bound():
    right = generator_event(1, "cancellation")
    left = copy.deepcopy(right)
    left["objective"]["fake"]["components"]["latent"] += torch.linspace(-2e-6, 3e-6, 128).reshape(
        1, 4, 2, 4, 4
    )
    left["objective"]["gen"]["components"]["latent"] += 3e-6
    assert direction_difference(left, right)["pass"]
