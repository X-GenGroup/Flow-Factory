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

"""Real CRD objectives against independent scalar and analytic gradient controls."""

import copy
from types import SimpleNamespace

import pytest
import torch
from crd_objective_reference import assert_objective, objective_reference

from flow_factory.samples import LatentState, NoisedState
from flow_factory.trainers.common import context_parallel
from tests.trainers.test_dgpo_crd_parity import _crd_trainer, _structured_adapter


def _case(
    monkeypatch,
    *,
    adaptive=True,
    temperature=0.7,
    loss_type="mse",
    reward_kl=True,
    dtype=torch.float64,
    cp_size=2,
    gas=4,
    advantages=(-0.7, 0.0, 0.8),
    kl_beta=0.03,
):
    adapter = _structured_adapter()
    trainer = _crd_trainer(adapter)
    trainer.adaptive_logp, trainer.weight_temp = adaptive, temperature
    trainer.crd_loss_type, trainer.crd_beta = loss_type, 0.6
    trainer.reward_adaptive_kl, trainer.kl_beta = reward_kl, kl_beta
    trainer.accelerator = SimpleNamespace(backward=lambda loss: (loss / gas).backward())
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    target = {
        name: torch.linspace(-0.4, 0.8, 3 * size, dtype=dtype).reshape(3, size)
        for name, size in (("video", 4), ("audio", 2))
    }
    new = {
        name: (
            value + torch.linspace(0.1, 0.7, value.numel(), dtype=dtype).reshape_as(value)
        ).requires_grad_()
        for name, value in target.items()
    }
    old = {name: value + 0.3 for name, value in target.items()}
    reference = {name: value - 0.17 for name, value in new.items()}
    reference = {name: value.detach() for name, value in reference.items()}
    masks = {
        "video": torch.tensor(
            [[True, True, False, True], [True, False, True, False], [False, True, True, True]]
        ),
        "audio": torch.tensor([[True, False], [True, True], [False, True]]),
    }
    noised = NoisedState(
        state=LatentState(target, active_masks=masks),
        noise=LatentState(target),
        target_velocity=LatentState(target),
    )
    velocity = LatentState(new)
    implicit = trainer._implicit_reward(velocity, LatentState(old), noised)
    adv = torch.tensor(advantages, dtype=torch.float32)
    policy = trainer._compute_crd_loss(adv, adv, implicit.detach(), implicit) * 5 / trainer.crd_beta
    kl = trainer._velocity_kl(velocity, LatentState(reference), noised)
    loss = policy + trainer._kl_loss(kl, adv)
    trainer._backward_full_output_loss(loss)
    row = dict(
        new={name: value.detach() for name, value in new.items()},
        old=old,
        reference=reference,
        target=target,
        masks=masks,
        advantage=adv,
        adaptive_logp=adaptive,
        weight_temp=temperature,
        crd_loss_type=loss_type,
        crd_beta=trainer.crd_beta,
        adv_clip_max=5.0,
        reward_adaptive_kl=reward_kl,
        kl_beta=kl_beta,
        kl_dtype=kl.dtype,
        cp_size=cp_size,
        gas=gas,
        loss=loss.detach(),
        implicit_reward=implicit.detach(),
        kl_div=kl.detach(),
        gradients={name: value.grad for name, value in new.items()},
    )
    return row


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("temperature", [-1.0, 0.0, 0.7])
@pytest.mark.parametrize("loss_type", ["mse", "bce"])
@pytest.mark.parametrize("reward_kl", [False, True])
@pytest.mark.parametrize("cp_size,gas", [(1, 1), (2, 4)])
def test_actual_crd_masked_objective_and_local_derivatives(
    monkeypatch, adaptive, temperature, loss_type, reward_kl, cp_size, gas
):
    assert_objective(
        _case(
            monkeypatch,
            adaptive=adaptive,
            temperature=temperature,
            loss_type=loss_type,
            reward_kl=reward_kl,
            cp_size=cp_size,
            gas=gas,
        )
    )


@pytest.mark.parametrize("advantages", [(0.0, 0.0, 0.0), (0.0, 0.2, 1.0), (-0.7, -0.2, 0.0)])
def test_hard_weighting_empty_pool_and_production_normalized_semantics(monkeypatch, advantages):
    assert_objective(_case(monkeypatch, temperature=0, advantages=advantages))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("adaptive", [False, True])
def test_native_residual_square_rounding_is_preserved(monkeypatch, dtype, adaptive):
    assert_objective(_case(monkeypatch, dtype=dtype, adaptive=adaptive, reward_kl=False))


def test_wrong_scaling_missing_component_and_differentiated_constants_are_rejected(monkeypatch):
    row = _case(monkeypatch)
    for factor in (0, 0.5, 2, -1):
        bad = copy.deepcopy(row)
        for gradient in bad["gradients"].values():
            gradient.mul_(factor)
        with pytest.raises(AssertionError):
            assert_objective(bad)
    bad = copy.deepcopy(row)
    del bad["new"]["audio"]
    with pytest.raises(AssertionError):
        assert_objective(bad)
    bad = copy.deepcopy(row)
    bad["target"]["video"] += 0.2
    with pytest.raises(AssertionError):
        assert_objective(bad)
    # If the MAD were differentiated, d(e^2/MAD)/de loses a term.
    bad = copy.deepcopy(row)
    expected = objective_reference(row)
    for name in bad["gradients"]:
        bad["gradients"][name] -= (
            expected["policy_gradients"][name].to(bad["gradients"][name].dtype) / 2
        )
    with pytest.raises(AssertionError):
        assert_objective(bad)


def test_differentiating_nonuniform_gathered_centers_changes_the_local_gradient(monkeypatch):
    row = _case(monkeypatch, temperature=0.7, loss_type="mse", reward_kl=False)
    expected = objective_reference(row)
    reward, advantage = expected["implicit_reward"], row["advantage"].double()
    correction = torch.zeros_like(reward)
    for sign in (1, -1):
        weights = (sign * advantage / 0.7).softmax(0)
        delta = row["crd_beta"] * (reward - (reward * weights).sum()) - (
            advantage - (advantage * weights).sum()
        )
        local_upstream = 2 * row["crd_beta"] * delta / len(reward) / 2
        local_upstream *= row["adv_clip_max"] / row["crd_beta"] * row["cp_size"] / row["gas"]
        correction -= weights * local_upstream.sum()
    assert correction.abs().max() > 1e-3
    total = sum(mask.sum(1) for mask in row["masks"].values()).double()
    bad = copy.deepcopy(row)
    for name, value in row["new"].items():
        residual = value.double() - row["target"][name].double()
        mask = row["masks"][name]
        mad = (residual.abs() * mask).sum(1) / mask.sum(1)
        erroneous_extra = -2 * residual * (correction / total / mad)[:, None] * mask
        bad["gradients"][name] += erroneous_extra
    with pytest.raises(AssertionError):
        assert_objective(bad)


@pytest.mark.parametrize("reward_kl", [False, True])
def test_zero_reference_kl_preserves_policy_only_derivatives(monkeypatch, reward_kl):
    row = _case(monkeypatch, kl_beta=0.0, reward_kl=reward_kl)
    expected = assert_objective(row)
    assert expected["kl_loss"] == 0
    for name, gradient in expected["gradients"].items():
        torch.testing.assert_close(gradient, expected["policy_gradients"][name], rtol=0, atol=0)
