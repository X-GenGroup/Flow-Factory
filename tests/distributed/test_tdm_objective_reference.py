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

"""Masked independent TDM losses and CP/GAS analytic gradient controls."""

from types import SimpleNamespace

import pytest
import torch
from tdm_objective_reference import assert_objective, objective_reference

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import LatentState
from flow_factory.trainers.common import context_parallel
from flow_factory.trainers.distillation.distribution_matching import (
    tdm_fake_loss,
    tdm_generator_loss,
)
from flow_factory.trainers.distillation.tdm import TDMTrainer


class _Adapter(BaseAdapter):
    trajectory_component_order = ("video", "audio")

    def load_pipeline(self):
        raise NotImplementedError

    def decode_latents(self, latents, **kwargs):
        return latents

    def inference(self, **kwargs):
        return []

    def forward(self, **kwargs):
        raise NotImplementedError


def _operands(kind, huber):
    generator = torch.Generator().manual_seed(723)
    live = {
        "video": torch.randn(2, 3, 4, generator=generator),
        "audio": torch.randn(2, 5, generator=generator),
    }
    masks = {
        "video": torch.tensor([[[True, False, True, True]], [[False, True, True, False]]]),
        "audio": torch.tensor([[True, True, False, True, False], [True, False, True, True, True]]),
    }
    row = {"kind": kind, "live": live, "masks": masks, "gas": 4}
    if kind == "fake":
        row.update(
            target={
                name: torch.randn(value.shape, generator=generator) for name, value in live.items()
            },
            sigma=torch.tensor([0.15, 0.8]),
            importance=torch.tensor([0.35, 2.5]),
            snr_gamma=5.0,
        )
    else:
        row.update(
            real={
                name: torch.randn(value.shape, generator=generator) for name, value in live.items()
            },
            fake={
                name: torch.randn(value.shape, generator=generator) for name, value in live.items()
            },
            use_huber=huber,
            huber_c=0.13,
        )
    return row


@pytest.mark.parametrize("kind,huber", [("fake", False), ("generator", False), ("generator", True)])
@pytest.mark.parametrize("degree", [1, 2])
def test_independent_masked_cp_partition_and_gas(monkeypatch, kind, huber, degree):
    adapter = object.__new__(_Adapter)
    row = _operands(kind, huber)
    expected_loss, expected_gradients = objective_reference(row)
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (degree, 0))
    all_gradients = []
    for peer in range(degree):
        live = {name: value.clone().requires_grad_() for name, value in row["live"].items()}
        partitioned = {}
        for name, value in live.items():
            mask = torch.arange(value.numel()).reshape(value.shape) % degree == peer
            partitioned[name] = torch.where(mask, value, value.detach())
        state = LatentState(partitioned, active_masks=row["masks"])
        if kind == "fake":
            loss = tdm_fake_loss(
                adapter,
                state,
                LatentState(row["target"], active_masks=row["masks"]),
                sigma=row["sigma"],
                importance=row["importance"],
                snr_gamma=row["snr_gamma"],
            )
        else:
            loss = tdm_generator_loss(
                adapter,
                state,
                LatentState(row["real"], active_masks=row["masks"]),
                LatentState(row["fake"], active_masks=row["masks"]),
                use_huber=huber,
                huber_c=row["huber_c"],
            )
        torch.testing.assert_close(loss.detach().double(), expected_loss, rtol=8e-6, atol=1e-7)
        trainer = object.__new__(TDMTrainer)
        trainer.accelerator = SimpleNamespace(
            backward=lambda value: (value / row["gas"]).backward()
        )
        # Use the public role backward entry, preserving CP normalization once.
        trainer.role_optimization = SimpleNamespace(backward=trainer.accelerator.backward)
        trainer._backward_full_output_loss(loss, role=True)
        all_gradients.append({name: value.grad.double() for name, value in live.items()})
    for name, expected in expected_gradients.items():
        actual = sum(peer[name] for peer in all_gradients) / degree
        torch.testing.assert_close(actual, expected / row["gas"], rtol=8e-6, atol=1e-8)
        assert torch.count_nonzero(expected)
        for wrong in (-1, 0, 0.5, 2, row["gas"]):
            assert not torch.allclose(actual, expected / row["gas"] * wrong, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("kind,huber", [("fake", False), ("generator", False), ("generator", True)])
def test_native_oracle_rejects_loss_and_derivative_corruption(kind, huber):
    row = _operands(kind, huber)
    loss, gradients = objective_reference(row)
    row.update(
        loss=loss.float(),
        gradients={
            name: (gradient * 2 / row["gas"]).float() for name, gradient in gradients.items()
        },
    )
    assert_objective(row, 2)
    for wrong in (-1, 0, 0.5, 2):
        corrupt = {
            **row,
            "gradients": {name: value * wrong for name, value in row["gradients"].items()},
        }
        with pytest.raises(AssertionError):
            assert_objective(corrupt, 2)
    with pytest.raises(AssertionError):
        assert_objective({**row, "loss": row["loss"] + 0.1}, 2)


def test_tdm_admission_does_not_admit_subclass():
    assert (
        context_parallel.context_parallel_training_capability(TDMTrainer).loss_gradient
        == "full_output"
    )

    class Unqualified(TDMTrainer):
        pass

    with pytest.raises(ValueError, match="explicitly declare"):
        context_parallel.context_parallel_training_capability(Unqualified)


def test_conditional_rng_and_projection_oracles_reject_corruption():
    from tdm_objective_reference import assert_conditional_renoise, assert_projection

    from flow_factory.samples import ComponentTimes
    from flow_factory.trainers.distillation.distribution_matching import tdm_conditional_renoise

    adapter = object.__new__(_Adapter)
    operands = _operands("generator", True)
    clean, implied = LatentState(operands["live"]), LatentState(operands["fake"])
    mid = {name: torch.tensor([0.1, 0.4]) for name in clean.components}
    target = {name: torch.tensor([0.35, 0.85]) for name in clean.components}

    def times(sigmas):
        return ComponentTimes(
            timestep={name: value * 1000 for name, value in sigmas.items()},
            next_timestep={name: torch.zeros_like(value) for name, value in sigmas.items()},
            sigma=sigmas,
        )

    generator = torch.Generator().manual_seed(931)
    clone = torch.Generator().set_state(generator.get_state())
    fresh = {
        name: torch.randn(value.shape, generator=clone) for name, value in clean.components.items()
    }
    noised, importance = tdm_conditional_renoise(
        adapter,
        clean,
        implied,
        mid_times=times(mid),
        target_times=times(target),
        generator=generator,
        importance_clip=2.0,
    )
    assert torch.equal(generator.get_state(), clone.get_state())
    row = {
        "clean": clean.components,
        "implied": implied.components,
        "mid": mid,
        "target_sigma": target,
        "fresh": fresh,
        "importance_clip": 2.0,
        "noise": noised.noise.components,
        "state": noised.state.components,
        "velocity_target": noised.target_velocity.components,
        "importance": importance,
    }
    assert_conditional_renoise(row)
    for key in ("noise", "state", "velocity_target", "fresh", "implied"):
        corrupted = {**row, key: {name: value + 0.03 for name, value in row[key].items()}}
        with pytest.raises(AssertionError):
            assert_conditional_renoise(corrupted)
    with pytest.raises(AssertionError):
        assert_conditional_renoise({**row, "importance": importance * 1.1})
    projected = adapter.project_velocity_to_clean_state(noised.state, times(target), implied)
    projection = {
        "state": noised.state.components,
        "sigma": target,
        "velocity": implied.components,
        "clean": projected.components,
    }
    assert_projection(projection)
    with pytest.raises(AssertionError):
        assert_projection(
            {**projection, "velocity": {name: -value for name, value in implied.components.items()}}
        )


def test_huber_near_zero_preserves_public_fp32_cancellation():
    adapter = object.__new__(_Adapter)
    live = {
        "video": torch.full((1, 2, 3), 0.125, requires_grad=True),
        "audio": torch.full((1, 5), 0.25, requires_grad=True),
    }
    real = {name: value.detach() + 1e-4 for name, value in live.items()}
    fake = {name: value + 2e-5 for name, value in real.items()}
    row = {
        "kind": "generator",
        "live": {name: value.detach() for name, value in live.items()},
        "real": real,
        "fake": fake,
        "use_huber": True,
        "huber_c": 0.1,
        "gas": 4,
    }
    loss = tdm_generator_loss(
        adapter,
        LatentState(live),
        LatentState(real),
        LatentState(fake),
        use_huber=True,
        huber_c=0.1,
    )
    (loss * 2 / row["gas"]).backward()
    row.update(loss=loss.detach(), gradients={name: value.grad for name, value in live.items()})
    assert_objective(row, 2)
    # The unrounded mathematical pseudo-Huber scalar differs materially here;
    # matching it would misdiagnose valid FP32 execution as an algorithm error.
    delta = 2e-5
    ideal = ((delta**2 + 0.1**2) ** 0.5 - 0.1) / 1e-4
    assert abs(float(loss) - ideal) > 1e-6
