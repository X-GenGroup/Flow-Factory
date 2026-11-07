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

"""Corruption controls for independent native OPD projection/gradient evidence."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
from opd_objective_reference import assert_objective, assert_ode_projection
from opd_observations import observe_objectives

from flow_factory.scheduler.sde_noise import seeded_sde_transitions


def _row(normalize=True, amplitude=1.0):
    state = torch.tensor([[0.5, -1.25, 0.75, 2.0]]) * amplitude
    velocity = torch.tensor([[0.25, -0.75, 1.5, -0.5]], requires_grad=True)
    target = state + velocity * -0.5
    target.retain_grad()
    teacher = torch.tensor([[0.3, -0.5, -0.1, 1.2]]) * amplitude
    error = target.float() - teacher.float()
    scale = error.abs().mean(1).detach()
    mse = error.square().mean(1)
    per_sample = mse / (scale + 1e-8) if normalize else mse
    loss = per_sample.mean()
    (loss * 2 / 4).backward()
    return {
        "student": target.detach(),
        "teacher": teacher,
        "teacher_detached": True,
        "self_normalize": normalize,
        "gas": 4,
        "reductions": [scale, mse.detach()] if normalize else [mse.detach()],
        "loss": loss.detach(),
        "target_gradient": target.grad,
        "projection": {
            "state": state,
            "velocity": velocity.detach(),
            "target": target.detach(),
            "timestep": torch.tensor([750.0]),
            "next_timestep": torch.tensor([250.0]),
            "velocity_gradient": velocity.grad,
        },
    }


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("amplitude", [1.0, 1e-10])
def test_opd_reference_checks_real_objective_and_scheduler_chain(normalize, amplitude):
    assert_objective(_row(normalize, amplitude), cp_size=2)


@pytest.mark.parametrize(
    "field", ["student", "teacher", "loss", "target_gradient", "reductions", "velocity_gradient"]
)
def test_opd_reference_rejects_corrupted_public_endpoint(field):
    row = copy.deepcopy(_row())
    if field == "reductions":
        row[field][0] *= 1.5
    elif field == "velocity_gradient":
        row["projection"][field] *= 0.5
    else:
        row[field] *= 0.5
    with pytest.raises(AssertionError):
        assert_objective(row, cp_size=2)


@pytest.mark.parametrize("field", ["state", "velocity", "timestep", "next_timestep", "target"])
def test_opd_projection_rejects_corrupted_replay_or_teacher_output(field):
    row = copy.deepcopy(_row()["projection"])
    row[field] += 0.25
    with pytest.raises(AssertionError):
        assert_ode_projection(row)


def test_opd_observer_preserves_seeded_scheduler_signature_and_restoration():
    calls = []

    class Scheduler:
        def step(self, velocity, timestep, latents, next_latents=None, generator=None):
            calls.append(generator)
            return latents if next_latents is None else next_latents

    class Group(dict):
        @property
        def primary(self):
            return self["latent"]

    def unused(*args, **kwargs):
        raise AssertionError("passive generation observation must not call training helpers")

    scheduler = Scheduler()
    original = scheduler.step
    adapter = SimpleNamespace(
        scheduler_group=Group(latent=scheduler),
        _named_parameters={},
        use_named_parameters=unused,
        reduce_latent_values=unused,
    )
    trainer = SimpleNamespace(
        adapter=adapter,
        training_args=SimpleNamespace(teachers=[]),
        _forward_step=unused,
        _precompute_teacher_targets=unused,
        _distill=unused,
        _backward_full_output_loss=unused,
    )
    latents = torch.zeros(2, 1)
    with observe_objectives(trainer, {}):
        observed = scheduler.step
        assert inspect.signature(observed) == inspect.signature(original)
        with seeded_sde_transitions(adapter, base_seed=3, sample_ids=[4, 9]):
            assert scheduler.step(None, None, latents) is latents
            assert scheduler.step(velocity=None, timestep=None, latents=latents) is latents
        assert scheduler.step is observed
    assert scheduler.step == original
    assert len(calls) == 2 and all(len(generators) == 2 for generators in calls)
    with pytest.raises(RuntimeError, match="intentional observation exit"):
        with observe_objectives(trainer, {}):
            with seeded_sde_transitions(adapter, base_seed=3, sample_ids=[4, 9]):
                raise RuntimeError("intentional observation exit")
    assert scheduler.step == original
