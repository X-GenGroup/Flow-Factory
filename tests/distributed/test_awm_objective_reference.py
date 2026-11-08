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


"""Actual AWM optimize against independent matching, ratio and snapshot derivatives."""

import copy
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from awm_objective_reference import assert_objective, matching
from awm_observations import observe_objectives

from flow_factory.samples import BaseSample, LatentState, MultiModalStepOutput
from flow_factory.trainers.common import context_parallel
from tests.trainers.test_decoupled_trainer_parity import (
    _adapter,
    _awm_trainer,
    _noised,
    _structured_adapter,
)


def _fixture(
    weighting="ghuber",
    off_policy=True,
    cp_size=2,
    gas=4,
    betas=(0.03, 0.02),
    shared=False,
    velocity_dtype=torch.float64,
    kl_compute_dtype=None,
):
    shape = (5, 3, 4)
    parameter = torch.nn.Parameter(
        torch.linspace(-0.4, 0.6, 60, dtype=torch.float64).reshape(shape)
    )
    clean = torch.linspace(-1.2, 1.6, 60, dtype=torch.float64).reshape(shape)
    advantages = [-8.0, -2.0, 0.0, 2.0, 8.0]
    samples = [
        BaseSample(
            timesteps=torch.tensor([1000.0, 500.0]),
            all_latents=value[None],
            latent_index_map=torch.tensor([-1, -1, 0]),
            prompt_embeds=torch.tensor([i + 1.0]),
            sampling_sample_id=i,
            sampling_group_id=0,
            sampling_group_member_id=i,
            extra_kwargs={"advantage": torch.tensor(adv, dtype=torch.float32)},
        )
        for i, (value, adv) in enumerate(zip(clean, advantages))
    ]
    batch = BaseSample.stack(samples)
    adapter = _adapter()
    trainer = _awm_trainer(adapter, weighting=weighting)
    trainer.off_policy = off_policy
    trainer.kl_beta, trainer.ema_kl_beta = betas
    trainer.training_args.update(
        trainer_type="awm",
        per_device_batch_size=5,
        num_inner_epochs=1,
        kl_beta=betas[0],
        ema_kl_beta=betas[1],
        adv_clip_range=(-5.0, 5.0),
        clip_range=(-0.001, 0.001),
        gradient_accumulation_steps=gas,
        use_shared_noise=shared,
    )
    trainer.epoch = trainer.step = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.logger = None
    trainer.config = SimpleNamespace(parallelism_args=SimpleNamespace(cp_size=cp_size))
    trainer.autocast = trainer.accumulate_gradients = nullcontext
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        sync_gradients=False,
        gradient_accumulation_steps=gas,
        backward=lambda loss: (loss / gas).backward(),
    )
    trainer._order_samples_for_optimize = lambda samples, inner_epoch: samples
    trainer._iter_prefetched_batches = lambda samples, size: iter([batch])
    snapshots = {
        name: torch.nn.Parameter(parameter.detach().clone() + offset)
        for name, offset in (
            ("ema", 0.4 * torch.tensor([1, -1, 1, -1, 1]).reshape(5, 1, 1)),
            ("reference", -0.3),
        )
    }
    scopes, forwards = [], []
    active = "live"

    @contextmanager
    def scope(name):
        nonlocal active
        before, active = active, name
        scopes.append(name)
        try:
            yield
        finally:
            active = before

    adapter.use_ema_parameters = lambda: scope("ema")
    adapter.use_ref_parameters = lambda: scope("reference")

    def forward_state(**kwargs):
        assert kwargs["compute_log_prob"] is False
        state = kwargs["state"]
        theta = parameter if active == "live" else snapshots[active]
        velocity = (theta + 0.13 * state.components["latent"]).to(velocity_dtype)
        forwards.append(
            {
                "scope": active,
                "grad": torch.is_grad_enabled(),
                "state": state,
                "velocity": velocity.detach().clone(),
            }
        )
        return MultiModalStepOutput(velocity=LatentState({"latent": velocity}))

    adapter.forward_state = forward_state
    if kl_compute_dtype is not None:

        def autocast_kl(new, other, noised):
            values = {
                name: (value - other.components[name]).to(kl_compute_dtype).square()
                for name, value in new.components.items()
            }
            return adapter.reduce_latent_values(values, state=noised.state)

        trainer._velocity_kl = autocast_kl
    return trainer, parameter, samples, scopes, forwards, snapshots


@pytest.mark.parametrize("weighting", ["Uniform", "t", "t**2", "huber", "ghuber"])
@pytest.mark.parametrize("off_policy", [False, True])
@pytest.mark.parametrize("cp_size,gas", [(1, 1), (2, 4)])
@pytest.mark.parametrize("betas", [(0.0, 0.0), (0.03, 0.0), (0.0, 0.02), (0.03, 0.02)])
def test_real_awm_optimize_matches_independent_derivatives(
    monkeypatch, weighting, off_policy, cp_size, gas, betas
):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    trainer, parameter, samples, scopes, forwards, snapshots = _fixture(
        weighting, off_policy, cp_size, gas, betas
    )
    before = {name: value.detach().clone() for name, value in snapshots.items()}
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    assert len(rows) == 2
    expected = sum(row["oracle"]["gradients"]["latent"] for row in rows)
    torch.testing.assert_close(parameter.grad, expected, rtol=2e-6, atol=2e-8)
    assert [row["scope"] for row in forwards[:2]] == ["ema" if off_policy else "live"] * 2
    assert all(not row["grad"] for row in forwards[:2])
    assert all(row["same_noised_inputs"] for row in rows)
    assert trainer.adapter.noise_calls == 2
    for name, value in snapshots.items():
        assert value.grad is None and torch.equal(value.detach(), before[name])
    for row in rows:
        assert_objective(row)
    assert not torch.equal(rows[0]["state"]["latent"], rows[1]["state"]["latent"])


def test_awm_clipped_and_unclipped_signs_and_corruption(monkeypatch):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, _, samples, *_ = _fixture()
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    active = torch.cat([row["oracle"]["active"] for row in rows])
    assert active.any() and (~active).any()
    for factor in (0.0, 0.5, 2.0, -1.0):
        bad = copy.deepcopy(rows[0])
        bad["gradients"]["latent"] *= factor
        with pytest.raises(AssertionError):
            assert_objective(bad)
    bad = copy.deepcopy(rows[0])
    bad["old_log_prob"] += 0.2
    with pytest.raises(AssertionError):
        assert_objective(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_public_operand_rounding_with_autocast_kl_precision(monkeypatch, dtype):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, _, samples, *_ = _fixture(velocity_dtype=dtype, kl_compute_dtype=torch.float32)
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    assert len(rows) == 2
    for row in rows:
        assert_objective(row)


@pytest.mark.parametrize("weighting", ["Uniform", "t", "t**2", "huber", "ghuber"])
def test_component_sigmas_and_active_masks_have_independent_derivatives(weighting):
    trainer = _awm_trainer(_structured_adapter(), weighting=weighting)
    clean = {
        "video": torch.tensor([[1.0, 2.0, 9.0]], dtype=torch.float64),
        "audio": torch.tensor([[3.0, 4.0]], dtype=torch.float64),
    }
    noise = {name: value + 0.3 for name, value in clean.items()}
    noised, times = _noised(
        clean,
        noise,
        {
            "video": torch.tensor([0.7], dtype=torch.float64),
            "audio": torch.tensor([0.3], dtype=torch.float64),
        },
    )
    noised.state.active_masks = {
        "video": torch.tensor([[True, True, False]]),
        "audio": torch.tensor([[True, True]]),
    }
    values = {
        "video": torch.tensor([[0.1, 0.2, 100.0]], dtype=torch.float64, requires_grad=True),
        "audio": torch.tensor([[0.6, -0.4]], dtype=torch.float64, requires_grad=True),
    }
    actual = trainer._matching_log_prob(LatentState(values), noised, times)
    row = {
        "target": noised.target_velocity.components,
        "sigma": times.sigma,
        "masks": noised.state.active_masks,
        "weighting": weighting,
        "ghuber_power": 0.25,
    }
    expected, slopes, _, _ = matching(values, row)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0)
    actual.sum().backward()
    for name, value in values.items():
        torch.testing.assert_close(value.grad, slopes[name], rtol=2e-12, atol=2e-12)
    assert values["video"].grad[0, 2] == 0


def test_same_shared_noise_and_time_reused_by_old_live_reference_and_ema(monkeypatch):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, _, samples, _, forwards, _ = _fixture(shared=True)
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    assert len(forwards) == 8
    assert trainer.adapter.noise_calls == 0
    for index in range(2):
        state = forwards[index]["state"]
        assert sum(item["state"] is state for item in forwards) == 4
        noise = rows[index]["noise"]["latent"]
        assert all(torch.equal(value, noise[0]) for value in noise)
    assert not torch.equal(rows[0]["noise"]["latent"], rows[1]["noise"]["latent"])


def test_sampling_scope_restores_after_exception():
    trainer, _, samples, scopes, forwards, _ = _fixture()
    with pytest.raises(RuntimeError, match="intentional"):
        with trainer.sampling_context():
            raise RuntimeError("intentional")
    trainer.off_policy = False
    batch = BaseSample.stack(samples)
    trainer._precompute_old_log_probs(
        batch, trainer.adapter.get_terminal_state(batch), len(samples)
    )
    assert scopes == ["ema"] and all(row["scope"] == "live" for row in forwards)


@pytest.mark.parametrize(
    "query,operand", [(1, "state"), (3, "state"), (4, "sigma"), (5, "timestep")]
)
def test_passive_observer_rejects_in_place_query_input_mutation(monkeypatch, query, operand):
    from flow_factory.trainers.rl import awm

    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, _, samples, *_ = _fixture()
    original = awm.forward_velocity_state
    calls = 0

    def corrupt(owner, batch, state, times, **kwargs):
        nonlocal calls
        calls += 1
        result = original(owner, batch, state, times, **kwargs)
        if calls == query:
            value = (
                state.components["latent"]
                if operand == "state"
                else getattr(times, operand)["latent"]
            )
            value[(0,) * value.ndim].add_(0.125)
        return result

    monkeypatch.setattr(awm, "forward_velocity_state", corrupt)
    with pytest.raises(AssertionError, match="AWM noised input changed in place"):
        with observe_objectives(trainer):
            trainer.optimize(samples)
