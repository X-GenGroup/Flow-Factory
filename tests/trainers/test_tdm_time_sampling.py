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

"""Independent conditional-distribution oracles and trajectory shift ownership."""

import math
from types import MethodType

import numpy as np
import pytest
import torch
from test_runtime_checkpoint_integration import _write_accelerate_artifacts
from test_runtime_identity import _Trainer
from test_tdm import _h3_trainer_and_sample, _objective_trainer, _sample, _trainer

from flow_factory.hparams import TDMQuerySamplingPolicy, TDMTrainingArguments
from flow_factory.hparams.training_args.tdm_r1 import TDMR1TrainingArguments
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler, SchedulerGroup
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.common.runtime_identity import build_trainer_runtime_identity
from flow_factory.trainers.common.runtime_state import TrainerRuntimeState
from flow_factory.trainers.distillation.tdm import TDMTrainer
from flow_factory.trainers.distillation.tdm_time_sampling import (
    RationalFlowShift,
    sample_interval_sigma,
    sample_query_timestep,
)


@pytest.fixture
def scipy_special():
    return pytest.importorskip("scipy.special")


@pytest.fixture
def truncnorm():
    return pytest.importorskip("scipy.stats").truncnorm


@pytest.mark.parametrize("args_cls", [TDMTrainingArguments, TDMR1TrainingArguments])
def test_sampling_defaults_and_no_separate_shift(args_cls):
    args = args_cls()
    assert args.tdm_query_distribution == "actual_uniform"
    assert (args.tdm_query_logit_mean, args.tdm_query_logit_std) == (0, 1)
    assert not hasattr(args, "tdm_time_shift")
    assert args.tdm_query_interval == "trajectory"
    assert args.tdm_query_max_sigma == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tdm_query_distribution": "uniform"},
        {"tdm_query_interval": "unknown"},
        *({"tdm_query_max_sigma": v} for v in [0, -1, 1.01, float("nan"), float("inf"), True]),
        {"tdm_query_logit_std": 0},
        {"tdm_query_logit_std": -1},
        {"tdm_query_logit_std": float("nan")},
        {"tdm_query_logit_std": float("inf")},
        {"tdm_query_logit_std": True},
        {"tdm_query_logit_mean": float("nan")},
        {"tdm_query_logit_mean": float("inf")},
        {"tdm_query_logit_mean": True},
    ],
)
def test_invalid_sampling_parameters(kwargs):
    with pytest.raises((ValueError, TypeError), match="tdm_"):
        TDMTrainingArguments(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [0, 73])
def test_actual_uniform_preserves_legacy_native_timestep_draws(dtype, seed):
    lower = torch.tensor([0.0, 500.0, 750.0], dtype=dtype)
    upper = torch.tensor([500.0, 750.0, 1000.0], dtype=dtype)
    policy = TDMQuerySamplingPolicy(
        interval="trajectory",
        distribution="actual_uniform",
        max_sigma=1.0,
        logit_mean=0.0,
        logit_std=1.0,
    )
    torch.manual_seed(seed)
    fraction = torch.rand(lower.shape, dtype=dtype)
    precision = torch.finfo(dtype)
    expected = lower + (upper - lower) * fraction.clamp(
        min=precision.eps,
        max=1.0 - precision.eps,
    )
    torch.manual_seed(seed)
    actual = sample_query_timestep(
        lower,
        upper,
        policy=policy,
        source_transform=None,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "lower,upper", [(0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1), (0.7, 0.700001)]
)
@pytest.mark.parametrize("mean,std", [(0, 1), (-2, 0.4), (3, 2), (-20, 1), (20, 1)])
def test_logit_normal_quantiles_match_independent_truncnorm_oracle(
    monkeypatch, lower, upper, mean, std, scipy_special, truncnorm
):
    fractions = torch.linspace(0.001, 0.999, 129, dtype=torch.float64)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: fractions)
    actual = sample_interval_sigma(
        torch.full_like(fractions, lower),
        torch.full_like(fractions, upper),
        distribution="conditional_logit_normal",
        logit_mean=mean,
        logit_std=std,
    )
    z_lo, z_hi = (scipy_special.logit(np.array([lower, upper])) - mean) / std
    expected = scipy_special.expit(truncnorm.ppf(fractions.numpy(), z_lo, z_hi) * std + mean)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=2e-12)
    assert bool(((actual > lower) & (actual < upper)).all())


@pytest.mark.parametrize("distribution", ["conditional_logit_normal", "source_uniform"])
@pytest.mark.parametrize("interval", ["reverse", "trajectory"])
def test_empirical_conditional_cdf_and_reproducibility(
    distribution, interval, scipy_special, truncnorm
):
    lower = torch.tensor([0, 0.5, 0.75, 0.9], dtype=torch.float64).repeat_interleave(6000)
    upper = torch.tensor([0.5, 0.75, 0.9, 1], dtype=torch.float64).repeat_interleave(6000)
    if interval == "reverse":
        upper.fill_(0.98)

    def draw():
        return sample_interval_sigma(
            lower,
            upper,
            distribution=distribution,
            logit_mean=0,
            logit_std=1,
            source_transform=RationalFlowShift(torch.full_like(lower, 3)),
        )

    torch.manual_seed(820)
    samples = draw()
    torch.manual_seed(820)
    assert torch.equal(samples, draw())
    assert not torch.equal(samples, draw())
    if distribution == "source_uniform":
        inverse = lambda value: value / (3 - 2 * value)
        cdf = ((inverse(samples) - inverse(lower)) / (inverse(upper) - inverse(lower))).numpy()
    else:
        cdf = truncnorm.cdf(
            scipy_special.logit(samples.numpy()),
            scipy_special.logit(lower.numpy()),
            scipy_special.logit(upper.numpy()),
        )
    for conditional in cdf.reshape(4, -1):
        expected = (np.arange(6000) + 0.5) / 6000
        assert np.max(np.abs(np.sort(conditional) - expected)) < 0.025
    assert bool(((samples > lower) & (samples < upper)).all())


def test_degenerate_probability_mass_fails_explicitly():
    with pytest.raises(ValueError, match="no reliable float64 probability interior"):
        sample_interval_sigma(
            torch.tensor([0.5]),
            torch.tensor([0.75]),
            distribution="conditional_logit_normal",
            logit_mean=100,
            logit_std=0.01,
        )


@pytest.mark.parametrize("distribution", ["source_uniform", "conditional_logit_normal"])
@pytest.mark.parametrize("interval", ["reverse", "trajectory"])
def test_independent_per_example_and_phase_draws(distribution, interval):
    trainer, adapter, _ = _objective_trainer()
    trainer.training_args.tdm_query_distribution = distribution
    trainer.training_args.tdm_query_interval = interval
    trainer.training_args.per_device_batch_size = 2
    samples = [_sample(), _sample()]
    trainer._record_tdm_generation_provenance(samples, 3.0)
    units = trainer._build_boundary_units(samples)
    batch = trainer._stack_replay_unit(units[0].samples)
    first = trainer._sample_score_query_times(units[0], batch).timestep["video"]
    second = trainer._sample_score_query_times(units[0], batch).timestep["video"]
    assert first[0] != first[1]
    assert not torch.equal(first, second)
    # Actual fake and generator losses each draw fresh times through the same path.
    adapter.mapped_primary_times.clear()
    assert torch.isfinite(trainer._fake_boundary_loss(units[0]))
    fake_time = adapter.mapped_primary_times[-1].clone()
    assert torch.isfinite(trainer._generator_boundary_loss(units[0]))
    assert not torch.equal(fake_time, adapter.mapped_primary_times[-1])


@pytest.mark.parametrize("dynamic", [False, True])
def test_snapshot_keeps_each_rollout_shift_after_scheduler_changes(monkeypatch, dynamic):
    trainer = _trainer(tdm_query_distribution="source_uniform")
    scheduler = FlowMatchEulerDiscreteSDEScheduler(
        shift=3, use_dynamic_shifting=dynamic, dynamics_type="ODE"
    )
    trainer.adapter.scheduler_group["video"] = scheduler

    generation_shifts = iter((3, 7))

    def generate(*args, **kwargs):
        shift = next(generation_shifts)
        if not dynamic:
            scheduler.set_shift(shift)
        scheduler.set_timesteps(sigmas=[1, 0.5], mu=math.log(shift) if dynamic else None)
        return [_sample()]

    monkeypatch.setattr(BaseTrainer, "sample_batch", generate)
    first = trainer.sample_batch({})
    second = trainer.sample_batch({})
    assert "_tdm_sampling_shift" not in first[0].extra_kwargs
    first_units = trainer._build_boundary_units(first)
    second_units = trainer._build_boundary_units(second)
    assert first_units[0].query_context.source_transform.gamma.item() == pytest.approx(3)
    assert second_units[0].query_context.source_transform.gamma.item() == pytest.approx(7)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.full(shape, 0.5, **kwargs))
    # Stored interval is (500, 1000): inverse endpoints for gamma=3 are (.25, 1).
    assert trainer._sample_perturbation_times(first_units[0]).item() == pytest.approx(
        833.333333, abs=1e-4
    )


def test_source_provenance_precedes_reward_submission_and_never_enters_sample(monkeypatch):
    trainer = _trainer(tdm_query_distribution="source_uniform")
    scheduler = FlowMatchEulerDiscreteSDEScheduler(shift=3, dynamics_type="ODE")
    trainer.adapter.scheduler_group["video"] = scheduler
    samples = [_sample()]
    events = []

    def generate(self, batch, reward_buffer=None, **kwargs):
        del self, batch, kwargs
        assert reward_buffer is None
        scheduler.set_timesteps(sigmas=[1.0, 0.5])
        events.append("generated")
        return samples

    class Buffer:
        def add_samples(self, submitted):
            assert trainer._tdm_generation_provenance_for(submitted[0]).source_shift == 3
            assert "_tdm_sampling_shift" not in submitted[0].extra_kwargs
            assert "_tdm_sampling_shift" not in submitted[0].to_dict()
            events.append("submitted")

    monkeypatch.setattr(BaseTrainer, "sample_batch", generate)
    assert trainer.sample_batch({}, reward_buffer=Buffer()) is samples
    assert events == ["generated", "submitted"]


def test_source_uniform_requires_generation_provenance():
    trainer = _trainer(tdm_query_distribution="source_uniform")
    with pytest.raises(ValueError, match="rollout-owned generation provenance"):
        trainer._build_boundary_units([_sample()])


@pytest.mark.parametrize(
    "field,value",
    [
        ("tdm_query_distribution", "conditional_logit_normal"),
        ("tdm_query_logit_mean", -0.5),
        ("tdm_query_logit_std", 2),
        ("tdm_query_interval", "trajectory"),
        ("tdm_query_max_sigma", 0.99),
        ("shift", 4),
        ("legacy", None),
    ],
)
def test_sampling_changes_reject_exact_state_resume(field, value, tmp_path):
    trainer = _Trainer()
    distribution = (
        "conditional_logit_normal"
        if field in ("tdm_query_logit_mean", "tdm_query_logit_std")
        else "source_uniform"
    )
    for name, default in dict(
        tdm_query_distribution=distribution,
        tdm_query_logit_mean=0.0,
        tdm_query_logit_std=1.0,
        tdm_query_interval="reverse",
        tdm_query_max_sigma=1.0,
    ).items():
        setattr(trainer.training_args, name, default)
    scheduler = FlowMatchEulerDiscreteSDEScheduler(shift=3)
    trainer.adapter.scheduler_group = SchedulerGroup({"latent": scheduler}, primary_name="latent")
    trainer.runtime_execution_identity_payload = MethodType(
        TDMTrainer.runtime_execution_identity_payload, trainer
    )
    original = build_trainer_runtime_identity(trainer)
    _write_accelerate_artifacts(tmp_path)
    TrainerRuntimeState(identity=original).prepare_save(tmp_path)
    if field == "shift":
        scheduler.set_shift(value)
    elif field == "legacy":
        del trainer.runtime_execution_identity_payload
        for name in (
            "tdm_query_distribution",
            "tdm_query_logit_mean",
            "tdm_query_logit_std",
            "tdm_query_interval",
            "tdm_query_max_sigma",
        ):
            delattr(trainer.training_args, name)
    else:
        setattr(trainer.training_args, field, value)
    modified = build_trainer_runtime_identity(trainer)
    with pytest.raises(ValueError, match="identity mismatch.*execution_contract_digest"):
        TrainerRuntimeState(identity=modified).validate_load(tmp_path)


@pytest.mark.parametrize(
    "distribution,interval,changes",
    [
        (
            "actual_uniform",
            "trajectory",
            {"tdm_query_logit_mean": -8.0, "tdm_query_logit_std": 9.0},
        ),
        (
            "conditional_logit_normal",
            "trajectory",
            {"tdm_query_max_sigma": 0.25},
        ),
    ],
)
def test_inactive_query_fields_do_not_change_resume_identity(distribution, interval, changes):
    trainer = _Trainer()
    for name, value in dict(
        tdm_query_distribution=distribution,
        tdm_query_logit_mean=0.0,
        tdm_query_logit_std=1.0,
        tdm_query_interval=interval,
        tdm_query_max_sigma=1.0,
    ).items():
        setattr(trainer.training_args, name, value)
    trainer.adapter.scheduler_group = SchedulerGroup(
        {"latent": FlowMatchEulerDiscreteSDEScheduler(shift=3)},
        primary_name="latent",
    )
    trainer.runtime_execution_identity_payload = MethodType(
        TDMTrainer.runtime_execution_identity_payload, trainer
    )
    before = build_trainer_runtime_identity(trainer)
    for name, value in changes.items():
        setattr(trainer.training_args, name, value)
    assert build_trainer_runtime_identity(trainer) == before


def test_dynamic_identity_does_not_depend_on_last_rollout():
    trainer = _Trainer()
    for name, value in dict(
        tdm_query_distribution="source_uniform",
        tdm_query_logit_mean=0.0,
        tdm_query_logit_std=1.0,
        tdm_query_interval="reverse",
        tdm_query_max_sigma=1.0,
    ).items():
        setattr(trainer.training_args, name, value)
    scheduler = FlowMatchEulerDiscreteSDEScheduler(use_dynamic_shifting=True)
    trainer.adapter.scheduler_group = SchedulerGroup({"latent": scheduler}, primary_name="latent")
    trainer.runtime_execution_identity_payload = MethodType(
        TDMTrainer.runtime_execution_identity_payload, trainer
    )
    before = build_trainer_runtime_identity(trainer)
    scheduler.set_timesteps(sigmas=[1, 0.5], mu=0.5)
    assert build_trainer_runtime_identity(trainer) == before
    scheduler.set_timesteps(sigmas=[1, 0.5], mu=0.9)
    assert build_trainer_runtime_identity(trainer) == before


def _four_step_sample():
    times = torch.tensor([1000.0, 900.0, 750.0, 500.0, 0.0])
    return _sample(
        state_index_map=torch.arange(5),
        video_states=torch.arange(5.0).reshape(-1, 1),
        audio_states=torch.arange(5.0).reshape(-1, 1),
        video_times=times,
        audio_times=times / 2,
    )


@pytest.mark.parametrize("interval", ["reverse", "trajectory"])
@pytest.mark.parametrize("distribution", ["source_uniform", "conditional_logit_normal"])
def test_four_step_intervals_match_conditional_oracle(
    monkeypatch, interval, distribution, scipy_special, truncnorm
):
    trainer = _trainer(
        num_inference_steps=4,
        tdm_query_interval=interval,
        tdm_query_distribution=distribution,
        tdm_query_max_sigma=0.98,
    )
    sample = _four_step_sample()
    trainer._record_tdm_generation_provenance([sample], 3.0)
    units = trainer._build_boundary_units([sample])
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.full(shape, 0.99, **kw))
    for unit, lower, stored_upper in zip(units, [0.9, 0.75, 0.5, 0], [1, 0.9, 0.75, 0.5]):
        upper = 0.98 if interval == "reverse" else stored_upper
        assert unit.interval_start.item() == pytest.approx(lower * 1000)
        assert unit.interval_end.item() == pytest.approx(stored_upper * 1000)
        assert unit.query_interval_end.item() == pytest.approx(upper * 1000)
        batch = trainer._stack_replay_unit(unit.samples)
        times = trainer._sample_score_query_times(unit, batch)
        if distribution == "source_uniform":
            inverse = lambda s: s / (3 - 2 * s)
            u = inverse(lower) + 0.99 * (inverse(upper) - inverse(lower))
            expected = 3 * u / (1 + 2 * u)
        else:
            expected = scipy_special.expit(
                truncnorm.ppf(0.99, scipy_special.logit(lower), scipy_special.logit(upper))
            )
        assert times.sigma["video"].item() == pytest.approx(expected, abs=1e-7)
        assert lower < times.sigma["video"].item() < upper
        # Secondary upper bound is adapter-mapped (490), not primary t_max (980).
        assert times.sigma["audio"].item() == pytest.approx(expected / 2, abs=1e-7)
        if interval == "reverse" and lower < 0.9:
            assert times.sigma["video"].item() > stored_upper


def test_trajectory_interval_ignores_max_sigma_and_keeps_sampling_reproducible():
    trainer = _trainer(num_inference_steps=4, tdm_query_max_sigma=0.2)
    units = trainer._build_boundary_units([_four_step_sample()])
    torch.manual_seed(43)
    before = [trainer._sample_perturbation_times(unit) for unit in units]
    trainer.training_args.tdm_query_max_sigma = 1.0
    units = trainer._build_boundary_units([_four_step_sample()])
    torch.manual_seed(43)
    after = [trainer._sample_perturbation_times(unit) for unit in units]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize("max_sigma", [0.5, 0.9])
def test_reverse_rejects_empty_intervals_before_loss(max_sigma):
    trainer = _trainer(
        num_inference_steps=4,
        tdm_query_interval="reverse",
        tdm_query_max_sigma=max_sigma,
    )
    with pytest.raises(ValueError, match="upper sigma above every lower boundary"):
        trainer._build_boundary_units([_four_step_sample()])


@pytest.mark.parametrize("max_sigma", [0.98, 1.0])
@pytest.mark.parametrize("fraction", [0.0, 1.0])
@pytest.mark.parametrize("distribution", ["source_uniform", "conditional_logit_normal"])
def test_reverse_protects_mapped_open_endpoints(monkeypatch, max_sigma, fraction, distribution):
    trainer = _trainer(
        num_inference_steps=4,
        tdm_query_interval="reverse",
        tdm_query_max_sigma=max_sigma,
        tdm_query_distribution=distribution,
    )
    sample = _four_step_sample()
    trainer._record_tdm_generation_provenance([sample], 3.0)
    units = trainer._build_boundary_units([sample])
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.full(shape, fraction, **kw))
    for unit in units:
        batch = trainer._stack_replay_unit(unit.samples)
        times = trainer._sample_score_query_times(unit, batch)
        for name in trainer.adapter.trajectory_component_order:
            assert bool((times.sigma[name] > unit.mid_times.sigma[name]).all())
            assert bool((times.sigma[name] < unit.query_bounds.upper.sigma[name]).all())


@pytest.mark.parametrize("distribution", ["source_uniform", "conditional_logit_normal"])
def test_reverse_h3_uses_mapped_cap_and_preserves_replay(monkeypatch, distribution):
    trainer, sample, adapter, video_times, audio_times = _h3_trainer_and_sample(4)
    trainer.training_args.tdm_query_interval = "reverse"
    trainer.training_args.tdm_query_distribution = distribution
    trainer.training_args.tdm_query_max_sigma = 0.98
    trainer._record_tdm_generation_provenance([sample], adapter.scheduler.shift)
    units = trainer._build_boundary_units([sample])
    cap = adapter.build_training_component_times(torch.tensor([980.0]))
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.full(shape, 0.99, **kw))
    for index, unit in enumerate(units):
        assert torch.equal(unit.mid_times.timestep["audio"], audio_times[index + 1 : index + 2])
        assert torch.equal(unit.times.timestep["video"], video_times[index : index + 1])
        assert torch.equal(unit.query_bounds.upper.sigma["audio"], cap.sigma["audio"])
        batch = trainer._stack_replay_unit(unit.samples)
        times = trainer._sample_score_query_times(unit, batch)
        for name in adapter.trajectory_component_order:
            assert bool((times.sigma[name] > unit.mid_times.sigma[name]).all())
            assert bool((times.sigma[name] < cap.sigma[name]).all())


def test_reverse_h3_six_step_production_geometry_requires_unit_cap():
    trainer, sample, adapter, _, _ = _h3_trainer_and_sample(6)
    trainer.training_args.tdm_query_interval = "reverse"
    trainer.training_args.tdm_query_distribution = "conditional_logit_normal"
    trainer.training_args.tdm_query_max_sigma = 0.98
    with pytest.raises(ValueError, match="upper sigma above every lower boundary"):
        trainer._build_boundary_units([sample])

    trainer.training_args.tdm_query_max_sigma = 1.0
    units = trainer._build_boundary_units([sample])
    assert len(units) == 6
    assert all(unit.query_context is units[0].query_context for unit in units)
    assert units[0].interval_start.item() == pytest.approx(983.6066365, abs=1e-4)
    assert units[0].query_interval_end.item() == pytest.approx(1000.0)
    cap = adapter.build_training_component_times(torch.tensor([1000.0]))
    assert torch.equal(units[0].query_bounds.upper.sigma["audio"], cap.sigma["audio"])


def test_non_source_distribution_does_not_wrap_scheduler(monkeypatch):
    trainer = _trainer()
    samples = [_sample()]
    received = []

    def generate(self, batch, reward_buffer=None, **kwargs):
        received.append((batch, reward_buffer, kwargs))
        return samples

    monkeypatch.setattr(BaseTrainer, "sample_batch", generate)
    # This scheduler fake has no set_timesteps method: no capture should be attempted.
    batch, reward_buffer = {}, object()
    assert trainer.sample_batch(batch, reward_buffer=reward_buffer, example=True) is samples
    assert received == [(batch, reward_buffer, {"example": True})]
    assert "_tdm_sampling_shift" not in samples[0].extra_kwargs
