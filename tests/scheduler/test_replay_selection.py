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

"""Rollout retention is independent of stochastic dynamics and re-noising times."""

from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import Arguments, SchedulerArguments
from flow_factory.models.trajectory_bridge.replay import get_train_step_indices
from flow_factory.scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    MiniMaxH3SDEScheduler,
    SchedulerGroup,
    UniPCMultistepSDEScheduler,
    load_scheduler,
)
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.utils.trajectory_collector import TrajectoryCollector, compute_trajectory_indices

SCHEDULERS = (FlowMatchEulerDiscreteSDEScheduler, UniPCMultistepSDEScheduler, MiniMaxH3SDEScheduler)


def scheduler(cls, selection="auto", dynamics="ODE", count=2):
    source = cls()
    args = SchedulerArguments(
        dynamics_type=dynamics, sde_steps=[0, 2, 4], num_sde_steps=count, replay_steps=selection
    )
    result = load_scheduler(source, args)
    result.set_timesteps(6)
    return result


@pytest.mark.parametrize("cls", SCHEDULERS)
@pytest.mark.parametrize("dynamics", ["ODE", "Flow-SDE"])
@pytest.mark.parametrize("selection", ["auto", "all", [4, 1], [], None])
def test_selection_and_storage_do_not_change_noise_or_schedule(cls, dynamics, selection):
    baseline = scheduler(cls, dynamics=dynamics)
    selected = scheduler(cls, selection, dynamics)
    torch.testing.assert_close(selected.timesteps, baseline.timesteps, rtol=0, atol=0)
    torch.testing.assert_close(selected.sigmas, baseline.sigmas, rtol=0, atol=0)
    for seed in (42, 127, 42):
        baseline.set_seed(seed)
        selected.set_seed(seed)
        torch.testing.assert_close(
            selected.current_sde_steps, baseline.current_sde_steps, rtol=0, atol=0
        )
        torch.testing.assert_close(
            selected.get_noise_levels(), baseline.get_noise_levels(), rtol=0, atol=0
        )
        expected = (
            list(range(6))
            if selection == "all" or selection == "auto" and dynamics == "ODE"
            else baseline.current_sde_steps.tolist() if selection == "auto" else selection or []
        )
        assert selected.train_timesteps.tolist() == expected
        expected_states = sorted({j for i in expected for j in (i, i + 1)})
        states = compute_trajectory_indices(selected.train_timesteps, 6)
        assert states == expected_states
        assert compute_trajectory_indices(
            selected.train_timesteps, 6, include_terminal=True
        ) == sorted(set(expected_states) | {6})
        collector = TrajectoryCollector(states, total_steps=6)
        for i in range(7):
            collector.collect(torch.tensor([i]), i)
        assert collector.collected_indices == expected_states
        if expected_states:
            index_map = collector.get_index_map()
            assert [int(index_map[i]) for i in expected_states] == list(range(len(expected_states)))


def parse(algorithm="art", selection="auto", *, training_values=None, **scheduler_values):
    return Arguments.from_dict(
        {
            "train": {
                "trainer_type": algorithm,
                "num_inference_steps": 6,
                "unique_sample_num_per_epoch": 8,
                "group_size": 2,
                "per_device_batch_size": 2,
                **(training_values or {}),
            },
            "scheduler": {
                "dynamics_type": "ODE" if algorithm == "art" else "Flow-SDE",
                "sde_steps": [0, 2, 4],
                "num_sde_steps": None,
                "replay_steps": selection,
                **scheduler_values,
            },
            "log": {"save_dir": ".scratch/art/replay-argument-saves"},
            "rewards": [{"name": "score", "reward_model": "clip"}],
        }
    )


@pytest.mark.parametrize("algorithm", ["art", "grpo", "grpo-guard", "dppo"])
def test_selected_replay_count_controls_accumulation(algorithm):
    baseline = parse(algorithm)
    selected = parse(algorithm, [4, 0])
    count = baseline.training_args.get_num_train_timesteps(baseline)
    assert selected.training_args.get_num_train_timesteps(selected) == 2
    assert (
        selected.training_args.gradient_accumulation_steps * count
        == baseline.training_args.gradient_accumulation_steps * 2
    )
    assert Arguments.from_dict(selected.to_dict()).scheduler_args.replay_steps == [4, 0]


@pytest.mark.parametrize("selection", [None, []])
@pytest.mark.parametrize("algorithm", ["art", "grpo", "dppo"])
def test_transition_objectives_require_recorded_transitions(algorithm, selection):
    with pytest.raises(ValueError, match="at least one"):
        parse(algorithm, selection)


@pytest.mark.parametrize("selection", [[6], [-1], [1, 1], [True], [0.5], "none", False])
def test_bad_replay_selection_fails_at_config_load(selection):
    with pytest.raises(ValueError, match="replay_steps"):
        parse(selection=selection)


@pytest.mark.parametrize("algorithm", ["grpo", "grpo-guard", "dppo"])
@pytest.mark.parametrize("accumulation", ["auto", 1])
@pytest.mark.parametrize(
    "selection,changes",
    [
        ([1], {}),
        ([0], {"num_sde_steps": 1}),
        ("all", {}),
        ("auto", {"dynamics_type": "ODE"}),
        ("auto", {"noise_level": 0}),
    ],
)
def test_likelihood_replay_is_validated_independently_of_accumulation(
    algorithm, accumulation, selection, changes
):
    with pytest.raises(ValueError, match="stochastic"):
        parse(
            algorithm,
            selection,
            training_values={"gradient_accumulation_steps": accumulation},
            **changes,
        )


@pytest.mark.parametrize("algorithm", ["art", "grpo", "grpo-guard", "dppo"])
@pytest.mark.parametrize("selection", [None, []])
def test_manual_accumulation_still_rejects_empty_replay(algorithm, selection):
    with pytest.raises(ValueError, match="at least one"):
        parse(algorithm, selection, training_values={"gradient_accumulation_steps": 2})


@pytest.mark.parametrize("algorithm", ["art", "grpo", "grpo-guard", "dppo"])
def test_explicit_accumulation_is_preserved_with_valid_subset(algorithm):
    config = parse(algorithm, [4, 0], training_values={"gradient_accumulation_steps": 7})
    assert config.training_args.gradient_accumulation_steps == 7
    assert config.training_args.get_num_train_timesteps(config) == 2


@pytest.mark.parametrize("accumulation", ["auto", 2])
def test_art_dynamics_are_validated_independently_of_accumulation(accumulation):
    with pytest.raises(ValueError, match="ART requires scheduler.dynamics_type='ODE'"):
        parse(
            "art",
            training_values={"gradient_accumulation_steps": accumulation},
            dynamics_type="Flow-SDE",
        )


def test_counting_is_pure_and_separate_from_objective_admission():
    from dataclasses import asdict

    args = SchedulerArguments(replay_steps=None)
    before = asdict(args)
    rng = torch.get_rng_state().clone()
    assert args.get_num_replay_steps(6) == 0
    assert asdict(args) == before
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    with pytest.raises(ValueError, match="at least one"):
        args.validate_replay_selection(6, require_stochastic=True)


@pytest.mark.parametrize("cls", SCHEDULERS)
def test_components_require_identical_replay_positions(cls):
    a = scheduler(cls, [0, 2])
    b = scheduler(cls, [0, 2])
    adapter = SimpleNamespace(
        scheduler_group=SchedulerGroup({"video": a, "audio": b}, primary_name="video")
    )
    assert get_train_step_indices(adapter).tolist() == [0, 2]
    b.replay_steps = [2, 0]
    with pytest.raises(ValueError, match="train"):
        get_train_step_indices(adapter)


@pytest.mark.parametrize("algorithm", ["nft", "dgpo", "awm"])
@pytest.mark.parametrize(
    "strategy", ["discrete", "discrete_with_init", "discrete_wo_init", "uniform", "logit_normal"]
)
def test_renoising_time_sampling_is_independent_of_replay_retention(algorithm, strategy):
    observations = []
    for selection in ("auto", "all", None, [1, 3]):
        args = parse(algorithm, selection)
        ta = args.training_args
        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            adapter=SimpleNamespace(
                scheduler=scheduler(FlowMatchEulerDiscreteSDEScheduler, selection)
            ),
            time_sampling_strategy=strategy,
            time_shift=ta.time_shift,
            timestep_range=ta.timestep_range,
            num_train_timesteps=ta.num_train_timesteps,
        )
        times = BaseTrainer._sample_timesteps(trainer, 2, torch.Generator().manual_seed(529))
        observations.append((times, ta.gradient_accumulation_steps))
    for times, accumulation in observations[1:]:
        torch.testing.assert_close(times, observations[0][0], rtol=0, atol=0)
        assert accumulation == observations[0][1]


def test_automatic_selection_preserves_legacy_scheduler_config_identity():
    assert "replay_steps" not in SchedulerArguments().to_dict()
    assert SchedulerArguments(replay_steps=None).to_dict()["replay_steps"] is None


@pytest.mark.parametrize("family", ["bagel", "sensenova", "h3"])
@pytest.mark.parametrize("selection", [None, "all", [1, 0]])
def test_custom_model_scheduler_factories_bind_replay_selection(family, selection, monkeypatch):
    from flow_factory.models.minimax_h3.workflow import build_h3_scheduler
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    args = SchedulerArguments(
        dynamics_type="ODE",
        replay_steps=list(selection) if isinstance(selection, list) else selection,
    )
    if family == "h3":
        selected = build_h3_scheduler(args, shift=12)
        audio = build_h3_scheduler(args, shift=3)
        audio.set_timesteps(6)
    else:
        from tests.models.test_bagel_tdm_contracts import _load_bagel_types

        BagelAdapter, _ = _load_bagel_types(monkeypatch)
        adapter = SimpleNamespace(config=SimpleNamespace(scheduler_args=args))
        selected = {"bagel": BagelAdapter, "sensenova": SenseNovaAdapter}[family].load_scheduler(
            adapter
        )
    selected.set_timesteps(6)
    expected = list(range(6)) if selection == "all" else list(selection or [])
    assert selected.train_timesteps.tolist() == expected
    if family == "h3":
        assert audio.train_timesteps.tolist() == expected
    if isinstance(selection, list):
        args.replay_steps.append(3)
        assert selected.train_timesteps.tolist() == expected


@pytest.mark.parametrize("cls", SCHEDULERS)
def test_direct_scheduler_construction_and_native_identity(cls):
    legacy = cls()
    selected = cls(replay_steps=[2, 0])
    selected.set_timesteps(6)
    assert selected.train_timesteps.tolist() == [2, 0]
    from flow_factory.trainers.common.runtime_identity import _scheduler_group_schema

    def identity(scheduler):
        return _scheduler_group_schema(
            SimpleNamespace(
                scheduler_group=SchedulerGroup({"latent": scheduler}, primary_name="latent")
            )
        )

    assert identity(selected) == identity(legacy)
