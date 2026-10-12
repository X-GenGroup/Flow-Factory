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

from dataclasses import fields
from types import SimpleNamespace

import pytest

from flow_factory.hparams import (
    AdamWOptimizerArguments,
    Arguments,
    DMD2TrainingArguments,
    MultiOptimizerArguments,
    TDMR1TrainingArguments,
    TDMTrainingArguments,
    get_training_args_class,
)
from flow_factory.hparams.training_args.dmd2 import DMD2_DEFAULT_OPTIMIZERS
from flow_factory.hparams.training_args.tdm_r1 import TDM_R1_DEFAULT_OPTIMIZERS
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.distillation.dmd2 import DMD2Trainer
from flow_factory.trainers.distillation.tdm import TDMTrainer
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer
from flow_factory.trainers.registry import list_registered_trainers

_MULTIROLE_TRAINERS = {"dmd2": DMD2Trainer, "tdm": TDMTrainer, "tdm-r1": TDMR1Trainer}


def _parse_train(
    trainer_type: str,
    *,
    data_overrides: dict | None = None,
    train_overrides: dict | None = None,
    scheduler_overrides: dict | None = None,
    rewards: list[dict] | None = None,
    optimizers: list[dict] | None = None,
) -> Arguments:
    train = {
        "trainer_type": trainer_type,
        "unique_sample_num_per_epoch": 8,
        "per_device_batch_size": 2 if trainer_type == "tdm-r1" else 1,
        "gradient_step_per_epoch": 1 if trainer_type in ("dmd2", "tdm", "tdm-r1") else 2,
    }
    train.update(train_overrides or {})
    config = {
        "data": data_overrides or {},
        "train": train,
        "scheduler": {"dynamics_type": "ODE", **(scheduler_overrides or {})},
    }
    if rewards is not None:
        config["rewards"] = rewards
    if optimizers is not None:
        config["optimizers"] = optimizers
    return Arguments.from_dict(config)


def _parse_multirole(
    trainer_type: str, *, optimizers: list[dict] | None = None, **train_overrides
) -> Arguments:
    """Parse a config that satisfies each multi-role algorithm's batch and reward contract."""
    rewards = None
    if trainer_type == "tdm-r1":
        train_overrides = {"group_size": 2, **train_overrides}
        rewards = [{"name": "score", "reward_model": "clip"}]
    return _parse_train(
        trainer_type, train_overrides=train_overrides, rewards=rewards, optimizers=optimizers
    )


@pytest.mark.parametrize(
    ("trainer_type", "expected_class"),
    [
        ("dmd2", DMD2TrainingArguments),
        ("tdm", TDMTrainingArguments),
        ("tdm-r1", TDMR1TrainingArguments),
    ],
)
def test_training_arguments_registry_resolves_multirole_classes(
    trainer_type: str,
    expected_class: type,
) -> None:
    assert get_training_args_class(trainer_type) is expected_class
    kwargs = {}
    if trainer_type == "tdm-r1":
        kwargs = {
            "train_overrides": {"group_size": 2},
            "rewards": [{"name": "score", "reward_model": "clip"}],
        }
    assert isinstance(_parse_train(trainer_type, **kwargs).training_args, expected_class)


def test_multirole_defaults_and_frozen_reference_only_surface() -> None:
    dmd2 = DMD2TrainingArguments()
    tdm = TDMTrainingArguments()
    tdm_r1 = TDMR1TrainingArguments()

    assert {config.name: config.learning_rate for config in DMD2_DEFAULT_OPTIMIZERS} == {
        "generator": 1e-5,
        "fake": 1e-5,
    }
    assert {config.name: config.update_frequency for config in DMD2_DEFAULT_OPTIMIZERS} == {
        "generator": 1,
        "fake": 5,
    }
    assert dmd2.default_role_optimizers() is DMD2_DEFAULT_OPTIMIZERS
    assert "ttur_fake_updates" not in {field.name for field in fields(DMD2TrainingArguments)}
    assert dmd2.gradient_step_per_epoch == 1
    assert dmd2.perturbation_timestep_range == (0.02, 0.98)

    assert tdm.default_role_optimizers() is DMD2_DEFAULT_OPTIMIZERS
    assert tdm.gradient_step_per_epoch == 1
    assert tdm.use_huber is True
    assert tdm.num_inference_steps == 4
    assert tdm.get_num_train_timesteps(None) == 4
    assert tdm.replay_rtol == 1e-4
    assert tdm.replay_atol == 1e-4

    tdm_r1_defaults = {config.name: config for config in TDM_R1_DEFAULT_OPTIMIZERS}
    assert tdm_r1.default_role_optimizers() is TDM_R1_DEFAULT_OPTIMIZERS
    assert {name: config.update_frequency for name, config in tdm_r1_defaults.items()} == {
        "generator": 1,
        "fake": 5,
        "surrogate": 1,
    }
    assert tdm_r1_defaults["generator"].learning_rate == 7.5e-5
    assert tdm_r1_defaults["generator"].betas == (0.0, 0.999)
    assert tdm_r1_defaults["fake"].learning_rate == 3e-4
    assert tdm_r1_defaults["fake"].betas == (0.0, 0.999)
    assert tdm_r1_defaults["surrogate"].learning_rate == 3e-4
    assert tdm_r1_defaults["surrogate"].betas == (0.9, 0.999)
    assert tdm_r1_defaults["generator"].learning_rate == tdm_r1_defaults["fake"].learning_rate / 4
    assert tdm_r1.advantage_aggregation == "gdpo"
    assert tdm_r1.tdm_weight == 0.3
    assert tdm_r1.surrogate_preference_beta == 1.0
    assert tdm_r1.advantage_clip_range == 5.0
    assert tdm_r1.use_huber is False
    assert {"surrogate_beta", "generator_kl_beta", "dm_step_scale", "dm_loss_type"}.isdisjoint(
        field.name for field in fields(TDMR1TrainingArguments)
    )


def test_per_role_optimizers_parse_from_the_top_level_list() -> None:
    """Optimizer hyperparameters are framework configuration, not algorithm fields."""
    config = Arguments.from_dict(
        {
            "train": {
                "trainer_type": "tdm-r1",
                "unique_sample_num_per_epoch": 8,
                "per_device_batch_size": 2,
                "gradient_step_per_epoch": 1,
                "group_size": 2,
            },
            "scheduler": {"dynamics_type": "ODE"},
            "rewards": [{"name": "score", "reward_model": "clip"}],
            "optimizers": [
                {
                    "name": "fake",
                    "learning_rate": "2e-5",
                    "betas": ["0.8", "0.95"],
                    "weight_decay": "3e-4",
                    "eps": "4e-8",
                    "max_grad_norm": "0.7",
                },
                {"name": "generator", "learning_rate": "5e-6"},
                {"name": "surrogate", "optimizer": "muon", "learning_rate": "6e-5"},
            ],
        }
    )

    fake = config.optimizer_args.get_by_name("fake")
    assert isinstance(fake, AdamWOptimizerArguments)
    assert fake.learning_rate == 2e-5
    assert fake.betas == (0.8, 0.95)
    assert fake.weight_decay == 3e-4
    assert fake.eps == 4e-8
    assert fake.max_grad_norm == 0.7
    assert config.optimizer_args.get_by_name("generator").learning_rate == 5e-6

    # A role can now select Muon without the algorithm knowing the optimizer exists.
    surrogate = config.optimizer_args.get_by_name("surrogate")
    assert surrogate.optimizer == "muon"
    assert surrogate.learning_rate == 6e-5


def test_multirole_config_rejects_invalid_declared_fields() -> None:
    with pytest.raises((TypeError, ValueError), match="perturbation_timestep_range"):
        _parse_train(
            "dmd2",
            train_overrides={"perturbation_timestep_range": [False, True]},
        )


@pytest.mark.parametrize(
    "args_cls",
    [DMD2TrainingArguments, TDMTrainingArguments, TDMR1TrainingArguments],
)
def test_multirole_training_args_declare_shared_model_inference_fields(args_cls) -> None:
    args = args_cls.from_dict(
        {
            "num_frames": 124,
            "frame_rate": 24.0,
            "extra_kwargs": {"timestep_shift": 3.0},
        }
    )

    assert args.num_frames == 124
    assert args.frame_rate == 24.0
    assert args.extra_kwargs == {"timestep_shift": 3.0}
    assert dict(args)["num_frames"] == 124
    assert dict(args)["frame_rate"] == 24.0
    assert dict(args)["timestep_shift"] == 3.0


@pytest.mark.parametrize(
    "field_name",
    ["surrogate_beta", "generator_kl_beta", "dm_step_scale", "dm_loss_type"],
)
def test_tdm_r1_forwards_undeclared_fields_through_extra_kwargs(field_name: str) -> None:
    config = _parse_train(
        "tdm-r1",
        train_overrides={"group_size": 2, field_name: 1.0},
        rewards=[{"name": "score", "reward_model": "clip"}],
    )

    assert config.training_args.extra_kwargs[field_name] == 1.0


def test_role_plans_and_base_trainer_config_seams_are_exact() -> None:
    dmd2 = DMD2TrainingArguments()
    tdm = TDMTrainingArguments()
    tdm_r1 = TDMR1TrainingArguments()

    assert dmd2.required_trainable_roles == ("generator", "fake")
    assert [
        (phase.role_name, phase.repeats)
        for phase in dmd2.role_update_plan({"generator": 1, "fake": 3}).phases
    ] == [("fake", 3), ("generator", 1)]
    assert [
        (phase.role_name, phase.repeats)
        for phase in tdm.role_update_plan({"generator": 1, "fake": 5}).phases
    ] == [("fake", 5), ("generator", 1)]
    assert tdm_r1.required_trainable_roles == ("generator", "fake", "surrogate")
    assert [
        (phase.role_name, phase.repeats)
        for phase in tdm_r1.role_update_plan({"generator": 1, "fake": 5, "surrogate": 1}).phases
    ] == [("fake", 5), ("surrogate", 1), ("generator", 1)]
    with pytest.raises(ValueError, match=r"missing \['fake'\]"):
        dmd2.role_update_plan({"generator": 1})

    # The trainer projects the framework's `optimizers` list onto the coordinator's
    # per-role view, and the plan repeats each role by its resolved update_frequency.
    optimizers = MultiOptimizerArguments(
        optimizer_configs=[
            AdamWOptimizerArguments(name="generator", learning_rate=3e-5),
            AdamWOptimizerArguments(name="fake", learning_rate=2e-5, update_frequency=3),
        ]
    )
    trainer = type(
        "TrainerConfigProbe",
        (),
        {
            "training_args": dmd2,
            "config": SimpleNamespace(optimizer_args=optimizers),
            "_optimizer_args_for_role": DMD2Trainer._optimizer_args_for_role,
            "_role_update_plan": BaseTrainer._role_update_plan,
        },
    )()
    role_configs = BaseTrainer._role_optimizer_configs(trainer)
    assert [
        (config.role_name, config.learning_rate, config.update_frequency) for config in role_configs
    ] == [
        ("generator", 3e-5, 1),
        ("fake", 2e-5, 3),
    ]
    assert [
        (phase.role_name, phase.repeats) for phase in BaseTrainer._role_update_plan(trainer).phases
    ] == [("fake", 3), ("generator", 1)]
    assert BaseTrainer._role_update_repeats(trainer, "fake") == 3
    with pytest.raises(ValueError, match="exactly one 'surrogate' phase"):
        BaseTrainer._role_update_repeats(trainer, "surrogate")


def _resolved_fake_repeats(config: Arguments) -> int:
    """Resolve the fake TTUR repeats exactly as the algorithm's trainer would."""
    trainer_cls = _MULTIROLE_TRAINERS[config.training_args.trainer_type]
    trainer = type(
        "CadenceProbe",
        (),
        {
            "training_args": config.training_args,
            "config": config,
            "_optimizer_args_for_role": trainer_cls._optimizer_args_for_role,
            "_role_update_plan": BaseTrainer._role_update_plan,
        },
    )()
    return BaseTrainer._role_update_repeats(trainer, "fake")


@pytest.mark.parametrize("trainer_type", ["dmd2", "tdm", "tdm-r1"])
def test_removed_ttur_fake_updates_fails_with_a_migration_hint(trainer_type: str) -> None:
    with pytest.raises(ValueError, match=r"ttur_fake_updates was replaced .*update_frequency"):
        _parse_multirole(trainer_type, ttur_fake_updates=2)


@pytest.mark.parametrize("trainer_type", ["dmd2", "tdm", "tdm-r1"])
def test_fake_update_frequency_sets_the_ttur_cadence(trainer_type: str) -> None:
    assert _resolved_fake_repeats(_parse_multirole(trainer_type)) == 5
    explicit = _parse_multirole(
        trainer_type,
        optimizers=[{"name": "fake", "learning_rate": 2e-5, "update_frequency": 2}],
    )
    assert _resolved_fake_repeats(explicit) == 2
    # An entry that leaves update_frequency unset keeps the algorithm's cadence.
    partial = _parse_multirole(trainer_type, optimizers=[{"name": "fake", "learning_rate": 2e-5}])
    assert _resolved_fake_repeats(partial) == 5


@pytest.mark.parametrize("entry_name", ["fakee", "default"])
def test_multirole_optimizer_entries_must_name_a_trainable_role(entry_name: str) -> None:
    """A misnamed entry would otherwise be ignored and silently keep the default cadence."""
    with pytest.raises(
        ValueError, match=rf"trainable roles \['generator', 'fake'\].*'{entry_name}'"
    ):
        _parse_multirole("dmd2", optimizers=[{"name": entry_name, "update_frequency": 3}])


@pytest.mark.parametrize(
    ("trainer_type", "role_name"),
    [("dmd2", "generator"), ("tdm", "generator"), ("tdm-r1", "generator"), ("tdm-r1", "surrogate")],
)
def test_roles_that_step_once_per_cycle_reject_update_frequency_at_parse(
    trainer_type: str, role_name: str
) -> None:
    with pytest.raises(ValueError, match=rf"steps \['{role_name}'\] once per update cycle"):
        _parse_multirole(trainer_type, optimizers=[{"name": role_name, "update_frequency": 2}])


def test_dmd2_stays_one_while_tdm_r1_auto_gas_counts_boundaries() -> None:
    automatic = _parse_train("dmd2").training_args
    tdm_r1 = _parse_train(
        "tdm-r1",
        train_overrides={"group_size": 2},
        rewards=[{"name": "score", "reward_model": "clip"}],
    ).training_args

    assert automatic.num_batches_per_epoch == 8
    assert automatic.gradient_accumulation_steps == 1
    assert tdm_r1.num_batches_per_epoch == 8
    assert tdm_r1.gradient_accumulation_steps == 32


@pytest.mark.parametrize("manual_gas", [4, 64, 128])
def test_tdm_r1_accepts_manual_gas(
    manual_gas: int,
) -> None:
    training_args = _parse_train(
        "tdm-r1",
        train_overrides={
            "group_size": 2,
            "gradient_accumulation_steps": manual_gas,
        },
        rewards=[{"name": "score", "reward_model": "clip"}],
    ).training_args

    assert training_args.gradient_accumulation_steps == manual_gas
    assert training_args._manual_gradient_accumulation_steps is True


@pytest.mark.parametrize("manual_gas", [2, 63, 65])
def test_tdm_r1_rejects_manual_gas_not_divisible_by_boundaries(
    manual_gas: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"gradient_accumulation_steps.*divisible.*num_inference_steps=4",
    ):
        _parse_train(
            "tdm-r1",
            train_overrides={
                "group_size": 2,
                "gradient_accumulation_steps": manual_gas,
            },
            rewards=[{"name": "score", "reward_model": "clip"}],
        )


def test_dmd2_default_geometry_resolves_one_batch_per_outer_iteration() -> None:
    training_args = Arguments.from_dict(
        {
            "train": {
                "trainer_type": "dmd2",
                "num_inference_steps": 1,
                "unique_sample_num_per_epoch": 8,
                "per_device_batch_size": 1,
            },
            "scheduler": {"dynamics_type": "ODE"},
        }
    ).training_args

    assert training_args.gradient_step_per_epoch == 1
    assert training_args.num_batches_per_epoch == 8
    assert training_args.gradient_accumulation_steps == 1


def test_dmd2_accepts_manual_single_batch_accumulation() -> None:
    training_args = _parse_train(
        "dmd2",
        train_overrides={"gradient_accumulation_steps": 1},
    ).training_args

    assert training_args.gradient_accumulation_steps == 1


def test_dmd2_rejects_multiple_auto_generator_steps() -> None:
    with pytest.raises(
        ValueError,
        match=(r"dmd2 requires gradient_step_per_epoch=1; " r"received gradient_step_per_epoch=2"),
    ):
        _parse_train(
            "dmd2",
            train_overrides={
                "num_inference_steps": 1,
                "gradient_step_per_epoch": 2,
            },
        )


@pytest.mark.parametrize("manual_gas", [2, 3, 8])
def test_dmd2_accepts_manual_gradient_accumulation(manual_gas: int) -> None:
    training_args = _parse_train(
        "dmd2",
        train_overrides={
            "num_inference_steps": 1,
            "gradient_accumulation_steps": manual_gas,
        },
    ).training_args

    assert training_args.gradient_accumulation_steps == manual_gas
    assert training_args.gradient_step_per_epoch == 1


def test_dmd2_rejects_untiled_unique_sample_without_auto_align() -> None:
    with pytest.raises(ValueError, match="does not auto-align unique_sample_num_per_epoch"):
        _parse_train(
            "dmd2",
            train_overrides={
                "unique_sample_num_per_epoch": 7,
                "per_device_batch_size": 2,
            },
        )


def test_tdm_r1_rejects_a_geometry_no_sampler_can_group() -> None:
    """One rank holding half a group cannot form a group logit under any layout."""
    with pytest.raises(ValueError, match="no compatible sampler"):
        _parse_train(
            "tdm-r1",
            train_overrides={
                "group_size": 2,
                "per_device_batch_size": 1,
            },
            rewards=[{"name": "score", "reward_model": "clip"}],
        )


def test_tdm_r1_splits_a_group_across_ranks_when_a_microbatch_cannot_hold_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group larger than the microbatch used to be rejected outright.

    ``group_distributed`` packs complete groups into the global microbatch, so the group
    logit is a cross-rank sum rather than a local one. That admits the common shape of
    one reward group spanning the whole global step.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")

    config = _parse_train(
        "tdm-r1",
        train_overrides={
            "group_size": 4,
            "per_device_batch_size": 2,
            "unique_sample_num_per_epoch": 4,
        },
        rewards=[{"name": "score", "reward_model": "clip"}],
    )

    assert config.data_args.sampler_type == "group_distributed"


def test_group_distributed_preserves_k16_on_32_single_sample_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "32")

    config = _parse_train(
        "dgpo",
        train_overrides={
            "group_size": 16,
            "per_device_batch_size": 1,
            "unique_sample_num_per_epoch": 48,
        },
        rewards=[{"name": "score", "reward_model": "clip"}],
    )

    assert config.data_args.sampler_type == "group_distributed"
    assert config.training_args.group_size == 16
    assert config.training_args.unique_sample_num_per_epoch == 48
    assert config.training_args.num_batches_per_epoch == 24


def test_group_tiled_aligns_to_gcd_derived_group_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "6")

    config = _parse_train(
        "grpo",
        data_overrides={"sampler_type": "group_tiled"},
        train_overrides={
            "group_size": 4,
            "per_device_batch_size": 1,
            "unique_sample_num_per_epoch": 4,
        },
        rewards=[{"name": "score", "reward_model": "clip"}],
    )

    assert config.data_args.sampler_type == "group_tiled"
    assert config.training_args.unique_sample_num_per_epoch % 3 == 0


def test_subgroup_tiled_aligns_each_contiguous_rank_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")

    config = _parse_train(
        "grpo",
        data_overrides={
            "sampler_type": "subgroup_tile",
            "sampler_subgroup_size": 2,
        },
        train_overrides={
            "group_size": 4,
            "per_device_batch_size": 1,
            "unique_sample_num_per_epoch": 5,
        },
        rewards=[{"name": "score", "reward_model": "clip"}],
    )

    # Four two-rank subgroups each close one K=4 group every two batches.
    assert config.training_args.unique_sample_num_per_epoch == 8


def test_subgroup_tiled_rejects_a_non_divisor_rank_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")

    with pytest.raises(ValueError, match="divide num_replicas"):
        _parse_train(
            "grpo",
            data_overrides={
                "sampler_type": "subgroup_tile",
                "sampler_subgroup_size": 3,
            },
            rewards=[{"name": "score", "reward_model": "clip"}],
        )


def test_tdm_r1_prefers_the_rank_local_layout_when_a_microbatch_holds_whole_groups() -> None:
    """Keeping a group on one rank saves the collective, so it wins when both fit."""
    config = _parse_train(
        "tdm-r1",
        train_overrides={
            "group_size": 2,
            "per_device_batch_size": 4,
            "unique_sample_num_per_epoch": 4,
        },
        rewards=[{"name": "score", "reward_model": "clip"}],
    )

    assert config.data_args.sampler_type == "group_contiguous"


@pytest.mark.parametrize("trainer_type", ["dmd2", "tdm"])
def test_reward_free_distillation_rejects_training_rewards(trainer_type: str) -> None:
    without_rewards = _parse_train(trainer_type)
    assert len(without_rewards.reward_args) == 0

    with pytest.raises(ValueError, match="does not accept training rewards"):
        _parse_train(
            trainer_type,
            rewards=[{"name": "score", "reward_model": "clip"}],
        )


def test_tdm_r1_requires_reward_and_complete_groups() -> None:
    with pytest.raises(ValueError, match="at least one training reward"):
        _parse_train("tdm-r1", train_overrides={"group_size": 2})
    with pytest.raises(ValueError, match="group_size"):
        _parse_train(
            "tdm-r1",
            rewards=[{"name": "score", "reward_model": "clip"}],
        )


@pytest.mark.parametrize("trainer_type", ["tdm", "tdm-r1"])
def test_tdm_variants_require_ode_dynamics(trainer_type: str) -> None:
    kwargs = {}
    if trainer_type == "tdm-r1":
        kwargs = {
            "train_overrides": {"group_size": 2},
            "rewards": [{"name": "score", "reward_model": "clip"}],
        }
    with pytest.raises(ValueError, match="dynamics_type.*ODE"):
        _parse_train(
            trainer_type,
            scheduler_overrides={"dynamics_type": "Flow-SDE"},
            **kwargs,
        )


def test_trainer_registry_exposes_future_lazy_paths_without_importing_modules() -> None:
    registered = list_registered_trainers()
    assert {key: registered[key] for key in ("dmd2", "tdm", "tdm-r1")} == {
        "dmd2": "flow_factory.trainers.distillation.dmd2.DMD2Trainer",
        "tdm": "flow_factory.trainers.distillation.tdm.TDMTrainer",
        "tdm-r1": "flow_factory.trainers.distillation.tdm_r1.TDMR1Trainer",
    }
