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

"""Tests for SFT and offline-DPO training argument contracts."""

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from flow_factory.contracts.execution import OFFLINE_EXECUTION_CONTRACT
from flow_factory.hparams import (
    Arguments,
    OfflineDPOTrainingArguments,
    SFTTrainingArguments,
    get_training_args_class,
)


def _offline_config(trainer_type: str, **train_overrides: object) -> Arguments:
    train = {
        "trainer_type": trainer_type,
        "gradient_accumulation_steps": 2,
        "max_epochs": 3,
        **train_overrides,
    }
    return Arguments.from_dict(
        {
            "data": {
                "datasets": [
                    {
                        "name": "offline",
                        "dataset_dir": "unused",
                        "train": {"weight": 1},
                    }
                ]
            },
            "scheduler": {"dynamics_type": "ODE"},
            "train": train,
        }
    )


@pytest.mark.parametrize(
    ("trainer_type", "arguments_class"),
    [
        ("sft", SFTTrainingArguments),
        ("offline-dpo", OfflineDPOTrainingArguments),
    ],
)
def test_offline_training_arguments_resolve_from_public_config(
    trainer_type: str,
    arguments_class: type,
) -> None:
    """Resolve both offline algorithms without entering grouped rollout geometry."""
    config = _offline_config(
        trainer_type,
        weighting_scheme="uniform",
        num_train_timesteps=3,
        timestep_range=[0.1, 0.9],
        time_shift=2,
        logit_mean=-0.5,
        logit_std=1.5,
    )

    assert get_training_args_class(trainer_type) is arguments_class
    assert isinstance(config.training_args, arguments_class)
    assert config.training_args.execution_contract is OFFLINE_EXECUTION_CONTRACT
    assert config.training_args.gradient_accumulation_steps == 2
    assert config.training_args.max_epochs == 3
    assert config.training_args.num_batches_per_epoch == 0
    assert config.training_args.num_train_timesteps == 3
    assert config.training_args.timestep_range == (0.1, 0.9)
    assert config.training_args.time_shift == 2.0
    assert config.data_args.sampler_type == "auto"


def test_offline_defaults_express_data_epoch_and_reference_semantics() -> None:
    """Keep SFT reference-free while DPO explicitly requires frozen reference losses."""
    sft = SFTTrainingArguments()
    dpo = OfflineDPOTrainingArguments()

    assert sft.trainer_type == "sft"
    assert dpo.trainer_type == "offline-dpo"
    assert sft.max_epochs == dpo.max_epochs == 1
    assert sft.gradient_accumulation_steps == dpo.gradient_accumulation_steps == 1
    assert sft.requires_ref_model is False
    assert dpo.requires_ref_model is True
    assert dpo.beta == 2000.0
    assert "reference_free" not in {field.name for field in fields(dpo)}
    assert "execution_contract" not in dpo.to_dict()


@pytest.mark.parametrize("arguments_class", [SFTTrainingArguments, OfflineDPOTrainingArguments])
@pytest.mark.parametrize("value", ["auto", 0, -1, True, 1.5])
def test_offline_gradient_accumulation_must_be_an_explicit_positive_integer(
    arguments_class: type,
    value: object,
) -> None:
    """Reject online automatic accumulation and ambiguous numeric values."""
    with pytest.raises((TypeError, ValueError), match="gradient_accumulation_steps"):
        arguments_class(gradient_accumulation_steps=value)


@pytest.mark.parametrize("arguments_class", [SFTTrainingArguments, OfflineDPOTrainingArguments])
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5])
def test_offline_max_epochs_counts_positive_complete_loader_traversals(
    arguments_class: type,
    value: object,
) -> None:
    """Require a finite positive count of complete offline data epochs."""
    with pytest.raises((TypeError, ValueError), match="train.max_epochs"):
        arguments_class(max_epochs=value)


@pytest.mark.parametrize("arguments_class", [SFTTrainingArguments, OfflineDPOTrainingArguments])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_offline_num_train_timesteps_is_a_positive_monte_carlo_count(
    arguments_class: type,
    value: object,
) -> None:
    """Reject invalid independent timestep-term counts."""
    with pytest.raises((TypeError, ValueError), match="train.num_train_timesteps"):
        arguments_class(num_train_timesteps=value)


@pytest.mark.parametrize(
    "value",
    [
        "0.9",
        [0.1],
        [0.1, 0.5, 0.9],
        [0.7, 0.2],
        [-0.1, 0.9],
        [0.1, 1.1],
        [0.1, float("nan")],
        [False, 0.9],
    ],
)
def test_offline_timestep_range_uses_strict_denoising_axis_fractions(value: object) -> None:
    """Reject malformed, non-finite, and out-of-domain timestep fractions."""
    with pytest.raises((TypeError, ValueError), match="train.timestep_range"):
        SFTTrainingArguments(timestep_range=value)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("time_shift", 0),
        ("time_shift", float("inf")),
        ("time_shift", True),
        ("logit_mean", float("nan")),
        ("logit_mean", "0"),
        ("logit_std", 0),
        ("logit_std", -1),
        ("logit_std", float("inf")),
    ],
)
def test_offline_distribution_parameters_are_finite_and_well_defined(
    field_name: str,
    value: object,
) -> None:
    """Validate every scalar used by the independent timestep sampler."""
    with pytest.raises((TypeError, ValueError), match=f"train.{field_name}"):
        SFTTrainingArguments(**{field_name: value})


@pytest.mark.parametrize("value", ["discrete", ["uniform"]])
def test_offline_weighting_scheme_rejects_online_or_discrete_modes(value: object) -> None:
    """Limit the public configuration to implemented offline samplers."""
    with pytest.raises((TypeError, ValueError), match="train.weighting_scheme"):
        SFTTrainingArguments(weighting_scheme=value)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_offline_dpo_requires_a_positive_finite_beta(value: object) -> None:
    """Reject scales that cannot represent the implemented DPO temperature."""
    with pytest.raises((TypeError, ValueError), match="train.beta"):
        OfflineDPOTrainingArguments(beta=value)


@pytest.mark.parametrize(
    "values",
    [
        {"reference_free": True},
        {"extra_kwargs": {"reference_free": False}},
    ],
)
def test_offline_dpo_rejects_unimplemented_reference_free_configuration(
    values: dict,
) -> None:
    """Keep the public surface aligned with the four-input shared DPO objective."""
    with pytest.raises(ValueError, match="requires frozen reference losses"):
        OfflineDPOTrainingArguments.from_dict(values)


@pytest.mark.parametrize("arguments_class", [SFTTrainingArguments, OfflineDPOTrainingArguments])
def test_user_cannot_override_offline_execution_contract(arguments_class: type) -> None:
    """Select offline acquisition only through the registered trainer type."""
    with pytest.raises(ValueError, match="selected by trainer_type"):
        arguments_class.from_dict({"execution_contract": "generation"})


@pytest.mark.parametrize(
    ("arguments_class", "wrong_trainer_type"),
    [
        (SFTTrainingArguments, "offline-dpo"),
        (OfflineDPOTrainingArguments, "sft"),
    ],
)
def test_direct_construction_rejects_mismatched_offline_trainer_identity(
    arguments_class: type,
    wrong_trainer_type: str,
) -> None:
    """Prevent direct class use from disagreeing with registry dispatch."""
    with pytest.raises(ValueError, match="requires train.trainer_type"):
        arguments_class(trainer_type=wrong_trainer_type)


def test_offline_config_rejects_runtime_training_rewards() -> None:
    """Keep dataset supervision independent from online reward feedback."""
    with pytest.raises(ValueError, match="does not accept training rewards"):
        Arguments.from_dict(
            {
                "data": {
                    "datasets": [
                        {
                            "name": "offline",
                            "dataset_dir": "unused",
                            "train": {"weight": 1},
                        }
                    ]
                },
                "train": {
                    "trainer_type": "sft",
                    "gradient_accumulation_steps": 1,
                },
                "rewards": [{"name": "score", "reward_model": "clip"}],
            }
        )


def test_wan22_ti2v5b_sft_example_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the public video recipe on supported offline configuration fields."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    path = Path(__file__).resolve().parents[2] / "examples/sft/lora/wan2_t2v/wan22_ti2v5b.yaml"
    config = Arguments.from_dict(yaml.safe_load(path.read_text()))
    assert isinstance(config.training_args, SFTTrainingArguments)
    assert config.training_args.max_epochs == 5
    assert config.training_args.num_frames == 49
    assert config.training_args.frame_rate == 10.0
    assert (config.training_args.height, config.training_args.width) == (480, 832)
    assert (config.model_args.lora_rank, config.model_args.lora_alpha) == (128, 256)
    assert config.model_args.target_components == ["transformer"]
    assert config.log_args.save_freq == 1
    assert config.log_args.save_model_only is False
    assert config.eval_args.eval_freq == 0
