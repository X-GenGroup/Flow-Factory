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

from types import SimpleNamespace
from typing import Any, List

import pytest
import torch

from flow_factory.hparams import (
    AdamWOptimizerArguments,
    Arguments,
    DMD2TrainingArguments,
    MultiOptimizerArguments,
)
from flow_factory.trainers.distillation.dmd2 import DMD2Trainer
from flow_factory.trainers.distillation.tdm import TDMTrainer
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer

DISTILLATION_TRAINERS = (DMD2Trainer, TDMTrainer, TDMR1Trainer)
FAKE_UPDATE_FREQUENCY = 2
# DMD2 draws one fresh GAS-sized rollout set per fake step; TDM and TDM-R1 share one.
ROLLOUT_SETS_PER_EPOCH = {DMD2Trainer: FAKE_UPDATE_FREQUENCY, TDMTrainer: 1, TDMR1Trainer: 1}


def _loop_trainer(trainer_cls: type, *, eval_freq: int, epochs: int = 2) -> Any:
    """Build a trainer that records the shared epoch loop's calls."""
    trainer = object.__new__(trainer_cls)
    trainer.events: List[str] = []
    trainer.epoch = 0
    trainer.step = 0
    trainer.adapter = SimpleNamespace(
        set_trajectory_seed=lambda seed: None,
        ema_step=lambda step: None,
    )
    # The epoch closes by reducing its buffered metrics across ranks, which is a
    # one-rank no-op here.
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda tensor, reduction: tensor,
    )
    trainer.training_args = SimpleNamespace(
        seed=0,
        gradient_accumulation_steps=2,
        get_num_train_timesteps=lambda config: 1,
        required_trainable_roles=("generator", "fake"),
        role_update_plan=DMD2TrainingArguments().role_update_plan,
    )
    trainer.config = SimpleNamespace(
        optimizer_args=MultiOptimizerArguments(
            optimizer_configs=[
                AdamWOptimizerArguments(name="fake", update_frequency=FAKE_UPDATE_FREQUENCY)
            ]
        )
    )
    trainer.log_args = SimpleNamespace(save_freq=0, save_dir=None, run_name="run", verbose=False)
    trainer.eval_args = SimpleNamespace(eval_freq=eval_freq)
    trainer.should_continue_training = lambda: trainer.epoch < epochs
    trainer.evaluate = lambda: trainer.events.append(f"evaluate:{trainer.epoch}")
    trainer.sample = lambda: trainer.events.append("sample") or []
    trainer.prepare_feedback = lambda samples: None
    trainer.optimize = lambda microbatches, **kwargs: trainer.events.append(
        f"optimize:{len(microbatches)}"
    )
    return trainer


@pytest.mark.parametrize("trainer_cls", DISTILLATION_TRAINERS)
def test_distillation_evaluates_on_eval_freq_like_every_other_trainer(
    trainer_cls: type,
) -> None:
    """Image-quality monitoring at eval time is not an algorithm-specific feature."""
    trainer = _loop_trainer(trainer_cls, eval_freq=1)

    trainer.start()

    assert [event for event in trainer.events if event.startswith("evaluate")] == [
        "evaluate:0",
        "evaluate:1",
    ]


@pytest.mark.parametrize("trainer_cls", DISTILLATION_TRAINERS)
def test_distillation_accumulates_gas_rollouts_before_one_optimize(
    trainer_cls: type,
) -> None:
    """Each optimize call receives one GAS-sized set of distinct rollouts."""
    trainer = _loop_trainer(trainer_cls, eval_freq=0, epochs=1)

    trainer.start()

    assert (
        trainer.events == ["sample", "sample", "optimize:2"] * ROLLOUT_SETS_PER_EPOCH[trainer_cls]
    )


@pytest.mark.parametrize("trainer_cls", DISTILLATION_TRAINERS)
def test_evaluation_is_skipped_when_eval_freq_is_disabled(trainer_cls: type) -> None:
    """A run without evaluation configured pays nothing for the shared branch."""
    trainer = _loop_trainer(trainer_cls, eval_freq=0)

    trainer.start()

    assert not [event for event in trainer.events if event.startswith("evaluate")]


def _distillation_config(trainer_type: str, **sections: Any) -> Arguments:
    """Build a minimal distillation config with one training and one eval dataset."""
    config = {
        "train": {
            "trainer_type": trainer_type,
            "unique_sample_num_per_epoch": 4,
            "per_device_batch_size": 2 if trainer_type == "tdm-r1" else 1,
            "gradient_step_per_epoch": 1,
        },
        "scheduler": {"dynamics_type": "ODE"},
        "data": {
            "datasets": [
                {"name": "prompts", "dataset_dir": "dataset/P", "train": {"weight": 1}},
                {"name": "bench", "dataset_dir": "dataset/B", "eval": {}},
            ]
        },
        "eval": {"eval_freq": 10},
    }
    if trainer_type == "tdm-r1":
        config["train"]["group_size"] = 2
    if trainer_type == "diffusion-opd":
        config["train"]["teachers"] = [
            {
                "name": "teacher",
                "path": "teacher/lora",
                "applicable_datasets": ["prompts"],
            }
        ]
    config.update(sections)
    return Arguments.from_dict(config)


@pytest.mark.parametrize("trainer_type", ["dmd2", "tdm", "diffusion-opd"])
def test_reward_free_distillation_accepts_eval_rewards_only(trainer_type: str) -> None:
    """Quality monitoring is an evaluation concern; the loss stays reward-free."""
    config = _distillation_config(
        trainer_type,
        eval_rewards=[{"name": "quality", "reward_model": "clip"}],
    )

    assert len(config.reward_args) == 0
    assert [reward.name for reward in config.eval_reward_args] == ["quality"]
    assert config.eval_reward_args.get_by_name("quality").applicable_datasets == ["bench"]


@pytest.mark.parametrize("trainer_type", ["dmd2", "tdm", "diffusion-opd"])
def test_reward_free_distillation_still_rejects_training_rewards(trainer_type: str) -> None:
    """An eval-only reward must not become a training signal by accident."""
    with pytest.raises(ValueError, match="rewards"):
        _distillation_config(
            trainer_type,
            rewards=[{"name": "quality", "reward_model": "clip"}],
        )
