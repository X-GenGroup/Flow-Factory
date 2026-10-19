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

"""Focused tests for generated and dataset acquisition execution."""

from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from flow_factory.contracts.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
)
from flow_factory.hparams.training_args import TrainingArguments
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.execution import (
    DatasetAcquisitionDriver,
    GenerationAcquisitionDriver,
    TrainingProgress,
    build_acquisition_driver,
)
from flow_factory.trainers.loader import load_trainer


class _RecordingDistributedSampler(DistributedSampler):
    """Record sampler epoch updates while retaining official behavior."""

    def __init__(self, dataset: List[int], events: List[str]) -> None:
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=False)
        self._events = events

    def set_epoch(self, epoch: int) -> None:
        """Record and apply one complete data-epoch index.

        Args:
            epoch: Completed data epochs before the next traversal.
        """
        self._events.append(f"set_epoch:{epoch}")
        super().set_epoch(epoch)


class _GenerationTrainer(BaseTrainer):
    """Exercise the shared generated-acquisition cadence."""

    def __init__(self, cycles: int) -> None:
        self.events: List[str] = []
        self.progress = TrainingProgress()
        self._cycles = cycles
        self.adapter = SimpleNamespace(
            set_trajectory_seed=lambda seed: self.events.append(f"seed:{seed}"),
            ema_step=lambda step: self.events.append(f"ema:{step}"),
        )
        self.training_args = SimpleNamespace(seed=10)
        self.log_args = SimpleNamespace(save_freq=0, save_dir=None, run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=0)
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            is_local_main_process=False,
        )
        self.logger = None

    def should_continue_training(self) -> bool:
        """Stop after the requested generated acquisitions."""
        return self.epoch < self._cycles

    def sample(self) -> List[Any]:
        """Record generated acquisition."""
        self.events.append("sample")
        return []

    def prepare_feedback(self, samples: List[Any]) -> None:
        """Record the runtime reward stage."""
        del samples
        self.events.append("feedback")

    def optimize(self, samples: List[Any]) -> None:
        """Record generated-example optimization."""
        del samples
        self.events.append("optimize")


class _RewardFreeGenerationTrainer(_GenerationTrainer):
    """Generated acquisition whose algorithm declares no runtime feedback."""

    execution_contract = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT


class _DatasetTrainer(BaseTrainer):
    """Exercise finite dataset acquisition without calling sample()."""

    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def __init__(self, cycles: int, *, fail_on_value: Optional[int] = None) -> None:
        self.events: List[str] = []
        self.progress = TrainingProgress()
        self._cycles = cycles
        self._fail_on_value = fail_on_value
        dataset = list(range(5))
        sampler = _RecordingDistributedSampler(dataset, self.events)
        self.dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
        self.adapter = SimpleNamespace(
            set_trajectory_seed=lambda seed: self.events.append(f"unexpected_seed:{seed}"),
            ema_step=lambda step: self.events.append(f"ema:{step}"),
        )
        self.training_args = SimpleNamespace(seed=10)
        self.log_args = SimpleNamespace(save_freq=0, save_dir=None, run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=0)

    def should_continue_training(self) -> bool:
        """Stop after the requested complete data epochs."""
        return self.epoch < self._cycles

    def sample(self) -> List[Any]:
        """Reject the online API on the dataset path."""
        raise AssertionError("dataset acquisition must not call sample()")

    def optimize_batch(self, batch: torch.Tensor) -> None:
        """Record one dataset batch and advance its independent optimizer count.

        Args:
            batch: Batch yielded by the finite official loader.
        """
        values = batch.tolist()
        self.events.append(f"batch:{values}")
        if self._fail_on_value is not None and values[0] == self._fail_on_value:
            raise RuntimeError("dataset update failed")
        self.step += 1


def test_progress_tracks_optimizer_and_acquisition_counters_independently() -> None:
    """Several optimizer updates may occur inside one complete data epoch."""
    progress = TrainingProgress(optimizer_step=3, rollout_iteration=2, data_epoch=1)

    assert progress.cycle_index(AcquisitionMode.GENERATION) == 2
    assert progress.cycle_index(AcquisitionMode.DATASET) == 1
    assert progress.advance_optimizer_step(2).optimizer_step == 5
    assert progress.advance_acquisition(
        AcquisitionMode.DATASET, completed=True
    ) == TrainingProgress(optimizer_step=3, rollout_iteration=2, data_epoch=2)


def test_generated_acquisition_preserves_existing_online_cadence() -> None:
    """Generation still seeds, samples, receives feedback, optimizes, and steps EMA."""
    trainer = _GenerationTrainer(cycles=2)

    trainer.start()

    assert trainer.events == [
        "seed:10",
        "sample",
        "feedback",
        "optimize",
        "ema:0",
        "seed:11",
        "sample",
        "feedback",
        "optimize",
        "ema:1",
    ]
    assert trainer.progress == TrainingProgress(rollout_iteration=2)


def test_generated_acquisition_may_omit_runtime_feedback() -> None:
    """Reward-free distillation is generated acquisition without a fake feedback stage."""
    trainer = _RewardFreeGenerationTrainer(cycles=1)

    trainer.start()

    assert trainer.events == ["seed:10", "sample", "optimize", "ema:0"]


def test_dataset_epoch_is_exactly_one_complete_dataloader_traversal() -> None:
    """Dataset acquisition exhausts the official loader before advancing its epoch."""
    trainer = _DatasetTrainer(cycles=2)

    trainer.start()

    assert trainer.events == [
        "set_epoch:0",
        "batch:[0, 1]",
        "batch:[2, 3]",
        "batch:[4]",
        "set_epoch:1",
        "batch:[0, 1]",
        "batch:[2, 3]",
        "batch:[4]",
    ]
    assert trainer.progress == TrainingProgress(optimizer_step=6, data_epoch=2)


def test_failed_dataset_batch_does_not_publish_a_partial_epoch() -> None:
    """A partial loader traversal never increments data_epoch."""
    trainer = _DatasetTrainer(cycles=1, fail_on_value=2)

    with pytest.raises(RuntimeError, match="dataset update failed"):
        trainer.start()

    assert trainer.events == ["set_epoch:0", "batch:[0, 1]", "batch:[2, 3]"]
    assert trainer.progress == TrainingProgress(optimizer_step=1)
    assert trainer._acquisition_cycle_incomplete is True


def _record_saves_and_evaluations(trainer: BaseTrainer, *, save_freq: int, eval_freq: int) -> None:
    """Record checkpoint saves and evaluations into the trainer's event stream."""
    trainer.log_args = SimpleNamespace(save_freq=save_freq, save_dir="save", run_name="run")
    trainer.eval_args = SimpleNamespace(eval_freq=eval_freq)
    trainer.save_checkpoint = lambda save_directory, *, epoch: trainer.events.append(
        f"save:{epoch}"
    )
    trainer.evaluate = lambda: trainer.events.append("eval")


def _saves(trainer: BaseTrainer) -> List[str]:
    return [event for event in trainer.events if event.startswith("save:")]


def test_generated_acquisition_saves_the_state_after_the_last_rollout() -> None:
    """Pre-rollout boundaries never reach the final state, so the loop saves it once more."""
    trainer = _GenerationTrainer(cycles=3)
    _record_saves_and_evaluations(trainer, save_freq=2, eval_freq=1)

    trainer.start()

    assert _saves(trainer) == ["save:0", "save:2", "save:3"]
    # The final save runs after the last EMA step and does not add an evaluation.
    assert trainer.events[-2:] == ["ema:2", "save:3"]


def test_resuming_from_the_final_online_checkpoint_matches_an_uninterrupted_run() -> None:
    """The final save sits where boundary N would save, before boundary N evaluates."""
    uninterrupted = _GenerationTrainer(cycles=3)
    _record_saves_and_evaluations(uninterrupted, save_freq=5, eval_freq=1)
    uninterrupted.start()

    first = _GenerationTrainer(cycles=2)
    _record_saves_and_evaluations(first, save_freq=5, eval_freq=1)
    first.start()
    assert first.events[-1] == "save:2"

    resumed = _GenerationTrainer(cycles=3)
    _record_saves_and_evaluations(resumed, save_freq=5, eval_freq=1)
    resumed.progress = TrainingProgress(rollout_iteration=2)
    resumed._exact_resume_source_checkpoint = resumed._canonical_checkpoint_path(
        "save/run/checkpoints/checkpoint-2"
    )
    resumed._exact_resume_boundary_pending = True
    resumed.start()

    # Resume skips rewriting checkpoint-2, then replays boundary 2's evaluation.
    assert first.events[:-1] + resumed.events == uninterrupted.events


@pytest.mark.parametrize(
    ("save_freq", "expected"),
    [(1, ["save:1", "save:2", "save:3"]), (2, ["save:2", "save:3"]), (3, ["save:3"])],
)
def test_dataset_acquisition_saves_the_last_epoch_exactly_once(
    save_freq: int, expected: List[str]
) -> None:
    """A last epoch on a save_freq multiple keeps its single boundary checkpoint."""
    trainer = _DatasetTrainer(cycles=3)
    _record_saves_and_evaluations(trainer, save_freq=save_freq, eval_freq=0)

    trainer.start()

    assert _saves(trainer) == expected
    assert trainer.events[-1] == "save:3"


@pytest.mark.parametrize(
    "trainer",
    [_GenerationTrainer(cycles=2), _DatasetTrainer(cycles=2)],
    ids=["generation", "dataset"],
)
def test_final_save_respects_disabled_saving_and_runs_without_progress(
    trainer: BaseTrainer,
) -> None:
    """save_freq: 0 disables the final save; a run resumed at its budget saves nothing new."""
    _record_saves_and_evaluations(trainer, save_freq=0, eval_freq=0)
    trainer.start()
    assert _saves(trainer) == []

    trainer.events.clear()
    _record_saves_and_evaluations(trainer, save_freq=1, eval_freq=0)
    trainer.start()
    assert trainer.events == []


def test_failed_cycle_does_not_save_a_final_checkpoint() -> None:
    """Only a completed budget publishes the final state."""
    trainer = _DatasetTrainer(cycles=1, fail_on_value=2)
    _record_saves_and_evaluations(trainer, save_freq=1, eval_freq=0)

    with pytest.raises(RuntimeError, match="dataset update failed"):
        trainer.start()

    assert _saves(trainer) == []


def test_dataset_driver_requires_official_distributed_sampler_on_one_process() -> None:
    """Single-process offline execution uses the same distribution contract."""
    trainer = _DatasetTrainer(cycles=1)
    trainer.dataloader = DataLoader(list(range(4)), batch_size=2)

    with pytest.raises(TypeError, match="DistributedSampler even when num_replicas=1"):
        trainer.start()

    assert trainer.events == []


def test_driver_selection_depends_only_on_acquisition() -> None:
    """Feedback choices do not duplicate loader or cycle selection."""
    assert isinstance(
        build_acquisition_driver(_GenerationTrainer.execution_contract),
        GenerationAcquisitionDriver,
    )
    assert isinstance(
        build_acquisition_driver(OFFLINE_EXECUTION_CONTRACT),
        DatasetAcquisitionDriver,
    )


def test_loader_rejects_contract_drift_before_adapter_or_accelerator_loading() -> None:
    """A stale registry pair fails before any heavyweight runtime side effect."""

    class _MismatchedTrainer(BaseTrainer):
        execution_contract = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    config = SimpleNamespace(training_args=TrainingArguments())
    with (
        patch(
            "flow_factory.trainers.loader.get_trainer_class",
            return_value=_MismatchedTrainer,
        ),
        patch("flow_factory.trainers.loader.get_model_adapter_class") as adapter_class,
        patch("flow_factory.trainers.loader.Accelerator") as accelerator,
        pytest.raises(ValueError, match="execution contract mismatch"),
    ):
        load_trainer(config)

    adapter_class.assert_not_called()
    accelerator.assert_not_called()
