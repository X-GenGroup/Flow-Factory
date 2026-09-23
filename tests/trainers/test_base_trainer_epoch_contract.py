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

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, List

import pytest
import torch

from flow_factory.contracts.execution import OFFLINE_EXECUTION_CONTRACT
from flow_factory.trainers.abc import BaseTrainer


class EpochTrainerFake(BaseTrainer):
    """Trainer exercising only the shared epoch loop."""

    paradigm = "decoupled"

    def __init__(self, total_epochs: int) -> None:
        self.events: List[str] = []
        self.epoch = 0
        self.step = 0
        self._total_epochs = total_epochs
        self.adapter = SimpleNamespace(
            set_trajectory_seed=lambda seed: self.events.append(f"seed:{seed}"),
            ema_step=lambda step: self.events.append(f"ema:{step}"),
        )
        self.training_args = SimpleNamespace(seed=100)
        self.log_args = SimpleNamespace(save_freq=0, save_dir=None, run_name="run")
        self.eval_args = SimpleNamespace(eval_freq=0)
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            is_local_main_process=False,
        )
        self.logger = None

    def should_continue_training(self) -> bool:
        """Stop after the configured number of epochs."""
        return self.epoch < self._total_epochs

    def sample(self) -> List[Any]:
        """Record the rollout stage."""
        self.events.append("sample")
        return []

    def prepare_feedback(self, samples: List[Any]) -> None:
        """Record the feedback stage."""
        self.events.append("feedback")

    def optimize(self, samples: List[Any]) -> None:
        """Record the optimization stage."""
        self.events.append("optimize")


class ScopedEpochTrainerFake(EpochTrainerFake):
    """Trainer that rolls out under a swapped parameter scope."""

    @contextmanager
    def sampling_context(self) -> Iterator[None]:
        """Record entering and leaving the sampling scope."""
        self.events.append("scope:enter")
        yield
        self.events.append("scope:exit")

    def _after_acquisition_cycle(self) -> None:
        """Record an algorithm-owned auxiliary update."""
        self.events.append("after_step")


class OfflineEpochTrainerFake(EpochTrainerFake):
    """Expose dataset-acquisition boundary semantics without a real loader."""

    execution_contract = OFFLINE_EXECUTION_CONTRACT


def test_shared_epoch_runs_the_stages_in_order() -> None:
    """Every algorithm inherits one reseed, sample, feedback, optimize, EMA sequence."""
    trainer = EpochTrainerFake(total_epochs=2)

    trainer.start()

    assert trainer.events == [
        "seed:100",
        "sample",
        "feedback",
        "optimize",
        "ema:0",
        "seed:101",
        "sample",
        "feedback",
        "optimize",
        "ema:1",
    ]
    assert trainer.epoch == 2


def test_sampling_scope_wraps_only_the_rollout() -> None:
    """An EMA or snapshot swap must close before feedback and the policy update."""
    trainer = ScopedEpochTrainerFake(total_epochs=1)

    trainer.start()

    assert trainer.events == [
        "seed:100",
        "scope:enter",
        "sample",
        "scope:exit",
        "feedback",
        "optimize",
        "ema:0",
        "after_step",
    ]


def test_on_policy_sampling_needs_no_scope_override() -> None:
    """The default scope is a no-op, so a coupled trainer inherits it unchanged."""
    trainer = EpochTrainerFake(total_epochs=1)

    with trainer.sampling_context():
        pass

    assert trainer.events == []


def test_offline_checkpoint_captures_post_eval_rng_state(tmp_path: Any) -> None:
    """Offline save dispatch occurs after evaluation advances exact-state RNG."""
    trainer = object.__new__(OfflineEpochTrainerFake)
    trainer.epoch = 1
    trainer.log_args = SimpleNamespace(save_freq=1, save_dir=str(tmp_path), run_name="run")
    trainer.eval_args = SimpleNamespace(eval_freq=1)
    events: List[str] = []
    post_eval_rng: List[torch.Tensor] = []
    saved_rng: List[torch.Tensor] = []

    def evaluate() -> None:
        events.append("eval")
        torch.rand(1)
        post_eval_rng.append(torch.get_rng_state().clone())

    def save_checkpoint(save_directory: str, *, epoch: int) -> None:
        events.append("save")
        saved_rng.append(torch.get_rng_state().clone())

    trainer.evaluate = evaluate
    trainer.save_checkpoint = save_checkpoint
    torch.manual_seed(1234)

    trainer._run_periodic_cycle_boundaries()

    assert events == ["eval", "save"]
    assert torch.equal(saved_rng[0], post_eval_rng[0])


def test_online_boundary_keeps_save_before_evaluation(tmp_path: Any) -> None:
    """Generated acquisition retains the established pre-rollout save cadence."""
    trainer = object.__new__(EpochTrainerFake)
    trainer.epoch = 1
    trainer.log_args = SimpleNamespace(save_freq=1, save_dir=str(tmp_path), run_name="run")
    trainer.eval_args = SimpleNamespace(eval_freq=1)
    events: List[str] = []
    trainer.save_checkpoint = lambda *args, **kwargs: events.append("save")
    trainer.evaluate = lambda: events.append("eval")

    trainer._run_periodic_cycle_boundaries()

    assert events == ["save", "eval"]


def test_coupled_paradigm_rejects_ode_dynamics() -> None:
    """A coupled algorithm on ODE dynamics has no transition density to differentiate."""
    trainer = object.__new__(EpochTrainerFake)
    type(trainer).paradigm = "coupled"
    trainer.adapter = SimpleNamespace(scheduler_group=_SchedulerGroupFake("Flow-ODE"))

    try:
        with pytest.raises(ValueError, match="requires stochastic dynamics"):
            trainer._validate_paradigm_dynamics()
    finally:
        type(trainer).paradigm = "decoupled"


def test_coupled_paradigm_accepts_stochastic_dynamics() -> None:
    """An SDE scheduler is exactly what a coupled algorithm needs."""
    trainer = object.__new__(EpochTrainerFake)
    type(trainer).paradigm = "coupled"
    trainer.adapter = SimpleNamespace(scheduler_group=_SchedulerGroupFake("Flow-SDE"))

    try:
        trainer._validate_paradigm_dynamics()
    finally:
        type(trainer).paradigm = "decoupled"


class _SchedulerGroupFake:
    """Scheduler group exposing one component with a fixed dynamics type."""

    def __init__(self, dynamics_type: str) -> None:
        self.names = ("latent",)
        self._dynamics_type = dynamics_type

    def __getitem__(self, name: str) -> SimpleNamespace:
        """Return the single component scheduler."""
        return SimpleNamespace(dynamics_type=self._dynamics_type)
