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

"""Cover the TDM-R1 objective as the official recipe defines it."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, List

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import BaseSample, LatentState
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.trainers.distillation import tdm_r1 as tdm_r1_module
from flow_factory.trainers.distillation.distribution_matching import revised_x0_loss
from flow_factory.trainers.distillation.group_preference import GroupPreferenceBatch
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter for objective reduction tests."""

    trajectory_component_order = ("latent",)

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        raise NotImplementedError


@contextmanager
def _null_context():
    """Stand in for the snapshot installation context."""
    yield


def _adapter() -> AdapterFake:
    return object.__new__(AdapterFake)


def _state(tensor: torch.Tensor) -> LatentState:
    return LatentState({"latent": tensor})


def test_the_student_is_pushed_along_the_correction() -> None:
    """The whole objective is a stop-grad regression, so this is what it must do."""
    student = _state(torch.zeros(2, 3, requires_grad=True))
    correction = _state(torch.tensor([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]]))
    reference = _state(torch.ones(2, 3))

    loss = revised_x0_loss(
        _adapter(), student, correction, reference, use_huber=False, huber_c=1e-3
    )
    loss.backward()

    # Descending the loss moves the student along the correction, so the gradient
    # points against it.
    gradient = student.components["latent"].grad
    assert torch.all(gradient * correction.components["latent"] <= 0)
    assert gradient[0, 0] < 0 and gradient[1, 1] > 0


def test_a_zero_correction_leaves_nothing_to_learn() -> None:
    """A direction that has collapsed must not still push the generator somewhere."""
    student = _state(torch.zeros(2, 3, requires_grad=True))

    loss = revised_x0_loss(
        _adapter(),
        student,
        _state(torch.zeros(2, 3)),
        _state(torch.ones(2, 3)),
        use_huber=False,
        huber_c=1e-3,
    )

    assert float(loss.detach()) == 0.0


def _generator_trainer(tdm_weight: float) -> TDMR1Trainer:
    """Build the trainer surface the generator objective reads."""
    trainer = object.__new__(TDMR1Trainer)
    trainer.adapter = _adapter()
    trainer.training_args = SimpleNamespace(
        tdm_weight=tdm_weight,
        use_huber=False,
        huber_c=1e-3,
        num_inference_steps=4,
        use_time_weighting=True,
        surrogate_reference_beta=0.001,
        surrogate_reference_threshold=0.05,
    )
    return trainer


def test_tdm_weight_mixes_the_two_rewards_and_leaves_the_anchor_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Official TDM-R1 weights only the rewards; the KL anchor carries no coefficient.

    Reading ``tdm_weight`` as a coefficient on a single reward term, which is what the
    name suggests, changes the ratio between matching and reward by an order of
    magnitude and quietly stops reproducing the published recipe.
    """
    trainer = _generator_trainer(tdm_weight=0.3)
    terms = SimpleNamespace(loss=torch.tensor(5.0), boundary_state=None, x0_real=None)
    monkeypatch.setattr(TDMR1Trainer, "_generator_score_terms", lambda self, unit: terms)
    monkeypatch.setattr(TDMR1Trainer, "_stack_replay_unit", lambda self, samples: None)
    monkeypatch.setattr(
        TDMR1Trainer, "_guidance_reward_direction", lambda self, batch, terms: "guidance"
    )
    monkeypatch.setattr(
        TDMR1Trainer,
        "_surrogate_reward_direction",
        lambda self, batch, terms: ("surrogate", "normalizer"),
    )
    losses = {"guidance": torch.tensor(2.0), "surrogate": torch.tensor(10.0)}
    monkeypatch.setattr(
        tdm_r1_module,
        "revised_x0_loss",
        lambda adapter, student, direction, reference, **kwargs: losses[direction],
    )
    monkeypatch.setattr(tdm_r1_module, "record_distillation_metric", lambda *args: None)

    loss = trainer._generator_boundary_loss(SimpleNamespace(samples=()))

    assert float(loss) == pytest.approx(5.0 + 0.3 * 2.0 + 0.7 * 10.0)


@pytest.mark.parametrize("scale", [1.0, 4.5])
def test_a_cfg_free_reference_simply_has_no_guidance_reward(scale: float) -> None:
    """Refusing to run would break every CFG-free config that trained fine before.

    With no guidance on the reference there is no conditional-minus-unconditional
    direction to follow, so the reward does not exist for that run; the generator
    still has its distribution match and its learned reward.
    """
    trainer = _generator_trainer(tdm_weight=0.3)
    trainer.training_args.guidance_scale = scale
    trainer.training_args.get_reference_guidance_scale = lambda: scale
    trainer.training_args.cfg_reward_scale = 0.0 if scale > 1.0 else 3.5

    assert trainer._guidance_reward_direction({}, None) is None


@pytest.mark.parametrize(
    ("boundary_index", "expected"),
    [(0, 0.25), (1, 0.5), (3, 1.0)],
)
def test_later_boundaries_weigh_more(boundary_index: int, expected: float) -> None:
    """A boundary nearer the image the reward scored gets proportionally more say."""
    trainer = _generator_trainer(tdm_weight=0.3)
    unit = SimpleNamespace(boundary_index=boundary_index)

    assert trainer._time_weight(unit) == pytest.approx(expected)


def test_time_weighting_can_be_turned_off() -> None:
    """Every boundary counts alike when the ramp is disabled."""
    trainer = _generator_trainer(tdm_weight=0.3)
    trainer.training_args.use_time_weighting = False

    assert trainer._time_weight(SimpleNamespace(boundary_index=0)) == 1.0


def _preference_batch(advantages: torch.Tensor) -> GroupPreferenceBatch:
    return GroupPreferenceBatch(
        local_group_indices=torch.zeros(advantages.shape[0], dtype=torch.int64),
        num_groups=1,
        group_size=advantages.shape[0],
        advantages=advantages,
        reduce_across_ranks=False,
    )


def test_a_surrogate_that_has_drifted_far_enough_stops_receiving_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only samples already moved the way their advantage asks are frozen."""
    monkeypatch.setattr(tdm_r1_module, "record_distillation_metric", lambda *args: None)
    trainer = _generator_trainer(tdm_weight=0.3)
    # Sample 0 improved (lower loss) with a positive advantage and sits past the
    # threshold; sample 1 improved but is still close; sample 2 moved the wrong way.
    trainable = torch.tensor([0.0, 0.9, 2.0], requires_grad=True)
    reference = torch.tensor([1.0, 1.0, 1.0])
    advantages = torch.tensor([1.0, 1.0, 1.0])

    clipped = trainer._clip_runaway_surrogate(trainable, reference, _preference_batch(advantages))
    clipped.sum().backward()

    torch.testing.assert_close(trainable.grad, torch.tensor([0.0, 1.0, 1.0]))


def test_the_slow_copy_is_scored_on_the_same_draw_as_the_live_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two draws would make the ratio measure perturbation noise, not the step taken."""
    monkeypatch.setattr(tdm_r1_module, "record_distillation_metric", lambda *args: None)
    trainer = _generator_trainer(tdm_weight=0.3)
    trainer.training_args.surrogate_clip_range = 0.1
    trainer.adapter = SimpleNamespace(use_variant_snapshot=lambda name: _null_context())
    seen: list = []

    def record_context(self: Any, unit: Any, *, trainable_role: str, context: Any = None):
        seen.append(context)
        return torch.ones(2), torch.ones(2)

    monkeypatch.setattr(TDMR1Trainer, "_boundary_preference_values", record_context)
    sentinel = ("batch", "noised", "times")

    trainer._clip_outside_trust_region(
        SimpleNamespace(),
        torch.ones(2),
        _preference_batch(torch.ones(2)),
        context=sentinel,
    )

    assert seen == [sentinel]


def test_samples_that_left_the_trust_region_stop_accumulating_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PPO ratio is measured against the slow copy, not the frozen reference.

    Sample 0 now denoises far better than the slow copy did and has a positive
    advantage, so it is outside the region; sample 1 barely moved; sample 2 moved the
    way a negative advantage asks, which for that sample is also outside.
    """
    monkeypatch.setattr(tdm_r1_module, "record_distillation_metric", lambda *args: None)
    trainer = _generator_trainer(tdm_weight=0.3)
    trainer.training_args.surrogate_clip_range = 0.1
    slow = torch.tensor([1.0, 1.0, 1.0])
    trainer.adapter = SimpleNamespace(use_variant_snapshot=lambda name: _null_context())
    monkeypatch.setattr(
        TDMR1Trainer,
        "_boundary_preference_values",
        lambda self, unit, *, trainable_role, context=None: (slow, slow),
    )
    trainable = torch.tensor([0.5, 0.99, 1.5], requires_grad=True)
    advantages = torch.tensor([1.0, 1.0, -1.0])

    kept = trainer._clip_outside_trust_region(
        SimpleNamespace(), trainable, _preference_batch(advantages)
    )
    kept.sum().backward()

    torch.testing.assert_close(trainable.grad, torch.tensor([0.0, 1.0, 0.0]))


def test_a_zero_clip_range_skips_the_slow_copy_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling the trust region must not still pay for an extra score query."""
    trainer = _generator_trainer(tdm_weight=0.3)
    trainer.training_args.surrogate_clip_range = 0.0

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the slow surrogate must not be queried when clipping is off")

    monkeypatch.setattr(TDMR1Trainer, "_boundary_preference_values", refuse)
    trainable = torch.tensor([0.5, 0.99])

    kept = trainer._clip_outside_trust_region(
        SimpleNamespace(), trainable, _preference_batch(torch.tensor([1.0, 1.0]))
    )

    assert kept is trainable


def test_a_negative_advantage_reverses_which_direction_counts_as_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected sample improves by scoring worse, so the clip must flip with it."""
    monkeypatch.setattr(tdm_r1_module, "record_distillation_metric", lambda *args: None)
    trainer = _generator_trainer(tdm_weight=0.3)
    trainable = torch.tensor([2.0, 0.0], requires_grad=True)
    reference = torch.tensor([1.0, 1.0])
    advantages = torch.tensor([-1.0, -1.0])

    clipped = trainer._clip_runaway_surrogate(trainable, reference, _preference_batch(advantages))
    clipped.sum().backward()

    torch.testing.assert_close(trainable.grad, torch.tensor([0.0, 1.0]))
