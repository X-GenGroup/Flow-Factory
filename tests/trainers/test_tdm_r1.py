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

import inspect
from types import MethodType, SimpleNamespace
from typing import Dict, List, Tuple

import pytest
import torch

from flow_factory.hparams import Arguments
from flow_factory.samples import BaseSample
from flow_factory.trainers.distillation.group_preference import (
    GroupPreferenceBatch,
    group_preference_loss,
    reduce_group_sums,
)
from flow_factory.trainers.distillation.tdm import TDMTrainer
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer


class AcceleratorFake:
    """CPU accelerator fake with shape-addressed peer reduction contributions."""

    def __init__(
        self,
        *,
        num_processes: int = 1,
        peer_contributions: Dict[Tuple[int, ...], torch.Tensor] | None = None,
    ) -> None:
        self.num_processes = num_processes
        self.peer_contributions = peer_contributions or {}
        self.reduced: List[torch.Tensor] = []

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Return the local tensor plus a configured peer contribution."""
        assert reduction == "sum"
        self.reduced.append(tensor.detach().clone())
        peer = self.peer_contributions.get(tuple(tensor.shape))
        return tensor if peer is None else tensor + peer


def _batch(
    advantages: torch.Tensor,
    indices: torch.Tensor | None = None,
    *,
    group_size: int | None = None,
) -> GroupPreferenceBatch:
    indices = (
        indices if indices is not None else torch.zeros(advantages.shape[0], dtype=torch.int64)
    )
    num_groups = int(indices.max().item()) + 1
    return GroupPreferenceBatch(
        local_group_indices=indices,
        num_groups=num_groups,
        group_size=group_size or advantages.shape[0] // num_groups,
        advantages=advantages,
    )


def _preference_oracle(
    trainable_values: torch.Tensor,
    reference_values: torch.Tensor,
    advantages: torch.Tensor,
    group_indices: torch.Tensor,
    num_groups: int,
    beta: float,
    *,
    group_size: int | None = None,
    peer_group_sums: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reproduce the detached DGPO/TDM-R1 group-preference surrogate."""
    group_size = group_size or advantages.shape[0] // num_groups
    delta = trainable_values.detach() - reference_values.detach()
    weighted_delta = advantages.detach() * beta * delta / group_size
    group_sums = torch.zeros(num_groups, dtype=weighted_delta.dtype).scatter_add_(
        0, group_indices, weighted_delta
    )
    if peer_group_sums is not None:
        group_sums = group_sums + peer_group_sums
    group_weights = torch.sigmoid(group_sums)[group_indices].detach()
    return (group_weights * advantages.detach() * trainable_values).mean()


@pytest.mark.parametrize(
    "advantages",
    [
        torch.tensor([1.0, 2.0]),
        torch.tensor([-1.0, -2.0]),
    ],
)
def test_group_preference_default_mode_preserves_sign_incomplete_dgpo_groups(
    advantages: torch.Tensor,
) -> None:
    trainable_values = torch.tensor([0.25, 0.75], requires_grad=True)
    reference_values = torch.tensor([0.5, 0.5])
    batch = _batch(advantages, group_size=2)

    loss = group_preference_loss(
        AcceleratorFake(), batch, trainable_values, reference_values, beta=2.0
    )

    expected = _preference_oracle(
        trainable_values,
        reference_values,
        advantages,
        batch.local_group_indices,
        batch.num_groups,
        2.0,
    )
    assert torch.equal(loss, expected)


def test_strict_group_preference_packs_membership_and_logits_into_one_reduce() -> None:
    accelerator = AcceleratorFake(num_processes=2)
    advantages = torch.tensor([1.0, -1.0])
    batch = _batch(advantages, group_size=2)
    trainable_values = torch.tensor([0.25, 0.75], requires_grad=True)

    loss = group_preference_loss(
        accelerator,
        batch,
        trainable_values,
        torch.tensor([0.5, 0.5]),
        beta=1.0,
        require_both_signs=True,
    )
    loss.backward()

    assert len(accelerator.reduced) == 1
    assert accelerator.reduced[0].shape == (1, 3)
    assert torch.isfinite(loss)
    assert torch.isfinite(trainable_values.grad).all()


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), float("-inf")])
def test_group_preference_rejects_a_nonfinite_beta(beta: float) -> None:
    with pytest.raises(ValueError, match=r"finite beta"):
        group_preference_loss(
            AcceleratorFake(),
            _batch(torch.tensor([1.0, -1.0])),
            torch.ones(2, requires_grad=True),
            torch.zeros(2),
            beta=beta,
        )


@pytest.mark.parametrize("group_size", [0, -1, True, 1.5])
def test_group_preference_rejects_an_invalid_configured_group_size(
    group_size: object,
) -> None:
    advantages = torch.tensor([1.0, -1.0])
    batch = GroupPreferenceBatch(
        local_group_indices=torch.tensor([0, 0], dtype=torch.int64),
        num_groups=1,
        group_size=group_size,  # type: ignore[arg-type]
        advantages=advantages,
    )
    error_type = TypeError if isinstance(group_size, (bool, float)) else ValueError

    with pytest.raises(error_type, match=r"positive non-bool int.*group_size"):
        group_preference_loss(
            AcceleratorFake(),
            batch,
            torch.ones(2, requires_grad=True),
            torch.zeros(2),
            beta=1.0,
        )


def test_group_preference_rejects_mismatched_shapes_dtypes_and_devices() -> None:
    batch = _batch(torch.tensor([1.0, -1.0]))
    trainable_values = torch.ones(2, requires_grad=True)

    with pytest.raises(ValueError, match=r"same shape"):
        group_preference_loss(AcceleratorFake(), batch, trainable_values, torch.zeros(3), beta=1.0)
    with pytest.raises(TypeError, match=r"same dtype"):
        group_preference_loss(
            AcceleratorFake(),
            batch,
            trainable_values,
            torch.zeros(2, dtype=torch.float64),
            beta=1.0,
        )
    with pytest.raises(ValueError, match=r"same device"):
        group_preference_loss(
            AcceleratorFake(),
            batch,
            trainable_values,
            torch.empty(2, device="meta"),
            beta=1.0,
        )


@pytest.mark.parametrize(
    ("indices", "num_groups", "error_type", "message"),
    [
        (torch.tensor([0.0, 0.0]), 1, TypeError, "int64"),
        (torch.tensor([[0, 0]], dtype=torch.int64), 1, ValueError, "one-dimensional"),
        (torch.tensor([0, 2], dtype=torch.int64), 3, ValueError, "dense"),
        (torch.tensor([0, 0], dtype=torch.int64), 2, ValueError, "dense"),
        (torch.tensor([0, 0], dtype=torch.int64), 0, ValueError, "positive num_groups"),
    ],
)
def test_reduce_group_sums_validates_dense_indices_and_num_groups(
    indices: torch.Tensor,
    num_groups: int,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        reduce_group_sums(AcceleratorFake(), torch.ones(2), indices, num_groups)


def test_reduce_group_sums_rejects_a_cross_rank_shape_change() -> None:
    class WrongShapeAccelerator(AcceleratorFake):
        def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
            """Return an invalid collective shape."""
            return torch.zeros(tensor.shape[0] + 1, dtype=tensor.dtype)

    with pytest.raises(ValueError, match=r"cross-rank reduction.*shape"):
        reduce_group_sums(
            WrongShapeAccelerator(num_processes=2),
            torch.tensor([1.0, 2.0]),
            torch.tensor([0, 0], dtype=torch.int64),
            1,
        )


def test_tdm_r1_inherits_tdm_math_without_guidance_mandate() -> None:
    assert TDMR1Trainer.__bases__ == (TDMTrainer,)
    assert TDMR1Trainer.paradigm == "decoupled"
    source = inspect.getsource(TDMR1Trainer)
    assert "_forward_guidance_branches" not in source
    assert "interleaved_microbatch" not in source
    assert "old_surrogate" not in source


def test_tdm_r1_constructs_without_guidance_branch_hook() -> None:
    trainer = object.__new__(TDMR1Trainer)
    trainer.training_args = SimpleNamespace(
        ttur_fake_updates=2,
        gradient_accumulation_steps=1,
        per_device_batch_size=1,
        num_inference_steps=1,
        num_inner_epochs=1,
    )
    trainer.training_args.get_num_train_timesteps = lambda config: 1
    trainer.config = SimpleNamespace()
    events: list[str] = []
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.adapter = SimpleNamespace(train=lambda: None, scheduler_group={})
    trainer._build_boundary_units = MethodType(
        lambda self, samples: [SimpleNamespace(boundary_index=1, samples=tuple(samples))],
        trainer,
    )
    trainer._fake_phase = MethodType(lambda self, units: events.append("fake"), trainer)
    trainer._surrogate_phase = MethodType(lambda self, units: events.append("surrogate"), trainer)
    trainer._generator_phase = MethodType(lambda self, units: events.append("generator"), trainer)

    trainer.optimize([BaseSample(timesteps=torch.tensor([1.0]), all_latents=torch.zeros(2, 1))])

    assert events == ["fake", "fake", "surrogate", "generator"]


def test_tdm_r1_replaces_a_sampler_that_scatters_group_members() -> None:
    """distributed_k_repeat leaves each rank a different set of partial groups."""
    config = Arguments.from_dict(
        {
            "data": {"sampler_type": "distributed_k_repeat"},
            "train": {
                "trainer_type": "tdm-r1",
                "group_size": 2,
                "per_device_batch_size": 2,
                "num_inference_steps": 1,
            },
            "scheduler": {"dynamics_type": "ODE"},
            "rewards": [{"name": "score", "reward_model": "clip"}],
        }
    )
    assert config.data_args.sampler_type == "group_contiguous"


def test_tdm_r1_sample_submits_dense_trajectory_endpoints_to_reward_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(TDMR1Trainer)
    trainer.training_args = SimpleNamespace(num_inference_steps=3)
    trainer.reward_buffer = object()
    received: dict[str, object] = {}

    def generate_one_rollout_batch(received_trainer: TDMR1Trainer, **kwargs: object) -> list:
        assert received_trainer is trainer
        received.update(kwargs)
        return []

    monkeypatch.setattr(
        "flow_factory.trainers.distillation.tdm_r1.generate_one_rollout_batch",
        generate_one_rollout_batch,
    )
    trainer._validate_trajectory_configuration = MethodType(lambda self: None, trainer)

    assert trainer.sample() == []
    assert received == {
        "reward_buffer": trainer.reward_buffer,
        "compute_log_prob": False,
        "trajectory_indices": [0, 1, 2, 3],
        "algorithm_name": "TDM-R1",
    }


def test_tdm_r1_feedback_uses_reward_buffer_and_configured_advantages() -> None:
    trainer = object.__new__(TDMR1Trainer)
    samples = [
        BaseSample(timesteps=torch.tensor([1.0]), all_latents=torch.zeros(2, 1)),
        BaseSample(timesteps=torch.tensor([1.0]), all_latents=torch.zeros(2, 1)),
    ]
    rewards = {"quality": torch.tensor([1.0, 3.0])}
    calls: list[tuple[str, object]] = []

    class RewardBufferFake:
        def finalize(self, *, store_to_samples: bool, split: str):
            calls.append(("finalize", (store_to_samples, split)))
            return rewards

    class AdvantageProcessorFake:
        def compute_advantages(self, **kwargs: object):
            calls.append(("advantages", kwargs))
            return torch.tensor([-1.0, 1.0])

        def pop_advantage_metrics(self):
            return {}

    trainer.reward_buffer = RewardBufferFake()
    trainer.advantage_processor = AdvantageProcessorFake()
    trainer.training_args = SimpleNamespace(advantage_aggregation="gdpo")
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.log_data = lambda *args, **kwargs: None

    trainer.prepare_feedback(samples)

    assert calls[0] == ("finalize", (True, "all"))
    assert calls[1][0] == "advantages"
    assert calls[1][1]["aggregation_func"] == "gdpo"


def _preference_trainer(sampler_type: str, *, num_processes: int = 1) -> TDMR1Trainer:
    """Build the trainer surface the group-preference batch reads."""
    trainer = object.__new__(TDMR1Trainer)
    trainer.training_args = SimpleNamespace(group_size=2, advantage_clip_range=5.0)
    trainer.config = SimpleNamespace(data_args=SimpleNamespace(sampler_type=sampler_type))
    trainer.accelerator = SimpleNamespace(
        num_processes=num_processes,
        reduce=lambda tensor, reduction: tensor,
    )
    return trainer


def test_tdm_r1_group_preference_batch_is_rank_local() -> None:
    trainer = _preference_trainer("group_contiguous")
    unit = SimpleNamespace(
        samples=(
            SimpleNamespace(unique_id=7, extra_kwargs={"advantage": 1.0}),
            SimpleNamespace(unique_id=7, extra_kwargs={"advantage": -0.5}),
        )
    )
    values = torch.tensor([0.25, 0.75])

    batch = trainer._group_preference_batch(unit, values)

    assert batch.num_groups == 1
    assert batch.reduce_across_ranks is False
    assert batch.group_size == 2
    torch.testing.assert_close(batch.local_group_indices, torch.zeros(2, dtype=torch.int64))


def test_tdm_r1_sums_group_logits_across_ranks_when_the_group_is_split() -> None:
    """Under group_distributed a rank holds group_size // num_replicas of each group."""
    trainer = _preference_trainer("group_distributed", num_processes=2)
    unit = SimpleNamespace(
        samples=(
            SimpleNamespace(unique_id=7, extra_kwargs={"advantage": 1.0}),
            SimpleNamespace(unique_id=9, extra_kwargs={"advantage": -0.5}),
        )
    )
    values = torch.tensor([0.25, 0.75])

    batch = trainer._group_preference_batch(unit, values)

    assert batch.num_groups == 2
    assert batch.reduce_across_ranks is True
    assert batch.group_size == 2
    torch.testing.assert_close(batch.local_group_indices, torch.tensor([0, 1]))


def test_tdm_r1_supports_packed_groups_smaller_than_world_size() -> None:
    trainer = _preference_trainer("group_distributed", num_processes=4)
    trainer.accelerator.gather = lambda _local: torch.tensor([7, 7, 9, 9])
    unit = SimpleNamespace(samples=(SimpleNamespace(unique_id=7, extra_kwargs={"advantage": 1.0}),))

    batch = trainer._group_preference_batch(unit, torch.tensor([0.25]))

    assert batch.num_groups == 2
    assert batch.reduce_across_ranks is True
    assert batch.allow_sparse_local_groups is True
    torch.testing.assert_close(batch.local_group_indices, torch.tensor([0]))


def test_tdm_r1_overlap_reuses_cached_group_indices_without_gathering() -> None:
    trainer = _preference_trainer("group_distributed", num_processes=4)
    trainer.accelerator.gather = lambda _local: (_ for _ in ()).throw(
        AssertionError("cached overlap group metadata must avoid an extra gather")
    )
    trainer._tdm_r1_reward_overlap_group_info = SimpleNamespace(
        local_unique_ids=torch.tensor([7]),
        local_group_indices=torch.tensor([0]),
        num_groups=2,
    )
    unit = SimpleNamespace(samples=(SimpleNamespace(unique_id=7, extra_kwargs={"advantage": 1.0}),))

    batch = trainer._group_preference_batch(unit, torch.tensor([0.25]))

    assert batch.num_groups == 2
    assert batch.reduce_across_ranks is True
    assert batch.allow_sparse_local_groups is True
    torch.testing.assert_close(batch.local_group_indices, torch.tensor([0]))


def test_tdm_r1_rejects_a_microbatch_holding_the_wrong_share_of_a_split_group() -> None:
    """Two members of one group on one rank means another rank has none of it."""
    trainer = _preference_trainer("group_distributed", num_processes=2)
    unit = SimpleNamespace(
        samples=(
            SimpleNamespace(unique_id=7, extra_kwargs={"advantage": 1.0}),
            SimpleNamespace(unique_id=7, extra_kwargs={"advantage": -0.5}),
        )
    )

    with pytest.raises(ValueError, match="exactly 1 members"):
        trainer._group_preference_batch(unit, torch.tensor([0.25, 0.75]))


def test_tdm_r1_rejects_ranks_that_disagree_on_the_group_id_space() -> None:
    """Summing group g on one rank into group g on another would mix unrelated samples."""
    trainer = _preference_trainer("group_distributed", num_processes=2)
    trainer.accelerator.reduce = lambda tensor, reduction: tensor + 1.0
    unit = SimpleNamespace(
        samples=(
            SimpleNamespace(unique_id=7, extra_kwargs={"advantage": 1.0}),
            SimpleNamespace(unique_id=9, extra_kwargs={"advantage": -0.5}),
        )
    )

    with pytest.raises(RuntimeError, match="same reward groups"):
        trainer._group_preference_batch(unit, torch.tensor([0.25, 0.75]))


def test_group_preference_rank_local_mode_skips_cross_rank_reduce() -> None:
    class ReduceMustNotRun:
        num_processes = 2

        def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
            raise AssertionError(
                f"rank-local preference must not call accelerator.reduce, "
                f"received reduction={reduction!r}"
            )

    advantages = torch.tensor([1.0, -1.0])
    batch = GroupPreferenceBatch(
        local_group_indices=torch.zeros(2, dtype=torch.int64),
        num_groups=1,
        group_size=2,
        advantages=advantages,
        reduce_across_ranks=False,
    )
    trainable_values = torch.tensor([0.25, 0.75], requires_grad=True)
    loss = group_preference_loss(
        ReduceMustNotRun(),
        batch,
        trainable_values,
        torch.tensor([0.5, 0.5]),
        beta=1.0,
    )
    assert torch.isfinite(loss)
