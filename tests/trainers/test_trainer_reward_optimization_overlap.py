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

"""Trainer capability tests for streamed reward-driven optimization."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flow_factory.contracts.execution import ONLINE_EXECUTION_CONTRACT
from flow_factory.rewards import (
    RewardTile,
    RewardTileGeometry,
    RewardTilePlan,
    build_reward_tile_plan,
)
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.registry import get_trainer_class, list_registered_trainers
from flow_factory.trainers.rl.dpo import DPOTrainer


class _TrainingArguments:
    reward_optimization_overlap = True
    reward_optimization_overlap_mode = "ordered"
    advantage_aggregation = "sum"
    global_std = False
    num_inner_epochs = 1
    shuffle_samples = False
    gradient_accumulation_steps = 4
    num_batches_per_epoch = 4
    per_device_batch_size = 1
    group_size = 1

    @staticmethod
    def get_num_train_timesteps(_config: SimpleNamespace) -> int:
        return 2


def _config(**training_overrides: object) -> SimpleNamespace:
    training_args = _TrainingArguments()
    for name, value in training_overrides.items():
        setattr(training_args, name, value)
    return SimpleNamespace(
        training_args=training_args,
        data_args=SimpleNamespace(sampler_type="group_contiguous"),
        reward_args=[SimpleNamespace(name="remote", async_reward=True, device=torch.device("cpu"))],
    )


def test_only_declared_trainers_expose_reward_optimization_overlap() -> None:
    supported = {
        name
        for name in list_registered_trainers()
        if get_trainer_class(name).reward_optimization_overlap_contract.supported
    }

    assert supported == {
        "awm",
        "crd",
        "dgpo",
        "dpo",
        "dppo",
        "grpo",
        "grpo-guard",
        "nft",
        "tdm-r1",
    }


def test_grpo_accepts_the_streaming_contract_geometry() -> None:
    trainer_cls = get_trainer_class("grpo")

    trainer_cls.validate_reward_optimization_overlap(_config())


def test_grpo_accepts_group_local_gdpo_overlap() -> None:
    trainer_cls = get_trainer_class("grpo")

    trainer_cls.validate_reward_optimization_overlap(_config(advantage_aggregation="gdpo"))


@pytest.mark.parametrize("trainer_name", ["nft", "awm", "crd", "dpo"])
def test_rank_local_group_relative_trainers_accept_overlap(trainer_name: str) -> None:
    get_trainer_class(trainer_name).validate_reward_optimization_overlap(_config())


@pytest.mark.parametrize("trainer_name", ["dgpo", "tdm-r1"])
def test_cross_rank_group_relative_trainers_accept_overlap(trainer_name: str) -> None:
    config = _config()
    config.data_args.sampler_type = "group_distributed"

    get_trainer_class(trainer_name).validate_reward_optimization_overlap(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"advantage_aggregation": "custom"}, "'sum' or 'gdpo'"),
        ({"global_std": True}, "global_std=false"),
        ({"num_inner_epochs": 2}, "num_inner_epochs=1"),
        ({"shuffle_samples": True}, "shuffle_samples=false"),
        ({"gradient_accumulation_steps": 3}, "complete gradient accumulation"),
    ],
)
def test_grpo_rejects_non_streamable_objective_geometry(
    overrides: dict[str, object],
    message: str,
) -> None:
    trainer_cls = get_trainer_class("grpo")

    with pytest.raises(ValueError, match=message):
        trainer_cls.validate_reward_optimization_overlap(_config(**overrides))


def test_unsupported_algorithm_rejects_overlap_before_runtime() -> None:
    trainer_cls = get_trainer_class("sft")

    with pytest.raises(ValueError, match="does not support"):
        trainer_cls.validate_reward_optimization_overlap(_config())


def test_overlap_rejects_a_reward_client_on_the_training_gpu() -> None:
    trainer_cls = get_trainer_class("grpo")
    config = _config()
    config.reward_args[0].device = torch.device("cuda")

    with pytest.raises(ValueError, match="device='cpu'"):
        trainer_cls.validate_reward_optimization_overlap(config)


class _GenerationTimingHarness:
    execution_contract = ONLINE_EXECUTION_CONTRACT

    def __init__(self) -> None:
        self.training_args = SimpleNamespace(reward_optimization_overlap=False)
        self.accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1)
        self.events: list[str] = []
        self.logged: dict[str, float] = {}
        self.step = 7

    def sampling_context(self):
        return nullcontext()

    def sample(self) -> list[object]:
        self.events.append("sample")
        return [object()]

    def prepare_feedback(self, _samples: list[object]) -> None:
        self.events.append("feedback")

    def optimize(self, _samples: list[object]) -> None:
        self.events.append("optimize")

    def _critical_path_timing_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return BaseTrainer._critical_path_timing_metrics(self, metrics)

    def log_data(self, metrics: dict[str, float], step: int) -> None:
        assert step == self.step
        self.logged = metrics


def test_standard_generation_logs_separately_grouped_stage_timings(monkeypatch) -> None:
    clock = iter((0.0, 2.0, 2.0, 5.0, 5.0, 11.0, 11.0))
    monkeypatch.setattr("flow_factory.trainers.abc.time.monotonic", lambda: next(clock))
    trainer = _GenerationTimingHarness()

    BaseTrainer._run_training_step(trainer)

    assert trainer.events == ["sample", "feedback", "optimize"]
    assert trainer.logged == {
        "timing/rollout_seconds": 2.0,
        "timing/feedback_seconds": 3.0,
        "timing/optimization_seconds": 6.0,
        "timing/cycle_seconds": 11.0,
    }


def test_timing_metrics_report_distributed_critical_path_maxima() -> None:
    def reduce_max(_local: torch.Tensor, reduction: str) -> torch.Tensor:
        assert reduction == "max"
        return torch.tensor([3.0, 5.0], dtype=torch.float64)

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=2,
        reduce=reduce_max,
    )
    trainer = SimpleNamespace(accelerator=accelerator)

    reduced = BaseTrainer._critical_path_timing_metrics(
        trainer,
        {
            "timing/rollout_seconds": 2.0,
            "timing/cycle_seconds": 5.0,
        },
    )

    assert reduced == {
        "timing/rollout_seconds": 3.0,
        "timing/cycle_seconds": 5.0,
    }


def test_ready_overlap_mode_bypasses_an_earlier_straggler_tile() -> None:
    selected = BaseTrainer._select_reward_overlap_tile(
        pending={0, 1, 2},
        globally_ready={1, 2},
        mode="ready",
    )

    assert selected == 1


def test_ordered_overlap_mode_waits_for_the_earliest_pending_tile() -> None:
    selected = BaseTrainer._select_reward_overlap_tile(
        pending={0, 1, 2},
        globally_ready={1, 2},
        mode="ordered",
    )

    assert selected is None


def test_ordered_overlap_only_coordinates_the_head_tile() -> None:
    candidates = BaseTrainer._reward_overlap_readiness_candidates(
        pending={7, 3, 5},
        mode="ordered",
    )

    assert candidates == (3,)


def test_ready_overlap_coordinates_every_pending_tile_in_stable_order() -> None:
    candidates = BaseTrainer._reward_overlap_readiness_candidates(
        pending={7, 3, 5},
        mode="ready",
    )

    assert candidates == (3, 5, 7)


def test_group_local_gdpo_optimizes_each_ready_tile() -> None:
    samples = [SimpleNamespace(extra_kwargs={}) for _ in range(4)]
    plan = RewardTilePlan(
        sample_count=4,
        samples_per_tile=2,
        batches_per_tile=2,
        geometry=RewardTileGeometry(),
        tiles=(
            RewardTile(tile_id=0, start=0, stop=2, group_ids=(0,)),
            RewardTile(tile_id=1, start=2, stop=4, group_ids=(1,)),
        ),
    )
    events: list[str] = []

    class RewardBuffer:
        consumed: set[int] = set()

        @staticmethod
        def poll_ready_tiles(_tiles):
            return {0, 1}

        def resolve_streaming_tile(self, indices):
            assert indices in ((0, 1), (2, 3))
            assert not self.consumed.intersection(indices)
            self.consumed.update(indices)
            events.append(f"resolve:{indices[0] // 2}")
            return {"reward": torch.ones(2)}

        def finish_streaming(self):
            assert self.consumed == {0, 1, 2, 3}
            events.append("finish")
            return {"reward": torch.ones(4)}

        @staticmethod
        def abort_streaming():
            events.append("abort")

    class Harness:
        training_args = SimpleNamespace(
            advantage_aggregation="gdpo",
            global_std=False,
            reward_optimization_overlap_mode="ready",
            reward_optimization_overlap_poll_interval=0.0,
        )
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            reduce=lambda value, reduction: value,
        )
        reward_buffer = RewardBuffer()
        advantage_processor = SimpleNamespace(pop_advantage_metrics=lambda: {})
        _reward_overlap_group_infos_by_tile = {}
        step = 0

        def _build_and_seal_reward_tile_plan(self, _samples):
            return plan

        def _prepare_reward_optimization_overlap(self, _samples, _plan):
            return None

        def _synchronize_reward_overlap_error(self, _phase, error):
            if error is not None:
                raise error

        def _compute_synchronized_reward_overlap_advantages(
            self,
            prepared_samples,
            _rewards,
            *,
            phase,
            build_metrics,
        ):
            assert phase == "final advantage"
            assert build_metrics
            events.append("final-advantage")
            for sample in prepared_samples:
                sample.extra_kwargs["advantage"] = torch.tensor(1.0)

        def _resolve_reward_overlap_tile_feedback(self, tile, tile_samples):
            self.reward_buffer.resolve_streaming_tile(tile.sample_indices)
            events.append(f"advantage:{tile.tile_id}")
            for sample in tile_samples:
                sample.extra_kwargs["advantage"] = torch.tensor(1.0)

        def _optimize_reward_overlap_tile(self, tile, tile_samples, _context):
            assert all("advantage" in sample.extra_kwargs for sample in tile_samples)
            events.append(f"optimize:{tile.tile_id}")

        def _finalize_reward_optimization_overlap(self, _context, _samples):
            return {}

        def _abort_reward_optimization_overlap(self, _context):
            events.append("abort-context")

        def _critical_path_timing_metrics(self, metrics):
            return BaseTrainer._critical_path_timing_metrics(self, metrics)

        def log_data(self, _metrics, step):
            assert step == 0

        _finish_reward_overlap_stream = BaseTrainer._finish_reward_overlap_stream
        _reward_overlap_readiness_candidates = staticmethod(
            BaseTrainer._reward_overlap_readiness_candidates
        )
        _select_reward_overlap_tile = staticmethod(BaseTrainer._select_reward_overlap_tile)

    BaseTrainer._run_reward_optimization_overlap(
        Harness(),
        samples,
        cycle_started=0.0,
        rollout_seconds=0.0,
    )

    assert events == [
        "resolve:0",
        "advantage:0",
        "optimize:0",
        "resolve:1",
        "advantage:1",
        "optimize:1",
        "finish",
        "final-advantage",
    ]


def test_cross_rank_plan_validates_fixed_header_before_gathering_uids() -> None:
    plan = RewardTilePlan(
        sample_count=2,
        samples_per_tile=1,
        batches_per_tile=1,
        geometry=RewardTileGeometry(group_layout="cross_rank_sharded"),
        tiles=(
            RewardTile(tile_id=0, start=0, stop=1, group_ids=(7,)),
            RewardTile(tile_id=1, start=1, stop=2, group_ids=(9,)),
        ),
    )
    rank_headers = torch.tensor(
        [
            [2, 1, 1, 2, 2, 1],
            [2, 1, 1, 2, 2, 1],
            [2, 1, 1, 2, 2, 1],
            [2, 1, 1, 2, 2, 1],
        ],
        dtype=torch.int64,
    )
    rank_uids = torch.tensor(
        [[7, 9], [7, 9], [8, 10], [8, 10]],
        dtype=torch.int64,
    )
    gather_calls: list[torch.Tensor] = []

    def gather(local: torch.Tensor) -> torch.Tensor:
        gather_calls.append(local.clone())
        if local.numel() == rank_headers.shape[1]:
            return rank_headers.flatten()
        return rank_uids.flatten()

    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=4,
            process_index=0,
            gather=gather,
        ),
        training_args=SimpleNamespace(group_size=2, per_device_batch_size=1),
        _synchronize_reward_overlap_error=lambda _phase, error: (
            (_ for _ in ()).throw(error) if error is not None else None
        ),
    )
    samples = [SimpleNamespace(unique_id=7), SimpleNamespace(unique_id=9)]

    BaseTrainer._validate_distributed_reward_tile_plan(trainer, plan, samples)

    assert len(gather_calls) == 2
    torch.testing.assert_close(gather_calls[0], rank_headers[0])
    torch.testing.assert_close(gather_calls[1], rank_uids[0])
    cached = trainer._reward_overlap_group_infos_by_tile
    assert tuple(cached) == (0, 1)
    assert cached[0][0].num_groups == 2
    torch.testing.assert_close(cached[0][0].local_unique_ids, torch.tensor([7]))
    torch.testing.assert_close(cached[0][0].local_group_indices, torch.tensor([0]))


def test_cross_rank_plan_rejects_sample_count_mismatch_before_uid_gather() -> None:
    plan = RewardTilePlan(
        sample_count=2,
        samples_per_tile=1,
        batches_per_tile=1,
        geometry=RewardTileGeometry(group_layout="cross_rank_sharded"),
        tiles=(
            RewardTile(tile_id=0, start=0, stop=1, group_ids=(7,)),
            RewardTile(tile_id=1, start=1, stop=2, group_ids=(9,)),
        ),
    )
    rank_headers = torch.tensor(
        [
            [2, 1, 1, 2, 2, 1],
            [3, 1, 1, 3, 3, 1],
        ],
        dtype=torch.int64,
    )
    gather_calls: list[torch.Tensor] = []

    def gather(local: torch.Tensor) -> torch.Tensor:
        gather_calls.append(local.clone())
        if len(gather_calls) > 1:
            raise AssertionError("UID gather must not run after a header mismatch")
        return rank_headers.flatten()

    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=2,
            process_index=0,
            gather=gather,
        ),
        training_args=SimpleNamespace(group_size=2, per_device_batch_size=1),
    )
    samples = [SimpleNamespace(unique_id=7), SimpleNamespace(unique_id=9)]

    with pytest.raises(RuntimeError, match="geometry differs across ranks"):
        BaseTrainer._validate_distributed_reward_tile_plan(trainer, plan, samples)

    assert len(gather_calls) == 1


def test_bagel_style_packed_batches_remain_intact_across_tiles() -> None:
    samples = [
        SimpleNamespace(unique_id=group) for group in range(2) for _sample_in_group in range(4)
    ]
    plan = build_reward_tile_plan(
        samples,
        group_size=4,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        optimizer_terms_per_batch=1,
    )
    rollout_batches = tuple(
        tuple(id(sample) for sample in samples[start : start + 2])
        for start in range(0, len(samples), 2)
    )
    trainer = SimpleNamespace(
        adapter=SimpleNamespace(requires_preserved_replay_batch_composition=True),
        training_args=SimpleNamespace(group_size=4, per_device_batch_size=2),
        _reward_overlap_rollout_batch_sample_ids=rollout_batches,
    )

    BaseTrainer._validate_reward_overlap_replay_batch_composition(trainer, plan, samples)

    assert [(tile.start, tile.stop) for tile in plan.tiles] == [(0, 4), (4, 8)]


def test_bagel_style_packed_batch_cannot_be_split_by_a_tile() -> None:
    samples = [SimpleNamespace(unique_id=index // 2) for index in range(6)]
    plan = RewardTilePlan(
        sample_count=6,
        samples_per_tile=3,
        batches_per_tile=1,
        geometry=RewardTileGeometry(),
        tiles=(
            RewardTile(tile_id=0, start=0, stop=3, group_ids=(0, 1)),
            RewardTile(tile_id=1, start=3, stop=6, group_ids=(1, 2)),
        ),
    )
    trainer = SimpleNamespace(
        adapter=SimpleNamespace(requires_preserved_replay_batch_composition=True),
        training_args=SimpleNamespace(group_size=2, per_device_batch_size=2),
        _reward_overlap_rollout_batch_sample_ids=tuple(
            tuple(id(sample) for sample in samples[start : start + 2])
            for start in range(0, len(samples), 2)
        ),
    )

    with pytest.raises(ValueError, match="splits a pack-composition-dependent"):
        BaseTrainer._validate_reward_overlap_replay_batch_composition(trainer, plan, samples)


def test_tile_reward_resolution_failure_is_synchronized_before_advantage_preparation() -> None:
    events: list[str] = []

    def synchronize(phase: str, error: Exception | None) -> None:
        events.append(f"sync:{phase}")
        if error is not None:
            raise RuntimeError("synchronized resolution failure") from error

    trainer = SimpleNamespace(
        reward_buffer=SimpleNamespace(
            resolve_streaming_tile=lambda _indices: (_ for _ in ()).throw(
                ValueError("rank-local reward failure")
            )
        ),
        _synchronize_reward_overlap_error=synchronize,
        _compute_synchronized_reward_overlap_advantages=lambda *_args, **_kwargs: events.append(
            "advantage"
        ),
    )
    tile = RewardTile(tile_id=0, start=0, stop=1, group_ids=(7,))

    with pytest.raises(RuntimeError, match="synchronized resolution failure"):
        BaseTrainer._resolve_reward_overlap_tile_feedback(trainer, tile, [object()])

    assert events == ["sync:tile reward resolution"]


def test_advantage_preparation_failure_is_synchronized_before_compute() -> None:
    events: list[str] = []

    def synchronize(phase: str, error: Exception | None) -> None:
        events.append(f"sync:{phase}")
        if error is not None:
            raise RuntimeError("synchronized preparation failure") from error

    trainer = SimpleNamespace(
        advantage_processor=SimpleNamespace(
            prepare_group_reward_collection=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("invalid local reward payload")
            )
        ),
        _synchronize_reward_overlap_error=synchronize,
        _compute_reward_overlap_advantages=lambda *_args, **_kwargs: events.append("compute"),
    )

    with pytest.raises(RuntimeError, match="synchronized preparation failure"):
        BaseTrainer._compute_synchronized_reward_overlap_advantages(
            trainer,
            [object()],
            {"reward": torch.tensor([1.0])},
            phase="tile advantage",
            build_metrics=False,
        )

    assert events == ["sync:tile advantage preparation"]


def test_dpo_overlap_tile_skips_redundant_pair_collectives() -> None:
    trainer = object.__new__(DPOTrainer)
    trainer.advantage_processor = SimpleNamespace(group_on_same_rank=True)
    trainer.accelerator = SimpleNamespace(
        num_processes=32,
        reduce=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("partial overlap pair metrics must not be reduced")
        ),
    )
    trainer._dpo_reward_overlap_tile_active = True
    samples = [
        SimpleNamespace(unique_id=7, extra_kwargs={"advantage": torch.tensor(-1.0)}),
        SimpleNamespace(unique_id=7, extra_kwargs={"advantage": torch.tensor(1.0)}),
    ]

    pairs, metrics = trainer._form_pairs(samples)
    aligned = trainer._align_dpo_pairs_across_ranks(pairs)

    assert aligned == pairs
    assert len(pairs) == 1
    assert metrics == {"train/dpo_num_pairs": 32}
