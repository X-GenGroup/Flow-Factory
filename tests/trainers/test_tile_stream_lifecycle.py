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

"""Controlled multi-cycle traces through the production tile stream and feedback."""

import threading
from types import SimpleNamespace

import pytest
import torch

from flow_factory.advantage import AdvantageProcessor
from flow_factory.hparams import ParallelismArguments, RewardArguments
from flow_factory.rewards import (
    PointwiseRewardModel,
    RewardBuffer,
    RewardModelOutput,
    RewardProcessor,
)
from flow_factory.samples import BaseSample
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.execution import TrainingProgress


class _ControlledReward(PointwiseRewardModel):
    required_fields = ("prompt",)

    def __call__(self, prompt):
        tile = int(prompt[0]) // 2
        assert self.gates[tile].wait(30), "test did not release the reward request"
        return RewardModelOutput(rewards=torch.tensor([float(value) for value in prompt]))


class _StreamTrainer(BaseTrainer):
    """Replace model acquisition/objective only; retain shared lifecycle and feedback."""

    def __init__(self, mode, failure=None):
        self.progress = TrainingProgress()
        self.failure = failure
        self.events = []
        self.metrics = []
        self.worker_threads = []
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.1)
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            process_index=0,
            num_processes=1,
            is_local_main_process=False,
            reduce=lambda value, reduction: value,
            gather=lambda value: value,
            wait_for_everyone=lambda: None,
        )
        self.adapter = SimpleNamespace(
            requires_preserved_replay_batch_composition=False,
            set_trajectory_seed=lambda seed: None,
            ema_step=lambda step: self.events.append((step, "ema")),
        )
        self.training_args = SimpleNamespace(
            seed=0,
            reward_optimization_overlap=True,
            reward_optimization_overlap_mode=mode,
            reward_optimization_overlap_poll_interval=0.001,
            group_size=2,
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
            get_num_train_timesteps=lambda config: 1,
            advantage_aggregation="sum",
        )
        self.config = SimpleNamespace(
            training_args=self.training_args,
            data_args=SimpleNamespace(sampler_type="group_contiguous"),
            parallelism_args=ParallelismArguments(),
        )
        self.log_args = SimpleNamespace(save_freq=0, save_dir=None, verbose=False)
        self.eval_args = SimpleNamespace(eval_freq=0)
        self._reward_overlap_acquisition_manifest = None
        reward_args = RewardArguments(
            name="score", device="cpu", batch_size=2, async_reward=True, num_workers=3
        )
        self.reward = _ControlledReward(reward_args, self.accelerator)
        self.reward_buffer = RewardBuffer(
            RewardProcessor(
                self.accelerator,
                {"score": self.reward},
                {"score": reward_args},
                group_on_same_rank=True,
                verbose=False,
            ),
            group_size=2,
        )
        self.advantage_processor = AdvantageProcessor(
            self.accelerator,
            {"score": {"a": 1.0, "b": 1.0}},
            group_size=2,
            global_std=False,
            sampler_type="group_contiguous",
            verbose=False,
            source_id_to_name=["a", "b"],
        )

    def should_continue_training(self):
        return self.epoch < 3

    def sample(self):
        self.reward_buffer.clear()
        self.reward.gates = [threading.Event() for _ in range(3)]
        self.first_tile = 1 if self.training_args.reward_optimization_overlap_mode == "ready" else 0
        self.reward.gates[self.first_tile].set()
        policy = self.parameter.detach().clone()
        samples = []
        for tile in range(3):
            assert torch.equal(self.parameter, policy)
            batch = [
                BaseSample(
                    prompt=str(tile * 2 + member),
                    source_id=tile % 2,
                    source="a" if tile % 2 == 0 else "b",
                    _unique_id=17 + tile // 2,
                )
                for member in range(2)
            ]
            self.reward_buffer.add_samples(batch)
            samples.extend(batch)
            self.events.append((self.epoch, "rollout", tile))
        self.reward_buffer._futures[self.first_tile][2].result(timeout=30)
        self.events.append((self.epoch, "rollout-complete"))
        return samples

    def _prepare_reward_optimization_overlap(self, samples, plan):
        assert [sample.source_id for sample in samples] == [0, 0, 1, 1, 0, 0]
        assert plan.tiles[0].group_ids[0] != plan.tiles[1].group_ids[0]
        return {"open": True}

    def _optimize_reward_overlap_tile(self, tile, samples, context):
        assert context["open"]
        cycle_events = [event[1:] for event in self.events if event[0] == self.epoch]
        assert ("rollout-complete",) in cycle_events
        if not any(event[0] == "update" for event in cycle_events):
            assert tile.tile_id == self.first_tile
            assert sum(not future.done() for _, _, future in self.reward_buffer._futures) == 2
            self.events.append((self.epoch, "overlap-with-pending-rewards"))
            for gate in self.reward.gates:
                gate.set()
        assert [sample.prompt for sample in samples] == [
            str(index) for index in tile.sample_indices
        ]
        with pytest.raises(RuntimeError, match="complete acquisition boundary"):
            self._save_exact_training_state("unused-active-cycle-checkpoint")
        if self.failure is not None:
            raise self.failure
        advantages = torch.stack([sample.extra_kwargs["advantage"] for sample in samples])
        assert torch.isfinite(advantages).all()
        torch.testing.assert_close(advantages[0], -advantages[1])
        assert advantages[1] > 0
        self.optimizer.zero_grad()
        (self.parameter * advantages[1]).backward()
        self.optimizer.step()
        self.step += 1
        self.events.append((self.epoch, "update", tile.tile_id))

    def _finalize_reward_optimization_overlap(self, context, samples):
        context["open"] = False
        self.events.append((self.epoch, "finish"))
        self.worker_threads.append(frozenset(self.reward_buffer._executors["score"]._threads))
        return {}

    def _abort_reward_optimization_overlap(self, context):
        if context is not None:
            context["open"] = False
        self.events.append((self.epoch, "abort-context"))

    def log_data(self, metrics, step):
        # Do not retain the media/sample objects carried by logging payloads.
        self.metrics.append(
            {key: value for key, value in metrics.items() if key != "train_samples"}
        )


@pytest.mark.parametrize("mode", ["ordered", "ready"])
def test_three_cycles_preserve_full_rollout_boundary_and_functional_overlap(mode):
    trainer = _StreamTrainer(mode)
    try:
        trainer.start()
    finally:
        for gate in getattr(trainer.reward, "gates", []):
            gate.set()
        trainer.cleanup(wait=True)

    assert trainer.progress == TrainingProgress(optimizer_step=9, rollout_iteration=3)
    assert trainer.parameter.item() < 1.0
    assert len(trainer.metrics) == 3
    for cycle in range(3):
        events = [event[1:] for event in trainer.events if event[0] == cycle]
        assert events[:4] == [("rollout", 0), ("rollout", 1), ("rollout", 2), ("rollout-complete",)]
        expected_order = [0, 1, 2] if mode == "ordered" else [1, 0, 2]
        assert [event[1] for event in events if event[0] == "update"] == expected_order
        assert events[-2:] == [("finish",), ("ema",)]
        assert ("overlap-with-pending-rewards",) in events
        assert trainer.metrics[cycle]["train/reward_mean"] == pytest.approx(2.5)
        assert trainer.metrics[cycle]["train/reward_overlap/tile_count"] == 3
        assert trainer.metrics[cycle]["train/reward_overlap/out_of_order_tiles"] == (
            mode == "ready"
        )
    assert len(trainer.worker_threads[0]) == 3
    assert all(threads == trainer.worker_threads[0] for threads in trainer.worker_threads)
    assert not trainer.reward_buffer.all_samples and not trainer.reward_buffer._futures
    assert all(not thread.is_alive() for thread in trainer.worker_threads[0])


@pytest.mark.parametrize("failure", [RuntimeError("objective failed"), KeyboardInterrupt()])
def test_stream_failure_unwinds_algorithm_context_and_keeps_cycle_incomplete(failure):
    trainer = _StreamTrainer("ready", failure=failure)
    try:
        with pytest.raises(type(failure)) as raised:
            trainer.start()
        assert raised.value is failure
    finally:
        for gate in getattr(trainer.reward, "gates", []):
            gate.set()
        trainer.cleanup(wait=True)
    assert (0, "abort-context") in trainer.events
    assert not any(event[1] == "ema" for event in trainer.events)
    assert trainer.progress == TrainingProgress()
    assert trainer._acquisition_cycle_incomplete
    assert not trainer._acquisition_cycle_active
    assert not trainer.reward_buffer._futures
    assert all(
        not thread.is_alive() for thread in trainer.reward_buffer._executors["score"]._threads
    )


@pytest.mark.parametrize("phase", ["plan", "preparation"])
def test_interrupt_before_stream_loop_still_cancels_rewards_when_context_cleanup_fails(phase):
    trainer = object.__new__(BaseTrainer)
    cancelled = []
    trainer.reward_buffer = SimpleNamespace(abort_streaming=lambda: cancelled.append(True))
    failure = KeyboardInterrupt()

    def interrupt(*_args):
        raise failure

    def failed_cleanup(_context):
        raise RuntimeError("secondary context cleanup failure")

    trainer._build_and_seal_reward_tile_plan = interrupt if phase == "plan" else lambda _: None
    trainer._prepare_reward_optimization_overlap = interrupt
    trainer._abort_reward_optimization_overlap = failed_cleanup
    with pytest.raises(KeyboardInterrupt) as raised:
        trainer._run_reward_optimization_overlap([], cycle_started=0, rollout_seconds=0)
    assert raised.value is failure
    assert cancelled == [True]
