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

"""Real Gloo tile loops; optimizer-window observations do not qualify CP training."""

import json
import math
import os
import threading
import time
import traceback
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.advantage import AdvantageProcessor
from flow_factory.data_utils.sampling_plan import build_sampling_plan
from flow_factory.hparams import ParallelismArguments, RewardArguments
from flow_factory.rewards import (
    PointwiseRewardModel,
    RewardBuffer,
    RewardModelOutput,
    RewardProcessor,
)
from flow_factory.samples import BaseSample
from flow_factory.trainers import abc as trainer_module
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.execution import TrainingProgress
from flow_factory.utils.dist import data_parallel_geometry
from flow_factory.utils.group_coordinator import GroupCoordinator


class _ControlledReward(PointwiseRewardModel):
    required_fields = ("prompt",)

    def __call__(self, prompt):
        assert len(prompt) == 1
        index, sample_id, source, member = json.loads(prompt[0])
        assert self.plan_ready.wait(20), "tile plan was not published"
        tile = self.row_to_tile[index]
        self.started[tile].set()
        assert self.gates[tile].wait(20), "reward request was not released"
        if self.fail and self.accelerator.process_index == 0 and tile == 1:
            raise ValueError("controlled next-tile reward failure")
        value = float(10 * source + member**2 + 1)
        self.scored.append(sample_id)
        return RewardModelOutput(rewards=torch.tensor([value], dtype=torch.float64))


class _LoopTrainer(BaseTrainer):
    """Retain production acquisition/feedback/lifecycle; observe optimizer terms only."""

    def __init__(self, accelerator, options):
        self.accelerator = accelerator
        self.options = options
        self.progress = TrainingProgress()
        self.cycles, self.orders, self.windows, self.readiness = [], [], [], []
        self.ema, self.aborts, self.worker_pools = [], [], []
        self.cleanup_snapshot = None
        self.dp_size, self.dp_rank = data_parallel_geometry(accelerator)
        group_size = 4 if options["layout"] in {"global_tile", "subgroup_tile"} else 2
        self.training_args = SimpleNamespace(
            seed=17,
            reward_optimization_overlap=True,
            reward_optimization_overlap_mode=options["mode"],
            reward_optimization_overlap_poll_interval=0.001,
            group_size=group_size,
            per_device_batch_size=1,
            gradient_accumulation_steps=4,
            get_num_train_timesteps=lambda config: 2,
            advantage_aggregation="sum",
        )
        self.config = SimpleNamespace(
            training_args=self.training_args,
            data_args=SimpleNamespace(
                sampler_type=options["layout"],
                sampler_subgroup_size=2 if options["layout"] == "subgroup_tile" else None,
            ),
            parallelism_args=ParallelismArguments(
                dp_shard_size=self.dp_size, tp_size=options["tp"], cp_size=2
            ),
        )
        self.adapter = SimpleNamespace(
            requires_preserved_replay_batch_composition=False,
            set_trajectory_seed=lambda seed: None,
            ema_step=lambda step: self.ema.append(step),
        )
        self.log_args = SimpleNamespace(save_freq=0, save_dir=None, verbose=False)
        self.eval_args = SimpleNamespace(eval_freq=0)
        self._reward_overlap_acquisition_manifest = None
        self.group_coordinator = GroupCoordinator(
            accelerator,
            sampler_type=options["layout"],
            subgroup_size=self.config.data_args.sampler_subgroup_size,
        )
        reward_args = RewardArguments(
            name="score",
            device="cpu",
            batch_size=1,
            async_reward=True,
            num_workers=1 if options.get("fail") else 2,
        )
        self.reward = _ControlledReward(reward_args, accelerator)
        self.reward.scored = []
        self.reward.fail = options.get("fail", False)
        self.reward_buffer = RewardBuffer(
            RewardProcessor(
                accelerator,
                {"score": self.reward},
                {"score": reward_args},
                group_on_same_rank=self.group_coordinator.groups_are_rank_local,
                verbose=False,
            ),
            group_size=group_size,
        )
        self.advantage_processor = AdvantageProcessor(
            accelerator,
            {"score": {"a": 1.0, "b": -2.0}},
            group_size,
            global_std=False,
            sampler_type=options["layout"],
            verbose=False,
            source_id_to_name=["a", "b"],
            group_coordinator=self.group_coordinator,
        )

    def should_continue_training(self):
        return self.epoch < 2

    def sample(self):
        self.reward_buffer.clear()
        local_rows = 6 if self.options.get("fail") else 4
        groups = local_rows * self.dp_size // self.training_args.group_size
        plan = build_sampling_plan(
            layout_name=self.options["layout"],
            dataset_size=groups,
            per_device_batch_size=1,
            group_size=self.training_args.group_size,
            unique_group_count=groups,
            num_replicas=self.dp_size,
            seed=17,
            epoch=self.epoch,
            subgroup_size=self.config.data_args.sampler_subgroup_size,
        )
        self.reward.plan_ready = threading.Event()
        self.reward.gates = [threading.Event() for _ in range(local_rows // 2)]
        self.reward.started = [threading.Event() for _ in self.reward.gates]
        self.first_tile = 1 if self.options["mode"] == "ready" else 0
        self.delay_rank = (2 * self.first_tile) % (2 * self.options["tp"])
        self.first_poll = True
        samples = []
        for batch in plan.batches_for_rank(self.dp_rank):
            (assignment,) = batch
            source = assignment.group_id % 2
            sample = BaseSample(
                prompt=json.dumps(
                    [len(samples), assignment.sample_id, source, assignment.group_member_id]
                ),
                source_id=source,
                source="a" if source == 0 else "b",
                sampling_group_id=2**53 + assignment.group_id // 2,
                sampling_group_member_id=assignment.group_member_id,
                sampling_sample_id=assignment.sample_id,
            )
            samples.append(sample)
            self.reward_buffer.add_samples([sample])
        assert len(samples) == local_rows
        self.cycles.append(
            {"ids": [s.sampling_sample_id for s in samples], "rollout_complete": True}
        )
        return samples

    def _prepare_reward_optimization_overlap(self, samples, plan):
        assert plan.sample_count == len(samples) and plan.samples_per_tile == 2
        assert plan.batches_per_tile == 2 and plan.geometry.optimizer_terms_per_batch == 2
        self.reward.row_to_tile = {
            i: tile.tile_id for tile in plan.tiles for i in tile.sample_indices
        }
        self.reward.plan_ready.set()
        if self.accelerator.process_index != self.delay_rank:
            self.reward.gates[self.first_tile].set()
            for _, indices, future in self.reward_buffer._futures:
                if all(self.reward.row_to_tile[i] == self.first_tile for i in indices):
                    future.result(timeout=20)
        self.cycles[-1]["tiles"] = [list(tile.sample_indices) for tile in plan.tiles]
        return {"open": True}

    def _optimize_reward_overlap_tile(self, tile, samples, context):
        assert context["open"] and self.cycles[-1]["rollout_complete"]
        assert (
            self.readiness[-1]["counts"][
                (
                    0
                    if self.options["mode"] == "ordered"
                    else self.readiness[-1]["candidates"].index(tile.tile_id)
                )
            ]
            == self.accelerator.num_processes
        )
        pending = sum(not f.done() for _, _, f in self.reward_buffer._futures)
        self.orders.append([self.epoch, tile.tile_id, pending])
        with pytest.raises(RuntimeError, match="complete acquisition boundary"):
            self._save_exact_training_state("unused-active-loop-checkpoint")
        terms = []
        for sample in samples:
            member = sample.sampling_group_member_id
            reward = float(sample.extra_kwargs["rewards"]["score"])
            assert reward == 10 * sample.source_id + member**2 + 1
            weight = 1.0 if sample.source_id == 0 else -2.0
            values = [
                weight * (10 * sample.source_id + m**2 + 1)
                for m in range(self.training_args.group_size)
            ]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            expected = torch.tensor((values[member] - mean) / std, dtype=torch.float64)
            actual = sample.extra_kwargs["advantage"].cpu()
            torch.testing.assert_close(actual, expected)
            for transition in range(2):
                terms.append(
                    {
                        "id": sample.sampling_sample_id,
                        "source": sample.source_id,
                        "group": sample.unique_id,
                        "member": member,
                        "transition": transition,
                        "reward_weight": weight,
                        "optimizer_weight": 0.25,
                        "advantage": float(actual),
                    }
                )
        assert len(terms) == self.training_args.gradient_accumulation_steps
        self.windows.append(
            {"cycle": self.epoch, "step": self.step, "tile": tile.tile_id, "terms": terms}
        )
        self.step += 1
        if len([row for row in self.orders if row[0] == self.epoch]) == 1:
            assert tile.tile_id == self.first_tile
            if self.options.get("fail"):
                assert self.reward.started[1].wait(20)
                if self.accelerator.process_index == 0:
                    self.reward.gates[1].set()
                    future = next(
                        f for _, indices, f in self.reward_buffer._futures if indices == [2]
                    )
                    assert isinstance(future.exception(timeout=20), ValueError)
                    # This worker may immediately take the next queued request
                    # after failing. Keep that request running while other ranks
                    # still have a genuinely queued request to cancel.
                    assert self.reward.started[2].wait(20)
            else:
                for gate in self.reward.gates:
                    gate.set()

    def _finalize_reward_optimization_overlap(self, context, samples):
        context["open"] = False
        assert not self.reward_buffer._streaming_sealed
        assert self.reward_buffer._streaming_consumed_indices == set(range(len(samples)))
        assert all(f.done() for _, _, f in self.reward_buffer._futures)
        self.worker_pools.append(frozenset(self.reward_buffer._executors["score"]._threads))
        return {}

    def _abort_reward_optimization_overlap(self, context):
        if context is not None:
            context["open"] = False
        self.aborts.append(self.epoch)

    def cleanup(self, *, wait=False):
        # Production abort already cancels queued requests. Release only running
        # test-controlled requests before the real shutdown owner joins them.
        if self.cleanup_snapshot is None:
            self.cleanup_snapshot = {
                "cancelled": sum(f.cancelled() for _, _, f in self.reward_buffer._futures),
                "incomplete": self._acquisition_cycle_incomplete,
                "active": self._acquisition_cycle_active,
            }
        for gate in getattr(self.reward, "gates", []):
            gate.set()
        started = time.monotonic()
        super().cleanup(wait=wait)
        assert time.monotonic() - started < 15

    def log_data(self, metrics, step):
        self.cycles[-1]["tile_count"] = int(metrics["train/reward_overlap/tile_count"])
        self.cycles[-1]["out_of_order"] = int(metrics["train/reward_overlap/out_of_order_tiles"])


def _worker(rank, store_path, result_dir, options):
    world = options["dp"] * options["tp"] * 2
    # Config-time topology reads the launch environment; the real process group
    # and runtime mesh below independently describe the same physical world.
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_path, world),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    result, trainer = {"rank": rank}, None
    original_gather = trainer_module.gather_object
    try:
        config = ParallelismConfig(dp_shard_size=options["dp"], tp_size=options["tp"], cp_size=2)
        mesh = config.build_device_mesh("cpu")

        def reduce(value, reduction):
            output = value.clone()
            dist.all_reduce(
                output, op=dist.ReduceOp.SUM if reduction == "sum" else dist.ReduceOp.MAX
            )
            if value.dtype == torch.int32 and value.numel() > 1:
                pending = set(range(len(trainer.reward.gates))) - {
                    row[1] for row in trainer.orders if row[0] == trainer.epoch
                }
                candidates = trainer._reward_overlap_readiness_candidates(
                    pending=pending, mode=options["mode"]
                )
                trainer.readiness.append(
                    {
                        "cycle": trainer.epoch,
                        "local": value.tolist(),
                        "counts": output.tolist(),
                        "candidates": list(candidates),
                    }
                )
                if trainer.first_poll:
                    trainer.first_poll = False
                    assert (
                        output[
                            (
                                0
                                if options["mode"] == "ordered"
                                else candidates.index(trainer.first_tile)
                            )
                        ]
                        < world
                    )
                    if rank == trainer.delay_rank:
                        trainer.reward.gates[trainer.first_tile].set()
            return output

        def gather(value):
            tensors = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(tensors, value)
            return torch.cat(tensors)

        def gather_objects(values):
            peers = [None] * world
            dist.all_gather_object(peers, values)
            return [item for peer in peers for item in peer]

        trainer_module.gather_object = gather_objects
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            process_index=rank,
            num_processes=world,
            is_local_main_process=False,
            parallelism_config=config,
            torch_device_mesh=mesh,
            reduce=reduce,
            gather=gather,
            wait_for_everyone=dist.barrier,
        )
        trainer = _LoopTrainer(accelerator, options)
        coordinator = trainer.group_coordinator
        result["runtime"] = {
            "world_size": dist.get_world_size(),
            "device": "cpu",
            "process_group_backend": dist.get_backend(),
            "distributed_type": "PROCESS_GROUP",
            "dp": options["dp"],
            "tp": options["tp"],
            "cp": 2,
            "mesh": mesh.mesh.tolist(),
            "dp_rank": coordinator.dp_rank,
            "group_ranks": (
                [rank]
                if coordinator.group_world_size == 1
                else dist.get_process_group_ranks(coordinator.collective_group)
            ),
            "group_world_size": coordinator.group_world_size,
        }
        try:
            trainer.start()
            assert not options.get("fail")
            assert trainer.progress == TrainingProgress(optimizer_step=4, rollout_iteration=2)
            assert trainer.ema == [0, 1] and not trainer.aborts
            assert trainer.worker_pools[0] == trainer.worker_pools[1]
        except RuntimeError as error:
            if not options.get("fail") or "controlled next-tile reward failure" not in str(error):
                raise
            result["reward_error"] = str(error)
            assert "reward polling" in str(error)
            assert trainer.progress == TrainingProgress(optimizer_step=1)
            assert not trainer.ema and trainer.aborts == [0]
            assert trainer.cleanup_snapshot == {
                "cancelled": 0 if rank == 0 else 1,
                "incomplete": True,
                "active": False,
            }
            with pytest.raises(RuntimeError, match="complete acquisition boundary"):
                trainer._save_exact_training_state("unused-incomplete-loop-checkpoint")
        trainer.cleanup(wait=True)
        assert not trainer.reward_buffer._futures and not trainer.reward_buffer.all_samples
        assert all(
            not thread.is_alive() for thread in trainer.reward_buffer._executors["score"]._threads
        )
        result.update(
            cycles=trainer.cycles,
            orders=trainer.orders,
            windows=trainer.windows,
            readiness=trainer.readiness,
            scored=trainer.reward.scored,
            cleanup=trainer.cleanup_snapshot,
            workers_stopped=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if trainer is not None:
            trainer.cleanup(wait=True)
        trainer_module.gather_object = original_gather
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize(
    "dp,tp,layout,mode",
    [
        (2, 1, "rank_local", "ordered"),
        (2, 1, "rank_local", "ready"),
        (2, 1, "global_batch", "ordered"),
        (2, 1, "global_tile", "ready"),
        (4, 1, "subgroup_tile", "ordered"),
        (4, 1, "subgroup_tile", "ready"),
        (2, 2, "rank_local", "ordered"),
        (2, 2, "rank_local", "ready"),
    ],
)
def test_complete_loop_preserves_group_and_optimizer_windows(tmp_path, dp, tp, layout, mode):
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        dp * tp * 2,
        {"dp": dp, "tp": tp, "layout": layout, "mode": mode},
        timeout=120,
    )
    assert all("error" not in row for row in rows), rows
    order = [0, 1] if mode == "ordered" else [1, 0]
    for row in rows:
        assert row["runtime"]["world_size"] == dp * tp * 2
        assert row["runtime"]["process_group_backend"] == "gloo" and row["workers_stopped"]
        assert [item[:2] for item in row["orders"]] == [
            [cycle, tile] for cycle in range(2) for tile in order
        ]
        for cycle, acquisition in enumerate(row["cycles"]):
            assert acquisition["tile_count"] == 2 and acquisition["out_of_order"] == (
                mode == "ready"
            )
            terms = [
                term
                for window in row["windows"]
                if window["cycle"] == cycle
                for term in window["terms"]
            ]
            assert Counter(
                (term["id"], term["transition"], term["optimizer_weight"]) for term in terms
            ) == Counter(
                (sample, transition, 0.25)
                for sample in acquisition["ids"]
                for transition in range(2)
            )
            assert [window["step"] for window in row["windows"] if window["cycle"] == cycle] == [
                2 * cycle,
                2 * cycle + 1,
            ]
        assert not row["cleanup"]["incomplete"] and not row["cleanup"]["active"]
    for cycle in range(2):
        assert any(row["orders"][2 * cycle][2] > 0 for row in rows)
    for replica in range(dp):
        peers = rows[replica * tp * 2 : (replica + 1) * tp * 2]
        assert all(peer["windows"] == peers[0]["windows"] for peer in peers)
        expected = [sample for cycle in peers[0]["cycles"] for sample in cycle["ids"]]
        assert Counter(sample for peer in peers for sample in peer["scored"]) == Counter(expected)
    if layout == "subgroup_tile":
        for row in rows:
            rank = row["rank"]
            first = (rank // 2 // 2) * 2
            assert row["runtime"]["group_ranks"] == [
                d * 2 + rank % 2 for d in range(first, first + 2)
            ]


@requires_gloo
def test_failed_next_tile_cancels_queued_work_and_closes_workers(tmp_path):
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        4,
        {"dp": 2, "tp": 1, "layout": "rank_local", "mode": "ordered", "fail": True},
        timeout=120,
    )
    assert all("error" not in row for row in rows), rows
    assert len({row["reward_error"] for row in rows}) == 1
    assert all(len(row["windows"]) == 1 and row["orders"][0][1:] == [0, 2] for row in rows)
    assert [row["cleanup"]["cancelled"] for row in rows] == [0, 1, 1, 1]
    assert all(row["workers_stopped"] for row in rows)
