# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Wan R1 three-role CP witness with acquisition-wide asynchronous feedback."""

import threading
import traceback
from collections import Counter
from contextlib import contextmanager

import torch
from tdm_r1_observations import observe_objectives
from test_context_parallel_training import _compare
from wan_tdm_fixture import build_fixture, config_dict, wrap_classes

from flow_factory.hparams import Arguments
from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput

checkpoint_run_name = "wan-tdm-r1"
execution_capability = "wan-tdm-r1-cp-v1"


class GatedReward(PointwiseRewardModel):
    """Each replica scores real decoded output once and keeps tile two pending."""

    use_tensor_inputs = True
    required_fields = ("sampling_sample_id", "sampling_group_member_id", "video")
    controller = None

    def __call__(self, sampling_sample_id, sampling_group_member_id, video):
        gates = self.controller
        assert gates["planned"].wait(90)
        index = gates["indices"][int(sampling_sample_id[0])]
        assert gates["ready"][index].wait(90)
        value = video[0]
        assert isinstance(value, torch.Tensor) and value.numel() and torch.isfinite(value).all()
        # Varying member rewards isolate feedback ownership from model quality.
        return RewardModelOutput(
            rewards=torch.tensor([-1.0 if int(sampling_group_member_id[0]) == 0 else 1.0])
        )


def make_config(root, result_dir, options):
    config = config_dict(root, result_dir, options)
    config["data"]["sampler_type"] = "global_batch"
    config["data"]["cache_dir"] = str(root / "tdm-r1-cache")
    config["train"].update(
        trainer_type="tdm-r1",
        group_size=2,
        unique_sample_num_per_epoch=2,
        global_std=False,
        advantage_aggregation="sum",
        reward_optimization_overlap=True,
        reward_optimization_overlap_mode="ordered",
        reward_optimization_overlap_poll_interval=0.002,
        surrogate_preference_beta=2.3,
        surrogate_reference_beta=0.17,
        surrogate_reference_threshold=0.05,
        surrogate_clip_range=0.1,
        surrogate_slow_decay_min=0.2,
        surrogate_slow_decay_max=0.5,
        tdm_weight=0.3,
        cfg_reward_scale=0.7,
        use_time_weighting=True,
    )
    config["optimizers"].append(
        {
            "name": "surrogate",
            "learning_rate": 7e-4,
            "eps": 1e-4,
            "max_grad_norm": 1.0,
            "update_frequency": 1,
        }
    )
    config["rewards"] = [
        {
            "name": "preference",
            "reward_model": "wan_tdm_r1_fixture.GatedReward",
            "device": "cpu",
            "batch_size": 1,
            "async_reward": True,
            "num_workers": 2,
        }
    ]
    config["log"]["run_name"] = checkpoint_run_name
    return Arguments.from_dict(config)


@contextmanager
def observe_training(trainer, records):
    originals = {
        name: getattr(trainer, name)
        for name in (
            "_run_overlap_training_step",
            "_prepare_reward_optimization_overlap",
            "_optimize_reward_overlap_tile",
            "_finalize_reward_optimization_overlap",
            "cleanup",
            "_synchronize_reward_overlap_error",
        )
    }
    records["overlap_cycles"] = []
    gates, current = None, None

    def run():
        nonlocal gates
        gates = {
            "planned": threading.Event(),
            "ready": [threading.Event(), threading.Event()],
            "indices": {},
        }
        GatedReward.controller = gates
        return originals["_run_overlap_training_step"]()

    def prepare(samples, plan):
        nonlocal current
        assert len(plan.tiles) == 2 and len(samples) == 2
        gates["indices"] = {
            int(samples[index].sampling_sample_id): tile.tile_id
            for tile in plan.tiles
            for index in tile.sample_indices
        }
        gates["planned"].set()
        gates["ready"][0].set()
        current = {
            "cycle": trainer.cycle_index,
            "tiles": [],
            "pending_before_fake": sum(
                not future.done() for _, _, future in trainer.reward_buffer._futures
            ),
        }
        records["overlap_cycles"].append(current)
        before = len(records["updates"])
        context = originals["_prepare_reward_optimization_overlap"](samples, plan)
        assert [row["role"] for row in records["updates"][before:]] == ["fake", "fake"]
        assert trainer.role_optimization._active_role_name == "surrogate"
        return context

    def optimize(tile, samples, context):
        before = len(records["updates"])
        pending = sum(not future.done() for _, _, future in trainer.reward_buffer._futures)
        previous = len(records["surrogate_objectives"])
        result = originals["_optimize_reward_overlap_tile"](tile, samples, context)
        assert len(records["surrogate_objectives"]) - previous == 2
        assert len(records["updates"]) - before == tile.tile_id
        if tile.tile_id:
            assert records["updates"][-1]["role"] == "surrogate"
        current["tiles"].append(
            {
                "tile": tile.tile_id,
                "pending": pending,
                "updates": len(records["updates"]) - before,
                "advantages": [float(sample.extra_kwargs["advantage"]) for sample in samples],
                "rewards": [
                    float(value)
                    for sample in samples
                    for value in sample.extra_kwargs["rewards"].values()
                ],
            }
        )
        gates["ready"][1].set()
        return result

    def finalize(context, samples):
        before = len(records["updates"])
        assert all(future.done() for _, _, future in trainer.reward_buffer._futures)
        result = originals["_finalize_reward_optimization_overlap"](context, samples)
        assert (
            len(records["updates"]) == before + 1 and records["updates"][-1]["role"] == "generator"
        )
        assert trainer.role_optimization._active_role_name is None
        current["drained"] = True
        return result

    def cleanup(*, wait=False):
        if gates is not None:
            gates["planned"].set()
            for event in gates["ready"]:
                event.set()
        return originals["cleanup"](wait=wait)

    def synchronize(phase, error):
        if error is not None:
            traceback.print_exception(type(error), error, error.__traceback__)
        return originals["_synchronize_reward_overlap_error"](phase, error)

    trainer._synchronize_reward_overlap_error = synchronize
    trainer._run_overlap_training_step, trainer._prepare_reward_optimization_overlap = run, prepare
    (
        trainer._optimize_reward_overlap_tile,
        trainer._finalize_reward_optimization_overlap,
        trainer.cleanup,
    ) = (optimize, finalize, cleanup)
    with observe_objectives(trainer, records):
        try:
            yield
        finally:
            for name, method in originals.items():
                setattr(trainer, name, method)


def assert_worker(trainer, records, loaded, final, options):
    cycles = 1 if options.get("resume") else 2
    assert Counter(row["role"] for row in records["updates"]) == {
        "fake": 2 * cycles,
        "surrogate": cycles,
        "generator": cycles,
    }
    assert Counter(row["kind"] for row in records["objectives"]) == {
        "fake": 8 * cycles,
        "generator": 4 * cycles,
    }
    assert len(records["surrogate_objectives"]) == len(records["generator_rewards"]) == 4 * cycles
    assert (
        len(records["snapshot_queries"]) == 4 * cycles
        and len(records["snapshot_updates"]) == cycles
    )
    assert all(len(row["loss"]) == 4 for row in records["updates"])
    for cycle in records["overlap_cycles"]:
        assert [tile["tile"] for tile in cycle["tiles"]] == [0, 1]
        assert [tile["updates"] for tile in cycle["tiles"]] == [0, 1]
        assert cycle["drained"]
    for reward in (0, 1):
        assert any(
            torch.count_nonzero(value)
            for row in records["generator_rewards"]
            for value in row["rewards"][reward]["direction"].values()
        )
    assert len(records["samples"]) == 2 * cycles


def assert_continuation(continuous, resumed, joined, options):
    for key in (
        "samples",
        "replays",
        "queries",
        "score_queries",
        "objectives",
        "evaluations",
        "surrogate_objectives",
        "generator_rewards",
        "snapshot_updates",
        "snapshot_queries",
        "reward_queries",
    ):
        _compare(joined[key], continuous["records"][key], exact=True)
    for actual, expected in zip(
        joined["overlap_cycles"], continuous["records"]["overlap_cycles"], strict=True
    ):
        # Pending thread counts prove overlap, while timing is not resume identity.
        assert actual["cycle"] == expected["cycle"] and actual["drained"] == expected["drained"]
        for a, b in zip(actual["tiles"], expected["tiles"], strict=True):
            _compare(
                {k: v for k, v in a.items() if k != "pending"},
                {k: v for k, v in b.items() if k != "pending"},
                exact=True,
            )
