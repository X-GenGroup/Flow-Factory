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

"""The peers of a tensor-parallel group split the scoring and train on one set of rewards.

They hold the same samples, so each is scored by exactly one peer and a summing
all-reduce hands every peer the whole set: half the reward work at tp=2, and
agreement by construction -- a scorer that does not return the same value twice
(a sampled VLM judgment, a remote call that falls back to a default) can no
longer give the peers different advantages, because no sample is scored twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput, RewardProcessor
from flow_factory.samples import BaseSample
from flow_factory.utils.dist import combine_tensor_parallel_shares, tensor_parallel_geometry

WORLD_SIZE = 4


class RankDependentReward(PointwiseRewardModel):
    """Scores each sample with this rank's index, as a non-repeatable scorer would."""

    required_fields = ("prompt",)
    scored: List[str] = []

    def __call__(self, prompt: List[str]) -> RewardModelOutput:
        RankDependentReward.scored.extend(prompt)
        rank = float(dist.get_rank())
        return RewardModelOutput(
            rewards=torch.tensor([rank + 0.25 * index for index in range(len(prompt))])
        )


class NoisyReward(PointwiseRewardModel):
    """Draws from the process RNG, which peers scoring different samples must not feel."""

    required_fields = ("prompt",)

    def __call__(self, prompt: List[str]) -> RewardModelOutput:
        return RewardModelOutput(rewards=torch.rand(len(prompt)) + 1.0)


def test_without_tensor_parallelism_values_are_returned_unchanged() -> None:
    values = {"score": torch.tensor([0.5, 1.5]), "count": torch.tensor([3], dtype=torch.int32)}

    combined = combine_tensor_parallel_shares(SimpleNamespace(), values)

    assert combined.keys() == values.keys()
    for name, value in values.items():
        assert combined[name] is value
    assert tensor_parallel_geometry(SimpleNamespace()) == (1, 0)


def _processor(accelerator: SimpleNamespace, models: Dict[str, type], **gates) -> RewardProcessor:
    configs = {
        name: RewardArguments(
            name=name,
            reward_model=f"test.{model.__name__}",
            device="cpu",
            batch_size=2,
            applicable_datasets=gates.get(name),
        )
        for name, model in models.items()
    }
    return RewardProcessor(
        accelerator=accelerator,
        reward_models={name: model(configs[name], accelerator) for name, model in models.items()},
        reward_configs=configs,
        group_on_same_rank=True,
        verbose=False,
    )


def _split_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        parallelism_config = ParallelismConfig(dp_shard_size=2, tp_size=2)
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            process_index=rank,
            num_processes=WORLD_SIZE,
            is_local_main_process=False,
            wait_for_everyone=lambda: None,
            parallelism_config=parallelism_config,
            torch_device_mesh=parallelism_config.build_device_mesh("cpu"),
        )
        result["geometry"] = list(tensor_parallel_geometry(accelerator))

        processor = _processor(accelerator, {"scorer": RankDependentReward})
        samples = [BaseSample(prompt=f"prompt-{index}") for index in range(4)]
        rewards = processor.compute_rewards(samples, split="pointwise")
        result["scored"] = list(RankDependentReward.scored)
        result["rewards"] = rewards["scorer"].tolist()
        result["stored"] = [float(s.extra_kwargs["rewards"]["scorer"]) for s in samples]

        # Three samples: peer 0 scores two and peer 1 one, so a reward drawing
        # from the RNG would advance their streams by different amounts.
        torch.manual_seed(7)
        noisy = _processor(accelerator, {"noisy": NoisyReward})
        noisy.compute_rewards([BaseSample(prompt=f"p{i}") for i in range(3)], split="pointwise")
        result["rng_after"] = torch.rand(3).tolist()

        # A reward applying only to one source: NaN where it does not, on every
        # peer, and applicability marked on samples a peer did not score.
        gated = _processor(accelerator, {"gated": RankDependentReward}, gated=["kept"])
        gated_samples = [BaseSample(prompt=f"g{i}") for i in range(4)]
        for index, sample in enumerate(gated_samples):
            sample.source = "kept" if index % 3 == 0 else "dropped"
        gated_rewards = gated.compute_rewards(gated_samples, split="pointwise")["gated"]
        result["gated_nan"] = torch.isnan(gated_rewards).tolist()
        result["gated_marked"] = ["gated" in s.applicable_rewards for s in gated_samples]

        _, tp_rank = tensor_parallel_geometry(accelerator)
        mixed = combine_tensor_parallel_shares(
            accelerator,
            {
                "half": torch.tensor([rank + 0.5, 0.0] if tp_rank == 0 else [0.0, rank + 0.5]).to(
                    torch.bfloat16
                ),
                "count": torch.tensor([rank if tp_rank == 0 else 0], dtype=torch.int32),
            },
        )
        result["mixed"] = {name: value.tolist() for name, value in mixed.items()}
        result["mixed_dtypes"] = {name: str(value.dtype) for name, value in mixed.items()}
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_peers_split_the_scoring_and_agree_and_replicas_keep_their_own(tmp_path: Path) -> None:
    """Ranks 0/1 and 2/3 are the two tensor-parallel groups of a 2x2 plan."""
    results = run_gloo_ranks(_split_worker, tmp_path, WORLD_SIZE, {})
    for result in results:
        assert "error" not in result, result["error"]
    by_rank = {result["rank"]: result for result in results}

    for first, second in ((0, 1), (2, 3)):
        assert by_rank[first]["geometry"] == [2, 0]
        assert by_rank[second]["geometry"] == [2, 1]
        # Each sample scored by exactly one peer of the group.
        assert by_rank[first]["scored"] == ["prompt-0", "prompt-2"]
        assert by_rank[second]["scored"] == ["prompt-1", "prompt-3"]
        # Every peer holds the scoring peer's value -- a non-repeatable scorer
        # agrees because each value has one source.
        expected = [float(first), float(second), first + 0.25, second + 0.25]
        for rank in (first, second):
            assert by_rank[rank]["rewards"] == expected
            assert by_rank[rank]["stored"] == expected
            assert by_rank[rank]["mixed"] == {"count": [first], "half": [first + 0.5, second + 0.5]}
            assert by_rank[rank]["mixed_dtypes"] == {
                "count": "torch.int32",
                "half": "torch.bfloat16",
            }
            assert by_rank[rank]["gated_nan"] == [False, True, True, False]
            assert by_rank[rank]["gated_marked"] == [True, False, False, True]
        # The peers' RNG streams are where they were before scoring.
        assert by_rank[first]["rng_after"] == by_rank[second]["rng_after"]
    torch.manual_seed(7)
    assert by_rank[0]["rng_after"] == torch.rand(3).tolist()
