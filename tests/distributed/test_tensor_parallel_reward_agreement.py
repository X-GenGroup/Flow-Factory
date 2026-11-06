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

"""TP x CP peers share each reward request exactly once.

Real Gloo groups cover pointwise, async, local-group and cross-DP-group scoring,
including non-repeatable values, applicability masks and isolated process RNG.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import (
    GroupwiseRewardModel,
    PointwiseRewardModel,
    RewardBuffer,
    RewardModelOutput,
    RewardProcessor,
)
from flow_factory.samples import BaseSample
from flow_factory.utils.dist import (
    combine_model_parallel_shares,
    combine_tensor_parallel_shares,
    model_parallel_geometry,
    tensor_parallel_geometry,
)


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


class GroupRankReward(GroupwiseRewardModel):
    """Keep complete groups on one peer and expose duplicate requests."""

    required_fields = ("prompt",)
    scored = []

    def __call__(self, prompt):
        type(self).scored.append(list(prompt))
        return RewardModelOutput(rewards=torch.full((len(prompt),), float(dist.get_rank())))


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


def _processor(
    accelerator: SimpleNamespace, models: Dict[str, type], *, async_reward=False, **gates
) -> RewardProcessor:
    configs = {
        name: RewardArguments(
            name=name,
            reward_model=f"test.{model.__name__}",
            device="cpu",
            batch_size=2,
            async_reward=async_reward,
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
    degree = options["tp_size"] * options["cp_size"]
    world_size = 2 * degree
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    result: Dict[str, Any] = {"rank": rank}
    try:
        parallelism_config = ParallelismConfig(
            dp_shard_size=2, tp_size=options["tp_size"], cp_size=options["cp_size"]
        )
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            process_index=rank,
            num_processes=world_size,
            is_local_main_process=False,
            wait_for_everyone=lambda: None,
            parallelism_config=parallelism_config,
            torch_device_mesh=parallelism_config.build_device_mesh("cpu"),
        )
        result["geometry"] = list(model_parallel_geometry(accelerator))

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

        _, peer_rank = model_parallel_geometry(accelerator)
        shares = torch.zeros(degree, dtype=torch.bfloat16)
        shares[peer_rank] = rank + 0.5
        mixed = combine_model_parallel_shares(
            accelerator,
            {
                "half": shares,
                "count": torch.tensor([rank if peer_rank == 0 else 0], dtype=torch.int32),
            },
        )
        result["mixed"] = {name: value.tolist() for name, value in mixed.items()}
        result["mixed_dtypes"] = {name: str(value.dtype) for name, value in mixed.items()}

        RankDependentReward.scored = []
        buffer = RewardBuffer(
            _processor(accelerator, {"async": RankDependentReward}, async_reward=True), group_size=2
        )
        buffer.add_samples([BaseSample(prompt=f"a{i}") for i in range(4)])
        try:
            result["async"] = buffer.finalize(split="all")["async"].tolist()
            result["async_scored"] = list(RankDependentReward.scored)
        finally:
            buffer.shutdown(wait=True)
        grouped = [BaseSample(prompt=f"g{i}", sampling_group_id=i // 2) for i in range(6)]
        result["grouped"] = (
            _processor(accelerator, {"group": GroupRankReward})
            .compute_rewards(grouped)["group"]
            .tolist()
        )
        result["groups_scored"] = list(GroupRankReward.scored)
        # A group's members may span distinct logical replicas. The coordinator
        # gathers only DP, and model-peer ownership must still score it once.
        from flow_factory.utils.group_coordinator import GroupCoordinator

        GroupRankReward.scored = []
        distributed = _processor(accelerator, {"group": GroupRankReward})
        distributed.group_on_same_rank = False
        distributed.group_coordinator = GroupCoordinator(accelerator, sampler_type="global_tile")
        dp_rank = rank // degree
        partial = [BaseSample(prompt=f"x{i}-{dp_rank}", sampling_group_id=i) for i in range(4)]
        result["distributed_groups"] = distributed.compute_rewards(partial)["group"].tolist()
        result["distributed_scored"] = list(GroupRankReward.scored)
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("tp_size,cp_size", [(2, 1), (1, 2), (2, 2)])
def test_peers_split_the_scoring_and_agree_and_replicas_keep_their_own(
    tmp_path, tp_size, cp_size
) -> None:
    """Each TP x CP peer set shares work; distinct logical replicas stay separate."""
    degree = tp_size * cp_size
    results = run_gloo_ranks(
        _split_worker, tmp_path, 2 * degree, {"tp_size": tp_size, "cp_size": cp_size}
    )
    for result in results:
        assert "error" not in result, result.get("error")
    by_rank = {result["rank"]: result for result in results}
    for first in (0, degree):
        expected = [float(first + i % degree) + 0.25 * (i // degree) for i in range(4)]
        group_values = [float(first + (i // 2) % degree) for i in range(6)]
        for peer in range(degree):
            row = by_rank[first + peer]
            assert row["geometry"] == [degree, peer]
            assert row["scored"] == [f"prompt-{i}" for i in range(peer, 4, degree)]
            assert row["async_scored"] == [f"a{i}" for i in range(peer, 4, degree)]
            assert row["rewards"] == row["stored"] == row["async"] == expected
            assert row["grouped"] == group_values
            assert row["distributed_groups"] == [
                float((i % 2) * degree + (i // 2) % degree) for i in range(4)
            ]
            expected_groups = [i for i in range(first // degree, 4, 2) if (i // 2) % degree == peer]
            assert row["distributed_scored"] == [[f"x{i}-0", f"x{i}-1"] for i in expected_groups]
            assert row["groups_scored"] == [
                [f"g{2*i}", f"g{2*i+1}"] for i in range(peer, 3, degree)
            ]
            assert row["mixed"] == {
                "count": [first],
                "half": [first + i + 0.5 for i in range(degree)],
            }
            assert row["mixed_dtypes"] == {"count": "torch.int32", "half": "torch.bfloat16"}
            assert row["gated_nan"] == [False, True, True, False]
            assert row["gated_marked"] == [True, False, False, True]
            assert row["rng_after"] == by_rank[first]["rng_after"]
    torch.manual_seed(7)
    assert by_rank[0]["rng_after"] == torch.rand(3).tolist()
