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

"""The peers of a tensor-parallel group train on one set of rewards.

Each peer scores its copy of the samples itself. A scorer that does not return
the same value twice -- a sampled VLM judgment, a remote call that falls back to
a default -- would give the peers different advantages, and the parameters the
plan replicates would drift apart with nothing reporting it.
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
from flow_factory.utils.dist import agree_across_tensor_parallel_peers

WORLD_SIZE = 4


class RankDependentReward(PointwiseRewardModel):
    """Scores each sample with this rank's index, as a non-repeatable scorer would."""

    required_fields = ("prompt",)

    def __call__(self, prompt: List[str]) -> RewardModelOutput:
        rank = float(dist.get_rank())
        return RewardModelOutput(
            rewards=torch.tensor([rank + 0.25 * index for index in range(len(prompt))])
        )


def test_without_tensor_parallelism_values_are_returned_unchanged() -> None:
    values = {"score": torch.tensor([0.5, 1.5]), "count": torch.tensor([3], dtype=torch.int32)}

    agreed = agree_across_tensor_parallel_peers(SimpleNamespace(), values)

    assert agreed.keys() == values.keys()
    for name, value in values.items():
        assert agreed[name] is value


def _agreement_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
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
        config = RewardArguments(
            name="scorer", reward_model="test.RankDependentReward", device="cpu", batch_size=2
        )
        processor = RewardProcessor(
            accelerator=accelerator,
            reward_models={"scorer": RankDependentReward(config, accelerator)},
            reward_configs={"scorer": config},
            group_on_same_rank=True,
            verbose=False,
        )
        samples = [BaseSample(prompt="first"), BaseSample(prompt="second")]
        rewards = processor.compute_rewards(samples, split="pointwise")
        result["rewards"] = rewards["scorer"].tolist()
        result["stored"] = [float(s.extra_kwargs["rewards"]["scorer"]) for s in samples]

        mixed = agree_across_tensor_parallel_peers(
            accelerator,
            {
                "half": torch.full((2,), rank + 0.5, dtype=torch.bfloat16),
                "count": torch.tensor([rank], dtype=torch.int32),
            },
        )
        result["mixed"] = {name: value.tolist() for name, value in mixed.items()}
        result["mixed_dtypes"] = {name: str(value.dtype) for name, value in mixed.items()}
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_peers_adopt_the_first_peers_rewards_and_replicas_keep_their_own(tmp_path: Path) -> None:
    """Ranks 0/1 and 2/3 are the two tensor-parallel groups of a 2x2 plan."""
    results = run_gloo_ranks(_agreement_worker, tmp_path, WORLD_SIZE, {})
    for result in results:
        assert "error" not in result, result["error"]
    by_rank = {result["rank"]: result for result in results}

    for rank, first_peer in ((0, 0), (1, 0), (2, 2), (3, 2)):
        expected = [float(first_peer), first_peer + 0.25]
        assert by_rank[rank]["rewards"] == expected
        assert by_rank[rank]["stored"] == expected
        assert by_rank[rank]["mixed"] == {"count": [first_peer], "half": [first_peer + 0.5] * 2}
        assert by_rank[rank]["mixed_dtypes"] == {"count": "torch.int32", "half": "torch.bfloat16"}
