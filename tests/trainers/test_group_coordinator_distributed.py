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

"""Real gloo coverage for subgroup-scoped group collectives."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from flow_factory.advantage import GroupCoordinator
from flow_factory.rewards import GroupwiseRewardModel, RewardModelOutput
from flow_factory.rewards.reward_processor import RewardProcessor
from flow_factory.samples import BaseSample
from flow_factory.utils.dist import gather_samples


class _PromptGroupReward(GroupwiseRewardModel):
    required_fields = ("prompt",)

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, prompt: list[str]) -> RewardModelOutput:
        self.batch_sizes.append(len(prompt))
        return RewardModelOutput(rewards=torch.arange(len(prompt), dtype=torch.float32))


class _Tokenizer:
    @staticmethod
    def decode(ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "-".join(str(value) for value in ids)


def _run_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=world_size,
        process_index=rank,
        is_local_main_process=rank == 0,
        wait_for_everyone=lambda: dist.barrier(),
        gather=lambda _tensor: (_ for _ in ()).throw(
            AssertionError("subgroup collectives must not use the global accelerator gather")
        ),
        reduce=lambda _tensor, reduction: (_ for _ in ()).throw(
            AssertionError(
                f"subgroup collectives must not use global accelerator reduce={reduction!r}"
            )
        ),
    )
    coordinator = GroupCoordinator(
        accelerator,
        sampler_type="subgroup_tile",
        subgroup_size=2,
    )

    gathered = coordinator.gather(torch.tensor([[rank]], dtype=torch.int64)).flatten()
    subgroup_start = rank - rank % 2
    torch.testing.assert_close(
        gathered,
        torch.tensor([subgroup_start, subgroup_start + 1], dtype=torch.int64),
    )
    reduced = coordinator.reduce_sum(torch.tensor([rank + 1.0]))
    expected_sum = float((subgroup_start + 1) + (subgroup_start + 2))
    torch.testing.assert_close(reduced, torch.tensor([expected_sum]))
    coordinator.gather_object = lambda _values: (_ for _ in ()).throw(
        AssertionError("tensorizable groupwise fields must not use pickle collectives")
    )
    string_sample = BaseSample(
        prompt=f"rank-{rank}-" + "短" * (rank + 1),
        source=f"source-{rank}",
        source_id=rank,
        sampling_group_id=rank // 2,
    )
    gathered_samples = gather_samples(
        accelerator,
        [string_sample],
        ["prompt", "source", "source_id", "sampling_group_id"],
        group_coordinator=coordinator,
    )
    assert [sample.prompt for sample in gathered_samples] == [
        f"rank-{peer}-" + "短" * (peer + 1) for peer in range(subgroup_start, subgroup_start + 2)
    ]

    reward_model = _PromptGroupReward()
    processor = RewardProcessor(
        accelerator=accelerator,
        reward_models={"group": reward_model},
        tokenizer=_Tokenizer(),
        group_on_same_rank=False,
        verbose=False,
        group_coordinator=coordinator,
    )
    sample = BaseSample(
        prompt=f"subgroup-{subgroup_start}",
        prompt_ids=torch.tensor([subgroup_start, rank], dtype=torch.int64),
        sampling_group_id=rank // 2,
        sampling_group_member_id=rank % 2,
        sampling_sample_id=rank,
    )
    rewards = processor.compute_rewards([sample], store_to_samples=False)
    torch.testing.assert_close(rewards["group"], torch.tensor([float(rank % 2)]))
    assert reward_model.batch_sizes == ([2] if rank % 2 == 0 else [])
    dist.barrier()
    dist.destroy_process_group()


def test_subgroup_coordinator_uses_only_its_two_rank_scope() -> None:
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repo_root), env.get("PYTHONPATH", "")) if value
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(test_file),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=90,
    )


if __name__ == "__main__":
    _run_worker()
