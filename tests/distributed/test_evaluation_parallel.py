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

"""Logical evaluation geometry and exactly-once async scoring on real meshes."""

import json
import os
import traceback
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.data_loader import prepare_data_loader
from accelerate.parallelism_config import ParallelismConfig
from accelerate.state import PartialState
from conftest import requires_gloo, run_gloo_ranks
from torch.utils.data import DataLoader

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import (
    PointwiseRewardModel,
    RewardBuffer,
    RewardModelOutput,
    RewardProcessor,
)
from flow_factory.samples import T2ISample
from flow_factory.trainers.abc import BaseTrainer


class EvaluationTrainer(BaseTrainer):
    def optimize(self, samples):
        raise AssertionError("This probe only evaluates")


class IndexReward(PointwiseRewardModel):
    required_fields = ("prompt",)

    def __call__(self, prompt):
        self.scored.extend(int(value) for value in prompt)
        return RewardModelOutput(rewards=torch.tensor([float(value) for value in prompt]))


def _worker(rank, store_path, result_dir, options):
    world = 2 * options["tp"] * options["cp"]
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world
    )
    result = {"rank": rank}
    buffers = []
    try:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(world),
            LOCAL_WORLD_SIZE=str(world),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )
        state = PartialState(cpu=True)
        config = ParallelismConfig(
            dp_replicate_size=1,
            dp_shard_size=2,
            tp_size=options["tp"],
            cp_size=options["cp"],
            sp_size=1,
        )
        mesh = config.build_device_mesh("cpu")
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=world,
            process_index=rank,
            is_main_process=rank == 0,
            is_local_main_process=rank == 0,
            distributed_type=state.distributed_type,
            parallelism_config=config,
            torch_device_mesh=mesh,
            wait_for_everyone=dist.barrier,
        )
        for total in (0, 1, 3, 4, 9):
            raw = DataLoader(
                [{"prompt": str(i)} for i in range(total)], batch_size=2, shuffle=False
            )
            loader = prepare_data_loader(
                raw,
                torch.device("cpu"),
                num_processes=world,
                process_index=rank,
                put_on_device=False,
                torch_device_mesh=mesh,
            )
            trainer = object.__new__(EvaluationTrainer)
            trainer.accelerator = accelerator
            trainer.log_args = SimpleNamespace(verbose=False)
            trainer.training_args = SimpleNamespace(seed=17)
            trainer.eval_args = SimpleNamespace()
            trainer.eval_dataloaders = {"bench": loader}
            trainer._eval_dataset_configs = {
                "bench": SimpleNamespace(
                    source_id=0, eval=SimpleNamespace(get_merged_eval_kwargs=lambda _: {})
                )
            }
            trainer.adapter = SimpleNamespace(eval=lambda: None, use_ema_parameters=nullcontext)
            trainer.autocast = nullcontext
            trainer.step = 1
            args = RewardArguments(name="score", device="cpu", async_reward=True, batch_size=2)
            reward = IndexReward(args, accelerator)
            reward.scored = []
            buffer = RewardBuffer(
                RewardProcessor(accelerator, {"score": reward}, {"score": args}, verbose=False),
                group_size=1,
            )
            buffers.append(buffer)
            trainer.eval_dataset_reward_buffers = {"bench": buffer}
            batches, logged = [], []

            def sample_batch(batch, **kwargs):
                assert "reward_buffer" not in kwargs
                batches.append(list(batch["prompt"]))
                return [
                    T2ISample(prompt=p, image=torch.zeros(3, 8, 8), height=8, width=8)
                    for p in batch["prompt"]
                ]

            trainer.sample_batch = sample_batch
            trainer.log_data = lambda data, step: logged.append(data)
            trainer.evaluate()
            scored, inputs = [None] * world, [None] * world
            dist.all_gather_object(scored, reward.scored)
            dist.all_gather_object(inputs, batches)
            assert sorted(i for values in scored for i in values) == list(range(total))
            model_degree = options["tp"] * options["cp"]
            for start in range(0, world, model_degree):
                assert all(inputs[r] == inputs[start] for r in range(start, start + model_degree))
            if rank == 0:
                row = logged[0]
                values = torch.arange(total, dtype=torch.float64)
                assert row["eval/bench/reward_score_count"] == total
                if total:
                    assert row["eval/bench/reward_score_mean"] == float(values.mean())
                    assert row["eval/bench/reward_score_std"] == pytest.approx(
                        float(values.std(correction=0)), rel=1e-12
                    )
                else:
                    assert torch.isnan(torch.tensor(row["eval/bench/reward_score_mean"]))
                assert len({s.prompt for s in row["eval/bench/samples"]}) == len(
                    row["eval/bench/samples"]
                )
            assert not buffer.all_samples
            buffer.shutdown(wait=True)
        result.update(
            status="pass",
            datasets=[0, 1, 3, 4, 9],
            exactly_once_rewards=True,
            model_peer_inputs=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        for buffer in buffers:
            buffer.shutdown(wait=True)
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_evaluation_uses_logical_replicas_and_excludes_padding(tmp_path, tp, cp):
    rows = run_gloo_ranks(_worker, tmp_path, 2 * tp * cp, {"tp": tp, "cp": cp})
    assert all(row.get("status") == "pass" for row in rows), rows
