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

"""Real DPO sample gathering uses logical replicas, preserves sources and counts unique pairs."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.samples import BaseSample
from flow_factory.trainers.rl.dpo import DPOTrainer
from flow_factory.utils.group_coordinator import GroupCoordinator


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 4), rank=rank, world_size=4)
    result = {"rank": rank}
    try:
        config = ParallelismConfig(dp_shard_size=2, cp_size=2)
        mesh = config.build_device_mesh("cpu")

        def forbidden(*args, **kwargs):
            raise AssertionError("DPO cannot gather/reduce duplicate model peers")

        accelerator = SimpleNamespace(
            parallelism_config=config,
            torch_device_mesh=mesh,
            process_index=rank,
            num_processes=4,
            device=torch.device("cpu"),
            gather=forbidden,
            reduce=forbidden,
        )
        coordinator = GroupCoordinator(accelerator, sampler_type="global_batch")
        trainer = object.__new__(DPOTrainer)
        trainer.accelerator = accelerator
        trainer.group_coordinator = coordinator
        trainer.epoch = 0
        trainer.advantage_processor = SimpleNamespace(group_on_same_rank=False)
        replica, peer = divmod(rank, 2)
        # Three complete groups distributed over DP2; source 0 and 1 reuse group id 7.
        # A peer-specific prompt payload exposes an accidental cross-model gather.
        specs = [(0, 7), (0, 8), (1, 7)]
        samples = [
            BaseSample(
                source=f"s{source}",
                source_id=source,
                prompt=f"peer-{peer}",
                sampling_group_id=group,
                sampling_sample_id=2 * i + replica,
                sampling_group_member_id=replica,
                extra_kwargs={"advantage": torch.tensor(float((i + 1) * (1 if replica else -1)))},
            )
            for i, (source, group) in enumerate(specs)
        ]
        pairs, metrics = trainer._form_pairs(samples)
        actual = [(a.sampling_sample_id, b.sampling_sample_id) for a, b in pairs]
        assert actual == ([(1, 0), (5, 4)] if replica == 0 else [(3, 2), (3, 2)])
        assert all(
            a.source_id == b.source_id and a.prompt == b.prompt == f"peer-{peer}" for a, b in pairs
        )
        assert metrics["train/dpo_num_pairs"] == 3
        assert metrics["train/dpo_chosen_adv_mean"] == pytest.approx(2)
        assert metrics["train/dpo_rejected_adv_mean"] == pytest.approx(-2)
        assert metrics["train/dpo_adv_margin_mean"] == pytest.approx(4)
        with pytest.raises(RuntimeError, match="need at least 2 chosen/rejected pairs.*got 1"):
            trainer._form_pairs(samples[:1])
        # Rank-local pair metrics also count replicas once, not both CP peers.
        trainer.group_coordinator = GroupCoordinator(accelerator, sampler_type="rank_local")
        trainer.advantage_processor.group_on_same_rank = True
        local = [
            BaseSample(
                prompt=f"peer-{peer}",
                sampling_group_id=replica,
                sampling_sample_id=2 * replica + i,
                extra_kwargs={"advantage": torch.tensor(float(i))},
            )
            for i in (0, 1)
        ]
        local_pairs, local_metrics = trainer._form_pairs(local)
        assert len(local_pairs) == 1 and local_metrics["train/dpo_num_pairs"] == 2
        result.update(
            status="pass",
            pairs=actual,
            unique_pairs=metrics["train/dpo_num_pairs"],
            local_unique_pairs=local_metrics["train/dpo_num_pairs"],
            dp_rank=replica,
            model_peer=peer,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
def test_real_dpo_pair_gather_counts_logical_replicas_and_unpadded_pairs(tmp_path):
    results = run_gloo_ranks(_worker, tmp_path, 4, {})
    assert all(row.get("status") == "pass" for row in results), results
    assert results[0]["pairs"] == results[1]["pairs"]
    assert results[2]["pairs"] == results[3]["pairs"]
