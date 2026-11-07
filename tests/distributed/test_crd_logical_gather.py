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

"""Real four-rank logical-DP gathering with deliberately different CP peers."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.utils.dist import gather_aligned_floating_tensors


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 4), rank=rank, world_size=4)
    result = dict(
        rank=rank, runtime=dict(rank=rank, world_size=4, device="cpu", process_group_backend="gloo")
    )
    try:
        config = ParallelismConfig(dp_shard_size=2, cp_size=2)
        mesh = config.build_device_mesh("cpu")

        def forbidden(*args, **kwargs):
            raise AssertionError("logical gathering must not use the launched world")

        accelerator = SimpleNamespace(
            parallelism_config=config,
            torch_device_mesh=mesh,
            num_processes=4,
            process_index=rank,
            gather=forbidden,
        )
        replica, peer = divmod(rank, 2)
        calls = []
        original = dist.all_gather

        def gather(outputs, tensor, *, group=None):
            calls.append((tuple(tensor.shape), tuple(dist.get_process_group_ranks(group))))
            return original(outputs, tensor, group=group)

        dist.all_gather = gather
        for reward_dtype in (torch.float32, torch.float64):
            fields = {
                "advantage": torch.tensor([rank + 0.25, rank + 0.75]),
                "implicit_reward": torch.tensor(
                    [rank * 10 + 0.5, rank * 10 + 1.5], dtype=reward_dtype
                ),
            }
            calls.clear()
            actual = gather_aligned_floating_tensors(accelerator, fields)
            expected_ranks = (peer, peer + 2)
            assert len(calls) == (1 if reward_dtype == torch.float32 else 2)
            assert all(group == expected_ranks for _, group in calls)
            assert actual["advantage"].tolist() == [
                r + offset for r in expected_ranks for offset in (0.25, 0.75)
            ]
            assert actual["implicit_reward"].tolist() == [
                r * 10 + offset for r in expected_ranks for offset in (0.5, 1.5)
            ]
            assert actual["implicit_reward"].dtype == reward_dtype
            assert actual["advantage"].shape == (4,)
        result.update(
            status="pass",
            logical_rows=4,
            launched_rows=8,
            peer=peer,
            advantages=actual["advantage"].tolist(),
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
def test_crd_gather_counts_distinct_replicas_with_aligned_packed_and_mixed_fields(tmp_path):
    rows = run_gloo_ranks(_worker, tmp_path, 4, {}, timeout=120)
    assert all(row.get("status") == "pass" for row in rows), rows
    assert rows[0]["advantages"] != rows[1]["advantages"]


@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_single_logical_replica_never_communicates(monkeypatch, tp, cp):
    def forbidden(*args, **kwargs):
        raise AssertionError("one logical replica must not communicate")

    config = SimpleNamespace(
        tp_size=tp, cp_size=cp, dp_shard_size=1, dp_replicate_size=1, sp_size=1
    )
    accelerator = SimpleNamespace(
        num_processes=tp * cp, process_index=0, parallelism_config=config, gather=forbidden
    )
    monkeypatch.setattr(dist, "all_gather", forbidden)
    for dtype in (torch.float32, torch.float64):
        fields = {"advantage": torch.ones(2), "implicit_reward": torch.ones(2, dtype=dtype)}
        actual = gather_aligned_floating_tensors(accelerator, fields)
        assert all(actual[key] is value for key, value in fields.items())
