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

"""Subgroup collectives partition logical replicas at fixed model-peer coordinates."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.rewards import GroupwiseRewardModel, RewardModelOutput, RewardProcessor
from flow_factory.samples import BaseSample
from flow_factory.utils.dist import gather_samples
from flow_factory.utils.group_coordinator import GroupCoordinator


class MemberOrderReward(GroupwiseRewardModel):
    required_fields = ("prompt",)

    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(list(prompt))
        return RewardModelOutput(rewards=torch.arange(len(prompt), dtype=torch.float32))


def _worker(rank, store_path, result_dir, options):
    model_degree = options["tp"] * options["cp"]
    world = 4 * model_degree
    device_type = options.get("device_type", "cpu")
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    backend = "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(
        backend, store=dist.FileStore(store_path, world), rank=rank, world_size=world
    )
    result = {
        "rank": rank,
        "runtime": {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "device": device.type,
            "process_group_backend": dist.get_backend(),
        },
    }
    try:
        config = ParallelismConfig(
            dp_replicate_size=options["replicate"],
            dp_shard_size=4 // options["replicate"],
            tp_size=options["tp"],
            cp_size=options["cp"],
        )
        mesh = config.build_device_mesh(device_type)

        def forbidden(*args, **kwargs):
            raise AssertionError("subgroup work cannot use Accelerate world collectives")

        accelerator = SimpleNamespace(
            parallelism_config=config,
            torch_device_mesh=mesh,
            process_index=rank,
            num_processes=world,
            device=device,
            gather=forbidden,
            reduce=forbidden,
        )
        dp_rank = rank // model_degree
        model_rank = rank % model_degree
        created = []
        memberships = {}
        new_group = dist.new_group

        def recorded_group(*args, **kwargs):
            created.append(list(kwargs["ranks"]))
            return new_group(*args, **kwargs)

        dist.new_group = recorded_group
        try:
            for size in (1, 2, 4):
                start = dp_rank // size * size
                rows = list(range(start, start + size))
                before = len(created)
                coordinator = GroupCoordinator(
                    accelerator, sampler_type="subgroup_tile", subgroup_size=size
                )
                assert coordinator.dp_rank == dp_rank and coordinator.dp_world_size == 4
                assert (
                    coordinator.group_rank == dp_rank % size
                    and coordinator.group_world_size == size
                )
                if size in (1, 4):
                    assert len(created) == before
                else:
                    expected = [
                        [d * model_degree + peer for d in range(first, first + size)]
                        for peer in range(model_degree)
                        for first in range(0, 4, size)
                    ]
                    assert created[before:] == expected
                if size > 1:
                    assert dist.get_process_group_ranks(coordinator.collective_group) == [
                        d * model_degree + model_rank for d in rows
                    ]
                memberships[size] = [d * model_degree + model_rank for d in rows]
                # Peer-specific values make crossing either a subgroup or model-peer
                # boundary observable in the payload, as well as in group membership.
                values = [d + 100 * model_rank for d in rows]
                value = dp_rank + 100 * model_rank
                tensor = torch.tensor([[value]], dtype=torch.int64, device=device)
                assert coordinator.gather(tensor).flatten().tolist() == values
                assert coordinator.gather_object([f"replica-{dp_rank}-peer-{model_rank}"]) == [
                    f"replica-{d}-peer-{model_rank}" for d in rows
                ]
                payload = torch.tensor([1.0, value, value**2], device=device)
                stats = coordinator.reduce_sum(payload)
                torch.testing.assert_close(
                    stats,
                    torch.tensor(
                        [float(size), float(sum(values)), float(sum(d * d for d in values))],
                        device=device,
                    ),
                    rtol=0,
                    atol=0,
                )
                all_values = [d + 100 * model_rank for d in range(4)]
                torch.testing.assert_close(
                    coordinator.reduce_sum_across_replicas(payload),
                    torch.tensor(
                        [4.0, float(sum(all_values)), float(sum(d * d for d in all_values))],
                        device=device,
                    ),
                    rtol=0,
                    atol=0,
                )
                sample = BaseSample(
                    prompt=f"group-{start} member-{dp_rank % size} 文",
                    source="fixture",
                    source_id=7,
                    sampling_group_id=2**53 + start,
                    sampling_group_member_id=dp_rank % size,
                    sampling_sample_id=dp_rank,
                )
                gathered = gather_samples(
                    accelerator,
                    [sample],
                    ["prompt", "source_id", "sampling_group_id", "sampling_group_member_id"],
                    device=device,
                    group_coordinator=coordinator,
                )
                assert [s.sampling_group_member_id for s in gathered] == list(range(size))
                assert all(
                    s.sampling_group_id == 2**53 + start and s.source_id == 7 for s in gathered
                )
                assert [s.prompt for s in gathered] == [
                    f"group-{start} member-{member} 文" for member in range(size)
                ]
                # The raw NCCL catalog cases qualify transport and scopes only.
                # Groupwise reward ownership remains a separate Gloo assertion.
                if device_type == "cuda":
                    continue
                reward = MemberOrderReward()
                processor = RewardProcessor(
                    accelerator,
                    {"member": reward},
                    group_on_same_rank=False,
                    verbose=False,
                    group_coordinator=coordinator,
                )
                scored = processor.compute_rewards([sample], store_to_samples=False)
                torch.testing.assert_close(
                    scored["member"], torch.tensor([float(dp_rank % size)]), rtol=0, atol=0
                )
                calls = [None] * world
                dist.all_gather_object(calls, reward.calls)
                actual = sorted(batch for peer_calls in calls for batch in peer_calls)
                expected_calls = [
                    [f"group-{first} member-{member} 文" for member in range(size)]
                    for first in range(0, 4, size)
                ]
                assert actual == expected_calls
            before = len(created)
            with pytest.raises(ValueError):
                GroupCoordinator(accelerator, sampler_type="subgroup_tile", subgroup_size=3)
            assert len(created) == before
        finally:
            dist.new_group = new_group
        result.update(
            status="pass",
            subgroup_sizes=[1, 2, 4],
            created_groups=created,
            memberships=memberships,
            exact_identity=True,
            single_owner_group_rewards=device_type == "cpu",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("tp,cp,replicate", [(2, 1, 1), (1, 2, 1), (2, 2, 1), (2, 1, 2)])
def test_subgroup_tiles_follow_logical_replica_order(tmp_path, tp, cp, replicate):
    rows = run_gloo_ranks(
        _worker, tmp_path, 4 * tp * cp, {"tp": tp, "cp": cp, "replicate": replicate}
    )
    assert all(row.get("status") == "pass" for row in rows), rows


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 8,
    reason="eight CUDA devices and NCCL are required",
)
@pytest.mark.parametrize("tp,cp", [(2, 1), (1, 2)], ids=["tp2-cp1", "tp1-cp2"])
def test_cuda_subgroup_tiles_follow_logical_replica_order(tmp_path, tp, cp):
    """Two R2 subgroups retain isolated NCCL payloads at each model-peer coordinate."""
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        8,
        {"tp": tp, "cp": cp, "replicate": 1, "device_type": "cuda"},
    )
    for rank, row in enumerate(rows):
        assert row.get("status") == "pass", row
        assert row["runtime"] == {
            "rank": rank,
            "world_size": 8,
            "device": "cuda",
            "process_group_backend": "nccl",
        }
        assert row["created_groups"] == rows[0]["created_groups"]
        assert row["single_owner_group_rewards"] is False
    print(
        json.dumps(
            {
                "probe": "subgroup_topology_only",
                "tp_size": tp,
                "cp_size": cp,
                "dp_size": 4,
                "subgroup_sizes": [1, 2, 4],
                "ranks_completed": [row["rank"] for row in rows],
                "runtime": rows[0]["runtime"],
                "r2_memberships": [row["memberships"]["2"] for row in rows],
                "isolated_payloads_and_replica_reductions": "passed",
            }
        )
    )
