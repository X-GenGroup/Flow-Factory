# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""TDM-R1 identity admission uses logical replicas and fails collectively."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer


def _host(accelerator):
    trainer = object.__new__(TDMR1Trainer)
    trainer.accelerator = accelerator
    trainer.config = SimpleNamespace(data_args=SimpleNamespace(sampler_type="global_batch"))
    return trainer


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 4), rank=rank, world_size=4)
    result = dict(
        rank=rank, runtime=dict(rank=rank, world_size=4, device="cpu", process_group_backend="gloo")
    )
    original = dist.all_reduce
    try:
        cp = options["cp"]
        config = ParallelismConfig(dp_shard_size=4 // cp, cp_size=cp)
        mesh = config.build_device_mesh("cpu")
        calls = []

        def forbidden(*args, **kwargs):
            raise AssertionError("identity extrema must use the explicit logical DP owner")

        def observed(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
            calls.append(
                (
                    tuple(tensor.shape),
                    tuple(dist.get_process_group_ranks(group or dist.group.WORLD)),
                )
            )
            return original(tensor, op=op, group=group, **kwargs)

        dist.all_reduce = observed
        accelerator = SimpleNamespace(
            parallelism_config=config,
            torch_device_mesh=mesh,
            num_processes=4,
            process_index=rank,
            reduce=forbidden,
        )
        trainer = _host(accelerator)
        replica, peer = divmod(rank, cp)
        members = tuple(range(peer, 4, cp))
        identities = torch.tensor([[peer * 100, 7], [peer * 100, 9]], dtype=torch.int64)
        before = identities.clone()
        trainer._validate_shared_group_identity(identities)
        assert torch.equal(identities, before)
        assert len(calls) == 4 and all(group == members for _, group in calls)
        valid_identity = identities.tolist()

        failures = []
        for corruption in ("value", "shape"):
            calls.clear()
            candidate = identities.clone()
            if replica == 0:
                if corruption == "value":
                    candidate[0, 1] += 3
                else:
                    candidate = candidate[:1]
            before = candidate.clone()
            try:
                trainer._validate_shared_group_identity(candidate)
            except RuntimeError as error:
                assert "same reward groups" in str(error)
                failures.append(corruption)
            else:
                raise AssertionError(
                    f"all replicas, including the maximum, must reject {corruption}"
                )
            assert torch.equal(candidate, before)
            assert len(calls) == (2 if corruption == "shape" else 4)
            assert all(group == members for _, group in calls)
            # Every process reached this point instead of a peer continuing to a different collective.
            dist.barrier()
        result.update(
            status="pass",
            valid_identity=valid_identity,
            logical_members=members,
            synchronized_rejections=failures,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        dist.all_reduce = original
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("cp", [1, 2], ids=["ordinary-dp4", "dp2-cp2"])
def test_equal_share_identity_is_scoped_and_all_replicas_reject_mismatches(tmp_path, cp):
    rows = run_gloo_ranks(_worker, tmp_path, 4, {"cp": cp}, timeout=120)
    assert all(row.get("status") == "pass" for row in rows), rows
    if cp == 2:
        assert rows[0]["valid_identity"] != rows[1]["valid_identity"]
        assert rows[0]["logical_members"] == [0, 2]
        assert rows[1]["logical_members"] == [1, 3]


@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_single_replica_identity_never_communicates(monkeypatch, tp, cp):
    def forbidden(*args, **kwargs):
        raise AssertionError("one logical replica must not communicate")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    config = SimpleNamespace(
        tp_size=tp, cp_size=cp, dp_shard_size=1, dp_replicate_size=1, sp_size=1
    )
    trainer = _host(
        SimpleNamespace(
            parallelism_config=config, num_processes=tp * cp, process_index=0, reduce=forbidden
        )
    )
    trainer._validate_shared_group_identity(torch.tensor([[-1, 7], [-1, 9]]))
