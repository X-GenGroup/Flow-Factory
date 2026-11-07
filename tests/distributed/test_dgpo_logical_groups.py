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


"""Actual DGPO group collectives count logical replicas once, including packed groups."""

import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.samples import BaseSample, LatentState, NoisedState
from flow_factory.utils.group_coordinator import GroupCoordinator
from tests.trainers.test_dgpo_crd_parity import _adapter, _dgpo_trainer


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 4), rank=rank, world_size=4)
    result = {
        "rank": rank,
        "runtime": {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "device": "cpu",
            "process_group_backend": dist.get_backend(),
        },
    }
    try:
        config = ParallelismConfig(dp_shard_size=2, cp_size=2)
        mesh = config.build_device_mesh("cpu")

        def forbidden(*args, **kwargs):
            raise AssertionError("DGPO must not use a launched-world accelerator collective")

        accelerator = SimpleNamespace(
            parallelism_config=config,
            torch_device_mesh=mesh,
            device=torch.device("cpu"),
            process_index=rank,
            num_processes=4,
            gather=forbidden,
            reduce=forbidden,
        )
        replica, model_peer = divmod(rank, 2)
        samples = [
            BaseSample(
                source=f"source-{replica}",
                source_id=replica + 1,
                sampling_group_id=2**53 + 3,
                sampling_group_member_id=i,
                sampling_sample_id=replica * 3 + i,
            )
            for i in range(3)
        ]
        trainer = _dgpo_trainer(_adapter(), group_size=3, dpo_beta=2.0)
        trainer.accelerator = accelerator
        trainer.group_coordinator = GroupCoordinator(accelerator, sampler_type="global_batch")
        assert dist.get_process_group_ranks(trainer.group_coordinator.collective_group) == [
            model_peer,
            model_peer + 2,
        ]
        info = trainer._precompute_group_info(samples)
        assert info["num_groups"] == 2
        assert info["local_group_indices"].tolist() == [replica] * 3
        values = torch.tensor([[0.2, 0.8], [0.5, -0.3], [1.0, 0.4]], dtype=torch.float64)
        values = (values + 0.3 * replica + 0.25 * model_peer).requires_grad_()
        target = torch.ones_like(values) * 0.1
        reference = torch.zeros_like(values)
        noised = NoisedState(
            state=LatentState({"latent": target}),
            target_velocity=LatentState({"latent": target}),
            noise=LatentState({"latent": torch.zeros_like(values)}),
        )
        dsm = trainer._compute_dsm_loss(
            noised.target_velocity, LatentState({"latent": values}), noised
        )
        adv = torch.tensor([1.0, -0.75, 0.5], dtype=torch.float64)
        result_loss = trainer._compute_group_dgpo_loss(
            LatentState({"latent": reference}), noised.target_velocity, adv, info, dsm, noised
        )
        result_loss.backward()
        errors = (values.detach() - target).square().mean(1)
        group_sum = math.fsum(
            float(a * 2 * (error - 0.01) / 3) for a, error in zip(adv, errors, strict=True)
        )
        weight = 1 / (1 + math.exp(-group_sum))
        expected_loss = (
            weight * math.fsum(float(a * error) for a, error in zip(adv, errors, strict=True)) / 3
        )
        expected_gradient = weight * adv[:, None] * (values.detach() - target) / 3
        torch.testing.assert_close(
            result_loss.detach(),
            torch.tensor(expected_loss, dtype=torch.float64),
            rtol=2e-12,
            atol=2e-12,
        )
        torch.testing.assert_close(values.grad, expected_gradient, rtol=2e-12, atol=2e-12)
        result.update(
            status="pass",
            groups=2,
            local_group=replica,
            model_peer=model_peer,
            group_sum=group_sum,
            value=float(result_loss.detach()),
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
def test_dgpo_packed_groups_use_logical_dp_scope(tmp_path):
    rows = run_gloo_ranks(_worker, tmp_path, 4, {}, timeout=120)
    assert all(row.get("status") == "pass" for row in rows), rows
    assert rows[0]["group_sum"] != pytest.approx(rows[1]["group_sum"])
    assert rows[2]["group_sum"] != pytest.approx(rows[3]["group_sum"])
