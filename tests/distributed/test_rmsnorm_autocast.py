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

"""Native CUDA autocast and TP normalization agree within each Torch version."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_model_preparation import initialize_cuda
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint


def _worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, 2)
    result = {
        "rank": rank,
        "torch": torch.__version__,
        "cases": [],
        "runtime": {
            "world_size": dist.get_world_size(),
            "device": "cuda",
            "distributed_type": "PROCESS_GROUP",
            "process_group_backend": dist.get_backend(),
        },
    }
    try:
        from flow_factory.models.parallelism.guards import _install_norm_pair
        from flow_factory.models.parallelism.tensor import ShardedAcrossHeadRMSNorm

        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        for dtype in (torch.bfloat16, torch.float16):
            for enabled in (False, True):
                for paired in (False, True):
                    for recompute in (False, True):
                        torch.manual_seed(138)
                        reference = nn.ModuleDict(
                            {"norm_q": nn.RMSNorm(64, eps=1e-6), "norm_k": nn.RMSNorm(64, eps=1e-6)}
                        ).cuda()
                        actual = copy.deepcopy(reference)
                        for norm in actual.values():
                            ShardedAcrossHeadRMSNorm()._apply(norm, mesh)
                        # The pairing owner expects named attributes and a K producer.
                        attention = nn.Module()
                        attention.norm_q, attention.norm_k = actual["norm_q"], actual["norm_k"]
                        attention.to_k = nn.Identity()
                        if paired:
                            _install_norm_pair(attention, attention.to_k, mesh.get_group())
                        inputs = [
                            torch.randn(2, length, 64, device="cuda", dtype=dtype).requires_grad_()
                            for length in (7, 5)
                        ]
                        local = [
                            x.detach().chunk(2, -1)[rank].clone().requires_grad_() for x in inputs
                        ]
                        upstream = [torch.randn_like(x) for x in inputs]

                        def forward(q, k):
                            k = attention.to_k(k)
                            return attention.norm_q(q), attention.norm_k(k)

                        with torch.autocast("cuda", enabled=enabled, dtype=dtype):
                            expected = [
                                reference[name](x) for name, x in zip(("norm_q", "norm_k"), inputs)
                            ]
                            outputs = (
                                checkpoint(forward, *local, use_reentrant=False)
                                if recompute
                                else forward(*local)
                            )
                        assert [x.dtype for x in outputs] == [x.dtype for x in expected]
                        sum(
                            (x.float() * u.float()).sum() for x, u in zip(expected, upstream)
                        ).backward()
                        sum(
                            (x.float() * u.chunk(2, -1)[rank].float()).sum()
                            for x, u in zip(outputs, upstream)
                        ).backward()
                        pairs = []
                        for i, name in enumerate(("norm_q", "norm_k")):
                            pairs.extend(
                                [
                                    (outputs[i], expected[i].chunk(2, -1)[rank]),
                                    (local[i].grad, inputs[i].grad.chunk(2, -1)[rank]),
                                    (
                                        actual[name].weight.grad.to_local(),
                                        reference[name].weight.grad.chunk(2)[rank],
                                    ),
                                ]
                            )
                        maximum = 0.0
                        for value, target in pairs:
                            assert torch.isfinite(value).all() and torch.count_nonzero(value)
                            # Floating reductions differ in order. Bound errors at
                            # the endpoint's precision, including BF16 input grads.
                            quantum = torch.finfo(value.dtype).eps
                            torch.testing.assert_close(
                                value, target, rtol=0, atol=8 * quantum * target.abs().max().item()
                            )
                            error = (
                                (value.float() - target.float()).norm() / target.float().norm()
                            ).item()
                            assert error <= 8 * quantum
                            maximum = max(maximum, error)
                        result["cases"].append(
                            {
                                "dtype": str(dtype),
                                "autocast": enabled,
                                "paired": paired,
                                "checkpoint": recompute,
                                "output_dtype": str(outputs[0].dtype),
                                "max_relative_error": maximum,
                            }
                        )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA ranks")
def test_cuda_tp_rmsnorm_matches_native_autocast(tmp_path):
    rows = run_gloo_ranks(_worker, tmp_path, 2, {})
    assert all(row.get("status") == "pass" and len(row["cases"]) == 16 for row in rows), rows
    print(json.dumps(rows))
