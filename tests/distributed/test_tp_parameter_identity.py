# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


"""Prepared TP placement identity across rank-invariant distributed layouts."""

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def _identity_worker(rank, store_path, result_dir, options):
    from flow_factory.trainers.common.runtime_identity import _parameter_schema, _schema_digest

    result = {"rank": rank}
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "tp"))
        value = torch.arange(63, dtype=torch.float64).reshape(7, 9)

        def schema(parameter):
            record = SimpleNamespace(
                parameter=parameter, component_name="transformer", parameter_name="weight"
            )
            trainer = SimpleNamespace(
                adapter=SimpleNamespace(
                    component_variant_registry=SimpleNamespace(
                        parameter_records=lambda role: [record]
                    )
                ),
                _required_trainable_roles=lambda: ("base",),
            )
            return _parameter_schema(trainer)[0]

        ordinary = schema(torch.nn.Parameter(value))
        assert ordinary == [
            {
                "key": "base/transformer/weight",
                "shape": [7, 9],
                "dtype": "torch.float64",
                "requires_grad": True,
            }
        ]
        data_only = torch.nn.Parameter(distribute_tensor(value, mesh["dp_shard"], [Shard(0)]))
        assert schema(data_only) == ordinary  # Preserve the data-only checkpoint schema.
        identities = []
        for placements in ([Shard(0), Replicate()], [Shard(0), Shard(1)]):
            parameter = torch.nn.Parameter(distribute_tensor(value, mesh, placements))
            current = schema(parameter)
            digest = _schema_digest(current)
            peers = [None] * 4
            dist.all_gather_object(peers, digest)
            assert len(set(peers)) == 1, peers
            torch.testing.assert_close(parameter.full_tensor(), value, rtol=0, atol=0)
            identities.append(digest)
        assert identities[0] != identities[1]
        assert _schema_digest(ordinary) not in identities
        result.update(status="pass", schemas=identities)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
def test_tp_parameter_identity_detects_placement_and_ignores_rank_local_shape(tmp_path):
    results = run_gloo_ranks(_identity_worker, tmp_path, 4, {})
    assert all(item.get("status") == "pass" for item in results), results
