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

"""Initial FSDP2 state must be independent of rank-local initialization RNG."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.loading.backend import FSDPBackendLoadRuntime


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    try:
        torch.manual_seed(123 + rank)
        module = torch.nn.Linear(4, 4)
        module.register_buffer("running", torch.randn(4))
        mismatch = options["mismatch"]
        if mismatch == "shape" and rank:
            module.register_buffer("extra", torch.zeros(1))
        if mismatch == "requires_grad" and rank:
            module.weight.requires_grad_(False)
        if mismatch == "meta":
            module.to_empty(device="meta")
        if mismatch == "dtensor":
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import DTensor, Replicate

            mesh = init_device_mesh("cpu", (2,))
            module.weight = torch.nn.Parameter(
                DTensor.from_local(module.weight.detach(), mesh, [Replicate()], run_check=False)
            )
        identities = [id(t) for t in (*module.parameters(), *module.buffers())]
        rng = torch.get_rng_state().clone()
        accelerator = SimpleNamespace(
            num_processes=2,
            device=torch.device("cpu"),
            state=SimpleNamespace(
                fsdp_plugin=SimpleNamespace(
                    fsdp_version=2,
                    cpu_ram_efficient_loading=False,
                )
            ),
        )
        runtime = FSDPBackendLoadRuntime(accelerator, None, None)
        if mismatch != "none":
            before = {
                k: (v.to_local() if hasattr(v, "to_local") else v).clone()
                for k, v in module.state_dict().items()
                if not v.is_meta
            }
            with pytest.raises(ValueError, match="matching, materialized"):
                runtime.synchronize_initial_state((module,))
            for k, v in module.state_dict().items():
                if not v.is_meta:
                    local = v.to_local() if hasattr(v, "to_local") else v
                    torch.testing.assert_close(local, before[k], rtol=0, atol=0)
            assert identities == [id(t) for t in (*module.parameters(), *module.buffers())]
        else:
            runtime.synchronize_initial_state((module, module))
            assert identities == [id(t) for t in (*module.parameters(), *module.buffers())]
            assert torch.equal(rng, torch.get_rng_state())
            torch.manual_seed(123)
            expected = torch.nn.Linear(4, 4)
            expected.register_buffer("running", torch.randn(4))
            for k, v in module.state_dict().items():
                torch.testing.assert_close(v, expected.state_dict()[k], rtol=0, atol=0)
        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "status": "pass"})
        )
    finally:
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("mismatch", ["none", "shape", "requires_grad", "meta", "dtensor"])
def test_fsdp2_initial_state_is_synchronized_before_sharding(tmp_path, mismatch):
    results = run_gloo_ranks(_worker, tmp_path, 2, {"mismatch": mismatch})
    assert all(row["status"] == "pass" for row in results)
