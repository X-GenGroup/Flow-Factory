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

"""Multiple checkpointed forwards share nested FSDP units before backward."""

import copy
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from conftest import run_gloo_ranks
from packaging.version import Version
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.utils.checkpoint import checkpoint

from flow_factory.loading.backend import FSDPBackendLoadRuntime, _unshard_fsdp2_checkpoint_replay
from flow_factory.loading.domain import ComponentDescriptor, ComponentRole, LoadPlanner


class ReplayBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, x):
        return torch.nn.functional.gelu(checkpoint(self.proj, x, use_reentrant=False))


class ReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ReplayBlock(), ReplayBlock()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _worker(rank, store_path, result_dir, options):
    result = {}
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{store_path}", rank=rank, world_size=2)
    try:
        torch.manual_seed(71)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = ReplayModel().cuda()
        dense = copy.deepcopy(model)
        mesh = init_device_mesh("cuda", (2,))
        plugin = SimpleNamespace(
            fsdp_version=2, cpu_ram_efficient_loading=False, activation_checkpointing=True
        )

        def prepare(module):
            for block in module.blocks:
                fully_shard(block.proj, mesh=mesh)
                fully_shard(block, mesh=mesh)
            fully_shard(module, mesh=mesh)
            return module

        accelerator = SimpleNamespace(
            state=SimpleNamespace(fsdp_plugin=plugin),
            device=torch.device("cuda", rank),
            distributed_type=DistributedType.FSDP,
            prepare=prepare,
        )
        plan = LoadPlanner().build(
            [ComponentDescriptor(name="transformer", root="transformer", role=ComponentRole.TARGET)]
        )
        runtime = FSDPBackendLoadRuntime(
            accelerator,
            plan,
            SimpleNamespace(
                fsdp2_use_in_forward_activation_checkpointing=True,
                configure_fsdp2_in_forward_activation_checkpointing=lambda root: len(root.blocks),
            ),
        )
        model = runtime.prepare(model)
        fsdp_units = sum(isinstance(m, FSDPModule) for m in model.modules())
        assert fsdp_units == 5
        result["runtime"] = {
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
            "device": next(model.parameters()).device.type,
            "distributed_type": "FSDP",
            "fsdp_version": 2,
            "prepared_fsdp_units": fsdp_units,
            "preparation_api": "torch.distributed.fsdp.fully_shard",
            "process_group_backend": dist.get_backend(),
        }
        hooks = sum(
            h is _unshard_fsdp2_checkpoint_replay
            for m in model.modules()
            for h in m._forward_pre_hooks.values()
        )
        native_replay = Version(torch.__version__).release[:2] >= (2, 11)
        assert (hooks == 0) if native_replay else (hooks == 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        reference_optimizer = torch.optim.AdamW(dense.parameters(), lr=0.001, foreach=False)
        generator = torch.Generator(device="cuda").manual_seed(123)
        errors = []
        for _ in range(3):
            inputs = [torch.randn(8, 16, generator=generator, device="cuda") for _ in range(2)]
            # Both graphs exist before either backward; the first backward can
            # reshard a block that the second graph must recompute.
            outputs = [model(x.chunk(2)[rank]) for x in inputs]
            reference_outputs = [dense(x) for x in inputs]
            for actual, expected in zip(outputs, reference_outputs):
                torch.testing.assert_close(actual, expected.chunk(2)[rank], rtol=2e-5, atol=2e-6)
            for actual, expected in zip(outputs, reference_outputs):
                actual.square().mean().backward()
                expected.square().mean().backward()
            for (name, param), (ref_name, reference) in zip(
                model.named_parameters(), dense.named_parameters()
            ):
                assert name == ref_name
                grad = param.grad.full_tensor()
                torch.testing.assert_close(grad, reference.grad, rtol=2e-5, atol=2e-6)
                errors.append(float((grad - reference.grad).abs().max()))
            optimizer.step()
            reference_optimizer.step()
            for param, reference in zip(model.parameters(), dense.parameters()):
                torch.testing.assert_close(param.full_tensor(), reference, rtol=2e-5, atol=2e-6)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        result.update(
            status="pass",
            torch=torch.__version__,
            replay_hooks=hooks,
            max_gradient_error=max(errors),
            steps=3,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA ranks")
def test_cuda_nested_fsdp_multiple_checkpointed_forwards(tmp_path):
    rows = run_gloo_ranks(_worker, tmp_path, 2, {})
    assert all(row.get("status") == "pass" for row in rows), rows
    print(json.dumps(rows))
