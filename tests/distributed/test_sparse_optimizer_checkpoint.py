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

"""Two-rank FSDP2 sparse AdamW restore and first-activation continuation."""

import json
import os
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_model_preparation import full_value, initialize_cuda, runtime_observation


class SparseLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8))
        self.bias = torch.nn.Parameter(torch.randn(8))

    def forward(self, value, activate_bias):
        output = value @ self.weight.T
        return output + self.bias if activate_bias else output


def _snapshot(model, optimizer):
    parameters = dict(model.named_parameters())
    return {
        "parameters": {
            name: full_value(parameter).cpu().clone() for name, parameter in parameters.items()
        },
        "state": {
            name: {
                key: full_value(value).cpu().clone()
                for key, value in optimizer.state[parameter].items()
            }
            for name, parameter in parameters.items()
            if parameter in optimizer.state
        },
    }


def _assert_exact(expected, actual):
    assert expected.keys() == actual.keys()
    for name in expected:
        if isinstance(expected[name], dict):
            _assert_exact(expected[name], actual[name])
        else:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


def _worker(rank, store_path, result_dir, options):
    del options
    os.environ.update(
        FSDP_VERSION="2", FSDP_AUTO_WRAP_POLICY="NO_WRAP", FSDP_CPU_RAM_EFFICIENT_LOADING="false"
    )
    initialize_cuda(rank, store_path, world_size=2)
    result = {"rank": rank}
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig

        from flow_factory.trainers.loader import Accelerator

        accelerator = Accelerator(
            mixed_precision="bf16",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2, reshard_after_forward=True, state_dict_type="SHARDED_STATE_DICT"
            ),
            parallelism_config=ParallelismConfig(dp_shard_size=2),
        )
        torch.manual_seed(73)
        model = SparseLinear()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        model, optimizer = accelerator.prepare(model, optimizer)
        result["runtime"] = runtime_observation(accelerator)
        sparse_dir, dense_dir = Path(result_dir) / "sparse", Path(result_dir) / "dense"

        def step(index, activate_bias):
            optimizer.zero_grad(set_to_none=True)
            value = (
                torch.arange(32, device=accelerator.device, dtype=torch.float32).reshape(4, 8) / 32
                + rank / 10
                + index / 100
            )
            with accelerator.autocast():
                loss = model(value, activate_bias).float().square().mean()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step(0, False)
        step(1, False)
        initial = _snapshot(model, optimizer)
        assert "bias" not in initial["state"]
        accelerator.save_state(str(sparse_dir))
        step(2, False)
        step(3, True)
        final = _snapshot(model, optimizer)
        assert final["state"]["bias"]["step"].item() == 1
        accelerator.save_state(str(dense_dir))

        for warm in (True, False):
            if not warm:
                accelerator._optimizers.clear()
                optimizer = accelerator.prepare_optimizer(
                    torch.optim.AdamW(model.parameters(), lr=0.03)
                )
                assert not optimizer.state
            accelerator.load_state(str(sparse_dir))
            _assert_exact(initial, _snapshot(model, optimizer))
            step(2, False)
            step(3, True)
            _assert_exact(final, _snapshot(model, optimizer))
        accelerator.load_state(str(dense_dir))
        _assert_exact(final, _snapshot(model, optimizer))
        torch.save(
            {"initial": initial, "continued": final},
            Path(result_dir) / f"observations-rank{rank}.pt",
        )
        result.update(
            sparse_membership_exact=True,
            fresh_resume_exact=True,
            warm_resume_exact=True,
            inactive_first_activation_exact=True,
            dense_native_resume_exact=True,
            status="pass",
        )
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA ranks")
def test_cuda_sparse_adamw_fsdp2_resume(tmp_path):
    results = run_gloo_ranks(_worker, tmp_path, 2, {}, timeout=120)
    assert all(result.get("status") == "pass" for result in results), results
