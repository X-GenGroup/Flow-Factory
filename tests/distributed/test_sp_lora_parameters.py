# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


"""Replicated LoRA parameters on real token shards retain every derivative/update."""

import copy
import json
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor


def _lora_worker(rank, store_path, result_dir, options):
    from peft import LoraConfig, get_peft_model

    from flow_factory.models.tensor_parallel import (
        SequenceBroadcastParameter,
        apply_tensor_parallel,
        peel_peft_wrapper,
    )

    result = {"rank": rank}
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        torch.manual_seed(199)
        model = torch.nn.Sequential(
            torch.nn.Linear(12, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4)
        ).double()
        model = get_peft_model(
            model, LoraConfig(r=3, lora_alpha=3, target_modules=["0", "2"], init_lora_weights=False)
        )
        reference = copy.deepcopy(model)
        apply_tensor_parallel(
            peel_peft_wrapper(model),
            mesh,
            {"0": SequenceBroadcastParameter(), "2": SequenceBroadcastParameter()},
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        active = []
        for step in range(3):
            torch.manual_seed(211 + step)
            inputs = torch.randn(1, 8, 12, dtype=torch.float64, requires_grad=True)
            refinputs = inputs.detach().clone().requires_grad_()
            target = torch.randn(1, 8, 4, dtype=torch.float64)
            local = inputs[:, rank * 4 : (rank + 1) * 4]
            actual = model(local)
            expected = reference(refinputs)
            torch.testing.assert_close(
                actual, expected[:, rank * 4 : (rank + 1) * 4], rtol=1e-12, atol=1e-12
            )
            (
                (actual - target[:, rank * 4 : (rank + 1) * 4]).square().sum() / target.numel()
            ).backward()
            (expected - target).square().mean().backward()
            dist.all_reduce(inputs.grad)
            torch.testing.assert_close(inputs.grad, refinputs.grad, rtol=1e-12, atol=1e-12)
            for (name, parameter), (other, refparam) in zip(
                model.named_parameters(), reference.named_parameters()
            ):
                assert name == other and parameter.requires_grad == refparam.requires_grad
                assert (parameter.grad is None) == (refparam.grad is None)
                if parameter.grad is not None:
                    torch.testing.assert_close(
                        parameter.grad.full_tensor(), refparam.grad, rtol=1e-12, atol=1e-12
                    )
                    active.append(name)
            optimizer.step()
            ref_optimizer.step()
            for parameter, refparam in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(
                    parameter.full_tensor(), refparam, rtol=1e-12, atol=1e-12
                )
                for key, value in optimizer.state[parameter].items():
                    torch.testing.assert_close(
                        value.full_tensor() if isinstance(value, DTensor) else value,
                        ref_optimizer.state[refparam][key],
                        rtol=1e-12,
                        atol=1e-12,
                    )
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
        assert len(set(active)) == 4
        result.update(status="pass", optimizer_steps=3, active_parameters=sorted(set(active)))
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
def test_replicated_lora_token_shards_preserve_input_and_parameter_updates(tmp_path):
    results = run_gloo_ranks(_lora_worker, tmp_path, 2, {})
    assert all(item.get("status") == "pass" for item in results), results
