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

"""Dense clipping and Adam equations, including actual mixed DTensor norm meshes."""

import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from test_context_parallel_training import (
    _check_adam_trajectory,
    _check_clipping,
    _clipping_norm_interval,
    _optimizer_fixture,
)


@pytest.mark.parametrize("counts", [(6_000_000,), (655_360,) * 32])
def test_dense_active_clipping_rejects_zero_half_double_and_no_clipping(counts):
    parameters = [torch.nn.Parameter(torch.zeros(count)) for count in counts]
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 2 / sum(counts) ** 0.5)
    gradients = {str(i): parameter.grad.clone() for i, parameter in enumerate(parameters)}
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=True)
    clipped = {str(i): parameter.grad for i, parameter in enumerate(parameters)}
    update = {
        "gradients": gradients,
        "clipped_gradients": clipped,
        "clipping_world_size": 1,
        "optimizer": {"max_grad_norm": 1.0},
    }
    coefficient, _ = _check_clipping(update)
    assert 0.49 < coefficient < 0.51
    for factor in (0.0, 0.5, 2.0):
        update["clipped_gradients"] = {name: value * factor for name, value in clipped.items()}
        with pytest.raises(AssertionError, match="clipping:"):
            _check_clipping(update)
    update["clipped_gradients"] = gradients
    with pytest.raises(AssertionError, match="clipping:"):
        _check_clipping(update)


def test_single_huge_active_tensor_fails_closed_when_bound_is_uninformative():
    parameter = torch.nn.Parameter(torch.zeros(20_000_000))
    parameter.grad = torch.full_like(parameter, 2 / parameter.numel() ** 0.5)
    raw = parameter.grad.clone()
    torch.nn.utils.clip_grad_norm_([parameter], 1.0, foreach=True)
    with pytest.raises(AssertionError, match="cannot distinguish factor-of-two"):
        _check_clipping(
            {
                "gradients": {"p": raw},
                "clipped_gradients": {"p": parameter.grad},
                "clipping_world_size": 1,
                "optimizer": {"max_grad_norm": 1.0},
            }
        )


@pytest.mark.parametrize("world_size", [None, True, 0, -1, 1.5])
def test_clipping_requires_explicit_valid_observed_topology(world_size):
    update = _optimizer_fixture([torch.linspace(-2, 3, 16)])["updates"][0]
    if world_size is None:
        del update["clipping_world_size"]
    else:
        update["clipping_world_size"] = world_size
    with pytest.raises(AssertionError):
        _check_clipping(update)


def test_dense_three_adam_steps_preserve_every_element_and_reject_state_corruption():
    # >20M entries crosses the former flat gamma domain while each native norm
    # follows a finite per-parameter hierarchy. Both signed warm steps are real.
    parameters = [torch.nn.Parameter(torch.full((655_360,), 0.1 + i * 0.001)) for i in range(32)]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, eps=0.0001, weight_decay=0.0001)
    records = {
        "initial_parameters": {str(i): p.detach().clone() for i, p in enumerate(parameters)},
        "updates": [],
    }
    settings = {
        key: optimizer.param_groups[0][key] for key in ("lr", "betas", "eps", "weight_decay")
    }
    settings["max_grad_norm"] = 1.0
    for factor in (1.0, -0.9, 0.7):
        before = {str(i): p.detach().clone() for i, p in enumerate(parameters)}
        gradients = {}
        for i, parameter in enumerate(parameters):
            parameter.grad = (
                torch.linspace(-0.002, 0.003, parameter.numel()) * factor * (1 + i / 32)
            )
            gradients[str(i)] = parameter.grad.clone()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=True)
        clipped = {str(i): p.grad.clone() for i, p in enumerate(parameters)}
        optimizer.step()
        records["updates"].append(
            {
                "gradients": gradients,
                "clipped_gradients": clipped,
                "clipping_world_size": 1,
                "parameters": {str(i): p.detach().clone() for i, p in enumerate(parameters)},
                "deltas": {str(i): p.detach() - before[str(i)] for i, p in enumerate(parameters)},
                "moments": {
                    str(i): {key: value.clone() for key, value in optimizer.state[p].items()}
                    for i, p in enumerate(parameters)
                },
                "optimizer": settings,
            }
        )
    contexts = _check_adam_trajectory(records)
    assert len(contexts) == 3 and all(row["coefficient"] < 1 for row in contexts)
    del contexts
    update = records["updates"][1]
    for corruption in ("warm_moment", "missing_update", "stale_step"):
        if corruption == "warm_moment":
            target = update["moments"]["0"]["exp_avg"][0]
            original = target.clone()
            target.add_(0.001)
        elif corruption == "missing_update":
            target = update["parameters"]["0"][0]
            original = target.clone()
            target.copy_(records["updates"][0]["parameters"]["0"][0])
        else:
            target = update["moments"]["0"]["step"]
            original = target.clone()
            target.fill_(1)
        try:
            with pytest.raises(AssertionError):
                _check_adam_trajectory(records)
        finally:
            target.copy_(original)


def _mixed_mesh_clipping_worker(rank, store_path, result_dir, options):
    from accelerate import Accelerator
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2)
    result = {"rank": rank, "scope": "native CPU DTensor multi-mesh clipping"}
    try:
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("shards", "singleton"))
        world, singleton = mesh["shards"], mesh["singleton"]
        operands = [
            torch.linspace(-0.2, 0.3, 1024),
            torch.linspace(0.1, 0.5, 300),
            torch.linspace(-0.4, 0.1, 500),
            torch.linspace(-0.3, 0.3, 700),
        ]
        layouts = [(world, [Shard(0)]), (world, [Replicate()]), (singleton, [Shard(0)]), None]
        parameters, gradients, geometry = [], {}, []
        for i, (gradient, layout) in enumerate(zip(operands, layouts)):
            if layout is not None:
                parameter = torch.nn.Parameter(
                    distribute_tensor(torch.ones_like(gradient), *layout)
                )
                parameter.grad = distribute_tensor(gradient, *layout)
                geometry.append(
                    {
                        "mesh": parameter.device_mesh.mesh.tolist(),
                        "placements": [str(p) for p in parameter.placements],
                    }
                )
            else:
                parameter = torch.nn.Parameter(torch.ones_like(gradient))
                parameter.grad = gradient.clone()
                geometry.append({"mesh": None, "placements": []})
            parameters.append(parameter)
            gradients[str(i)] = gradient.clone()
        observed_norm = Accelerator._clip_grad_norm_dtensor_aware(None, parameters, 1.0)
        clipped = {
            str(i): (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).clone()
            for i, p in enumerate(parameters)
        }
        update = {
            "gradients": gradients,
            "clipped_gradients": clipped,
            "clipping_world_size": dist.get_world_size(),
            "optimizer": {"max_grad_norm": 1.0},
        }
        norm, lower, upper = _clipping_norm_interval(gradients, update["clipping_world_size"])
        assert lower <= observed_norm <= upper
        coefficient, _ = _check_clipping(update)
        assert coefficient < 1
        for factor in (0.0, 0.5, 2.0):
            broken = {
                **update,
                "clipped_gradients": {name: value * factor for name, value in clipped.items()},
            }
            with pytest.raises(AssertionError, match="clipping:"):
                _check_clipping(broken)
        result.update(
            status="pass",
            world_size=dist.get_world_size(),
            geometry=geometry,
            observed_norm=float(observed_norm),
            reference_norm=float(norm),
            norm_interval=[float(lower), float(upper)],
            negative_factors=[0.0, 0.5, 2.0],
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
def test_mixed_mesh_clipping_uses_actual_shard_replica_and_plain_groups(tmp_path):
    rows = run_gloo_ranks(_mixed_mesh_clipping_worker, tmp_path, 2, {})
    assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows
    print(json.dumps({"dense_optimizer_mixed_mesh": rows}))
