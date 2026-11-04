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

"""Fixed-topology auxiliary state and next-update parity on real FSDP2 ranks."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate import FullyShardedDataParallelPlugin
from accelerate.parallelism_config import ParallelismConfig
from conftest import run_gloo_ranks
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from flow_factory.ema.ema import EMAModuleWrapper
from flow_factory.models.model_bundle import ModelBundle
from flow_factory.models.tensor_parallel import apply_tensor_parallel
from flow_factory.models.variants import ComponentVariantRegistry, ComponentVariantSpec
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.common.runtime_state import TrainerRuntimeState
from flow_factory.trainers.execution import TrainingProgress
from flow_factory.trainers.loader import Accelerator
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(8, 16)
        self.down = nn.Linear(16, 8)

    def forward(self, value):
        return self.down(torch.nn.functional.gelu(self.up(value)))


class _Adapter:
    trainable_component_names = ("transformer",)
    target_module_map = {"transformer": "all"}

    def __init__(self, accelerator):
        self.module = _Block()
        self.accelerator = accelerator
        self._named_parameters = {}
        self.load_calls = 0

    def get_component(self, name):
        return self.module

    def save_checkpoint(self, *, save_directory, model_only, include_training_roles):
        assert not model_only and include_training_roles
        self.accelerator.save_state(save_directory)

    def _load_training_state(self, path):
        self.load_calls += 1
        self.accelerator.load_state(path)


class _RoleState:
    """Use the existing coordinator and snapshot validators as one runtime child."""

    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        return {
            "coordinator": self.trainer.role_optimization.state_dict(),
            "snapshots": self.trainer.adapter.component_variant_registry.snapshot_state_dict(),
        }

    def validate_state_dict(self, state):
        assert set(state) == {"coordinator", "snapshots"}
        self.trainer._validate_multirole_coordinator_state(state["coordinator"])
        self.trainer.adapter.component_variant_registry.validate_snapshot_state_dict(
            state["snapshots"]
        )

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.trainer.role_optimization.load_state_dict(state["coordinator"])
        self.trainer.adapter.component_variant_registry.load_snapshot_state_dict(state["snapshots"])


def _local(tensor):
    return (tensor.to_local() if isinstance(tensor, DTensor) else tensor).detach().cpu().clone()


def _parameters_and_moments(parameters, optimizer):
    return [
        {
            "parameter": _local(parameter),
            **{
                key: _local(value) if isinstance(value, torch.Tensor) else value
                for key, value in optimizer.state[parameter].items()
            },
        }
        for parameter in parameters
    ]


def _assert_states_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            _assert_states_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_states_equal(a, b)
    else:
        assert left == right


def _worker(rank, store_path, result_dir, options):
    world_size, tp = options["world_size"], options["tp"]
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size), ACCELERATE_USE_FSDP="true"
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        store=dist.FileStore(store_path, world_size),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    result = {
        "rank": rank,
        "world_size": world_size,
        "dp": 2,
        "tp": tp,
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("torch", "accelerate")
        },
    }
    try:
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="NO_WRAP",
            reshard_after_forward=True,
            cpu_ram_efficient_loading=False,
        )
        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=plugin,
            parallelism_config=ParallelismConfig(dp_replicate_size=1, dp_shard_size=2, tp_size=tp),
        )
        result["runtime"] = {
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
            "device": str(accelerator.device),
            "distributed_type": accelerator.distributed_type.name,
            "fsdp_version": accelerator.state.fsdp_plugin.fsdp_version,
            "process_group_backend": dist.get_backend(),
        }
        torch.manual_seed(73)
        adapter = _Adapter(accelerator)
        registry = ComponentVariantRegistry(adapter)
        for name, route in (("generator", "transformer"), ("surrogate", "transformer_surrogate")):
            registry.declare(
                ComponentVariantSpec(
                    name=name, storage_mode="full", component_routes={"transformer": route}
                )
            )
        registry.materialize(("generator", "surrogate"))
        adapter.component_variant_registry = registry
        members = registry.bundle_members()
        if tp > 1:
            for module in members.values():
                apply_tensor_parallel(
                    module,
                    accelerator.torch_device_mesh["tp"],
                    {"up": ColwiseParallel(), "down": RowwiseParallel()},
                )
        registry.rebind_parameters(members)
        bundle = ModelBundle(members)
        groups = [
            {"params": list(registry.parameters(name)), "role_name": name}
            for name in registry.variant_names
        ]
        optimizer = torch.optim.AdamW(groups, lr=1e-3, eps=1e-4, weight_decay=0.01)
        bundle, optimizer = accelerator.prepare(bundle, optimizer)
        registry.rebind_parameters(bundle.members)
        registry.add_snapshot("surrogate", "slow_surrogate")
        roles = {
            name: OptimizationRole(
                config=RoleOptimizerConfig(
                    role_name=name,
                    learning_rate=1e-3,
                    adam_betas=(0.9, 0.999),
                    adam_weight_decay=0.01,
                    adam_epsilon=1e-4,
                    max_grad_norm=1.0,
                ),
                parameters=tuple(optimizer.param_groups[index]["params"]),
                optimizer_group_ids=(index,),
            )
            for index, name in enumerate(registry.variant_names)
        }
        trainer = object.__new__(BaseTrainer)
        trainer.accelerator, trainer.adapter = accelerator, adapter
        trainer.model_bundle, trainer.optimizer = bundle, optimizer
        trainer.role_optimization = RoleOptimizationCoordinator(
            accelerator, bundle, optimizer, roles
        )
        generator_params = tuple(registry.parameters("generator"))
        ema = EMAModuleWrapper(
            generator_params, decay=0.5, update_step_interval=2, device=torch.device("cpu")
        )
        reference = EMAModuleWrapper(
            generator_params, decay=0.0, update_step_interval=0, device=torch.device("cpu")
        )
        old = EMAModuleWrapper(
            generator_params, decay=0.0, update_step_interval=1, device=accelerator.device
        )
        children = {
            "ema": ema,
            "reference": reference,
            "old_policy": old,
            "role_state": _RoleState(trainer),
        }
        identity = {
            **TrainerRuntimeState().identity,
            "trainer": "parameter_state_fixture",
            "world_size": world_size,
            "distributed_type": "FSDP",
            "gradient_scaler": "none",
        }
        trainer.runtime_state = TrainerRuntimeState(child_names=tuple(children), identity=identity)
        for name, child in children.items():
            trainer.runtime_state.attach_child(name, child)
        trainer._trainer_runtime_children = children
        trainer._runtime_children_attached = True
        reference_initial = [_local(p) for p in reference.ema_parameters]
        initial_ema = [_local(p) for p in ema.ema_parameters]
        isolation_checks = []

        def cycle(step):
            inputs = torch.randn(
                2, 3, 8, generator=torch.Generator().manual_seed(100 + step * 2 + rank // tp)
            ).to(accelerator.device)
            for role in registry.variant_names:
                inactive = "surrogate" if role == "generator" else "generator"
                inactive_parameters = tuple(registry.parameters(inactive))
                before = _parameters_and_moments(inactive_parameters, optimizer)
                if step:
                    assert all(
                        "exp_avg" in value and "exp_avg_sq" in value and value["step"].item() > 0
                        for value in before
                    )
                route = registry.resolve_route(role, "transformer")
                with torch.no_grad():
                    if role == "generator":
                        with ema.use_ema_parameters(generator_params):
                            target = bundle(route, inputs).detach() * 0.3
                        with reference.use_ema_parameters(generator_params):
                            target = target + bundle(route, inputs).detach() * 0.2
                        with old.use_ema_parameters(generator_params):
                            target = target + bundle(route, inputs).detach() * 0.1
                    else:
                        live = [_local(p) for p in registry.parameters(role)]
                        with registry.use_snapshot("slow_surrogate"):
                            target = bundle(route, inputs).detach() * 0.5
                        _assert_states_equal(live, [_local(p) for p in registry.parameters(role)])
                with trainer.role_optimization.phase(role):
                    with trainer.role_optimization.microbatch():
                        trainer.role_optimization.backward(
                            (bundle(route, inputs) - target).square().mean()
                        )
                        assert all(
                            parameter.grad is not None
                            and torch.count_nonzero(_local(parameter.grad)).item() > 0
                            for parameter in roles[role].parameters
                        )
                        trainer.role_optimization.finish_microbatch()
                _assert_states_equal(
                    before, _parameters_and_moments(inactive_parameters, optimizer)
                )
                isolation_checks.append(
                    {"step": step, "active_role": role, "populated_inactive_state": bool(step)}
                )
            ema.step(generator_params, step)
            reference.step(generator_params, step)
            old.step(generator_params, step)
            if (step + 1) % 2 == 0:
                registry.update_snapshot("slow_surrogate", decay=0.5)
            trainer.runtime_state.progress = TrainingProgress(
                optimizer_step=step + 1, rollout_iteration=step + 1
            )

        for step in range(3):
            cycle(step)
        assert ema.num_updates == 1 and old.num_updates == 3
        assert registry.snapshot_state_dict()["update_counts"] == {"slow_surrogate": 1}
        assert any(not torch.equal(a, _local(b)) for a, b in zip(initial_ema, ema.ema_parameters))
        _assert_states_equal(reference_initial, [_local(p) for p in reference.ema_parameters])
        checkpoint = str(Path(result_dir, "checkpoint"))
        trainer._save_exact_training_state(checkpoint)
        cycle(3)
        expected = _parameters_and_moments(bundle.parameters(), optimizer)
        expected_aux = {name: child.state_dict() for name, child in children.items()}
        expected_aux = _local_tree(expected_aux)
        with torch.no_grad():
            for parameter in bundle.parameters():
                parameter.add_(0.25)
            for wrapper in (ema, reference, old):
                for parameter in wrapper.ema_parameters:
                    parameter.add_(0.375)
            registry.update_snapshot("slow_surrogate", decay=0.0)
        assert any(
            not torch.equal(a["parameter"], b["parameter"])
            for a, b in zip(expected, _parameters_and_moments(bundle.parameters(), optimizer))
        )
        corrupted = Path(result_dir, "bad_layout")
        if rank == 0:
            shutil.copytree(checkpoint, corrupted)
            _corrupt_layout_with_valid_hashes(corrupted)
        dist.barrier()
        before_bad_load = _parameters_and_moments(bundle.parameters(), optimizer)
        aux_before_bad_load = _local_tree(
            {name: child.state_dict() for name, child in children.items()}
        )
        trainer.runtime_state = TrainerRuntimeState(child_names=tuple(children), identity=identity)
        trainer._runtime_children_attached = False
        trainer.model_args = SimpleNamespace(resume_path=str(corrupted), resume_type="state")
        try:
            trainer._finalize_adapter_runtime()
        except (ValueError, RuntimeError) as error:
            assert "layout mismatch" in str(error)
        else:
            raise AssertionError("bad DTensor layout reached backend restoration")
        assert adapter.load_calls == 0
        _assert_states_equal(
            before_bad_load, _parameters_and_moments(bundle.parameters(), optimizer)
        )
        _assert_states_equal(
            aux_before_bad_load,
            _local_tree({name: child.state_dict() for name, child in children.items()}),
        )
        result["layout_preflight_before_backend_mutation"] = True
        trainer.runtime_state = TrainerRuntimeState(child_names=tuple(children), identity=identity)
        trainer._runtime_children_attached = False
        trainer.model_args = SimpleNamespace(resume_path=checkpoint, resume_type="state")
        trainer._finalize_adapter_runtime()
        assert adapter.load_calls == 1 and trainer.step == 3
        assert ema.num_updates == 1 and old.num_updates == 3
        cycle(3)
        _assert_states_equal(expected, _parameters_and_moments(bundle.parameters(), optimizer))
        _assert_states_equal(
            expected_aux,
            _local_tree({name: child.state_dict() for name, child in children.items()}),
        )
        assert ema.num_updates == 2 and old.num_updates == 4
        assert all(
            not p.requires_grad and p.grad is None
            for p in ema.ema_parameters + reference.ema_parameters + old.ema_parameters
        )
        assert all(
            p.device.type == "cpu" and p.device_mesh.device_type == "cuda"
            for p in ema.ema_parameters
        )
        plain = EMAModuleWrapper([torch.nn.Parameter(torch.ones(3, device=accelerator.device))])
        plain_state = plain.state_dict()
        plain_state["ema_parameters"] = [torch.zeros(3)]
        plain.load_state_dict(plain_state)
        assert plain.ema_parameters[0].device == accelerator.device
        assert plain.ema_parameters[0].sum().item() == 0
        result["replicated_default_device_preserved"] = True
        result.update(
            next_update_exact=True,
            optimizer_moments_exact=True,
            nonzero_gradients=True,
            ema_updates=ema.num_updates,
            old_policy_updates=old.num_updates,
            reference_updates=reference.num_updates,
            snapshot_updates=registry.snapshot_state_dict()["update_counts"],
            cpu_offload_preserved=True,
            inactive_role_checks=isolation_checks,
            backend_load_calls=adapter.load_calls,
            placements=sorted({str(p.placements) for p in generator_params}),
            runtime_manifest_version=json.loads(
                Path(checkpoint, "flow_factory_trainer_runtime.json").read_text()
            )["version"],
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def _corrupt_layout_with_valid_hashes(directory):
    manifest_path = directory / "flow_factory_trainer_runtime.json"
    manifest = json.loads(manifest_path.read_text())
    rank_path = directory / manifest["rank_files"][0]["path"]
    payload = json.loads(rank_path.read_text())

    def corrupt(node):
        if node["type"] == "dtensor":
            node["layout"]["mesh_dim_names"] = ["wrong_dimension"]
            return True
        children = (
            [child for _, child in node["items"]]
            if node["type"] == "mapping"
            else node.get("items", [])
        )
        return any(corrupt(child) for child in children)

    assert corrupt(payload["state"])
    rank_path.write_text(json.dumps(payload))
    descriptor = {
        "path": rank_path.name,
        "size": rank_path.stat().st_size,
        "sha256": hashlib.sha256(rank_path.read_bytes()).hexdigest(),
    }
    manifest["rank_files"][0] = descriptor
    manifest["state_files"] = [
        descriptor if entry["path"] == rank_path.name else entry
        for entry in manifest["state_files"]
    ]
    manifest_path.write_text(json.dumps(manifest))


def _local_tree(value):
    if isinstance(value, torch.Tensor):
        return _local(value)
    if isinstance(value, dict):
        return {key: _local_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_local_tree(item) for item in value]
    return value


@pytest.mark.parametrize("tp", [1, 2], ids=["fsdp2", "fsdp2_tp"])
def test_parameter_state_resume_matches_the_next_update(tmp_path, tp):
    if torch.cuda.device_count() < 2 * tp:
        pytest.skip(f"requires {2 * tp} CUDA devices")
    results = run_gloo_ranks(
        _worker, tmp_path, 2 * tp, {"world_size": 2 * tp, "tp": tp}, timeout=240
    )
    assert all("error" not in result for result in results), results
    assert all(
        result["next_update_exact"] and result["optimizer_moments_exact"] for result in results
    )
    assert all(
        result["ema_updates"] == 2 and result["reference_updates"] == 0 for result in results
    )
