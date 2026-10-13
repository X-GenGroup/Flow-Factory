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

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Iterator, Mapping, Tuple

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs

from flow_factory.hparams import (
    AdamWOptimizerArguments,
    DMD2TrainingArguments,
    MultiOptimizerArguments,
    TDMR1TrainingArguments,
    TDMTrainingArguments,
)
from flow_factory.samples import LatentState
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.distillation.dmd2 import DMD2Trainer
from flow_factory.trainers.distillation.tdm import TDMTrainer
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
)


@contextmanager
def _variant_context(role_name: str) -> Iterator[None]:
    del role_name
    yield


class TinyRoleLayer(torch.nn.Module):
    """Apply one scalar trainable role weight."""

    def __init__(self, initial_value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([initial_value]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale the input by this role's parameter."""
        return value * self.weight


class TinyMultiRoleBundle(torch.nn.Module):
    """Expose role-local branches through one prepared model root."""

    _no_split_modules = ["TinyRoleLayer"]

    def __init__(self, storage_mode: str) -> None:
        super().__init__()
        self.storage_mode = storage_mode
        if storage_mode == "lora":
            self.base_weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
            self.roles = torch.nn.ModuleDict(
                {
                    "generator": TinyRoleLayer(0.1),
                    "fake": TinyRoleLayer(-0.2),
                    "surrogate": TinyRoleLayer(0.3),
                }
            )
        elif storage_mode == "full":
            self.register_parameter("base_weight", None)
            self.roles = torch.nn.ModuleDict(
                {
                    "generator": TinyRoleLayer(1.1),
                    "fake": TinyRoleLayer(0.8),
                    "surrogate": TinyRoleLayer(1.3),
                }
            )
        else:
            raise ValueError(
                "expected storage_mode in ('lora', 'full'), " f"received {storage_mode!r}"
            )

    def forward(self, value: torch.Tensor, role_name: str) -> torch.Tensor:
        """Route one forward through the selected trainable role."""
        if role_name not in self.roles:
            raise KeyError(f"expected role_name in {tuple(self.roles)!r}, received {role_name!r}")
        role_output = self.roles[role_name](value)
        if self.base_weight is None:
            return role_output
        return value * self.base_weight + role_output


class RoleCounters:
    """Checkpoint the closed role-phase counters used by the harness."""

    def __init__(self, role_names: Tuple[str, ...]) -> None:
        self.steps = {role_name: 0 for role_name in role_names}

    def state_dict(self) -> Dict[str, Dict[str, int]]:
        """Return role counters in Accelerate's custom-state format."""
        return {"steps": dict(self.steps)}

    def load_state_dict(self, state: Mapping[str, Mapping[str, int]]) -> None:
        """Restore role counters with exact role ownership."""
        received = tuple(state["steps"])
        expected = tuple(self.steps)
        if received != expected:
            raise ValueError(
                f"expected checkpoint role counters {expected!r}, received {received!r}"
            )
        self.steps = dict(state["steps"])


def _algorithm_roles(algorithm: str) -> Tuple[str, ...]:
    if algorithm in {"dmd2", "tdm"}:
        return ("generator", "fake")
    if algorithm == "tdm-r1":
        return ("generator", "fake", "surrogate")
    raise ValueError("expected algorithm in ('dmd2', 'tdm', 'tdm-r1'), " f"received {algorithm!r}")


def _fake_cadence_config(fake_update_frequency: int) -> SimpleNamespace:
    """Return a trainer config whose fake optimizer entry sets the TTUR cadence."""
    return SimpleNamespace(
        optimizer_args=MultiOptimizerArguments(
            optimizer_configs=[
                AdamWOptimizerArguments(name="fake", update_frequency=fake_update_frequency)
            ]
        )
    )


def _phase_order(algorithm: str) -> Tuple[str, ...]:
    raise ValueError(
        "expected production optimize() paths for dmd2/tdm/tdm-r1; "
        f"received leftover harness algorithm={algorithm!r}"
    )


def _optimizer(
    model: TinyMultiRoleBundle,
    role_names: Tuple[str, ...],
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {
                "params": tuple(model.roles[role_name].parameters()),
                "role_name": role_name,
                "lr": 0.03,
            }
            for role_name in role_names
        ]
    )


def _roles_from_optimizer_groups(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, OptimizationRole]:
    roles = {}
    for group_id, group in enumerate(optimizer.param_groups):
        role_name = group["role_name"]
        config = RoleOptimizerConfig(
            role_name=role_name,
            learning_rate=group["lr"],
            adam_betas=group["betas"],
            adam_weight_decay=group["weight_decay"],
            adam_epsilon=group["eps"],
            max_grad_norm=1.0,
        )
        roles[role_name] = OptimizationRole(
            config=config,
            parameters=tuple(group["params"]),
            optimizer_group_ids=(group_id,),
        )
    return roles


def _run_sequential_boundary(
    coordinator: RoleOptimizationCoordinator,
    model: torch.nn.Module,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    observations: Dict[str, Any] = {}
    stepped_roles = []
    role_trace = []
    value = torch.tensor([2.0], device=accelerator.device)
    target = torch.tensor([0.25], device=accelerator.device)
    for role_name in ("fake", "surrogate", "generator"):
        role_trace.append(role_name)
        parameter_values = {
            name: role.parameters[0].detach().clone() for name, role in coordinator.roles.items()
        }
        observations[f"{role_name}_sees_fake"] = model(value, "fake").detach().clone()
        with coordinator.phase(role_name):
            with coordinator.microbatch():
                loss = (model(value, role_name) - target).square().mean()
                coordinator.backward(loss)
                stepped = coordinator.finish_microbatch()
        if stepped:
            stepped_roles.append(role_name)
            assert not torch.equal(
                coordinator.roles[role_name].parameters[0].detach(),
                parameter_values[role_name],
            )
            assert all(
                torch.equal(
                    coordinator.roles[other_name].parameters[0].detach(),
                    parameter_values[other_name],
                )
                for other_name in coordinator.roles
                if other_name != role_name
            )
    observations["role_trace"] = tuple(role_trace)
    observations["stepped_roles"] = tuple(stepped_roles)
    return observations


def _run_production_tdm_r1_optimize(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    roles: Dict[str, OptimizationRole],
    coordinator: RoleOptimizationCoordinator,
    *,
    gradient_accumulation_steps: int,
) -> Dict[str, Any]:
    """Execute the real TDM-R1 optimize method on a tiny prepared bundle."""

    class EMARegistry:
        def __init__(self) -> None:
            self.live = {
                "old_surrogate": roles["surrogate"].parameters[0],
                "generator_ema": roles["generator"].parameters[0],
            }
            self.snapshots = {
                name: parameter.detach().clone() for name, parameter in self.live.items()
            }

        def update_parameter_ema(self, name: str, decay: float) -> None:
            self.snapshots[name].lerp_(self.live[name].detach(), 1 - decay)

        def has_snapshot(self, name: str) -> bool:
            return name in self.snapshots

        def add_snapshot(self, variant_name: str, name: str) -> None:
            self.live[name] = roles[variant_name].parameters[0]
            self.snapshots[name] = self.live[name].detach().clone()

        def update_snapshot(self, name: str, decay: float) -> None:
            self.update_parameter_ema(name, decay)

        @contextmanager
        def use_snapshot(self, name: str) -> Iterator[None]:
            live = self.live[name]
            restored = live.detach().clone()
            live.data.copy_(self.snapshots[name])
            try:
                yield
            finally:
                live.data.copy_(restored)

    class ProductionPathTDMR1Trainer(TDMR1Trainer):
        optimize_calls = 0

        def optimize(self, samples: list[Any]) -> None:
            self.optimize_calls += 1
            super().optimize(samples)

    registry = EMARegistry()
    initial_parameters = {name: role.parameters[0].detach().clone() for name, role in roles.items()}
    initial_old = registry.snapshots["old_surrogate"].clone()
    initial_generator = registry.snapshots["generator_ema"].clone()
    trainer = object.__new__(ProductionPathTDMR1Trainer)
    trainer.accelerator = accelerator
    trainer.model_bundle = model
    trainer.optimizer = optimizer
    trainer.optimization_roles = roles
    trainer.role_optimization = coordinator
    # Delegated to the real registry so the slow surrogate copy TDM-R1 keeps is
    # exercised by the distributed wiring rather than stubbed away.
    trainer.adapter = SimpleNamespace(
        component_variant_registry=registry,
        train=lambda: None,
        use_component_variant=_variant_context,
        has_variant_snapshot=registry.has_snapshot,
        declare_variant_snapshot=registry.add_snapshot,
        update_variant_snapshot=registry.update_snapshot,
        use_variant_snapshot=registry.use_snapshot,
    )
    algorithm_args = TDMR1TrainingArguments()
    trainer.training_args = SimpleNamespace(
        gradient_accumulation_steps=gradient_accumulation_steps,
        get_num_train_timesteps=lambda config: 1,
        required_trainable_roles=algorithm_args.required_trainable_roles,
        role_update_plan=algorithm_args.role_update_plan,
        per_device_batch_size=2,
        surrogate_slow_decay_min=0.001,
        surrogate_slow_decay_max=0.3,
    )
    trainer.config = _fake_cadence_config(1)
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    units = [SimpleNamespace(boundary_index=1)]
    samples = [SimpleNamespace(unique_id=7), SimpleNamespace(unique_id=7)]
    microbatches = [samples for _ in range(gradient_accumulation_steps)]
    trace: list[str] = []
    noise_checksums: list[torch.Tensor] = []
    value = torch.tensor([2.0], device=accelerator.device)
    trainer._build_boundary_units = MethodType(lambda self, received: units, trainer)

    def role_loss(role_name: str) -> torch.Tensor:
        trace.append(role_name)
        return model(value, role_name).square().mean()

    trainer._fake_boundary_loss = MethodType(
        lambda self, unit: role_loss("fake"),
        trainer,
    )
    trainer._surrogate_boundary_loss = MethodType(
        lambda self, unit: role_loss("surrogate"),
        trainer,
    )
    trainer._generator_boundary_loss = MethodType(
        lambda self, unit: role_loss("generator"),
        trainer,
    )

    trainer.optimize(microbatches)

    return {
        "trace": tuple(trace),
        "optimize_calls": trainer.optimize_calls,
        "role_deltas": {
            name: role.parameters[0].detach() - initial_parameters[name]
            for name, role in roles.items()
        },
    }


def _run_production_tdm_optimize(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    roles: Dict[str, OptimizationRole],
    coordinator: RoleOptimizationCoordinator,
    *,
    ratio: int,
    iterations: int,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """Execute the real TDM optimize method on a tiny prepared bundle."""

    class ProductionPathTDMTrainer(TDMTrainer):
        optimize_calls = 0

        def optimize(self, samples: list[Any]) -> None:
            self.optimize_calls += 1
            super().optimize(samples)

    initial_parameters = {name: role.parameters[0].detach().clone() for name, role in roles.items()}
    trainer = object.__new__(ProductionPathTDMTrainer)
    trainer.accelerator = accelerator
    trainer.model_bundle = model
    trainer.optimizer = optimizer
    trainer.optimization_roles = roles
    trainer.role_optimization = coordinator
    trainer.adapter = SimpleNamespace(
        train=lambda: None,
        use_component_variant=_variant_context,
    )
    algorithm_args = TDMTrainingArguments()
    trainer.training_args = SimpleNamespace(
        per_device_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        get_num_train_timesteps=lambda config: 1,
        required_trainable_roles=algorithm_args.required_trainable_roles,
        role_update_plan=algorithm_args.role_update_plan,
        num_inference_steps=1,
        num_inner_epochs=1,
    )
    trainer.config = _fake_cadence_config(ratio)
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trace: list[str] = []
    value = torch.tensor([2.0], device=accelerator.device)
    target = torch.tensor([0.25], device=accelerator.device)

    def role_loss(role_name: str) -> torch.Tensor:
        trace.append(role_name)
        return (model(value, role_name) - target).square().mean()

    trainer._build_boundary_units = MethodType(
        lambda self, received: [SimpleNamespace(boundary_index=1)],
        trainer,
    )
    trainer._fake_boundary_loss = MethodType(lambda self, unit: role_loss("fake"), trainer)
    trainer._generator_boundary_loss = MethodType(
        lambda self, unit: role_loss("generator"),
        trainer,
    )

    for _ in range(iterations):
        trainer.optimize([[SimpleNamespace()] for _ in range(gradient_accumulation_steps)])

    return {
        "trace": tuple(trace),
        "optimize_calls": trainer.optimize_calls,
        "trainer_step": trainer.step,
        "role_deltas": {
            name: role.parameters[0].detach() - initial_parameters[name]
            for name, role in roles.items()
        },
    }


def _run_production_dmd2_optimize(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    roles: Dict[str, OptimizationRole],
    coordinator: RoleOptimizationCoordinator,
    *,
    ratio: int,
    iterations: int,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """Execute the real DMD2 optimize method on a tiny prepared bundle."""

    class ProductionPathDMD2Trainer(DMD2Trainer):
        optimize_calls = 0

        def optimize(self, samples: list[Any]) -> None:
            self.optimize_calls += 1
            super().optimize(samples)

    initial_parameters = {name: role.parameters[0].detach().clone() for name, role in roles.items()}
    trainer = object.__new__(ProductionPathDMD2Trainer)
    trainer.accelerator = accelerator
    trainer.model_bundle = model
    trainer.optimizer = optimizer
    trainer.optimization_roles = roles
    trainer.role_optimization = coordinator
    trainer.adapter = SimpleNamespace(
        train=lambda: None,
        use_component_variant=_variant_context,
    )
    trainer.training_args = SimpleNamespace(
        num_inference_steps=1,
        num_inner_epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # The public step is paced by the primary role, which is the first declared.
        required_trainable_roles=("generator", "fake"),
        role_update_plan=DMD2TrainingArguments().role_update_plan,
    )
    trainer.config = _fake_cadence_config(ratio)
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trace: list[str] = []
    value = torch.tensor([2.0], device=accelerator.device)
    target = torch.tensor([0.25], device=accelerator.device)

    def role_loss(role_name: str) -> torch.Tensor:
        trace.append(role_name)
        return (model(value, role_name) - target).square().mean()

    trainer._stack_replay_unit = MethodType(lambda self, unit: unit, trainer)
    trainer._fake_replay_loss = MethodType(
        lambda self, batch: role_loss("fake"),
        trainer,
    )
    trainer._generator_replay_loss = MethodType(
        lambda self, batch: role_loss("generator"),
        trainer,
    )

    for _ in range(iterations):
        trainer.optimize([[SimpleNamespace()] for _ in range(gradient_accumulation_steps)])

    return {
        "trace": tuple(trace),
        "optimize_calls": trainer.optimize_calls,
        "trainer_step": trainer.step,
        "role_deltas": {
            name: role.parameters[0].detach() - initial_parameters[name]
            for name, role in roles.items()
        },
    }


def _run_role_phase(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    counters: RoleCounters,
    role_name: str,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    value = torch.tensor([2.0], device=accelerator.device)
    target = torch.tensor([0.25], device=accelerator.device)
    loss = (model(value, role_name) - target).square().mean()
    accelerator.backward(loss)
    for group in optimizer.param_groups:
        group_role = group["role_name"]
        gradients = tuple(parameter.grad for parameter in group["params"])
        if group_role == role_name:
            assert any(gradient is not None for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    counters.steps[role_name] += 1


def _optimizer_steps(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Tuple[float, ...]]:
    result = {}
    for group in optimizer.param_groups:
        role_name = group["role_name"]
        result[role_name] = tuple(
            float(optimizer.state[parameter]["step"]) for parameter in group["params"]
        )
    return result


def _shared_checkpoint_directory(
    accelerator: Accelerator,
    backend: str,
    storage_mode: str,
    algorithm: str,
) -> Path:
    path_values = [None]
    if accelerator.is_main_process:
        path_values[0] = tempfile.mkdtemp(
            prefix=f"flow-factory-{backend}-{storage_mode}-{algorithm}-"
        )
    dist.broadcast_object_list(path_values, src=0)
    if not isinstance(path_values[0], str):
        raise TypeError(
            "expected rank-zero checkpoint path as str, "
            f"received {type(path_values[0]).__name__}: {path_values[0]!r}"
        )
    return Path(path_values[0])


def _validate_backend_variant(backend: str, accelerator: Any) -> None:
    """Validate the exact plugin variant selected by one backend label."""
    expected_distributed_types = {
        "ddp": "MULTI_GPU",
        "fsdp2": "FSDP",
        "zero1": "DEEPSPEED",
        "zero2": "DEEPSPEED",
    }
    if backend not in expected_distributed_types:
        raise ValueError(
            f"expected backend label in {tuple(expected_distributed_types)!r}, received {backend!r}"
        )

    distributed_type = accelerator.distributed_type.name
    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    fsdp_version = getattr(fsdp_plugin, "fsdp_version", None)
    zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
    matches = distributed_type == expected_distributed_types[backend]
    if backend == "fsdp2":
        matches = matches and fsdp_plugin is not None and fsdp_version == 2
    elif backend == "zero1":
        matches = matches and deepspeed_plugin is not None and zero_stage == 1
    elif backend == "zero2":
        matches = matches and deepspeed_plugin is not None and zero_stage == 2

    if not matches:
        fsdp_plugin_name = None if fsdp_plugin is None else type(fsdp_plugin).__name__
        deepspeed_plugin_name = (
            None if deepspeed_plugin is None else type(deepspeed_plugin).__name__
        )
        raise AssertionError(
            f"expected backend label {backend!r}; received "
            f"distributed_type={distributed_type!r}, "
            f"fsdp_plugin={fsdp_plugin_name!r}, fsdp_version={fsdp_version!r}, "
            f"deepspeed_plugin={deepspeed_plugin_name!r}, zero_stage={zero_stage!r}"
        )


def _fake_accelerator(
    distributed_type: str,
    *,
    fsdp_version: int | None = None,
    zero_stage: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        distributed_type=SimpleNamespace(name=distributed_type),
        state=SimpleNamespace(
            fsdp_plugin=(
                None if fsdp_version is None else SimpleNamespace(fsdp_version=fsdp_version)
            ),
            deepspeed_plugin=(
                None if zero_stage is None else SimpleNamespace(zero_stage=zero_stage)
            ),
        ),
        # The multi-role check compares the tracked root through `unwrap_model`, because
        # accelerate registers the module before DDP wraps it. These fakes register the
        # module itself, so unwrapping is the identity here.
        unwrap_model=lambda model: model,
    )


@pytest.mark.parametrize(
    "backend,accelerator",
    [
        ("ddp", _fake_accelerator("MULTI_GPU")),
        ("fsdp2", _fake_accelerator("FSDP", fsdp_version=2)),
        ("zero1", _fake_accelerator("DEEPSPEED", zero_stage=1)),
        ("zero2", _fake_accelerator("DEEPSPEED", zero_stage=2)),
    ],
)
def test_backend_variant_accepts_exact_plugin_contract(
    backend: str,
    accelerator: SimpleNamespace,
) -> None:
    _validate_backend_variant(backend, accelerator)


@pytest.mark.parametrize(
    "backend,accelerator,match",
    [
        (
            "fsdp2",
            _fake_accelerator("FSDP", fsdp_version=1),
            "expected backend label 'fsdp2'.*distributed_type='FSDP'.*fsdp_version=1",
        ),
        (
            "fsdp2",
            _fake_accelerator("MULTI_GPU"),
            "expected backend label 'fsdp2'.*distributed_type='MULTI_GPU'.*fsdp_plugin=None",
        ),
        (
            "zero1",
            _fake_accelerator("DEEPSPEED", zero_stage=2),
            "expected backend label 'zero1'.*distributed_type='DEEPSPEED'.*zero_stage=2",
        ),
        (
            "zero2",
            _fake_accelerator("DEEPSPEED", zero_stage=1),
            "expected backend label 'zero2'.*distributed_type='DEEPSPEED'.*zero_stage=1",
        ),
    ],
)
def test_backend_variant_rejects_wrong_plugin_version_or_stage(
    backend: str,
    accelerator: SimpleNamespace,
    match: str,
) -> None:
    with pytest.raises(AssertionError, match=match):
        _validate_backend_variant(backend, accelerator)


def test_deepspeed_tdm_r1_zero2_is_allowed(algorithm: str) -> None:
    if algorithm != "tdm-r1":
        pytest.skip("DeepSpeed fail-fast applies only to TDM-R1")
    model = object()
    optimizer = SimpleNamespace(
        param_groups=[
            {"role_name": "generator"},
            {"role_name": "fake"},
            {"role_name": "surrogate"},
        ]
    )
    accelerator = _fake_accelerator("DEEPSPEED", zero_stage=2)
    accelerator._models = [model]
    accelerator._optimizers = [optimizer]
    trainer = SimpleNamespace(
        training_args=SimpleNamespace(
            trainer_type="tdm-r1",
            required_trainable_roles=("generator", "fake", "surrogate"),
        ),
        accelerator=accelerator,
        model_bundle=model,
        optimizer=optimizer,
        _unprepared_optimizer_group_roles=("generator", "fake", "surrogate"),
    )
    trainer._required_trainable_roles = MethodType(
        BaseTrainer._required_trainable_roles,
        trainer,
    )

    BaseTrainer._validate_multirole_backend(trainer)


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2])
def test_backend_role_phases(
    backend: str,
    storage_mode: str,
    algorithm: str,
    gradient_accumulation_steps: int,
) -> None:
    """Exercise one-root multi-role phases and backend state restoration."""
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    _validate_backend_variant(backend, accelerator)

    role_names = _algorithm_roles(algorithm)
    model = TinyMultiRoleBundle(storage_mode)
    optimizer = _optimizer(model, role_names)
    unprepared_group_roles = tuple(group["role_name"] for group in optimizer.param_groups)
    model, optimizer = accelerator.prepare(model, optimizer)
    counters = RoleCounters(role_names)

    assert len(accelerator._models) == 1
    # Accelerate registers a DDP model before wrapping it, so compare the underlying modules.
    assert accelerator.unwrap_model(accelerator._models[0]) is accelerator.unwrap_model(model)
    assert len(accelerator._optimizers) == 1
    assert accelerator._optimizers[0] is optimizer
    assert tuple(group["role_name"] for group in optimizer.param_groups) == unprepared_group_roles
    if algorithm == "tdm-r1" and backend in {"zero1", "zero2"}:
        trainer = SimpleNamespace(
            training_args=SimpleNamespace(
                trainer_type="tdm-r1",
                required_trainable_roles=role_names,
            ),
            accelerator=accelerator,
            model_bundle=model,
            optimizer=optimizer,
            _unprepared_optimizer_group_roles=unprepared_group_roles,
        )
        trainer._required_trainable_roles = MethodType(
            BaseTrainer._required_trainable_roles,
            trainer,
        )
        BaseTrainer._validate_multirole_backend(trainer)
    if backend == "fsdp2":
        prepared_parameter_ids = {id(parameter) for parameter in model.parameters()}
        assert all(
            id(parameter) in prepared_parameter_ids
            for group in optimizer.param_groups
            for parameter in group["params"]
        )

    if algorithm == "tdm-r1":
        roles = _roles_from_optimizer_groups(optimizer)
        coordinator = RoleOptimizationCoordinator(
            accelerator,
            model,
            optimizer,
            roles,
        )
        accelerator.register_for_checkpointing(coordinator)
        evidence = _run_production_tdm_r1_optimize(
            accelerator,
            model,
            optimizer,
            roles,
            coordinator,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        expected_trace = (
            ("fake",) * gradient_accumulation_steps
            + ("surrogate",) * gradient_accumulation_steps
            + ("generator",) * gradient_accumulation_steps
        )
        assert evidence["trace"] == expected_trace
        assert evidence["optimize_calls"] == 1
        assert all(
            torch.count_nonzero(delta).item() > 0 for delta in evidence["role_deltas"].values()
        )
        assert {name: role.step for name, role in roles.items()} == {
            "generator": 1,
            "fake": 1,
            "surrogate": 1,
        }
        counters.steps = {name: role.step for name, role in roles.items()}
    elif algorithm == "dmd2":
        roles = _roles_from_optimizer_groups(optimizer)
        coordinator = RoleOptimizationCoordinator(
            accelerator,
            model,
            optimizer,
            roles,
        )
        accelerator.register_for_checkpointing(coordinator)
        evidence = _run_production_dmd2_optimize(
            accelerator,
            model,
            optimizer,
            roles,
            coordinator,
            ratio=3,
            iterations=6,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        fake_trace = ("fake",) * (3 * gradient_accumulation_steps)
        generator_trace = ("generator",) * gradient_accumulation_steps
        assert evidence["trace"] == (fake_trace + generator_trace) * 6
        assert evidence["optimize_calls"] == 6
        assert evidence["trainer_step"] == 6
        assert {name: role.step for name, role in roles.items()} == {
            "generator": 6,
            "fake": 18,
        }
        assert all(
            torch.count_nonzero(delta).item() > 0 for delta in evidence["role_deltas"].values()
        )
        counters.steps = {name: role.step for name, role in roles.items()}
    else:
        roles = _roles_from_optimizer_groups(optimizer)
        coordinator = RoleOptimizationCoordinator(
            accelerator,
            model,
            optimizer,
            roles,
        )
        accelerator.register_for_checkpointing(coordinator)
        evidence = _run_production_tdm_optimize(
            accelerator,
            model,
            optimizer,
            roles,
            coordinator,
            ratio=1,
            iterations=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        assert evidence["trace"] == (
            ("fake",) * gradient_accumulation_steps + ("generator",) * gradient_accumulation_steps
        )
        assert evidence["optimize_calls"] == 1
        assert {name: role.step for name, role in roles.items()} == {
            "generator": 1,
            "fake": 1,
        }
        assert all(
            torch.count_nonzero(delta).item() > 0 for delta in evidence["role_deltas"].values()
        )
        counters.steps = {name: role.step for name, role in roles.items()}

    input_value = torch.tensor([1.75], device=accelerator.device)
    # An evaluation forward with grad enabled leaves DDP's reducer expecting a backward, which
    # the next no_sync accumulation micro-step would then complete with stale unused parameters.
    with torch.no_grad():
        expected_generator_output = model(input_value, "generator").detach().clone()
    expected_counters = dict(counters.steps)
    expected_optimizer_steps = _optimizer_steps(optimizer)
    checkpoint_directory = _shared_checkpoint_directory(
        accelerator,
        backend,
        storage_mode,
        algorithm,
    )
    accelerator.save_state(str(checkpoint_directory))

    if algorithm == "tdm-r1":
        _run_production_tdm_r1_optimize(
            accelerator,
            model,
            optimizer,
            roles,
            coordinator,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        counters.steps = {name: role.step for name, role in roles.items()}
    elif algorithm == "dmd2":
        _run_production_dmd2_optimize(
            accelerator,
            model,
            optimizer,
            roles,
            coordinator,
            ratio=3,
            iterations=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        counters.steps = {name: role.step for name, role in roles.items()}
    else:
        _run_production_tdm_optimize(
            accelerator,
            model,
            optimizer,
            roles,
            coordinator,
            ratio=1,
            iterations=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        counters.steps = {name: role.step for name, role in roles.items()}
    accelerator.load_state(str(checkpoint_directory))
    if algorithm in ("tdm-r1", "dmd2", "tdm"):
        counters.steps = {name: role.step for name, role in roles.items()}

    with torch.no_grad():
        restored_generator_output = model(input_value, "generator").detach()
    torch.testing.assert_close(restored_generator_output, expected_generator_output)
    assert counters.steps == expected_counters
    assert _optimizer_steps(optimizer) == expected_optimizer_steps
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(checkpoint_directory)


def test_cpu_sequential_wrapper_preserves_role_local_state_and_chronology() -> None:
    gradient_accumulation_steps = 1
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    model = TinyMultiRoleBundle("lora")
    optimizer = _optimizer(model, _algorithm_roles("tdm-r1"))
    model, optimizer = accelerator.prepare(model, optimizer)
    roles = _roles_from_optimizer_groups(optimizer)
    coordinator = RoleOptimizationCoordinator(
        accelerator,
        model,
        optimizer,
        roles,
    )
    value = torch.tensor([2.0], device=accelerator.device)
    initial_fake = model(value, "fake").detach().clone()

    first = _run_sequential_boundary(coordinator, model, accelerator)
    assert not torch.equal(first["surrogate_sees_fake"], initial_fake)

    assert {name: role.step for name, role in roles.items()} == {
        "generator": 1,
        "fake": 1,
        "surrogate": 1,
    }
    assert set(optimizer.state) == {
        parameter for role in roles.values() for parameter in role.parameters
    }
    assert all(
        optimizer.state[parameter]["step"].item() == 1
        for role in roles.values()
        for parameter in role.parameters
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_distributed_wiring_executes_production_tdm_r1_optimize() -> None:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    model = TinyMultiRoleBundle("lora")
    optimizer = _optimizer(model, _algorithm_roles("tdm-r1"))
    model, optimizer = accelerator.prepare(model, optimizer)
    roles = _roles_from_optimizer_groups(optimizer)
    coordinator = RoleOptimizationCoordinator(
        accelerator,
        model,
        optimizer,
        roles,
    )

    evidence = _run_production_tdm_r1_optimize(
        accelerator,
        model,
        optimizer,
        roles,
        coordinator,
        gradient_accumulation_steps=1,
    )

    assert evidence["trace"] == ("fake", "surrogate", "generator")
    assert evidence["optimize_calls"] == 1
    assert len(accelerator._models) == 1
    assert accelerator._models[0] is model
    assert len(accelerator._optimizers) == 1
    assert accelerator._optimizers[0] is optimizer
    assert all(torch.count_nonzero(delta).item() > 0 for delta in evidence["role_deltas"].values())


def test_cpu_distributed_wiring_executes_production_dmd2_outer_iterations() -> None:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    model = TinyMultiRoleBundle("lora")
    optimizer = _optimizer(model, _algorithm_roles("dmd2"))
    model, optimizer = accelerator.prepare(model, optimizer)
    roles = _roles_from_optimizer_groups(optimizer)
    coordinator = RoleOptimizationCoordinator(
        accelerator,
        model,
        optimizer,
        roles,
    )

    evidence = _run_production_dmd2_optimize(
        accelerator,
        model,
        optimizer,
        roles,
        coordinator,
        ratio=3,
        iterations=6,
    )

    assert evidence["trace"] == ("fake", "fake", "fake", "generator") * 6
    assert evidence["optimize_calls"] == 6
    assert evidence["trainer_step"] == 6
    assert {name: role.step for name, role in roles.items()} == {"generator": 6, "fake": 18}
    assert _optimizer_steps(optimizer) == {"generator": (6.0,), "fake": (18.0,)}
    assert len(accelerator._models) == 1
    assert accelerator._models[0] is model
    assert len(accelerator._optimizers) == 1
    assert accelerator._optimizers[0] is optimizer
    assert all(torch.count_nonzero(delta).item() > 0 for delta in evidence["role_deltas"].values())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_tdm_r1_cli_collects_selected_matrix() -> None:
    repository_root = Path(__file__).parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/distributed/test_multirole_backend.py::test_backend_role_phases",
        "tests/distributed/test_multirole_backend.py::" "test_deepspeed_tdm_r1_zero2_is_allowed",
        "--collect-only",
        "-q",
        "--backend=zero2",
        "--algorithm=tdm-r1",
        "--storage-mode=lora",
    ]

    result = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_backend_role_phases" in result.stdout
    assert "test_deepspeed_tdm_r1_zero2_is_allowed" in result.stdout
