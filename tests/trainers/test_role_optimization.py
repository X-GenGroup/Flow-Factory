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

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, List, Mapping, Optional, Tuple

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

from flow_factory.hparams.optimizer_args import (
    AdamWOptimizerArguments,
    MultiOptimizerArguments,
)
from flow_factory.models.variants import (
    DEFAULT_BASE_VARIANT,
    ComponentVariantRegistry,
    ComponentVariantSpec,
)
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
    RolePhase,
    RoleUpdatePlan,
)


class TinyBundle(torch.nn.Module):
    """Expose two independently optimized scalar roles."""

    def __init__(self, generator: float = 1.0, fake: float = -1.0) -> None:
        super().__init__()
        self.generator = torch.nn.Parameter(torch.tensor(generator))
        self.fake = torch.nn.Parameter(torch.tensor(fake))


class RecordingScheduler:
    """Record role-local scheduler cadence."""

    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        """Record one scheduler step."""
        self.steps += 1


class MinimalTrainer(BaseTrainer):
    """Provide concrete hooks for uninitialized BaseTrainer seam tests."""

    def start(self) -> None:
        """Provide an unused concrete training hook."""

    def prepare_feedback(self, samples: List[Any]) -> None:
        """Provide an unused concrete feedback hook."""

    def optimize(self, *args: Any, **kwargs: Any) -> None:
        """Provide an unused concrete optimization hook."""


class DeterministicAccelerator:
    """Implement the accumulation surface needed by the coordinator tests."""

    def __init__(
        self,
        sync_sequence: Tuple[bool, ...],
        distributed_type: DistributedType = DistributedType.MULTI_GPU,
    ) -> None:
        self._sync_sequence = sync_sequence
        self._microbatch_index = 0
        self.sync_gradients = False
        self.accumulate_depth = 0
        self.clipped_parameter_ids: List[Tuple[int, ...]] = []
        # The coordinator reads this to decide whether `parameter.grad` is authoritative.
        self.distributed_type = distributed_type
        self.state = SimpleNamespace(
            fsdp_plugin=(
                SimpleNamespace(fsdp_version=1)
                if distributed_type == DistributedType.FSDP
                else None
            )
        )

    @contextmanager
    def accumulate(self, model: torch.nn.Module) -> Iterator[None]:
        """Set the next explicit synchronization decision."""
        del model
        if self._microbatch_index >= len(self._sync_sequence):
            raise RuntimeError(
                "expected another sync decision for microbatch, "
                f"received only {self._sync_sequence!r}"
            )
        self.sync_gradients = self._sync_sequence[self._microbatch_index]
        self._microbatch_index += 1
        self.accumulate_depth += 1
        try:
            yield
        finally:
            self.accumulate_depth -= 1

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate while proving accumulation is active."""
        if self.accumulate_depth != 1:
            raise RuntimeError(
                "expected backward inside one accumulate context, "
                f"received depth {self.accumulate_depth}"
            )
        loss.backward()

    def clip_grad_norm_(
        self, parameters: Tuple[torch.nn.Parameter, ...], max_norm: float
    ) -> torch.Tensor:
        """Record exact clipping ownership and delegate to PyTorch."""
        parameter_tuple = tuple(parameters)
        self.clipped_parameter_ids.append(tuple(id(parameter) for parameter in parameter_tuple))
        return torch.nn.utils.clip_grad_norm_(parameter_tuple, max_norm)


class ReplacingPrepareAccelerator(DeterministicAccelerator):
    """Simulate preparation replacing model and optimizer parameter identities."""

    def prepare(
        self,
        model: TinyBundle,
        optimizer: torch.optim.AdamW,
    ) -> Tuple[TinyBundle, torch.optim.AdamW]:
        """Return equivalent objects backed by fresh parameters."""
        replacement = TinyBundle(
            generator=model.generator.detach().item(),
            fake=model.fake.detach().item(),
        )
        replacement_parameters = {
            "generator": replacement.generator,
            "fake": replacement.fake,
        }
        replacement_groups = []
        for group in optimizer.param_groups:
            role_name = group["role_name"]
            replacement_groups.append(
                {
                    "params": [replacement_parameters[role_name]],
                    "role_name": role_name,
                    "lr": group["lr"],
                    "betas": group["betas"],
                    "weight_decay": group["weight_decay"],
                    "eps": group["eps"],
                }
            )
        return replacement, torch.optim.AdamW(replacement_groups)


class SkippingOptimizer:
    """Expose AcceleratedOptimizer's overflow signal while skipping updates."""

    def __init__(self, optimizer: torch.optim.AdamW) -> None:
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.step_was_skipped = True
        self.step_calls = 0

    def step(self) -> None:
        """Consume a prepared step call without updating AdamW state."""
        self.step_calls += 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients through the wrapped optimizer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)


class BackendContractAccelerator:
    """Expose Accelerate's prepared-object tracking contract."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        fsdp_plugin: Optional[Any] = None,
        deepspeed_plugin: Optional[Any] = None,
    ) -> None:
        self.state = SimpleNamespace(
            fsdp_plugin=fsdp_plugin,
            deepspeed_plugin=deepspeed_plugin,
        )
        self._models = [model]
        self._optimizers = [optimizer]

    @staticmethod
    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        """Return the tiny prepared root unchanged."""
        return model


def _config(
    role_name: str,
    *,
    learning_rate: float = 0.05,
    max_grad_norm: float = 10.0,
) -> RoleOptimizerConfig:
    return RoleOptimizerConfig(
        role_name=role_name,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        adam_betas=(0.8, 0.9),
        adam_weight_decay=0.01,
        adam_epsilon=1e-6,
        max_grad_norm=max_grad_norm,
    )


def _runtime(
    accelerator: Any,
    *,
    generator: float = 1.0,
    fake: float = -1.0,
    generator_scheduler: Optional[Any] = None,
    fake_scheduler: Optional[Any] = None,
    optimizer_factory: Optional[Any] = None,
) -> Tuple[
    TinyBundle,
    torch.optim.AdamW,
    Mapping[str, OptimizationRole],
    RoleOptimizationCoordinator,
]:
    bundle = TinyBundle(generator=generator, fake=fake)
    configs = (_config("generator", learning_rate=0.03), _config("fake", learning_rate=0.07))
    optimizer_class = optimizer_factory or torch.optim.AdamW
    optimizer = optimizer_class(
        [
            {
                "params": [bundle.generator],
                "role_name": "generator",
                "lr": configs[0].learning_rate,
                "betas": configs[0].adam_betas,
                "weight_decay": configs[0].adam_weight_decay,
                "eps": configs[0].adam_epsilon,
            },
            {
                "params": [bundle.fake],
                "role_name": "fake",
                "lr": configs[1].learning_rate,
                "betas": configs[1].adam_betas,
                "weight_decay": configs[1].adam_weight_decay,
                "eps": configs[1].adam_epsilon,
            },
        ]
    )
    roles = {
        "generator": OptimizationRole(
            config=configs[0],
            parameters=(bundle.generator,),
            optimizer_group_ids=(0,),
            scheduler=generator_scheduler,
        ),
        "fake": OptimizationRole(
            config=configs[1],
            parameters=(bundle.fake,),
            optimizer_group_ids=(1,),
            scheduler=fake_scheduler,
        ),
    }
    coordinator = RoleOptimizationCoordinator(
        accelerator,
        bundle,
        optimizer,
        roles,
    )
    return bundle, optimizer, roles, coordinator


def _run_phase(
    coordinator: RoleOptimizationCoordinator,
    role_name: str,
    losses: Tuple[torch.Tensor, ...],
) -> List[bool]:
    stepped: List[bool] = []
    with coordinator.phase(role_name):  # type: ignore[arg-type]
        for loss in losses:
            with coordinator.microbatch():
                coordinator.backward(loss)
                stepped.append(coordinator.finish_microbatch())
    return stepped


def _backend_contract_trainer(
    *,
    fsdp_plugin: Optional[Any] = None,
    deepspeed_plugin: Optional[Any] = None,
    trainer_type: str = "dmd2",
) -> MinimalTrainer:
    bundle, optimizer, roles, _ = _runtime(DeterministicAccelerator((True,)))
    trainer = object.__new__(MinimalTrainer)
    trainer.training_args = SimpleNamespace(
        trainer_type=trainer_type,
        required_trainable_roles=("generator", "fake"),
    )
    trainer.model_bundle = bundle
    trainer.optimizer = optimizer
    trainer.optimization_roles = dict(roles)
    trainer._unprepared_optimizer_group_roles = ("generator", "fake")
    trainer.accelerator = BackendContractAccelerator(
        bundle,
        optimizer,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
    )
    return trainer


def test_base_trainer_builds_one_ordered_adamw_with_role_hyperparameters() -> None:
    bundle = TinyBundle()

    class Registry:
        variant_names = ("generator", "fake")

        @staticmethod
        def get_spec(role_name: str) -> Any:
            return SimpleNamespace(storage_mode="full")

        @staticmethod
        def parameters(role_name: str) -> Tuple[torch.nn.Parameter, ...]:
            return (getattr(bundle, role_name),)

    trainer = object.__new__(MinimalTrainer)
    trainer.adapter = SimpleNamespace(component_variant_registry=Registry())
    trainer.training_args = SimpleNamespace(
        learning_rate=0.1,
        adam_betas=(0.7, 0.8),
        adam_weight_decay=0.02,
        adam_epsilon=1e-5,
        max_grad_norm=3.0,
    )
    trainer._role_optimizer_configs = lambda: (
        _config("generator", learning_rate=0.1, max_grad_norm=3.0),
        _config("fake", learning_rate=0.2, max_grad_norm=4.0),
    )
    trainer.config = SimpleNamespace(
        optimizer_args=MultiOptimizerArguments(
            optimizer_configs=[
                AdamWOptimizerArguments(
                    name="generator", learning_rate=0.1, betas=(0.8, 0.9), weight_decay=0.02
                ),
                AdamWOptimizerArguments(
                    name="fake", learning_rate=0.2, betas=(0.8, 0.9), weight_decay=0.02
                ),
            ]
        )
    )
    trainer.accelerator = SimpleNamespace(distributed_type=DistributedType.NO)

    optimizer = BaseTrainer._init_optimizer(trainer)

    assert type(optimizer) is torch.optim.AdamW
    assert optimizer is trainer.optimizer
    assert [group["role_name"] for group in optimizer.param_groups] == ["generator", "fake"]
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.2]
    assert [group["betas"] for group in optimizer.param_groups] == [(0.8, 0.9)] * 2
    assert [role.parameters for role in trainer.optimization_roles.values()] == [
        (bundle.generator,),
        (bundle.fake,),
    ]


def test_flat_training_fields_reach_the_role_config_through_one_translation() -> None:
    """The single-optimizer shorthand resolves to the same list an explicit config writes."""
    trainer = object.__new__(MinimalTrainer)
    trainer.training_args = SimpleNamespace()
    trainer.config = SimpleNamespace(
        optimizer_args=MultiOptimizerArguments(
            optimizer_configs=[
                AdamWOptimizerArguments(
                    name="base",
                    learning_rate=0.001,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                    eps=1e-7,
                    max_grad_norm=1.5,
                )
            ]
        )
    )

    assert BaseTrainer._role_optimizer_configs(trainer) == (
        RoleOptimizerConfig(
            role_name="base",
            learning_rate=0.001,
            adam_betas=(0.9, 0.95),
            adam_weight_decay=0.1,
            adam_epsilon=1e-7,
            max_grad_norm=1.5,
        ),
    )


def test_rebinds_roles_to_prepared_optimizer_parameter_identities() -> None:
    accelerator = ReplacingPrepareAccelerator((True,))
    stale_bundle, stale_optimizer, stale_roles, _ = _runtime(accelerator)
    stale_parameter_ids = {
        id(parameter) for role in stale_roles.values() for parameter in role.parameters
    }
    prepared_bundle, prepared_optimizer = accelerator.prepare(stale_bundle, stale_optimizer)
    trainer = object.__new__(MinimalTrainer)
    trainer.training_args = SimpleNamespace(
        trainer_type="dmd2", required_trainable_roles=("generator", "fake")
    )
    trainer.optimization_roles = dict(stale_roles)
    trainer.model_bundle = prepared_bundle
    trainer.optimizer = prepared_optimizer
    trainer.accelerator = accelerator

    BaseTrainer._init_prepared_role_optimization(trainer)
    coordinator = trainer.role_optimization
    prepared_parameter_ids = {
        id(parameter) for group in prepared_optimizer.param_groups for parameter in group["params"]
    }
    rebound_parameter_ids = {
        id(parameter)
        for role in trainer.optimization_roles.values()
        for parameter in role.parameters
    }

    assert tuple(trainer.optimization_roles) == ("generator", "fake")
    assert trainer.optimization_roles["generator"].config is stale_roles["generator"].config
    assert trainer.optimization_roles["generator"].optimizer_group_ids == (0,)
    assert trainer.optimization_roles["fake"].optimizer_group_ids == (1,)
    assert rebound_parameter_ids == prepared_parameter_ids
    assert rebound_parameter_ids.isdisjoint(stale_parameter_ids)
    stepped = _run_phase(
        coordinator,
        "fake",
        ((prepared_bundle.fake - 3.0).square(),),
    )
    assert stepped == [True]
    assert set(prepared_optimizer.state) == {prepared_bundle.fake}


def test_backend_contract_accepts_fsdp2_original_parameter_identity() -> None:
    trainer = _backend_contract_trainer(
        fsdp_plugin=SimpleNamespace(fsdp_version=2, use_orig_params=None)
    )

    BaseTrainer._validate_multirole_backend(trainer)


def test_backend_contract_rejects_fsdp2_optimizer_parameter_outside_root() -> None:
    trainer = _backend_contract_trainer(
        fsdp_plugin=SimpleNamespace(fsdp_version=2, use_orig_params=None)
    )
    trainer.optimizer.param_groups[1]["params"] = [torch.nn.Parameter(torch.ones(1))]

    with pytest.raises(RuntimeError, match="FSDP2.*optimizer parameter.*prepared model root.*dmd2"):
        BaseTrainer._validate_multirole_backend(trainer)


@pytest.mark.parametrize("zero_stage", [0, 3])
def test_backend_contract_rejects_unsupported_deepspeed_stage(zero_stage: int) -> None:
    trainer = _backend_contract_trainer(deepspeed_plugin=SimpleNamespace(zero_stage=zero_stage))

    with pytest.raises(
        ValueError,
        match=rf"DeepSpeed.*dmd2.*generator.*fake.*zero_stage.*{zero_stage}",
    ):
        BaseTrainer._validate_multirole_backend(trainer)


@pytest.mark.parametrize("zero_stage", [1, 2])
def test_backend_contract_accepts_deepspeed_zero_one_and_two(zero_stage: int) -> None:
    trainer = _backend_contract_trainer(deepspeed_plugin=SimpleNamespace(zero_stage=zero_stage))

    BaseTrainer._validate_multirole_backend(trainer)


@pytest.mark.parametrize("zero_stage", [1, 2])
def test_backend_contract_accepts_deepspeed_zero_one_and_two_for_tdm_r1(zero_stage: int) -> None:
    trainer = _backend_contract_trainer(
        trainer_type="tdm-r1",
        deepspeed_plugin=SimpleNamespace(zero_stage=zero_stage),
    )

    BaseTrainer._validate_multirole_backend(trainer)


def test_backend_contract_requires_one_tracked_root_and_optimizer() -> None:
    trainer = _backend_contract_trainer()
    trainer.accelerator._models.append(TinyBundle())
    with pytest.raises(RuntimeError, match="dmd2.*one prepared model root.*received 2"):
        BaseTrainer._validate_multirole_backend(trainer)

    trainer = _backend_contract_trainer()
    trainer.accelerator._optimizers.clear()
    with pytest.raises(RuntimeError, match="dmd2.*one prepared optimizer.*received 0"):
        BaseTrainer._validate_multirole_backend(trainer)


def test_backend_contract_rejects_changed_optimizer_group_role_mapping() -> None:
    trainer = _backend_contract_trainer()
    trainer.optimizer.param_groups[1]["role_name"] = "surrogate"

    with pytest.raises(
        RuntimeError,
        match="dmd2.*optimizer group role mapping.*generator.*fake.*generator.*surrogate",
    ):
        BaseTrainer._validate_multirole_backend(trainer)


def test_backend_contract_leaves_single_role_legacy_backend_unchanged() -> None:
    trainer = _backend_contract_trainer(deepspeed_plugin=SimpleNamespace(zero_stage=3))
    trainer.training_args.required_trainable_roles = ("generator",)
    trainer.accelerator._models.clear()
    trainer.accelerator._optimizers.clear()

    BaseTrainer._validate_multirole_backend(trainer)


def test_rejects_duplicate_and_non_exhaustive_optimizer_ownership() -> None:
    bundle, optimizer, roles, _ = _runtime(DeterministicAccelerator((True,)))
    duplicate_roles = dict(roles)
    duplicate_roles["fake"] = OptimizationRole(
        config=roles["fake"].config,
        parameters=(bundle.generator,),
        optimizer_group_ids=(1,),
    )
    with pytest.raises(ValueError, match="disjoint.*fake.*generator"):
        RoleOptimizationCoordinator(
            DeterministicAccelerator((True,)), bundle, optimizer, duplicate_roles
        )

    missing_roles = {"generator": roles["generator"]}
    with pytest.raises(ValueError, match="expected exhaustive.*received"):
        RoleOptimizationCoordinator(
            DeterministicAccelerator((True,)), bundle, optimizer, missing_roles
        )


def test_accepts_deepspeed_style_flattened_optimizer_tensor() -> None:
    flat_parameter = torch.zeros(8, requires_grad=True)
    optimizer = torch.optim.AdamW([{"params": [flat_parameter], "role_name": "base"}])
    role = OptimizationRole(
        config=_config("base"),
        parameters=(flat_parameter,),
        optimizer_group_ids=(0,),
    )

    coordinator = RoleOptimizationCoordinator(
        SimpleNamespace(),
        torch.nn.Identity(),
        optimizer,
        {"base": role},
    )

    assert coordinator.roles["base"].parameters[0] is flat_parameter


def test_active_role_only_receives_adamw_state_and_clipping() -> None:
    accelerator = DeterministicAccelerator((True,))
    bundle, optimizer, roles, coordinator = _runtime(accelerator)

    stepped = _run_phase(coordinator, "fake", ((bundle.fake - 2.0).square(),))

    assert stepped == [True]
    assert set(optimizer.state) == {bundle.fake}
    assert bundle.generator not in optimizer.state
    assert optimizer.state[bundle.fake]["step"].item() == 1
    assert accelerator.clipped_parameter_ids == [(id(bundle.fake),)]
    assert roles["fake"].step == 1
    assert roles["generator"].step == 0


def test_fsdp1_uses_collective_root_clipping_and_role_exclusive_step() -> None:
    accelerator = DeterministicAccelerator((True,), DistributedType.FSDP)
    bundle, optimizer, roles, coordinator = _runtime(accelerator)

    stepped = _run_phase(coordinator, "fake", ((bundle.fake - 2.0).square(),))

    assert stepped == [True]
    assert set(optimizer.state) == {bundle.fake}
    assert accelerator.clipped_parameter_ids == [
        tuple(id(parameter) for parameter in bundle.parameters())
    ]
    assert roles["fake"].step == 1
    assert roles["generator"].step == 0


def test_active_role_hides_inactive_groups_from_zero_style_optimizer_step() -> None:
    """ZeRO materializes zero gradients, so it must not even see inactive params."""

    class UpdatesEveryVisibleParameter(torch.optim.Optimizer):
        """Simulate a backend optimizer that updates zero-gradient partitions."""

        def __init__(self, groups: List[dict[str, Any]]) -> None:
            super().__init__(groups, {"lr": 0.1})

        @torch.no_grad()
        def step(self, closure: Any = None) -> None:
            del closure
            for group in self.param_groups:
                for parameter in group["params"]:
                    parameter.add_(group["lr"])

    accelerator = DeterministicAccelerator((True,))
    bundle, optimizer, roles, coordinator = _runtime(
        accelerator,
        optimizer_factory=UpdatesEveryVisibleParameter,
    )
    generator_before = bundle.generator.detach().clone()
    original_generator_group = tuple(optimizer.param_groups[0]["params"])

    _run_phase(coordinator, "fake", ((bundle.fake - 2.0).square(),))

    torch.testing.assert_close(bundle.generator, generator_before, rtol=0, atol=0)
    assert tuple(optimizer.param_groups[0]["params"]) == original_generator_group
    assert optimizer.param_groups[0]["params"][0] is bundle.generator
    assert roles["fake"].step == 1
    assert roles["generator"].step == 0


def test_deepspeed_group_filter_calls_only_active_private_optimizer_step() -> None:
    """ZeRO-2 iterates internal bit16 groups instead of public param_groups."""

    class ZeroOptimizerFake:
        def __init__(self) -> None:
            self.calls: List[int] = []

        def _optimizer_step(self, group_no: int) -> None:
            self.calls.append(group_no)

    class DeepSpeedWrapperFake:
        def __init__(self, groups: List[dict[str, Any]]) -> None:
            self.param_groups = groups
            self.optimizer = ZeroOptimizerFake()

    bundle = TinyBundle(generator=0.4, fake=-0.2)
    configs = (_config("generator"), _config("fake"))
    roles = {
        "generator": OptimizationRole(configs[0], (bundle.generator,), (0,)),
        "fake": OptimizationRole(configs[1], (bundle.fake,), (1,)),
    }
    wrapper = DeepSpeedWrapperFake(
        [
            {"params": [bundle.generator], "role_name": "generator"},
            {"params": [bundle.fake], "role_name": "fake"},
        ]
    )
    coordinator = RoleOptimizationCoordinator(
        SimpleNamespace(distributed_type=DistributedType.DEEPSPEED),
        bundle,
        wrapper,  # type: ignore[arg-type]
        roles,
    )

    with coordinator._active_optimizer_groups_only(roles["fake"]):
        wrapper.optimizer._optimizer_step(0)
        wrapper.optimizer._optimizer_step(1)

    assert wrapper.optimizer.calls == [1]
    # The exact original method is restored after the role-local window.
    wrapper.optimizer._optimizer_step(0)
    assert wrapper.optimizer.calls == [1, 0]


def test_rejects_inactive_and_stale_gradients_with_role_context() -> None:
    bundle, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))
    bundle.fake.grad = torch.ones_like(bundle.fake)
    with pytest.raises(RuntimeError, match="phase entry.*expected.*None.*fake"):
        with coordinator.phase("generator"):
            pass

    bundle.fake.grad = None
    with pytest.raises(RuntimeError, match="inactive.*fake.*generator.*expected.*None"):
        with coordinator.phase("generator"):
            with coordinator.microbatch():
                coordinator.backward(bundle.generator.square())
                bundle.fake.grad = torch.ones_like(bundle.fake)
                coordinator.finish_microbatch()


def test_rejects_sync_without_an_active_role_gradient() -> None:
    _, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))

    with pytest.raises(RuntimeError, match="active role.*generator.*at least one gradient"):
        with coordinator.phase("generator"):
            with coordinator.microbatch():
                coordinator.backward(torch.tensor(1.0, requires_grad=True))
                coordinator.finish_microbatch()


def test_rejects_nested_phase_and_wrong_microbatch_order() -> None:
    _, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))
    with pytest.raises(RuntimeError, match="cannot enter role.*fake.*open.*generator"):
        with coordinator.phase("generator"):
            with coordinator.phase("fake"):
                pass

    with pytest.raises(RuntimeError, match="microbatch.*open phase"):
        with coordinator.microbatch():
            pass


def test_requires_exactly_one_finish_call_per_microbatch() -> None:
    bundle, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))
    with pytest.raises(RuntimeError, match="microbatch exit.*exactly one.*finish_microbatch"):
        with coordinator.phase("generator"):
            with coordinator.microbatch():
                coordinator.backward(bundle.generator.square())

    bundle, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))
    with pytest.raises(RuntimeError, match="finish_microbatch.*already called"):
        with coordinator.phase("generator"):
            with coordinator.microbatch():
                coordinator.backward(bundle.generator.square())
                coordinator.finish_microbatch()
                coordinator.finish_microbatch()


def test_requires_exactly_one_backward_call_per_microbatch() -> None:
    _, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))
    with pytest.raises(RuntimeError, match="finish_microbatch.*exactly one backward.*received 0"):
        with coordinator.phase("generator"):
            with coordinator.microbatch():
                coordinator.finish_microbatch()

    bundle, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))
    with pytest.raises(RuntimeError, match="backward.*already called.*generator"):
        with coordinator.phase("generator"):
            with coordinator.microbatch():
                coordinator.backward(bundle.generator.square())
                coordinator.backward((bundle.generator - 1.0).square())


def test_rejects_incomplete_and_extra_sync_boundaries() -> None:
    bundle, _, _, coordinator = _runtime(DeterministicAccelerator((False,)))
    with pytest.raises(RuntimeError, match="phase exit.*exactly one.*received 0"):
        _run_phase(coordinator, "generator", (bundle.generator.square(),))

    bundle, _, _, coordinator = _runtime(DeterministicAccelerator((True, True)))
    with pytest.raises(RuntimeError, match="extra sync.*generator.*already stepped"):
        _run_phase(
            coordinator,
            "generator",
            (bundle.generator.square(), bundle.generator.square()),
        )


def test_gas_one_preserves_order_and_advances_role_scheduler_only() -> None:
    generator_scheduler = RecordingScheduler()
    fake_scheduler = RecordingScheduler()
    accelerator = DeterministicAccelerator((True, True))
    bundle, _, roles, coordinator = _runtime(
        accelerator,
        generator_scheduler=generator_scheduler,
        fake_scheduler=fake_scheduler,
    )

    fake_stepped = _run_phase(coordinator, "fake", ((bundle.fake - 1.0).square(),))
    generator_stepped = _run_phase(coordinator, "generator", ((bundle.generator + 1.0).square(),))

    assert fake_stepped == [True]
    assert generator_stepped == [True]
    assert roles["fake"].step == roles["generator"].step == 1
    assert fake_scheduler.steps == generator_scheduler.steps == 1


def test_real_accelerator_gas_two_matches_explicit_large_batch() -> None:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    bundle, _, roles, coordinator = _runtime(accelerator, generator=0.4)
    large_batch_parameter = torch.nn.Parameter(torch.tensor(0.4))
    large_batch_optimizer = torch.optim.AdamW(
        [large_batch_parameter],
        lr=roles["generator"].config.learning_rate,
        betas=roles["generator"].config.adam_betas,
        weight_decay=roles["generator"].config.adam_weight_decay,
        eps=roles["generator"].config.adam_epsilon,
    )
    samples = (torch.tensor(1.0), torch.tensor(3.0))

    stepped = _run_phase(
        coordinator,
        "generator",
        tuple((bundle.generator - sample).square() for sample in samples),
    )
    torch.stack([(large_batch_parameter - sample).square() for sample in samples]).mean().backward()
    large_batch_optimizer.step()

    assert stepped == [False, True]
    torch.testing.assert_close(bundle.generator, large_batch_parameter, rtol=0, atol=1e-7)
    assert roles["generator"].step == 1


def test_generator_only_public_step_callback() -> None:
    accelerator = DeterministicAccelerator((True, True))
    bundle, _, _, coordinator = _runtime(accelerator)
    trainer = object.__new__(MinimalTrainer)
    trainer.step = 0
    trainer.role_optimization = coordinator
    # The public step is paced by the primary role, which is the first declared.
    trainer.training_args = SimpleNamespace(required_trainable_roles=("generator", "fake"))

    with coordinator.phase("fake"):
        with coordinator.microbatch():
            coordinator.backward(bundle.fake.square())
            assert BaseTrainer._finish_role_microbatch(trainer) is True
    assert trainer.step == 0

    with coordinator.phase("generator"):
        with coordinator.microbatch():
            coordinator.backward(bundle.generator.square())
            assert BaseTrainer._finish_role_microbatch(trainer) is True
    assert trainer.step == 1


def test_skipped_prepared_step_consumes_phase_without_advancing_state() -> None:
    scheduler = RecordingScheduler()
    accelerator = DeterministicAccelerator((True,))
    bundle, optimizer, roles, _ = _runtime(
        accelerator,
        generator_scheduler=scheduler,
    )
    skipping_optimizer = SkippingOptimizer(optimizer)
    coordinator = RoleOptimizationCoordinator(
        accelerator,
        bundle,
        skipping_optimizer,  # type: ignore[arg-type]
        roles,
    )
    trainer = object.__new__(MinimalTrainer)
    trainer.step = 0
    trainer.role_optimization = coordinator
    generator_before = bundle.generator.detach().clone()

    with coordinator.phase("generator"):
        with coordinator.microbatch():
            coordinator.backward(bundle.generator.square())
            assert BaseTrainer._finish_role_microbatch(trainer) is False

    torch.testing.assert_close(bundle.generator, generator_before)
    assert skipping_optimizer.step_calls == 1
    assert skipping_optimizer.state == {}
    assert roles["generator"].step == 0
    assert scheduler.steps == 0
    assert trainer.step == 0
    assert bundle.generator.grad is None


def test_state_round_trip_restores_only_closed_phase_counters() -> None:
    accelerator = DeterministicAccelerator((True,))
    bundle, _, roles, coordinator = _runtime(accelerator)
    _run_phase(coordinator, "fake", (bundle.fake.square(),))
    state = coordinator.state_dict()

    _, _, restored_roles, restored = _runtime(DeterministicAccelerator((True,)))
    restored.load_state_dict(state)

    assert restored_roles["generator"].step == 0
    assert restored_roles["fake"].step == 1
    assert restored.state_dict() == state


def test_state_dict_rejects_an_open_phase() -> None:
    _, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))

    with pytest.raises(RuntimeError, match="state_dict.*closed phase.*generator"):
        with coordinator.phase("generator"):
            coordinator.state_dict()


@pytest.mark.parametrize(
    "state,match",
    [
        ({}, "expected coordinator state version 3.*received None"),
        (
            {
                "version": 1,
                "role_steps": {"generator": 0, "fake": 0},
                "optimizer_group_roles": ("generator", "fake"),
                "active_phase": None,
            },
            r"version 1.*retired.*cannot be migrated",
        ),
        (
            {
                "version": 2,
                "role_steps": {"generator": 0, "fake": 0},
                "optimizer_group_roles": ("generator", "fake"),
                "active_phase": None,
            },
            r"version 2.*retired.*cannot be migrated",
        ),
        (
            {
                "version": 3,
                "role_steps": {"generator": 0},
                "optimizer_group_roles": ("generator", "fake"),
                "active_phase": None,
            },
            "role_steps.*generator.*fake.*received",
        ),
        (
            {
                "version": 3,
                "role_steps": {"generator": 0, "fake": -1},
                "optimizer_group_roles": ("generator", "fake"),
                "active_phase": None,
            },
            "non-negative int.*fake.*-1",
        ),
        (
            {
                "version": 3,
                "role_steps": {"generator": 0, "fake": 0},
                "optimizer_group_roles": ("fake", "generator"),
                "active_phase": None,
            },
            "optimizer group roles.*expected.*generator.*fake.*received",
        ),
        (
            {
                "version": 3,
                "role_steps": {"generator": 0, "fake": 0},
                "optimizer_group_roles": ("generator", "fake"),
                "active_phase": "generator",
            },
            "closed phase.*None.*generator",
        ),
    ],
)
def test_rejects_malformed_state_with_expected_received_context(
    state: Mapping[str, Any], match: str
) -> None:
    _, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))

    with pytest.raises((TypeError, ValueError), match=match):
        coordinator.load_state_dict(state)


def test_coordinator_has_no_retired_chronology_api() -> None:
    _, _, _, coordinator = _runtime(DeterministicAccelerator((True,)))
    for name in (
        "begin_outer_iteration",
        "interleaved_microbatch",
        "active_role",
        "backward_active",
        "finish_active_role",
        "chronology_policy",
        "outer_step",
    ):
        assert not hasattr(coordinator, name)


def test_six_iterations_with_ttur_three_step_only_active_role() -> None:
    bundle, optimizer, roles, coordinator = _runtime(DeterministicAccelerator((True,) * 24))
    for _ in range(6):
        for _ in range(3):
            _run_phase(coordinator, "fake", (bundle.fake.square(),))
        _run_phase(coordinator, "generator", (bundle.generator.square(),))

    assert roles["fake"].step == 18
    assert roles["generator"].step == 6
    assert bundle.fake.grad is None
    assert bundle.generator.grad is None
    assert set(optimizer.state) == {bundle.fake, bundle.generator}


def test_role_plan_values_are_immutable_and_validate_ranges() -> None:
    plan = RoleUpdatePlan(phases=(RolePhase("fake", repeats=2), RolePhase("generator", repeats=1)))

    assert plan.phases[0].repeats == 2
    with pytest.raises(ValueError, match="repeats.*>= 1.*received 0"):
        RolePhase("fake", repeats=0)
    with pytest.raises(ValueError, match="update_frequency.*>= 1.*received 0"):
        RoleOptimizerConfig(
            role_name="base",
            learning_rate=0.1,
            adam_betas=(0.9, 0.999),
            adam_weight_decay=0.0,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            update_frequency=0,
        )


def test_required_roles_fall_back_to_every_declared_variant() -> None:
    """Without a declaration on training_args the registry is the answer.

    This guards registry fallback after trainability moved out of variant specs.
    """
    registry = ComponentVariantRegistry(SimpleNamespace(trainable_component_names=("transformer",)))
    registry.declare(
        ComponentVariantSpec(
            name="generator",
            storage_mode="full",
            component_routes={"transformer": "transformer"},
        )
    )
    registry.declare(
        ComponentVariantSpec(
            name="fake",
            storage_mode="full",
            component_routes={"transformer": "fake__transformer"},
        )
    )
    trainer = SimpleNamespace(
        training_args=SimpleNamespace(required_trainable_roles=None),
        adapter=SimpleNamespace(component_variant_registry=registry),
    )

    assert BaseTrainer._required_trainable_roles(trainer) == ("generator", "fake")


def test_required_roles_default_to_one_base_variant_without_a_registry() -> None:
    trainer = SimpleNamespace(
        training_args=SimpleNamespace(required_trainable_roles=None),
        adapter=SimpleNamespace(component_variant_registry=None),
    )

    assert BaseTrainer._required_trainable_roles(trainer) == (DEFAULT_BASE_VARIANT,)


def test_the_lone_optimizer_entry_adopts_the_role_name_it_configures() -> None:
    """Every shipped example names its single entry `default`, not after the role.

    `build_optimizer` looks parameters up by `OptimizerArguments.name` while the
    lookup table is keyed by role, so returning the entry under its own name found
    no parameters and every one of those configs died at startup with
    "expected optimizer 'default' to own parameters, received none".
    """
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            optimizer_args=MultiOptimizerArguments.from_dict(
                [{"name": "default", "learning_rate": 1e-4}]
            )
        )
    )

    resolved = BaseTrainer._optimizer_args_for_role(trainer, DEFAULT_BASE_VARIANT)

    assert resolved.name == DEFAULT_BASE_VARIANT
    assert resolved.learning_rate == 1e-4


def test_a_named_optimizer_entry_still_wins_over_the_fallback() -> None:
    """Renaming the lone entry must not mask a config that names its roles."""
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            optimizer_args=MultiOptimizerArguments.from_dict(
                [
                    {"name": "generator", "learning_rate": 1e-4},
                    {"name": "fake", "learning_rate": 2e-4},
                ]
            )
        )
    )

    assert BaseTrainer._optimizer_args_for_role(trainer, "fake").learning_rate == 2e-4

    with pytest.raises(ValueError, match="expected an optimizer configuration named 'missing'"):
        BaseTrainer._optimizer_args_for_role(trainer, "missing")


def test_inactive_populated_adam_moments_remain_unchanged() -> None:
    """A previously trained role must retain both parameters and nonempty optimizer state."""
    bundle, optimizer, roles, coordinator = _runtime(DeterministicAccelerator((True,) * 4))
    _run_phase(coordinator, "fake", ((bundle.fake - 2.0).square(),))
    _run_phase(coordinator, "generator", ((bundle.generator + 1.0).square(),))
    for active, inactive in (("fake", "generator"), ("generator", "fake")):
        parameter = getattr(bundle, inactive)
        before = parameter.detach().clone()
        state = {key: value.clone() for key, value in optimizer.state[parameter].items()}
        assert state["step"].item() >= 1
        assert torch.count_nonzero(state["exp_avg"]).item() > 0
        assert torch.count_nonzero(state["exp_avg_sq"]).item() > 0
        _run_phase(coordinator, active, ((getattr(bundle, active) - 3.0).square(),))
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        assert set(optimizer.state[parameter]) == set(state)
        for key, value in state.items():
            torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
