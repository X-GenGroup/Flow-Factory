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

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Tuple

import pytest
import yaml
from accelerate.utils import DistributedType

from flow_factory.hparams import ParallelismArguments
from flow_factory.hparams.training_args import TrainingArguments
from flow_factory.trainers import loader
from flow_factory.trainers.loader import _requires_ddp_unused_parameter_detection


class DefaultAdapter:
    """Represent an adapter that uses the legacy DDP default."""

    ddp_find_unused_parameters = False


class UnusedParameterAdapter:
    """Represent an adapter that explicitly enables unused-parameter detection."""

    ddp_find_unused_parameters = True


@dataclass
class CustomMultiRoleTrainingArguments(TrainingArguments):
    """Expose roles through the direct-path training-arguments contract."""

    required_trainable_roles: ClassVar[Tuple[str, ...]] = ("generator", "fake")


def _config(
    *,
    trainer_type: str = "grpo",
    required_trainable_roles: Tuple[str, ...] = ("generator",),
) -> SimpleNamespace:
    return SimpleNamespace(
        mixed_precision="no",
        optimizer_args=(),
        parallelism_args=ParallelismArguments(),
        model_args=SimpleNamespace(model_type="tiny"),
        training_args=SimpleNamespace(
            trainer_type=trainer_type,
            required_trainable_roles=required_trainable_roles,
            gradient_accumulation_steps=1,
            seed=7,
            max_grad_norm=1.0,
        ),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="multirole-loader"),
    )


@pytest.mark.parametrize(
    "config,adapter_cls,expected",
    [
        (_config(), DefaultAdapter, False),
        (_config(required_trainable_roles=("generator", "fake")), DefaultAdapter, True),
        (_config(), UnusedParameterAdapter, True),
    ],
)
def test_loader_resolves_ddp_unused_parameter_detection(
    config: SimpleNamespace,
    adapter_cls: type,
    expected: bool,
) -> None:
    assert _requires_ddp_unused_parameter_detection(config, adapter_cls) is expected


def test_loader_honors_direct_path_training_arguments_roles() -> None:
    trainer_type = f"{__name__}.CustomMultiRoleTrainingArguments"
    config = _config(trainer_type=trainer_type)
    del config.training_args.required_trainable_roles

    assert _requires_ddp_unused_parameter_detection(config, DefaultAdapter) is True


def test_fsdp2_config_uses_installed_original_parameter_contract() -> None:
    config_path = Path(__file__).parents[2] / "config/accelerate_configs/fsdp2.yaml"
    fsdp_config = yaml.safe_load(config_path.read_text())["fsdp_config"]

    assert fsdp_config["fsdp_version"] == 2
    assert "fsdp_forward_prefetch" not in fsdp_config
    assert "fsdp_use_orig_params" not in fsdp_config


def test_loader_builds_ddp_handler_before_one_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    handler = object()
    adapter = object()
    config = _config(required_trainable_roles=("generator", "fake"))

    def fake_get_model_adapter_class(identifier: str) -> type:
        events.append(("adapter_class", identifier))
        return DefaultAdapter

    def fake_get_training_args_class(identifier: str) -> type:
        events.append(("training_args_class", identifier))
        return TrainingArguments

    def fake_ddp_kwargs(*, find_unused_parameters: bool) -> object:
        events.append(("ddp_kwargs", find_unused_parameters))
        return handler

    class FakeAccelerator:
        # `load_trainer` validates the distributed plan on the fresh Accelerator, so the
        # fake has to report one; DDP is the plan this test is describing.
        distributed_type = DistributedType.MULTI_GPU

        def __init__(self, **kwargs: object) -> None:
            events.append(("accelerator", kwargs["kwargs_handlers"]))

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(loader, "get_model_adapter_class", fake_get_model_adapter_class)
    monkeypatch.setattr(loader, "get_training_args_class", fake_get_training_args_class)
    monkeypatch.setattr(loader, "DistributedDataParallelKwargs", fake_ddp_kwargs)
    monkeypatch.setattr(loader, "Accelerator", FakeAccelerator)
    monkeypatch.setattr(loader, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "reconcile_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "load_model", lambda **kwargs: adapter)
    monkeypatch.setattr(loader, "get_trainer_class", lambda identifier: FakeTrainer)

    trainer = loader.load_trainer(config)

    assert [event[0] for event in events] == [
        "adapter_class",
        "training_args_class",
        "ddp_kwargs",
        "accelerator",
    ]
    assert events[2] == ("ddp_kwargs", True)
    assert events[3][1] == [handler]
    assert trainer.kwargs["adapter"] is adapter
