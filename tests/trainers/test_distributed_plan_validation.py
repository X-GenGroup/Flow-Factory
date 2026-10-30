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

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from flow_factory.hparams.optimizer_args import (
    AdamWOptimizerArguments,
    MuonOptimizerArguments,
)
from flow_factory.trainers import loader
from flow_factory.trainers.abc import (
    BaseTrainer,
    configure_deepspeed_micro_batch_size,
    validate_supported_distributed_plan,
)
from flow_factory.trainers.multirole import validate_tensor_parallel_backend_plan


def _accelerator(distributed_type: DistributedType, zero_stage: object = None) -> SimpleNamespace:
    plugin = None if zero_stage is None else SimpleNamespace(zero_stage=zero_stage)
    return SimpleNamespace(
        distributed_type=distributed_type,
        state=SimpleNamespace(deepspeed_plugin=plugin),
    )


def test_zero_three_is_rejected_before_any_weights_load() -> None:
    """The backend validator rejects parameter-sharded DeepSpeed."""
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=3)

    with pytest.raises(ValueError, match="ZeRO-3 is not supported"):
        validate_supported_distributed_plan(accelerator)


def test_loader_rejects_zero_three_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trainer factory must reject ZeRO-3 before constructing an adapter."""
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=3)
    model_load_attempted = False

    class Adapter:
        ddp_find_unused_parameters = False

    def unexpected_model_load(**kwargs: object) -> None:
        del kwargs
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("load_model must not run for DeepSpeed ZeRO-3")

    config = SimpleNamespace(
        mixed_precision="bf16",
        optimizer_args=(),
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="zero3-rejection-test"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            # Read before the Accelerator exists, to size DDP's unused-parameter
            # detection from the algorithm's role count.
            trainer_type="grpo",
            required_trainable_roles=None,
        ),
    )
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(loader, "load_model", unexpected_model_load)

    with pytest.raises(ValueError, match="ZeRO-3 is not supported"):
        loader.load_trainer(config)

    assert model_load_attempted is False


def _fsdp_accelerator(fsdp_version: int) -> SimpleNamespace:
    """Build an Accelerator stub reporting the requested FSDP major version."""
    accelerator = _accelerator(DistributedType.FSDP)
    accelerator.state.fsdp_plugin = SimpleNamespace(
        fsdp_version=fsdp_version,
        activation_checkpointing=False,
        cpu_ram_efficient_loading=False,
    )
    return accelerator


@pytest.mark.parametrize(
    ("accelerator", "error_pattern", "run_name"),
    [
        (
            _accelerator(DistributedType.DEEPSPEED, zero_stage=2),
            "Muon with DeepSpeed is not verified",
            "muon-deepspeed-rejection-test",
        ),
        (
            _fsdp_accelerator(fsdp_version=1),
            "Muon with FSDP1 does not work",
            "muon-fsdp1-rejection-test",
        ),
    ],
)
def test_loader_rejects_unsupported_muon_backend_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    accelerator: SimpleNamespace,
    error_pattern: str,
    run_name: str,
) -> None:
    """The trainer factory must reject unsupported Muon plans before model loading."""
    model_load_attempted = False

    class Adapter:
        ddp_find_unused_parameters = False

    def unexpected_model_load(**kwargs: object) -> None:
        del kwargs
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("load_model must not run for an unsupported Muon backend")

    config = SimpleNamespace(
        mixed_precision="bf16",
        optimizer_args=(MuonOptimizerArguments(name="base"),),
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name=run_name),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            trainer_type="grpo",
            required_trainable_roles=None,
        ),
    )
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(loader, "load_model", unexpected_model_load)

    with pytest.raises(ValueError, match=error_pattern):
        loader.load_trainer(config)

    assert model_load_attempted is False


def test_loader_rejects_unavailable_muon_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported backend still requires the optimizer API before model loading."""
    accelerator = _accelerator(DistributedType.MULTI_GPU)
    model_load_attempted = False

    class Adapter:
        ddp_find_unused_parameters = False

    def unexpected_model_load(**kwargs: object) -> None:
        del kwargs
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("load_model must not run without torch.optim.Muon")

    config = SimpleNamespace(
        mixed_precision="bf16",
        optimizer_args=(MuonOptimizerArguments(name="base"),),
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="muon-api-rejection-test"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            trainer_type="grpo",
            required_trainable_roles=None,
        ),
    )
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(loader, "load_model", unexpected_model_load)

    with pytest.raises(ValueError, match="torch.optim.Muon is unavailable"):
        loader.load_trainer(config)

    assert model_load_attempted is False


@pytest.mark.parametrize("backend_checkpointing", [False, True])
def test_loader_rejects_selective_fsdp2_checkpointing_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    backend_checkpointing: bool,
) -> None:
    """The trainer factory must reject selective FSDP2 checkpointing before loading."""
    accelerator = _fsdp_accelerator(fsdp_version=2)
    accelerator.state.fsdp_plugin.activation_checkpointing = backend_checkpointing
    model_load_attempted = False

    class Adapter:
        ddp_find_unused_parameters = False

    def unexpected_model_load(**kwargs: object) -> None:
        del kwargs
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("load_model must not run for selective FSDP2 checkpointing")

    config = SimpleNamespace(
        mixed_precision="bf16",
        optimizer_args=(),
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="checkpoint-plan-rejection-test"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            trainer_type="grpo",
            required_trainable_roles=None,
            enable_gradient_checkpointing=SimpleNamespace(mode="every_n"),
        ),
    )
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(loader, "load_model", unexpected_model_load)

    with pytest.raises(ValueError, match="cannot preserve selective"):
        loader.load_trainer(config)

    assert model_load_attempted is False


@pytest.mark.parametrize("backend_checkpointing", [False, True])
def test_loader_selects_fsdp2_backend_checkpointing_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    backend_checkpointing: bool,
) -> None:
    """A full model policy must become the safe backend owner before loading."""
    accelerator = _fsdp_accelerator(fsdp_version=2)
    accelerator.state.fsdp_plugin.activation_checkpointing = backend_checkpointing
    observed_plan = []
    adapter = object()

    class Adapter:
        ddp_find_unused_parameters = False

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    config = SimpleNamespace(
        mixed_precision="bf16",
        optimizer_args=(),
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="checkpoint-plan-owner-test"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            trainer_type="grpo",
            required_trainable_roles=None,
            enable_gradient_checkpointing=True,
        ),
    )

    def load_model_with_resolved_plan(**kwargs: object) -> object:
        del kwargs
        observed_plan.append(
            (
                config.training_args.enable_gradient_checkpointing,
                accelerator.state.fsdp_plugin.activation_checkpointing,
            )
        )
        return adapter

    monkeypatch.setattr(loader, "get_trainer_class", lambda trainer_type: FakeTrainer)
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(loader, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "reconcile_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "load_model", load_model_with_resolved_plan)

    trainer = loader.load_trainer(config)

    assert observed_plan == [(False, True)]
    assert trainer.kwargs["adapter"] is adapter


@pytest.mark.parametrize("zero_stage", [1, 2])
def test_supported_deepspeed_stages_pass(zero_stage: int) -> None:
    """ZeRO-1 and ZeRO-2 are the supported DeepSpeed configurations."""
    validate_supported_distributed_plan(_accelerator(DistributedType.DEEPSPEED, zero_stage))


@pytest.mark.parametrize(
    "distributed_type",
    [DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.FSDP],
)
def test_non_deepspeed_plans_pass(distributed_type: DistributedType) -> None:
    """DDP and FSDP carry no DeepSpeed plugin and are supported unchanged."""
    validate_supported_distributed_plan(_accelerator(distributed_type))


def test_deepspeed_without_a_plugin_is_not_rejected() -> None:
    """A DeepSpeed distributed type with no plugin has no stage to reject."""
    validate_supported_distributed_plan(_accelerator(DistributedType.DEEPSPEED))


def test_deepspeed_micro_batch_size_is_set_for_custom_train_loader() -> None:
    accelerator = _accelerator(DistributedType.DEEPSPEED, zero_stage=2)
    accelerator.state.deepspeed_plugin.deepspeed_config = {}

    configure_deepspeed_micro_batch_size(accelerator, per_device_batch_size=3)

    assert (
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == 3
    )


def test_muon_with_deepspeed_is_rejected_as_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Muon runs inside a composite; DeepSpeed rebuilds its own optimizer wrapper."""
    trainer = SimpleNamespace(accelerator=_accelerator(DistributedType.DEEPSPEED, zero_stage=2))

    with pytest.raises(ValueError, match="Muon with DeepSpeed is not verified"):
        BaseTrainer._validate_optimizer_backend(trainer, (MuonOptimizerArguments(name="base"),))

    # AdamW is unaffected, and Muon is fine on the backends that preserve parameter rank.
    BaseTrainer._validate_optimizer_backend(trainer, (AdamWOptimizerArguments(name="base"),))
    fsdp2_trainer = SimpleNamespace(accelerator=_fsdp_accelerator(fsdp_version=2))
    monkeypatch.setattr(torch.optim, "Muon", object(), raising=False)
    BaseTrainer._validate_optimizer_backend(fsdp2_trainer, (MuonOptimizerArguments(name="base"),))


def test_muon_with_fsdp1_is_rejected_before_a_rollout_is_paid_for() -> None:
    """FSDP1 flattens to 1D, so Muon would only fail after the first full rollout."""
    trainer = SimpleNamespace(accelerator=_fsdp_accelerator(fsdp_version=1))

    with pytest.raises(ValueError, match="Muon with FSDP1 does not work"):
        BaseTrainer._validate_optimizer_backend(trainer, (MuonOptimizerArguments(name="base"),))

    # FSDP1 stays available to every optimizer that accepts a flattened parameter.
    BaseTrainer._validate_optimizer_backend(trainer, (AdamWOptimizerArguments(name="base"),))


def test_no_zero_three_profile_is_shipped() -> None:
    """A shipped profile would invite a configuration the trainer refuses."""
    config_dir = Path(__file__).resolve().parents[2] / "config" / "deepspeed"

    assert config_dir.is_dir()
    assert not (config_dir / "deepspeed_zero3.yaml").exists()


def test_deepspeed_gradient_clipping_is_wired_from_the_configured_norm() -> None:
    """DeepSpeed clips inside its engine and ignores the value passed at the call site.

    accelerate reads the threshold from this environment variable when building the
    plugin, so leaving it unset ships an unresolved "auto" and max_grad_norm never
    takes effect on that backend.
    """
    source = inspect.getsource(loader.load_trainer)
    assert "ACCELERATE_GRADIENT_CLIPPING" in source
    assert source.index("ACCELERATE_GRADIENT_CLIPPING") < source.index("accelerator = Accelerator(")


def _prepared_trainer(distributed_type: DistributedType, local: int, others: int = 0):
    """Trainer stub owning ``local`` trainable elements, with ``others`` on the peers.

    ``reduce`` stands in for the collective: the guard asks whether ANY rank holds
    trainable elements, so the stub adds what the peers would report.
    """
    accelerator = _accelerator(distributed_type)
    accelerator.device = torch.device("cpu")
    accelerator.num_processes = 2
    accelerator.reduce = lambda tensor, reduction="sum": tensor + float(others)
    parameter = torch.nn.Parameter(torch.zeros(local or 1))
    parameter.requires_grad = local > 0
    return (
        SimpleNamespace(
            accelerator=accelerator,
            model_bundle=SimpleNamespace(parameters=lambda: iter([parameter])),
        ),
        BaseTrainer,
    )


def test_a_prepared_root_no_rank_can_train_is_rejected() -> None:
    """Nothing anywhere requires a gradient, so an optimizer step would change nothing."""
    trainer, BaseTrainer = _prepared_trainer(DistributedType.FSDP, local=0, others=0)

    with pytest.raises(RuntimeError, match="received 0 across all 2 rank"):
        BaseTrainer._validate_trainable_parameters_survived_prepare(trainer)


def test_a_rank_holding_no_shard_of_the_adapter_is_accepted() -> None:
    """FSDP splits by byte range: a rank can own none of a small adapter and be healthy."""
    trainer, BaseTrainer = _prepared_trainer(DistributedType.FSDP, local=0, others=2048)
    BaseTrainer._validate_trainable_parameters_survived_prepare(trainer)

    trainer, BaseTrainer = _prepared_trainer(DistributedType.DEEPSPEED, local=8)
    BaseTrainer._validate_trainable_parameters_survived_prepare(trainer)


def test_tdm_r1_fsdp1_disables_incompatible_activation_checkpointing() -> None:
    plugin = SimpleNamespace(fsdp_version=1, activation_checkpointing=True)
    disabled = []
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=plugin),
        ),
        training_args=SimpleNamespace(
            trainer_type="tdm-r1",
            enable_gradient_checkpointing=True,
        ),
        adapter=SimpleNamespace(disable_gradient_checkpointing=lambda: disabled.append(True)),
    )

    BaseTrainer._apply_backend_checkpointing_constraints(trainer)

    assert disabled == [True]
    assert trainer.training_args.enable_gradient_checkpointing is False
    assert plugin.activation_checkpointing is False


@pytest.mark.parametrize(
    "checkpoint_policy",
    [True, SimpleNamespace(mode="full")],
)
@pytest.mark.parametrize("backend_checkpointing", [False, True])
def test_fsdp2_disables_model_checkpointing_and_keeps_backend_checkpointing(
    checkpoint_policy: object,
    backend_checkpointing: bool,
) -> None:
    plugin = SimpleNamespace(
        fsdp_version=2,
        activation_checkpointing=backend_checkpointing,
    )
    disabled = []
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=plugin),
        ),
        training_args=SimpleNamespace(
            trainer_type="tdm-r1",
            enable_gradient_checkpointing=checkpoint_policy,
        ),
        adapter=SimpleNamespace(disable_gradient_checkpointing=lambda: disabled.append(True)),
    )

    BaseTrainer._apply_backend_checkpointing_constraints(trainer)

    assert disabled == [True]
    assert trainer.training_args.enable_gradient_checkpointing is False
    assert plugin.activation_checkpointing is True


def test_fsdp1_keeps_model_checkpointing_and_disables_nested_backend_checkpointing() -> None:
    plugin = SimpleNamespace(fsdp_version=1, activation_checkpointing=True)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=plugin),
        ),
        training_args=SimpleNamespace(
            trainer_type="grpo",
            enable_gradient_checkpointing=True,
        ),
        adapter=SimpleNamespace(
            disable_gradient_checkpointing=lambda: pytest.fail(
                "FSDP1 must keep model checkpointing enabled"
            )
        ),
    )

    BaseTrainer._apply_backend_checkpointing_constraints(trainer)

    assert trainer.training_args.enable_gradient_checkpointing is True
    assert plugin.activation_checkpointing is False


@pytest.mark.parametrize("backend_checkpointing", [False, True])
def test_fsdp2_rejects_selective_model_checkpointing(
    backend_checkpointing: bool,
) -> None:
    plugin = SimpleNamespace(
        fsdp_version=2,
        activation_checkpointing=backend_checkpointing,
    )
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=plugin),
        ),
        training_args=SimpleNamespace(
            trainer_type="grpo",
            enable_gradient_checkpointing=SimpleNamespace(mode="every_n"),
        ),
        adapter=SimpleNamespace(
            disable_gradient_checkpointing=lambda: pytest.fail(
                "selective checkpointing must fail before mutating the adapter"
            )
        ),
    )

    with pytest.raises(ValueError, match="cannot preserve selective"):
        BaseTrainer._apply_backend_checkpointing_constraints(trainer)

    assert trainer.training_args.enable_gradient_checkpointing.mode == "every_n"
    assert plugin.activation_checkpointing is backend_checkpointing


def test_fsdp2_keeps_backend_checkpointing_when_model_policy_is_disabled() -> None:
    plugin = SimpleNamespace(fsdp_version=2, activation_checkpointing=True)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=plugin),
        ),
        training_args=SimpleNamespace(
            trainer_type="grpo",
            enable_gradient_checkpointing=False,
        ),
        adapter=SimpleNamespace(),
    )

    BaseTrainer._apply_backend_checkpointing_constraints(trainer)

    assert plugin.activation_checkpointing is True


def _tp_accelerator(
    distributed_type: DistributedType,
    tp_size: object = 2,
    fsdp_version: object = None,
    multi_device: bool = False,
) -> SimpleNamespace:
    """Build an Accelerator stub requesting the given tensor-parallel degree."""
    parallelism_config = None if tp_size is None else SimpleNamespace(tp_size=tp_size)
    plugin = None if fsdp_version is None else SimpleNamespace(fsdp_version=fsdp_version)
    return SimpleNamespace(
        distributed_type=distributed_type,
        parallelism_config=parallelism_config,
        multi_device=multi_device,
        state=SimpleNamespace(deepspeed_plugin=None, fsdp_plugin=plugin),
    )


def test_tensor_parallel_without_a_parallelism_config_is_not_rejected() -> None:
    """An Accelerator with no ParallelismConfig requests no tensor parallelism."""
    validate_tensor_parallel_backend_plan(_tp_accelerator(DistributedType.FSDP, tp_size=None))


@pytest.mark.parametrize(
    ("distributed_type", "fsdp_version"),
    [(DistributedType.FSDP, 2), (DistributedType.NO, None), (DistributedType.MULTI_GPU, None)],
)
def test_disabled_tensor_parallel_is_not_rejected(
    distributed_type: DistributedType, fsdp_version: object
) -> None:
    """``tp_size=1`` is the default and must not constrain the backend."""
    accelerator = _tp_accelerator(distributed_type, tp_size=1, fsdp_version=fsdp_version)
    validate_tensor_parallel_backend_plan(accelerator)


def test_fsdp1_with_tensor_parallel_is_rejected() -> None:
    """FSDP1 cannot host the DTensor parameters ``_prepare_tp`` installs."""
    accelerator = _tp_accelerator(DistributedType.FSDP, tp_size=2, fsdp_version=1)

    with pytest.raises(ValueError, match="not supported with FSDP1"):
        validate_tensor_parallel_backend_plan(accelerator)


def test_fsdp2_tensor_parallel_requires_replicate_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accelerate's FSDP2 TP branch imports ReplicateParallel unconditionally."""
    from transformers.integrations import tensor_parallel

    monkeypatch.delattr(tensor_parallel, "ReplicateParallel", raising=False)
    accelerator = _tp_accelerator(DistributedType.FSDP, tp_size=2, fsdp_version=2)

    with pytest.raises(ValueError, match="ReplicateParallel"):
        validate_tensor_parallel_backend_plan(accelerator)


def test_fsdp2_tensor_parallel_passes_with_replicate_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transformers build exposing ReplicateParallel clears the FSDP2 TP plan."""
    from transformers.integrations import tensor_parallel

    monkeypatch.setattr(tensor_parallel, "ReplicateParallel", object, raising=False)
    accelerator = _tp_accelerator(DistributedType.FSDP, tp_size=2, fsdp_version=2)

    validate_tensor_parallel_backend_plan(accelerator)


@pytest.mark.parametrize(
    ("distributed_type", "multi_device"),
    [(DistributedType.DEEPSPEED, False), (DistributedType.MULTI_GPU, True)],
)
def test_tensor_parallel_supported_backends_pass(
    distributed_type: DistributedType, multi_device: bool
) -> None:
    """DeepSpeed and multi-device backends accept a tensor-parallel degree."""
    accelerator = _tp_accelerator(distributed_type, tp_size=2, multi_device=multi_device)

    validate_tensor_parallel_backend_plan(accelerator)


def test_tensor_parallel_unsupported_backend_is_rejected() -> None:
    """A single-process backend has no mesh for tensor parallelism to shard across."""
    accelerator = _tp_accelerator(DistributedType.NO, tp_size=2)

    with pytest.raises(ValueError, match="requires DistributedType.FSDP"):
        validate_tensor_parallel_backend_plan(accelerator)


def test_loader_rejects_fsdp1_tensor_parallel_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trainer factory must reject FSDP1 + TP before constructing an adapter."""
    accelerator = _tp_accelerator(DistributedType.FSDP, tp_size=2, fsdp_version=1)
    model_load_attempted = False

    class Adapter:
        ddp_find_unused_parameters = False

    def unexpected_model_load(**kwargs: object) -> None:
        del kwargs
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("load_model must not run for FSDP1 tensor parallelism")

    config = SimpleNamespace(
        mixed_precision="bf16",
        optimizer_args=(AdamWOptimizerArguments(name="base"),),
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="tp-fsdp1-rejection-test"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            trainer_type="grpo",
            required_trainable_roles=None,
        ),
    )
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(loader, "load_model", unexpected_model_load)

    with pytest.raises(ValueError, match="not supported with FSDP1"):
        loader.load_trainer(config)

    assert model_load_attempted is False
