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

from flow_factory.hparams import ParallelismArguments
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
        parallelism_args=ParallelismArguments(),
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
        parallelism_args=ParallelismArguments(),
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
        parallelism_args=ParallelismArguments(),
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
        parallelism_args=ParallelismArguments(),
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
        parallelism_args=ParallelismArguments(),
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
    monkeypatch.setattr(loader, "seed_data_parallel_replica", lambda *args, **kwargs: None)
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


def _tensor_parallel_accelerator(tp_size: int) -> SimpleNamespace:
    """The least a tensor-parallel gate reads: a degree and an FSDP2 flag."""
    return SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        is_fsdp2=True,
        parallelism_config=SimpleNamespace(tp_size=tp_size),
        state=SimpleNamespace(deepspeed_plugin=None),
    )


def test_a_sequence_parallel_overlay_that_applies_without_a_plan_is_refused_before_loading() -> (
    None
):
    """Neither an explicit request nor an adapter default may find no plan after a load.

    `sequence_parallel: true` against an adapter without an overlay asked for
    something it cannot have; an adapter that enables the overlay by default
    without declaring one is that adapter's bug. The only other place either was
    noticed was `_apply_tensor_parallel`, after the denoiser is in memory.
    """
    from flow_factory.trainers.multirole.backend import validate_tensor_parallel_backend_plan

    def adapter(by_default: bool) -> SimpleNamespace:
        return SimpleNamespace(
            __name__="SilentAdapter",
            tensor_parallel_plan=lambda: {"blocks.0.to_q": object()},
            sequence_parallel_plan=lambda: {},
            sequence_parallel_by_default=by_default,
        )

    optimizers = (AdamWOptimizerArguments(name="default"),)

    with pytest.raises(ValueError, match="declares no sequence-parallel plan"):
        validate_tensor_parallel_backend_plan(
            _tensor_parallel_accelerator(2), adapter(False), optimizers, sequence_parallel=True
        )
    with pytest.raises(ValueError, match="sequence_parallel_by_default is True"):
        validate_tensor_parallel_backend_plan(
            _tensor_parallel_accelerator(2), adapter(True), optimizers, sequence_parallel=None
        )

    # Left unset on an adapter that does not enable it, tensor parallelism alone.
    validate_tensor_parallel_backend_plan(
        _tensor_parallel_accelerator(2), adapter(False), optimizers, sequence_parallel=None
    )


def _run_config(
    tmp_path,
    *,
    tp_size: int,
    heads: int = 24,
    sampler_type: str = "global_batch",
    sequence_parallel=None,
    accelerators: tuple = (),
) -> SimpleNamespace:
    import json

    transformer = tmp_path / "checkpoint" / "transformer"
    transformer.mkdir(parents=True, exist_ok=True)
    (transformer / "config.json").write_text(json.dumps({"num_attention_heads": heads}))
    return SimpleNamespace(
        parallelism_args=ParallelismArguments(tp_size=tp_size, sequence_parallel=sequence_parallel),
        data_args=SimpleNamespace(sampler_type=sampler_type),
        acceleration_args=SimpleNamespace(
            shared=[], rollout=[SimpleNamespace(name=name) for name in accelerators]
        ),
        model_args=SimpleNamespace(model_name_or_path=str(tmp_path / "checkpoint")),
    )


_HEADED_ADAPTER = SimpleNamespace(
    __name__="HeadedAdapter",
    tensor_parallel_head_config=("transformer", ("num_attention_heads",)),
    sequence_parallel_by_default=False,
)


def test_an_indivisible_head_count_is_refused_before_the_weights_load(tmp_path) -> None:
    """SD3.5-Large's 38 heads take tp=2 and nothing above it; Z-Image's 30 the same."""
    from flow_factory.trainers.multirole.backend import validate_tensor_parallel_run_config

    validate_tensor_parallel_run_config(_run_config(tmp_path, tp_size=2, heads=38), _HEADED_ADAPTER)
    with pytest.raises(ValueError, match=r"num_attention_heads=38.*tp_size=4.*\[2, 19, 38\]"):
        validate_tensor_parallel_run_config(
            _run_config(tmp_path, tp_size=4, heads=38), _HEADED_ADAPTER
        )
    # Without tensor parallelism nothing is read.
    validate_tensor_parallel_run_config(_run_config(tmp_path, tp_size=1, heads=38), _HEADED_ADAPTER)


def test_head_counts_nested_in_a_root_config_are_read_by_dotted_key(tmp_path) -> None:
    """SenseNova keeps its counts in the root ``config.json`` under ``llm_config``."""
    import json

    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
    from flow_factory.trainers.multirole.backend import validate_tensor_parallel_run_config

    checkpoint = tmp_path / "sensenova"
    checkpoint.mkdir()
    adapter = SimpleNamespace(
        __name__="SenseNovaAdapter",
        tensor_parallel_head_config=SenseNovaAdapter.tensor_parallel_head_config,
        sequence_parallel_by_default=SenseNovaAdapter.sequence_parallel_by_default,
    )

    def run_config(tp_size: int, llm_config) -> SimpleNamespace:
        (checkpoint / "config.json").write_text(json.dumps({"llm_config": llm_config}))
        return SimpleNamespace(
            parallelism_args=ParallelismArguments(tp_size=tp_size),
            data_args=SimpleNamespace(sampler_type="global_batch"),
            acceleration_args=SimpleNamespace(shared=[], rollout=[]),
            model_args=SimpleNamespace(model_name_or_path=str(checkpoint)),
        )

    shipped = {"num_attention_heads": 32, "num_key_value_heads": 8}
    for tp_size in (2, 4, 8):
        validate_tensor_parallel_run_config(run_config(tp_size, shipped), adapter)
    # 32 query heads take tp=16; the 8 kv heads do not.
    with pytest.raises(
        ValueError, match=r"llm_config\.num_key_value_heads=8.*tp_size=16.*\[2, 4, 8\]"
    ):
        validate_tensor_parallel_run_config(run_config(16, shipped), adapter)
    with pytest.raises(TypeError, match=r"above 'num_attention_heads' in config\.json"):
        validate_tensor_parallel_run_config(run_config(2, 5), adapter)


def test_settings_tensor_parallelism_cannot_honour_are_refused_before_loading(tmp_path) -> None:
    from flow_factory.trainers.multirole.backend import validate_tensor_parallel_run_config

    with pytest.raises(ValueError, match="subgroup_tile"):
        validate_tensor_parallel_run_config(
            _run_config(tmp_path, tp_size=2, sampler_type="subgroup_tile"), _HEADED_ADAPTER
        )
    with pytest.raises(ValueError, match="diffusers_cache"):
        validate_tensor_parallel_run_config(
            _run_config(
                tmp_path, tp_size=2, sequence_parallel=True, accelerators=("diffusers_cache",)
            ),
            _HEADED_ADAPTER,
        )
    # The cache is only refused where the overlay applies.
    validate_tensor_parallel_run_config(
        _run_config(tmp_path, tp_size=2, accelerators=("diffusers_cache",)), _HEADED_ADAPTER
    )


def test_cpu_ram_efficient_loading_is_refused_under_tensor_parallelism() -> None:
    """Rank 0 broadcasts its own shard where every other rank expects the whole tensor."""
    from flow_factory.trainers.multirole.backend import validate_tensor_parallel_backend_plan

    accelerator = _tensor_parallel_accelerator(2)
    accelerator.state.fsdp_plugin = SimpleNamespace(cpu_ram_efficient_loading=True)
    adapter_cls = SimpleNamespace(
        __name__="PlannedAdapter",
        tensor_parallel_plan=lambda: {"blocks.0.to_q": object()},
        sequence_parallel_plan=lambda: {},
        sequence_parallel_by_default=False,
    )

    with pytest.raises(ValueError, match="cpu_ram_efficient_loading"):
        validate_tensor_parallel_backend_plan(
            accelerator, adapter_cls, (AdamWOptimizerArguments(name="default"),)
        )


def test_only_verified_overlays_are_on_by_default() -> None:
    """Every registered adapter either verifies its overlay or leaves it to the run.

    A default-on overlay is a promise that the overlay holds on every input the
    adapter accepts; it is made where the overlay ran end to end on real weights.
    """
    from flow_factory.models.registry import get_model_adapter_class, list_registered_models

    on_by_default = set()
    for model_type in list_registered_models():
        try:
            adapter_cls = get_model_adapter_class(model_type)
        except ImportError:
            # An optional backend (Bagel's flash-attn) is absent in this environment.
            continue
        assert isinstance(adapter_cls.sequence_parallel_by_default, bool), model_type
        if adapter_cls.sequence_parallel_by_default:
            assert adapter_cls.sequence_parallel_plan(), model_type
            on_by_default.add(adapter_cls.__name__)

    assert on_by_default == {"SD3_5Adapter", "Wan2_T2V_Adapter"}


def test_the_two_readings_of_the_tensor_parallel_degree_must_agree() -> None:
    """The degree has two readers by necessity, and one origin by construction.

    Batch geometry reads the config -- `data_parallel_world_size()` is called
    from `Arguments.__post_init__`, before any `Accelerator` exists -- while the
    mesh, the sharding and the samplers read `accelerator.parallelism_config`.
    The second is built from the first, so agreement is structural; this pins it
    anyway, because a divergence is silent: the global batch simply comes out a
    factor of `tp_size` wrong.

    Both directions are checked. A config asking for more than the runtime built
    is the one that would otherwise slip through, because `tp_size <= 1` returns
    early.
    """
    from flow_factory.trainers.multirole.backend import validate_tensor_parallel_backend_plan

    adapter_cls = SimpleNamespace(
        __name__="Adapter",
        tensor_parallel_plan=lambda: {"blocks.0.to_q": object()},
        sequence_parallel_plan=lambda: {"blocks.0": object()},
    )
    optimizers = (AdamWOptimizerArguments(name="default"),)

    def check(runtime: int, declared: object) -> None:
        validate_tensor_parallel_backend_plan(
            _tensor_parallel_accelerator(runtime), adapter_cls, optimizers, False, declared
        )

    check(2, 2)
    check(1, 1)
    # Omitted entirely, for a caller with no parsed config.
    check(2, None)

    with pytest.raises(ValueError, match="disagrees between its two readers"):
        check(2, 1)
    # The runtime built nothing while the config divided the world by 2: caught
    # before the `tp_size <= 1` early return, or it would pass unexamined.
    with pytest.raises(ValueError, match="disagrees between its two readers"):
        check(1, 2)


def test_an_unset_dimension_cannot_be_refilled_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_build_parallelism_config` must hand Accelerate resolved degrees.

    `ParallelismConfig.__post_init__` fills any dimension passed as `None` from
    `PARALLELISM_CONFIG_{DP_REPLICATE,DP_SHARD,TP}_SIZE`. Passing
    `parallelism_args.tp_size` straight through therefore let a config that
    wrote only `dp_shard_size` come back with a tensor-parallel degree the
    derivation never saw -- measured at 1 on the config side and 2 on the
    runtime side -- which also made the derived `dp_shard_size` inconsistent
    with the config it was returned in.
    """
    from flow_factory.hparams import ParallelismArguments

    monkeypatch.setenv("PARALLELISM_CONFIG_TP_SIZE", "2")
    monkeypatch.setenv("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "4")
    monkeypatch.setenv("PARALLELISM_CONFIG_CP_SIZE", "2")
    monkeypatch.setenv("PARALLELISM_CONFIG_SP_SIZE", "2")
    monkeypatch.setattr(loader, "get_world_size", lambda: 8)

    config = SimpleNamespace(parallelism_args=ParallelismArguments(dp_shard_size=8))
    built = loader._build_parallelism_config(config)

    assert built.tp_size == config.parallelism_args.effective_tensor_parallel_size == 1
    assert built.dp_replicate_size == 1
    assert built.dp_shard_size == 8
    assert built.cp_size == built.sp_size == 1


@pytest.mark.parametrize("dimension", ["cp_size", "sp_size"])
def test_unsupported_launcher_dimensions_fail_before_model_loading(
    monkeypatch: pytest.MonkeyPatch, dimension: str
) -> None:
    """An unconfigured public block must not admit a launcher-only CP/SP plan."""
    config = SimpleNamespace(
        mixed_precision="bf16",
        parallelism_args=ParallelismArguments(),
        optimizer_args=(),
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="unsupported-topology"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            trainer_type="grpo",
            required_trainable_roles=None,
        ),
    )
    accelerator = _fsdp_accelerator(2)
    accelerator.num_processes = 4
    sizes = dict(dp_replicate_size=1, dp_shard_size=2, tp_size=1, cp_size=1, sp_size=1)
    sizes[dimension] = 2
    accelerator.parallelism_config = SimpleNamespace(**sizes)
    monkeypatch.setattr(
        loader,
        "get_model_adapter_class",
        lambda _: SimpleNamespace(ddp_find_unused_parameters=False),
    )
    monkeypatch.setattr(loader, "Accelerator", lambda **_: accelerator)
    monkeypatch.setattr(loader, "load_model", lambda **_: pytest.fail("must reject before weights"))
    message = "context parallelism cp_size=2" if dimension == "cp_size" else "independent sequence"
    with pytest.raises(ValueError, match=message):
        loader.load_trainer(config)
