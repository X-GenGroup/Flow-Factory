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

"""Tests for deterministic realized trainer resume identities."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
from accelerate.utils import DistributedType
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from flow_factory.contracts import NegativePromptPolicy
from flow_factory.contracts.execution import OFFLINE_EXECUTION_CONTRACT
from flow_factory.data_utils.dataset import GeneralDataset
from flow_factory.data_utils.multi_source import (
    MultiSourceTrainDataLoader,
    WeightedSourceBatchScheduler,
)
from flow_factory.data_utils.offline_dataset import OfflineDataset
from flow_factory.data_utils.sampler import DistributedKRepeatSampler
from flow_factory.hparams.optimizer_args import (
    AdamWOptimizerArguments,
    MultiOptimizerArguments,
)
from flow_factory.models.pipeline_contracts import image_output_contract
from flow_factory.trainers.common.runtime_identity import (
    build_trainer_runtime_identity,
)


@dataclass
class _Record:
    """Expose the stable parameter ownership fields used by the identity builder."""

    component_name: str
    parameter_name: str
    parameter: torch.nn.Parameter


class _Registry:
    """Return ordered role-owned parameter records."""

    def __init__(self, records: dict[str, tuple[_Record, ...]]) -> None:
        self.records = records

    def parameter_records(self, role_name: str) -> tuple[_Record, ...]:
        """Return records for one role."""
        return self.records[role_name]


class _Adapter:
    """Carry the realized registry without model-loading behavior."""


class _ConfigBlock:
    """Expose resolved config values through the production ``to_dict`` contract."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached resolved-config mapping."""
        return dict(self.__dict__)


class _FingerprintDataset:
    """Minimal Hugging Face-like dataset carrying a stable content fingerprint."""

    def __init__(self, fingerprint: str, length: int = 8) -> None:
        self._fingerprint = fingerprint
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


class _SchedulerGroup:
    """Provide the realized scheduler-group surface hashed by the trainer."""

    names = ("latent",)
    primary_name = "latent"

    def __getitem__(self, name: str) -> SimpleNamespace:
        assert name == "latent"
        return SimpleNamespace(dynamics_type="Flow-ODE")


def _online_loader(
    *,
    fingerprint: str = "online-content-v1",
    seed: int = 17,
    rank: int = 0,
) -> DataLoader:
    """Build the same rank-aware loader geometry used by online acquisition."""
    dataset = GeneralDataset.__new__(GeneralDataset)
    dataset.processed_dataset = _FingerprintDataset(fingerprint)
    sampler = DistributedKRepeatSampler(
        dataset,
        batch_size=2,
        group_size=2,
        unique_sample_num=4,
        num_replicas=2,
        rank=rank,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=GeneralDataset.collate_fn)


def _multi_source_online_loader(
    *,
    source_order: tuple[str, ...] = ("train-a", "train-b"),
    source_name_to_id: dict[str, int] | None = None,
    batches_per_block: int = 1,
) -> MultiSourceTrainDataLoader:
    """Build an identity-only multi-source loader with ordered training names."""
    loaders = {
        source_name: _online_loader(fingerprint=f"content:{source_name}")
        for source_name in source_order
    }
    scheduler = WeightedSourceBatchScheduler(
        {
            source_name: loader.batch_sampler.num_batches_per_epoch
            for source_name, loader in loaders.items()
        },
        seed=42,
        batches_per_block=batches_per_block,
    )
    return MultiSourceTrainDataLoader(
        loaders,
        scheduler,
        source_name_to_id=source_name_to_id,
        batch_size=2,
    )


def _prepared_eval_loader(
    *,
    fingerprint: str = "eval-content-v1",
    rank: int = 0,
) -> DataLoaderShard:
    """Build the Accelerate wrapper shape used by realized eval loaders."""
    dataset = _FingerprintDataset(fingerprint, length=8)
    source = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_sampler = BatchSamplerShard(
        source.batch_sampler,
        num_processes=2,
        process_index=rank,
    )
    return DataLoaderShard(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=source.collate_fn,
    )


def _configure_evaluation(
    trainer: "_Trainer",
    loaders: tuple[tuple[str, DataLoader], ...],
) -> None:
    """Attach ordered realized eval loaders and matching dataset configs."""
    trainer.eval_dataloaders = dict(loaders)
    trainer._eval_dataset_configs = {
        name: _ConfigBlock(
            name=name,
            source_id=index,
            eval={"enabled": True, "guidance_scale": None},
        )
        for index, (name, _) in enumerate(loaders)
    }


class _Trainer:
    """Provide the identity builder's structural trainer interface."""

    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def __init__(
        self,
        *,
        width: int = 2,
        learning_rate: float = 1e-3,
        beta: float = 0.1,
        weighting_scheme: str = "uniform",
        timestep_range: tuple[float, float] = (0.0, 1.0),
        seed: int = 42,
        max_epochs: int = 2,
        log_every: int = 10,
        max_grad_norm: float = 1.0,
        update_frequency: int = 1,
        accepts_image_input: bool = False,
        dataloader: DataLoader | None = None,
    ) -> None:
        parameter = torch.nn.Parameter(torch.zeros(width, width))
        self.adapter = _Adapter()
        self.adapter.effective_pipeline_io_contract = image_output_contract(
            negative_prompt=NegativePromptPolicy.OPTIONAL,
            input_image_min_count=0 if accepts_image_input else None,
            input_image_max_count=1 if accepts_image_input else None,
        )
        self.adapter.component_variant_registry = _Registry(
            {"base": (_Record("transformer", "weight", parameter),)}
        )
        self.adapter.scheduler_group = _SchedulerGroup()
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameter],
                    "role_name": "base",
                    "lr": learning_rate,
                }
            ]
        )
        self.model_args = _ConfigBlock(
            model_type="tiny",
            model_name_or_path="tests/tiny",
            finetune_type="full",
            target_components=["transformer"],
            forward_variant="epsilon",
            resume_path="checkpoint-source",
            resume_type="state",
        )
        self.training_args = _ConfigBlock(
            trainer_type="sft",
            beta=beta,
            weighting_scheme=weighting_scheme,
            timestep_range=timestep_range,
            seed=seed,
            max_epochs=max_epochs,
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
        )
        self.config = SimpleNamespace(
            scheduler_args=_ConfigBlock(
                dynamics_type="Flow-ODE",
                seed=9,
                num_sde_steps=0,
            ),
            acceleration_args=_ConfigBlock(shared=[], rollout=[]),
            reward_args=_ConfigBlock(
                rewards=[{"name": "quality", "reward_model": "tests.Reward", "weight": 1.0}]
            ),
            eval_args=_ConfigBlock(eval_freq=1, guidance_scale=4.0),
            log_args=_ConfigBlock(log_every=log_every, save_freq=1),
            optimizer_args=MultiOptimizerArguments(
                optimizer_configs=[
                    AdamWOptimizerArguments(
                        name="base",
                        learning_rate=learning_rate,
                        weight_decay=1e-2,
                        max_grad_norm=max_grad_norm,
                        update_frequency=update_frequency,
                    )
                ]
            ),
        )
        self.reward_args = self.config.reward_args
        self.eval_args = self.config.eval_args
        self.eval_reward_args = self.reward_args
        self.eval_dataloaders: dict[str, DataLoader] = {}
        self._eval_dataset_configs: dict[str, _ConfigBlock] = {}
        self.dataloader = _online_loader() if dataloader is None else dataloader
        self.accelerator = SimpleNamespace(
            num_processes=2,
            distributed_type=DistributedType.NO,
            mixed_precision="no",
            scaler=None,
            state=SimpleNamespace(fsdp_plugin=None, deepspeed_plugin=None),
        )

    def _required_trainable_roles(self) -> tuple[str, ...]:
        """Return the realized optimizer-role order."""
        return ("base",)

    def _optimizer_args_for_role(self, role_name: str) -> AdamWOptimizerArguments:
        """Return the realized optimizer arguments for one role."""
        optimizer_args = self.config.optimizer_args.get_by_name(role_name)
        assert optimizer_args is not None
        return optimizer_args


def test_identity_covers_concrete_types_model_roles_and_world_size() -> None:
    """Human-readable fields and opaque schemas cover separate compatibility axes."""
    trainer = _Trainer()

    identity = build_trainer_runtime_identity(trainer)

    assert identity["trainer"].endswith("._Trainer")
    assert identity["adapter"].endswith("._Adapter")
    assert identity["algorithm"] == "sft"
    assert identity["model"] == "tiny:tests/tiny"
    assert identity["finetune_type"] == "full"
    assert identity["optimizer_roles"] == ("base",)
    assert identity["world_size"] == 2
    assert identity["distributed_type"] == "NO"
    assert identity["mixed_precision"] == "no"
    assert identity["gradient_scaler"] == "none"
    assert len(identity["backend_schema_digest"]) == 64
    assert len(identity["parameter_schema_digest"]) == 64
    assert len(identity["optimizer_schema_digest"]) == 64
    assert len(identity["execution_contract_digest"]) == 64
    assert len(identity["data_contract_digest"]) == 64


def test_parameter_and_optimizer_schema_changes_have_independent_digests() -> None:
    """Shape and optimizer configuration drift are both exact-resume incompatibilities."""
    baseline = build_trainer_runtime_identity(_Trainer(width=2, learning_rate=1e-3))
    changed_shape = build_trainer_runtime_identity(_Trainer(width=3, learning_rate=1e-3))
    changed_optimizer = build_trainer_runtime_identity(_Trainer(width=2, learning_rate=2e-3))

    assert changed_shape["parameter_schema_digest"] != baseline["parameter_schema_digest"]
    assert changed_optimizer["parameter_schema_digest"] == baseline["parameter_schema_digest"]
    assert changed_optimizer["optimizer_schema_digest"] != baseline["optimizer_schema_digest"]


def test_effective_pipeline_contract_changes_execution_identity() -> None:
    """Checkpoint-realized input semantics are exact-resume boundaries."""
    baseline = build_trainer_runtime_identity(_Trainer(accepts_image_input=False))
    changed = build_trainer_runtime_identity(_Trainer(accepts_image_input=True))

    assert changed["execution_contract_digest"] != baseline["execution_contract_digest"]
    assert changed["data_contract_digest"] == baseline["data_contract_digest"]


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (("max_grad_norm", 0.5), ("update_frequency", 3)),
)
def test_non_param_group_optimizer_semantics_change_execution_contract_digest(
    field: str,
    changed_value: Any,
) -> None:
    """Gradient clipping and role cadence cannot drift across an exact resume."""
    baseline = build_trainer_runtime_identity(_Trainer())
    changed = build_trainer_runtime_identity(_Trainer(**{field: changed_value}))

    assert changed["optimizer_schema_digest"] == baseline["optimizer_schema_digest"]
    assert changed["execution_contract_digest"] != baseline["execution_contract_digest"]


def test_optimizer_schema_rejects_parameters_outside_rebound_registry() -> None:
    """A prepared optimizer cannot restore state onto an unowned physical parameter."""
    trainer = _Trainer()
    trainer.optimizer.param_groups[0]["params"].append(torch.nn.Parameter(torch.zeros(1)))

    with pytest.raises(ValueError, match="not owned by the rebound variant registry"):
        build_trainer_runtime_identity(trainer)


@pytest.mark.parametrize("zero_stage", (1, 2))
def test_deepspeed_optimizer_schema_maps_flat_partitions_through_logical_groups(
    zero_stage: int,
) -> None:
    """ZeRO flat optimizer partitions retain their logical variant ownership."""

    def identity(partition_widths: tuple[int, int]) -> dict[str, Any]:
        trainer = _Trainer()
        fake_parameter = torch.nn.Parameter(torch.ones(3, 3))
        trainer.adapter.component_variant_registry.records["fake"] = (
            _Record("transformer", "fake_weight", fake_parameter),
        )
        trainer.optimizer.add_param_group(
            {"params": [fake_parameter], "role_name": "fake", "lr": 2e-3}
        )
        trainer.config.optimizer_args = MultiOptimizerArguments(
            optimizer_configs=[
                AdamWOptimizerArguments(name="base", learning_rate=1e-3),
                AdamWOptimizerArguments(name="fake", learning_rate=2e-3),
            ]
        )
        trainer._required_trainable_roles = lambda: ("base", "fake")

        basic_optimizer = trainer.optimizer
        logical_groups = [list(group["params"]) for group in basic_optimizer.param_groups]
        for group, width in zip(basic_optimizer.param_groups, partition_widths):
            group["params"] = [torch.nn.Parameter(torch.zeros(width), requires_grad=True)]
        zero_optimizer = SimpleNamespace(
            bit16_groups=logical_groups,
            optimizer=basic_optimizer,
        )
        trainer.optimizer = SimpleNamespace(
            param_groups=basic_optimizer.param_groups,
            optimizer=zero_optimizer,
        )
        trainer.accelerator.distributed_type = DistributedType.DEEPSPEED
        trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(
            zero_stage=zero_stage,
            deepspeed_config={"zero_optimization": {"stage": zero_stage}},
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            is_train_batch_min=True,
        )
        return build_trainer_runtime_identity(trainer)

    rank_zero = identity((5, 7))
    rank_one = identity((6, 8))

    assert rank_zero["parameter_schema_digest"] == rank_one["parameter_schema_digest"]
    assert rank_zero["optimizer_schema_digest"] == rank_one["optimizer_schema_digest"]


def test_deepspeed_optimizer_schema_rejects_foreign_logical_parameters() -> None:
    """ZeRO logical groups must still exhaust only registry-owned parameters."""
    trainer = _Trainer()
    trainer.accelerator.distributed_type = DistributedType.DEEPSPEED
    trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(zero_stage=2)
    trainer.optimizer.optimizer = SimpleNamespace(
        bit16_groups=[[torch.nn.Parameter(torch.zeros(1))]]
    )

    with pytest.raises(ValueError, match="not owned by the rebound variant registry"):
        build_trainer_runtime_identity(trainer)


@pytest.mark.parametrize(
    ("logical_groups", "error_type", "message"),
    (
        (None, TypeError, "requires the logical model parameter groups"),
        ([], ValueError, "logical and partitioned group counts to match"),
    ),
)
def test_deepspeed_optimizer_schema_rejects_invalid_logical_groups(
    logical_groups: list[list[torch.nn.Parameter]] | None,
    error_type: type[Exception],
    message: str,
) -> None:
    """ZeRO identity fails closed when its logical parameter seam is unavailable."""
    trainer = _Trainer()
    trainer.accelerator.distributed_type = DistributedType.DEEPSPEED
    trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(zero_stage=2)
    trainer.optimizer.optimizer = SimpleNamespace()
    if logical_groups is not None:
        trainer.optimizer.optimizer.bit16_groups = logical_groups

    with pytest.raises(error_type, match=message):
        build_trainer_runtime_identity(trainer)


def test_backend_and_precision_drift_change_the_resume_identity() -> None:
    """Backend checkpoint layouts are rejected before prepared-state mutation."""
    baseline_trainer = _Trainer()
    baseline = build_trainer_runtime_identity(baseline_trainer)

    fsdp_trainer = _Trainer()
    fsdp_trainer.accelerator.distributed_type = DistributedType.FSDP
    fsdp_trainer.accelerator.state.fsdp_plugin = SimpleNamespace(
        fsdp_version=2,
        state_dict_type="SHARDED_STATE_DICT",
        state_dict_config={"offload_to_cpu": True, "rank0_only": False},
        optim_state_dict_config={"offload_to_cpu": True, "rank0_only": False},
        sharding_strategy="FULL_SHARD",
        reshard_after_forward=True,
        use_orig_params=True,
        cpu_offload=False,
        mixed_precision_policy=None,
        backward_prefetch=None,
        forward_prefetch=False,
        auto_wrap_policy=None,
        transformer_cls_names_to_wrap=None,
        min_num_params=None,
        limit_all_gathers=True,
        sync_module_states=True,
        cpu_ram_efficient_loading=True,
        activation_checkpointing=False,
    )
    fsdp = build_trainer_runtime_identity(fsdp_trainer)

    fp16_trainer = _Trainer()
    fp16_trainer.accelerator.mixed_precision = "fp16"
    fp16_trainer.accelerator.scaler = SimpleNamespace()
    fp16 = build_trainer_runtime_identity(fp16_trainer)

    assert fsdp["distributed_type"] == "FSDP"
    assert fsdp["backend_schema_digest"] != baseline["backend_schema_digest"]
    assert fp16["mixed_precision"] == "fp16"
    assert fp16["gradient_scaler"].endswith("SimpleNamespace")


def test_fsdp_wrap_and_checkpoint_topology_drift_changes_backend_digest() -> None:
    """FSDP wrap units and state-dict policy are exact-resume identity fields."""

    def identity(*, min_num_params: int, rank0_only: bool) -> dict[str, Any]:
        trainer = _Trainer()
        trainer.accelerator.distributed_type = DistributedType.FSDP
        trainer.accelerator.state.fsdp_plugin = SimpleNamespace(
            fsdp_version=1,
            state_dict_type="FULL_STATE_DICT",
            state_dict_config={"offload_to_cpu": True, "rank0_only": rank0_only},
            optim_state_dict_config={"offload_to_cpu": True, "rank0_only": rank0_only},
            sharding_strategy="FULL_SHARD",
            reshard_after_forward=None,
            use_orig_params=False,
            cpu_offload=False,
            mixed_precision_policy=None,
            backward_prefetch=None,
            forward_prefetch=False,
            auto_wrap_policy="size_based_auto_wrap_policy",
            transformer_cls_names_to_wrap=None,
            min_num_params=min_num_params,
            limit_all_gathers=True,
            sync_module_states=True,
            cpu_ram_efficient_loading=False,
            activation_checkpointing=False,
        )
        return build_trainer_runtime_identity(trainer)

    baseline = identity(min_num_params=1_000, rank0_only=True)
    changed_wrap = identity(min_num_params=2_000, rank0_only=True)
    changed_state_dict = identity(min_num_params=1_000, rank0_only=False)

    assert changed_wrap["backend_schema_digest"] != baseline["backend_schema_digest"]
    assert changed_state_dict["backend_schema_digest"] != baseline["backend_schema_digest"]


def test_deepspeed_batch_and_accumulation_drift_changes_backend_digest() -> None:
    """DeepSpeed engine cadence and batch geometry participate in exact resume."""

    def identity(*, micro_batch: int, accumulation_steps: int) -> dict[str, Any]:
        trainer = _Trainer()
        trainer.accelerator.distributed_type = DistributedType.DEEPSPEED
        trainer.accelerator.gradient_accumulation_steps = accumulation_steps
        config = {
            "train_micro_batch_size_per_gpu": micro_batch,
            "gradient_accumulation_steps": accumulation_steps,
            "zero_optimization": {"stage": 2},
        }
        trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(
            zero_stage=2,
            deepspeed_config=config,
            gradient_accumulation_steps=accumulation_steps,
            gradient_clipping="auto",
            is_train_batch_min=True,
        )
        trainer.optimizer.optimizer = SimpleNamespace(
            bit16_groups=[list(group["params"]) for group in trainer.optimizer.param_groups]
        )
        return build_trainer_runtime_identity(trainer)

    baseline = identity(micro_batch=1, accumulation_steps=2)
    changed_micro_batch = identity(micro_batch=2, accumulation_steps=2)
    changed_accumulation = identity(micro_batch=1, accumulation_steps=4)

    assert changed_micro_batch["backend_schema_digest"] != baseline["backend_schema_digest"]
    assert changed_accumulation["backend_schema_digest"] != baseline["backend_schema_digest"]


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("beta", 0.2),
        ("weighting_scheme", "logit_normal"),
        ("timestep_range", (0.2, 0.8)),
        ("seed", 123),
    ),
)
def test_objective_sampling_and_seed_drift_change_execution_contract_digest(
    field: str,
    changed_value: Any,
) -> None:
    """Resolved objective, time sampling, and RNG cadence gate exact resume."""
    baseline = build_trainer_runtime_identity(_Trainer())
    changed = build_trainer_runtime_identity(_Trainer(**{field: changed_value}))

    assert changed["execution_contract_digest"] != baseline["execution_contract_digest"]


def test_scheduler_reward_acceleration_and_model_forward_drift_change_execution_digest() -> None:
    """Every model-forward input outside the optimizer schema is still identity-locked."""
    baseline_trainer = _Trainer()
    baseline = build_trainer_runtime_identity(baseline_trainer)

    changed_scheduler = _Trainer()
    changed_scheduler.config.scheduler_args.seed = 10
    changed_reward = _Trainer()
    changed_reward.reward_args.rewards[0]["weight"] = 2.0
    changed_acceleration = _Trainer()
    changed_acceleration.config.acceleration_args.shared = [
        {"name": "attention_backend", "params": {"backend": "sdpa"}}
    ]
    changed_forward = _Trainer()
    changed_forward.model_args.forward_variant = "velocity"

    for changed_trainer in (
        changed_scheduler,
        changed_reward,
        changed_acceleration,
        changed_forward,
    ):
        changed = build_trainer_runtime_identity(changed_trainer)
        assert changed["execution_contract_digest"] != baseline["execution_contract_digest"]


def test_budget_logging_checkpoint_cadence_and_resume_location_are_operational() -> None:
    """Non-computational controls may change without pretending the math changed."""
    baseline_trainer = _Trainer(max_epochs=2, log_every=10)
    changed_trainer = _Trainer(max_epochs=20, log_every=1)
    changed_trainer.config.log_args.save_freq = 5
    changed_trainer.model_args.resume_path = "different-checkpoint"
    changed_trainer.model_args.resume_type = "full"

    baseline = build_trainer_runtime_identity(baseline_trainer)
    changed = build_trainer_runtime_identity(changed_trainer)

    assert changed["execution_contract_digest"] == baseline["execution_contract_digest"]


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (("eval_freq", 4), ("guidance_scale", 7.0)),
)
def test_evaluation_cadence_and_sampling_change_execution_contract_digest(
    field: str,
    changed_value: Any,
) -> None:
    """Online exact resume replays eval, so its RNG-consuming semantics are locked."""
    baseline = build_trainer_runtime_identity(_Trainer())
    changed_trainer = _Trainer()
    setattr(changed_trainer.eval_args, field, changed_value)
    changed = build_trainer_runtime_identity(changed_trainer)

    assert changed["execution_contract_digest"] != baseline["execution_contract_digest"]


def test_evaluation_reward_semantics_change_execution_contract_digest() -> None:
    """A replayed eval reward implementation is part of exact continuation."""
    baseline = build_trainer_runtime_identity(_Trainer())
    changed_trainer = _Trainer()
    changed_trainer.eval_reward_args = _ConfigBlock(
        rewards=[{"name": "aesthetic", "reward_model": "tests.OtherReward", "weight": 1.0}]
    )
    changed = build_trainer_runtime_identity(changed_trainer)

    assert changed["execution_contract_digest"] != baseline["execution_contract_digest"]


def test_evaluation_dataset_overrides_change_execution_contract_digest() -> None:
    """Per-dataset eval overrides control replayed adapter sampling."""
    baseline_trainer = _Trainer()
    changed_trainer = _Trainer()
    for trainer in (baseline_trainer, changed_trainer):
        _configure_evaluation(trainer, (("eval-a", _prepared_eval_loader()),))
    changed_trainer._eval_dataset_configs["eval-a"].eval["guidance_scale"] = 6.0

    baseline = build_trainer_runtime_identity(baseline_trainer)
    changed = build_trainer_runtime_identity(changed_trainer)

    assert changed["execution_contract_digest"] != baseline["execution_contract_digest"]
    assert changed["data_contract_digest"] == baseline["data_contract_digest"]


def test_evaluation_loader_content_and_order_change_data_contract_digest() -> None:
    """Replayed eval inputs and their RNG-consuming traversal order stay locked."""
    baseline_trainer = _Trainer()
    _configure_evaluation(
        baseline_trainer,
        (
            ("eval-a", _prepared_eval_loader(fingerprint="eval-a")),
            ("eval-b", _prepared_eval_loader(fingerprint="eval-b")),
        ),
    )
    changed_content_trainer = _Trainer()
    _configure_evaluation(
        changed_content_trainer,
        (
            ("eval-a", _prepared_eval_loader(fingerprint="eval-a-v2")),
            ("eval-b", _prepared_eval_loader(fingerprint="eval-b")),
        ),
    )
    reordered_trainer = _Trainer()
    _configure_evaluation(
        reordered_trainer,
        (
            ("eval-b", _prepared_eval_loader(fingerprint="eval-b")),
            ("eval-a", _prepared_eval_loader(fingerprint="eval-a")),
        ),
    )

    baseline = build_trainer_runtime_identity(baseline_trainer)
    changed_content = build_trainer_runtime_identity(changed_content_trainer)
    reordered = build_trainer_runtime_identity(reordered_trainer)

    assert changed_content["data_contract_digest"] != baseline["data_contract_digest"]
    assert reordered["data_contract_digest"] != baseline["data_contract_digest"]


def test_prepared_evaluation_loader_identity_excludes_process_index() -> None:
    """All ranks agree on one eval data contract after Accelerate sharding."""
    rank_zero_trainer = _Trainer()
    rank_one_trainer = _Trainer()
    _configure_evaluation(
        rank_zero_trainer,
        (("eval-a", _prepared_eval_loader(rank=0)),),
    )
    _configure_evaluation(
        rank_one_trainer,
        (("eval-a", _prepared_eval_loader(rank=1)),),
    )

    rank_zero = build_trainer_runtime_identity(rank_zero_trainer)
    rank_one = build_trainer_runtime_identity(rank_one_trainer)

    assert rank_one["data_contract_digest"] == rank_zero["data_contract_digest"]


def _offline_dataset(
    *,
    record_ids: tuple[str, ...] = ("record-a", "record-b"),
    condition_ids: tuple[str, ...] = ("condition-a", "condition-b"),
    source_name: str = "source",
    source_id: int = 0,
) -> OfflineDataset:
    """Create an identity-only offline dataset without decoding target media."""
    dataset = OfflineDataset.__new__(OfflineDataset)
    dataset._records = tuple(None for _ in record_ids)
    dataset._record_ids = record_ids
    dataset._condition_ids = condition_ids
    dataset._condition_cache = _FingerprintDataset("condition-cache-v1", len(record_ids))
    dataset.source_name = source_name
    dataset.source_id = source_id
    dataset.supervision_type = "demonstration"
    return dataset


def _offline_loader(
    *,
    dataset: OfflineDataset | None = None,
    seed: int = 42,
    rank: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
    batch_size: int = 1,
) -> DataLoader:
    """Build an official distributed offline loader for identity tests."""
    concatenated = ConcatDataset([_offline_dataset() if dataset is None else dataset])
    sampler = DistributedSampler(
        concatenated,
        num_replicas=2,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    return DataLoader(
        concatenated,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
    )


@pytest.mark.parametrize(
    "loader",
    (
        _offline_loader(dataset=_offline_dataset(record_ids=("record-a", "target-changed"))),
        _offline_loader(
            dataset=_offline_dataset(condition_ids=("condition-a", "manifest-changed"))
        ),
        _offline_loader(dataset=_offline_dataset(source_name="renamed-source")),
        _offline_loader(seed=99),
        _offline_loader(shuffle=False),
        _offline_loader(drop_last=True),
        _offline_loader(batch_size=2),
    ),
)
def test_offline_records_source_and_sampler_semantics_change_data_contract_digest(
    loader: DataLoader,
) -> None:
    """Ordered targets, conditions, provenance, and sampler policy are exact inputs."""
    baseline = build_trainer_runtime_identity(_Trainer(dataloader=_offline_loader()))
    changed = build_trainer_runtime_identity(_Trainer(dataloader=loader))

    assert changed["data_contract_digest"] != baseline["data_contract_digest"]


def test_data_contract_tracks_source_order_and_gradient_accumulation() -> None:
    """Cross-source traversal order and complete optimizer windows are locked."""
    source_a = _offline_dataset(
        record_ids=("record-a1", "record-a2"),
        condition_ids=("condition-a1", "condition-a2"),
        source_name="a",
        source_id=0,
    )
    source_b = _offline_dataset(
        record_ids=("record-b1", "record-b2"),
        condition_ids=("condition-b1", "condition-b2"),
        source_name="b",
        source_id=1,
    )

    def loader(sources: tuple[OfflineDataset, ...]) -> DataLoader:
        dataset = ConcatDataset(sources)
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=True,
            seed=42,
        )
        return DataLoader(dataset, batch_size=1, sampler=sampler)

    baseline_trainer = _Trainer(dataloader=loader((source_a, source_b)))
    reordered_trainer = _Trainer(dataloader=loader((source_b, source_a)))
    changed_accumulation = _Trainer(dataloader=loader((source_a, source_b)))
    changed_accumulation.training_args.gradient_accumulation_steps = 2

    baseline = build_trainer_runtime_identity(baseline_trainer)
    reordered = build_trainer_runtime_identity(reordered_trainer)
    changed_gas = build_trainer_runtime_identity(changed_accumulation)

    assert reordered["data_contract_digest"] != baseline["data_contract_digest"]
    assert changed_gas["data_contract_digest"] != baseline["data_contract_digest"]


def test_offline_data_identity_excludes_global_transport_source_ids() -> None:
    """Eval-only entries may renumber offline sources without changing training."""

    def identity(source_ids: tuple[int, int]) -> dict[str, Any]:
        sources = (
            _offline_dataset(
                record_ids=("record-a1", "record-a2"),
                condition_ids=("condition-a1", "condition-a2"),
                source_name="train-a",
                source_id=source_ids[0],
            ),
            _offline_dataset(
                record_ids=("record-b1", "record-b2"),
                condition_ids=("condition-b1", "condition-b2"),
                source_name="train-b",
                source_id=source_ids[1],
            ),
        )
        dataset = ConcatDataset(sources)
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=True,
            seed=42,
        )
        return build_trainer_runtime_identity(
            _Trainer(dataloader=DataLoader(dataset, batch_size=1, sampler=sampler))
        )

    baseline = identity((0, 2))
    eval_sources_inserted_and_reordered = identity((3, 1))

    assert (
        eval_sources_inserted_and_reordered["data_contract_digest"]
        == baseline["data_contract_digest"]
    )


def test_loader_identity_excludes_rank_but_tracks_online_content_and_sampler_seed() -> None:
    """Every rank hashes one contract while content and order changes remain visible."""
    rank_zero = build_trainer_runtime_identity(_Trainer(dataloader=_online_loader(rank=0)))
    rank_one = build_trainer_runtime_identity(_Trainer(dataloader=_online_loader(rank=1)))
    changed_content = build_trainer_runtime_identity(
        _Trainer(dataloader=_online_loader(fingerprint="online-content-v2"))
    )
    changed_seed = build_trainer_runtime_identity(_Trainer(dataloader=_online_loader(seed=18)))

    assert rank_one["data_contract_digest"] == rank_zero["data_contract_digest"]
    assert changed_content["data_contract_digest"] != rank_zero["data_contract_digest"]
    assert changed_seed["data_contract_digest"] != rank_zero["data_contract_digest"]


def test_eval_only_source_registry_changes_do_not_renumber_multi_source_identity() -> None:
    """Full global source-ID maps are transport metadata, not training semantics."""
    baseline = build_trainer_runtime_identity(
        _Trainer(
            dataloader=_multi_source_online_loader(
                source_name_to_id={"train-a": 0, "eval-only": 1, "train-b": 2}
            )
        )
    )
    eval_sources_inserted_and_reordered = build_trainer_runtime_identity(
        _Trainer(
            dataloader=_multi_source_online_loader(
                source_name_to_id={
                    "eval-second": 0,
                    "train-a": 1,
                    "eval-only": 2,
                    "train-b": 3,
                }
            )
        )
    )
    reversed_training_order = build_trainer_runtime_identity(
        _Trainer(
            dataloader=_multi_source_online_loader(
                source_order=("train-b", "train-a"),
                source_name_to_id={"train-b": 0, "train-a": 1},
            )
        )
    )
    one_training_source_removed = build_trainer_runtime_identity(
        _Trainer(
            dataloader=_multi_source_online_loader(
                source_order=("train-a",),
                source_name_to_id={"train-a": 0},
            )
        )
    )

    assert (
        eval_sources_inserted_and_reordered["data_contract_digest"]
        == baseline["data_contract_digest"]
    )
    assert reversed_training_order["data_contract_digest"] != baseline["data_contract_digest"]
    assert one_training_source_removed["data_contract_digest"] != baseline["data_contract_digest"]


def test_multi_source_block_geometry_changes_data_identity() -> None:
    one_batch_blocks = build_trainer_runtime_identity(
        _Trainer(dataloader=_multi_source_online_loader(batches_per_block=1))
    )
    group_complete_blocks = build_trainer_runtime_identity(
        _Trainer(dataloader=_multi_source_online_loader(batches_per_block=2))
    )

    assert group_complete_blocks["data_contract_digest"] != one_batch_blocks["data_contract_digest"]
