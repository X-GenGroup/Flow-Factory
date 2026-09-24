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

"""Build deterministic exact-resume identities from realized trainer state."""

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any

import torch
from accelerate.utils import DistributedType
from torch.utils.data import ConcatDataset, DataLoader, Subset

_EXECUTION_IDENTITY_HOOK = "runtime_execution_identity_payload"
_DATA_IDENTITY_HOOK = "runtime_data_identity_payload"
_OPERATIONAL_TRAINING_FIELDS = frozenset({"max_epochs"})
_RESUME_MODEL_FIELDS = frozenset({"resume_path", "resume_type"})


def build_trainer_runtime_identity(trainer: Any) -> dict[str, Any]:
    """Describe the realized trainer, model, and optimizer compatibility boundary.

    This function intentionally runs after ``accelerator.prepare`` and variant
    parameter rebinding. Consequently the schema is derived from the physical
    parameter roots and optimizer groups that Accelerate will restore, rather than
    from a pre-prepare configuration approximation.

    Args:
        trainer: Initialized trainer exposing adapter, optimizer, and accelerator.

    Returns:
        Strict identity mapping accepted by :class:`TrainerRuntimeState`.
    """
    parameter_schema, parameter_keys = _parameter_schema(trainer)
    optimizer_schema = _optimizer_schema(trainer, parameter_keys)
    backend_schema = _backend_schema(trainer.accelerator)
    execution_schema = _trainer_identity_payload(
        trainer,
        hook_name=_EXECUTION_IDENTITY_HOOK,
        default_builder=build_default_execution_identity_payload,
    )
    data_schema = _trainer_identity_payload(
        trainer,
        hook_name=_DATA_IDENTITY_HOOK,
        default_builder=build_default_data_identity_payload,
    )
    model_args = trainer.model_args
    training_args = trainer.training_args
    return {
        "trainer": _qualified_type_name(type(trainer)),
        "adapter": _qualified_type_name(type(trainer.adapter)),
        "algorithm": _require_non_empty_string(
            getattr(training_args, "trainer_type", None),
            "training_args.trainer_type",
        ),
        "model": (
            f"{_require_non_empty_string(getattr(model_args, 'model_type', None), 'model.model_type')}"
            f":{_require_non_empty_string(getattr(model_args, 'model_name_or_path', None), 'model.model_name_or_path')}"
        ),
        "finetune_type": _require_non_empty_string(
            getattr(model_args, "finetune_type", None),
            "model.finetune_type",
        ),
        "optimizer_roles": tuple(trainer._required_trainable_roles()),
        "parameter_schema_digest": _schema_digest(parameter_schema),
        "optimizer_schema_digest": _schema_digest(optimizer_schema),
        "execution_contract_digest": _schema_digest(execution_schema),
        "data_contract_digest": _schema_digest(data_schema),
        "distributed_type": backend_schema["distributed_type"],
        "backend_schema_digest": _schema_digest(backend_schema),
        "mixed_precision": _require_non_empty_string(
            getattr(trainer.accelerator, "mixed_precision", None),
            "accelerator.mixed_precision",
        ),
        "gradient_scaler": (
            "none"
            if getattr(trainer.accelerator, "scaler", None) is None
            else _qualified_type_name(type(trainer.accelerator.scaler))
        ),
        "world_size": _require_positive_int(
            getattr(trainer.accelerator, "num_processes", None),
            "accelerator.num_processes",
        ),
    }


def build_default_execution_identity_payload(trainer: Any) -> dict[str, Any]:
    """Return resolved objective and model-forward semantics for exact resume.

    The run budget, logging, checkpoint cadence, and resume location intentionally
    remain operational controls. Evaluation is identity-locked because the online
    checkpoint boundary replays evaluation before the next acquisition, and adapter
    or reward evaluation may consume global device RNG. Everything that can alter a
    training forward, objective, time sample, reward, optimizer cadence, or replayed
    evaluation stays locked. Trainers with additional realized semantics may override
    ``runtime_execution_identity_payload`` and extend this mapping.
    """
    training = _export_config(trainer.training_args, "training_args")
    for field_name in _OPERATIONAL_TRAINING_FIELDS:
        training.pop(field_name, None)

    model = _export_config(trainer.model_args, "model_args")
    for field_name in _RESUME_MODEL_FIELDS:
        model.pop(field_name, None)

    config = trainer.config
    scheduler = _export_config(config.scheduler_args, "config.scheduler_args")
    acceleration = _export_config(config.acceleration_args, "config.acceleration_args")
    rewards = _export_config(trainer.reward_args, "reward_args")
    evaluation = _evaluation_execution_schema(trainer)
    optimizer_execution = _resolved_optimizer_execution_schema(trainer)
    execution_contract = type(trainer).execution_contract
    acquisition = getattr(execution_contract, "acquisition", None)
    feedback = getattr(execution_contract, "feedback", None)
    pipeline_io_contract = getattr(
        trainer.adapter,
        "effective_pipeline_io_contract",
        None,
    )
    return {
        "contract": {
            "acquisition": _enum_identity_value(
                acquisition,
                "execution_contract.acquisition",
            ),
            "feedback": _enum_identity_value(
                feedback,
                "execution_contract.feedback",
            ),
            "paradigm": getattr(type(trainer), "paradigm", None),
        },
        "pipeline_io_contract": pipeline_io_contract,
        "training": training,
        "scheduler": scheduler,
        "realized_scheduler_group": _scheduler_group_schema(trainer.adapter),
        "training_rewards": rewards,
        "evaluation": evaluation,
        "acceleration": acceleration,
        "model_forward": model,
        "optimizer_execution": optimizer_execution,
    }


def build_default_data_identity_payload(trainer: Any) -> dict[str, Any]:
    """Return rank-free manifest/fingerprint and loader-order semantics.

    Offline record IDs cover normalized manifest semantics and build-local streaming
    SHA-256 digests for input and target/chosen/rejected media. Online sources use their
    resolved Hugging Face preprocessing fingerprints. Ordered training source names are
    locked, while global numeric source IDs and full name-to-ID registries are excluded
    because eval-only entries can renumber them without changing training. Ordered
    realized evaluation loaders are locked separately because online exact resume
    replays evaluation after the source checkpoint. Sampler rank and mutable epoch/index
    state are excluded, while seed, shuffle/drop policy, batch geometry, and accumulation
    cadence remain locked. A trainer with a new loader abstraction can override
    ``runtime_data_identity_payload`` instead of coupling it to this inspector.
    """
    accumulation_steps = getattr(
        trainer.training_args,
        "gradient_accumulation_steps",
        None,
    )
    return {
        "gradient_accumulation_steps": accumulation_steps,
        "training_loader": _loader_schema(
            getattr(trainer, "dataloader", None),
            "dataloader",
        ),
        "evaluation_loaders": _evaluation_loader_schema(trainer),
    }


def _evaluation_execution_schema(trainer: Any) -> dict[str, Any]:
    """Describe evaluation semantics in their realized execution order."""
    eval_args = _export_config(trainer.eval_args, "eval_args")
    eval_rewards = _export_config(trainer.eval_reward_args, "eval_reward_args")
    eval_loaders = _require_named_eval_loaders(trainer)
    eval_configs = getattr(trainer, "_eval_dataset_configs", None)
    if not isinstance(eval_configs, Mapping):
        raise TypeError(
            "trainer _eval_dataset_configs must be a mapping, received "
            f"{type(eval_configs).__name__}: {eval_configs!r}"
        )

    datasets = []
    for dataset_name in eval_loaders:
        if dataset_name not in eval_configs:
            raise KeyError(
                "trainer evaluation loader has no realized dataset configuration: "
                f"{dataset_name!r}"
            )
        datasets.append(
            {
                "name": dataset_name,
                "configuration": _export_config(
                    eval_configs[dataset_name],
                    f"_eval_dataset_configs[{dataset_name!r}]",
                ),
            }
        )
    return {
        "arguments": eval_args,
        "rewards": eval_rewards,
        "datasets": datasets,
    }


def _evaluation_loader_schema(trainer: Any) -> list[dict[str, Any]]:
    """Describe ordered eval loaders without mutable iterator or rank state."""
    return [
        {
            "name": dataset_name,
            "loader": _loader_schema(
                loader,
                f"eval_dataloaders[{dataset_name!r}]",
            ),
        }
        for dataset_name, loader in _require_named_eval_loaders(trainer).items()
    ]


def _require_named_eval_loaders(trainer: Any) -> Mapping[str, Any]:
    """Return the ordered realized evaluation-loader mapping."""
    eval_loaders = getattr(trainer, "eval_dataloaders", None)
    if not isinstance(eval_loaders, Mapping):
        raise TypeError(
            "trainer eval_dataloaders must be a mapping, received "
            f"{type(eval_loaders).__name__}: {eval_loaders!r}"
        )
    for dataset_name in eval_loaders:
        if type(dataset_name) is not str or not dataset_name:
            raise TypeError(
                "trainer evaluation-loader names must be non-empty strings, "
                f"received {dataset_name!r}"
            )
    return eval_loaders


def _trainer_identity_payload(
    trainer: Any,
    *,
    hook_name: str,
    default_builder: Any,
) -> Any:
    """Call one trainer extension hook and canonicalize its strict mapping."""
    hook = getattr(trainer, hook_name, None)
    payload = default_builder(trainer) if hook is None else hook()
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"trainer {hook_name} must return a mapping, received "
            f"{type(payload).__name__}: {payload!r}"
        )
    return _canonical_contract_value(payload, hook_name)


def _export_config(value: Any, path: str) -> dict[str, Any]:
    """Export one resolved config block without accepting object repr fallbacks."""
    if isinstance(value, Mapping):
        exported = value
    else:
        exporter = getattr(value, "to_dict", None)
        if not callable(exporter):
            raise TypeError(
                f"trainer runtime execution config at {path} must be a mapping or "
                f"provide to_dict(), received {type(value).__name__}: {value!r}"
            )
        exported = exporter()
    if not isinstance(exported, Mapping):
        raise TypeError(
            f"trainer runtime execution config {path}.to_dict() must return a mapping, "
            f"received {type(exported).__name__}: {exported!r}"
        )
    return dict(exported)


def _resolved_optimizer_execution_schema(trainer: Any) -> list[dict[str, Any]]:
    """Describe per-role optimizer semantics absent from parameter groups.

    The realized optimizer groups already lock update-rule fields such as learning
    rate, moments, and weight decay. Gradient clipping and role update frequency are
    consumed through ``RoleOptimizerConfig`` instead, so a param-group-only identity
    would accept a resume that changes the optimization trajectory. Resolve through
    the trainer hook to include algorithm-provided defaults as well as explicit
    ``optimizers`` entries.
    """
    resolved = []
    for role_name in trainer._required_trainable_roles():
        optimizer_args = trainer._optimizer_args_for_role(role_name)
        resolved.append(
            {
                "role_name": role_name,
                "arguments_type": _qualified_type_name(type(optimizer_args)),
                "arguments": _export_config(
                    optimizer_args,
                    f"config.optimizer_args[{role_name!r}]",
                ),
            }
        )
    return resolved


def _enum_identity_value(value: Any, path: str) -> str:
    """Require a concrete enum-valued execution axis."""
    if not isinstance(value, Enum):
        raise TypeError(
            f"trainer runtime identity {path} must be an enum member, received "
            f"{type(value).__name__}: {value!r}"
        )
    return str(value.value)


def _scheduler_group_schema(adapter: Any) -> Any:
    """Describe realized scheduler types and immutable configs in component order."""
    group = getattr(adapter, "scheduler_group", None)
    if group is None:
        return None
    names = getattr(group, "names", None)
    if isinstance(names, (str, bytes)):
        raise TypeError("adapter.scheduler_group.names must be a sequence, not a string")
    try:
        names = tuple(names)
    except TypeError as error:
        raise TypeError("adapter.scheduler_group.names must be a sequence") from error
    if not names:
        raise ValueError("adapter.scheduler_group.names cannot be empty")
    schedulers = []
    for name in names:
        if type(name) is not str or not name:
            raise TypeError(
                "adapter.scheduler_group component names must be non-empty strings, "
                f"received {name!r}"
            )
        scheduler = group[name]
        scheduler_config = getattr(scheduler, "config", None)
        if scheduler_config is not None:
            scheduler_config = _export_config(
                scheduler_config,
                f"adapter.scheduler_group[{name!r}].config",
            )
        schedulers.append(
            {
                "name": name,
                "type": _qualified_type_name(type(scheduler)),
                "dynamics_type": getattr(scheduler, "dynamics_type", None),
                "config": scheduler_config,
            }
        )
    return {
        "primary_name": _require_non_empty_string(
            getattr(group, "primary_name", None),
            "adapter.scheduler_group.primary_name",
        ),
        "schedulers": schedulers,
    }


def _loader_schema(loader: Any, path: str) -> Any:
    """Describe one framework train loader without mutable iterator state."""
    if loader is None:
        return None
    loaders_by_source = getattr(loader, "_loaders_by_source", None)
    source_scheduler = getattr(loader, "_scheduler", None)
    if isinstance(loaders_by_source, Mapping) and source_scheduler is not None:
        sources = []
        for source_name, source_loader in loaders_by_source.items():
            if type(source_name) is not str or not source_name:
                raise TypeError(
                    f"multi-source loader name at {path} must be a non-empty str, "
                    f"received {source_name!r}"
                )
            sources.append(
                {
                    "name": source_name,
                    "loader": _loader_schema(
                        source_loader,
                        f"{path}.sources[{source_name!r}]",
                    ),
                }
            )
        source_schedule = {
            "counts": getattr(source_scheduler, "_counts", None),
            "seed": getattr(source_scheduler, "_seed", None),
        }
        batches_per_block = getattr(source_scheduler, "_batches_per_block", 1)
        if batches_per_block != 1:
            source_schedule["batches_per_block"] = batches_per_block
        return {
            "type": _qualified_type_name(type(loader)),
            "batch_size": getattr(loader, "_batch_size", None),
            "length": len(loader),
            "sources": sources,
            "source_schedule": source_schedule,
        }
    if not isinstance(loader, DataLoader):
        raise TypeError(
            f"unsupported train loader at {path}: {type(loader).__name__}; "
            f"override {_DATA_IDENTITY_HOOK}() for a custom acquisition loader"
        )
    return {
        "type": _qualified_type_name(type(loader)),
        "length": _loader_length(loader, path),
        "dataset": _dataset_schema(loader.dataset, f"{path}.dataset"),
        "sampler": _sampler_schema(loader.sampler, f"{path}.sampler"),
        "batch_sampler": _sampler_schema(
            loader.batch_sampler,
            f"{path}.batch_sampler",
        ),
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "num_workers": loader.num_workers,
        "persistent_workers": loader.persistent_workers,
        "prefetch_factor": loader.prefetch_factor,
        "pin_memory": loader.pin_memory,
        "pin_memory_device": getattr(loader, "pin_memory_device", ""),
        "timeout": loader.timeout,
        "in_order": getattr(loader, "in_order", True),
        "collate": _callable_identity_schema(loader.collate_fn, f"{path}.collate_fn"),
        "worker_init": _callable_identity_schema(
            loader.worker_init_fn,
            f"{path}.worker_init_fn",
        ),
    }


def _loader_length(loader: DataLoader, path: str) -> int:
    """Return finite batch geometry for finite or epoch-bounded samplers."""
    try:
        length = len(loader)
    except TypeError:
        length = getattr(loader.batch_sampler, "num_batches_per_epoch", None)
    if type(length) is not int or length < 1:
        raise TypeError(
            f"train loader length at {path} must be a positive int or expose "
            f"batch_sampler.num_batches_per_epoch, received {length!r}"
        )
    return length


def _dataset_schema(dataset: Any, path: str) -> dict[str, Any]:
    """Describe ordered dataset provenance without decoding large media files."""
    if isinstance(dataset, ConcatDataset):
        return {
            "type": _qualified_type_name(type(dataset)),
            "length": _dataset_length(dataset, path),
            "sources": [
                _dataset_schema(source, f"{path}.sources[{index}]")
                for index, source in enumerate(dataset.datasets)
            ],
        }
    if isinstance(dataset, Subset):
        indices = tuple(dataset.indices)
        return {
            "type": _qualified_type_name(type(dataset)),
            "length": len(indices),
            "indices_digest": _schema_digest(indices),
            "dataset": _dataset_schema(dataset.dataset, f"{path}.dataset"),
        }

    record_ids = getattr(dataset, "_record_ids", None)
    condition_ids = getattr(dataset, "_condition_ids", None)
    if record_ids is not None or condition_ids is not None:
        if isinstance(record_ids, (str, bytes)) or isinstance(condition_ids, (str, bytes)):
            raise TypeError(f"offline dataset IDs at {path} must be ordered sequences")
        try:
            record_ids = tuple(record_ids)
            condition_ids = tuple(condition_ids)
        except TypeError as error:
            raise TypeError(f"offline dataset IDs at {path} must be ordered sequences") from error
        if len(record_ids) != len(condition_ids) or len(record_ids) != _dataset_length(
            dataset,
            path,
        ):
            raise ValueError(
                f"offline dataset ID cardinality mismatch at {path}: "
                f"records={len(record_ids)}, conditions={len(condition_ids)}, "
                f"dataset={len(dataset)}"
            )
        for identifier_name, identifiers in (
            ("record", record_ids),
            ("condition", condition_ids),
        ):
            invalid = tuple(
                identifier
                for identifier in identifiers
                if type(identifier) is not str or not identifier
            )
            if invalid:
                raise TypeError(
                    f"offline {identifier_name} IDs at {path} must be non-empty strings, "
                    f"received {invalid!r}"
                )
        return {
            "type": _qualified_type_name(type(dataset)),
            "length": len(record_ids),
            "source_name": _require_non_empty_string(
                getattr(dataset, "source_name", None),
                f"{path}.source_name",
            ),
            "supervision_type": _require_non_empty_string(
                getattr(dataset, "supervision_type", None),
                f"{path}.supervision_type",
            ),
            "record_ids_digest": _schema_digest(record_ids),
            "condition_ids_digest": _schema_digest(condition_ids),
            "condition_cache": _fingerprint_schema(
                getattr(dataset, "_condition_cache", None),
                f"{path}.condition_cache",
            ),
        }

    processed_dataset = getattr(dataset, "processed_dataset", None)
    if processed_dataset is not None:
        return {
            "type": _qualified_type_name(type(dataset)),
            "length": _dataset_length(dataset, path),
            "processed": _fingerprint_schema(
                processed_dataset,
                f"{path}.processed_dataset",
            ),
        }
    if getattr(dataset, "_fingerprint", None) is not None:
        return _fingerprint_schema(dataset, path)
    raise TypeError(
        f"unsupported train dataset at {path}: {type(dataset).__name__}; "
        f"override {_DATA_IDENTITY_HOOK}() for a custom dataset contract"
    )


def _fingerprint_schema(dataset: Any, path: str) -> dict[str, Any]:
    """Require the stable cache/source fingerprint already owned by the dataset."""
    if dataset is None:
        raise TypeError(f"dataset fingerprint source at {path} cannot be None")
    fingerprint = getattr(dataset, "_fingerprint", None)
    if type(fingerprint) is not str or not fingerprint:
        raise TypeError(
            f"dataset at {path} must expose a non-empty _fingerprint, received "
            f"{type(fingerprint).__name__}: {fingerprint!r}"
        )
    return {
        "type": _qualified_type_name(type(dataset)),
        "length": _dataset_length(dataset, path),
        "fingerprint": fingerprint,
    }


def _dataset_length(dataset: Any, path: str) -> int:
    """Require a finite non-negative dataset cardinality."""
    try:
        length = len(dataset)
    except TypeError as error:
        raise TypeError(f"train dataset at {path} must have a finite length") from error
    if type(length) is not int or length < 0:
        raise TypeError(
            f"train dataset length at {path} must be a non-negative int, received {length!r}"
        )
    return length


def _sampler_schema(sampler: Any, path: str) -> Any:
    """Describe rank-free sampler order and batch geometry."""
    if sampler is None:
        return None
    module = type(sampler).__module__
    if not (
        module.startswith("torch.utils.data")
        or module.startswith("accelerate.data_loader")
        or module.startswith("flow_factory.data_utils.sampler")
    ):
        raise TypeError(
            f"unsupported sampler at {path}: {_qualified_type_name(type(sampler))}; "
            f"override {_DATA_IDENTITY_HOOK}() for custom sampler semantics"
        )
    schema: dict[str, Any] = {"type": _qualified_type_name(type(sampler))}
    for field_name in (
        "batch_size",
        "drop_last",
        "shuffle",
        "seed",
        "num_replicas",
        "num_processes",
        "num_samples",
        "total_size",
        "replacement",
        "k",
        "m",
        "sample_num_per_iteration",
        "groups_per_rank",
        "copies_per_rank",
        "num_batches_per_epoch",
        "split_batches",
        "even_batches",
    ):
        if hasattr(sampler, field_name):
            schema[field_name] = getattr(sampler, field_name)
    nested_sampler = getattr(sampler, "sampler", None)
    if nested_sampler is not None and nested_sampler is not sampler:
        schema["sampler"] = _sampler_schema(nested_sampler, f"{path}.sampler")
    nested_batch_sampler = getattr(sampler, "batch_sampler", None)
    if nested_batch_sampler is not None and nested_batch_sampler is not sampler:
        schema["batch_sampler"] = _sampler_schema(
            nested_batch_sampler,
            f"{path}.batch_sampler",
        )
    generator = getattr(sampler, "generator", None)
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError(
                f"sampler generator at {path} must be torch.Generator or None, "
                f"received {type(generator).__name__}"
            )
        schema["generator_initial_seed"] = generator.initial_seed()
    # ``rank`` and mutable ``epoch``/iterator offsets are intentionally absent.
    return schema


def _callable_identity_schema(value: Any, path: str) -> Any:
    """Describe a loader callable without process-local object identity."""
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": _qualified_type_name(type(value)),
            "state": {
                field.name: _canonical_contract_value(
                    getattr(value, field.name),
                    f"{path}.{field.name}",
                )
                for field in fields(value)
            },
        }
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if type(module) is str and type(qualname) is str:
        return {"callable": f"{module}.{qualname}"}
    if callable(value):
        return {"type": _qualified_type_name(type(value))}
    raise TypeError(
        f"loader callable at {path} must be callable or None, received "
        f"{type(value).__name__}: {value!r}"
    )


def _canonical_contract_value(value: Any, path: str) -> Any:
    """Convert execution/data hook output into strict deterministic JSON values."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"runtime contract value at {path} must be finite")
        return value
    if isinstance(value, Enum):
        if type(value.value) in (bool, int, float, str):
            return value.value
        return f"{_qualified_type_name(type(value))}.{value.name}"
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_contract_value(
                getattr(value, field.name),
                f"{path}.{field.name}",
            )
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result = {}
        for key in value:
            if type(key) is not str:
                raise TypeError(
                    f"runtime contract mapping key at {path} must be str, "
                    f"received {type(key).__name__}: {key!r}"
                )
        for key in sorted(value):
            result[key] = _canonical_contract_value(value[key], f"{path}.{key}")
        return result
    if isinstance(value, (set, frozenset)):
        items = [_canonical_contract_value(item, f"{path}[set]") for item in value]
        return {
            "set": sorted(
                items,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical_contract_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, type):
        return {"type": _qualified_type_name(value)}
    if callable(value):
        return _callable_backend_schema(value, path=path)
    raise TypeError(
        f"unsupported runtime contract value at {path}: " f"{type(value).__name__}: {value!r}"
    )


def _backend_schema(accelerator: Any) -> dict[str, Any]:
    """Return the prepared-state backend plan that controls checkpoint layout."""
    distributed_type = _canonical_backend_value(
        getattr(accelerator, "distributed_type", None),
        "accelerator.distributed_type",
    )
    if type(distributed_type) is not str or not distributed_type:
        raise TypeError(
            "trainer runtime identity accelerator.distributed_type must resolve to a "
            f"non-empty str, received {distributed_type!r}"
        )
    state = getattr(accelerator, "state", None)
    fsdp_plugin = getattr(state, "fsdp_plugin", None)
    deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    fsdp_schema = None
    if fsdp_plugin is not None:
        fsdp_schema = {
            name: _canonical_backend_value(
                getattr(fsdp_plugin, name, None),
                f"accelerator.fsdp_plugin.{name}",
            )
            for name in (
                "fsdp_version",
                "state_dict_type",
                "state_dict_config",
                "optim_state_dict_config",
                "sharding_strategy",
                "reshard_after_forward",
                "use_orig_params",
                "cpu_offload",
                "mixed_precision_policy",
                "backward_prefetch",
                "forward_prefetch",
                "auto_wrap_policy",
                "transformer_cls_names_to_wrap",
                "min_num_params",
                "limit_all_gathers",
                "sync_module_states",
                "cpu_ram_efficient_loading",
                "activation_checkpointing",
            )
        }
    deepspeed_schema = None
    if deepspeed_plugin is not None:
        deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", None)
        zero_optimization = (
            deepspeed_config.get("zero_optimization")
            if isinstance(deepspeed_config, Mapping)
            else None
        )
        deepspeed_schema = {
            "zero_stage": _canonical_backend_value(
                getattr(deepspeed_plugin, "zero_stage", None),
                "accelerator.deepspeed_plugin.zero_stage",
            ),
            "config": _canonical_backend_value(
                deepspeed_config,
                "accelerator.deepspeed_plugin.deepspeed_config",
            ),
            "gradient_accumulation_steps": _canonical_backend_value(
                getattr(deepspeed_plugin, "gradient_accumulation_steps", None),
                "accelerator.deepspeed_plugin.gradient_accumulation_steps",
            ),
            "gradient_clipping": _canonical_backend_value(
                getattr(deepspeed_plugin, "gradient_clipping", None),
                "accelerator.deepspeed_plugin.gradient_clipping",
            ),
            "is_train_batch_min": _canonical_backend_value(
                getattr(deepspeed_plugin, "is_train_batch_min", None),
                "accelerator.deepspeed_plugin.is_train_batch_min",
            ),
            "zero_optimization": _canonical_backend_value(
                zero_optimization,
                "accelerator.deepspeed_plugin.zero_optimization",
            ),
        }
    return {
        "distributed_type": distributed_type,
        "gradient_accumulation_steps": _canonical_backend_value(
            getattr(accelerator, "gradient_accumulation_steps", None),
            "accelerator.gradient_accumulation_steps",
        ),
        "fsdp": fsdp_schema,
        "deepspeed": deepspeed_schema,
    }


def _canonical_backend_value(value: Any, path: str) -> Any:
    """Convert backend plugin settings into deterministic JSON-compatible values."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if math.isnan(value):
            raise ValueError(f"backend schema value at {path} cannot be NaN")
        if math.isinf(value):
            return {"float": "+infinity" if value > 0 else "-infinity"}
        return value
    if isinstance(value, Enum):
        if type(value.value) is str:
            return value.value
        return f"{_qualified_type_name(type(value))}.{value.name}"
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, partial):
        return {
            "partial": _callable_backend_schema(value.func, path=f"{path}.func"),
            "args": [
                _canonical_backend_value(item, f"{path}.args[{index}]")
                for index, item in enumerate(value.args)
            ],
            "keywords": _canonical_backend_value(
                value.keywords or {},
                f"{path}.keywords",
            ),
        }
    if isinstance(value, type):
        return {"type": _qualified_type_name(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_backend_value(
                getattr(value, field.name),
                f"{path}.{field.name}",
            )
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result = {}
        for key in value:
            if type(key) is not str:
                raise TypeError(
                    f"backend schema mapping key at {path} must be str, "
                    f"received {type(key).__name__}: {key!r}"
                )
        for key in sorted(value):
            result[key] = _canonical_backend_value(value[key], f"{path}.{key}")
        return result
    if isinstance(value, (set, frozenset)):
        items = [_canonical_backend_value(item, f"{path}[set]") for item in value]
        return {
            "set": sorted(
                items,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical_backend_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if callable(value):
        return _callable_backend_schema(value, path=path)
    raise TypeError(
        f"unsupported backend schema value at {path}: " f"{type(value).__name__}: {value!r}"
    )


def _callable_backend_schema(value: Any, *, path: str) -> dict[str, Any]:
    """Describe a wrap-policy callable and its bound configuration."""
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if type(module) is not str or type(qualname) is not str:
        raise TypeError(
            f"backend schema callable at {path} must expose module and qualname, "
            f"received {type(value).__name__}: {value!r}"
        )
    schema: dict[str, Any] = {"callable": f"{module}.{qualname}"}
    defaults = getattr(value, "__defaults__", None)
    if defaults:
        schema["defaults"] = _canonical_backend_value(defaults, f"{path}.defaults")
    keyword_defaults = getattr(value, "__kwdefaults__", None)
    if keyword_defaults:
        schema["keyword_defaults"] = _canonical_backend_value(
            keyword_defaults,
            f"{path}.keyword_defaults",
        )
    closure = getattr(value, "__closure__", None)
    if closure:
        schema["closure"] = [
            _canonical_backend_value(cell.cell_contents, f"{path}.closure[{index}]")
            for index, cell in enumerate(closure)
        ]
    return schema


def _parameter_schema(trainer: Any) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Return stable variant-owned parameter records and identity lookup."""
    registry = trainer.adapter.component_variant_registry
    schema: list[dict[str, Any]] = []
    parameter_keys: dict[int, str] = {}
    stable_keys: set[str] = set()
    for role_name in trainer._required_trainable_roles():
        for record in registry.parameter_records(role_name):
            parameter = record.parameter
            key = f"{role_name}/{record.component_name}/{record.parameter_name}"
            if id(parameter) in parameter_keys:
                raise ValueError(
                    "trainer runtime parameter schema contains duplicate parameter identity: "
                    f"{parameter_keys[id(parameter)]!r} and {key!r}"
                )
            if key in stable_keys:
                raise ValueError(f"trainer runtime parameter schema contains duplicate key {key!r}")
            parameter_keys[id(parameter)] = key
            stable_keys.add(key)
            schema.append(
                {
                    "key": key,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": bool(parameter.requires_grad),
                }
            )
    if not schema:
        raise RuntimeError("trainer runtime identity requires at least one parameter record")
    return schema, parameter_keys


def _optimizer_schema(trainer: Any, parameter_keys: Mapping[int, str]) -> dict[str, Any]:
    """Return ordered optimizer groups linked to the stable parameter schema."""
    optimizer = trainer.optimizer
    optimizer_groups = optimizer.param_groups
    logical_parameter_groups = _logical_optimizer_parameter_groups(trainer, optimizer_groups)
    groups = []
    consumed_parameters: set[str] = set()
    for group_index, (group, raw_parameters) in enumerate(
        zip(optimizer_groups, logical_parameter_groups)
    ):
        if not isinstance(raw_parameters, Sequence):
            raise TypeError(
                f"optimizer group {group_index} params must be a sequence, "
                f"received {type(raw_parameters).__name__}: {raw_parameters!r}"
            )
        group_parameters = []
        for parameter_index, parameter in enumerate(raw_parameters):
            key = parameter_keys.get(id(parameter))
            if key is None:
                raise ValueError(
                    "optimizer schema contains a parameter not owned by the rebound "
                    f"variant registry at group {group_index}, index {parameter_index}"
                )
            if key in consumed_parameters:
                raise ValueError(f"optimizer schema references parameter {key!r} more than once")
            consumed_parameters.add(key)
            group_parameters.append(key)
        settings = {}
        for key, value in group.items():
            if key == "params":
                continue
            if type(key) is not str or not key:
                raise TypeError(
                    f"optimizer group {group_index} setting key must be a non-empty str, "
                    f"received {type(key).__name__}: {key!r}"
                )
            settings[key] = _canonical_optimizer_value(
                value,
                f"group[{group_index}].{key}",
            )
        groups.append(
            {
                "parameters": group_parameters,
                "settings": settings,
            }
        )
    missing_parameters = frozenset(parameter_keys.values()).difference(consumed_parameters)
    if missing_parameters:
        raise ValueError(
            "optimizer schema does not exhaust rebound variant parameters: "
            f"{tuple(sorted(missing_parameters))!r}"
        )
    return {
        "type_chain": _optimizer_type_chain(optimizer),
        "groups": groups,
    }


def _logical_optimizer_parameter_groups(
    trainer: Any,
    optimizer_groups: Sequence[Mapping[str, Any]],
) -> Sequence[Sequence[torch.Tensor]]:
    """Return model-owned parameters before ZeRO replaces groups with flat partitions."""
    accelerator = trainer.accelerator
    deepspeed_plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    if getattr(accelerator, "distributed_type", None) != DistributedType.DEEPSPEED or getattr(
        deepspeed_plugin, "zero_stage", None
    ) not in (1, 2):
        return tuple(group.get("params") for group in optimizer_groups)

    deepspeed_optimizer = getattr(trainer.optimizer, "optimizer", None)
    logical_groups = getattr(deepspeed_optimizer, "bit16_groups", None)
    if not isinstance(logical_groups, Sequence) or isinstance(logical_groups, (str, bytes)):
        raise TypeError(
            "DeepSpeed ZeRO-1/2 optimizer schema requires the logical model parameter "
            f"groups from optimizer.bit16_groups, received {type(logical_groups).__name__}: "
            f"{logical_groups!r}"
        )
    if len(logical_groups) != len(optimizer_groups):
        raise ValueError(
            "DeepSpeed ZeRO-1/2 optimizer schema expected logical and partitioned group "
            f"counts to match, received {len(logical_groups)} and {len(optimizer_groups)}"
        )
    return logical_groups


def _optimizer_type_chain(optimizer: Any) -> list[str]:
    """Describe transparent optimizer wrappers without following cycles."""
    names = []
    seen: set[int] = set()
    current = optimizer
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(_qualified_type_name(type(current)))
        children = getattr(current, "optimizers", None)
        if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
            names.extend(
                f"child[{index}]={_qualified_type_name(type(child))}"
                for index, child in enumerate(children)
            )
        nested = getattr(current, "optimizer", None)
        current = nested if nested is not current else None
    return names


def _canonical_optimizer_value(value: Any, path: str) -> Any:
    """Convert optimizer group configuration into strict canonical JSON values."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"optimizer schema value at {path} must be finite")
        return value
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, torch.device):
        return {"torch_device": str(value)}
    if isinstance(value, tuple):
        return {
            "tuple": [
                _canonical_optimizer_value(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        }
    if isinstance(value, list):
        return [
            _canonical_optimizer_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value):
            if type(key) is not str:
                raise TypeError(
                    f"optimizer schema mapping key at {path} must be str, "
                    f"received {type(key).__name__}: {key!r}"
                )
            result[key] = _canonical_optimizer_value(value[key], f"{path}.{key}")
        return result
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if type(module) is str and type(qualname) is str:
            return {"callable": f"{module}.{qualname}"}
    raise TypeError(
        f"unsupported optimizer schema value at {path}: " f"{type(value).__name__}: {value!r}"
    )


def _schema_digest(schema: Any) -> str:
    """Hash one canonical JSON schema with explicit stable separators."""
    payload = json.dumps(
        schema,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _qualified_type_name(type_: type) -> str:
    """Return one import-qualified concrete type name."""
    return f"{type_.__module__}.{type_.__qualname__}"


def _require_non_empty_string(value: Any, field_name: str) -> str:
    """Require a concrete non-empty identity string."""
    if type(value) is not str or not value:
        raise TypeError(
            f"trainer runtime identity {field_name} must be a non-empty str, "
            f"received {type(value).__name__}: {value!r}"
        )
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    """Require a concrete positive identity integer."""
    if type(value) is not int or value < 1:
        raise TypeError(
            f"trainer runtime identity {field_name} must be a positive int, "
            f"received {type(value).__name__}: {value!r}"
        )
    return value


__all__ = [
    "build_default_data_identity_payload",
    "build_default_execution_identity_payload",
    "build_trainer_runtime_identity",
]
