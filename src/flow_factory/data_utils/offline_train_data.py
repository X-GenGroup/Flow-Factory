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

"""Build finite offline train data from normalized V2 dataset sources."""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Mapping, MutableMapping, Optional, Sequence

import torch
from accelerate import Accelerator
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from ..contracts import (
    BatchCapability,
    InputMediaBinding,
    PipelineIOContract,
    validate_pipeline_model_input,
    validate_pipeline_output_candidate,
)
from ..hparams import Arguments
from ..utils.base import filter_kwargs
from ..utils.dist import data_parallel_geometry
from .dataset import (
    METADATA_COLUMN,
    PreprocessCallable,
    _supports_ordered_references,
)
from .loader import _create_or_load_dataset
from .offline_condition_cache import (
    compute_offline_condition_source_hash,
    project_offline_condition_dataset,
)
from .offline_dataset import (
    DEFAULT_MEDIA_DECODERS,
    OFFLINE_CONDITION_ID_COLUMN,
    MediaDecoder,
    OfflineDataset,
    OfflineSupervisionType,
    load_offline_manifest,
)
from .offline_loader import build_offline_dataloader
from .schema import (
    DemonstrationSupervision,
    MediaAsset,
    MediaType,
    NormalizedDatasetRecord,
    PreferenceSupervision,
)


def build_offline_train_dataloader(
    config: Arguments,
    accelerator: Accelerator,
    preprocess_func: PreprocessCallable,
    *,
    supervision_type: OfflineSupervisionType,
    pipeline_io_contract: PipelineIOContract,
    preprocess_kwargs: Optional[Mapping[str, Any]] = None,
    extra_hash_strs: Optional[Sequence[str]] = None,
    media_decoders: Optional[Mapping[MediaType, MediaDecoder]] = None,
    shuffle: bool = True,
    sampler_drop_last: bool = False,
    batch_drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """Build one already-distributed offline train dataloader from config sources.

    Every enabled training source is parsed from
    ``{dataset_dir}/{train.split}.jsonl`` as homogeneous V2 supervision. Only the
    normalized input projection enters distributed Arrow preprocessing; target,
    chosen, rejected, and metadata fields remain attached to ``OfflineDataset`` and
    are decoded on demand.

    Args:
        config: Fully resolved framework arguments.
        accelerator: Accelerator used only for rank metadata and cache barriers.
        preprocess_func: Adapter input-preprocessing callable.
        supervision_type: Homogeneous supervision required from every source.
        pipeline_io_contract: Adapter-owned input/output declaration used to reject
            unsupported conditions before preprocessing.
        preprocess_kwargs: Optional overrides layered onto config-derived train
            preprocessing arguments.
        extra_hash_strs: Additional input-cache fingerprint components.
        media_decoders: Optional module-level target decoders by media type.
        shuffle: Whether the official ``DistributedSampler`` shuffles each epoch.
        sampler_drop_last: Whether the sampler drops its non-divisible rank tail.
        batch_drop_last: Whether the DataLoader drops a rank-local partial batch.
        pin_memory: Whether the DataLoader pins CPU tensors. Defaults to ``False``
            for macOS and MPS safety.

    Returns:
        A finite DataLoader whose dataset retains only normalized records and
        detached Hugging Face condition datasets, never a bound adapter wrapper.

    Raises:
        TypeError: If callable, mapping, or source fields have invalid types.
        ValueError: If no source is enabled or offline source contracts disagree.
        FileNotFoundError: If an enabled source split JSONL does not exist.
        NotImplementedError: If target supervision needs an unavailable decoder.

    Note:
        The returned loader is already sharded by PyTorch ``DistributedSampler``
        and must not be passed to ``Accelerator.prepare``.
    """
    if not callable(preprocess_func):
        raise TypeError(
            "offline train data requires a callable preprocess_func, "
            f"got {type(preprocess_func).__name__}"
        )
    if not isinstance(pipeline_io_contract, PipelineIOContract):
        raise TypeError(
            "offline train data requires a PipelineIOContract, "
            f"got {type(pipeline_io_contract).__name__}"
        )
    ordered_references = _supports_ordered_references(preprocess_func)
    expected_ordered_references = (
        pipeline_io_contract.input_media.binding is InputMediaBinding.ORDERED_REFERENCES
    )
    if ordered_references != expected_ordered_references:
        raise ValueError(
            "offline input preprocessing binding disagrees with pipeline contract: "
            f"contract={pipeline_io_contract.input_media.binding.value!r}, "
            f"preprocess supports_ordered_references={ordered_references}"
        )
    if preprocess_kwargs is not None and not isinstance(preprocess_kwargs, Mapping):
        raise TypeError(
            "preprocess_kwargs must be a mapping or None, "
            f"got {type(preprocess_kwargs).__name__}"
        )
    if extra_hash_strs is not None and isinstance(extra_hash_strs, (str, bytes)):
        raise TypeError("extra_hash_strs must be a sequence of strings, not a string")
    if media_decoders is not None and not isinstance(media_decoders, Mapping):
        raise TypeError(
            "media_decoders must be a mapping or None, " f"got {type(media_decoders).__name__}"
        )

    data_args = config.data_args
    training_args = config.training_args
    if (
        pipeline_io_contract.batch_capability is BatchCapability.SINGLE_SAMPLE
        and training_args.per_device_batch_size != 1
    ):
        raise ValueError(
            "offline pipeline contract requires per_device_batch_size=1 for "
            f"single-sample execution, received {training_args.per_device_batch_size!r}"
        )
    if not data_args.enable_preprocess:
        raise ValueError(
            "offline train data requires data.enable_preprocess=True so cached rows contain "
            "adapter input conditions rather than raw V2 projections"
        )
    training_sources = tuple(data_args.training_datasets)
    _validate_training_sources(training_sources)

    normalized_extra_hash_strs = _build_extra_hash_strs(config, extra_hash_strs)
    normalized_preprocess_kwargs = _build_preprocess_kwargs(
        config,
        preprocess_func,
        preprocess_kwargs,
    )
    available_decoder_types = _available_decoder_types(media_decoders)

    source_records = []
    for source in training_sources:
        train_spec = source.train
        if train_spec is None:
            raise RuntimeError(
                f"enabled offline source {source.name!r} unexpectedly has train=None"
            )
        dataset_dir = os.path.abspath(os.path.expanduser(os.fspath(source.dataset_dir)))
        manifest_path = os.path.join(dataset_dir, f"{train_spec.split}.jsonl")
        records = load_offline_manifest(
            manifest_path,
            supervision_type=supervision_type,
            dataset_dir=dataset_dir,
        )
        max_dataset_size = (
            train_spec.max_dataset_size
            if train_spec.max_dataset_size is not None
            else data_args.max_dataset_size
        )
        records = _slice_records(records, max_dataset_size, source_name=source.name)
        _validate_pipeline_inputs(
            records,
            contract=pipeline_io_contract,
            source_name=source.name,
        )
        _validate_pipeline_outputs(
            records,
            contract=pipeline_io_contract,
            source_name=source.name,
        )
        _require_decoder_coverage(
            records,
            available_decoder_types,
            source_name=source.name,
        )
        source_records.append((source, dataset_dir, records))

    offline_datasets = []
    for source, dataset_dir, records in source_records:
        media_digest_cache: MutableMapping[str, str] = {}
        condition_cache = _build_distributed_condition_cache(
            records,
            source_name=source.name,
            split=source.train.split,
            dataset_dir=dataset_dir,
            cache_dir=data_args.cache_dir,
            preprocess_func=preprocess_func,
            preprocess_kwargs=normalized_preprocess_kwargs,
            preprocessing_batch_size=data_args.preprocessing_batch_size,
            pipeline_io_contract=pipeline_io_contract,
            force_reprocess=data_args.force_reprocess,
            extra_hash_strs=[*normalized_extra_hash_strs, f"offline_train_source:{source.name}"],
            preprocess_parallelism=data_args.preprocess_parallelism,
            accelerator=accelerator,
            _media_digest_cache=media_digest_cache,
        )
        offline_datasets.append(
            OfflineDataset(
                records,
                condition_cache,
                source_name=source.name,
                source_id=source.source_id,
                supervision_type=supervision_type,
                media_decoders=media_decoders,
                _media_digest_cache=media_digest_cache,
            )
        )

    # Ranks of one tensor-parallel group consume the same sample, so the official
    # DistributedSampler must be sized by the data-parallel group, not the world.
    dp_world_size, dp_rank = data_parallel_geometry(accelerator)
    return build_offline_dataloader(
        offline_datasets,
        source_weights=[source.train.weight for source in training_sources],
        batch_size=training_args.per_device_batch_size,
        num_replicas=dp_world_size,
        rank=dp_rank,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_workers=data_args.dataloader_num_workers,
        shuffle=shuffle,
        seed=training_args.seed,
        sampler_drop_last=sampler_drop_last,
        batch_drop_last=batch_drop_last,
        pin_memory=pin_memory,
    )


def _validate_pipeline_inputs(
    records: Sequence[NormalizedDatasetRecord],
    *,
    contract: PipelineIOContract,
    source_name: str,
) -> None:
    """Validate every normalized input before any condition cache can be reused."""
    for row_index, record in enumerate(records):
        try:
            validate_pipeline_model_input(record.model_input, contract)
        except (TypeError, ValueError) as exc:
            error_type = type(exc)
            raise error_type(
                f"offline source {source_name!r} row {row_index} violates its pipeline "
                f"input contract: {exc}"
            ) from exc


def _validate_pipeline_outputs(
    records: Sequence[NormalizedDatasetRecord],
    *,
    contract: PipelineIOContract,
    source_name: str,
) -> None:
    """Validate every supervision candidate before condition preprocessing."""
    for row_index, record in enumerate(records):
        supervision = record.supervision
        if isinstance(supervision, DemonstrationSupervision):
            candidates = (("target", supervision.target.media),)
        elif isinstance(supervision, PreferenceSupervision):
            candidates = (
                ("chosen", supervision.chosen.media),
                ("rejected", supervision.rejected.media),
            )
        else:
            raise RuntimeError("normalized offline record unexpectedly lacks supervision")
        for candidate_name, media in candidates:
            try:
                validate_pipeline_output_candidate(media, contract)
            except (TypeError, ValueError) as exc:
                error_type = type(exc)
                raise error_type(
                    f"offline source {source_name!r} row {row_index} {candidate_name} "
                    f"violates its pipeline output contract: {exc}"
                ) from exc


def _build_distributed_condition_cache(
    records: Sequence[NormalizedDatasetRecord],
    *,
    source_name: str,
    split: str,
    dataset_dir: str,
    cache_dir: str,
    preprocess_func: PreprocessCallable,
    preprocess_kwargs: Mapping[str, Any],
    preprocessing_batch_size: int,
    pipeline_io_contract: PipelineIOContract,
    force_reprocess: bool,
    extra_hash_strs: Sequence[str],
    preprocess_parallelism: Literal["global", "local"],
    accelerator: Accelerator,
    _media_digest_cache: MutableMapping[str, str] | None = None,
) -> HFDataset:
    """Build one input-only cache through the existing rank-safe orchestrator."""
    ordered_references = _supports_ordered_references(preprocess_func)
    effective_batch_size = (
        1
        if ordered_references
        or pipeline_io_contract.batch_capability is BatchCapability.SINGLE_SAMPLE
        else preprocessing_batch_size
    )
    raw_dataset = project_offline_condition_dataset(
        records,
        source_name=source_name,
        ordered_references=ordered_references,
        pipeline_io_contract=pipeline_io_contract,
        _media_digest_cache=_media_digest_cache,
    )
    condition_ids = tuple(raw_dataset[OFFLINE_CONDITION_ID_COLUMN])
    source_hash = compute_offline_condition_source_hash(
        condition_ids,
        pipeline_io_contract=pipeline_io_contract,
    )
    dataset_builder = _create_or_load_dataset(
        split=split,
        accelerator=accelerator,
        base_kwargs={
            "dataset_dir": dataset_dir,
            "cache_dir": cache_dir,
            "enable_preprocess": True,
            "force_reprocess": force_reprocess,
            "preprocessing_batch_size": effective_batch_size,
            "max_dataset_size": None,
            "preprocess_func": preprocess_func,
            "preprocess_kwargs": dict(preprocess_kwargs),
            "extra_hash_strs": list(extra_hash_strs),
            "image_dir": dataset_dir,
            "video_dir": dataset_dir,
            "audio_dir": dataset_dir,
            "raw_dataset": raw_dataset,
            "source_hash_override": source_hash,
            "passthrough_columns": (OFFLINE_CONDITION_ID_COLUMN,),
        },
        enable_distributed=accelerator.num_processes > 1,
        preprocess_parallelism=preprocess_parallelism,
    )
    condition_cache = dataset_builder.processed_dataset
    if METADATA_COLUMN in condition_cache.column_names:
        condition_cache = condition_cache.remove_columns(METADATA_COLUMN)
    return condition_cache


def _build_preprocess_kwargs(
    config: Arguments,
    preprocess_func: PreprocessCallable,
    overrides: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    data_args = config.data_args
    training_args = config.training_args
    kwargs = filter_kwargs(preprocess_func, **data_args)
    kwargs.update({"is_train": True, **training_args})
    kwargs["guidance_scale"] = training_args.get_preprocess_guidance_scale()
    if overrides is not None:
        kwargs.update(overrides)
    return filter_kwargs(preprocess_func, **kwargs)


def _build_extra_hash_strs(
    config: Arguments,
    extra_hash_strs: Optional[Sequence[str]],
) -> list[str]:
    model_args = config.model_args
    precision_policy = json.dumps(
        {
            "component_load_dtypes": _canonical_dtype_policy(
                getattr(model_args, "component_load_dtypes", None)
            ),
            "frozen_parameters_dtype": _canonical_dtype_policy(
                getattr(model_args, "frozen_parameters_dtype", None)
            ),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    values = [
        model_args.model_type,
        model_args.model_name_or_path,
        f"preprocess_precision:{precision_policy}",
    ]
    if extra_hash_strs is not None:
        values.extend(extra_hash_strs)
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"offline cache hash component {index} must be str, "
                f"got {type(value).__name__}: {value!r}"
            )
    return values


def _canonical_dtype_policy(value: Any) -> Any:
    """Return stable JSON data for one normalized model dtype policy."""
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return str(value).split(".")[-1]
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        canonical = {}
        for selector in sorted(value):
            if not isinstance(selector, str) or not selector:
                raise TypeError(
                    "offline preprocessing dtype policy selectors must be non-empty strings, "
                    f"got {selector!r}"
                )
            canonical[selector] = _canonical_dtype_policy(value[selector])
        return canonical
    raise TypeError(
        "offline preprocessing dtype policy must be a dtype, string, mapping, or None, "
        f"got {type(value).__name__}: {value!r}"
    )


def _validate_training_sources(sources: Sequence[Any]) -> None:
    if not sources:
        raise ValueError("offline train data requires at least one enabled data.datasets source")
    names = set()
    source_ids = set()
    for index, source in enumerate(sources):
        if source.train is None or not source.train.enabled:
            raise ValueError(f"offline source {index} is not enabled for training")
        if type(source.train.weight) not in (int, float) or source.train.weight != 1:
            raise ValueError(
                f"offline source {source.name!r} requires train.weight=1, "
                f"got {source.train.weight!r}"
            )
        if not isinstance(source.train.split, str) or not source.train.split:
            raise ValueError(
                f"offline source {source.name!r} requires a non-empty train.split string"
            )
        if not isinstance(source.source_id, int) or isinstance(source.source_id, bool):
            raise ValueError(
                f"offline source {source.name!r} requires a resolved integer source_id, "
                f"got {source.source_id!r}"
            )
        if source.source_id < 0:
            raise ValueError(
                f"offline source {source.name!r} requires source_id >= 0, got {source.source_id}"
            )
        if source.name in names:
            raise ValueError(f"offline source name {source.name!r} is duplicated")
        if source.source_id in source_ids:
            raise ValueError(f"offline source_id {source.source_id} is duplicated")
        names.add(source.name)
        source_ids.add(source.source_id)


def _slice_records(
    records: Sequence[NormalizedDatasetRecord],
    max_dataset_size: Optional[int],
    *,
    source_name: str,
) -> tuple[NormalizedDatasetRecord, ...]:
    if max_dataset_size is None:
        return tuple(records)
    if type(max_dataset_size) is not int or max_dataset_size < 1:
        raise ValueError(
            f"offline source {source_name!r} max_dataset_size must be an int >= 1 or None, "
            f"got {max_dataset_size!r}"
        )
    return tuple(records[:max_dataset_size])


def _available_decoder_types(
    media_decoders: Optional[Mapping[MediaType, MediaDecoder]],
) -> frozenset[MediaType]:
    available = set(DEFAULT_MEDIA_DECODERS)
    if media_decoders is not None:
        for media_type in media_decoders:
            if media_type not in ("image", "video", "audio"):
                raise ValueError(f"unsupported offline media decoder type: {media_type!r}")
            available.add(media_type)
    return frozenset(available)


def _require_decoder_coverage(
    records: Sequence[NormalizedDatasetRecord],
    available_decoder_types: frozenset[MediaType],
    *,
    source_name: str,
) -> None:
    for asset in _iter_supervision_assets(records):
        if asset.type not in available_decoder_types:
            raise NotImplementedError(
                f"offline source {source_name!r} has no decoder for target media type "
                f"{asset.type!r} at {asset.path!r}"
            )


def _iter_supervision_assets(
    records: Sequence[NormalizedDatasetRecord],
) -> Sequence[MediaAsset]:
    assets = []
    for record in records:
        supervision = record.supervision
        if isinstance(supervision, DemonstrationSupervision):
            assets.extend(supervision.target.media)
        elif isinstance(supervision, PreferenceSupervision):
            assets.extend(supervision.chosen.media)
            assets.extend(supervision.rejected.media)
        else:
            raise RuntimeError("normalized offline record unexpectedly lacks supervision")
    return tuple(assets)


__all__ = ["build_offline_train_dataloader"]
