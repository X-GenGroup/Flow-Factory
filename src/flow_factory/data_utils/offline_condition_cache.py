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

"""Project V2 inputs into model condition preprocessing and caching.

Offline supervision has a deliberately separate lifecycle from input
conditions. This module exposes only ``NormalizedDatasetRecord.model_input`` to
``GeneralDataset``. Target, chosen, rejected, and record metadata are never
inserted into the raw Arrow table and therefore cannot leak into an adapter or
invalidate an input-condition cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from datasets import Dataset as HFDataset

from ..contracts import (
    BatchCapability,
    InputMediaBinding,
    MediaRepresentation,
    NegativePromptPolicy,
    PipelineIOContract,
    resolve_pipeline_input_media_slots,
)
from ..samples.references import canonicalize_reference_manifest
from .dataset import (
    METADATA_COLUMN,
    GeneralDataset,
    PreprocessCallable,
    _supports_ordered_references,
)
from .offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    compute_offline_condition_id,
)
from .schema import MediaAsset, NormalizedDatasetRecord

_CONDITION_SOURCE_FORMAT = "flow-factory-offline-condition-v4"


def project_offline_condition_dataset(
    records: Sequence[NormalizedDatasetRecord],
    *,
    source_name: str,
    ordered_references: bool,
    pipeline_io_contract: PipelineIOContract | None = None,
    _media_digest_cache: MutableMapping[str, str] | None = None,
) -> HFDataset:
    """Build an input-only raw dataset for ``GeneralDataset`` preprocessing.

    Grouped adapters receive ``prompt`` plus per-modality ``images``, ``videos``,
    and ``audios`` columns. Ordered-reference adapters receive one canonical JSON
    string per row instead of an Arrow list-of-struct column. The string preserves
    validated ``type`` entries through the adapter preprocess boundary, avoiding
    Arrow's heterogeneous-struct null-key expansion.
    """
    if not isinstance(ordered_references, bool):
        raise TypeError(
            "ordered_references must be a bool, " f"got {type(ordered_references).__name__}"
        )
    if pipeline_io_contract is not None and not isinstance(
        pipeline_io_contract, PipelineIOContract
    ):
        raise TypeError(
            "pipeline_io_contract must be a PipelineIOContract or None, "
            f"got {type(pipeline_io_contract).__name__}"
        )
    if pipeline_io_contract is not None:
        expected_ordered_references = (
            pipeline_io_contract.input_media.binding is InputMediaBinding.ORDERED_REFERENCES
        )
        if ordered_references != expected_ordered_references:
            raise ValueError(
                "offline condition projection binding disagrees with pipeline contract: "
                f"ordered_references={ordered_references}, "
                f"contract={pipeline_io_contract.input_media.binding.value!r}"
            )
    stable_records = tuple(records)
    if not stable_records:
        raise ValueError("offline condition projection requires at least one record")
    for index, record in enumerate(stable_records):
        if not isinstance(record, NormalizedDatasetRecord):
            raise TypeError(
                "offline condition projection accepts normalized V2 records only, "
                f"got {type(record).__name__} at index {index}"
            )

    media_digest_cache: MutableMapping[str, str] = (
        {} if _media_digest_cache is None else _media_digest_cache
    )
    condition_ids = [
        compute_offline_condition_id(
            record,
            index=index,
            source_name=source_name,
            _media_digest_cache=media_digest_cache,
        )
        for index, record in enumerate(stable_records)
    ]
    columns: Dict[str, List[Any]] = {
        "prompt": [record.model_input.prompt for record in stable_records],
        OFFLINE_CONDITION_ID_COLUMN: condition_ids,
    }
    resolved_slots = []
    for record in stable_records:
        if pipeline_io_contract is None:
            slots = tuple(None for _ in record.model_input.media)
            if any(asset.slot is not None for asset in record.model_input.media):
                raise ValueError(
                    "offline condition projection requires pipeline_io_contract when "
                    "input media declares semantic slots"
                )
        else:
            slots = resolve_pipeline_input_media_slots(
                record.model_input,
                pipeline_io_contract,
            )
        resolved_slots.append(slots)

    negative_prompts = [record.model_input.negative_prompt for record in stable_records]
    projects_negative_prompt = (
        pipeline_io_contract is not None
        and pipeline_io_contract.negative_prompt is not NegativePromptPolicy.UNSUPPORTED
    )
    if projects_negative_prompt or any(value is not None for value in negative_prompts):
        # Adapter tokenizers consume a homogeneous text batch. In a mixed V2
        # batch, an omitted optional negative prompt is semantically the empty
        # prompt, not a tokenizer-level ``None`` value.
        columns["negative_prompt"] = [
            value if value is not None else "" for value in negative_prompts
        ]

    if ordered_references:
        columns["references"] = [
            canonicalize_reference_manifest(
                [_to_ordered_reference(asset) for asset in record.model_input.media],
                row_index=index,
            )
            for index, record in enumerate(stable_records)
        ]
    else:
        grouped_rows = [
            _group_media_by_type_and_slot(
                record.model_input.media,
                slots,
                pipeline_io_contract,
            )
            for record, slots in zip(stable_records, resolved_slots)
        ]
        grouped_columns = {
            "images": [[asset.path for asset, _ in row["image"]] for row in grouped_rows],
            "videos": [
                [_to_grouped_rate_spec(asset, rate_name="fps") for asset, _ in row["video"]]
                for row in grouped_rows
            ],
            "audios": [
                [_to_grouped_rate_spec(asset, rate_name="sample_rate") for asset, _ in row["audio"]]
                for row in grouped_rows
            ],
            "image_slots": [[slot for _, slot in row["image"]] for row in grouped_rows],
            "video_slots": [[slot for _, slot in row["video"]] for row in grouped_rows],
            "audio_slots": [[slot for _, slot in row["audio"]] for row in grouped_rows],
        }
        declared_rules = (
            {}
            if pipeline_io_contract is None
            else {rule.format.type.value: rule for rule in pipeline_io_contract.input_media.rules}
        )
        column_media_types = {
            "images": "image",
            "videos": "video",
            "audios": "audio",
            "image_slots": "image",
            "video_slots": "video",
            "audio_slots": "audio",
        }
        for column_name, values in grouped_columns.items():
            media_type = column_media_types[column_name]
            rule = declared_rules.get(media_type)
            if column_name.endswith("_slots"):
                include = rule is not None and bool(rule.slots)
            else:
                include = rule is not None if pipeline_io_contract is not None else any(values)
            if include:
                columns[column_name] = values

    return HFDataset.from_dict(columns)


def compute_offline_condition_source_hash(
    condition_ids: Sequence[str],
    *,
    pipeline_io_contract: PipelineIOContract | None = None,
) -> str:
    """Hash ordered inputs and their effective projection contract."""
    stable_ids = tuple(condition_ids)
    if not stable_ids:
        raise ValueError("offline condition source hash requires at least one condition id")
    for index, condition_id in enumerate(stable_ids):
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError(
                "offline condition ids must be non-empty strings, "
                f"got {condition_id!r} at index {index}"
            )
    if pipeline_io_contract is not None and not isinstance(
        pipeline_io_contract,
        PipelineIOContract,
    ):
        raise TypeError(
            "pipeline_io_contract must be a PipelineIOContract or None, "
            f"got {type(pipeline_io_contract).__name__}"
        )
    payload = json.dumps(
        {
            "format": _CONDITION_SOURCE_FORMAT,
            "condition_ids": stable_ids,
            "input_projection_contract": _input_projection_contract_identity(pipeline_io_contract),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_offline_condition_cache(
    records: Sequence[NormalizedDatasetRecord],
    *,
    source_name: str,
    dataset_dir: str | os.PathLike[str],
    preprocess_func: PreprocessCallable,
    preprocess_kwargs: Mapping[str, Any] | None = None,
    pipeline_io_contract: PipelineIOContract | None = None,
    cache_dir: str | os.PathLike[str] = "~/.cache/flow_factory/datasets",
    force_reprocess: bool = False,
    preprocessing_batch_size: int | None = None,
    extra_hash_strs: Sequence[str] | None = None,
    target_arrow_path: str | os.PathLike[str] | None = None,
) -> HFDataset:
    """Preprocess and cache only offline input conditions.

    The default Arrow target is derived from the same input-only merged-cache
    fingerprint used by ``GeneralDataset``. Rebuilding from records whose target
    or metadata changed therefore loads the existing cache without invoking the
    adapter again. Callers orchestrating distributed preprocessing may provide a
    rank-specific ``target_arrow_path`` instead.
    """
    if not callable(preprocess_func):
        raise TypeError(
            "offline condition cache requires a callable preprocess_func, "
            f"got {type(preprocess_func).__name__}"
        )
    ordered_references = _supports_ordered_references(preprocess_func)
    raw_dataset = project_offline_condition_dataset(
        records,
        source_name=source_name,
        ordered_references=ordered_references,
        pipeline_io_contract=pipeline_io_contract,
    )
    condition_ids = tuple(raw_dataset[OFFLINE_CONDITION_ID_COLUMN])
    source_hash = compute_offline_condition_source_hash(
        condition_ids,
        pipeline_io_contract=pipeline_io_contract,
    )
    normalized_dataset_dir = os.path.expanduser(os.fspath(dataset_dir))
    normalized_cache_dir = os.path.expanduser(os.fspath(cache_dir))
    normalized_preprocess_kwargs = dict(preprocess_kwargs or {})
    normalized_extra_hash_strs = list(extra_hash_strs or ())

    requires_single_sample_batches = ordered_references or (
        pipeline_io_contract is not None
        and pipeline_io_contract.batch_capability is BatchCapability.SINGLE_SAMPLE
    )
    if preprocessing_batch_size is None:
        preprocessing_batch_size = 1 if requires_single_sample_batches else 16
    if (
        not isinstance(preprocessing_batch_size, int)
        or isinstance(preprocessing_batch_size, bool)
        or preprocessing_batch_size <= 0
    ):
        raise ValueError(
            "preprocessing_batch_size must be a positive integer, "
            f"got {preprocessing_batch_size!r}"
        )
    if requires_single_sample_batches:
        preprocessing_batch_size = 1

    normalized_target_arrow_path: str
    if target_arrow_path is None:
        merged_cache_path = GeneralDataset.compute_cache_path(
            dataset_dir=normalized_dataset_dir,
            split="train",
            cache_dir=normalized_cache_dir,
            max_dataset_size=None,
            preprocess_func=preprocess_func,
            preprocess_kwargs=normalized_preprocess_kwargs,
            extra_hash_strs=normalized_extra_hash_strs,
            source_hash_override=source_hash,
        )
        normalized_target_arrow_path = f"{merged_cache_path}.arrow"
    else:
        normalized_target_arrow_path = os.path.expanduser(os.fspath(target_arrow_path))

    dataset_builder = GeneralDataset(
        dataset_dir=normalized_dataset_dir,
        split="train",
        cache_dir=normalized_cache_dir,
        preprocess_func=preprocess_func,
        preprocess_kwargs=normalized_preprocess_kwargs,
        preprocessing_batch_size=preprocessing_batch_size,
        force_reprocess=force_reprocess,
        extra_hash_strs=normalized_extra_hash_strs,
        image_dir=normalized_dataset_dir,
        video_dir=normalized_dataset_dir,
        audio_dir=normalized_dataset_dir,
        target_arrow_path=normalized_target_arrow_path,
        raw_dataset=raw_dataset,
        source_hash_override=source_hash,
        passthrough_columns=(OFFLINE_CONDITION_ID_COLUMN,),
    )
    condition_cache = dataset_builder.processed_dataset
    if METADATA_COLUMN in condition_cache.column_names:
        condition_cache = condition_cache.remove_columns(METADATA_COLUMN)
    return condition_cache


def _to_ordered_reference(asset: MediaAsset) -> Dict[str, Any]:
    reference: Dict[str, Any] = {"type": asset.type, "path": asset.path}
    if asset.type == "video" and asset.fps is not None:
        reference["fps"] = asset.fps
    elif asset.type == "audio" and asset.sample_rate is not None:
        reference["sample_rate"] = asset.sample_rate
    return reference


def _input_projection_contract_identity(
    contract: PipelineIOContract | None,
) -> Dict[str, Any] | None:
    """Return the canonical contract fields that can alter input preprocessing."""
    if contract is None:
        return None
    input_media = contract.input_media
    return {
        "binding": input_media.binding.value,
        "order": input_media.order.value,
        "min_total_count": input_media.min_total_count,
        "max_total_count": input_media.max_total_count,
        "required_any_types": [value.value for value in input_media.required_any_types],
        "rules": [
            {
                "type": rule.format.type.value,
                "fps": rule.format.fps.value,
                "sample_rate": rule.format.sample_rate.value,
                "representation": _media_representation_identity(rule.format.representation),
                "min_count": rule.min_count,
                "max_count": rule.max_count,
                "slots": list(rule.slots),
                "required_slots": list(rule.required_slots),
            }
            for rule in input_media.rules
        ],
        "negative_prompt": contract.negative_prompt.value,
        "batch_capability": contract.batch_capability.value,
    }


def _media_representation_identity(
    representation: MediaRepresentation,
) -> Dict[str, Any]:
    """Serialize every physical representation field that can alter preprocessing."""
    return {
        "container": representation.container.value,
        "layout": representation.layout.value,
        "dtype": representation.dtype.value,
        "value_range": {
            "minimum": representation.value_range.minimum,
            "maximum": representation.value_range.maximum,
            "finite": representation.value_range.finite,
        },
        "channels": representation.channels,
        "color_space": (
            None if representation.color_space is None else representation.color_space.value
        ),
        "device": representation.device.value,
        "requires_contiguous": representation.requires_contiguous,
        "requires_detached": representation.requires_detached,
    }


def _to_grouped_rate_spec(asset: MediaAsset, *, rate_name: str) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"path": asset.path}
    rate = getattr(asset, rate_name)
    if rate is not None:
        spec[rate_name] = rate
    return spec


def _group_media_by_type_and_slot(
    media: Sequence[MediaAsset],
    slots: Sequence[str | None],
    contract: PipelineIOContract | None,
) -> Dict[str, List[tuple[MediaAsset, str | None]]]:
    """Group one record and canonicalize slotted media into declaration order."""
    grouped: Dict[str, List[tuple[MediaAsset, str | None]]] = {
        "image": [],
        "video": [],
        "audio": [],
    }
    for asset, slot in zip(media, slots):
        grouped[asset.type].append((asset, slot))
    if contract is None:
        return grouped
    for rule in contract.input_media.rules:
        if not rule.slots:
            continue
        slot_order = {slot: index for index, slot in enumerate(rule.slots)}
        grouped[rule.format.type.value].sort(key=lambda item: slot_order[item[1]])
    return grouped


__all__ = [
    "build_offline_condition_cache",
    "compute_offline_condition_source_hash",
    "project_offline_condition_dataset",
]
