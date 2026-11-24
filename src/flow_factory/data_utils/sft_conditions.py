"""Bind an SFT record's sampled conditioning to a cache row or an on-demand encoder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
from datasets import Dataset

from ..models.sft_dropout import CaptionDropoutPolicy, SFTConditionDropoutPolicy, record_draw
from ..utils.base import filter_kwargs
from .dataset import (
    EXTRA_PYTHON_FORMAT_COLUMNS,
    decode_preprocessing_inputs,
    format_preprocessed_columns,
)
from .offline_dataset import OFFLINE_CONDITION_ID_COLUMN, _collate_condition_mappings
from .schema import NormalizedDatasetRecord, NormalizedModelInput


def absent_caption_records(
    records: Sequence[NormalizedDatasetRecord],
    empty_prompt: str,
    *,
    share_across_references: bool = False,
) -> tuple[list[NormalizedDatasetRecord], list[int]]:
    """Index deterministic alternatives once per distinct retained input, in source order."""
    alternatives, positions, indices = [], {}, []
    for record in records:
        absent = replace(
            record.model_input,
            prompt=empty_prompt,
            media=() if share_across_references else record.model_input.media,
        )
        if absent not in positions:
            positions[absent] = len(alternatives)
            alternatives.append(replace(record, model_input=absent))
        indices.append(positions[absent])
    return alternatives, indices


@dataclass
class SFTConditionSource:
    """Worker-side state. All model components stay on the training rank."""

    policy: SFTConditionDropoutPolicy | None
    mode: float | str
    seed: int
    raw: bool
    dataset_dir: str
    ordered: bool
    alternatives: Dataset | None = None
    indices: tuple[int, ...] = ()
    epoch: int = 0

    def select(
        self,
        index: int,
        model_input: NormalizedModelInput,
        record_id: str,
        condition: dict[str, Any],
    ) -> dict[str, Any]:
        """Realize a sampled input while preserving the positive cache and supervision."""
        policy = self.policy
        mask = (
            policy.sample(self.mode, model_input, record_draw(self.seed, self.epoch, record_id))
            if policy is not None
            else frozenset()
        )
        if self.raw:
            batch = {k: [v] for k, v in condition.items()}
            if mask and isinstance(policy, CaptionDropoutPolicy):
                batch["prompt"] = [policy.empty_prompt]
            args = decode_preprocessing_inputs(
                batch,
                [index],
                image_dir=self.dataset_dir,
                video_dir=self.dataset_dir,
                audio_dir=self.dataset_dir,
                data_root=self.dataset_dir,
                ordered_references=self.ordered,
            )
            condition = {"_sft_raw": (batch, args)}
        elif mask and isinstance(policy, CaptionDropoutPolicy):
            if self.alternatives is None:
                raise RuntimeError(
                    "SFT caption dropout selected a record without an alternate cache"
                )
            alternative = self.alternatives[self.indices[index]]
            absent = {k: v for k, v in alternative.items() if k != OFFLINE_CONDITION_ID_COLUMN}
            condition = {**condition, **absent} if policy.share_across_references else absent
        if policy is not None:
            condition["_sft_mask"] = mask
        return condition


def _cache_tensor(value):
    if not isinstance(value, torch.Tensor) or not value.numel() or value.is_complex():
        return NotImplemented
    if value.dtype == torch.bool:
        return value
    return value.to(torch.float32 if value.is_floating_point() else torch.int64)


def _cache_value(value):
    """Keep large tensor payloads out of Arrow while reproducing its torch row formatter."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and value:
        items = [_cache_tensor(item) for item in value]
        if all(isinstance(item, torch.Tensor) for item in items):
            if all(item.shape == items[0].shape and item.dtype == items[0].dtype for item in items):
                return torch.stack(items)
            return items
    return _cache_tensor(value)


def _cached_row_types(conditions, python_columns):
    """Match the cache's tensor, ragged-list and scalar representations before collation."""
    keys = conditions[0].keys()
    if any(row.keys() != keys for row in conditions):
        raise ValueError("raw SFT conditions must expose identical fields within a batch")
    metadata = {}
    for key in keys - python_columns:
        values = [row[key] for row in conditions]
        formatted = [_cache_value(value) for value in values]
        if any(item is NotImplemented for item in formatted):
            # Infer a shared Arrow schema for the small nullable and nested metadata columns.
            metadata[key] = values
        else:
            for row, item in zip(conditions, formatted, strict=True):
                row[key] = item
    if metadata:
        for row, extra in zip(
            conditions, Dataset.from_dict(metadata).with_format("torch"), strict=True
        ):
            row.update(extra)


def encode_raw_sft_conditions(rows, preprocess_func, kwargs):
    """Encode only the sampled presentation and use the cached path's row collation."""
    conditions = []
    for batch, inputs in rows:
        encoded = preprocess_func(**filter_kwargs(preprocess_func, **{**inputs, **kwargs}))
        columns = {**batch, **format_preprocessed_columns(preprocess_func, encoded)}
        conditions.append({k: v[0] for k, v in columns.items()})
    owner = getattr(preprocess_func, "__self__", None)
    python_columns = {
        "images",
        *EXTRA_PYTHON_FORMAT_COLUMNS,
        *getattr(owner, "python_format_columns", ()),
    }
    _cached_row_types(conditions, python_columns)
    return _collate_condition_mappings(conditions)
