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

"""Materialize offline model inputs on demand using the same row format as the cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset

from ..utils.base import filter_kwargs
from .dataset import (
    EXTRA_PYTHON_FORMAT_COLUMNS,
    decode_preprocessing_inputs,
    format_preprocessed_columns,
)
from .offline_dataset import _collate_condition_mappings
from .schema import NormalizedModelInput


@dataclass
class RawOfflineConditionSource:
    """Decode input media in workers. The frozen model encoders stay on the training rank."""

    dataset_dir: str
    ordered: bool

    def select(
        self,
        index: int,
        model_input: NormalizedModelInput,
        record_id: str,
        condition: dict[str, Any],
    ) -> dict[str, Any]:
        """Decode the positive input projection for one record."""
        batch = {key: [value] for key, value in condition.items()}
        args = decode_preprocessing_inputs(
            batch,
            [index],
            image_dir=self.dataset_dir,
            video_dir=self.dataset_dir,
            audio_dir=self.dataset_dir,
            data_root=self.dataset_dir,
            ordered_references=self.ordered,
        )
        return {"_offline_raw": (batch, args)}


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
        raise ValueError("raw offline conditions must expose identical fields within a batch")
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


def encode_raw_offline_conditions(rows, preprocess_func, kwargs):
    """Encode the input presentation and use the cached path's row collation."""
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
