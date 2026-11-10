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

"""Preprocessing modules load only when some rank must preprocess a split."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from flow_factory.data_utils.loader import _create_or_load_dataset, deferred_preprocessing_modules


class RecordingPreprocessor:
    """Fails unless the preprocessing modules were loaded before it runs."""

    def __init__(self, events: List[str]) -> None:
        self.events = events

    def preprocess(self, prompt: List[str]) -> Dict[str, Any]:
        if "load" not in self.events:
            raise AssertionError("preprocessing ran before its modules were loaded")
        self.events.append("preprocess")
        return {"prompt_ids": [torch.tensor([len(text)]) for text in prompt]}


def _accelerator() -> SimpleNamespace:
    return SimpleNamespace(
        num_processes=1,
        process_index=0,
        is_main_process=True,
        is_local_main_process=True,
        device=torch.device("cpu"),
    )


def _base_kwargs(tmp_path: Path, preprocessor: RecordingPreprocessor, **overrides) -> dict:
    dataset_dir = tmp_path / "data"
    if not dataset_dir.exists():
        dataset_dir.mkdir()
        rows = "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in ("a", "bbb", "cc"))
        (dataset_dir / "train.jsonl").write_text(rows)
    return {
        "dataset_dir": str(dataset_dir),
        "cache_dir": str(tmp_path / "cache"),
        "preprocess_func": preprocessor.preprocess,
        "preprocess_kwargs": {},
        "preprocessing_batch_size": 2,
        **overrides,
    }


def _load(events: List[str]):
    def load() -> None:
        events.append("load")

    return load


def _ids(dataset) -> List[int]:
    return [int(row["prompt_ids"][0]) for row in dataset.processed_dataset]


def test_a_miss_loads_once_before_preprocessing_and_a_hit_never_loads(tmp_path: Path) -> None:
    events: List[str] = []
    preprocessor = RecordingPreprocessor(events)
    with deferred_preprocessing_modules(_load(events)):
        built = _create_or_load_dataset(
            "train", _accelerator(), _base_kwargs(tmp_path, preprocessor), False
        )
    assert events[0] == "load" and events.count("load") == 1 and "preprocess" in events

    events.clear()
    with deferred_preprocessing_modules(_load(events)):
        cached = _create_or_load_dataset(
            "train", _accelerator(), _base_kwargs(tmp_path, preprocessor), False
        )
    assert events == []
    assert _ids(cached) == _ids(built) == [1, 3, 2]


def test_force_reprocess_loads_even_with_a_merged_cache(tmp_path: Path) -> None:
    events: List[str] = []
    preprocessor = RecordingPreprocessor(events)
    with deferred_preprocessing_modules(_load(events)):
        _create_or_load_dataset(
            "train", _accelerator(), _base_kwargs(tmp_path, preprocessor), False
        )
    events.clear()
    with deferred_preprocessing_modules(_load(events)):
        _create_or_load_dataset(
            "train",
            _accelerator(),
            _base_kwargs(tmp_path, preprocessor, force_reprocess=True),
            False,
        )
    assert events[0] == "load" and "preprocess" in events


def test_disabled_preprocessing_never_loads(tmp_path: Path) -> None:
    events: List[str] = []
    preprocessor = RecordingPreprocessor(events)
    with deferred_preprocessing_modules(_load(events)):
        _create_or_load_dataset(
            "train",
            _accelerator(),
            _base_kwargs(tmp_path, preprocessor, enable_preprocess=False),
            False,
        )
    assert events == []


def test_without_a_registered_loader_the_caller_owns_the_modules(tmp_path: Path) -> None:
    events: List[str] = ["load"]
    preprocessor = RecordingPreprocessor(events)
    dataset = _create_or_load_dataset(
        "train", _accelerator(), _base_kwargs(tmp_path, preprocessor), False
    )
    assert events == ["load", "preprocess", "preprocess"] and _ids(dataset) == [1, 3, 2]
