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

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from datasets import Dataset as HFDataset

from flow_factory.data_utils.dataset import GeneralDataset
from flow_factory.data_utils.loader import _create_or_load_dataset


class PromptLengthPreprocessor:
    def preprocess(self, prompt: List[str]) -> Dict[str, Any]:
        return {"prompt_ids": [torch.tensor([len(text)]) for text in prompt]}


def _build_rank_parts(dataset_dir: Path, cache_dir: Path, num_shards: int) -> str:
    preprocessor = PromptLengthPreprocessor()
    merged_cache_path = GeneralDataset.compute_cache_path(
        dataset_dir=str(dataset_dir),
        split="train",
        cache_dir=str(cache_dir),
        max_dataset_size=None,
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={},
    )
    for rank in range(num_shards):
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            split="train",
            cache_dir=str(cache_dir),
            force_reprocess=True,
            preprocessing_batch_size=2,
            preprocess_func=preprocessor.preprocess,
            num_shards=num_shards,
            shard_index=rank,
            target_arrow_path=GeneralDataset.build_part_arrow_path(
                merged_cache_path, rank, num_shards
            ),
        )
    return merged_cache_path


def test_fewer_rows_than_ranks_still_consolidate_every_row_in_order(tmp_path: Path) -> None:
    """Give row-less ranks an empty part so consolidation keeps every row in order."""
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    rows = "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in ("a", "bbb"))
    (dataset_dir / "train.jsonl").write_text(rows)

    merged_cache_path = _build_rank_parts(dataset_dir, tmp_path / "cache", num_shards=4)
    GeneralDataset.consolidate_parts(merged_cache_path, 4, split="train")
    merged = GeneralDataset.load_merged(merged_cache_path).processed_dataset

    assert [int(row["prompt_ids"][0]) for row in merged] == [1, 3]
    build_part = Path(GeneralDataset.build_part_arrow_path(merged_cache_path, 0, 4))
    consolidated_part = Path(merged_cache_path) / build_part.relative_to(merged_cache_path + ".tmp")
    empty_part = HFDataset.from_file(str(consolidated_part))
    assert len(empty_part) == 0
    assert empty_part.features == merged.features


@pytest.mark.parametrize("num_shards", [1, 4])
def test_whole_empty_source_round_trips_without_inventing_preprocessed_columns(
    tmp_path: Path, monkeypatch, num_shards: int
) -> None:
    """A whole empty source retains its input schema and never invokes the encoder."""
    dataset_dir = tmp_path / "empty"
    dataset_dir.mkdir()
    (dataset_dir / "train.txt").write_text("")

    def unexpected_preprocess(*args, **kwargs):
        raise AssertionError("a whole empty source has no rows to preprocess")

    monkeypatch.setattr(PromptLengthPreprocessor, "preprocess", unexpected_preprocess)
    cache_dir = tmp_path / "cache"
    merged_cache_path = _build_rank_parts(dataset_dir, cache_dir, num_shards)
    raw_features = HFDataset.from_dict({"prompt": []}).features
    for rank in range(num_shards):
        part = HFDataset.from_file(
            GeneralDataset.build_part_arrow_path(merged_cache_path, rank, num_shards)
        )
        assert len(part) == 0 and part.features == raw_features
        assert "prompt_ids" not in part.column_names

    GeneralDataset.consolidate_parts(merged_cache_path, num_shards, split="train")
    merged = GeneralDataset.load_merged(merged_cache_path).processed_dataset
    assert len(merged) == 0 and merged.features == raw_features
    accelerator = SimpleNamespace(
        num_processes=num_shards, process_index=0, is_local_main_process=True
    )
    cached = _create_or_load_dataset(
        "train",
        accelerator,
        {
            "dataset_dir": str(dataset_dir),
            "cache_dir": str(cache_dir),
            "preprocess_func": PromptLengthPreprocessor().preprocess,
            "preprocess_kwargs": {},
        },
        enable_distributed=num_shards > 1,
    )
    assert len(cached) == 0 and cached.processed_dataset.features == raw_features


def test_single_rank_image_schema_survives_map_and_cache_reload(tmp_path):
    from PIL import Image

    directory = tmp_path / "images-source"
    directory.mkdir()
    image = directory / "input.png"
    Image.new("RGB", (12, 8), (20, 40, 60)).save(image)
    (directory / "train.jsonl").write_text(
        json.dumps({"prompt": "a bird", "images": [str(image)]}) + "\n"
    )
    dataset = GeneralDataset(
        dataset_dir=str(directory),
        cache_dir=str(tmp_path / "cache"),
        preprocessing_batch_size=1,
        preprocess_func=PromptLengthPreprocessor().preprocess,
    )
    # Both the immediate map and the merged-cache entrypoint must decode PIL.
    assert isinstance(dataset[0]["images"][0], Image.Image)
    dataset.processed_dataset.save_to_disk(tmp_path / "saved")
    reloaded = GeneralDataset.load_merged(str(tmp_path / "saved"))
    assert isinstance(reloaded[0]["images"][0], Image.Image)
    assert reloaded[0]["images"][0].getpixel((0, 0)) == (20, 40, 60)
