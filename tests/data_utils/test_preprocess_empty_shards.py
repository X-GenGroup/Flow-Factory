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
from typing import Any, Dict, List

import torch
from datasets import Dataset as HFDataset

from flow_factory.data_utils.dataset import GeneralDataset


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
