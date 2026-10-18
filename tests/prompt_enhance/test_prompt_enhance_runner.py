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

"""CPU tests for the ff-prompt-enhance dataset runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from PIL import Image

from flow_factory.data_utils.dataset import GeneralDataset
from flow_factory.prompt_enhance import (
    BasePromptEnhancer,
    PromptEnhanceArguments,
    get_prompt_enhancer_class,
    list_registered_prompt_enhancers,
    runner,
)


class _RecordingEnhancer(BasePromptEnhancer):
    uses_images = True
    supported_generation_keys = frozenset({"temperature"})

    def load(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def enhance(self, prompts, images, seeds):
        self.calls.append({"prompts": list(prompts), "seeds": list(seeds)})
        return [
            f"{prompt} | seed={seed} | images={len(row_images)}"
            for prompt, row_images, seed in zip(prompts, images, seeds)
        ]


def _single_process_state() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        process_index=0,
        num_processes=1,
        is_main_process=True,
        wait_for_everyone=lambda: None,
    )


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "source"
    (dataset_dir / "images").mkdir(parents=True)
    Image.new("RGB", (8, 6), (10, 20, 30)).save(dataset_dir / "images" / "ref.png")
    _write_jsonl(
        dataset_dir / "train.jsonl",
        [
            {"prompt": "a cat", "images": ["ref.png"], "tag": "first"},
            {"prompt": "a dog", "images": [], "tag": "second"},
            {"prompt": "a bird", "images": ["ref.png", "ref.png"], "tag": "third"},
        ],
    )
    (dataset_dir / "test.txt").write_text("a fox\n\na owl\n", encoding="utf-8")
    return dataset_dir


def _args(source_dataset: Path, output_dir: Path, **overrides: Any) -> PromptEnhanceArguments:
    values = {
        "enhancer": "recording",
        "input_dir": str(source_dataset),
        "output_dir": str(output_dir),
        "num_variants": 2,
        "batch_size": 2,
        "seed": 42,
    }
    values.update(overrides)
    return PromptEnhanceArguments.from_dict(values)


def test_run_writes_row_major_variants_with_source_prompt_and_media_links(
    source_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enhancers: List[_RecordingEnhancer] = []

    def build_enhancer(args, device):
        enhancer = _RecordingEnhancer(args, device)
        enhancers.append(enhancer)
        return enhancer

    monkeypatch.setattr(runner, "get_prompt_enhancer_class", lambda name: build_enhancer)
    output_dir = tmp_path / "enhanced"
    runner.run_prompt_enhancement(_args(source_dataset, output_dir), state=_single_process_state())

    train_rows = _read_jsonl(output_dir / "train.jsonl")
    # seed = 42 + variant * 3 + row, independent of the [0, 1] / [2] batching.
    expected = [
        ("a cat", "first", 0, 42, 1),
        ("a cat", "first", 1, 45, 1),
        ("a dog", "second", 0, 43, 0),
        ("a dog", "second", 1, 46, 0),
        ("a bird", "third", 0, 44, 2),
        ("a bird", "third", 1, 47, 2),
    ]
    assert len(train_rows) == len(expected)
    for row, (source, tag, variant, seed, num_images) in zip(train_rows, expected):
        assert row["prompt"] == f"{source} | seed={seed} | images={num_images}"
        assert row["source_prompt"] == source
        assert row["pe_variant"] == variant
        assert row["tag"] == tag

    test_rows = _read_jsonl(output_dir / "test.jsonl")
    assert [row["source_prompt"] for row in test_rows] == ["a fox", "a fox", "a owl", "a owl"]
    assert all(row["prompt"].endswith("images=0") for row in test_rows)

    assert (output_dir / "images").is_symlink()
    assert (output_dir / "images").resolve() == (source_dataset / "images").resolve()

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["num_variants"] == 2
    assert manifest["seed_formula"] == runner.SEED_FORMULA
    assert manifest["linked_media"] == ["images"]
    assert manifest["splits"] == {
        "train": {"source_rows": 3, "output_rows": 6},
        "test": {"source_rows": 2, "output_rows": 4},
    }
    assert len(enhancers) == 1


def test_enhanced_dataset_feeds_general_dataset_without_training_changes(
    source_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "get_prompt_enhancer_class", lambda name: _RecordingEnhancer)
    output_dir = tmp_path / "enhanced"
    runner.run_prompt_enhancement(
        _args(source_dataset, output_dir, num_variants=1, splits=["train"]),
        state=_single_process_state(),
    )
    seen: Dict[str, Any] = {}

    def preprocess(prompt, images=None):
        seen["prompt"] = list(prompt)
        seen["image_counts"] = [len(row_images) for row_images in images]
        return {"prompt_length": [len(text) for text in prompt]}

    dataset = GeneralDataset(
        dataset_dir=str(output_dir),
        split="train",
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocess,
        force_reprocess=True,
    )

    enhanced_rows = _read_jsonl(output_dir / "train.jsonl")
    assert seen["prompt"] == [row["prompt"] for row in enhanced_rows]
    assert seen["image_counts"] == [1, 0, 2]
    metadata = dataset[0]["metadata"]
    assert metadata["source_prompt"] == "a cat"
    assert metadata["pe_variant"] == 0


def _enhance_shard(enhancer, records, args, source_dataset, parts_dir, rank, world_size):
    return runner.enhance_split_shard(
        enhancer,
        records,
        split="train",
        image_dir=str(source_dataset / "images"),
        parts_dir=parts_dir,
        rank=rank,
        world_size=world_size,
        num_variants=args.num_variants,
        batch_size=4,
        base_seed=args.seed,
        fingerprint=runner.split_fingerprint(args, records),
    )


def test_two_rank_shards_merge_and_resume_skips_completed_pairs(
    source_dataset: Path,
    tmp_path: Path,
) -> None:
    records = runner.load_split_records(str(source_dataset), "train")
    args = _args(source_dataset, tmp_path / "out")
    parts_dir = tmp_path / "out" / runner.PARTS_DIRNAME
    parts_dir.mkdir(parents=True)
    header = {runner.FINGERPRINT_KEY: runner.split_fingerprint(args, records)}
    runner.part_file_path(parts_dir, "train", 0).write_text(
        json.dumps(header) + "\n" + json.dumps({"row": 0, "variant": 0, "prompt": "cached"}) + "\n",
        encoding="utf-8",
    )
    calls_by_rank = {}
    for rank in (0, 1):
        enhancer = _RecordingEnhancer(args, torch.device("cpu"))
        enhancer.load()
        _enhance_shard(enhancer, records, args, source_dataset, parts_dir, rank, world_size=2)
        calls_by_rank[rank] = [(call["prompts"], call["seeds"]) for call in enhancer.calls]

    # Rank 0 owns rows 0 and 2; (0, 0) was already in its part file. Seeds stay per row.
    assert calls_by_rank[0] == [(["a bird"], [44]), (["a cat", "a bird"], [45, 47])]
    assert calls_by_rank[1] == [(["a dog"], [43]), (["a dog"], [46])]

    written = runner.merge_split(
        records,
        split="train",
        parts_dir=parts_dir,
        world_size=2,
        num_variants=2,
        output_dir=tmp_path / "out",
    )
    rows = _read_jsonl(tmp_path / "out" / "train.jsonl")
    assert written == 6
    assert rows[0]["prompt"] == "cached"
    assert [(row["source_prompt"], row["pe_variant"]) for row in rows] == [
        ("a cat", 0),
        ("a cat", 1),
        ("a dog", 0),
        ("a dog", 1),
        ("a bird", 0),
        ("a bird", 1),
    ]


def test_resume_with_changed_settings_or_source_fails(source_dataset: Path, tmp_path: Path) -> None:
    records = runner.load_split_records(str(source_dataset), "train")
    parts_dir = tmp_path / "out" / runner.PARTS_DIRNAME
    args = _args(source_dataset, tmp_path / "out", num_variants=1)
    enhancer = _RecordingEnhancer(args, torch.device("cpu"))
    enhancer.load()
    _enhance_shard(enhancer, records, args, source_dataset, parts_dir, rank=0, world_size=1)

    reseeded = _args(source_dataset, tmp_path / "out", num_variants=1, seed=7)
    with pytest.raises(ValueError, match=r"different \['seed'\]"):
        _enhance_shard(enhancer, records, reseeded, source_dataset, parts_dir, 0, 1)

    edited = [dict(records[0], prompt="a lion")] + list(records[1:])
    with pytest.raises(ValueError, match=r"different \['source_sha256'\]"):
        _enhance_shard(enhancer, edited, args, source_dataset, parts_dir, 0, 1)

    more_variants = _args(source_dataset, tmp_path / "out", num_variants=2)
    enhancer.calls.clear()
    _enhance_shard(enhancer, records, more_variants, source_dataset, parts_dir, 0, 1)
    assert [call["prompts"] for call in enhancer.calls] == [["a cat", "a dog", "a bird"]]


def test_merge_requires_every_row_variant_pair(source_dataset: Path, tmp_path: Path) -> None:
    records = runner.load_split_records(str(source_dataset), "train")
    parts_dir = tmp_path / runner.PARTS_DIRNAME
    parts_dir.mkdir()
    runner.part_file_path(parts_dir, "train", 0).write_text(
        json.dumps({"row": 0, "variant": 0, "prompt": "only"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.merge_split(
            records,
            split="train",
            parts_dir=parts_dir,
            world_size=1,
            num_variants=1,
            output_dir=tmp_path,
        )


def test_structural_errors_fail_fast(source_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="generation keys"):
        _RecordingEnhancer(
            _args(source_dataset, tmp_path / "out", generation={"top_k": 5}),
            torch.device("cpu"),
        )
    with pytest.raises(FileNotFoundError, match="validation"):
        runner.resolve_splits(str(source_dataset), ["train", "validation"])
    assert runner.resolve_splits(str(source_dataset), None) == ["train", "test"]

    (source_dataset / "broken.jsonl").write_text(json.dumps({"text": "no prompt"}) + "\n")
    with pytest.raises(ValueError, match="no 'prompt' field"):
        runner.load_split_records(str(source_dataset), "broken")

    class _ShortEnhancer(_RecordingEnhancer):
        def enhance(self, prompts, images, seeds):
            return prompts[:-1]

    args = _args(source_dataset, tmp_path / "out")
    records = runner.load_split_records(str(source_dataset), "train")
    with pytest.raises(ValueError, match="returned 2 prompts for a batch of 3"):
        _enhance_shard(
            _ShortEnhancer(args, torch.device("cpu")),
            records,
            args,
            source_dataset,
            tmp_path / "parts",
            rank=0,
            world_size=1,
        )


def test_single_image_key_resolves_like_general_dataset(source_dataset: Path) -> None:
    images = runner.load_record_images(
        {"prompt": "x", "image": "ref.png"}, str(source_dataset / "images")
    )
    assert [image.size for image in images] == [(8, 6)]
    assert runner.load_record_images({"prompt": "x"}, str(source_dataset / "images")) == []


def test_registry_resolves_every_builtin_enhancer() -> None:
    registered = list_registered_prompt_enhancers()
    assert set(registered) == {
        "flux2",
        "ltx2",
        "flux3",
        "qwen-image-2.1-t2i",
        "qwen-image-2.1-i2i",
        "openai-compatible",
    }
    for name in registered:
        assert issubclass(get_prompt_enhancer_class(name), BasePromptEnhancer)
    with pytest.raises(ImportError, match="Could not load prompt enhancer"):
        get_prompt_enhancer_class("missing.module.Enhancer")
