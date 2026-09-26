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

from __future__ import annotations

import functools
import json
import multiprocessing
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import av
import numpy as np
import pytest
import torch
from PIL import Image

import flow_factory.data_utils.offline_dataset as offline_dataset_module
from flow_factory.data_utils.offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    DemonstrationOutput,
    DemonstrationOutputBatch,
    OfflineCollator,
    OfflineDataset,
    PreferenceOutput,
    PreferenceOutputBatch,
    compute_offline_condition_id,
    compute_offline_record_id,
    decode_audio,
    decode_video,
    load_offline_manifest,
)
from flow_factory.data_utils.schema import MediaAsset, NormalizedDatasetRecord, normalize_v2_record

SOURCE_NAME = "test-source"
SOURCE_ID = 7


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3, 2), color=color).save(path)


def _save_video(path: Path, values: Sequence[int], *, fps: int = 24) -> None:
    """Write a tiny deterministic RGB video for decoder and spawn-worker tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 6
        stream.height = 4
        stream.pix_fmt = "yuv420p"
        for value in values:
            pixels = np.full((stream.height, stream.width, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _demonstration_record(
    dataset_dir: Path,
    target_paths: Sequence[str],
    *,
    prompt: str = "Draw the target.",
) -> NormalizedDatasetRecord:
    return normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": prompt, "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {
                    "media": [
                        {"type": "image", "path": target_path} for target_path in target_paths
                    ]
                },
            },
            "metadata": {"prompt": prompt},
        },
        dataset_dir=dataset_dir,
    )


def _preference_record(
    dataset_dir: Path,
    chosen_paths: Sequence[str],
    rejected_paths: Sequence[str],
) -> NormalizedDatasetRecord:
    return normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "Choose the better output.", "media": []},
            "supervision": {
                "type": "preference",
                "chosen": {
                    "media": [
                        {"type": "image", "path": chosen_path} for chosen_path in chosen_paths
                    ]
                },
                "rejected": {
                    "media": [
                        {"type": "image", "path": rejected_path} for rejected_path in rejected_paths
                    ]
                },
            },
        },
        dataset_dir=dataset_dir,
    )


def _condition_cache(
    records: Sequence[NormalizedDatasetRecord],
    conditions: Sequence[Dict[str, Any]],
    *,
    source_name: str = SOURCE_NAME,
) -> List[Dict[str, Any]]:
    assert len(records) == len(conditions)
    return [
        {
            **condition,
            OFFLINE_CONDITION_ID_COLUMN: compute_offline_condition_id(
                record,
                index=index,
                source_name=source_name,
            ),
        }
        for index, (record, condition) in enumerate(zip(records, conditions))
    ]


def _record_ids(
    records: Sequence[NormalizedDatasetRecord],
    *,
    source_name: str = SOURCE_NAME,
) -> List[str]:
    return [
        compute_offline_record_id(record, index=index, source_name=source_name)
        for index, record in enumerate(records)
    ]


def _custom_video_decoder(asset: MediaAsset) -> Dict[str, str]:
    return {"decoded_path": asset.path}


class _BoundDecoder:
    def decode(self, asset: MediaAsset) -> str:
        return asset.path


class _CallableDecoder:
    def __call__(self, asset: MediaAsset) -> str:
        return asset.path


def _spawn_dataset_worker(
    dataset: OfflineDataset,
    collator: OfflineCollator,
    result_queue: Any,
) -> None:
    batch = collator([dataset[0]])
    result_queue.put(
        (
            batch.output.target_media[0][0].payload.getpixel((0, 0)),
            batch.condition["cached_text"],
            batch.sources,
            batch.source_ids.tolist(),
        )
    )


def _spawn_video_dataset_worker(dataset: OfflineDataset, result_queue: Any) -> None:
    payload = dataset[0].output.target_media[0].payload
    result_queue.put((payload.shape, payload.dtype.str, payload.flags.c_contiguous))


def test_manifest_reader_preserves_order_resolves_paths_and_requires_one_type(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "train.jsonl"
    rows = [
        {
            "schema_version": 2,
            "input": {"prompt": "first", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": "targets/first.png"}]},
            },
        },
        {
            "schema_version": 2,
            "input": {"prompt": "second", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": "targets/second.png"}]},
            },
        },
    ]
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    records = load_offline_manifest(
        manifest_path,
        supervision_type="demonstration",
    )

    assert [record.model_input.prompt for record in records] == ["first", "second"]
    assert records[0].supervision.target.media[0].path == str(tmp_path / "targets" / "first.png")


@pytest.mark.parametrize(
    "invalid_line,error_fragment",
    [
        ('{"schema_version": 2,', "invalid offline JSONL record"),
        (
            json.dumps(
                {
                    "schema_version": 2,
                    "input": {
                        "prompt": "unknown media key",
                        "media": [{"type": "image", "path": "input.png", "media_type": "image"}],
                    },
                    "supervision": {
                        "type": "demonstration",
                        "target": {"media": [{"type": "image", "path": "target.png"}]},
                    },
                }
            ),
            "invalid offline V2 record",
        ),
        ("   ", "blank line"),
    ],
)
def test_manifest_reader_reports_failing_line_context(
    tmp_path: Path,
    invalid_line: str,
    error_fragment: str,
) -> None:
    manifest_path = tmp_path / "train.jsonl"
    valid = {
        "schema_version": 2,
        "input": {"prompt": "valid", "media": []},
        "supervision": {
            "type": "demonstration",
            "target": {"media": [{"type": "image", "path": "target.png"}]},
        },
    }
    manifest_path.write_text(
        json.dumps(valid) + "\n" + invalid_line + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_offline_manifest(manifest_path, supervision_type="demonstration")
    message = str(exc_info.value)
    assert error_fragment in message
    assert "train.jsonl:2" in message


def test_manifest_reader_rejects_missing_and_mixed_supervision(tmp_path: Path) -> None:
    prompt_only = tmp_path / "prompt-only.jsonl"
    prompt_only.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "input": {"prompt": "no output", "media": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        load_offline_manifest(prompt_only, supervision_type="demonstration")
    message = str(exc_info.value)
    assert "invalid offline V2 record" in message
    assert "prompt-only.jsonl:1" in message
    assert "supervision" in message

    mixed = tmp_path / "mixed.jsonl"
    preference = {
        "schema_version": 2,
        "input": {"prompt": "pair", "media": []},
        "supervision": {
            "type": "preference",
            "chosen": {"media": [{"type": "image", "path": "chosen.png"}]},
            "rejected": {"media": [{"type": "image", "path": "rejected.png"}]},
        },
    }
    demonstration = {
        "schema_version": 2,
        "input": {"prompt": "target", "media": []},
        "supervision": {
            "type": "demonstration",
            "target": {"media": [{"type": "image", "path": "target.png"}]},
        },
    }
    mixed.write_text(
        json.dumps(demonstration) + "\n" + json.dumps(preference) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"mixed\.jsonl:2 has 'preference' supervision"):
        load_offline_manifest(mixed, supervision_type="demonstration")


def test_empty_manifest_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "empty.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="offline JSONL manifest is empty"):
        load_offline_manifest(manifest_path, supervision_type="demonstration")


def test_dataset_requires_normalized_homogeneous_records_and_matching_cache_ids(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target.png"
    _save_image(target_path, (10, 20, 30))
    demonstration = _demonstration_record(tmp_path, ["target.png"])
    preference = _preference_record(tmp_path, ["target.png"], ["target.png"])
    condition = _condition_cache(
        [demonstration],
        [{"prompt_embeds": torch.ones(2)}],
    )

    with pytest.raises(ValueError, match="equal length"):
        OfflineDataset(
            [demonstration],
            [],
            source_name=SOURCE_NAME,
            source_id=SOURCE_ID,
            supervision_type="demonstration",
        )
    with pytest.raises(ValueError, match="condition cache identity mismatch at index 0"):
        OfflineDataset(
            [demonstration],
            [
                {
                    "prompt_embeds": torch.ones(2),
                    OFFLINE_CONDITION_ID_COLUMN: "stale-cache-id",
                }
            ],
            source_name=SOURCE_NAME,
            source_id=SOURCE_ID,
            supervision_type="demonstration",
        )
    with pytest.raises(ValueError, match="missing reserved identity column"):
        OfflineDataset(
            [demonstration],
            [{"prompt_embeds": torch.ones(2)}],
            source_name=SOURCE_NAME,
            source_id=SOURCE_ID,
            supervision_type="demonstration",
        )
    with pytest.raises(ValueError, match="has 'preference' supervision"):
        OfflineDataset(
            [preference],
            _condition_cache([preference], [{"prompt_embeds": torch.ones(2)}]),
            source_name=SOURCE_NAME,
            source_id=SOURCE_ID,
            supervision_type="demonstration",
        )
    with pytest.raises(TypeError, match="normalized V2 records only"):
        OfflineDataset(
            [{"raw": True}],
            condition,
            source_name=SOURCE_NAME,
            source_id=SOURCE_ID,
            supervision_type="demonstration",
        )


def test_record_id_distinguishes_duplicate_rows_by_stable_index(tmp_path: Path) -> None:
    _save_image(tmp_path / "target.png", (1, 2, 3))
    record = _demonstration_record(tmp_path, ["target.png"])

    first = compute_offline_record_id(record, index=0, source_name=SOURCE_NAME)
    second = compute_offline_record_id(record, index=1, source_name=SOURCE_NAME)

    assert len(first) == 64
    assert first != second
    assert first == compute_offline_record_id(record, index=0, source_name=SOURCE_NAME)
    assert first != compute_offline_record_id(record, index=0, source_name="other-source")


def test_dataset_decodes_target_on_every_access_and_strips_reserved_condition_id(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target.png"
    _save_image(target_path, (255, 0, 0))
    record = _demonstration_record(tmp_path, ["target.png"])
    condition = {"prompt_embeds": torch.ones(2)}
    dataset = OfflineDataset(
        [record],
        _condition_cache([record], [condition]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )

    first = dataset[0]
    assert tuple(first.condition) == ("prompt_embeds",)
    assert first.condition["prompt_embeds"] is condition["prompt_embeds"]
    assert OFFLINE_CONDITION_ID_COLUMN not in first.condition
    assert isinstance(first.output, DemonstrationOutput)
    first_image = first.output.target_media[0].payload
    assert first_image.mode == "RGB"
    assert first_image.size == (3, 2)
    assert first_image.getpixel((0, 0)) == (255, 0, 0)

    _save_image(target_path, (0, 0, 255))
    second = dataset[0]

    assert second.output.target_media[0].payload.getpixel((0, 0)) == (0, 0, 255)
    assert first_image.getpixel((0, 0)) == (255, 0, 0)


def test_condition_id_is_input_only_while_record_id_tracks_full_provenance(
    tmp_path: Path,
) -> None:
    _save_image(tmp_path / "first.png", (1, 2, 3))
    _save_image(tmp_path / "second.png", (4, 5, 6))
    first = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "same input", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": "first.png"}]},
            },
            "metadata": {"revision": 1},
        },
        dataset_dir=tmp_path,
    )
    second = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "same input", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": "second.png"}]},
            },
            "metadata": {"revision": 2},
        },
        dataset_dir=tmp_path,
    )

    assert compute_offline_condition_id(
        first, index=0, source_name=SOURCE_NAME
    ) == compute_offline_condition_id(second, index=0, source_name=SOURCE_NAME)
    assert compute_offline_record_id(
        first, index=0, source_name=SOURCE_NAME
    ) != compute_offline_record_id(second, index=0, source_name=SOURCE_NAME)


def test_same_path_media_replacement_changes_the_owning_identity(tmp_path: Path) -> None:
    input_path = tmp_path / "condition.png"
    target_path = tmp_path / "target.png"
    _save_image(input_path, (1, 2, 3))
    _save_image(target_path, (4, 5, 6))
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {
                "prompt": "content-addressed",
                "media": [{"type": "image", "path": "condition.png"}],
            },
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": "target.png"}]},
            },
        },
        dataset_dir=tmp_path,
    )

    first_condition_id = compute_offline_condition_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )
    first_record_id = compute_offline_record_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )
    _save_image(input_path, (7, 8, 9))
    second_condition_id = compute_offline_condition_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )
    second_record_id = compute_offline_record_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )
    _save_image(target_path, (10, 11, 12))
    target_changed_condition_id = compute_offline_condition_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )
    target_changed_record_id = compute_offline_record_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )

    assert second_condition_id != first_condition_id
    assert second_record_id != first_record_id
    assert target_changed_condition_id == second_condition_id
    assert target_changed_record_id != second_record_id


@pytest.mark.parametrize("changed_arm", ("chosen", "rejected"))
def test_preference_record_id_tracks_each_output_arm_content(
    tmp_path: Path,
    changed_arm: str,
) -> None:
    chosen_path = tmp_path / "chosen.png"
    rejected_path = tmp_path / "rejected.png"
    _save_image(chosen_path, (1, 2, 3))
    _save_image(rejected_path, (4, 5, 6))
    record = _preference_record(tmp_path, ["chosen.png"], ["rejected.png"])

    first = compute_offline_record_id(record, index=0, source_name=SOURCE_NAME)
    changed_path = chosen_path if changed_arm == "chosen" else rejected_path
    _save_image(changed_path, (7, 8, 9))

    assert compute_offline_record_id(record, index=0, source_name=SOURCE_NAME) != first


def test_dataset_hashes_each_unique_media_path_once_per_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_path = tmp_path / "shared.png"
    _save_image(shared_path, (1, 2, 3))
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {
                "prompt": "shared bytes",
                "media": [
                    {"type": "image", "path": "shared.png"},
                    {"type": "image", "path": "shared.png"},
                ],
            },
            "supervision": {
                "type": "demonstration",
                "target": {
                    "media": [
                        {"type": "image", "path": "shared.png"},
                        {"type": "image", "path": "shared.png"},
                    ]
                },
            },
        },
        dataset_dir=tmp_path,
    )
    records = [record, record]
    condition_cache = _condition_cache(
        records,
        [{"prompt_embeds": torch.ones(2)}, {"prompt_embeds": torch.zeros(2)}],
    )
    original_hasher = offline_dataset_module._stream_media_content_sha256
    hashed_paths: List[str] = []

    def counted_hasher(path: str) -> str:
        hashed_paths.append(path)
        return original_hasher(path)

    monkeypatch.setattr(
        offline_dataset_module,
        "_stream_media_content_sha256",
        counted_hasher,
    )

    OfflineDataset(
        records,
        condition_cache,
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )

    assert hashed_paths == [str(shared_path)]


def test_identity_helpers_report_unreadable_media_path(tmp_path: Path) -> None:
    missing_input = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {
                "prompt": "missing input",
                "media": [{"type": "image", "path": "missing-input.png"}],
            },
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": "unused-target.png"}]},
            },
        },
        dataset_dir=tmp_path,
    )
    missing_target = _demonstration_record(tmp_path, ["missing-target.png"])

    with pytest.raises(OSError, match=r"hash offline media content.*missing-input\.png"):
        compute_offline_condition_id(missing_input, index=0, source_name=SOURCE_NAME)
    with pytest.raises(OSError, match=r"hash offline media content.*missing-target\.png"):
        compute_offline_record_id(missing_target, index=0, source_name=SOURCE_NAME)


def test_dataset_snapshots_plain_condition_cache_order(tmp_path: Path) -> None:
    for name in ("first.png", "second.png"):
        _save_image(tmp_path / name, (1, 2, 3))
    records = [
        _demonstration_record(tmp_path, ["first.png"], prompt="first"),
        _demonstration_record(tmp_path, ["second.png"], prompt="second"),
    ]
    cache = _condition_cache(
        records,
        [{"label": "first"}, {"label": "second"}],
    )
    dataset = OfflineDataset(
        records,
        cache,
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )

    cache.reverse()

    assert dataset[0].condition["label"] == "first"
    assert dataset[1].condition["label"] == "second"


def test_dataset_revalidates_embedded_condition_id_when_row_changes(tmp_path: Path) -> None:
    _save_image(tmp_path / "target.png", (1, 2, 3))
    record = _demonstration_record(tmp_path, ["target.png"])
    cache = _condition_cache([record], [{"label": "original"}])
    dataset = OfflineDataset(
        [record],
        cache,
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )

    cache[0][OFFLINE_CONDITION_ID_COLUMN] = "mutated"

    with pytest.raises(RuntimeError, match="condition cache identity mismatch at index 0"):
        dataset[0]


def test_dataset_precomputes_ids_and_normalizes_negative_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_image(tmp_path / "target.png", (1, 2, 3))
    record = _demonstration_record(tmp_path, ["target.png"])
    dataset = OfflineDataset(
        [record],
        _condition_cache([record], [{"label": "cached"}]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )
    expected_condition_id = compute_offline_condition_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )
    expected_record_id = compute_offline_record_id(
        record,
        index=0,
        source_name=SOURCE_NAME,
    )
    monkeypatch.setattr(
        offline_dataset_module,
        "compute_offline_condition_id",
        lambda *args, **kwargs: pytest.fail("condition id was recomputed"),
    )
    monkeypatch.setattr(
        offline_dataset_module,
        "compute_offline_record_id",
        lambda *args, **kwargs: pytest.fail("record id was recomputed"),
    )

    item = dataset[-1]

    assert item.condition_id == expected_condition_id
    assert item.record_id == expected_record_id
    with pytest.raises(IndexError, match="out of range"):
        dataset[1]
    with pytest.raises(IndexError, match="out of range"):
        dataset[-2]
    with pytest.raises(TypeError, match="must be an integer"):
        dataset[True]


def test_demonstration_collator_stacks_conditions_and_preserves_ragged_media(
    tmp_path: Path,
) -> None:
    for name, color in (
        ("first.png", (1, 0, 0)),
        ("second-a.png", (2, 0, 0)),
        ("second-b.png", (3, 0, 0)),
    ):
        _save_image(tmp_path / name, color)
    records = [
        _demonstration_record(tmp_path, ["first.png"], prompt="first"),
        _demonstration_record(
            tmp_path,
            ["second-a.png", "second-b.png"],
            prompt="second",
        ),
    ]
    conditions = [
        {
            "prompt_embeds": torch.tensor([1.0, 2.0]),
            "ragged": torch.ones(1),
            "optional_image_latents": None,
            "label": "first",
        },
        {
            "prompt_embeds": torch.tensor([3.0, 4.0]),
            "ragged": torch.ones(2),
            "optional_image_latents": torch.ones(2, 3),
            "label": "second",
        },
    ]
    cache = _condition_cache(records, conditions)
    dataset = OfflineDataset(
        records,
        cache,
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )

    batch = OfflineCollator("demonstration")([dataset[0], dataset[1]])

    assert torch.equal(
        batch.condition["prompt_embeds"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert [tensor.shape for tensor in batch.condition["ragged"]] == [
        torch.Size([1]),
        torch.Size([2]),
    ]
    assert batch.condition["label"] == ["first", "second"]
    assert batch.condition["optional_image_latents"][0] is None
    assert batch.condition["optional_image_latents"][1].shape == (2, 3)
    assert isinstance(batch.output, DemonstrationOutputBatch)
    assert [len(sample_media) for sample_media in batch.output.target_media] == [1, 2]
    assert [media.path for media in batch.output.target_media[1]] == [
        str(tmp_path / "second-a.png"),
        str(tmp_path / "second-b.png"),
    ]
    assert batch.condition_ids == tuple(row[OFFLINE_CONDITION_ID_COLUMN] for row in cache)
    assert batch.record_ids == tuple(_record_ids(records))
    assert batch.sources == (SOURCE_NAME, SOURCE_NAME)
    assert torch.equal(batch.source_ids, torch.tensor([SOURCE_ID, SOURCE_ID]))


def test_collator_transports_mixed_source_identity_from_concat_batches(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    _save_image(target_path, (1, 2, 3))
    record = _demonstration_record(tmp_path, ["target.png"])
    first_source = OfflineDataset(
        [record],
        _condition_cache(
            [record],
            [{"prompt_embeds": torch.ones(2)}],
            source_name="first-source",
        ),
        source_name="first-source",
        source_id=2,
        supervision_type="demonstration",
    )
    second_source = OfflineDataset(
        [record],
        _condition_cache(
            [record],
            [{"prompt_embeds": torch.zeros(2)}],
            source_name="second-source",
        ),
        source_name="second-source",
        source_id=5,
        supervision_type="demonstration",
    )

    batch = OfflineCollator("demonstration")([first_source[0], second_source[0]])

    assert batch.sources == ("first-source", "second-source")
    assert torch.equal(batch.source_ids, torch.tensor([2, 5], dtype=torch.long))
    assert batch.record_ids[0] != batch.record_ids[1]


def test_preference_dataset_and_collator_keep_chosen_and_rejected_ragged(
    tmp_path: Path,
) -> None:
    for name, color in (
        ("chosen-a.png", (1, 0, 0)),
        ("chosen-b.png", (2, 0, 0)),
        ("rejected.png", (3, 0, 0)),
    ):
        _save_image(tmp_path / name, color)
    record = _preference_record(
        tmp_path,
        ["chosen-a.png", "chosen-b.png"],
        ["rejected.png"],
    )
    dataset = OfflineDataset(
        [record],
        _condition_cache([record], [{"prompt_embeds": torch.ones(2)}]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="preference",
    )

    item = dataset[0]
    batch = OfflineCollator("preference")([item])

    assert isinstance(item.output, PreferenceOutput)
    assert [media.path for media in item.output.chosen_media] == [
        str(tmp_path / "chosen-a.png"),
        str(tmp_path / "chosen-b.png"),
    ]
    assert isinstance(batch.output, PreferenceOutputBatch)
    assert [len(media) for media in batch.output.chosen_media] == [2]
    assert [len(media) for media in batch.output.rejected_media] == [1]


def test_collator_uses_declared_supervision_instead_of_first_item_union(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    _save_image(target_path, (1, 2, 3))
    demonstration = _demonstration_record(tmp_path, ["target.png"])
    preference = _preference_record(tmp_path, ["target.png"], ["target.png"])
    demonstration_dataset = OfflineDataset(
        [demonstration],
        _condition_cache([demonstration], [{"prompt_embeds": torch.ones(2)}]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )
    preference_dataset = OfflineDataset(
        [preference],
        _condition_cache([preference], [{"prompt_embeds": torch.ones(2)}]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="preference",
    )

    with pytest.raises(ValueError, match="supervision mismatch at batch index 1"):
        OfflineCollator("demonstration")([demonstration_dataset[0], preference_dataset[0]])


def test_default_audio_decoder_preserves_samples_and_source_clock_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(b"identity-only audio payload")
    calls: List[tuple[str, int | None]] = []

    def fake_load_audio(path: str, sample_rate: int | None = None) -> torch.Tensor:
        calls.append((str(path), sample_rate))
        return torch.arange(12, dtype=torch.float64).reshape(2, 6).requires_grad_()

    monkeypatch.setattr(offline_dataset_module, "load_audio", fake_load_audio)
    media: Dict[str, Any] = {
        "type": "audio",
        "path": "target.wav",
        "sample_rate": 16000,
    }
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "audio target", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [media]},
            },
        },
        dataset_dir=tmp_path,
    )
    dataset = OfflineDataset(
        [record],
        _condition_cache([record], [{"prompt_embeds": torch.ones(2)}]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )

    item = dataset[0]

    assert isinstance(item.output, DemonstrationOutput)
    decoded = item.output.target_media[0]
    assert calls == [(str(target_path), None)]
    assert decoded.payload.shape == (2, 6)
    assert decoded.payload.dtype is torch.float32
    assert decoded.payload.device.type == "cpu"
    assert decoded.payload.is_contiguous()
    assert decoded.payload.requires_grad is False
    assert decoded.sample_rate == 16000
    assert pickle.loads(pickle.dumps(decode_audio)) is decode_audio
    pickle.dumps(dataset)


def test_default_video_decoder_returns_diffusers_compatible_cpu_frames(tmp_path: Path) -> None:
    target_path = tmp_path / "target.mp4"
    _save_video(target_path, [16, 64, 192])
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "video", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "video", "path": "target.mp4", "fps": 24.0}]},
            },
        },
        dataset_dir=tmp_path,
    )
    dataset = OfflineDataset(
        [record],
        _condition_cache([record], [{"prompt_embeds": torch.ones(2)}]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )

    item = dataset[0]
    assert isinstance(item.output, DemonstrationOutput)
    decoded = item.output.target_media[0]
    assert isinstance(decoded.payload, np.ndarray)
    assert decoded.payload.shape == (3, 4, 6, 3)
    assert decoded.payload.dtype == np.uint8
    assert decoded.payload.flags.c_contiguous
    assert decoded.fps == 24.0
    assert pickle.loads(pickle.dumps(decode_video)) is decode_video
    pickle.dumps(dataset)


def test_default_video_decoder_survives_spawn_worker_pickling(tmp_path: Path) -> None:
    target_path = tmp_path / "spawn.mp4"
    _save_video(target_path, [12, 24])
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "spawn video", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "video", "path": "spawn.mp4", "fps": 24.0}]},
            },
        },
        dataset_dir=tmp_path,
    )
    dataset = OfflineDataset(
        [record],
        tuple(_condition_cache([record], [{"cached_text": "encoded"}])),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_spawn_video_dataset_worker, args=(dataset, result_queue))

    process.start()
    process.join(timeout=20)

    assert process.exitcode == 0
    assert result_queue.get(timeout=5) == ((2, 4, 6, 3), "|u1", True)


def test_module_level_media_decoder_can_be_injected_explicitly(tmp_path: Path) -> None:
    (tmp_path / "target.mp4").write_bytes(b"identity-only test payload")
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "video", "media": []},
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "video", "path": "target.mp4", "fps": 24.0}]},
            },
        },
        dataset_dir=tmp_path,
    )
    dataset = OfflineDataset(
        [record],
        _condition_cache([record], [{"prompt_embeds": torch.ones(2)}]),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
        media_decoders={"video": _custom_video_decoder},
    )

    item = dataset[0]

    assert isinstance(item.output, DemonstrationOutput)
    assert item.output.target_media[0].payload == {"decoded_path": str(tmp_path / "target.mp4")}
    assert item.output.target_media[0].fps == 24.0


@pytest.mark.parametrize(
    "decoder,error_fragment",
    [
        (_BoundDecoder().decode, "must be a module-level function"),
        (_CallableDecoder(), "must be a module-level function"),
        (
            functools.partial(_custom_video_decoder),
            "must be a module-level function",
        ),
        (lambda asset: asset.path, "must be defined at module scope"),
    ],
)
def test_decoder_injection_rejects_non_function_or_local_callables(
    tmp_path: Path,
    decoder: Any,
    error_fragment: str,
) -> None:
    record = _demonstration_record(tmp_path, ["target.png"])

    with pytest.raises(TypeError, match=error_fragment):
        OfflineDataset(
            [record],
            _condition_cache([record], [{"prompt_embeds": torch.ones(2)}]),
            source_name=SOURCE_NAME,
            source_id=SOURCE_ID,
            supervision_type="demonstration",
            media_decoders={"image": decoder},
        )


def test_dataset_and_collator_are_pickleable_with_spawn_workers(tmp_path: Path) -> None:
    target_path = tmp_path / "target.png"
    _save_image(target_path, (7, 8, 9))
    record = _demonstration_record(tmp_path, ["target.png"])
    dataset = OfflineDataset(
        [record],
        tuple(_condition_cache([record], [{"cached_text": "encoded"}])),
        source_name=SOURCE_NAME,
        source_id=SOURCE_ID,
        supervision_type="demonstration",
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_spawn_dataset_worker,
        args=(dataset, OfflineCollator("demonstration"), result_queue),
    )

    process.start()
    process.join(timeout=20)

    assert process.exitcode == 0
    assert result_queue.get(timeout=5) == (
        (7, 8, 9),
        ["encoded"],
        (SOURCE_NAME,),
        [SOURCE_ID],
    )
