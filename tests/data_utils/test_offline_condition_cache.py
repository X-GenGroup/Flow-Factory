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

import gc
import json
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from datasets import Dataset as HFDataset
from datasets import Image as HFImage
from PIL import Image

from flow_factory.contracts import (
    DECODED_IMAGE_REPRESENTATION,
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaRule,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    MediaValueRange,
    NegativePromptPolicy,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
)
from flow_factory.data_utils.dataset import GeneralDataset, _cross_chunk_schema_probe_batches
from flow_factory.data_utils.offline_condition_cache import (
    build_offline_condition_cache,
    compute_offline_condition_source_hash,
    project_offline_condition_dataset,
)
from flow_factory.data_utils.offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    OfflineDataset,
    compute_offline_condition_id,
)
from flow_factory.data_utils.offline_loader import build_offline_dataloader
from flow_factory.data_utils.schema import NormalizedDatasetRecord, normalize_v2_record

_IMAGE_FORMAT = MediaFormat(
    type=MediaType.IMAGE,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.NOT_APPLICABLE,
    representation=DECODED_IMAGE_REPRESENTATION,
)
_SLOTTED_IMAGE_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(
            InputMediaRule(
                format=_IMAGE_FORMAT,
                min_count=1,
                max_count=2,
                slots=("first_frame", "last_frame"),
            ),
        ),
        binding=InputMediaBinding.GROUPED_BY_TYPE,
        order=InputMediaOrder.WITHIN_TYPE,
    ),
    negative_prompt=NegativePromptPolicy.UNSUPPORTED,
    output_media=OutputMediaSequence(items=(_IMAGE_FORMAT,)),
    geometry_source=GeometrySource.CONFIGURED,
    batch_capability=BatchCapability.SINGLE_SAMPLE,
)
_OPTIONAL_IMAGE_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(InputMediaRule(format=_IMAGE_FORMAT, min_count=0, max_count=1),),
        binding=InputMediaBinding.GROUPED_BY_TYPE,
        order=InputMediaOrder.INSENSITIVE,
    ),
    negative_prompt=NegativePromptPolicy.OPTIONAL,
    output_media=OutputMediaSequence(items=(_IMAGE_FORMAT,)),
    geometry_source=GeometrySource.CONFIGURED,
    batch_capability=BatchCapability.UNIFORM,
)


class CountingPreprocessor:
    def __init__(self) -> None:
        self.calls = 0
        self.received_kwargs: Dict[str, Any] = {}

    def preprocess(self, prompt: List[str], **kwargs: Any) -> Dict[str, torch.Tensor]:
        self.calls += 1
        self.received_kwargs = kwargs
        return {"prompt_embeds": torch.ones(len(prompt), 2)}


class GroupedPreprocessor:
    def __init__(self) -> None:
        self.images: Any = None
        self.videos: Any = None
        self.audios: Any = None

    def preprocess(
        self,
        prompt: List[str],
        images: Any = None,
        videos: Any = None,
        audios: Any = None,
    ) -> Dict[str, torch.Tensor]:
        self.images = images
        self.videos = videos
        self.audios = audios
        return {"encoded": torch.ones(len(prompt), 1)}


class OrderedPreprocessor:
    supports_ordered_references = True

    def __init__(self) -> None:
        self.references: Any = None

    def preprocess(
        self,
        prompt: List[str],
        references: List[List[Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        self.references = references
        return {"encoded": torch.ones(len(prompt), 1)}


class CollisionPreprocessor:
    def preprocess(self, prompt: List[str]) -> Dict[str, List[str]]:
        return {OFFLINE_CONDITION_ID_COLUMN: ["adapter-owned"] * len(prompt)}


class OptionalImagePreprocessor:
    python_format_columns = frozenset({"condition_images"})

    def preprocess(
        self,
        prompt: List[str],
        images: List[List[Image.Image]],
    ) -> Dict[str, Any]:
        condition_images = []
        image_latents = []
        image_latent_ids = []
        for sample_images in images:
            condition_images.append(sample_images)
            if sample_images:
                image_latents.append(torch.ones(2, 3))
                image_latent_ids.append(torch.zeros(2, 4))
            else:
                image_latents.append(None)
                image_latent_ids.append(None)
        return {
            "condition_images": condition_images,
            "image_latents": image_latents,
            "image_latent_ids": image_latent_ids,
        }


class OptionalSourcePreprocessor:
    """Expose stable output keys for optional condition columns across sources."""

    def preprocess(
        self,
        prompt: List[str],
        negative_prompt: List[str],
        images: List[List[Image.Image]],
    ) -> Dict[str, torch.Tensor]:
        return {
            "prompt_lengths": torch.tensor([len(value) for value in prompt]),
            "negative_prompt_lengths": torch.tensor([len(value) for value in negative_prompt]),
            "image_counts": torch.tensor([len(value) for value in images]),
        }


class SingleSamplePreprocessor:
    def __init__(self) -> None:
        self.batch_sizes: List[int] = []

    def preprocess(
        self,
        prompt: List[str],
        images: List[List[Image.Image]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        del images, kwargs
        self.batch_sizes.append(len(prompt))
        return {"encoded": torch.ones(len(prompt), 1)}


class ChunkBoundedDatasetSpy:
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        *,
        columns: List[str] | None = None,
        accesses: List[tuple[tuple[str, ...], int, int]] | None = None,
        max_chunk_size: int = 2,
    ) -> None:
        self.rows = rows
        self.column_names = list(rows[0]) if columns is None else columns
        self.accesses = [] if accesses is None else accesses
        self.max_chunk_size = max_chunk_size

    def __len__(self) -> int:
        return len(self.rows)

    def select_columns(self, column_names: List[str]) -> "ChunkBoundedDatasetSpy":
        return ChunkBoundedDatasetSpy(
            self.rows,
            columns=column_names,
            accesses=self.accesses,
            max_chunk_size=self.max_chunk_size,
        )

    def __getitem__(self, key: slice) -> Dict[str, List[Any]]:
        if not isinstance(key, slice):
            raise AssertionError(f"schema discovery requested an unbounded column: {key!r}")
        start = 0 if key.start is None else key.start
        stop = len(self.rows) if key.stop is None else key.stop
        if key.step not in (None, 1):
            raise AssertionError(f"schema discovery requested a strided slice: {key!r}")
        if stop - start > self.max_chunk_size:
            raise AssertionError(f"schema discovery requested an oversized slice: {key!r}")
        self.accesses.append((tuple(self.column_names), start, stop))
        return {
            column_name: [self.rows[index][column_name] for index in range(start, stop)]
            for column_name in self.column_names
        }


def _demonstration_record(
    dataset_dir: Path,
    *,
    input_media: List[Dict[str, Any]] | None = None,
    target_path: str = "target.png",
    metadata: Dict[str, Any] | None = None,
    negative_prompt: str | None = None,
) -> NormalizedDatasetRecord:
    return normalize_v2_record(
        {
            "schema_version": 2,
            "input": {
                "prompt": "condition prompt",
                "negative_prompt": negative_prompt,
                "media": input_media or [],
            },
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": target_path}]},
            },
            "metadata": metadata or {},
        },
        dataset_dir=dataset_dir,
    )


def _preference_record(dataset_dir: Path) -> NormalizedDatasetRecord:
    return normalize_v2_record(
        {
            "schema_version": 2,
            "input": {"prompt": "preference condition", "media": []},
            "supervision": {
                "type": "preference",
                "chosen": {"media": [{"type": "image", "path": "chosen-private.png"}]},
                "rejected": {"media": [{"type": "image", "path": "rejected-private.png"}]},
            },
            "metadata": {"annotator": "private"},
        },
        dataset_dir=dataset_dir,
    )


def test_target_and_metadata_changes_reuse_the_input_only_cache(tmp_path: Path) -> None:
    first = _demonstration_record(
        tmp_path,
        target_path="first-target-does-not-exist.png",
        metadata={"revision": 1},
    )
    second = _demonstration_record(
        tmp_path,
        target_path="second-target-does-not-exist.png",
        metadata={"revision": 2},
    )
    preprocessor = CountingPreprocessor()

    first_cache = build_offline_condition_cache(
        [first],
        source_name="demo-source",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )
    second_cache = build_offline_condition_cache(
        [second],
        source_name="demo-source",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )

    expected_condition_id = compute_offline_condition_id(
        first,
        index=0,
        source_name="demo-source",
    )
    assert preprocessor.calls == 1
    assert first_cache._fingerprint == second_cache._fingerprint
    assert first_cache.cache_files == second_cache.cache_files
    assert second_cache[0][OFFLINE_CONDITION_ID_COLUMN] == expected_condition_id
    assert "metadata" not in second_cache.column_names
    assert "metadata" not in second_cache[0]
    assert OFFLINE_CONDITION_ID_COLUMN not in preprocessor.received_kwargs
    assert "first-target-does-not-exist.png" not in repr(second_cache[0])
    assert "second-target-does-not-exist.png" not in repr(second_cache[0])


def test_input_media_replacement_invalidates_the_condition_cache(tmp_path: Path) -> None:
    input_path = tmp_path / "reference.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(input_path)
    record = _demonstration_record(
        tmp_path,
        input_media=[{"type": "image", "path": "reference.png"}],
        target_path="target-does-not-enter-condition-cache.png",
    )
    preprocessor = CountingPreprocessor()

    first_cache = build_offline_condition_cache(
        [record],
        source_name="content-addressed-input",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )
    first_condition_id = first_cache[0][OFFLINE_CONDITION_ID_COLUMN]
    Image.new("RGB", (2, 2), color=(4, 5, 6)).save(input_path)
    second_cache = build_offline_condition_cache(
        [record],
        source_name="content-addressed-input",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )

    assert preprocessor.calls == 2
    assert second_cache[0][OFFLINE_CONDITION_ID_COLUMN] != first_condition_id
    assert second_cache._fingerprint != first_cache._fingerprint
    assert second_cache.cache_files != first_cache.cache_files


def test_condition_cache_does_not_retain_the_bound_preprocessor(tmp_path: Path) -> None:
    preprocessor = CountingPreprocessor()
    preprocessor_ref = weakref.ref(preprocessor)

    condition_cache = build_offline_condition_cache(
        [_demonstration_record(tmp_path)],
        source_name="lifecycle",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
    )
    del preprocessor
    gc.collect()

    assert isinstance(condition_cache, HFDataset)
    assert not hasattr(condition_cache, "_preprocess_func")
    assert preprocessor_ref() is None


def test_projection_contains_only_input_and_identity_columns(tmp_path: Path) -> None:
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "reference.png")
    record = _demonstration_record(
        tmp_path,
        input_media=[{"type": "image", "path": "reference.png"}],
        target_path="private-target.png",
        metadata={"private": "metadata"},
        negative_prompt="bad quality",
    )

    projected = project_offline_condition_dataset(
        [record],
        source_name="source",
        ordered_references=False,
    )

    assert set(projected.column_names) == {
        "prompt",
        "negative_prompt",
        "images",
        OFFLINE_CONDITION_ID_COLUMN,
    }
    assert projected[0]["images"] == [str(tmp_path / "reference.png")]
    assert "private-target.png" not in repr(projected[0])


def test_projection_canonicalizes_explicit_and_positional_semantic_slots(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "first.png")
    Image.new("RGB", (2, 2), color=(4, 5, 6)).save(tmp_path / "last.png")
    record = _demonstration_record(
        tmp_path,
        input_media=[
            {"type": "image", "path": "last.png", "slot": "last_frame"},
            {"type": "image", "path": "first.png"},
        ],
    )

    projected = project_offline_condition_dataset(
        [record],
        source_name="slotted",
        ordered_references=False,
        pipeline_io_contract=_SLOTTED_IMAGE_CONTRACT,
    )

    assert projected[0]["images"] == [
        str(tmp_path / "first.png"),
        str(tmp_path / "last.png"),
    ]
    assert projected[0]["image_slots"] == ["first_frame", "last_frame"]


def test_projection_normalizes_missing_optional_negative_prompts_in_mixed_batch(
    tmp_path: Path,
) -> None:
    """Tokenizers receive strings when only some V2 rows specify a negative prompt."""
    projected = project_offline_condition_dataset(
        [
            _demonstration_record(tmp_path, negative_prompt=None),
            _demonstration_record(tmp_path, negative_prompt="bad quality"),
        ],
        source_name="mixed-negative-prompts",
        ordered_references=False,
    )

    assert projected["negative_prompt"] == ["", "bad quality"]


def test_optional_condition_columns_are_stable_across_offline_sources(
    tmp_path: Path,
) -> None:
    """ConcatDataset batches can mix an all-empty source with a populated source."""
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "reference.png")
    Image.new("RGB", (2, 2), color=(4, 5, 6)).save(tmp_path / "target.png")
    empty_record = _demonstration_record(tmp_path)
    populated_record = _demonstration_record(
        tmp_path,
        input_media=[{"type": "image", "path": "reference.png"}],
        negative_prompt="bad quality",
    )
    preprocessor = OptionalSourcePreprocessor()

    empty_cache = build_offline_condition_cache(
        [empty_record],
        source_name="empty-source",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        pipeline_io_contract=_OPTIONAL_IMAGE_CONTRACT,
        preprocessing_batch_size=1,
    )
    populated_cache = build_offline_condition_cache(
        [populated_record],
        source_name="populated-source",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        pipeline_io_contract=_OPTIONAL_IMAGE_CONTRACT,
        preprocessing_batch_size=1,
    )
    assert set(empty_cache.column_names) == set(populated_cache.column_names)

    empty_dataset = OfflineDataset(
        [empty_record],
        empty_cache,
        source_name="empty-source",
        source_id=0,
        supervision_type="demonstration",
    )
    populated_dataset = OfflineDataset(
        [populated_record],
        populated_cache,
        source_name="populated-source",
        source_id=1,
        supervision_type="demonstration",
    )
    loader = build_offline_dataloader(
        [empty_dataset, populated_dataset],
        source_weights=[1, 1],
        batch_size=2,
        num_replicas=1,
        rank=0,
        gradient_accumulation_steps=1,
        shuffle=False,
    )

    batch = next(iter(loader))
    assert batch.condition["image_counts"].tolist() == [0, 1]
    assert batch.condition["negative_prompt_lengths"].tolist() == [0, 11]


def test_preference_arms_never_enter_the_condition_projection(tmp_path: Path) -> None:
    projected = project_offline_condition_dataset(
        [_preference_record(tmp_path)],
        source_name="preference",
        ordered_references=False,
    )

    assert set(projected.column_names) == {"prompt", OFFLINE_CONDITION_ID_COLUMN}
    assert "chosen-private.png" not in repr(projected[0])
    assert "rejected-private.png" not in repr(projected[0])
    assert "annotator" not in repr(projected[0])


def test_condition_source_hash_is_order_sensitive() -> None:
    assert compute_offline_condition_source_hash(["first", "second"]) != (
        compute_offline_condition_source_hash(["second", "first"])
    )


def test_condition_source_hash_includes_effective_slot_projection() -> None:
    reversed_rule = replace(
        _SLOTTED_IMAGE_CONTRACT.input_media.rules[0],
        slots=("last_frame", "first_frame"),
    )
    reversed_contract = replace(
        _SLOTTED_IMAGE_CONTRACT,
        input_media=replace(
            _SLOTTED_IMAGE_CONTRACT.input_media,
            rules=(reversed_rule,),
        ),
    )

    baseline = compute_offline_condition_source_hash(
        ["same-input"],
        pipeline_io_contract=_SLOTTED_IMAGE_CONTRACT,
    )
    changed = compute_offline_condition_source_hash(
        ["same-input"],
        pipeline_io_contract=reversed_contract,
    )

    assert changed != baseline


def test_condition_source_hash_includes_decoded_media_representation() -> None:
    """A physical input-boundary change must invalidate cached preprocessing."""
    changed_representation = replace(
        DECODED_IMAGE_REPRESENTATION,
        value_range=MediaValueRange(minimum=0.0, maximum=254.0, finite=True),
    )
    changed_rule = replace(
        _OPTIONAL_IMAGE_CONTRACT.input_media.rules[0],
        format=replace(_IMAGE_FORMAT, representation=changed_representation),
    )
    changed_contract = replace(
        _OPTIONAL_IMAGE_CONTRACT,
        input_media=replace(
            _OPTIONAL_IMAGE_CONTRACT.input_media,
            rules=(changed_rule,),
        ),
    )

    baseline = compute_offline_condition_source_hash(
        ["same-input"],
        pipeline_io_contract=_OPTIONAL_IMAGE_CONTRACT,
    )
    changed = compute_offline_condition_source_hash(
        ["same-input"],
        pipeline_io_contract=changed_contract,
    )

    assert changed != baseline


def test_schema_probe_scans_only_bounded_pending_column_chunks() -> None:
    dataset = ChunkBoundedDatasetSpy(
        [
            {"a": None, "b": [], "always": "set"},
            {"a": None, "b": [], "always": "set"},
            {"a": None, "b": [], "always": "set"},
            {"a": "typed", "b": [], "always": "set"},
            {"a": None, "b": ["typed"], "always": "set"},
            {"a": None, "b": [], "always": "set"},
        ],
        max_chunk_size=2,
    )

    probe_batches = _cross_chunk_schema_probe_batches(
        dataset,
        preprocessing_batch_size=2,
    )

    assert probe_batches == [[0, 1], [2, 3], [4, 5]]
    assert dataset.accesses == [
        (("a", "b", "always"), 0, 2),
        (("a", "b"), 2, 4),
        (("b",), 4, 6),
    ]


def test_grouped_projection_preserves_per_modality_order_and_rate_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, color in (("first.png", (255, 0, 0)), ("second.png", (0, 255, 0))):
        Image.new("RGB", (2, 2), color=color).save(tmp_path / name)
    for name in ("first.mp4", "second.mp4", "voice.wav"):
        (tmp_path / name).write_bytes(f"identity bytes for {name}".encode("utf-8"))
    video_calls: List[tuple[str, float | None]] = []
    audio_calls: List[tuple[str, int | None]] = []

    def fake_load_video(path: str, fps: float | None = None) -> List[Image.Image]:
        video_calls.append((path, fps))
        value = 10 if path.endswith("first.mp4") else 20
        return [Image.new("RGB", (2, 2), color=(value, value, value))]

    def fake_load_audio(path: str, sample_rate: int | None = None) -> torch.Tensor:
        audio_calls.append((path, sample_rate))
        return torch.ones(1, 4)

    monkeypatch.setattr("flow_factory.data_utils.dataset.load_video_frames", fake_load_video)
    monkeypatch.setattr("flow_factory.data_utils.dataset.load_audio", fake_load_audio)
    record = _demonstration_record(
        tmp_path,
        input_media=[
            {"type": "image", "path": "first.png"},
            {"type": "video", "path": "first.mp4", "fps": 12.5},
            {"type": "audio", "path": "voice.wav", "sample_rate": 16000},
            {"type": "image", "path": "second.png"},
            {"type": "video", "path": "second.mp4"},
        ],
    )
    preprocessor = GroupedPreprocessor()

    build_offline_condition_cache(
        [record],
        source_name="grouped",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        force_reprocess=True,
        preprocessing_batch_size=1,
    )

    assert [image.getpixel((0, 0)) for image in preprocessor.images[0]] == [
        (255, 0, 0),
        (0, 255, 0),
    ]
    assert video_calls == [
        (str(tmp_path / "first.mp4"), 12.5),
        (str(tmp_path / "second.mp4"), None),
    ]
    assert audio_calls == [(str(tmp_path / "voice.wav"), 16000)]
    assert len(preprocessor.videos[0]) == 2
    assert len(preprocessor.audios[0]) == 1


def test_condition_cache_preserves_optional_image_schema_across_map_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrow keeps one optional-image schema using datasets 3.3 map keywords."""
    original_map = HFDataset.map
    captured_features = []

    def datasets_3_3_compatible_map(
        self: HFDataset,
        function: Any,
        *,
        batched: bool,
        with_indices: bool,
        batch_size: int,
        fn_kwargs: Dict[str, Any],
        remove_columns: List[str],
        new_fingerprint: str,
        cache_file_name: str,
        features: Any,
        desc: str,
        load_from_cache_file: bool,
    ) -> HFDataset:
        captured_features.append(features)
        return original_map(
            self,
            function,
            batched=batched,
            with_indices=with_indices,
            batch_size=batch_size,
            fn_kwargs=fn_kwargs,
            remove_columns=remove_columns,
            new_fingerprint=new_fingerprint,
            cache_file_name=cache_file_name,
            features=features,
            desc=desc,
            load_from_cache_file=load_from_cache_file,
        )

    monkeypatch.setattr(HFDataset, "map", datasets_3_3_compatible_map)
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "reference.png")
    records = [
        _demonstration_record(tmp_path),
        _demonstration_record(tmp_path),
        _demonstration_record(
            tmp_path,
            input_media=[{"type": "image", "path": "reference.png"}],
        ),
        _demonstration_record(
            tmp_path,
            input_media=[{"type": "image", "path": "reference.png"}],
        ),
    ]

    cache = build_offline_condition_cache(
        records,
        source_name="optional-images",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=OptionalImagePreprocessor().preprocess,
        force_reprocess=True,
        preprocessing_batch_size=2,
    )

    assert len(captured_features) == 1
    assert captured_features[0] is not None
    assert isinstance(cache.features["condition_images"].feature, HFImage)
    assert cache.features["image_latents"].feature.feature.dtype == "float32"
    assert cache.features["image_latent_ids"].feature.feature.dtype == "float32"
    assert cache[0]["condition_images"] == []
    assert cache[0]["image_latents"] is None
    assert cache[0]["image_latent_ids"] is None
    assert cache[1]["condition_images"] == []
    assert cache[1]["image_latents"] is None
    assert cache[1]["image_latent_ids"] is None
    for row in (2, 3):
        assert len(cache[row]["condition_images"]) == 1
        assert cache[row]["image_latents"].shape == (2, 3)
        assert cache[row]["image_latent_ids"].shape == (2, 4)


def test_distributed_condition_cache_uses_one_global_optional_media_schema(
    tmp_path: Path,
) -> None:
    """Disjoint empty/populated ranks consolidate with identical Arrow features."""
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "reference.png")
    records = [
        _demonstration_record(tmp_path),
        _demonstration_record(tmp_path),
        _demonstration_record(
            tmp_path,
            input_media=[{"type": "image", "path": "reference.png"}],
        ),
        _demonstration_record(
            tmp_path,
            input_media=[{"type": "image", "path": "reference.png"}],
        ),
    ]
    raw_dataset = project_offline_condition_dataset(
        records,
        source_name="distributed-optional-images",
        ordered_references=False,
        pipeline_io_contract=_OPTIONAL_IMAGE_CONTRACT,
    )
    condition_ids = tuple(raw_dataset[OFFLINE_CONDITION_ID_COLUMN])
    source_hash = compute_offline_condition_source_hash(
        condition_ids,
        pipeline_io_contract=_OPTIONAL_IMAGE_CONTRACT,
    )
    preprocessor = OptionalImagePreprocessor()
    merged_cache_path = GeneralDataset.compute_cache_path(
        dataset_dir=str(tmp_path),
        split="train",
        cache_dir=str(tmp_path / "cache"),
        max_dataset_size=None,
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={},
        source_hash_override=source_hash,
    )

    for rank in range(2):
        GeneralDataset(
            dataset_dir=str(tmp_path),
            split="train",
            cache_dir=str(tmp_path / "cache"),
            force_reprocess=True,
            preprocessing_batch_size=2,
            preprocess_func=preprocessor.preprocess,
            num_shards=2,
            shard_index=rank,
            image_dir=str(tmp_path),
            video_dir=str(tmp_path),
            audio_dir=str(tmp_path),
            target_arrow_path=GeneralDataset.build_part_arrow_path(
                merged_cache_path,
                rank,
                2,
            ),
            raw_dataset=raw_dataset,
            source_hash_override=source_hash,
            passthrough_columns=(OFFLINE_CONDITION_ID_COLUMN,),
        )

    GeneralDataset.consolidate_parts(merged_cache_path, 2, split="train")
    merged = GeneralDataset.load_merged(merged_cache_path).processed_dataset

    assert isinstance(merged.features["condition_images"].feature, HFImage)
    assert merged[0]["condition_images"] == []
    assert len(merged[2]["condition_images"]) == 1


def test_standalone_condition_cache_honors_single_sample_contract(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "first.png")
    records = [
        _demonstration_record(
            tmp_path,
            input_media=[{"type": "image", "path": "first.png", "slot": "first_frame"}],
        )
        for _ in range(2)
    ]
    preprocessor = SingleSamplePreprocessor()

    build_offline_condition_cache(
        records,
        source_name="single-sample",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        pipeline_io_contract=_SLOTTED_IMAGE_CONTRACT,
        force_reprocess=True,
        preprocessing_batch_size=8,
    )

    assert preprocessor.batch_sizes == [1, 1]


def test_ordered_heterogeneous_references_cross_arrow_as_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(tmp_path / "image.png")
    (tmp_path / "video.mp4").write_bytes(b"identity-only video payload")
    (tmp_path / "audio.wav").write_bytes(b"identity-only audio payload")
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._decode_ordered_video",
        lambda path: (np.zeros((1, 2, 2, 3), dtype=np.uint8), 30.0, None, None),
    )
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._decode_ordered_audio",
        lambda path: (torch.zeros(1, 8), 22050),
    )
    record = _demonstration_record(
        tmp_path,
        input_media=[
            {"type": "image", "path": "image.png"},
            {"type": "video", "path": "video.mp4", "fps": 24.0},
            {"type": "audio", "path": "audio.wav", "sample_rate": 44100},
        ],
    )
    projected = project_offline_condition_dataset(
        [record],
        source_name="ordered",
        ordered_references=True,
    )
    raw_manifest = projected[0]["references"]
    raw_references = json.loads(raw_manifest)

    assert isinstance(raw_manifest, str)
    assert [set(reference) for reference in raw_references] == [
        {"type", "path"},
        {"type", "path", "fps"},
        {"type", "path", "sample_rate"},
    ]
    assert [reference["type"] for reference in raw_references] == [
        "image",
        "video",
        "audio",
    ]

    preprocessor = OrderedPreprocessor()
    cache = build_offline_condition_cache(
        [record],
        source_name="ordered",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=preprocessor.preprocess,
        force_reprocess=True,
    )
    loaded = preprocessor.references[0]

    assert set(loaded[0]) == {"type", "path", "media"}
    assert set(loaded[1]) == {"type", "path", "fps", "frames"}
    assert set(loaded[2]) == {"type", "path", "sample_rate", "media"}
    assert cache[0][OFFLINE_CONDITION_ID_COLUMN] == compute_offline_condition_id(
        record,
        index=0,
        source_name="ordered",
    )


def test_preprocess_cannot_overwrite_reserved_condition_identity(tmp_path: Path) -> None:
    record = _demonstration_record(tmp_path)

    with pytest.raises(ValueError, match="collides with passthrough columns"):
        build_offline_condition_cache(
            [record],
            source_name="collision",
            dataset_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            preprocess_func=CollisionPreprocessor().preprocess,
            force_reprocess=True,
            preprocessing_batch_size=1,
        )


def test_in_memory_general_dataset_requires_explicit_source_identity(tmp_path: Path) -> None:
    raw_dataset = HFDataset.from_dict({"prompt": ["hello"]})

    with pytest.raises(ValueError, match="source_hash_override is required"):
        GeneralDataset(
            dataset_dir=str(tmp_path),
            raw_dataset=raw_dataset,
            enable_preprocess=False,
        )


def test_file_backed_online_dataset_keeps_legacy_loading_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "datasets.config.HF_DATASETS_CACHE",
        str(tmp_path / "huggingface-cache"),
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"prompt": "online prompt", "label": 7}) + "\n",
        encoding="utf-8",
    )
    preprocessor = CountingPreprocessor()

    dataset = GeneralDataset(
        dataset_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        force_reprocess=True,
        preprocessing_batch_size=1,
    )

    assert dataset[0]["prompt"] == "online prompt"
    assert dataset[0]["metadata"] == {"label": 7}
    assert preprocessor.calls == 1
