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
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import ConcatDataset, DistributedSampler

import flow_factory.data_utils.offline_dataset as offline_dataset_module
import flow_factory.data_utils.offline_train_data as offline_train_data
from flow_factory.contracts import (
    DECODED_AUDIO_REPRESENTATION,
    DECODED_IMAGE_REPRESENTATION,
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaRule,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
)
from flow_factory.data_utils.offline_dataset import (
    OFFLINE_CONDITION_ID_COLUMN,
    DemonstrationOutput,
    OfflineDataset,
    PreferenceOutput,
)
from flow_factory.data_utils.offline_train_data import build_offline_train_dataloader
from flow_factory.hparams.data_args import DataArguments
from flow_factory.hparams.dataset_args import DatasetArguments, DatasetTrainSpec

_IMAGE_FORMAT = MediaFormat(
    type=MediaType.IMAGE,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.NOT_APPLICABLE,
    representation=DECODED_IMAGE_REPRESENTATION,
)
_TEXT_TO_IMAGE_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(),
        binding=InputMediaBinding.GROUPED_BY_TYPE,
        order=InputMediaOrder.INSENSITIVE,
    ),
    negative_prompt=NegativePromptPolicy.OPTIONAL,
    output_media=OutputMediaSequence(items=(_IMAGE_FORMAT,)),
    geometry_source=GeometrySource.CONFIGURED,
    batch_capability=BatchCapability.UNIFORM,
)
_ORDERED_IMAGE_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(InputMediaRule(format=_IMAGE_FORMAT, min_count=1, max_count=1),),
        binding=InputMediaBinding.ORDERED_REFERENCES,
        order=InputMediaOrder.GLOBAL,
    ),
    negative_prompt=NegativePromptPolicy.OPTIONAL,
    output_media=OutputMediaSequence(items=(_IMAGE_FORMAT,)),
    geometry_source=GeometrySource.CONFIGURED,
    batch_capability=BatchCapability.UNIFORM,
)
_TEXT_TO_AUDIO_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(),
        binding=InputMediaBinding.GROUPED_BY_TYPE,
        order=InputMediaOrder.INSENSITIVE,
    ),
    negative_prompt=NegativePromptPolicy.OPTIONAL,
    output_media=OutputMediaSequence(
        items=(
            MediaFormat(
                type=MediaType.AUDIO,
                fps=RateRequirement.NOT_APPLICABLE,
                sample_rate=RateRequirement.REQUIRED,
                representation=DECODED_AUDIO_REPRESENTATION,
            ),
        )
    ),
    geometry_source=GeometrySource.OUTPUT_MEDIA,
    batch_capability=BatchCapability.UNIFORM,
)


class _TrainingArguments(dict):
    def __init__(
        self,
        *,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        seed: int = 17,
        guidance_scale: float = 3.0,
    ) -> None:
        super().__init__(
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seed=seed,
            guidance_scale=guidance_scale,
        )
        self.__dict__.update(self)

    def get_preprocess_guidance_scale(self) -> float:
        return self.guidance_scale


class _Accelerator:
    def __init__(self, *, num_processes: int = 1, process_index: int = 0) -> None:
        self.num_processes = num_processes
        self.process_index = process_index
        self.local_process_index = process_index
        self.is_main_process = process_index == 0
        self.is_local_main_process = process_index == 0
        self.wait_calls = 0
        self.prepare_calls = 0

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def prepare(self, *args: Any) -> None:
        self.prepare_calls += 1
        raise AssertionError("offline train loader must not call Accelerator.prepare")


class _CountingPreprocessor:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: List[int] = []
        self.is_train: bool | None = None
        self.guidance_scale: float | None = None

    def preprocess(
        self,
        prompt: List[str],
        *,
        is_train: bool,
        guidance_scale: float,
    ) -> Dict[str, torch.Tensor]:
        self.calls += 1
        self.batch_sizes.append(len(prompt))
        self.is_train = is_train
        self.guidance_scale = guidance_scale
        return {"prompt_embeds": torch.ones(len(prompt), 2)}


class _OrderedPreprocessor:
    supports_ordered_references = True

    def __init__(self) -> None:
        self.references: Any = None

    def preprocess(
        self,
        prompt: List[str],
        references: List[List[Dict[str, Any]]],
        *,
        is_train: bool,
        guidance_scale: float,
    ) -> Dict[str, torch.Tensor]:
        del is_train, guidance_scale
        self.references = references
        return {"prompt_embeds": torch.ones(len(prompt), 2)}


def _source(
    name: str,
    dataset_dir: Path,
    source_id: int | None,
    *,
    weight: int = 1,
    max_dataset_size: int | None = None,
) -> DatasetArguments:
    return DatasetArguments(
        name=name,
        dataset_dir=str(dataset_dir),
        train=DatasetTrainSpec(
            split="train",
            weight=weight,
            max_dataset_size=max_dataset_size,
        ),
        source_id=source_id,
    )


def _config(
    tmp_path: Path,
    sources: List[DatasetArguments],
    *,
    max_dataset_size: int | None = None,
    force_reprocess: bool = False,
    preprocess_parallelism: str = "local",
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_args=DataArguments(
            datasets=sources,
            cache_dir=str(tmp_path / "cache"),
            preprocessing_batch_size=2,
            dataloader_num_workers=0,
            enable_preprocess=True,
            force_reprocess=force_reprocess,
            max_dataset_size=max_dataset_size,
            preprocess_parallelism=preprocess_parallelism,
        ),
        training_args=_TrainingArguments(
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        model_args=SimpleNamespace(
            model_type="test-model",
            model_name_or_path="test/model",
            component_load_dtypes=None,
            frozen_parameters_dtype=None,
        ),
    )


def _write_image(path: Path, value: int) -> None:
    Image.new("RGB", (2, 2), color=(value, value, value)).save(path)


def test_offline_cache_fingerprint_tracks_precision_aware_model_policies(
    tmp_path: Path,
) -> None:
    """Condition embeddings cannot cross component load or frozen dtype policies."""
    baseline = _config(tmp_path, [])
    load_override = _config(tmp_path, [])
    load_override.model_args.component_load_dtypes = {
        "text_encoder": torch.float16,
        "default": None,
    }
    reordered_override = _config(tmp_path, [])
    reordered_override.model_args.component_load_dtypes = {
        "default": None,
        "text_encoder": torch.float16,
    }
    frozen_override = _config(tmp_path, [])
    frozen_override.model_args.frozen_parameters_dtype = {
        "text_encoders": torch.bfloat16,
        "default": None,
    }

    baseline_hash = offline_train_data._build_extra_hash_strs(baseline, None)
    load_hash = offline_train_data._build_extra_hash_strs(load_override, None)

    assert load_hash != baseline_hash
    assert load_hash == offline_train_data._build_extra_hash_strs(reordered_override, None)
    assert offline_train_data._build_extra_hash_strs(frozen_override, None) != baseline_hash


def _demonstration_row(prompt: str, target_path: str, *, metadata: int = 0) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "input": {"prompt": prompt, "media": []},
        "supervision": {
            "type": "demonstration",
            "target": {"media": [{"type": "image", "path": target_path}]},
        },
        "metadata": {"revision": metadata},
    }


def _preference_row(prompt: str, chosen_path: str, rejected_path: str) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "input": {"prompt": prompt, "media": []},
        "supervision": {
            "type": "preference",
            "chosen": {"media": [{"type": "image", "path": chosen_path}]},
            "rejected": {"media": [{"type": "image", "path": rejected_path}]},
        },
        "metadata": {"annotator": "offline"},
    }


def _write_manifest(dataset_dir: Path, rows: List[Dict[str, Any]]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_datasets(loader: Any) -> tuple[OfflineDataset, ...]:
    assert isinstance(loader.dataset, ConcatDataset)
    return tuple(loader.dataset.datasets)


def test_builder_returns_detached_input_cache_and_decodes_target_on_demand(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "demo"
    dataset_dir.mkdir()
    _write_image(dataset_dir / "target.png", 10)
    _write_manifest(dataset_dir, [_demonstration_row("a prompt", "target.png")])
    preprocessor = _CountingPreprocessor()
    preprocessor_ref = weakref.ref(preprocessor)
    accelerator = _Accelerator()

    loader = build_offline_train_dataloader(
        _config(tmp_path, [_source("demo", dataset_dir, 0)]),
        accelerator,
        preprocessor.preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        shuffle=False,
    )

    (dataset,) = _source_datasets(loader)
    assert isinstance(loader.sampler, DistributedSampler)
    assert preprocessor.calls == 1
    assert preprocessor.is_train is True
    assert preprocessor.guidance_scale == 3.0
    assert accelerator.prepare_calls == 0
    assert isinstance(dataset._condition_cache, HFDataset)
    assert not hasattr(dataset._condition_cache, "_preprocess_func")
    assert "metadata" not in dataset._condition_cache.column_names
    assert "target.png" not in repr(dataset._condition_cache[0])

    first = dataset[0]
    assert isinstance(first.output, DemonstrationOutput)
    assert first.output.target_media[0].payload.getpixel((0, 0)) == (10, 10, 10)
    _write_image(dataset_dir / "target.png", 200)
    assert dataset[0].output.target_media[0].payload.getpixel((0, 0)) == (200, 200, 200)

    del preprocessor
    gc.collect()
    assert preprocessor_ref() is None


def test_builder_hashes_a_shared_input_and_target_path_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Condition projection and full record identity share one build-local memo."""
    dataset_dir = tmp_path / "shared-identity-media"
    dataset_dir.mkdir()
    shared_path = dataset_dir / "shared.png"
    _write_image(shared_path, 10)
    row = _demonstration_row("shared media", "shared.png")
    row["input"]["media"] = [{"type": "image", "path": "shared.png"}]
    _write_manifest(dataset_dir, [row])
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

    build_offline_train_dataloader(
        _config(tmp_path, [_source("shared-media", dataset_dir, 0)]),
        _Accelerator(),
        _OrderedPreprocessor().preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=_ORDERED_IMAGE_CONTRACT,
        shuffle=False,
    )

    assert hashed_paths == [str(shared_path)]


def test_builder_slices_each_source_and_preserves_resolved_source_identity(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    for dataset_dir, offset in ((first_dir, 0), (second_dir, 10)):
        dataset_dir.mkdir()
        rows = []
        for index in range(3):
            target = f"target-{index}.png"
            _write_image(dataset_dir / target, offset + index)
            rows.append(_demonstration_row(f"prompt-{offset + index}", target))
        _write_manifest(dataset_dir, rows)

    loader = build_offline_train_dataloader(
        _config(
            tmp_path,
            [
                _source("first", first_dir, 3, max_dataset_size=1),
                _source("second", second_dir, 7),
            ],
            max_dataset_size=2,
        ),
        _Accelerator(),
        _CountingPreprocessor().preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        shuffle=False,
    )

    first, second = _source_datasets(loader)
    assert (len(first), len(second)) == (1, 2)
    assert (first.source_name, first.source_id) == ("first", 3)
    assert (second.source_name, second.source_id) == ("second", 7)
    assert len(loader.dataset) == 3


def test_target_and_metadata_only_manifest_change_reuses_condition_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "reuse"
    dataset_dir.mkdir()
    _write_image(dataset_dir / "first.png", 1)
    _write_image(dataset_dir / "second.png", 2)
    source = _source("reuse", dataset_dir, 0)
    config = _config(tmp_path, [source])
    preprocessor = _CountingPreprocessor()

    _write_manifest(
        dataset_dir,
        [_demonstration_row("stable prompt", "first.png", metadata=1)],
    )
    first_loader = build_offline_train_dataloader(
        config,
        _Accelerator(),
        preprocessor.preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        shuffle=False,
    )
    _write_manifest(
        dataset_dir,
        [_demonstration_row("stable prompt", "second.png", metadata=2)],
    )
    second_loader = build_offline_train_dataloader(
        config,
        _Accelerator(),
        preprocessor.preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        shuffle=False,
    )

    assert preprocessor.calls == 1
    first_cache = _source_datasets(first_loader)[0]._condition_cache
    second_dataset = _source_datasets(second_loader)[0]
    assert first_cache.cache_files == second_dataset._condition_cache.cache_files
    assert second_dataset[0].output.target_media[0].path.endswith("second.png")
    assert "second.png" not in repr(second_dataset._condition_cache[0])


def test_builder_supports_homogeneous_offline_preference_sources(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "preference"
    dataset_dir.mkdir()
    _write_image(dataset_dir / "chosen.png", 220)
    _write_image(dataset_dir / "rejected.png", 20)
    _write_manifest(
        dataset_dir,
        [_preference_row("shared condition", "chosen.png", "rejected.png")],
    )

    loader = build_offline_train_dataloader(
        _config(tmp_path, [_source("preference", dataset_dir, 0)]),
        _Accelerator(),
        _CountingPreprocessor().preprocess,
        supervision_type="preference",
        pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        shuffle=False,
    )

    (dataset,) = _source_datasets(loader)
    item = dataset[0]
    assert isinstance(item.output, PreferenceOutput)
    assert item.output.chosen_media[0].payload.getpixel((0, 0)) == (220, 220, 220)
    assert item.output.rejected_media[0].payload.getpixel((0, 0)) == (20, 20, 20)
    assert "chosen.png" not in repr(dataset._condition_cache[0])
    assert "rejected.png" not in repr(dataset._condition_cache[0])


def test_builder_preserves_ordered_reference_type_with_single_row_batches(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "ordered"
    dataset_dir.mkdir()
    _write_image(dataset_dir / "reference.png", 50)
    _write_image(dataset_dir / "target.png", 100)
    row = _demonstration_row("ordered condition", "target.png")
    row["input"]["media"] = [{"type": "image", "path": "reference.png"}]
    _write_manifest(dataset_dir, [row])
    preprocessor = _OrderedPreprocessor()

    loader = build_offline_train_dataloader(
        _config(tmp_path, [_source("ordered", dataset_dir, 0)]),
        _Accelerator(),
        preprocessor.preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=_ORDERED_IMAGE_CONTRACT,
        shuffle=False,
    )

    assert preprocessor.references is not None
    assert preprocessor.references[0][0]["type"] == "image"
    (dataset,) = _source_datasets(loader)
    assert dataset[0].model_input.media[0].type == "image"


def test_builder_rejects_unsupported_input_media_before_condition_preprocessing(
    tmp_path: Path,
) -> None:
    """Text-to-image adapters cannot silently discard V2 conditioning media."""
    dataset_dir = tmp_path / "unsupported-input"
    dataset_dir.mkdir()
    _write_image(dataset_dir / "reference.png", 25)
    _write_image(dataset_dir / "target.png", 75)
    row = _demonstration_row("must use the reference", "target.png")
    row["input"]["media"] = [{"type": "image", "path": "reference.png"}]
    _write_manifest(dataset_dir, [row])
    preprocessor = _CountingPreprocessor()

    with pytest.raises(
        ValueError,
        match=r"source 'unsupported-input' row 0.*does not accept input media type 'image'",
    ):
        build_offline_train_dataloader(
            _config(tmp_path, [_source("unsupported-input", dataset_dir, 0)]),
            _Accelerator(),
            preprocessor.preprocess,
            supervision_type="demonstration",
            pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        )

    assert preprocessor.calls == 0


def test_builder_rejects_preprocessor_binding_drift_before_dataset_io(tmp_path: Path) -> None:
    """Projection layout must agree with the adapter-owned binding declaration."""
    with pytest.raises(ValueError, match=r"binding disagrees.*ordered_references.*False"):
        build_offline_train_dataloader(
            _config(tmp_path, [_source("missing", tmp_path / "missing", 0)]),
            _Accelerator(),
            _CountingPreprocessor().preprocess,
            supervision_type="demonstration",
            pipeline_io_contract=_ORDERED_IMAGE_CONTRACT,
        )


def test_builder_rejects_single_sample_pipeline_batching_before_dataset_io(
    tmp_path: Path,
) -> None:
    """Adapter batching capability is enforced before cache or manifest work."""
    contract = replace(
        _TEXT_TO_IMAGE_CONTRACT,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )
    preprocessor = _CountingPreprocessor()

    with pytest.raises(
        ValueError,
        match=r"requires per_device_batch_size=1.*received 2",
    ):
        build_offline_train_dataloader(
            _config(
                tmp_path,
                [_source("missing", tmp_path / "missing", 0)],
                per_device_batch_size=2,
            ),
            _Accelerator(),
            preprocessor.preprocess,
            supervision_type="demonstration",
            pipeline_io_contract=contract,
        )

    assert preprocessor.calls == 0


def test_single_sample_contract_forces_condition_preprocessing_batch_size_one(
    tmp_path: Path,
) -> None:
    """Single-sample adapters must also preprocess their condition cache at B=1."""
    dataset_dir = tmp_path / "single-sample-cache"
    dataset_dir.mkdir()
    rows = []
    for index in range(2):
        target = f"target-{index}.png"
        _write_image(dataset_dir / target, index)
        rows.append(_demonstration_row(f"prompt-{index}", target))
    _write_manifest(dataset_dir, rows)
    contract = replace(
        _TEXT_TO_IMAGE_CONTRACT,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )
    preprocessor = _CountingPreprocessor()

    build_offline_train_dataloader(
        _config(tmp_path, [_source("single-sample-cache", dataset_dir, 0)]),
        _Accelerator(),
        preprocessor.preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=contract,
        shuffle=False,
    )

    assert preprocessor.calls == 2
    assert preprocessor.batch_sizes == [1, 1]


def test_builder_rejects_non_unit_weight_and_unresolved_source_id_before_io(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing"
    preprocessor = _CountingPreprocessor()

    with pytest.raises(ValueError, match=r"weight=1.*got 2"):
        build_offline_train_dataloader(
            _config(tmp_path, [_source("weighted", missing_dir, 0, weight=2)]),
            _Accelerator(),
            preprocessor.preprocess,
            supervision_type="demonstration",
            pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        )
    with pytest.raises(ValueError, match="resolved integer source_id"):
        build_offline_train_dataloader(
            _config(tmp_path, [_source("unresolved", missing_dir, None)]),
            _Accelerator(),
            preprocessor.preprocess,
            supervision_type="demonstration",
            pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        )
    assert preprocessor.calls == 0


def test_builder_rejects_missing_target_decoder_before_condition_preprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        offline_train_data,
        "DEFAULT_MEDIA_DECODERS",
        {
            media_type: decoder
            for media_type, decoder in offline_dataset_module.DEFAULT_MEDIA_DECODERS.items()
            if media_type != "audio"
        },
    )
    dataset_dir = tmp_path / "audio"
    _write_manifest(
        dataset_dir,
        [
            {
                "schema_version": 2,
                "input": {"prompt": "audio target", "media": []},
                "supervision": {
                    "type": "demonstration",
                    "target": {
                        "media": [
                            {
                                "type": "audio",
                                "path": "target.wav",
                                "sample_rate": 16000,
                            }
                        ]
                    },
                },
            }
        ],
    )
    preprocessor = _CountingPreprocessor()

    with pytest.raises(
        NotImplementedError,
        match=r"source 'audio'.*type 'audio'.*target\.wav",
    ):
        build_offline_train_dataloader(
            _config(tmp_path, [_source("audio", dataset_dir, 0)]),
            _Accelerator(),
            preprocessor.preprocess,
            supervision_type="demonstration",
            pipeline_io_contract=_TEXT_TO_AUDIO_CONTRACT,
        )
    assert preprocessor.calls == 0


def test_builder_rejects_output_contract_before_condition_preprocessing(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "wrong-output"
    _write_manifest(
        dataset_dir,
        [
            {
                "schema_version": 2,
                "input": {"prompt": "video target", "media": []},
                "supervision": {
                    "type": "demonstration",
                    "target": {
                        "media": [
                            {
                                "type": "video",
                                "path": "target.mp4",
                                "fps": 24.0,
                            }
                        ]
                    },
                },
            }
        ],
    )
    preprocessor = _CountingPreprocessor()

    with pytest.raises(
        ValueError,
        match=r"source 'wrong-output' row 0 target.*expected.*type 'image'.*'video'",
    ):
        build_offline_train_dataloader(
            _config(tmp_path, [_source("wrong-output", dataset_dir, 0)]),
            _Accelerator(),
            preprocessor.preprocess,
            supervision_type="demonstration",
            pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        )

    assert preprocessor.calls == 0


def test_builder_delegates_distributed_condition_cache_to_rank_safe_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "distributed"
    dataset_dir.mkdir()
    rows = []
    for index in range(4):
        target = f"target-{index}.png"
        _write_image(dataset_dir / target, index)
        rows.append(_demonstration_row(f"prompt-{index}", target, metadata=index))
    _write_manifest(dataset_dir, rows)
    calls: List[Dict[str, Any]] = []

    def fake_create_or_load_dataset(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        raw_dataset = kwargs["base_kwargs"]["raw_dataset"]
        return SimpleNamespace(
            processed_dataset=HFDataset.from_dict(
                {
                    "prompt_embeds": [[1.0, 1.0] for _ in range(len(raw_dataset))],
                    OFFLINE_CONDITION_ID_COLUMN: raw_dataset[OFFLINE_CONDITION_ID_COLUMN],
                }
            )
        )

    monkeypatch.setattr(
        offline_train_data,
        "_create_or_load_dataset",
        fake_create_or_load_dataset,
    )
    accelerator = _Accelerator(num_processes=2, process_index=1)

    loader = build_offline_train_dataloader(
        _config(
            tmp_path,
            [_source("distributed", dataset_dir, 0)],
            preprocess_parallelism="global",
        ),
        accelerator,
        _CountingPreprocessor().preprocess,
        supervision_type="demonstration",
        pipeline_io_contract=_TEXT_TO_IMAGE_CONTRACT,
        shuffle=False,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["accelerator"] is accelerator
    assert call["enable_distributed"] is True
    assert call["preprocess_parallelism"] == "global"
    raw_dataset = call["base_kwargs"]["raw_dataset"]
    assert set(raw_dataset.column_names) == {
        "prompt",
        "negative_prompt",
        OFFLINE_CONDITION_ID_COLUMN,
    }
    assert raw_dataset["negative_prompt"] == ["", "", "", ""]
    assert "target-0.png" not in repr(raw_dataset[0])
    assert "revision" not in repr(raw_dataset[0])
    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.num_replicas == 2
    assert loader.sampler.rank == 1
    assert accelerator.prepare_calls == 0
