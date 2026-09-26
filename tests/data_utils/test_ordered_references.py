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
import wave
from pathlib import Path
from typing import Any, Dict, List

import av
import numpy as np
import pytest
import torch
from PIL import Image

from flow_factory.data_utils.dataset import GeneralDataset, _load_ordered_reference
from flow_factory.models.minimax_h3.adapters import MiniMaxH3Ref2VAAdapter

REFERENCES = [
    {"type": "image", "path": "subject.png"},
    {"type": "video", "path": "motion.mp4", "fps": 29.97},
    {"type": "audio", "path": "voice.wav", "sample_rate": 44100},
]


class OrderedPreprocessor:
    supports_ordered_references = True

    def __init__(self) -> None:
        self.received: List[List[Dict[str, Any]]] = []
        self.received_manifests: List[str] = []

    def preprocess(
        self,
        prompt: List[str],
        references: List[List[Dict[str, Any]]],
        reference_manifest: List[str],
        workflow: str,
        width: int,
        height: int = 512,
        duration: float = 2.0,
        num_frames: int = 49,
        model_name_or_path: str = "MiniMaxAI/MiniMax-H3",
    ) -> Dict[str, Any]:
        self.received = references
        self.received_manifests = reference_manifest
        assert workflow == "ref2va"
        return {"encoded": torch.tensor([[len(references[0])]], dtype=torch.float32)}


class GenericPreprocessor:
    def __init__(self) -> None:
        self.received: Dict[str, Any] = {}

    def preprocess(
        self,
        prompt: List[str],
        images: List[List[Image.Image]],
        videos: List[List[List[Image.Image]]],
        audios: List[List[torch.Tensor]],
    ) -> Dict[str, Any]:
        self.received = {
            "prompt": prompt,
            "images": images,
            "videos": videos,
            "audios": audios,
        }
        return {"encoded": torch.ones(len(prompt), 1)}


def _write_jsonl(dataset_dir: Path, references: List[Dict[str, Any]]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.jsonl").write_text(
        json.dumps({"prompt": "animate", "references": references}) + "\n",
        encoding="utf-8",
    )


def _write_wav(path: Path, sample_rate: int) -> None:
    samples = (np.sin(np.linspace(0, np.pi * 4, sample_rate // 20)) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def _write_video(path: Path, with_audio: bool) -> None:
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("mpeg4", rate=12)
        video_stream.width = 8
        video_stream.height = 6
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=16000) if with_audio else None
        if audio_stream is not None:
            audio_stream.layout = "mono"
        for value in (20, 80, 140):
            frame = av.VideoFrame.from_ndarray(
                np.full((6, 8, 3), value, dtype=np.uint8), format="rgb24"
            )
            for packet in video_stream.encode(frame):
                container.mux(packet)
        if audio_stream is not None:
            waveform = np.zeros((1, 1600), dtype=np.int16)
            audio_frame = av.AudioFrame.from_ndarray(waveform, format="s16", layout="mono")
            audio_frame.sample_rate = 16000
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
        for stream in (video_stream, audio_stream):
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)


def test_ordered_references_round_trip_real_media_and_merged_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    references = [
        {"type": "image", "path": "subject.png"},
        {"type": "audio", "path": "voice.wav", "sample_rate": 22050},
    ]
    _write_jsonl(dataset_dir, references)
    Image.new("RGB", (4, 3), color=(12, 34, 56)).save(dataset_dir / "subject.png")
    _write_wav(dataset_dir / "voice.wav", sample_rate=22050)
    preprocessor = OrderedPreprocessor()
    dataset = GeneralDataset(
        dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={"workflow": "ref2va", "width": 768},
        preprocessing_batch_size=1,
        force_reprocess=True,
    )

    assert len(preprocessor.received) == 1
    assert [entry["type"] for entry in preprocessor.received[0]] == ["image", "audio"]
    assert preprocessor.received[0][0]["media"].size == (4, 3)
    assert preprocessor.received[0][1]["sample_rate"] == 22050
    waveform = preprocessor.received[0][1]["media"]
    assert waveform.shape[0] == 1
    assert waveform.dtype is torch.float32
    assert waveform.device.type == "cpu"
    assert waveform.is_contiguous()
    assert waveform.requires_grad is False
    assert json.loads(preprocessor.received_manifests[0]) == references
    row = dataset[0]
    assert json.loads(row["reference_manifest"]) == references
    assert all(value is not None for value in row.values())
    assert not any(isinstance(value, Image.Image) for value in row.values())
    merged_path = tmp_path / "merged"
    dataset.processed_dataset.save_to_disk(str(merged_path))
    reloaded = GeneralDataset.load_merged(str(merged_path))
    reloaded_row = reloaded[0]
    collated = GeneralDataset.collate_fn([reloaded_row])
    assert collated["reference_manifest"] == [row["reference_manifest"]]
    assert collated["encoded"].shape == (1, 1)


def test_video_reference_preserves_frames_fps_embedded_audio_and_rate(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    references = [{"type": "video", "path": "motion.mp4"}]
    _write_jsonl(dataset_dir, references)
    video_path = dataset_dir / "motion.mp4"
    _write_video(video_path, with_audio=True)

    preprocessor = OrderedPreprocessor()
    GeneralDataset(
        dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        preprocess_kwargs={"workflow": "ref2va", "width": 768},
        preprocessing_batch_size=1,
        force_reprocess=True,
    )

    video = preprocessor.received[0][0]
    assert video["frames"].shape == (3, 6, 8, 3)
    assert video["frames"].dtype == np.uint8
    assert video["frames"].flags.c_contiguous
    assert video["fps"] == pytest.approx(12.0)
    assert video["audio"].shape[0] == 1
    assert video["audio"].dtype is torch.float32
    assert video["audio"].device.type == "cpu"
    assert video["audio"].is_contiguous()
    assert video["audio"].requires_grad is False
    assert video["sample_rate"] == 16000


def test_reference_decode_error_has_row_reference_and_cause(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    rows = [
        {"prompt": "valid", "references": [{"type": "image", "path": "valid.png"}]},
        {"prompt": "missing", "references": [{"type": "image", "path": "missing.png"}]},
    ]
    (dataset_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    Image.new("RGB", (2, 2)).save(dataset_dir / "valid.png")
    preprocessor = OrderedPreprocessor()

    with pytest.raises(
        ValueError,
        match=r"row 1.*reference 0.*image.*missing\.png",
    ) as caught:
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "cache"),
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
            preprocessing_batch_size=1,
            force_reprocess=True,
        )

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_soundtrack_decode_error_reports_soundtrack_path_and_context(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    references = [
        {
            "type": "video",
            "path": "motion.mp4",
            "audio_path": "missing-soundtrack.wav",
        }
    ]
    _write_jsonl(dataset_dir, references)
    _write_video(dataset_dir / "motion.mp4", with_audio=False)
    preprocessor = OrderedPreprocessor()

    with pytest.raises(
        ValueError,
        match=r"row 0.*reference 0.*type='video'.*missing-soundtrack\.wav",
    ) as caught:
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "cache"),
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
            preprocessing_batch_size=1,
            force_reprocess=True,
        )

    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    ("media_type", "decode_target", "decode_result", "rate_name"),
    [
        (
            "video",
            "flow_factory.data_utils.dataset._decode_ordered_video",
            (np.zeros((1, 2, 2, 3), dtype=np.uint8), float("nan"), None, None),
            "fps",
        ),
        (
            "audio",
            "flow_factory.data_utils.dataset._decode_ordered_audio",
            (torch.zeros(1, 8), None),
            "sample_rate",
        ),
        (
            "audio",
            "flow_factory.data_utils.dataset._decode_ordered_audio",
            (torch.zeros(1, 8), -1),
            "sample_rate",
        ),
    ],
)
def test_decoded_rates_must_be_finite_positive_with_context(
    tmp_path: Path,
    monkeypatch,
    media_type: str,
    decode_target: str,
    decode_result: Any,
    rate_name: str,
) -> None:
    dataset_dir = tmp_path / f"dataset-{media_type}-{rate_name}"
    path = "media.mp4" if media_type == "video" else "media.wav"
    references = [
        {"type": "image", "path": "valid.png"},
        {"type": media_type, "path": path},
    ]
    _write_jsonl(dataset_dir, references)
    Image.new("RGB", (2, 2)).save(dataset_dir / "valid.png")
    monkeypatch.setattr(decode_target, lambda media_path: decode_result)
    preprocessor = OrderedPreprocessor()

    with pytest.raises(
        ValueError,
        match=rf"row 0.*reference 1.*type='{media_type}'.*{rate_name}.*finite positive",
    ):
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "cache"),
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
            preprocessing_batch_size=1,
            force_reprocess=True,
        )


def test_video_uses_manifest_fps_when_decoder_has_no_rate(tmp_path: Path, monkeypatch) -> None:
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._decode_ordered_video",
        lambda media_path: (frames, None, None, None),
    )

    loaded = _load_ordered_reference(
        {"type": "video", "path": "media.mp4", "fps": 24.0},
        data_root=str(tmp_path),
        row_index=3,
        reference_index=4,
    )

    assert loaded["frames"] is frames
    assert loaded["fps"] == 24.0


def test_video_without_override_rejects_missing_decoded_fps(tmp_path: Path, monkeypatch) -> None:
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "flow_factory.data_utils.dataset._decode_ordered_video",
        lambda media_path: (frames, None, None, None),
    )

    with pytest.raises(
        ValueError,
        match=r"row 3.*reference 4.*type='video'.*effective fps.*finite positive.*None",
    ):
        _load_ordered_reference(
            {"type": "video", "path": "media.mp4"},
            data_root=str(tmp_path),
            row_index=3,
            reference_index=4,
        )


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
@pytest.mark.parametrize(
    ("media_type", "rate_name", "decode_target", "decode_result"),
    [
        (
            "video",
            "fps",
            "flow_factory.data_utils.dataset._decode_ordered_video",
            (np.zeros((1, 2, 2, 3), dtype=np.uint8), 24.0, None, None),
        ),
        (
            "audio",
            "sample_rate",
            "flow_factory.data_utils.dataset._decode_ordered_audio",
            (torch.zeros(1, 8), 16000),
        ),
    ],
)
def test_effective_override_rates_must_be_finite_positive(
    tmp_path: Path,
    monkeypatch,
    rate: float,
    media_type: str,
    rate_name: str,
    decode_target: str,
    decode_result: Any,
) -> None:
    monkeypatch.setattr(decode_target, lambda media_path: decode_result)
    entry = {"type": media_type, "path": f"media.{media_type}", rate_name: rate}

    with pytest.raises(
        ValueError,
        match=rf"row 3.*reference 4.*type='{media_type}'.*effective {rate_name}.*finite positive",
    ):
        _load_ordered_reference(
            entry,
            data_root=str(tmp_path),
            row_index=3,
            reference_index=4,
        )


def test_generic_real_media_loaders_preserve_structure_cache_and_collate(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    for directory in ("images", "videos", "audios"):
        (dataset_dir / directory).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "prompt": "generic",
                "images": "subject.png",
                "videos": "motion.mp4",
                "audios": "voice.wav",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    Image.new("RGB", (4, 3), color=(1, 2, 3)).save(dataset_dir / "images" / "subject.png")
    _write_video(dataset_dir / "videos" / "motion.mp4", with_audio=False)
    _write_wav(dataset_dir / "audios" / "voice.wav", sample_rate=16000)
    preprocessor = GenericPreprocessor()
    dataset = GeneralDataset(
        dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=preprocessor.preprocess,
        preprocessing_batch_size=1,
        force_reprocess=True,
    )

    assert len(preprocessor.received["images"]) == 1
    assert len(preprocessor.received["images"][0]) == 1
    assert preprocessor.received["images"][0][0].size == (4, 3)
    assert len(preprocessor.received["videos"][0][0]) == 3
    assert preprocessor.received["audios"][0][0].shape[0] == 1
    row = dataset[0]
    merged_path = tmp_path / "generic-merged"
    dataset.processed_dataset.save_to_disk(str(merged_path))
    reloaded = GeneralDataset.load_merged(str(merged_path))
    reloaded_row = reloaded[0]
    collated = GeneralDataset.collate_fn([reloaded_row])
    assert len(collated["images"][0]) == 1
    assert len(collated["videos"][0]) == 1
    assert len(collated["audios"][0]) == 1
    assert collated["videos"][0][0].shape == row["videos"][0].shape
    assert collated["audios"][0][0].shape == row["audios"][0].shape
    assert collated["encoded"].shape == (1, 1)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("workflow", "other-workflow"),
        ("duration", 3.0),
        ("num_frames", 73),
        ("model_name_or_path", "other/model"),
        ("width", 1024),
        ("height", 768),
    ],
)
def test_semantic_preprocess_argument_changes_cache_fingerprint(
    tmp_path: Path, field: str, changed: Any
) -> None:
    dataset_dir = tmp_path / f"dataset-{field}"
    _write_jsonl(dataset_dir, REFERENCES)
    preprocessor = OrderedPreprocessor()
    source = GeneralDataset(
        dataset_dir=str(dataset_dir),
        preprocess_func=preprocessor.preprocess,
        enable_preprocess=False,
    )
    kwargs = {
        "workflow": "ref2va",
        "duration": 2.0,
        "num_frames": 49,
        "model_name_or_path": "MiniMaxAI/MiniMax-H3",
        "width": 768,
        "height": 512,
    }
    changed_kwargs = {**kwargs, field: changed}

    def fingerprint(preprocess_kwargs: Dict[str, Any]) -> str:
        return GeneralDataset.compute_cache_path(
            dataset_dir=str(dataset_dir),
            split="train",
            cache_dir=str(tmp_path / "cache"),
            max_dataset_size=None,
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs=preprocess_kwargs,
            extra_hash_strs=[source._ordered_reference_source_hash],
        )

    assert fingerprint(kwargs) != fingerprint(changed_kwargs)


@pytest.mark.parametrize(
    "changed_references",
    [
        [
            {"type": "image", "path": "subject.png"},
            {"type": "video", "path": "motion.mp4", "fps": 24.0},
            {"type": "audio", "path": "voice.wav", "sample_rate": 44100},
        ],
        [
            {"type": "image", "path": "subject.png"},
            {"type": "video", "path": "motion.mp4", "fps": 29.97},
            {"type": "audio", "path": "voice.wav", "sample_rate": 48000},
        ],
        list(reversed(REFERENCES)),
    ],
)
def test_manifest_rates_and_order_change_source_fingerprint(
    tmp_path: Path, changed_references: List[Dict[str, Any]]
) -> None:
    first_dir = tmp_path / "first" / "dataset"
    second_dir = tmp_path / "second" / "dataset"
    _write_jsonl(first_dir, REFERENCES)
    _write_jsonl(second_dir, changed_references)
    preprocessor = OrderedPreprocessor()

    first = GeneralDataset(
        dataset_dir=str(first_dir),
        preprocess_func=preprocessor.preprocess,
        enable_preprocess=False,
    )
    second = GeneralDataset(
        dataset_dir=str(second_dir),
        preprocess_func=preprocessor.preprocess,
        enable_preprocess=False,
    )

    assert first._ordered_reference_source_hash != second._ordered_reference_source_hash


def test_ref2va_adapter_enables_ordered_reference_preprocessing() -> None:
    assert MiniMaxH3Ref2VAAdapter.supports_ordered_references is True


def test_source_content_changes_the_merged_cache_path_in_place(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_jsonl(dataset_dir, REFERENCES)
    preprocessor = OrderedPreprocessor()

    def cache_path() -> str:
        return GeneralDataset.compute_cache_path(
            dataset_dir=str(dataset_dir),
            split="train",
            cache_dir=str(tmp_path / "cache"),
            max_dataset_size=None,
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
        )

    before = cache_path()
    _write_jsonl(dataset_dir, list(reversed(REFERENCES)))
    after = cache_path()

    assert before != after


def test_dataset_root_participates_in_merged_cache_identity(tmp_path: Path) -> None:
    first_dir = tmp_path / "first" / "dataset"
    second_dir = tmp_path / "second" / "dataset"
    _write_jsonl(first_dir, REFERENCES)
    _write_jsonl(second_dir, REFERENCES)
    preprocessor = OrderedPreprocessor()

    def cache_path(dataset_dir: Path) -> str:
        return GeneralDataset.compute_cache_path(
            dataset_dir=str(dataset_dir),
            split="train",
            cache_dir=str(tmp_path / "cache"),
            max_dataset_size=None,
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
        )

    assert cache_path(first_dir) != cache_path(second_dir)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("height", 768),
        ("width", 1024),
        ("num_frames", 73),
    ],
)
def test_h3_kwargs_wrapper_declares_cache_relevant_geometry(
    tmp_path: Path,
    field: str,
    changed: int,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_jsonl(dataset_dir, REFERENCES)
    adapter = object.__new__(MiniMaxH3Ref2VAAdapter)
    kwargs = {"height": 512, "width": 768, "num_frames": 49}

    def cache_path(preprocess_kwargs: Dict[str, Any]) -> str:
        return GeneralDataset.compute_cache_path(
            dataset_dir=str(dataset_dir),
            split="train",
            cache_dir=str(tmp_path / "cache"),
            max_dataset_size=None,
            preprocess_func=adapter.preprocess_func,
            preprocess_kwargs=preprocess_kwargs,
        )

    assert cache_path(kwargs) != cache_path({**kwargs, field: changed})


def test_adapter_preprocess_cache_version_changes_the_cache_path(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_jsonl(dataset_dir, REFERENCES)
    adapter = object.__new__(MiniMaxH3Ref2VAAdapter)

    def cache_path() -> str:
        return GeneralDataset.compute_cache_path(
            dataset_dir=str(dataset_dir),
            split="train",
            cache_dir=str(tmp_path / "cache"),
            max_dataset_size=None,
            preprocess_func=adapter.preprocess_func,
            preprocess_kwargs={"height": 512, "width": 768, "num_frames": 49},
        )

    adapter.preprocess_cache_version = "first"
    before = cache_path()
    adapter.preprocess_cache_version = "second"
    after = cache_path()

    assert before != after


def test_generic_media_preprocessing_does_not_enable_ordered_references(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_jsonl(dataset_dir, REFERENCES)

    def generic(prompt: List[str], images=None, videos=None, audios=None) -> Dict[str, Any]:
        assert images is None
        assert videos is None
        assert audios is None
        return {"encoded": torch.ones(len(prompt), 1)}

    dataset = GeneralDataset(
        dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "cache"),
        preprocess_func=generic,
        preprocessing_batch_size=1,
        force_reprocess=True,
    )

    assert "reference_manifest" not in dataset.processed_dataset.column_names
    assert dataset[0]["metadata"]["references"] == REFERENCES


def test_ordered_reference_preprocessing_rejects_outer_batch_larger_than_one(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    rows = [
        {"prompt": "first", "references": REFERENCES},
        {"prompt": "second", "references": REFERENCES},
    ]
    (dataset_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    preprocessor = OrderedPreprocessor()
    with pytest.raises(ValueError, match=r"ordered-reference.*expected B=1.*received B=2"):
        GeneralDataset(
            dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "cache"),
            preprocess_func=preprocessor.preprocess,
            preprocess_kwargs={"workflow": "ref2va", "width": 768},
            preprocessing_batch_size=2,
            force_reprocess=True,
        )
