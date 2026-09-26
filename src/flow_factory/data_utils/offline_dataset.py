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

"""Offline records, on-demand target decoding, and explicit collation.

The input-condition cache and output supervision intentionally have different
lifecycles. Prompt and condition encodings are fetched from a cache by stable
record index, while target, chosen, and rejected media are decoded from their
source paths on every ``__getitem__`` call. This module never caches output
pixels or model-specific output latents and never holds a model adapter.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Mapping, MutableMapping, Sequence, Union

import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import Dataset

from ..utils.audio import load_audio

try:
    import av
except ImportError:
    av = None

from .schema import (
    DatasetRecordV2,
    DemonstrationSupervision,
    MediaAsset,
    MediaType,
    NormalizedDatasetRecord,
    NormalizedModelInput,
    NormalizedOutputCandidate,
    PreferenceSupervision,
    normalize_v2_record,
)

OfflineSupervisionType = Literal["demonstration", "preference"]
MediaDecoder = Callable[[MediaAsset], Any]
ConditionCache = Union[Dataset, Sequence[Mapping[str, Any]]]
OFFLINE_CONDITION_ID_COLUMN = "__offline_condition_id__"
_MEDIA_CONTENT_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DecodedMedia:
    """One source media asset decoded on the CPU for the current item access."""

    type: MediaType
    path: str
    payload: Any
    fps: float | None = None
    sample_rate: int | None = None


@dataclass(frozen=True, slots=True)
class DemonstrationOutput:
    """Decoded output for one demonstration item."""

    target_media: tuple[DecodedMedia, ...]


@dataclass(frozen=True, slots=True)
class PreferenceOutput:
    """Decoded pairwise output for one preference item."""

    chosen_media: tuple[DecodedMedia, ...]
    rejected_media: tuple[DecodedMedia, ...]


DecodedOutput = Union[DemonstrationOutput, PreferenceOutput]


@dataclass(frozen=True, slots=True)
class OfflineItem:
    """One cached input condition paired with freshly decoded supervision."""

    condition: Mapping[str, Any]
    condition_id: str
    record_id: str
    source: str
    source_id: int
    model_input: NormalizedModelInput
    supervision_type: OfflineSupervisionType
    output: DecodedOutput
    metadata_json: str


@dataclass(frozen=True, slots=True)
class DemonstrationOutputBatch:
    """Ragged sample-to-ordered-media demonstration outputs."""

    target_media: tuple[tuple[DecodedMedia, ...], ...]


@dataclass(frozen=True, slots=True)
class PreferenceOutputBatch:
    """Ragged sample-to-ordered-media pairwise outputs."""

    chosen_media: tuple[tuple[DecodedMedia, ...], ...]
    rejected_media: tuple[tuple[DecodedMedia, ...], ...]


DecodedOutputBatch = Union[DemonstrationOutputBatch, PreferenceOutputBatch]


@dataclass(frozen=True, slots=True)
class OfflineBatch:
    """Explicit offline batch without model- or algorithm-specific tensor fields."""

    condition: Dict[str, Any]
    condition_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    sources: tuple[str, ...]
    source_ids: torch.Tensor
    model_inputs: tuple[NormalizedModelInput, ...]
    supervision_type: OfflineSupervisionType
    output: DecodedOutputBatch
    metadata_json: tuple[str, ...]


def load_offline_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    supervision_type: OfflineSupervisionType,
    dataset_dir: str | os.PathLike[str] | None = None,
) -> tuple[NormalizedDatasetRecord, ...]:
    """Strictly parse and normalize one homogeneous offline JSONL split.

    Args:
        manifest_path: Path to the split JSONL manifest.
        supervision_type: Required supervision discriminator for every row.
        dataset_dir: Root for relative media paths. Defaults to the manifest's
            parent directory.

    Returns:
        Immutable normalized records in manifest order.

    Raises:
        ValueError: If the manifest is empty, a line is blank or invalid, or a
            row does not carry the required homogeneous supervision type.
    """
    _validate_supervision_type(supervision_type)
    resolved_manifest_path = os.path.abspath(os.path.expanduser(os.fspath(manifest_path)))
    resolved_dataset_dir = (
        os.path.dirname(resolved_manifest_path)
        if dataset_dir is None
        else os.path.abspath(os.path.expanduser(os.fspath(dataset_dir)))
    )
    records: List[NormalizedDatasetRecord] = []

    with open(resolved_manifest_path, encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            context = f"{resolved_manifest_path}:{line_number}"
            if not line.strip():
                raise ValueError(f"invalid offline JSONL record at {context}: blank line")
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid offline JSONL record at {context}: {exc.msg}") from exc
            try:
                parsed_record = DatasetRecordV2.model_validate(raw_record)
            except ValidationError as exc:
                raise ValueError(f"invalid offline V2 record at {context}: {exc}") from exc

            normalized_record = normalize_v2_record(
                parsed_record,
                dataset_dir=resolved_dataset_dir,
            )
            _require_record_supervision(
                normalized_record,
                supervision_type,
                context=context,
            )
            records.append(normalized_record)

    if not records:
        raise ValueError(f"offline JSONL manifest is empty: {resolved_manifest_path}")
    return tuple(records)


class OfflineDataset(Dataset):
    """Bind normalized offline records to a separately built condition cache.

    Every condition-cache row must carry :data:`OFFLINE_CONDITION_ID_COLUMN`.
    Its input-only identity must match ``records[index]`` before the cache is
    accepted and is checked again when the row is fetched. Plain list and tuple
    caches are snapshotted so caller-side reordering cannot change the binding.
    """

    def __init__(
        self,
        records: Sequence[NormalizedDatasetRecord],
        condition_cache: ConditionCache,
        *,
        source_name: str,
        source_id: int,
        supervision_type: OfflineSupervisionType,
        media_decoders: Mapping[MediaType, MediaDecoder] | None = None,
        _media_digest_cache: MutableMapping[str, str] | None = None,
    ) -> None:
        _validate_supervision_type(supervision_type)
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError(f"offline source_name must be a non-empty string, got {source_name!r}")
        if not isinstance(source_id, int) or isinstance(source_id, bool) or source_id < 0:
            raise ValueError(f"offline source_id must be a non-negative integer, got {source_id!r}")
        normalized_records = tuple(records)
        stable_condition_cache = (
            tuple(condition_cache)
            if isinstance(condition_cache, (list, tuple))
            else condition_cache
        )
        if not normalized_records:
            raise ValueError("offline dataset requires at least one normalized record")
        if len(normalized_records) != len(stable_condition_cache):
            raise ValueError(
                "offline records and condition cache must have equal length, "
                f"got {len(normalized_records)} and {len(stable_condition_cache)}"
            )

        for index, record in enumerate(normalized_records):
            if not isinstance(record, NormalizedDatasetRecord):
                raise TypeError(
                    "offline dataset accepts normalized V2 records only, "
                    f"got {type(record).__name__} at index {index}"
                )
            _require_record_supervision(
                record,
                supervision_type,
                context=f"offline dataset index {index}",
            )

        decoders: Dict[MediaType, MediaDecoder] = dict(DEFAULT_MEDIA_DECODERS)
        if media_decoders is not None:
            for media_type, decoder in media_decoders.items():
                if media_type not in ("image", "video", "audio"):
                    raise ValueError(f"unsupported media decoder type: {media_type!r}")
                _require_picklable_unbound_decoder(media_type, decoder)
                decoders[media_type] = decoder
        _require_supervision_decoder_coverage(
            normalized_records,
            decoders,
            source_name=source_name,
        )

        condition_ids: List[str] = []
        record_ids: List[str] = []
        media_digest_cache: MutableMapping[str, str] = (
            {} if _media_digest_cache is None else _media_digest_cache
        )
        for index, record in enumerate(normalized_records):
            condition_id = compute_offline_condition_id(
                record,
                index=index,
                source_name=source_name,
                _media_digest_cache=media_digest_cache,
            )
            record_id = compute_offline_record_id(
                record,
                index=index,
                source_name=source_name,
                _media_digest_cache=media_digest_cache,
            )
            _extract_condition(
                stable_condition_cache[index],
                expected_condition_id=condition_id,
                index=index,
                mismatch_error=ValueError,
            )
            condition_ids.append(condition_id)
            record_ids.append(record_id)

        self._records = normalized_records
        self._condition_cache = stable_condition_cache
        self._condition_cache_length = len(stable_condition_cache)
        self._condition_ids = tuple(condition_ids)
        self._record_ids = tuple(record_ids)
        self._media_decoders = decoders
        self.source_name = source_name
        self.source_id = source_id
        self.supervision_type = supervision_type

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> OfflineItem:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"offline dataset index must be an integer, got {index!r}")
        if index < 0:
            index += len(self._records)
        if index < 0 or index >= len(self._records):
            raise IndexError(f"offline dataset index out of range: {index}")
        if len(self._condition_cache) != self._condition_cache_length:
            raise RuntimeError(
                "condition cache length changed after offline dataset construction; "
                "stable index binding is no longer guaranteed"
            )
        record = self._records[index]
        condition_id = self._condition_ids[index]
        record_id = self._record_ids[index]
        condition = _extract_condition(
            self._condition_cache[index],
            expected_condition_id=condition_id,
            index=index,
            mismatch_error=RuntimeError,
        )

        supervision = record.supervision
        if self.supervision_type == "demonstration":
            if not isinstance(supervision, DemonstrationSupervision):
                raise RuntimeError(
                    f"offline dataset supervision changed unexpectedly at index {index}"
                )
            output: DecodedOutput = DemonstrationOutput(
                target_media=self._decode_candidate(supervision.target)
            )
        else:
            if not isinstance(supervision, PreferenceSupervision):
                raise RuntimeError(
                    f"offline dataset supervision changed unexpectedly at index {index}"
                )
            output = PreferenceOutput(
                chosen_media=self._decode_candidate(supervision.chosen),
                rejected_media=self._decode_candidate(supervision.rejected),
            )

        return OfflineItem(
            condition=condition,
            condition_id=condition_id,
            record_id=record_id,
            source=self.source_name,
            source_id=self.source_id,
            model_input=record.model_input,
            supervision_type=self.supervision_type,
            output=output,
            metadata_json=record.metadata_json,
        )

    def _decode_candidate(
        self,
        candidate: NormalizedOutputCandidate,
    ) -> tuple[DecodedMedia, ...]:
        return tuple(self._decode_media(asset) for asset in candidate.media)

    def _decode_media(self, asset: MediaAsset) -> DecodedMedia:
        decoder = self._media_decoders.get(asset.type)
        if decoder is None:
            raise NotImplementedError(
                f"offline target media type {asset.type!r} has no decoder for {asset.path!r}; "
                "inject a module-level media decoder explicitly"
            )
        payload = decoder(asset)
        return DecodedMedia(
            type=asset.type,
            path=asset.path,
            payload=payload,
            fps=asset.fps,
            sample_rate=asset.sample_rate,
        )


@dataclass(frozen=True, slots=True)
class OfflineCollator:
    """Collate one known offline supervision branch without row-based guessing."""

    supervision_type: OfflineSupervisionType

    def __post_init__(self) -> None:
        _validate_supervision_type(self.supervision_type)

    def __call__(self, items: Sequence[OfflineItem]) -> OfflineBatch:
        if not items:
            raise ValueError("cannot collate an empty offline batch")
        for index, item in enumerate(items):
            if not isinstance(item, OfflineItem):
                raise TypeError(
                    f"offline collator expected OfflineItem at index {index}, "
                    f"got {type(item).__name__}"
                )
            if item.supervision_type != self.supervision_type:
                raise ValueError(
                    "offline collator supervision mismatch at batch index "
                    f"{index}: expected {self.supervision_type!r}, "
                    f"got {item.supervision_type!r}"
                )

        if self.supervision_type == "demonstration":
            target_media: List[tuple[DecodedMedia, ...]] = []
            for index, item in enumerate(items):
                if not isinstance(item.output, DemonstrationOutput):
                    raise TypeError(
                        "demonstration collator expected DemonstrationOutput at batch index "
                        f"{index}, got {type(item.output).__name__}"
                    )
                target_media.append(item.output.target_media)
            output: DecodedOutputBatch = DemonstrationOutputBatch(target_media=tuple(target_media))
        else:
            chosen_media: List[tuple[DecodedMedia, ...]] = []
            rejected_media: List[tuple[DecodedMedia, ...]] = []
            for index, item in enumerate(items):
                if not isinstance(item.output, PreferenceOutput):
                    raise TypeError(
                        "preference collator expected PreferenceOutput at batch index "
                        f"{index}, got {type(item.output).__name__}"
                    )
                chosen_media.append(item.output.chosen_media)
                rejected_media.append(item.output.rejected_media)
            output = PreferenceOutputBatch(
                chosen_media=tuple(chosen_media),
                rejected_media=tuple(rejected_media),
            )

        return OfflineBatch(
            condition=_collate_condition_mappings([item.condition for item in items]),
            condition_ids=tuple(item.condition_id for item in items),
            record_ids=tuple(item.record_id for item in items),
            sources=tuple(item.source for item in items),
            source_ids=torch.tensor([item.source_id for item in items], dtype=torch.long),
            model_inputs=tuple(item.model_input for item in items),
            supervision_type=self.supervision_type,
            output=output,
            metadata_json=tuple(item.metadata_json for item in items),
        )


def decode_image(asset: MediaAsset) -> Image.Image:
    """Decode one target image as detached RGB pixels on the CPU.

    Args:
        asset: Normalized image reference with a resolved local path.

    Returns:
        Detached RGB PIL image.

    Raises:
        ValueError: If PIL cannot decode the target image.
    """
    try:
        with Image.open(asset.path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to decode target image {asset.path!r}: {exc}") from exc


def decode_video(asset: MediaAsset) -> np.ndarray:
    """Decode one target video into native-rate RGB frames on the CPU.

    The returned canonical decoded-target representation is a C-contiguous ``uint8``
    array shaped ``(frames, height, width, 3)``. Output codecs validate that byte
    boundary and convert it exactly once to floating ``[0, 1]`` pixels before any
    processor that expects unit-range NumPy input. Temporal sampling, spatial resizing,
    and model-specific normalization remain adapter-owned. Keeping this function at
    module scope makes the default decoder safe to pickle under spawn-based DataLoader
    workers.

    Args:
        asset: Normalized video reference with a resolved local path.

    Returns:
        Contiguous ``uint8`` RGB array shaped ``(frames, height, width, 3)``.

    Raises:
        ImportError: If PyAV is unavailable.
        ValueError: If the container, stream, or decoded geometry is invalid.
    """
    if av is None:
        raise ImportError(
            "offline target video decoding requires PyAV>=17.0.0; "
            "install with `pip install 'av>=17.0.0'`"
        )
    try:
        with av.open(asset.path) as container:
            if not container.streams.video:
                raise ValueError("container has no video stream")
            stream = container.streams.video[0]
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        raise ValueError(f"failed to decode target video {asset.path!r}: {exc}") from exc

    if not frames:
        raise ValueError(f"failed to decode target video {asset.path!r}: decoded no frames")
    try:
        video = np.stack(frames, axis=0)
    except ValueError as exc:
        raise ValueError(
            f"failed to decode target video {asset.path!r}: decoded frames have inconsistent geometry"
        ) from exc
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(
            f"failed to decode target video {asset.path!r}: expected RGB frames shaped "
            f"(F,H,W,3), received {video.shape}"
        )
    return np.ascontiguousarray(video, dtype=np.uint8)


def decode_audio(asset: MediaAsset) -> torch.Tensor:
    """Decode one target audio asset as a detached CPU waveform.

    The returned contiguous ``float32`` tensor has shape ``(channels, samples)``.
    The manifest ``sample_rate`` is a logical source-clock override, so decoding
    intentionally preserves the file's samples instead of resampling them. The
    :class:`DecodedMedia` boundary carries that override to the model codec, which
    owns source-clock truncation, channel conversion, and the single model-rate
    resample. This mirrors the source-clock semantics of video ``fps`` overrides.
    Keeping this function at module scope makes the default decoder safe to pickle
    under spawn-based DataLoader workers.

    Args:
        asset: Normalized audio reference with a resolved local path and optional rate override.

    Returns:
        Detached contiguous CPU waveform shaped ``(channels, samples)``.

    Raises:
        TypeError: If the audio backend returns a non-tensor or non-floating payload.
        ValueError: If the decoded waveform is empty, non-finite, or not two-dimensional.
    """
    waveform = load_audio(asset.path, sample_rate=None)
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(
            f"failed to decode target audio {asset.path!r}: expected torch.Tensor, "
            f"received {type(waveform).__name__}"
        )
    if not waveform.is_floating_point():
        raise TypeError(
            f"failed to decode target audio {asset.path!r}: expected floating waveform, "
            f"received {waveform.dtype}"
        )
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise ValueError(
            f"failed to decode target audio {asset.path!r}: expected non-empty waveform "
            f"shaped (channels,samples), received {tuple(waveform.shape)}"
        )
    if not torch.isfinite(waveform).all():
        raise ValueError(
            f"failed to decode target audio {asset.path!r}: waveform contains non-finite values"
        )
    return waveform


DEFAULT_MEDIA_DECODERS: Mapping[MediaType, MediaDecoder] = MappingProxyType(
    {
        "image": decode_image,
        "video": decode_video,
        "audio": decode_audio,
    }
)


def compute_offline_condition_id(
    record: NormalizedDatasetRecord,
    *,
    index: int,
    source_name: str,
    _media_digest_cache: MutableMapping[str, str] | None = None,
) -> str:
    """Build an input-only identity for one prompt/condition cache row.

    Media bytes are streamed into the identity. Callers building several rows may
    supply one private path-to-digest memo so a repeated asset is read only once;
    the ordinary standalone call remains self-contained.
    """
    _validate_identity_inputs(record, index=index, source_name=source_name)
    media_digest_cache: MutableMapping[str, str] = (
        {} if _media_digest_cache is None else _media_digest_cache
    )
    identity_payload = {
        "source_name": source_name,
        "index": index,
        "schema_version": record.schema_version,
        "input": {
            "prompt": record.model_input.prompt,
            "negative_prompt": record.model_input.negative_prompt,
            "media": [
                _media_identity(media, media_digest_cache=media_digest_cache)
                for media in record.model_input.media
            ],
        },
    }
    return _hash_identity(identity_payload)


def compute_offline_record_id(
    record: NormalizedDatasetRecord,
    *,
    index: int,
    source_name: str,
    _media_digest_cache: MutableMapping[str, str] | None = None,
) -> str:
    """Build a full provenance identity including supervision bytes and metadata."""
    media_digest_cache: MutableMapping[str, str] = (
        {} if _media_digest_cache is None else _media_digest_cache
    )
    condition_id = compute_offline_condition_id(
        record,
        index=index,
        source_name=source_name,
        _media_digest_cache=media_digest_cache,
    )
    supervision = record.supervision
    if isinstance(supervision, DemonstrationSupervision):
        supervision_payload: Any = {
            "type": "demonstration",
            "target": _candidate_identity(
                supervision.target,
                media_digest_cache=media_digest_cache,
            ),
        }
    elif isinstance(supervision, PreferenceSupervision):
        supervision_payload = {
            "type": "preference",
            "chosen": _candidate_identity(
                supervision.chosen,
                media_digest_cache=media_digest_cache,
            ),
            "rejected": _candidate_identity(
                supervision.rejected,
                media_digest_cache=media_digest_cache,
            ),
        }
    else:
        supervision_payload = None
    return _hash_identity(
        {
            "condition_id": condition_id,
            "supervision": supervision_payload,
            "metadata_json": record.metadata_json,
        }
    )


def _validate_identity_inputs(
    record: NormalizedDatasetRecord,
    *,
    index: int,
    source_name: str,
) -> None:
    if not isinstance(record, NormalizedDatasetRecord):
        raise TypeError(
            "offline identity requires NormalizedDatasetRecord, " f"got {type(record).__name__}"
        )
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"offline record index must be a non-negative integer, got {index!r}")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError(f"offline source_name must be a non-empty string, got {source_name!r}")


def _hash_identity(identity_payload: Mapping[str, Any]) -> str:
    canonical_identity = json.dumps(
        identity_payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()


def _candidate_identity(
    candidate: NormalizedOutputCandidate,
    *,
    media_digest_cache: MutableMapping[str, str],
) -> Dict[str, Any]:
    return {
        "media": [
            _media_identity(media, media_digest_cache=media_digest_cache)
            for media in candidate.media
        ]
    }


def _media_identity(
    media: MediaAsset,
    *,
    media_digest_cache: MutableMapping[str, str],
) -> Dict[str, Any]:
    return {
        "type": media.type,
        "path": media.path,
        "fps": media.fps,
        "sample_rate": media.sample_rate,
        "slot": media.slot,
        "content_sha256": _media_content_sha256(
            media.path,
            media_digest_cache=media_digest_cache,
        ),
    }


def _media_content_sha256(
    path: str,
    *,
    media_digest_cache: MutableMapping[str, str],
) -> str:
    """Return one memoized content digest for a normalized local media path."""
    cached = media_digest_cache.get(path)
    if cached is not None:
        return cached
    content_digest = _stream_media_content_sha256(path)
    media_digest_cache[path] = content_digest
    return content_digest


def _stream_media_content_sha256(path: str) -> str:
    """Hash one media file incrementally without decoding or retaining its bytes."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as media_file:
            while chunk := media_file.read(_MEDIA_CONTENT_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as error:
        raise OSError(f"failed to hash offline media content at {path!r}: {error}") from error
    return digest.hexdigest()


def _extract_condition(
    row: Any,
    *,
    expected_condition_id: str,
    index: int,
    mismatch_error: type[Exception],
) -> Dict[str, Any]:
    if not isinstance(row, Mapping):
        raise mismatch_error(
            "condition cache row must be a mapping at index " f"{index}, got {type(row).__name__}"
        )
    if OFFLINE_CONDITION_ID_COLUMN not in row:
        raise mismatch_error(
            f"condition cache row at index {index} is missing reserved identity column "
            f"{OFFLINE_CONDITION_ID_COLUMN!r}"
        )
    actual_condition_id = row[OFFLINE_CONDITION_ID_COLUMN]
    if actual_condition_id != expected_condition_id:
        raise mismatch_error(
            "condition cache identity mismatch at index "
            f"{index}: expected {expected_condition_id!r}, got {actual_condition_id!r}"
        )
    return {key: value for key, value in row.items() if key != OFFLINE_CONDITION_ID_COLUMN}


def _require_supervision_decoder_coverage(
    records: Sequence[NormalizedDatasetRecord],
    decoders: Mapping[MediaType, MediaDecoder],
    *,
    source_name: str,
) -> None:
    for record_index, record in enumerate(records):
        supervision = record.supervision
        candidates: tuple[tuple[str, NormalizedOutputCandidate], ...]
        if isinstance(supervision, DemonstrationSupervision):
            candidates = (("target", supervision.target),)
        elif isinstance(supervision, PreferenceSupervision):
            candidates = (
                ("chosen", supervision.chosen),
                ("rejected", supervision.rejected),
            )
        else:
            continue
        for candidate_name, candidate in candidates:
            for media_index, asset in enumerate(candidate.media):
                if asset.type not in decoders:
                    raise NotImplementedError(
                        f"offline source {source_name!r} dataset index {record_index} "
                        f"{candidate_name} media {media_index} at {asset.path!r} has type "
                        f"{asset.type!r} with no decoder; "
                        "inject a pickleable module-level function explicitly"
                    )


def _collate_condition_mappings(
    conditions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_keys = tuple(conditions[0].keys())
    expected_key_set = set(expected_keys)
    for index, condition in enumerate(conditions):
        if set(condition.keys()) != expected_key_set:
            raise ValueError(
                "condition cache mappings must expose identical keys within a batch; "
                f"expected {sorted(expected_key_set)!r}, got "
                f"{sorted(condition.keys())!r} at batch index {index}"
            )

    collated: Dict[str, Any] = {}
    for key in expected_keys:
        values = [condition[key] for condition in conditions]
        if all(isinstance(value, torch.Tensor) for value in values):
            shapes = [value.shape for value in values]
            collated[key] = (
                torch.stack(values, dim=0)
                if all(shape == shapes[0] for shape in shapes)
                else values
            )
        elif any(isinstance(value, torch.Tensor) for value in values) and any(
            isinstance(value, list) for value in values
        ):
            collated[key] = [
                list(torch.unbind(value, dim=0)) if isinstance(value, torch.Tensor) else value
                for value in values
            ]
        else:
            collated[key] = values
    return collated


def _validate_supervision_type(supervision_type: str) -> None:
    if supervision_type not in ("demonstration", "preference"):
        raise ValueError(
            "offline supervision_type must be 'demonstration' or 'preference', "
            f"got {supervision_type!r}"
        )


def _require_picklable_unbound_decoder(
    media_type: MediaType,
    decoder: MediaDecoder,
) -> None:
    if not inspect.isfunction(decoder):
        raise TypeError(
            f"decoder for media type {media_type!r} must be a module-level function, "
            f"got {type(decoder).__name__}"
        )
    qualified_name = getattr(decoder, "__qualname__", "")
    if qualified_name == "<lambda>" or "<locals>" in qualified_name:
        raise TypeError(
            f"decoder for media type {media_type!r} must be defined at module scope for spawn"
        )
    try:
        pickle.dumps(decoder)
    except (AttributeError, pickle.PicklingError, TypeError) as exc:
        raise TypeError(
            f"decoder for media type {media_type!r} must be pickleable for spawn workers"
        ) from exc


def _require_record_supervision(
    record: NormalizedDatasetRecord,
    supervision_type: OfflineSupervisionType,
    *,
    context: str,
) -> None:
    supervision = record.supervision
    if supervision is None:
        raise ValueError(
            f"{context} is prompt-only; offline datasets require {supervision_type!r} supervision"
        )
    actual_type: OfflineSupervisionType = (
        "demonstration" if isinstance(supervision, DemonstrationSupervision) else "preference"
    )
    if actual_type != supervision_type:
        raise ValueError(
            f"{context} has {actual_type!r} supervision, expected homogeneous "
            f"{supervision_type!r} supervision"
        )


__all__ = [
    "DEFAULT_MEDIA_DECODERS",
    "DecodedMedia",
    "DemonstrationOutput",
    "DemonstrationOutputBatch",
    "MediaDecoder",
    "OfflineBatch",
    "OfflineCollator",
    "OfflineDataset",
    "OfflineItem",
    "OfflineSupervisionType",
    "OFFLINE_CONDITION_ID_COLUMN",
    "PreferenceOutput",
    "PreferenceOutputBatch",
    "compute_offline_condition_id",
    "compute_offline_record_id",
    "decode_audio",
    "decode_image",
    "decode_video",
    "load_offline_manifest",
]
