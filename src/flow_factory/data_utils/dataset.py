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

# src/flow_factory/data_utils/dataset.py
import hashlib
import inspect
import json
import logging
import math
import os
import random
import shutil
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Union

import imageio.v3 as iio
import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import Features as HFFeatures
from datasets import Image as HFImage
from datasets import Sequence as HFSequence
from datasets import load_dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar
from PIL import Image
from torch.utils.data import Dataset

from ..samples.references import canonicalize_reference_manifest, parse_reference_manifest
from ..utils.audio import load_audio, require_decoded_audio_waveform
from ..utils.base import (
    filter_kwargs,
    pil_image_to_tensor,
    standardize_image_batch,
)
from ..utils.image import require_decoded_rgb_image
from ..utils.logger_utils import setup_logger
from ..utils.video import require_decoded_video_frames

try:
    import av
except ImportError:
    av = None

logger = setup_logger(__name__, rank_zero_only=True)

# Bump when the on-disk preprocessed format changes in a backward-incompatible
# way (e.g. image columns switched from raw tensors to the HF Image feature), so
# stale caches written by an older format are not silently reused.
_PREPROCESS_FORMAT_VERSION = 2

# Column that carries raw per-sample JSONL fields (packed by ``_preprocess_batch``,
# consumed by ``BaseTrainer._inject_batch_metadata`` via ``json.dumps``).
METADATA_COLUMN = "metadata"

# Non-image columns kept in the HF "python" format by name (in addition to image
# columns, which are auto-detected by feature type via ``_is_image_feature``).
# The torch formatter would recursively tensorize ``METADATA_COLUMN``'s numeric
# values (e.g. an int becomes a 0-dim Tensor), breaking JSON serialization.
EXTRA_PYTHON_FORMAT_COLUMNS = frozenset({METADATA_COLUMN})


# ========================================================================================
# Protocol Definitions
# ========================================================================================


class TextEncodeCallable(Protocol):
    """Protocol for text encoding functions."""

    def __call__(self, prompt: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]: ...


class ImageEncodeCallable(Protocol):
    """Protocol for image encoding functions."""

    def __call__(
        self, image: Union[Image.Image, List[Image.Image]], **kwargs: Any
    ) -> Dict[str, Any]: ...


class VideoEncodeCallable(Protocol):
    """Protocol for video encoding functions."""

    def __call__(
        self, video: Union[List[Image.Image], List[List[Image.Image]]], **kwargs: Any
    ) -> Dict[str, Any]: ...


class PreprocessCallable(Protocol):
    """Protocol for preprocessing functions that handle multi-modal inputs."""

    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]],
        videos: Optional[
            Union[List[Image.Image], List[List[Image.Image]], List[List[List[Image.Image]]]]
        ],
        **kwargs: Any,
    ) -> Dict[str, Any]: ...


# ========================================================================================
# GeneralDataset Class
# ========================================================================================


class GeneralDataset(Dataset):
    """
    General-purpose dataset for multi-modal data (text, images, videos).

    Supports:
    - Loading from JSONL or TXT files
    - Optional preprocessing with caching
    - Distributed preprocessing across multiple GPUs
    - Automatic cache management and merging
    """

    @staticmethod
    def check_exists(dataset_dir: str, split: str) -> bool:
        """Check if dataset files exist for a given split."""
        dataset_dir = os.path.expanduser(dataset_dir)
        jsonl_path = os.path.join(dataset_dir, f"{split}.jsonl")
        txt_path = os.path.join(dataset_dir, f"{split}.txt")
        return os.path.exists(jsonl_path) or os.path.exists(txt_path)

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        cache_dir: str = "~/.cache/flow_factory/datasets",
        enable_preprocess: bool = True,
        force_reprocess: bool = False,
        preprocessing_batch_size: int = 16,
        max_dataset_size: Optional[int] = None,
        preprocess_func: Optional[PreprocessCallable] = None,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        num_shards: Optional[int] = None,
        shard_index: Optional[int] = None,
        extra_hash_strs: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        audio_dir: Optional[str] = None,
        target_arrow_path: Optional[str] = None,
        raw_dataset: Optional[HFDataset] = None,
        source_hash_override: Optional[str] = None,
        passthrough_columns: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        Initialize GeneralDataset.

        Args:
            dataset_dir: Path to dataset directory
            split: Dataset split ('train', 'test', etc.)
            cache_dir: Directory for caching preprocessed data
            enable_preprocess: Whether to enable preprocessing
            force_reprocess: Force reprocessing even if cache exists
            preprocessing_batch_size: Batch size for preprocessing
            max_dataset_size: Limit dataset size to this many samples
            preprocess_func: Function to preprocess batches
            preprocess_kwargs: Additional kwargs for preprocess_func
            num_shards: Total number of shards for distributed preprocessing
            shard_index: Current shard index (0 to num_shards-1)
            extra_hash_strs: Extra strings concatenated into the cache
                fingerprint (e.g. model identifiers) so two runs that differ
                only in those strings get distinct caches.
            image_dir: Override for the image root directory. When ``None``,
                JSONL datasets default to ``{dataset_dir}/images`` and TXT
                datasets stay ``None`` (no image loading).
            video_dir: Override for the video root directory. Same default
                resolution as ``image_dir``, with ``{dataset_dir}/videos``.
            audio_dir: Override for the audio root directory. Same default
                resolution as ``image_dir``, with ``{dataset_dir}/audios``.
            target_arrow_path: If provided, route ``Dataset.map`` output directly
                to this Arrow file via ``cache_file_name=``. The orchestrator
                (``loader._create_or_load_dataset``) sets this so each rank's
                preprocessed bytes land at their final per-rank location and
                the main rank can metadata-merge them without re-serialization.
                When ``None``, HF falls back to its default cache path under
                ``~/.cache/huggingface/datasets`` (single-process / legacy).
            raw_dataset: Optional in-memory HuggingFace dataset. When provided,
                file discovery is skipped and ``source_hash_override`` is required.
                This is intended for narrow projections such as offline input-only
                condition caches, not for passing a complete source manifest.
            source_hash_override: Stable source-content identity used in place of
                hashing ``{dataset_dir}/{split}.jsonl`` or ``.txt``. Required with
                ``raw_dataset`` so unrelated in-memory datasets cannot share a cache.
            passthrough_columns: Raw columns that must survive preprocessing at the
                top level unchanged. They are never forwarded to ``preprocess_func``
                or copied into ``metadata``. A preprocess result using one of these
                names is rejected as a collision.
            **kwargs: Additional arguments (ignored)

        Note:
            ``image_dir``, ``video_dir`` and ``audio_dir`` are NOT included in
            the cache fingerprint. If your JSONL stores RELATIVE asset paths
            and you switch one of these directories between runs while
            keeping every other config bit identical, the existing cache will
            be reused with stale data. Set ``force_reprocess=True`` once after
            such a switch, or include the directory in ``extra_hash_strs``.
        """
        super().__init__()
        self.data_root = os.path.expanduser(dataset_dir)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.split = split
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.image_dir = image_dir
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self._uses_ordered_references = _supports_ordered_references(preprocess_func)
        self._source_hash_override = _validate_source_hash_override(source_hash_override)
        self._passthrough_columns = _normalize_passthrough_columns(passthrough_columns)

        if self.shard_index is not None and self.shard_index > 0:
            disable_progress_bar()

        if raw_dataset is None:
            loaded_raw_dataset = self._load_raw_dataset()
        else:
            if not isinstance(raw_dataset, HFDataset):
                raise TypeError(
                    "raw_dataset must be a datasets.Dataset, " f"got {type(raw_dataset).__name__}"
                )
            if self._source_hash_override is None:
                raise ValueError("source_hash_override is required when raw_dataset is provided")
            loaded_raw_dataset = raw_dataset

        missing_passthrough_columns = set(self._passthrough_columns) - set(
            loaded_raw_dataset.column_names
        )
        if missing_passthrough_columns:
            raise ValueError(
                "passthrough columns are missing from the raw dataset: "
                f"{sorted(missing_passthrough_columns)!r}"
            )

        if max_dataset_size is not None and len(loaded_raw_dataset) > max_dataset_size:
            loaded_raw_dataset = loaded_raw_dataset.select(range(max_dataset_size))
            logger.info(f"Dataset size limited to {max_dataset_size} samples.")

        self._ordered_reference_source_hash = ""
        if self._uses_ordered_references:
            if "references" not in loaded_raw_dataset.column_names:
                raise ValueError(
                    "ordered-reference preprocessing requires a references column, "
                    f"got columns={loaded_raw_dataset.column_names!r} in {self.data_root!r}"
                )
            canonical_manifests = [
                _canonicalize_ordered_reference_value(references, row_index=row_index)
                for row_index, references in enumerate(loaded_raw_dataset["references"])
            ]
            self._ordered_reference_source_hash = hashlib.sha256(
                "\n".join(canonical_manifests).encode("utf-8")
            ).hexdigest()
            if self._source_hash_override is None:
                extra_hash_strs = list(extra_hash_strs or []) + [
                    self._ordered_reference_source_hash
                ]

        if enable_preprocess:
            self.processed_dataset = self._preprocess_dataset(
                raw_dataset=loaded_raw_dataset,
                preprocess_func=preprocess_func,
                preprocess_kwargs=preprocess_kwargs or {},
                preprocessing_batch_size=preprocessing_batch_size,
                force_reprocess=force_reprocess,
                max_dataset_size=max_dataset_size,
                extra_hash_strs=extra_hash_strs,
                target_arrow_path=target_arrow_path,
                source_hash_override=self._source_hash_override,
            )
        else:
            self.processed_dataset = loaded_raw_dataset
            self.merged_cache_path = None

    def _load_raw_dataset(self) -> HFDataset:
        """Load raw dataset from JSONL or TXT file."""
        jsonl_path = os.path.join(self.data_root, f"{self.split}.jsonl")
        txt_path = os.path.join(self.data_root, f"{self.split}.txt")

        if os.path.exists(jsonl_path):
            raw_dataset = load_dataset("json", data_files=jsonl_path, split="train")
            self.image_dir = (
                os.path.join(self.data_root, "images") if self.image_dir is None else self.image_dir
            )
            self.video_dir = (
                os.path.join(self.data_root, "videos") if self.video_dir is None else self.video_dir
            )
            self.audio_dir = (
                os.path.join(self.data_root, "audios") if self.audio_dir is None else self.audio_dir
            )
        elif os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            raw_dataset = HFDataset.from_dict({"prompt": prompts})
            self.image_dir = None if self.image_dir is None else self.image_dir
            self.video_dir = None if self.video_dir is None else self.video_dir
            self.audio_dir = None if self.audio_dir is None else self.audio_dir
            logger.info(f"Loaded {len(prompts)} prompts from {txt_path}")
        else:
            raise FileNotFoundError(f"Could not find {jsonl_path} or {txt_path}")

        return raw_dataset

    def _preprocess_dataset(
        self,
        raw_dataset: HFDataset,
        preprocess_func: PreprocessCallable,
        preprocess_kwargs: Dict[str, Any],
        preprocessing_batch_size: int,
        force_reprocess: bool,
        max_dataset_size: Optional[int],
        extra_hash_strs: Optional[List[str]] = None,
        target_arrow_path: Optional[str] = None,
        source_hash_override: Optional[str] = None,
    ) -> HFDataset:
        """Apply preprocessing to raw dataset with caching.

        Args:
            target_arrow_path: If set, ``map()`` writes its Arrow output directly
                to this file via ``cache_file_name=`` (and reads it back on a
                cache hit). When ``None``, HF derives a path under its own
                ``~/.cache/huggingface/datasets`` cache (legacy behavior).

        Returns:
            Preprocessed HuggingFace Dataset.
        """
        self._preprocess_func = preprocess_func
        self._preprocess_kwargs = preprocess_kwargs

        self.merged_cache_path = self.compute_cache_path(
            dataset_dir=self.data_root,
            split=self.split,
            cache_dir=self.cache_dir,
            max_dataset_size=max_dataset_size,
            preprocess_func=preprocess_func,
            preprocess_kwargs=preprocess_kwargs,
            extra_hash_strs=extra_hash_strs,
            source_hash_override=source_hash_override,
        )

        # Every distributed rank must write the same Arrow schema. Infer it from
        # the unsharded source before selecting the rank-local rows; otherwise an
        # all-empty optional-media shard can become ``List(null)`` while another
        # rank writes ``List(Image)`` and the merged cache cannot be loaded.
        preprocess_features = self._infer_cross_chunk_features(
            raw_dataset=raw_dataset,
            preprocessing_batch_size=preprocessing_batch_size,
            image_dir=self.image_dir,
            video_dir=self.video_dir,
            audio_dir=self.audio_dir,
            force_reprocess=force_reprocess,
            target_arrow_path=target_arrow_path,
            require_explicit_features=bool(self.num_shards and self.num_shards > 1),
        )

        if self.num_shards and self.num_shards > 1:
            if self.shard_index is None:
                raise ValueError(
                    f"shard_index must be set when num_shards > 1, "
                    f"got num_shards={self.num_shards}, shard_index=None"
                )
            raw_dataset = self._shard_dataset(raw_dataset, self.shard_index, self.num_shards)
            shard_fingerprint = (
                f"{os.path.basename(self.merged_cache_path)}"
                f"{self._shard_suffix(self.shard_index, self.num_shards)}"
            )
            # Display convention matches :meth:`_shard_suffix`: the second
            # number is the last shard index (``num_shards - 1``), not the total.
            desc = (
                f"[Preprocessing {self.split} dataset] "
                f"Shard {self.shard_index:04d}/{self.num_shards - 1:04d}"
            )
        else:
            shard_fingerprint = os.path.basename(self.merged_cache_path)
            desc = f"[Preprocessing {self.split} dataset]"

        os.makedirs(self.cache_dir, exist_ok=True)
        if target_arrow_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(target_arrow_path)), exist_ok=True)

        processed_dataset = raw_dataset.map(
            self._preprocess_batch,
            batched=True,
            with_indices=True,
            batch_size=preprocessing_batch_size,
            fn_kwargs={
                "image_dir": self.image_dir,
                "video_dir": self.video_dir,
                "audio_dir": self.audio_dir,
            },
            remove_columns=raw_dataset.column_names,
            new_fingerprint=shard_fingerprint,
            cache_file_name=target_arrow_path,
            features=preprocess_features,
            desc=desc,
            load_from_cache_file=not force_reprocess,
        )

        _apply_torch_format(processed_dataset)

        return processed_dataset

    def _infer_cross_chunk_features(
        self,
        *,
        raw_dataset: HFDataset,
        preprocessing_batch_size: int,
        image_dir: Optional[str],
        video_dir: Optional[str],
        audio_dir: Optional[str],
        force_reprocess: bool,
        target_arrow_path: Optional[str],
        require_explicit_features: bool = False,
    ) -> Optional[HFFeatures]:
        """Infer one explicit schema when later map chunks introduce typed values.

        HuggingFace infers a batched map's writer schema from its first output
        chunk. Optional columns therefore need a representative schema probe when
        that chunk is empty but a later chunk is populated. This narrow probe may
        call a preprocessor before the real ordered map, so preprocessors must keep
        their existing cache-oriented determinism contract. Global Python, NumPy,
        torch, MPS, and explicit ``torch.Generator`` states are restored; arbitrary
        adapter-owned mutable state is intentionally outside that guarantee.

        Args:
            raw_dataset: Full input dataset before distributed sharding.
            preprocessing_batch_size: Map chunk size.
            image_dir: Image root forwarded to preprocessing.
            video_dir: Video root forwarded to preprocessing.
            audio_dir: Audio root forwarded to preprocessing.
            force_reprocess: Whether an existing explicit Arrow target is rebuilt.
            target_arrow_path: Optional explicit Arrow cache target.
            require_explicit_features: Whether to infer a schema even when the
                first source chunk is already representative. Distributed ranks
                use this to share one global schema across disjoint shards.

        Returns:
            Explicit output features for a cross-chunk nullable transition, or
            ``None`` when first-chunk inference is already sufficient.
        """
        if (
            not force_reprocess
            and target_arrow_path is not None
            and os.path.isfile(target_arrow_path)
        ):
            return None

        probe_batches = _cross_chunk_schema_probe_batches(
            raw_dataset,
            preprocessing_batch_size=preprocessing_batch_size,
        )
        if probe_batches is None:
            if not require_explicit_features or not len(raw_dataset):
                return None
            probe_batches = [list(range(min(preprocessing_batch_size, len(raw_dataset))))]

        explicit_generators = list(_iter_torch_generators(self._preprocess_kwargs))
        generator_states = [generator.get_state() for generator in explicit_generators]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        mps_state = None
        if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
            mps_state = torch.mps.get_rng_state()

        try:
            with torch.random.fork_rng():
                probe_results = [
                    (
                        indices,
                        self._preprocess_batch(
                            raw_dataset[indices],
                            indices,
                            image_dir,
                            video_dir,
                            audio_dir,
                        ),
                    )
                    for indices in probe_batches
                ]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
            for generator, state in zip(explicit_generators, generator_states):
                generator.set_state(state)

        expected_columns = tuple(probe_results[0][1])
        combined_result = {column_name: [] for column_name in expected_columns}
        for indices, probe_result in probe_results:
            if set(probe_result) != set(expected_columns):
                raise ValueError(
                    "preprocess output columns must remain stable across map chunks: "
                    f"expected {expected_columns!r}, got {tuple(probe_result)!r} "
                    f"for source rows {indices!r}"
                )
            for column_name in expected_columns:
                values = probe_result[column_name]
                if not isinstance(values, (list, tuple, np.ndarray)):
                    raise TypeError(
                        "batched preprocess output columns must be sequences, "
                        f"got {type(values).__name__} for {column_name!r} "
                        f"at source rows {indices!r}"
                    )
                combined_result[column_name].extend(values)

        return HFDataset.from_dict(combined_result).features

    def _shard_dataset(self, dataset: HFDataset, shard_index: int, num_shards: int) -> HFDataset:
        """
        Split dataset into shards for distributed preprocessing.

        Args:
            dataset: Full dataset to shard
            shard_index: Index of current shard (0 to num_shards-1)
            num_shards: Total number of shards

        Returns:
            Sharded subset of the dataset
        """
        shard_size = len(dataset) // num_shards
        start_idx = shard_index * shard_size
        end_idx = start_idx + shard_size if shard_index < num_shards - 1 else len(dataset)
        return dataset.select(range(start_idx, end_idx))

    def _preprocess_batch(
        self,
        batch: Dict[str, Any],
        indices: List[int],
        image_dir: Optional[str],
        video_dir: Optional[str],
        audio_dir: Optional[str],
    ) -> Dict[str, Any]:
        """
        Preprocess a batch of samples.

        Workflow:
            1. Prepare prompt inputs (text)
            2. Load and prepare image inputs
            3. Load and prepare video inputs
            4. Load and prepare audio inputs
            5. Load ordered references and forward their canonical manifest sidecars
            6. Call preprocess function
            7. Move result tensors to CPU for caching
            8. Pack non-preprocessed columns into ``metadata``

        Args:
            batch: Dictionary with batch data.
            indices: Source row indices aligned with batch rows, used in validation diagnostics.
            image_dir: Directory containing images (``None`` skips image loading).
                Per-sample paths are loaded as PIL Images and kept as a
                ``List[Image]``; the column-level ``images`` field is therefore
                always a ``MultiImageBatch`` of shape ``List[List[Image]]`` —
                single-image samples produce ``[Image]`` and empty samples
                produce ``[]``.
            video_dir: Directory containing videos (``None`` skips video loading).
                Same shape as ``image_dir``: column-level ``videos`` is a
                ``MultiVideoBatch`` (``List[List[VideoFrames]]``).
            audio_dir: Directory containing audio files (``None`` skips audio
                loading). Each per-sample list of paths is loaded via
                :func:`flow_factory.utils.audio.load_audio` and stored as a
                ``List[torch.Tensor]`` (one Tensor per audio clip), so the
                column-level ``audios`` field is always a ``MultiAudioBatch``
                of shape ``List[List[Tensor]]`` — single-audio samples produce
                ``[Tensor]`` and empty samples produce ``[]``.

        Returns:
            Dictionary with preprocessed data, plus an additional ``metadata``
            list carrying every non-preprocess column from ``batch``.

        Note:
            The ``[]``-for-empty contract is what keeps every column length
            equal to the input batch size, which HF ``Dataset.map(batched=True)``
            requires. Mixing in ``None`` or unwrapping single-element lists to a
            bare ``Tensor`` breaks Arrow's homogeneous-column requirement and
            forces every downstream consumer to handle three input shapes.
        """
        assert self._preprocess_func is not None, "Preprocess function must be provided."
        if self._uses_ordered_references and len(batch["prompt"]) != 1:
            raise ValueError(
                "ordered-reference preprocessing expected B=1, "
                f"received B={len(batch['prompt'])} for split={self.split!r}"
            )
        # The columns that are used in preprocess and maintained in the final results.
        PREPROCESS_COLUMNS = (
            "prompt",
            "negative_prompt",
            "images",
            "videos",
            "audios",
            "image_slots",
            "video_slots",
            "audio_slots",
        )
        metadata_excluded_columns = set(PREPROCESS_COLUMNS)
        metadata_excluded_columns.update(self._passthrough_columns)
        if self._uses_ordered_references:
            metadata_excluded_columns.update({"references", "reference_manifest"})

        # 1. Prepare prompt inputs (text)
        prompt = batch["prompt"]
        negative_prompt = batch.get("negative_prompt", None)
        prompt_args = {"prompt": prompt}
        if negative_prompt is not None:
            prompt_args["negative_prompt"] = negative_prompt

        # 2. Prepare image inputs (only when image_dir exists and batch has images)
        if "image" in batch:
            batch["images"] = batch.pop("image")  # Rename for consistency

        image_args = {"images": None}
        if image_dir is not None and "images" in batch:
            img_paths_list = batch["images"]
            batch["images"] = []  # Clear
            image_args["images"] = []
            for img_paths in img_paths_list:
                if not img_paths:
                    # Empty sample contributes [] to both args and batch so the
                    # column stays a homogeneous List[List[...]] (MultiImageBatch)
                    # and HF.map(batched=True) sees matching column lengths.
                    image_args["images"].append([])
                    batch["images"].append([])
                else:
                    if isinstance(img_paths, str):
                        img_paths = [img_paths]
                    images = [
                        _load_rgb_image(
                            _resolve_path(image_dir, img_path),
                            source="grouped input image",
                        )
                        for img_path in img_paths
                    ]
                    image_args["images"].append(images)
                    # Persist as PIL (not tensors) so HF stores this column via the
                    # Image feature. Ragged image tensors (variable size/count, e.g.
                    # multi-reference I2I) are not Arrow-serializable.
                    batch["images"].append(images)

        # 3. Prepare video inputs (only when video_dir exists and batch has videos)
        if "video" in batch:
            batch["videos"] = batch.pop("video")  # Rename for consistency

        video_args = {"videos": None}
        if video_dir is not None and "videos" in batch:
            video_paths_list = batch["videos"]
            batch["videos"] = []  # Clear
            video_args["videos"] = []
            for video_paths in video_paths_list:
                if not video_paths:
                    # Empty sample contributes [] to both args and batch so the
                    # column stays a homogeneous List[List[...]] (MultiVideoBatch)
                    # and HF.map(batched=True) sees matching column lengths.
                    video_args["videos"].append([])
                    batch["videos"].append([])
                else:
                    if isinstance(video_paths, str):
                        video_paths = [video_paths]

                    videos = [
                        _load_grouped_video(video_dir, video_spec) for video_spec in video_paths
                    ]
                    video_pts = [pil_image_to_tensor(video) for video in videos]
                    video_args["videos"].append(videos)
                    batch["videos"].append(video_pts)

        # 4. Prepare audio inputs (only when audio_dir exists and batch has audios)
        if "audio" in batch:
            batch["audios"] = batch.pop("audio")  # Rename for consistency

        audio_args = {"audios": None}
        if audio_dir is not None and "audios" in batch:
            audio_paths_list = batch["audios"]
            batch["audios"] = []  # Clear
            audio_args["audios"] = []
            for audio_paths in audio_paths_list:
                if not audio_paths:
                    # Empty sample contributes [] to both args and batch so the
                    # column stays a homogeneous List[List[Tensor]] (MultiAudioBatch)
                    # and HF.map(batched=True) sees matching column lengths.
                    audio_args["audios"].append([])
                    batch["audios"].append([])
                else:
                    if isinstance(audio_paths, str):
                        audio_paths = [audio_paths]
                    audios = [
                        _load_grouped_audio(audio_dir, audio_spec) for audio_spec in audio_paths
                    ]
                    # Always store as List[Tensor] (no single-audio unwrap) so
                    # downstream encode_audio sees a uniform type within the batch.
                    audio_args["audios"].append(audios)
                    batch["audios"].append(audios)

        # 5. Load ordered references and retain their canonical reconstruction sidecars.
        reference_args: Dict[str, Any] = {}
        if self._uses_ordered_references:
            raw_references = batch.pop("references")
            loaded_reference_batch = []
            canonical_manifests = []
            for row_offset, references in enumerate(raw_references):
                row_index = indices[row_offset]
                manifest = _canonicalize_ordered_reference_value(
                    references,
                    row_index=row_index,
                )
                canonical_manifests.append(manifest)
                loaded_reference_batch.append(
                    [
                        _load_ordered_reference(
                            entry,
                            self.data_root,
                            row_index=row_index,
                            reference_index=reference_index,
                        )
                        for reference_index, entry in enumerate(json.loads(manifest))
                    ]
                )
            reference_args.update(
                references=loaded_reference_batch,
                reference_manifest=canonical_manifests,
            )
            batch["reference_manifest"] = canonical_manifests

        slot_args = {
            column: batch[column]
            for column in ("image_slots", "video_slots", "audio_slots")
            if column in batch
        }

        # 6. Call preprocess function with filtered kwargs
        input_args = {
            **prompt_args,
            **image_args,
            **video_args,
            **audio_args,
            **reference_args,
            **slot_args,
            **self._preprocess_kwargs,
        }
        filtered_args = filter_kwargs(self._preprocess_func, **input_args)
        preprocess_res = self._preprocess_func(**filtered_args)
        passthrough_collisions = set(preprocess_res) & set(self._passthrough_columns)
        if passthrough_collisions:
            raise ValueError(
                "preprocess result collides with passthrough columns: "
                f"{sorted(passthrough_collisions)!r}"
            )

        # 7. Process results - move tensors to CPU for caching.
        # Image-valued adapter outputs (declared via `python_format_columns`)
        # are stored as per-sample List[PIL] so HF serializes them via the Image
        # feature; ragged image tensors (variable size/count, e.g. multi-ref I2I)
        # are not Arrow-serializable.
        adapter = getattr(self._preprocess_func, "__self__", None)
        adapter_python_format_cols = getattr(adapter, "python_format_columns", frozenset())
        final_res = {}
        for k, v in preprocess_res.items():
            if k in adapter_python_format_cols:
                # Image column: canonicalize each per-sample value
                # (Tensor(N,C,H,W) / List[Tensor] / List[PIL]) to List[PIL].
                # Empty samples stay []. The per-batch value is a per-sample list
                # by the column-homogeneity contract.
                if not isinstance(v, list):
                    raise TypeError(
                        f"image column {k!r} must be a per-sample list for HF "
                        f"Image serialization, got {type(v).__name__}"
                    )
                final_res[k] = [_to_pil_image_list(per_sample) for per_sample in v]
            elif isinstance(v, torch.Tensor):
                # Case A: Dense Batch Tensor
                # Move entire batch to CPU first (faster than moving slices), then unbind
                final_res[k] = list(torch.unbind(v.cpu(), dim=0))
            elif isinstance(v, list):
                # Case B: Ragged List (e.g. Flux image latents of varying sizes,
                # or nested lists like List[List[Tensor]] for multi-ref condition images)
                final_res[k] = [_move_to_cpu(x) for x in v]
            else:
                # Case C: Other types (None, int, etc)
                final_res[k] = v

        # 8. Prepare final results
        batch_dict = {**batch, **final_res}
        if self._uses_ordered_references:
            _validate_arrow_safe_ordered_result(batch_dict, len(batch["prompt"]))
        # Pack non-preprocess fields into the METADATA_COLUMN (dict[list] -> list[dict]).
        # At sample time, BaseTrainer._inject_batch_metadata stores each per-sample
        # dict as a single JSON string under `sample.extra_kwargs['metadata']`.
        # Reward models that need metadata fields call `json.loads(sample.metadata)`
        # to access them (see GenEvalRewardModel for a full example).
        # Complex values (nested lists/dicts) in the source JSONL must already be
        # stored as JSON strings for Arrow compatibility.
        batch_dict[METADATA_COLUMN] = [
            {k: v[idx] for k, v in batch.items() if k not in metadata_excluded_columns}
            for idx in range(len(batch["prompt"]))
        ]

        return batch_dict

    @classmethod
    def load_merged(cls, merged_cache_path: str) -> "GeneralDataset":
        """
        Load preprocessed dataset from merged cache.

        Args:
            merged_cache_path: Path to merged cache directory

        Returns:
            GeneralDataset instance with loaded data
        """
        instance = cls.__new__(cls)
        loaded = load_from_disk(merged_cache_path)
        if not isinstance(loaded, HFDataset):
            raise TypeError(
                f"expected a Dataset at {merged_cache_path!r}, got {type(loaded).__name__}"
            )
        instance.processed_dataset = loaded
        _apply_torch_format(instance.processed_dataset)
        return instance

    @staticmethod
    def compute_cache_path(
        dataset_dir: str,
        split: str,
        cache_dir: str,
        max_dataset_size: Optional[int],
        preprocess_func: Optional[Callable],
        preprocess_kwargs: Optional[Dict[str, Any]],
        extra_hash_strs: Optional[List[str]] = None,
        source_hash_override: Optional[str] = None,
        digits: int = 32,
    ) -> str:
        """Compute merged cache path by hashing all components.

        ``kwargs_hash`` is computed via *deep signature collection*: the set
        of relevant keys is the union of named parameters from
        ``preprocess_func`` and (when ``preprocess_func`` accepts ``**kwargs``
        and is a bound adapter method) all ``encode_*`` methods on the same
        adapter instance.  Keys outside this union (e.g.
        ``num_batches_per_epoch``, ``gradient_accumulation_steps``) are
        excluded — they are training-infrastructure fields that do not affect
        preprocessing output.

        To force a value into the cache key without adding it to any function
        signature, include it in ``extra_hash_strs``.

        Args:
            digits: Length of hash fingerprint (default: 32, max: 32)

        Returns:
            Cache path with fingerprint of specified length
        """
        dataset_root = os.path.abspath(os.path.expanduser(dataset_dir))
        dataset_name = os.path.basename(dataset_root)
        validated_source_hash_override = _validate_source_hash_override(source_hash_override)
        if validated_source_hash_override is not None:
            source_hash = validated_source_hash_override
        else:
            source_candidates = (
                os.path.join(dataset_root, f"{split}.jsonl"),
                os.path.join(dataset_root, f"{split}.txt"),
            )
            source_path = next(
                (path for path in source_candidates if os.path.isfile(path)),
                None,
            )
            if source_path is None:
                source_hash = "missing"
            else:
                hasher = hashlib.sha256()
                with open(source_path, "rb") as source_file:
                    for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                        hasher.update(chunk)
                source_hash = hasher.hexdigest()
        cutoff_str = str(max_dataset_size) if max_dataset_size else "full"
        funcs_hash = _compute_encode_funcs_hash(preprocess_func, digits=16)
        hashable_kwargs = _select_cache_relevant_kwargs(preprocess_func, preprocess_kwargs)
        owner = getattr(preprocess_func, "__self__", None)
        cache_fields = getattr(owner, "preprocess_cache_fields", frozenset())
        for field_name in cache_fields:
            if preprocess_kwargs is not None and field_name in preprocess_kwargs:
                hashable_kwargs[field_name] = preprocess_kwargs[field_name]
        cache_version = getattr(owner, "preprocess_cache_version", "")
        kwargs_hash = hashlib.md5(str(sorted(hashable_kwargs.items())).encode()).hexdigest()[:16]
        extra_hash = "|".join(extra_hash_strs) if extra_hash_strs else ""

        combined = (
            f"{dataset_root}|{dataset_name}|{source_hash}|{split}|{cutoff_str}|"
            f"{funcs_hash}|{kwargs_hash}|cachev={cache_version}"
            f"|{extra_hash}|fmtv{_PREPROCESS_FORMAT_VERSION}"
        )
        fingerprint = hashlib.md5(combined.encode()).hexdigest()[: min(digits, 32)]

        logger.debug(
            "compute_cache_path: dataset=%s split=%s cutoff=%s funcs=%s kwargs=%s "
            "extra=%s hashable_keys=%s -> %s",
            dataset_name,
            split,
            cutoff_str,
            funcs_hash,
            kwargs_hash,
            extra_hash,
            sorted(hashable_kwargs),
            fingerprint,
        )
        return os.path.join(os.path.expanduser(cache_dir), fingerprint)

    @staticmethod
    def _shard_suffix(shard_idx: int, num_shards: int) -> str:
        """Per-rank suffix ``_shard{X:04d}of{Y:04d}`` where ``Y = num_shards - 1``.

        IMPORTANT: ``Y`` is the *last* shard index (inclusive), **not** the
        total shard count. The rank range covered is ``[0, Y]``, i.e.
        ``num_shards`` ranks in total. Example::

            _shard_suffix(shard_idx=0, num_shards=4) -> "_shard0000of0003"
            _shard_suffix(shard_idx=3, num_shards=4) -> "_shard0003of0003"

        Shared by:

          * the Arrow filename embedded in :meth:`build_part_arrow_path`
          * the HF ``Dataset.map(new_fingerprint=...)`` string in
            :meth:`_preprocess_dataset`

        Changing the format here keeps both in lockstep; no caller of this class
        may hand-craft the suffix.
        """
        return f"_shard{shard_idx:04d}of{num_shards - 1:04d}"

    @staticmethod
    def build_part_arrow_path(merged_cache_path: str, shard_idx: int, num_shards: int) -> str:
        """Deterministic per-rank Arrow file path inside ``{merged_cache_path}.tmp``.

        Single source of truth for the per-rank cache file layout. Called by:

          * the writer (each rank's ``Dataset.map(cache_file_name=...)`` target
            in :func:`flow_factory.data_utils.loader._create_or_load_dataset`)
          * :meth:`consolidate_parts` (reconstructs every rank's path to build
            the merged dataset's ``state.json``)

        Layout::

            {merged_cache_path}.tmp/_parts/rank_{X:04d}_of_{N:04d}/
                cache-{basename}{_shard_suffix(X, N)}.arrow

        Args:
            merged_cache_path: Final merged-cache directory (without the
                ``.tmp`` suffix). A leading ``~`` is expanded internally; no
                other normalization is applied, so the return value is
                absolute iff ``merged_cache_path`` is absolute after
                ``expanduser``. ``{merged_cache_path}.tmp`` is the build dir.
            shard_idx: Shard index (``0 <= shard_idx < num_shards``).
            num_shards: Total number of shards participating in preprocessing.

        Returns:
            Path to this rank's Arrow file inside the build directory.
        """
        merged_cache_path = os.path.expanduser(merged_cache_path)
        build_dir = merged_cache_path + ".tmp"
        merged_fp = os.path.basename(merged_cache_path)
        return os.path.join(
            build_dir,
            "_parts",
            f"rank_{shard_idx:04d}_of_{num_shards:04d}",
            f"cache-{merged_fp}{GeneralDataset._shard_suffix(shard_idx, num_shards)}.arrow",
        )

    @classmethod
    def consolidate_parts(
        cls,
        merged_cache_path: str,
        num_shards: int,
        split: Optional[str] = None,
    ) -> None:
        """Promote per-rank Arrow files into a valid HF dataset directory without copying data.

        Builds the top-level ``state.json`` and ``dataset_info.json`` that turn the
        directory ``merged_cache_path + ".tmp"`` (which already contains every rank's
        Arrow file under ``_parts/rank_*/``) into a structure ``load_from_disk`` can
        read, then atomically renames ``.tmp`` -> ``merged_cache_path``. No row data
        is re-serialized: each shard's bytes stay where ``Dataset.map(cache_file_name=...)``
        wrote them.

        Paths of the ``num_shards`` per-rank Arrow files are derived via
        :meth:`build_part_arrow_path`, so the writer and the consolidator cannot
        drift.

        Args:
            merged_cache_path: Final destination directory. The function reads from
                ``merged_cache_path + ".tmp"`` and renames it to this path on success.
                A leading ``~`` is expanded internally to keep ``build_dir`` and
                :meth:`build_part_arrow_path` outputs on the same form (otherwise
                ``os.path.relpath`` would cross forms and produce bogus prefixes).
            num_shards: Total number of per-rank Arrow files expected under the
                build directory, in rank order. Listed in the produced
                ``state.json`` as ``_data_files`` (relative to ``merged_cache_path``);
                ``load_from_disk`` will memory-map them in this order.
            split: Optional split tag stored as ``state["_split"]`` (round-trips to
                ``dataset.split`` after ``load_from_disk``).

        Raises:
            FileNotFoundError: If the build directory or any expected per-rank Arrow
                file is missing. The message includes ``merged_cache_path`` and
                ``num_shards`` to make distributed debugging tractable.
        """
        merged_cache_path = os.path.expanduser(merged_cache_path)
        build_dir = merged_cache_path + ".tmp"
        if not os.path.isdir(build_dir):
            raise FileNotFoundError(
                f"expected build dir for consolidation, missing: {build_dir} "
                f"(merged_cache_path={merged_cache_path})"
            )
        part_arrow_paths = [
            cls.build_part_arrow_path(merged_cache_path, i, num_shards) for i in range(num_shards)
        ]
        for p in part_arrow_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"expected per-rank arrow file, missing: {p} "
                    f"(merged_cache_path={merged_cache_path}, "
                    f"num_shards={num_shards})"
                )

        template = HFDataset.from_file(part_arrow_paths[0])
        state = {
            "_data_files": [{"filename": os.path.relpath(p, build_dir)} for p in part_arrow_paths],
            "_fingerprint": os.path.basename(merged_cache_path),
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": split,
        }
        with open(os.path.join(build_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        dataset_info = asdict(template.info)
        with open(os.path.join(build_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump({k: dataset_info[k] for k in sorted(dataset_info)}, f, indent=2)
        if os.path.exists(merged_cache_path):
            shutil.rmtree(merged_cache_path)
        os.replace(build_dir, merged_cache_path)

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.

        Stacks tensors with same shape, keeps ragged tensors as lists. Image
        columns stored via the HF Image feature -- the raw ``images`` column always,
        plus any adapter output declared in ``python_format_columns`` (e.g. Bagel's
        ``condition_images``) -- decode to per-sample ``List[PIL.Image]``, so they
        land in Case 3 and are kept as a ``List[List[PIL.Image]]`` MultiImageBatch.
        ``condition_images`` not declared in ``python_format_columns`` stay tensors.

        Args:
            batch: List of samples

        Returns:
            Collated batch dictionary
        """
        if not batch:
            return {}

        collated_batch = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            # Classify value types
            is_tensor = [isinstance(v, torch.Tensor) for v in values]
            is_list = [isinstance(v, list) for v in values]

            if all(is_tensor):
                # Case 1: All elements are tensors
                shapes = [v.shape for v in values]
                if all(s == shapes[0] for s in shapes):
                    # Same shape → stack into batch tensor
                    collated_batch[key] = torch.stack(values, dim=0)
                else:
                    # Different shapes → keep as List[Tensor]
                    collated_batch[key] = values

            elif any(is_tensor) and any(is_list):
                # Case 2: Mixed tensor/list → normalize to List[List[Tensor]].
                # Handles ragged tensor columns (e.g. image latents) where the
                # dataset auto-stacks same-shape samples into a Tensor but keeps
                # variable-shape samples as List[Tensor]; unbind the stacked ones
                # so the whole column is a consistent List[List[Tensor]].
                collated_batch[key] = [
                    list(torch.unbind(v, dim=0)) if isinstance(v, torch.Tensor) else v
                    for v in values
                ]

            else:
                # Case 3: Other types (lists, ints, strs, and python-format
                # columns: metadata dicts and PIL image columns).
                # Image columns arrive here as per-sample List[PIL.Image]; keeping
                # `values` yields a List[List[PIL.Image]] MultiImageBatch.
                collated_batch[key] = values

        return collated_batch


# ========================================================================================
# Utility Functions
# ========================================================================================


def _validate_source_hash_override(value: Optional[str]) -> Optional[str]:
    """Validate an optional caller-owned source-content identity."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            "source_hash_override must be a non-empty string or None, "
            f"got {type(value).__name__}"
        )
    if not value.strip():
        raise ValueError("source_hash_override must be a non-empty string")
    return value


def _normalize_passthrough_columns(columns: Optional[Sequence[str]]) -> tuple[str, ...]:
    """Return validated passthrough column names in caller-declared order."""
    if columns is None:
        return ()
    if isinstance(columns, (str, bytes)):
        raise TypeError("passthrough_columns must be a sequence of column names, not a string")
    normalized = tuple(columns)
    for column in normalized:
        if not isinstance(column, str) or not column:
            raise ValueError(
                "passthrough column names must be non-empty strings, " f"got {column!r}"
            )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"passthrough_columns contains duplicates: {normalized!r}")
    if METADATA_COLUMN in normalized:
        raise ValueError(
            f"{METADATA_COLUMN!r} is owned by GeneralDataset and cannot be a " "passthrough column"
        )
    return normalized


def _cross_chunk_schema_probe_batches(
    dataset: HFDataset,
    *,
    preprocessing_batch_size: int,
) -> Optional[List[List[int]]]:
    """Return real map chunks needed to cover later typed source values."""
    if preprocessing_batch_size <= 0 or len(dataset) <= preprocessing_batch_size:
        return None

    first_chunk_stop = min(preprocessing_batch_size, len(dataset))
    first_chunk = dataset[:first_chunk_stop]
    pending_columns = {
        column_name
        for column_name in dataset.column_names
        if not any(_has_typed_payload(value) for value in first_chunk[column_name])
    }
    if not pending_columns:
        return None

    probe_chunk_starts = {0}
    for start in range(first_chunk_stop, len(dataset), preprocessing_batch_size):
        stop = min(start + preprocessing_batch_size, len(dataset))
        pending_chunk = dataset.select_columns(sorted(pending_columns))[start:stop]
        typed_columns = {
            column_name
            for column_name in pending_columns
            if any(_has_typed_payload(value) for value in pending_chunk[column_name])
        }
        if typed_columns:
            probe_chunk_starts.add(start)
            pending_columns.difference_update(typed_columns)
            if not pending_columns:
                break

    if len(probe_chunk_starts) == 1:
        return None
    return [
        list(range(start, min(start + preprocessing_batch_size, len(dataset))))
        for start in sorted(probe_chunk_starts)
    ]


def _has_typed_payload(value: Any) -> bool:
    """Return whether a value can contribute a non-null Arrow child type."""
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, Mapping):
        return any(_has_typed_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_typed_payload(item) for item in value)
    return True


def _iter_torch_generators(
    value: Any,
    _seen: Optional[set[int]] = None,
) -> Iterator[torch.Generator]:
    """Yield distinct explicit torch generators nested in preprocessing kwargs."""
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return
    _seen.add(value_id)

    if isinstance(value, torch.Generator):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_torch_generators(item, _seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_torch_generators(item, _seen)


def _supports_ordered_references(preprocess_func: Optional[Callable]) -> bool:
    """Return whether a bound preprocessor explicitly opts into ordered references."""
    if preprocess_func is None:
        return False
    owner = getattr(preprocess_func, "__self__", None)
    return bool(
        getattr(preprocess_func, "supports_ordered_references", False)
        or getattr(owner, "supports_ordered_references", False)
    )


def _canonicalize_ordered_reference_value(value: Any, row_index: int) -> str:
    """Canonicalize either an ordered reference list or an opaque Arrow string."""
    if isinstance(value, str):
        references = parse_reference_manifest(value, row_index=row_index)
    else:
        references = value
    return canonicalize_reference_manifest(references, row_index=row_index)


def _load_rgb_image(path: str, *, source: str) -> Image.Image:
    """Decode one image into the shared positive-size RGB PIL boundary."""
    with Image.open(path) as image:
        decoded = image.convert("RGB")
    return require_decoded_rgb_image(decoded, source=source)


def _load_grouped_video(base_dir: str, spec: Any) -> List[Image.Image]:
    """Decode one grouped video path, honoring an optional FPS override."""
    path, fps = _parse_grouped_media_spec(
        spec,
        media_type="video",
        rate_name="fps",
    )
    return load_video_frames(_resolve_path(base_dir, path), fps=fps)


def _load_grouped_audio(base_dir: str, spec: Any) -> torch.Tensor:
    """Decode one grouped audio path, honoring an optional sample-rate override."""
    path, sample_rate = _parse_grouped_media_spec(
        spec,
        media_type="audio",
        rate_name="sample_rate",
    )
    if sample_rate is not None and not isinstance(sample_rate, int):
        raise TypeError(
            "grouped audio entry requires an integer sample_rate, " f"got {sample_rate!r}"
        )
    return require_decoded_audio_waveform(
        load_audio(_resolve_path(base_dir, path), sample_rate=sample_rate),
        source="grouped input audio",
    )


def _parse_grouped_media_spec(
    spec: Any,
    *,
    media_type: str,
    rate_name: str,
) -> tuple[str, Optional[Union[int, float]]]:
    """Normalize a legacy path string or a V2 projected path/rate mapping."""
    if isinstance(spec, str):
        return spec, None
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"expected grouped {media_type} entry to be a path string or mapping, "
            f"got {type(spec).__name__}: {spec!r}"
        )
    unknown_keys = set(spec) - {"path", rate_name}
    if unknown_keys:
        raise ValueError(f"grouped {media_type} entry has unknown keys: {sorted(unknown_keys)!r}")
    path = spec.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"grouped {media_type} entry requires a non-empty path string, got {path!r}"
        )
    rate = spec.get(rate_name)
    if rate is not None and (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise ValueError(
            f"grouped {media_type} entry requires finite positive {rate_name}, got {rate!r}"
        )
    return path, rate


def _load_ordered_reference(
    entry: Dict[str, Any],
    data_root: str,
    row_index: int,
    reference_index: int,
) -> Dict[str, Any]:
    """Decode one ordered reference with dataset row/reference context."""
    reference_type = entry["type"]
    resolved_path = _resolve_path(data_root, entry["path"])
    failing_path = resolved_path
    loaded = dict(entry)
    try:
        if reference_type == "image":
            loaded["media"] = _load_rgb_image(
                resolved_path,
                source="ordered input image",
            )
        elif reference_type == "video":
            frames, fps, audio, sample_rate = _decode_ordered_video(resolved_path)
            effective_fps = entry.get("fps", fps)
            _require_finite_positive_rate(
                effective_fps,
                "effective fps",
                row_index,
                reference_index,
                reference_type,
                resolved_path,
            )
            loaded["frames"] = frames
            loaded["fps"] = effective_fps
            if "audio_path" in entry:
                audio_path = _resolve_path(data_root, entry["audio_path"])
                failing_path = audio_path
                audio, sample_rate = _decode_ordered_audio(audio_path)
            if audio is not None:
                _require_finite_positive_rate(
                    sample_rate,
                    "sample_rate",
                    row_index,
                    reference_index,
                    reference_type,
                    failing_path,
                )
                effective_sample_rate = entry.get("sample_rate", sample_rate)
                _require_finite_positive_rate(
                    effective_sample_rate,
                    "effective sample_rate",
                    row_index,
                    reference_index,
                    reference_type,
                    failing_path,
                )
                loaded["audio"] = audio
                loaded["sample_rate"] = effective_sample_rate
        elif reference_type == "audio":
            audio, sample_rate = _decode_ordered_audio(resolved_path)
            _require_finite_positive_rate(
                sample_rate,
                "sample_rate",
                row_index,
                reference_index,
                reference_type,
                resolved_path,
            )
            effective_sample_rate = entry.get("sample_rate", sample_rate)
            _require_finite_positive_rate(
                effective_sample_rate,
                "effective sample_rate",
                row_index,
                reference_index,
                reference_type,
                resolved_path,
            )
            loaded["media"] = audio
            loaded["sample_rate"] = effective_sample_rate
        else:
            raise ValueError(
                "expected ordered reference type in ('image', 'video', 'audio'), "
                f"got {reference_type!r}"
            )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"failed to decode ordered reference at row {row_index}, "
            f"reference {reference_index}, type={reference_type!r}, "
            f"path={failing_path!r}: {error}"
        ) from error
    return loaded


def _require_finite_positive_rate(
    value: Any,
    rate_name: str,
    row_index: int,
    reference_index: int,
    reference_type: str,
    media_path: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"at row {row_index}, reference {reference_index}, type={reference_type!r}, "
            f"path={media_path!r}, expected decoded {rate_name} to be finite positive, "
            f"got {value!r}"
        )


def _require_pyav() -> Any:
    if av is None:
        raise ImportError(
            "ordered video/audio references require PyAV>=17.0.0; "
            "install with `pip install 'av>=17.0.0'`"
        )
    return av


def _decode_ordered_video(
    video_path: str,
) -> tuple[np.ndarray, Any, Optional[torch.Tensor], Optional[int]]:
    """Decode frames, FPS, and optional soundtrack without resampling."""
    av_module = _require_pyav()
    with av_module.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"expected a video stream in {video_path!r}, got none")
        video_stream = container.streams.video[0]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video_stream)]
        reported_frame_rate = video_stream.average_rate or video_stream.guessed_rate
        frame_rate = None if reported_frame_rate is None else float(reported_frame_rate)
        audio = None
        sample_rate = None
        if container.streams.audio:
            container.seek(0)
            audio, sample_rate = _decode_av_audio_stream(container, container.streams.audio[0])
    if not frames:
        raise ValueError(f"expected video frames in {video_path!r}, decoded none")
    decoded_frames = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)
    return (
        require_decoded_video_frames(decoded_frames, source="ordered input video"),
        frame_rate,
        audio,
        sample_rate,
    )


def _decode_ordered_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """Decode an audio file and preserve its source sample rate."""
    av_module = _require_pyav()
    with av_module.open(audio_path) as container:
        if not container.streams.audio:
            raise ValueError(f"expected an audio stream in {audio_path!r}, got none")
        return _decode_av_audio_stream(container, container.streams.audio[0])


def _decode_av_audio_stream(container: Any, stream: Any) -> tuple[torch.Tensor, int]:
    """Decode one PyAV audio stream without changing its sample rate."""
    av_module = _require_pyav()
    reported_sample_rate = stream.codec_context.sample_rate
    if reported_sample_rate is None:
        raise ValueError("expected decoded audio sample_rate, got None")
    sample_rate = int(reported_sample_rate)
    resampler = av_module.audio.resampler.AudioResampler(
        format="fltp", layout=stream.layout, rate=sample_rate
    )
    chunks = []
    for frame in container.decode(stream):
        chunks.extend(
            torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(frame)
        )
    chunks.extend(
        torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(None)
    )
    if not chunks:
        raise ValueError("expected decoded audio samples, got none")
    waveform = torch.cat(chunks, dim=-1).to(torch.float32).contiguous()
    return (
        require_decoded_audio_waveform(waveform, source="ordered input audio"),
        sample_rate,
    )


def _validate_arrow_safe_ordered_result(
    batch_dict: Dict[str, Any],
    expected_batch_size: int,
) -> None:
    """Reject transient media and malformed columns before Arrow serialization."""
    for column_name, values in batch_dict.items():
        if not isinstance(values, list) or len(values) != expected_batch_size:
            received_length = len(values) if isinstance(values, list) else None
            raise ValueError(
                f"expected ordered-reference column {column_name!r} to have outer "
                f"B={expected_batch_size}, got type={type(values).__name__}, "
                f"length={received_length}"
            )
        _validate_arrow_safe_value(values, column_name)


def _validate_arrow_safe_value(value: Any, field_path: str) -> None:
    if value is None:
        raise ValueError(
            f"expected Arrow-safe ordered-reference value for {field_path!r}, got None"
        )
    if isinstance(value, Image.Image):
        raise TypeError(
            f"expected Arrow-safe ordered-reference value for {field_path!r}, "
            f"got PIL.Image {value!r}"
        )
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_arrow_safe_value(item, f"{field_path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_arrow_safe_value(item, f"{field_path}.{key}")
        return
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError(
            f"expected Arrow-safe scalar/list/dict/tensor for {field_path!r}, "
            f"got {type(value).__name__}: {value!r}"
        )


def _move_to_cpu(obj):
    """Recursively move tensors to CPU within nested lists."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, list):
        return [_move_to_cpu(x) for x in obj]
    return obj


def _to_pil_image_list(per_sample: Any) -> List[Image.Image]:
    """Convert one sample's image value to a ``List[PIL.Image]``.

    Accepts ``Tensor(N,C,H,W)``, ``List[Tensor]``, or ``List[PIL]``; an empty
    sample stays ``[]``. Used to store image columns via the HF Image feature.
    """
    if len(per_sample) == 0:
        return []
    return standardize_image_batch(per_sample, output_type="pil")


def _is_image_feature(feature: Any) -> bool:
    """Return True if a HuggingFace feature stores images (``Image`` or a
    sequence/list of ``Image``).

    Image columns decode to PIL and must be excluded from the ``torch`` format.
    """
    if isinstance(feature, HFImage):
        return True
    if isinstance(feature, HFSequence):
        # ``Sequence.feature`` holds the inner feature; getattr avoids a stub gap.
        return _is_image_feature(getattr(feature, "feature", None))
    # Nested-list features (e.g. ``[Image()]``) also denote a sequence of images.
    if isinstance(feature, list):
        return len(feature) == 1 and _is_image_feature(feature[0])
    return False


def _python_format_column_names(dataset: HFDataset) -> set:
    """Names of columns surfaced as plain Python objects (HF "python" format)
    instead of torch tensors.

    Two sources:
        1. Image columns (detected by feature type): decode to PIL images of
           varying sizes, which cannot be cast to tensors.
        2. ``EXTRA_PYTHON_FORMAT_COLUMNS`` (by name): non-image columns such as
           ``metadata`` that must survive untouched for JSON serialization.
    """
    image_cols = {name for name, feat in dataset.features.items() if _is_image_feature(feat)}
    return image_cols | EXTRA_PYTHON_FORMAT_COLUMNS


def _apply_torch_format(dataset: HFDataset) -> None:
    """Split columns between the ``torch`` and ``python`` formats.

    Tensorizable columns get the ``torch`` format; columns from
    ``_python_format_column_names`` (image columns + ``metadata``) are excluded
    and surfaced via ``output_all_columns``, so ``__getitem__`` returns them as
    plain Python objects (PIL images / dicts) for ``collate_fn`` to handle.
    """
    python_cols = _python_format_column_names(dataset)
    torch_cols = [c for c in dataset.column_names if c not in python_cols]
    dataset.set_format(type="torch", columns=torch_cols, output_all_columns=True)


def _resolve_path(base_dir: str, path: str) -> str:
    """Resolve path: use as-is if absolute, otherwise join with base_dir."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_video_frames(video_path: str, fps: Optional[float] = None) -> List[Image.Image]:
    """
    Load video frames using imageio (diffusers standard).

    Args:
        video_path: Path to video file
        fps: If specified, resample video to this frame rate

    Returns:
        List of PIL Images representing video frames
    """
    frames = [Image.fromarray(frame).convert("RGB") for frame in iio.imread(video_path)]

    if fps is not None:
        # Uniform resampling based on target fps
        metadata = iio.immeta(video_path)
        original_fps = metadata.get("fps", 30)
        step = original_fps / fps
        indices = [int(i * step) for i in range(int(len(frames) / step))]
        frames = [frames[i] for i in indices if i < len(frames)]

    return frames


def _compute_function_hash(func: Optional[Callable], digits: int = 16) -> str:
    """
    Compute stable hash for function caching.
    For bound methods, includes class name to distinguish subclass implementations.
    """
    _MAX_DIGITS = 32
    digits = min(digits, _MAX_DIGITS)

    if func is None:
        return "none" * 4

    # Extract class context for bound methods
    class_prefix = ""
    if hasattr(func, "__self__"):
        class_name = func.__self__.__class__.__qualname__
        class_prefix = f"{class_name}:"

    try:
        # Method 1: Source code + class context
        source = inspect.getsource(func)
        source = "".join(source.split())
        combined = class_prefix + source
        return hashlib.md5(combined.encode()).hexdigest()[:digits]
    except (TypeError, OSError):
        # Method 2: Module path + class context
        try:
            module = inspect.getmodule(func)
            module_name = module.__name__ if module else "unknown"
            func_name = getattr(func, "__qualname__", getattr(func, "__name__", "anonymous"))
            signature = class_prefix + f"{module_name}.{func_name}"
            return hashlib.md5(signature.encode()).hexdigest()[:digits]
        except:
            # Method 3: Fallback with class context
            logger.warning(f"Could not compute stable hash for {func}, using id() fallback")
            signature = class_prefix + str(id(func))
            return hashlib.md5(signature.encode()).hexdigest()[:digits]


_ENCODER_METHOD_NAMES = ("encode_prompt", "encode_image", "encode_video", "encode_audio")


def _collect_named_params(func: Optional[Callable]) -> set[str]:
    """Named (non-VAR_KEYWORD / VAR_POSITIONAL) parameter names, minus ``self``."""
    if func is None:
        return set()
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return set()
    return {
        p.name
        for p in sig.parameters.values()
        if p.kind
        not in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    } - {"self"}


def _select_cache_relevant_kwargs(
    preprocess_func: Optional[Callable],
    preprocess_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the subset of *preprocess_kwargs* that can affect preprocessing output.

    Collects named parameters from:
      1. ``preprocess_func`` itself
      2. If ``preprocess_func`` accepts ``**kwargs`` AND is a bound method,
         also every ``encode_*`` method on the same adapter instance
         (``encode_prompt``, ``encode_image``, ``encode_video``,
         ``encode_audio``) — because ``BaseAdapter.preprocess_func``
         forwards its ``**kwargs`` to these methods via ``filter_kwargs``.

    The union of these parameter names becomes the key-filter. Keys not in
    the union (e.g. ``num_batches_per_epoch``, ``gradient_accumulation_steps``)
    are excluded from the cache fingerprint.

    Safety properties:
      - Over-hash is safe (worst case: unnecessary re-preprocess).
      - Under-hash is dangerous (cache corruption). This approach can only
        over-hash (includes encoder params for encoders that might not run
        at runtime), never under-hash.
      - Falls back to the full dict when signature inspection fails.

    To make a value influence the cache key without adding it to any
    function signature, pass it via ``extra_hash_strs`` instead.
    """
    kwargs = preprocess_kwargs or {}
    if preprocess_func is None or not kwargs:
        return dict(kwargs)

    relevant_keys = _collect_named_params(preprocess_func)

    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(preprocess_func).parameters.values()
    )
    if has_var_kw and hasattr(preprocess_func, "__self__"):
        adapter = preprocess_func.__self__
        for name in _ENCODER_METHOD_NAMES:
            encoder = getattr(adapter, name, None)
            if callable(encoder):
                relevant_keys |= _collect_named_params(encoder)

    if not relevant_keys:
        return dict(kwargs)

    return {k: v for k, v in kwargs.items() if k in relevant_keys}


def _compute_encode_funcs_hash(*funcs: Optional[Callable], digits: int = 16) -> str:
    """
    Compute joint hash for multiple functions.

    Ensures cache is invalidated when any preprocessing logic changes.

    Args:
        *funcs: Variable number of functions to hash
        digits: Number of hash digits to return

    Returns:
        Hexadecimal hash string representing joint hash
    """
    _MAX_DIGITS = 32
    digits = min(digits, _MAX_DIGITS)
    individual_hashes = [_compute_function_hash(func) for func in funcs]
    combined_parts = [f"func{i}:{hash_val}" for i, hash_val in enumerate(individual_hashes)]
    combined = "|".join(combined_parts)
    return hashlib.md5(combined.encode()).hexdigest()[:digits]
