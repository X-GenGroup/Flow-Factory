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

# src/flow_factory/models/samples.py
from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.utils.import_utils import is_torch_available, is_torch_version
from PIL import Image

from ..utils.base import (
    ImageBatch,
    ImageSingle,
    VideoBatch,
    VideoSingle,
    audio_to_tensor,
    hash_pil_image,
    hash_pil_image_list,
    hash_tensor,
    hash_tensor_list,
    is_tensor_list,
    map_tensor_leaves,
    standardize_image_batch,
    standardize_video_batch,
)
from ..utils.logger_utils import setup_logger
from .references import canonicalize_reference_manifest, parse_reference_manifest
from .trajectory import StructuredTrajectory

logger = setup_logger(__name__)


__all__ = [
    "BaseSample",
    "StackedSampleBatch",
    "ImageConditionSample",
    "VideoConditionSample",
    "T2ISample",
    "T2VSample",
    "T2AVSample",
    "I2ISample",
    "I2VSample",
    "I2AVSample",
    "V2VSample",
    "OrderedReferenceConditionSample",
    "Ref2AVSample",
    "MiniMaxH3T2VASample",
    "MiniMaxH3FL2VASample",
    "MiniMaxH3Ref2VASample",
]


@dataclass
class BaseSample:
    """Per-sample output of an Adapter (strictly without a batch dimension).

    Invariant: every tensor field holds a single sample (e.g. ``image`` is
    ``(C, H, W)``, never ``(B, C, H, W)``); ``__post_init__`` relies on this to
    canonicalize media. To batch samples for training use ``BaseSample.stack``,
    which returns a :class:`StackedSampleBatch` (a dict subclass) rather than a
    batched BaseSample -- a batched BaseSample would violate this invariant and
    corrupt media in ``__post_init__`` (which keeps only index 0).
    """

    _id_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "prompt",
            "prompt_ids",
            "negative_prompt",
            "negative_prompt_ids",
        }
    )

    # Fields that are shared across the batch
    _shared_fields: ClassVar[frozenset[str]] = frozenset(
        {"height", "width", "latent_index_map", "log_prob_index_map"}
    )

    # Fields ``gather_samples`` must transport to satisfy concrete-class reconstruction
    # invariants, even when omitted from the consumer-requested field list.
    reconstruction_required_fields: ClassVar[frozenset[str]] = frozenset()

    # Denoiseing trajectory
    timesteps: Optional[torch.Tensor] = None  # (T+1,)
    all_latents: Optional[torch.Tensor] = None  # (num_steps, Seq_len, C)
    latent_index_map: Optional[torch.Tensor] = None  # (T+1,) LongTensor
    log_probs: Optional[torch.Tensor] = None  # (num_steps,)
    log_prob_index_map: Optional[torch.Tensor] = None  # (T+1,) LongTensor
    trajectory: Optional[StructuredTrajectory] = None
    # Output dimensions
    height: Optional[int] = None
    width: Optional[int] = None
    # Generated media
    image: Optional[ImageSingle] = (
        None  # PIL.Image | torch.Tensor | np.ndarray. This field will be convert to a tensor of shape (C, H, W) for canonicalization.
    )
    video: Optional[VideoSingle] = (
        None  # List[Image.Image] | torch.Tensor | np.ndarray. This field will be convert to a tensor of shape (T, C, H, W) for canonicalization.
    )
    audio: Optional[torch.Tensor] = (
        None  # torch.Tensor (C, T) | (T,) waveform, float32 [-1, 1]. This field will be promoted to (C, T) for canonicalization.
    )
    audio_sample_rate: Optional[int] = None  # Sample rate in Hz (e.g. 24000 for LTX2)
    # Prompt information
    prompt: Optional[str] = None
    prompt_ids: Optional[torch.Tensor] = None
    prompt_embeds: Optional[torch.Tensor] = None
    # Negative prompt information
    negative_prompt: Optional[str] = None
    negative_prompt_ids: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None

    # --- Multi-source training bookkeeping ---
    # Populated by `BaseTrainer._inject_batch_metadata` for every sample
    # produced under `data.datasets`. Both fields are None in legacy
    # single-source mode (no `data.datasets`); the reward gate treats
    # `source_id is None` as "applies to every reward" so byte-identical
    # legacy behavior is preserved.
    #
    # `source` is the human-readable name (used for log keys / metric prefixes /
    # debugging); `source_id` is the small monotonic int assigned by
    # `Arguments._assign_source_ids` and is the form used in hot-path
    # comparisons (set membership in `RewardArguments._datasets_resolved`,
    # cross-rank gather of an int8/int16 vector instead of strings).
    source: Optional[str] = field(default=None, repr=False, compare=False)
    source_id: Optional[int] = field(default=None, repr=False, compare=False)

    # Deterministic acquisition identity assigned by the sampling plan before
    # model inference. ``sampling_group_id`` is the authoritative group key;
    # ``unique_id`` remains its compatibility-facing accessor and falls back to
    # the legacy content fingerprint only for samples created outside a planned
    # training dataloader (evaluation, tests, and custom user code).
    sampling_group_id: Optional[int] = field(default=None, repr=False, compare=False)
    sampling_group_member_id: Optional[int] = field(default=None, repr=False, compare=False)
    sampling_sample_id: Optional[int] = field(default=None, repr=False, compare=False)

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Set of reward names that COULD have applied to this sample given
    # the current routing config (i.e. whose ``RewardArguments.applicable_datasets``
    # contained this sample's ``source``, or was None).  Populated
    # by ``RewardProcessor`` whenever a reward is computed.  Read by
    # ``AdvantageProcessor`` to aggregate authoritatively rather than
    # relying on ``np.isnan`` (which would silently mask in-model NaN
    # bugs).  See plan §6 for the design.
    applicable_rewards: Set[str] = field(default_factory=set, repr=False, compare=False)

    _unique_id: Optional[int] = field(default=None, repr=False, compare=False)

    def __init_subclass__(cls) -> None:
        """
        **Copied from diffusers.utils.outputs.BaseOutput.__init_subclass__**
        Register subclasses as PyTorch pytree nodes for DDP/FSDP compatibility.
        """
        super().__init_subclass__()
        if is_torch_available():
            import torch.utils._pytree as pytree

            def flatten(obj):
                """Flatten dataclass to (values, context)."""
                values = []
                keys = []
                for f in fields(obj):
                    keys.append(f.name)
                    values.append(getattr(obj, f.name))
                return values, keys

            def unflatten(values, keys):
                """Reconstruct dataclass from (values, context)."""
                return cls(**dict(zip(keys, values)))

            if is_torch_available() and is_torch_version("<", "2.2"):
                pytree._register_pytree_node(cls, flatten, unflatten)
            else:
                pytree.register_pytree_node(
                    cls, flatten, unflatten, serialized_type_name=f"{cls.__module__}.{cls.__name__}"
                )

    def __post_init__(self):
        """Canonicalize media fields, assuming a single sample (no batch dim).

        Each media field is normalized to one sample's tensor, so this must never
        run on batched data (batches live in :class:`StackedSampleBatch`).
        """
        # Standardize image field to tensor (C, H, W)
        if self.image is not None:
            # -> (1, C, H, W) -> (C, H, W)
            self.image = standardize_image_batch(self.image, "pt")[0]

        # Standardize video field to tensor (T, C, H, W)
        if self.video is not None:
            # -> (1, T, C, H, W) -> (T, C, H, W)
            self.video = standardize_video_batch(self.video, "pt")[0]

        # Standardize audio field to tensor (C, T)
        if self.audio is not None:
            self.audio = audio_to_tensor(self.audio)

    @classmethod
    def shared_fields(cls) -> frozenset[str]:
        """Merge all _shared_fields from inheritance chain."""
        fields = set()
        for base in cls.__mro__[:-1]:  # Exclude object
            if hasattr(base, "_shared_fields"):
                fields.update(base._shared_fields)
        return frozenset(fields)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a plain dict, lifting ``extra_kwargs`` keys to the top level.

        Used by ``stack()`` and ``**sample`` unpacking (e.g. reward-model kwargs).
        """
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        extra = result.pop("extra_kwargs", {})
        collisions = set(extra) & set(result)
        if collisions:
            raise ValueError(
                f"{type(self).__name__}.extra_kwargs keys collide with declared "
                f"fields {sorted(collisions)}; extra_kwargs must hold only "
                f"non-declared keys (got extra keys {sorted(extra)})."
            )
        result.update(extra)
        return result

    def __getattr__(self, key: str) -> Any:
        """Access attributes. Check extra_kwargs if not found.

        Note for callers: do NOT use ``object.__getattribute__(sample, ...)``
        to bypass this fallback — plain ``sample.<key>`` already does the
        right thing (real dataclass fields short-circuit before this method
        runs; only "missing" lookups fall through to ``extra_kwargs``).
        ``object.__getattribute__`` is appropriate ONLY inside this method
        body to avoid infinite recursion when reading ``extra_kwargs``
        itself.
        """
        try:
            extra = object.__getattribute__(self, "extra_kwargs")
        except AttributeError:
            # extra_kwargs not yet initialized (during __init__)
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

        if key in extra:
            return extra[key]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """Set attributes."""
        if key in type(self)._id_fields:
            object.__setattr__(self, "_unique_id", None)  # Reset unique_id cache

        super().__setattr__(key, value)

    def keys(self):
        return self.to_dict().keys()  # Keep consistent

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: sample['prompt']."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __iter__(self):
        """Allow iteration over keys (required for some mapping operations)."""
        return iter(self.keys())

    def to(
        self,
        device: Union[torch.device, str],
        depth: int = 1,
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> BaseSample:
        """Move all tensor fields to ``device`` (in place).

        Args:
            device: Target device.
            depth: 0 moves only direct tensor fields; 1 also moves tensor-list fields.
            non_blocking: Forwarded to ``Tensor.to``. Enables truly asynchronous
                H2D when the host source is pinned (used by the copy-stream prefetch
                in the optimize loop).
            pin_memory: When moving to CPU, return page-locked (pinned) tensors so a
                subsequent H2D copy can be issued asynchronously. Ignored for
                non-CPU targets. Implies a blocking D2H (the copy must finish before
                the pinned buffer is filled).
        """
        assert 0 <= depth <= 1, "Only depth 0 and 1 are supported."
        device = torch.device(device)
        pin = pin_memory and device.type == "cpu"

        def _move(t: torch.Tensor) -> torch.Tensor:
            if pin:
                # One D2H straight into pinned memory, avoiding the
                # pageable->pinned double copy of t.to("cpu").pin_memory().
                out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                return out.copy_(t, non_blocking=non_blocking)
            return t.to(device, non_blocking=non_blocking)

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, _move(value))
            elif depth == 1 and is_tensor_list(value):
                setattr(
                    self,
                    field.name,
                    [_move(t) if isinstance(t, torch.Tensor) else t for t in value],
                )
            elif depth == 1 and isinstance(value, dict):
                # Move tensors nested in ANY dict field (in practice only
                # extra_kwargs: advantage, mu_teacher, ...) so they follow the
                # sample across offload/prefetch. Non-tensor leaves pass through.
                setattr(self, field.name, map_tensor_leaves(value, _move))
            elif isinstance(value, StructuredTrajectory):
                value.map_tensors(_move)

        return self

    def _hash_id_fields(self, hasher: hashlib._Hash) -> None:
        """Feed identity fields into *hasher*.

        Subclasses extend via ``super()._hash_id_fields(hasher)`` then
        hash their own fields into the same hasher.
        """
        if self.prompt is not None:
            hasher.update(self.prompt.encode("utf-8"))
        elif self.prompt_ids is not None:
            hasher.update(self.prompt_ids.cpu().numpy().tobytes())

        if self.negative_prompt is not None:
            hasher.update(self.negative_prompt.encode("utf-8"))
        elif self.negative_prompt_ids is not None:
            hasher.update(self.negative_prompt_ids.cpu().numpy().tobytes())

    def compute_unique_id(self, num_bytes: int = 8) -> int:
        """Compute a signed integer identifier for distributed grouping.

        Args:
            num_bytes: Number of digest bytes to use (default 8 = 64-bit,
                fits ``torch.int64`` used by ``collect_group_rewards``).
        """
        if not 1 <= num_bytes <= 32:
            raise ValueError(f"num_bytes must be in [1, 32] (sha256 digest), got {num_bytes}")
        hasher = hashlib.sha256()
        self._hash_id_fields(hasher)
        return int.from_bytes(hasher.digest()[:num_bytes], byteorder="big", signed=True)

    @property
    def unique_id(self) -> int:
        """Return the planned group ID or a legacy content fingerprint."""
        if self.sampling_group_id is not None:
            return self.sampling_group_id
        if self._unique_id is None:
            self._unique_id = self.compute_unique_id()
        return self._unique_id

    def assign_sampling_identity(
        self,
        *,
        group_id: int,
        group_member_id: int,
        sample_id: int,
    ) -> None:
        """Attach one sampler-owned identity without changing model inputs."""

        for value, name in (
            (group_id, "group_id"),
            (group_member_id, "group_member_id"),
            (sample_id, "sample_id"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"expected {name} to be a non-negative integer, got {value!r}")
        self.sampling_group_id = group_id
        self.sampling_group_member_id = group_member_id
        self.sampling_sample_id = sample_id

    def reset_unique_id(self):
        """Reset cached unique_id (call after modifying relevant fields)."""
        self._unique_id = None

    @classmethod
    def _stack_values(cls, key: str, values: List[Any]) -> Union[torch.Tensor, Dict, List, Any]:
        """
        Recursively stack values based on field configuration.

        Processing order:
            1. Shared fields → return first element only
            2. Stackable fields → attempt stacking (tensors/dicts)
            3. Other fields → return as list

        Args:
            key: Field name to determine stacking behavior
            values: List of values to stack

        Returns:
            - Any: If shared, returns first element
            - torch.Tensor: If stackable and all values are matching tensors
            - Dict: If stackable and all values are dicts (recursively stacked)
            - List: Otherwise
        """
        if not values:
            return values

        # All are None - return None
        if all(v is None for v in values):
            return None

        first = values[0]

        if any(isinstance(value, StructuredTrajectory) for value in values):
            if not all(isinstance(value, StructuredTrajectory) for value in values):
                raise ValueError(
                    "expected all samples to use trajectory=None or StructuredTrajectory, "
                    f"received {[type(value).__name__ for value in values]}"
                )
            return StructuredTrajectory.stack(values)

        # 1. Shared fields - take first element only
        if key in cls.shared_fields():
            return first

        # 2. Tensor fields, try to stack
        # Stack tensors with matching shapes
        if isinstance(first, torch.Tensor):
            # Assume all tensors
            if all(v.shape == first.shape for v in values):
                return torch.stack(values)
            return values

        # 3. Recursively stack dictionaries
        if isinstance(first, dict):
            if all(isinstance(v, dict) for v in values):
                return {k: cls._stack_values(k, [v[k] for v in values]) for k in first.keys()}
            return values

        # 3. Default - return as list
        return values

    @classmethod
    def stack(cls, samples: List[BaseSample]) -> StackedSampleBatch:
        """Collate per-sample BaseSamples into one batched ``StackedSampleBatch``.

        Thin alias for ``StackedSampleBatch(samples)`` -- the batch type owns the
        collation; this stays the discoverable stacking entry point. See
        :class:`StackedSampleBatch` for the per-key collation rules and the
        returned type.

        Raises:
            ValueError: If ``samples`` is empty.
        """
        return StackedSampleBatch(samples)


class StackedSampleBatch(dict):
    """Batched, collated view of a ``List[BaseSample]`` (built from ``samples`` by
    its constructor; also returned by ``BaseSample.stack``).

    A plain ``str``-keyed ``dict`` subclass, so it is a drop-in for the
    optimize-loop consumption: ``batch[key]``, ``batch.get(key)``, ``.items()``,
    ``**batch`` into ``adapter.forward``, and in-place ``batch[key] = value`` all
    keep working unchanged. Beyond a bare dict it:

    - names the "batched samples" concept (instead of an anonymous ``Dict[str, Any]``),
    - keeps a handle to the source per-sample objects (``batch.samples``) and the
      originating type (``batch.sample_cls``) for per-sample access / introspection,
    - exposes read-only attribute access (``batch.all_latents`` == ``batch["all_latents"]``).

    It is intentionally NOT a ``BaseSample``: a batched sample would violate
    ``BaseSample``'s "tensors without batch dim" invariant and re-trigger
    ``__post_init__`` media canonicalization (which assumes no batch dim and keeps
    only index 0).
    """

    def __init__(self, samples: List[BaseSample]) -> None:
        """Collate a non-empty ``List[BaseSample]`` into the batched mapping.

        Each sample is flattened via ``BaseSample.to_dict()`` (which lifts
        ``extra_kwargs`` keys to the top level), then ``BaseSample._stack_values``
        collates per key:
            - ``shared_fields()`` keys: take the first sample's value (not stacked).
            - tensors with matching shapes: ``torch.stack``; mismatched: list.
            - dicts: collated recursively; other types (str, Set, PIL): list.

        Raises:
            ValueError: If ``samples`` is empty.
        """
        if not samples:
            raise ValueError("No samples to stack.")

        sample_cls = type(samples[0])
        sample_dicts = [s.to_dict() for s in samples]
        all_keys: set = set()
        for d in sample_dicts:
            all_keys.update(d.keys())

        super().__init__(
            {
                key: sample_cls._stack_values(key, [d.get(key) for d in sample_dicts])
                for key in all_keys
            }
        )
        # ``samples``: the per-sample objects this batch was collated from. Stored
        # as an instance attribute (NOT a mapping key) -> excluded from ``**batch``
        # / ``.items()`` and resolved before ``__getattr__``. Callers needing
        # per-sample access (e.g. OPD teacher routing / ``mu_teacher`` write-back)
        # read ``batch.samples``.
        self.samples = samples

    @property
    def sample_cls(self) -> type[BaseSample]:
        """Originating ``BaseSample`` subclass (all samples share one type)."""
        return type(self.samples[0])

    def __getattr__(self, name: str) -> Any:
        """Read-only attribute access over the mapping (sugar for ``batch[name]``).

        Only invoked when normal attribute lookup fails, so the real ``samples``
        attribute, the ``sample_cls`` property, and ``dict`` methods short-circuit
        first.
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute or key '{name}'")


@dataclass
class ImageConditionSample(BaseSample):
    """Sample for tasks with image conditions.

    ``condition_images`` is canonicalized in ``__post_init__`` to a deterministic
    per-sample type so gather/stack dispatch stays consistent across samples/ranks:
    ``List[torch.Tensor(C, H, W)]`` in [0, 1] by default, or ``List[PIL.Image]``
    when the subclass sets ``condition_images_as_pil = True`` (see that field).
    """

    _id_fields: ClassVar[frozenset[str]] = BaseSample._id_fields | frozenset({"condition_images"})
    # Opt-in for adapters that persist condition_images via the HF Image feature
    # (``python_format_columns``); keep in sync with that ClassVar.
    condition_images_as_pil: ClassVar[bool] = False

    condition_images: Optional[ImageBatch] = (
        None  # Image.Image | torch.Tensor | np.ndarray, per-sample list or batched
    )

    def __post_init__(self):
        super().__post_init__()
        if self.condition_images is not None:
            output_type = "pil" if self.condition_images_as_pil else "pt"
            self.condition_images = standardize_image_batch(self.condition_images, output_type)
            if isinstance(self.condition_images, torch.Tensor):
                self.condition_images = list(self.condition_images.unbind(0))

    def _hash_id_fields(self, hasher: hashlib._Hash) -> None:
        super()._hash_id_fields(hasher)
        if self.condition_images is not None:
            cond_images = standardize_image_batch(self.condition_images, output_type="pil")
            hasher.update(hash_pil_image_list(cond_images).encode())


@dataclass
class VideoConditionSample(BaseSample):
    """Sample for tasks with video conditions.

    ``condition_videos`` is canonicalized in ``__post_init__`` to a deterministic
    per-sample type so gather/stack dispatch stays consistent across samples/ranks:
    ``List[torch.Tensor(T, C, H, W)]`` by default, or ``List[List[PIL.Image]]``
    (per-video frame lists) when the subclass sets ``condition_videos_as_pil = True``
    (see that field).
    """

    _id_fields: ClassVar[frozenset[str]] = BaseSample._id_fields | frozenset({"condition_videos"})
    # Mirror of ``ImageConditionSample.condition_images_as_pil`` for video frames;
    # opt-in for adapters that persist condition_videos as PIL frames.
    condition_videos_as_pil: ClassVar[bool] = False

    condition_videos: Optional[VideoBatch] = (
        None  # List[Image.Image] | torch.Tensor | np.ndarray, per-sample list or batched
    )

    def __post_init__(self):
        super().__post_init__()
        if self.condition_videos is not None:
            output_type = "pil" if self.condition_videos_as_pil else "pt"
            self.condition_videos = standardize_video_batch(self.condition_videos, output_type)
            if isinstance(self.condition_videos, torch.Tensor):
                self.condition_videos = list(self.condition_videos.unbind(0))

    def _hash_id_fields(self, hasher: hashlib._Hash) -> None:
        super()._hash_id_fields(hasher)
        if self.condition_videos is not None:
            cond_videos = standardize_video_batch(self.condition_videos, output_type="pil")
            for v in cond_videos:
                hasher.update(hash_pil_image_list(v).encode())


@dataclass
class T2ISample(BaseSample):
    """Text-to-Image sample output."""

    pass


@dataclass
class T2VSample(BaseSample):
    """Text-to-Video sample output."""

    pass


@dataclass
class I2ISample(ImageConditionSample):
    """Image-to-Image sample output."""

    pass


@dataclass
class I2VSample(ImageConditionSample):
    """Image-to-Video sample output."""

    pass


@dataclass
class I2AVSample(ImageConditionSample):
    """Image-to-Audio-Video sample output."""

    pass


@dataclass
class V2VSample(VideoConditionSample):
    """Video-to-Video sample output."""

    pass


@dataclass
class T2AVSample(BaseSample):
    """Text-to-Audio-Video sample output."""

    pass


@dataclass
class OrderedReferenceConditionSample(BaseSample):
    """Sample conditioned by an ordered heterogeneous reference manifest."""

    _id_fields: ClassVar[frozenset[str]] = BaseSample._id_fields | frozenset({"reference_manifest"})
    reconstruction_required_fields: ClassVar[frozenset[str]] = (
        BaseSample.reconstruction_required_fields | frozenset({"reference_manifest"})
    )

    reference_manifest: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reference_manifest is None:
            raise ValueError(
                f"expected reference_manifest string for {type(self).__name__}, got None"
            )
        references = parse_reference_manifest(self.reference_manifest, row_index=0)
        self.reference_manifest = canonicalize_reference_manifest(references, row_index=0)

    def _hash_id_fields(self, hasher: hashlib._Hash) -> None:
        super()._hash_id_fields(hasher)
        if self.reference_manifest is None:
            raise ValueError(
                f"expected reference_manifest string for {type(self).__name__}, got None"
            )
        hasher.update(self.reference_manifest.encode("utf-8"))


@dataclass
class Ref2AVSample(OrderedReferenceConditionSample):
    """Reference-to-Audio-Video sample output."""

    pass


@dataclass
class MiniMaxH3T2VASample(T2AVSample):
    """MiniMax H3 Text-to-Audio-Video sample output."""

    pass


@dataclass
class MiniMaxH3FL2VASample(I2AVSample):
    """MiniMax H3 First/Last-frame-to-Audio-Video sample output."""

    pass


@dataclass
class MiniMaxH3Ref2VASample(Ref2AVSample):
    """MiniMax H3 ordered-reference-to-Audio-Video sample output."""

    pass
