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

"""Dependency-neutral media metadata, representation, and geometry contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class MediaType(str, Enum):
    """Media modalities understood by pipeline I/O contracts."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class RateRequirement(str, Enum):
    """Declare whether a modality-specific rate field is accepted or required."""

    NOT_APPLICABLE = "not_applicable"
    OPTIONAL = "optional"
    REQUIRED = "required"


class MediaContainer(str, Enum):
    """Declare the concrete container crossing one media boundary."""

    PIL_IMAGE = "pil_image"
    NUMPY_ARRAY = "numpy_array"
    TORCH_TENSOR = "torch_tensor"


class MediaLayout(str, Enum):
    """Declare the semantic axis order of one media payload."""

    HWC = "hwc"
    FHWC = "fhwc"
    CHANNELS_SAMPLES = "channels_samples"
    BCHW = "bchw"
    BCFHW = "bcfhw"


class MediaColorSpace(str, Enum):
    """Declare semantic color-channel ordering for visual media."""

    RGB = "rgb"


class MediaDType(str, Enum):
    """Declare an exact or family-level payload dtype."""

    UINT8 = "uint8"
    FLOAT32 = "float32"
    FLOATING = "floating"


class MediaDevice(str, Enum):
    """Declare whether a payload is CPU-owned or may live on any device."""

    CPU = "cpu"
    ANY = "any"


@dataclass(frozen=True, slots=True)
class MediaValueRange:
    """Declare finite-value and optional closed-bound requirements."""

    minimum: Optional[float]
    maximum: Optional[float]
    finite: bool

    def __post_init__(self) -> None:
        """Validate a coherent immutable numerical interval."""
        if type(self.finite) is not bool:
            raise TypeError(
                "expected MediaValueRange.finite to be bool, "
                f"received {type(self.finite).__name__}: {self.finite!r}"
            )
        for field_name in ("minimum", "maximum"):
            value = getattr(self, field_name)
            if value is not None:
                _require_finite_float(value, f"MediaValueRange.{field_name}")
        if self.minimum is not None and self.maximum is not None:
            if self.maximum < self.minimum:
                raise ValueError(
                    "expected MediaValueRange.maximum >= minimum, received "
                    f"minimum={self.minimum} and maximum={self.maximum}"
                )
        if (self.minimum is not None or self.maximum is not None) and not self.finite:
            raise ValueError("bounded MediaValueRange must require finite values")


@dataclass(frozen=True, slots=True)
class MediaRepresentation:
    """Declare container, axes, channels, dtype, ownership, and range at a boundary."""

    container: MediaContainer
    layout: MediaLayout
    dtype: MediaDType
    value_range: MediaValueRange
    channels: Optional[int]
    color_space: Optional[MediaColorSpace]
    device: MediaDevice
    requires_contiguous: bool
    requires_detached: bool

    def __post_init__(self) -> None:
        """Validate representation fields and container-specific coherence."""
        _require_enum(self.container, MediaContainer, "MediaRepresentation.container")
        _require_enum(self.layout, MediaLayout, "MediaRepresentation.layout")
        _require_enum(self.dtype, MediaDType, "MediaRepresentation.dtype")
        _require_exact_type(
            self.value_range,
            MediaValueRange,
            "MediaRepresentation.value_range",
        )
        _require_enum(self.device, MediaDevice, "MediaRepresentation.device")
        if self.color_space is not None:
            _require_enum(
                self.color_space,
                MediaColorSpace,
                "MediaRepresentation.color_space",
            )
        for field_name in ("requires_contiguous", "requires_detached"):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise TypeError(
                    f"expected MediaRepresentation.{field_name} to be bool, "
                    f"received {type(value).__name__}: {value!r}"
                )
        if self.channels is not None:
            _require_positive_int(self.channels, "MediaRepresentation.channels")

        if self.container is MediaContainer.PIL_IMAGE:
            if self.layout is not MediaLayout.HWC:
                raise ValueError("PIL image representation requires HWC layout")
            if self.dtype is not MediaDType.UINT8:
                raise ValueError("PIL image representation requires uint8 channel semantics")
            if self.device is not MediaDevice.CPU:
                raise ValueError("PIL image representation requires CPU ownership")
            if self.channels != 3:
                raise ValueError("PIL image representation requires three RGB channels")
            if self.color_space is not MediaColorSpace.RGB:
                raise ValueError("PIL image representation requires RGB color space")
            if self.requires_contiguous or self.requires_detached:
                raise ValueError(
                    "PIL image representation cannot require contiguous or detached storage"
                )
        elif self.container is MediaContainer.NUMPY_ARRAY:
            if self.layout not in (MediaLayout.HWC, MediaLayout.FHWC):
                raise ValueError("NumPy media representation requires HWC or FHWC layout")
            if self.device is not MediaDevice.CPU:
                raise ValueError("NumPy media representation requires CPU ownership")
            if self.requires_detached:
                raise ValueError("NumPy media representation cannot declare requires_detached=True")
        else:
            if self.layout not in (
                MediaLayout.CHANNELS_SAMPLES,
                MediaLayout.BCHW,
                MediaLayout.BCFHW,
            ):
                raise ValueError(
                    "torch media representation requires channels_samples, BCHW, or BCFHW layout"
                )

        if self.dtype is MediaDType.UINT8:
            minimum = self.value_range.minimum
            maximum = self.value_range.maximum
            if minimum is not None and minimum < 0.0:
                raise ValueError("uint8 media range cannot have a negative minimum")
            if maximum is not None and maximum > 255.0:
                raise ValueError("uint8 media range cannot exceed 255")


@dataclass(frozen=True, slots=True)
class MediaFormat:
    """Declare one modality's metadata and physical representation contract."""

    type: MediaType
    fps: RateRequirement
    sample_rate: RateRequirement
    representation: MediaRepresentation

    def __post_init__(self) -> None:
        """Validate strict field types and modality-specific coherence."""
        _require_enum(self.type, MediaType, "type")
        _require_enum(self.fps, RateRequirement, "fps")
        _require_enum(self.sample_rate, RateRequirement, "sample_rate")
        _require_exact_type(self.representation, MediaRepresentation, "representation")

        if self.type is MediaType.IMAGE:
            if (
                self.fps is not RateRequirement.NOT_APPLICABLE
                or self.sample_rate is not RateRequirement.NOT_APPLICABLE
            ):
                raise ValueError("image media cannot declare fps or sample_rate requirements")
            if self.representation.layout not in (MediaLayout.HWC, MediaLayout.BCHW):
                raise ValueError("image media requires HWC or BCHW representation")
            if self.representation.channels != 3:
                raise ValueError("image media requires exactly three RGB channels")
            if self.representation.color_space is not MediaColorSpace.RGB:
                raise ValueError("image media requires RGB color space")
            return
        if self.type is MediaType.VIDEO:
            if self.fps is RateRequirement.NOT_APPLICABLE:
                raise ValueError("video media must declare fps as optional or required")
            if self.sample_rate is not RateRequirement.NOT_APPLICABLE:
                raise ValueError("video media cannot declare a sample_rate requirement")
            if self.representation.layout not in (MediaLayout.FHWC, MediaLayout.BCFHW):
                raise ValueError("video media requires FHWC or BCFHW representation")
            if self.representation.channels != 3:
                raise ValueError("video media requires exactly three RGB channels")
            if self.representation.color_space is not MediaColorSpace.RGB:
                raise ValueError("video media requires RGB color space")
            return
        if self.fps is not RateRequirement.NOT_APPLICABLE:
            raise ValueError("audio media cannot declare an fps requirement")
        if self.sample_rate is RateRequirement.NOT_APPLICABLE:
            raise ValueError("audio media must declare sample_rate as optional or required")
        if self.representation.layout is not MediaLayout.CHANNELS_SAMPLES:
            raise ValueError("audio media requires channels_samples representation")
        if self.representation.color_space is not None:
            raise ValueError("audio media cannot declare a color space")


@runtime_checkable
class MediaMetadataLike(Protocol):
    """Structural media metadata shared by input and output references."""

    @property
    def type(self) -> str:
        """Return the public media type discriminator."""
        ...

    @property
    def fps(self) -> Optional[float]:
        """Return source frames per second when applicable."""
        ...

    @property
    def sample_rate(self) -> Optional[int]:
        """Return source samples per second when applicable."""
        ...


@dataclass(frozen=True, slots=True)
class MediaGeometry:
    """Describe resolved image, video, or audio geometry at any pipeline boundary."""

    type: MediaType
    height: Optional[int] = None
    width: Optional[int] = None
    frames: Optional[int] = None
    fps: Optional[float] = None
    samples: Optional[int] = None
    sample_rate: Optional[int] = None

    @property
    def frame_rate(self) -> Optional[float]:
        """Return the video clock using the configuration-facing field name."""
        return self.fps

    @property
    def num_frames(self) -> Optional[int]:
        """Return the temporal video extent using the configuration-facing field name."""
        return self.frames

    @property
    def num_samples(self) -> Optional[int]:
        """Return the audio extent using an explicit count-oriented field name."""
        return self.samples

    def __post_init__(self) -> None:
        """Validate strict modality-specific geometry fields."""
        _require_enum(self.type, MediaType, "MediaGeometry.type")
        for field_name in ("height", "width", "frames", "samples", "sample_rate"):
            value = getattr(self, field_name)
            if value is not None:
                _require_positive_int(value, f"MediaGeometry.{field_name}")
        if self.fps is not None:
            _require_positive_float(self.fps, "MediaGeometry.fps")

        populated = {
            name
            for name in ("height", "width", "frames", "fps", "samples", "sample_rate")
            if getattr(self, name) is not None
        }
        if self.type is MediaType.IMAGE:
            expected = {"height", "width"}
            if populated != expected:
                raise ValueError(
                    "expected image geometry fields ('height', 'width'), received "
                    f"{tuple(sorted(populated))}"
                )
            return
        if self.type is MediaType.VIDEO:
            required = {"height", "width", "frames"}
            allowed = required | {"fps"}
            if not required.issubset(populated) or not populated.issubset(allowed):
                raise ValueError(
                    "expected video geometry fields ('frames', 'height', 'width') with optional "
                    f"'fps', received {tuple(sorted(populated))}"
                )
            return
        required = {"samples"}
        allowed = required | {"sample_rate"}
        if not required.issubset(populated) or not populated.issubset(allowed):
            raise ValueError(
                "expected audio geometry field 'samples' with optional 'sample_rate', received "
                f"{tuple(sorted(populated))}"
            )


def validate_media_metadata(
    media: MediaMetadataLike,
    media_format: MediaFormat,
    *,
    identifier: str,
) -> None:
    """Validate shared modality and source-rate metadata.

    Args:
        media: Structural media metadata from an input or output reference.
        media_format: Common media declaration used by the role-specific contract.
        identifier: User-facing field prefix included in validation errors.

    Returns:
        None after successful validation.
    """
    _require_exact_type(media_format, MediaFormat, "media_format")
    if not isinstance(media, MediaMetadataLike):
        raise TypeError(
            f"expected {identifier} to implement MediaMetadataLike, "
            f"received {type(media).__name__}: {media!r}"
        )
    if type(media.type) is not str:
        raise TypeError(
            f"expected {identifier}.type to be str, "
            f"received {type(media.type).__name__}: {media.type!r}"
        )
    if media.type != media_format.type.value:
        raise ValueError(
            f"expected {identifier}.type {media_format.type.value!r}, received {media.type!r}"
        )
    _validate_rate(media.fps, media_format.fps, "fps", f"{identifier}.fps")
    _validate_rate(
        media.sample_rate,
        media_format.sample_rate,
        "sample_rate",
        f"{identifier}.sample_rate",
    )


def validate_media_geometry(
    geometry: MediaGeometry,
    media_format: MediaFormat,
    *,
    identifier: str,
) -> None:
    """Validate resolved geometry against a shared media declaration.

    Args:
        geometry: Runtime geometry resolved by an input or output path.
        media_format: Common media declaration used by the role-specific contract.
        identifier: User-facing field prefix included in validation errors.

    Returns:
        None after successful validation.
    """
    _require_exact_type(geometry, MediaGeometry, "geometry")
    _require_exact_type(media_format, MediaFormat, "media_format")
    if geometry.type is not media_format.type:
        raise ValueError(
            f"expected {identifier}.type {media_format.type.value!r}, "
            f"received {geometry.type.value!r}"
        )
    _validate_rate(geometry.fps, media_format.fps, "fps", f"{identifier}.fps")
    _validate_rate(
        geometry.sample_rate,
        media_format.sample_rate,
        "sample_rate",
        f"{identifier}.sample_rate",
    )


def _validate_rate(
    value: object,
    requirement: RateRequirement,
    rate_name: str,
    identifier: str,
) -> None:
    if requirement is RateRequirement.NOT_APPLICABLE:
        if value is not None:
            raise ValueError(f"expected {identifier}=None, received {value!r}")
        return
    if value is None:
        if requirement is RateRequirement.REQUIRED:
            raise ValueError(f"expected required {identifier}, received None")
        return
    if rate_name == "fps":
        _require_positive_float(value, identifier)
    else:
        _require_positive_int(value, identifier)


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"expected {field_name} to be {enum_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


def _require_exact_type(value: object, expected_type: type[object], field_name: str) -> None:
    if type(value) is not expected_type:
        raise TypeError(
            f"expected {field_name} to be {expected_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


def _require_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be positive int, received "
            f"{type(value).__name__}: {value!r}"
        )
    if value <= 0:
        raise ValueError(f"expected {field_name} > 0, received {value}")


def _require_finite_float(value: object, field_name: str) -> None:
    if type(value) is not float:
        raise TypeError(
            f"expected {field_name} to be finite float, received "
            f"{type(value).__name__}: {value!r}"
        )
    if not math.isfinite(value):
        raise ValueError(f"expected finite {field_name}, received {value!r}")


def _require_positive_float(value: object, field_name: str) -> None:
    _require_finite_float(value, field_name)
    if value <= 0:
        raise ValueError(f"expected {field_name} > 0, received {value!r}")


DECODED_IMAGE_REPRESENTATION = MediaRepresentation(
    container=MediaContainer.PIL_IMAGE,
    layout=MediaLayout.HWC,
    dtype=MediaDType.UINT8,
    value_range=MediaValueRange(minimum=0.0, maximum=255.0, finite=True),
    channels=3,
    color_space=MediaColorSpace.RGB,
    device=MediaDevice.CPU,
    requires_contiguous=False,
    requires_detached=False,
)
DECODED_VIDEO_REPRESENTATION = MediaRepresentation(
    container=MediaContainer.NUMPY_ARRAY,
    layout=MediaLayout.FHWC,
    dtype=MediaDType.UINT8,
    value_range=MediaValueRange(minimum=0.0, maximum=255.0, finite=True),
    channels=3,
    color_space=MediaColorSpace.RGB,
    device=MediaDevice.CPU,
    requires_contiguous=True,
    requires_detached=False,
)
DECODED_AUDIO_REPRESENTATION = MediaRepresentation(
    container=MediaContainer.TORCH_TENSOR,
    layout=MediaLayout.CHANNELS_SAMPLES,
    dtype=MediaDType.FLOAT32,
    value_range=MediaValueRange(minimum=None, maximum=None, finite=True),
    channels=None,
    color_space=None,
    device=MediaDevice.CPU,
    requires_contiguous=True,
    requires_detached=True,
)
UNIT_VIDEO_REPRESENTATION = MediaRepresentation(
    container=MediaContainer.NUMPY_ARRAY,
    layout=MediaLayout.FHWC,
    dtype=MediaDType.FLOAT32,
    value_range=MediaValueRange(minimum=0.0, maximum=1.0, finite=True),
    channels=3,
    color_space=MediaColorSpace.RGB,
    device=MediaDevice.CPU,
    requires_contiguous=True,
    requires_detached=False,
)
MODEL_IMAGE_REPRESENTATION = MediaRepresentation(
    container=MediaContainer.TORCH_TENSOR,
    layout=MediaLayout.BCHW,
    dtype=MediaDType.FLOATING,
    value_range=MediaValueRange(minimum=None, maximum=None, finite=True),
    channels=3,
    color_space=MediaColorSpace.RGB,
    device=MediaDevice.ANY,
    requires_contiguous=False,
    requires_detached=False,
)
MODEL_VIDEO_REPRESENTATION = MediaRepresentation(
    container=MediaContainer.TORCH_TENSOR,
    layout=MediaLayout.BCFHW,
    dtype=MediaDType.FLOATING,
    value_range=MediaValueRange(minimum=None, maximum=None, finite=True),
    channels=3,
    color_space=MediaColorSpace.RGB,
    device=MediaDevice.ANY,
    requires_contiguous=False,
    requires_detached=False,
)


__all__ = [
    "DECODED_AUDIO_REPRESENTATION",
    "DECODED_IMAGE_REPRESENTATION",
    "DECODED_VIDEO_REPRESENTATION",
    "MODEL_IMAGE_REPRESENTATION",
    "MODEL_VIDEO_REPRESENTATION",
    "MediaContainer",
    "MediaColorSpace",
    "MediaDType",
    "MediaDevice",
    "MediaFormat",
    "MediaGeometry",
    "MediaLayout",
    "MediaMetadataLike",
    "MediaRepresentation",
    "MediaType",
    "MediaValueRange",
    "RateRequirement",
    "UNIT_VIDEO_REPRESENTATION",
    "validate_media_geometry",
    "validate_media_metadata",
]
