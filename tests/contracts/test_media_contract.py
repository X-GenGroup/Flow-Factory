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

"""Tests for common media format, representation, and geometry contracts."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from PIL import Image

from flow_factory.contracts import (
    DECODED_AUDIO_REPRESENTATION,
    DECODED_IMAGE_REPRESENTATION,
    DECODED_VIDEO_REPRESENTATION,
    MODEL_IMAGE_REPRESENTATION,
    MODEL_VIDEO_REPRESENTATION,
    UNIT_VIDEO_REPRESENTATION,
    MediaColorSpace,
    MediaContainer,
    MediaDevice,
    MediaDType,
    MediaFormat,
    MediaGeometry,
    MediaLayout,
    MediaRepresentation,
    MediaType,
    MediaValueRange,
    RateRequirement,
    validate_media_geometry,
)
from flow_factory.utils.media import require_media_payload


def test_canonical_representations_cover_decoded_and_model_media_boundaries() -> None:
    """One shared vocabulary represents image, video, and audio physical payloads."""
    assert (
        DECODED_IMAGE_REPRESENTATION.container,
        DECODED_IMAGE_REPRESENTATION.layout,
        DECODED_IMAGE_REPRESENTATION.dtype,
        DECODED_IMAGE_REPRESENTATION.color_space,
        DECODED_IMAGE_REPRESENTATION.value_range.minimum,
        DECODED_IMAGE_REPRESENTATION.value_range.maximum,
    ) == (
        MediaContainer.PIL_IMAGE,
        MediaLayout.HWC,
        MediaDType.UINT8,
        MediaColorSpace.RGB,
        0.0,
        255.0,
    )
    assert (
        DECODED_VIDEO_REPRESENTATION.container,
        DECODED_VIDEO_REPRESENTATION.layout,
        DECODED_VIDEO_REPRESENTATION.channels,
    ) == (MediaContainer.NUMPY_ARRAY, MediaLayout.FHWC, 3)
    assert (
        DECODED_AUDIO_REPRESENTATION.container,
        DECODED_AUDIO_REPRESENTATION.layout,
        DECODED_AUDIO_REPRESENTATION.device,
    ) == (MediaContainer.TORCH_TENSOR, MediaLayout.CHANNELS_SAMPLES, MediaDevice.CPU)
    assert MODEL_IMAGE_REPRESENTATION.layout is MediaLayout.BCHW
    assert MODEL_VIDEO_REPRESENTATION.layout is MediaLayout.BCFHW
    assert MODEL_VIDEO_REPRESENTATION.value_range.minimum is None
    assert MODEL_VIDEO_REPRESENTATION.value_range.maximum is None
    assert hash(DECODED_VIDEO_REPRESENTATION)

    with pytest.raises(FrozenInstanceError):
        DECODED_VIDEO_REPRESENTATION.channels = 4  # type: ignore[misc]


def test_media_value_range_constructor_is_strict() -> None:
    with pytest.raises(ValueError, match="maximum >= minimum"):
        MediaValueRange(minimum=2.0, maximum=1.0, finite=True)
    with pytest.raises(ValueError, match="bounded.*finite"):
        MediaValueRange(minimum=0.0, maximum=1.0, finite=False)
    with pytest.raises(TypeError, match="minimum.*finite float"):
        MediaValueRange(minimum=0, maximum=1.0, finite=True)  # type: ignore[arg-type]


def test_media_format_rejects_a_representation_from_another_modality() -> None:
    with pytest.raises(ValueError, match="image media requires HWC or BCHW"):
        MediaFormat(
            type=MediaType.IMAGE,
            fps=RateRequirement.NOT_APPLICABLE,
            sample_rate=RateRequirement.NOT_APPLICABLE,
            representation=DECODED_VIDEO_REPRESENTATION,
        )
    with pytest.raises(ValueError, match="audio media requires channels_samples"):
        MediaFormat(
            type=MediaType.AUDIO,
            fps=RateRequirement.NOT_APPLICABLE,
            sample_rate=RateRequirement.REQUIRED,
            representation=DECODED_IMAGE_REPRESENTATION,
        )
    with pytest.raises(ValueError, match="image media requires RGB color space"):
        MediaFormat(
            type=MediaType.IMAGE,
            fps=RateRequirement.NOT_APPLICABLE,
            sample_rate=RateRequirement.NOT_APPLICABLE,
            representation=replace(MODEL_IMAGE_REPRESENTATION, color_space=None),
        )


def test_common_geometry_validates_the_same_rate_policy_at_any_boundary() -> None:
    video_format = MediaFormat(
        type=MediaType.VIDEO,
        fps=RateRequirement.REQUIRED,
        sample_rate=RateRequirement.NOT_APPLICABLE,
        representation=DECODED_VIDEO_REPRESENTATION,
    )
    geometry = MediaGeometry(
        type=MediaType.VIDEO,
        height=32,
        width=48,
        frames=9,
        fps=24.0,
    )

    validate_media_geometry(geometry, video_format, identifier="decoded video")
    assert (geometry.num_frames, geometry.frame_rate) == (9, 24.0)

    missing_rate = MediaGeometry(
        type=MediaType.VIDEO,
        height=32,
        width=48,
        frames=9,
    )
    with pytest.raises(ValueError, match=r"required decoded video\.fps"):
        validate_media_geometry(missing_rate, video_format, identifier="decoded video")


def test_generic_payload_validator_enforces_declared_range_and_exact_shape() -> None:
    unit_video = np.array([0.0, 0.5, 1.0], dtype=np.float32).reshape(1, 1, 1, 3)
    assert (
        require_media_payload(
            unit_video,
            representation=UNIT_VIDEO_REPRESENTATION,
            source="unit video",
            expected_shape=(1, 1, 1, 3),
        )
        is unit_video
    )

    outside_range = unit_video.copy()
    outside_range[0, 0, 0, 2] = np.float32(1.01)
    with pytest.raises(ValueError, match="values <= 1.0"):
        require_media_payload(
            outside_range,
            representation=UNIT_VIDEO_REPRESENTATION,
            source="unit video",
        )
    with pytest.raises(ValueError, match="expected FHWC shape"):
        require_media_payload(
            unit_video,
            representation=UNIT_VIDEO_REPRESENTATION,
            source="unit video",
            expected_shape=(2, 1, 1, 3),
        )


def test_generic_payload_validator_enforces_a_narrower_pil_byte_range() -> None:
    representation = replace(
        DECODED_IMAGE_REPRESENTATION,
        value_range=MediaValueRange(minimum=0.0, maximum=254.0, finite=True),
    )

    with pytest.raises(ValueError, match="values <= 254.0"):
        require_media_payload(
            Image.new("RGB", (1, 1), color=(255, 0, 0)),
            representation=representation,
            source="bounded image",
        )


def test_model_media_contract_requires_finite_values_without_a_global_interval() -> None:
    pixels = torch.tensor([-3.0, 0.0, 5.0]).reshape(1, 3, 1, 1, 1)
    assert (
        require_media_payload(
            pixels,
            representation=MODEL_VIDEO_REPRESENTATION,
            source="model video",
        )
        is pixels
    )

    invalid = pixels.clone()
    invalid[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        require_media_payload(
            invalid,
            representation=MODEL_VIDEO_REPRESENTATION,
            source="model video",
        )


def test_representation_constructor_rejects_container_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="NumPy media representation requires HWC or FHWC"):
        MediaRepresentation(
            container=MediaContainer.NUMPY_ARRAY,
            layout=MediaLayout.BCFHW,
            dtype=MediaDType.FLOAT32,
            value_range=MediaValueRange(minimum=None, maximum=None, finite=True),
            channels=3,
            color_space=MediaColorSpace.RGB,
            device=MediaDevice.CPU,
            requires_contiguous=True,
            requires_detached=False,
        )
