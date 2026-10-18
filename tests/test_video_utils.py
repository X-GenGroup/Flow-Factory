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

import numpy as np
import pytest
import torch

from flow_factory.contracts.media import PixelRange
from flow_factory.utils.video import (
    decoded_video_to_unit_float,
    normalize_video_to_uint8,
    require_decoded_video_frames,
    require_finite_bcfhw_video,
    standardize_video_batch,
)


def _drifted_unit_video(drift: float) -> torch.Tensor:
    video = torch.full((2, 3, 4, 4), 0.6)
    video[0, 0, 0, 0] = drift
    return video


@pytest.mark.parametrize("drift", [1.0 + 1e-6, 1.0078125, -1e-6])
def test_normalize_video_to_uint8_keeps_unit_scale_under_edge_drift(drift: float) -> None:
    """Scale unit pixels by 255 even when one pixel drifts just outside [0, 1]."""
    for video in (_drifted_unit_video(drift), _drifted_unit_video(drift).numpy()):
        pixels = np.asarray(normalize_video_to_uint8(video))

        assert pixels.dtype == np.uint8
        assert (pixels[1] == 153).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_normalize_video_to_uint8_rejects_non_finite_pixels(value: float) -> None:
    """Refuse a non-finite video instead of emitting a near-black one."""
    with pytest.raises(ValueError, match="1 non-finite"):
        normalize_video_to_uint8(_drifted_unit_video(value))


def test_normalize_video_to_uint8_keeps_integer_bytes() -> None:
    """Keep near-black uint8 frames as bytes instead of rescaling them as unit pixels."""
    video = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
    video[..., 0] = 1

    assert torch.equal(normalize_video_to_uint8(video), video)


@pytest.mark.parametrize(
    ("pixels", "value_range", "expected"),
    [
        (torch.tensor([-1.0, 0.0, 1.0]), PixelRange.SIGNED_UNIT, [0, 128, 255]),
        (np.array([-1.0, 0.0, 1.0], dtype=np.float32), PixelRange.SIGNED_UNIT, [0, 128, 255]),
        (torch.tensor([0.0, 100.0, 255.0]), PixelRange.BYTE, [0, 100, 255]),
    ],
)
def test_normalize_video_to_uint8_reads_declared_signed_and_byte_float_ranges(
    pixels: object,
    value_range: PixelRange,
    expected: list[int],
) -> None:
    """Read [-1, 1] and floating [0, 255] pixels only when the caller declares them."""
    assert np.asarray(normalize_video_to_uint8(pixels, value_range)).tolist() == expected


@pytest.mark.parametrize("peak", [1.5, 300.0, -0.5])
def test_normalize_video_to_uint8_rejects_undeclared_values_outside_unit_drift(
    peak: float,
) -> None:
    """Fail on floats beyond unit drift instead of guessing another interval for them."""
    with pytest.raises(ValueError, match=r"expected unit pixels in \[0, 1\].*PixelRange"):
        normalize_video_to_uint8(_drifted_unit_video(peak))


def test_standardize_video_batch_hands_rewards_visible_frames_after_edge_drift() -> None:
    """Give PIL-consuming rewards the decoded colors rather than a near-black video."""
    (frames,) = standardize_video_batch(_drifted_unit_video(1.0 + 1e-6), output_type="pil")

    assert (np.asarray(frames[1]) == 153).all()


def test_standardize_video_batch_reads_numpy_range_from_dtype() -> None:
    """Divide byte frames by 255 even when they are dark, and keep floating frames unit."""
    dark_bytes = np.ones((1, 2, 4, 4, 3), dtype=np.uint8)
    unit_floats = np.full((1, 2, 4, 4, 3), 0.6, dtype=np.float32)

    stacked = standardize_video_batch(dark_bytes, output_type="pt")
    (listed,) = standardize_video_batch([dark_bytes[0]], output_type="pt")

    torch.testing.assert_close(stacked, torch.full((1, 2, 3, 4, 4), 1.0 / 255.0))
    torch.testing.assert_close(listed, torch.full((2, 3, 4, 4), 1.0 / 255.0))
    torch.testing.assert_close(
        standardize_video_batch(unit_floats, output_type="pt"), torch.full((1, 2, 3, 4, 4), 0.6)
    )
    with pytest.raises(ValueError, match="standardize_video_batch expected unit pixels"):
        standardize_video_batch(unit_floats * 255.0, output_type="pt")


def test_require_decoded_video_frames_accepts_canonical_rgb_bytes() -> None:
    """Accept contiguous uint8 RGB frames with positive FHWC geometry."""
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)

    assert require_decoded_video_frames(frames, source="test video") is frames


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        (object(), TypeError, "NumPy array"),
        (np.zeros((2, 3, 4, 3), dtype=np.float32), TypeError, "dtype uint8"),
        (np.zeros((3, 4, 3), dtype=np.uint8), ValueError, "FHWC"),
        (np.zeros((2, 3, 4, 4), dtype=np.uint8), ValueError, "3 channels in FHWC"),
        (np.zeros((0, 3, 4, 3), dtype=np.uint8), ValueError, "positive FHWC dimensions"),
        (
            np.zeros((2, 3, 8, 3), dtype=np.uint8)[:, :, ::2, :],
            ValueError,
            "C-contiguous",
        ),
    ],
)
def test_require_decoded_video_frames_rejects_noncanonical_payloads(
    payload: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject ambiguous decoded representations before model preprocessing."""
    with pytest.raises(error_type, match=message):
        require_decoded_video_frames(payload, source="test video")


def test_decoded_video_to_unit_float_converts_once_without_mutating_source() -> None:
    """Map every byte exactly into one independent float32 unit-range buffer."""
    frames = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1).reshape(1, 1, 256, 3)
    original = frames.copy()
    expected = frames.astype(np.float32)
    expected /= np.float32(255.0)

    unit = decoded_video_to_unit_float(frames, source="test video")

    assert unit.dtype == np.float32
    assert unit.flags.c_contiguous
    assert not np.shares_memory(unit, frames)
    np.testing.assert_array_equal(frames, original)
    np.testing.assert_array_equal(unit, expected)


def test_require_finite_bcfhw_video_accepts_model_specific_floating_range() -> None:
    """Accept any finite model-owned interval while fixing exact RGB layout."""
    pixels = torch.tensor([-3.0, 0.0, 5.0]).view(1, 3, 1, 1, 1)

    assert (
        require_finite_bcfhw_video(
            pixels,
            source="test video pixels",
            batch_size=1,
            frames=1,
            height=1,
            width=1,
        )
        is pixels
    )


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        (np.zeros((1, 1, 1, 1, 3), dtype=np.float32), TypeError, "torch.Tensor"),
        (torch.zeros(1, 4, 2, 2, 2), ValueError, "3 channels in BCFHW"),
        (torch.zeros(1, 3, 2, 2, 2, dtype=torch.uint8), TypeError, "dtype floating"),
        (torch.full((1, 3, 2, 2, 2), float("inf")), ValueError, "non-finite values"),
    ],
)
def test_require_finite_bcfhw_video_rejects_ambiguous_model_pixels(
    payload: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        require_finite_bcfhw_video(
            payload,
            source="test video pixels",
            batch_size=1,
            frames=2,
            height=2,
            width=2,
        )
