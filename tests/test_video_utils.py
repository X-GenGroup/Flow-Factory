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

from flow_factory.utils.video import (
    decoded_video_to_unit_float,
    require_decoded_video_frames,
    require_finite_bcfhw_video,
)


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
