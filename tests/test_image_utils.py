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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from flow_factory.contracts.media import PixelRange
from flow_factory.models.configured_image_output import ConfiguredImageOutputCodec
from flow_factory.utils.image import (
    normalize_to_uint8,
    require_decoded_rgb_image,
    require_finite_bchw_image,
    standardize_image_batch,
)


def _drifted_unit_image(drift: float) -> torch.Tensor:
    image = torch.full((3, 4, 4), 0.6)
    image[0, 0, 0] = drift
    return image


@pytest.mark.parametrize("drift", [1.0 + 1e-6, 1.0078125, -1e-6])
def test_normalize_to_uint8_keeps_unit_scale_under_edge_drift(drift: float) -> None:
    """Scale unit pixels by 255 even when one pixel drifts just outside [0, 1]."""
    for image in (_drifted_unit_image(drift), _drifted_unit_image(drift).numpy()):
        pixels = np.asarray(normalize_to_uint8(image))

        assert pixels.dtype == np.uint8
        assert (pixels[1:] == 153).all()


def test_normalize_to_uint8_rejects_non_finite_pixels() -> None:
    """Refuse a non-finite image instead of emitting a near-black one."""
    with pytest.raises(ValueError, match="1 non-finite"):
        normalize_to_uint8(_drifted_unit_image(float("nan")))


def test_normalize_to_uint8_keeps_integer_bytes() -> None:
    """Keep near-black uint8 pixels as bytes instead of rescaling them as unit pixels."""
    image = np.ones((4, 4, 3), dtype=np.uint8)

    assert np.array_equal(normalize_to_uint8(image), image)


def test_standardize_image_batch_hands_rewards_visible_pixels_after_edge_drift() -> None:
    """Give PIL-consuming rewards the decoded colors rather than a near-black image."""
    (image,) = standardize_image_batch(_drifted_unit_image(1.0 + 1e-6), output_type="pil")

    assert (np.asarray(image)[1:] == 153).all()


def test_normalize_to_uint8_reads_signed_unit_pixels_only_when_declared() -> None:
    """Convert [-1, 1] images under an explicit declaration; undeclared negatives raise."""
    bright_signed = torch.full((3, 4, 4), 0.2)
    dark_signed = torch.full((3, 4, 4), -0.6)

    assert (normalize_to_uint8(bright_signed, PixelRange.SIGNED_UNIT) == 153).all()
    assert (normalize_to_uint8(dark_signed, PixelRange.SIGNED_UNIT) == 51).all()
    with pytest.raises(ValueError, match=r"expected unit pixels in \[0, 1\]"):
        normalize_to_uint8(dark_signed)


def test_normalize_to_uint8_rejects_floating_ranges_for_integer_pixels() -> None:
    """Refuse to reinterpret byte pixels under a floating interval declaration."""
    with pytest.raises(ValueError, match="hold \\[0, 255\\] bytes"):
        normalize_to_uint8(np.ones((4, 4, 3), dtype=np.uint8), PixelRange.UNIT)
    with pytest.raises(ValueError, match="expected integer pixels in \\[0, 255\\]"):
        normalize_to_uint8(torch.full((3, 4, 4), 256, dtype=torch.int32))


def test_standardize_image_batch_reads_numpy_range_from_dtype() -> None:
    """Divide dark byte images by 255 instead of mistaking them for unit pixels."""
    dark_bytes = np.ones((1, 4, 4, 3), dtype=np.uint8)
    unit_floats = np.full((1, 4, 4, 3), 0.6, dtype=np.float32)

    stacked = standardize_image_batch(dark_bytes, output_type="pt")
    (listed,) = standardize_image_batch([dark_bytes[0]], output_type="pt")

    torch.testing.assert_close(stacked, torch.full((1, 3, 4, 4), 1.0 / 255.0))
    torch.testing.assert_close(listed, torch.full((3, 4, 4), 1.0 / 255.0))
    torch.testing.assert_close(
        standardize_image_batch(unit_floats, output_type="pt"), torch.full((1, 3, 4, 4), 0.6)
    )
    with pytest.raises(ValueError, match="standardize_image_batch expected unit pixels"):
        standardize_image_batch(unit_floats * 255.0, output_type="pt")


def test_require_decoded_rgb_image_accepts_canonical_pil_target() -> None:
    """Accept positive-size RGB PIL targets without copying or converting them."""
    image = Image.new("RGB", (4, 3), color=(0, 127, 255))

    assert require_decoded_rgb_image(image, source="test image") is image


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        (object(), TypeError, "RGB PIL.Image"),
        (np.zeros((2, 3, 3), dtype=np.uint8), TypeError, "RGB PIL.Image"),
        (Image.new("L", (3, 2)), ValueError, "RGB mode"),
        (Image.new("RGBA", (3, 2)), ValueError, "RGB mode"),
        (Image.new("RGB", (0, 2)), ValueError, "positive HWC dimensions"),
        (Image.new("RGB", (3, 0)), ValueError, "positive HWC dimensions"),
    ],
)
def test_require_decoded_rgb_image_rejects_noncanonical_payloads(
    payload: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject ambiguous representations before a model-owned image processor."""
    with pytest.raises(error_type, match=message):
        require_decoded_rgb_image(payload, source="test image")


def test_configured_image_codec_rejects_uint8_numpy_before_preprocessing() -> None:
    """Do not let byte-domain arrays reach processors that interpret arrays as unit pixels."""
    payload = np.full((2, 3, 3), 255, dtype=np.uint8)
    media_batch = ((SimpleNamespace(payload=payload),),)

    with pytest.raises(TypeError, match="RGB PIL.Image"):
        ConfiguredImageOutputCodec._extract_images(media_batch)


def test_require_finite_bchw_image_accepts_model_specific_floating_range() -> None:
    """Share shape/dtype/finiteness without imposing one model normalization interval."""
    pixels = torch.tensor([-3.0, 0.0, 5.0]).view(1, 3, 1, 1)

    assert (
        require_finite_bchw_image(
            pixels,
            source="test image pixels",
            batch_size=1,
            height=1,
            width=1,
        )
        is pixels
    )


def test_require_finite_bchw_image_accepts_declared_rgba_channels() -> None:
    pixels = torch.zeros(2, 4, 3, 5)

    assert (
        require_finite_bchw_image(
            pixels,
            source="rgba pixels",
            batch_size=2,
            height=3,
            width=5,
            channels=4,
        )
        is pixels
    )
    with pytest.raises(ValueError, match="expected 3 channels in BCHW"):
        require_finite_bchw_image(pixels, source="rgb pixels", batch_size=2, height=3, width=5)


@pytest.mark.parametrize("channels", [2, 5, "4", 4.0])
def test_require_finite_bchw_image_rejects_unsupported_channel_counts(channels: object) -> None:
    with pytest.raises(ValueError, match="expected 3 \\(RGB\\) or 4 \\(RGBA\\)"):
        require_finite_bchw_image(
            torch.zeros(1, 4, 1, 1),
            source="pixels",
            batch_size=1,
            height=1,
            width=1,
            channels=channels,
        )


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        (np.zeros((1, 1, 1, 3), dtype=np.float32), TypeError, "torch.Tensor"),
        (torch.zeros(1, 4, 2, 2), ValueError, "3 channels in BCHW"),
        (torch.zeros(1, 3, 2, 2, dtype=torch.uint8), TypeError, "dtype floating"),
        (torch.full((1, 3, 2, 2), float("nan")), ValueError, "non-finite values"),
    ],
)
def test_require_finite_bchw_image_rejects_ambiguous_model_pixels(
    payload: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        require_finite_bchw_image(
            payload,
            source="test image pixels",
            batch_size=1,
            height=2,
            width=2,
        )


def test_configured_image_codec_enforces_complete_processor_tensor_contract() -> None:
    """The common configured-image codec must route through the shared tensor validator."""
    with pytest.raises(TypeError, match="dtype floating"):
        ConfiguredImageOutputCodec._validate_pixel_values(
            torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
            batch_size=1,
            height=2,
            width=2,
        )


def test_diffusers_image_processor_maps_rgb_pil_bytes_to_model_pixel_endpoints() -> None:
    """The shared PIL boundary gives Diffusers one and only one byte-to-unit conversion."""
    image = Image.new("RGB", (2, 2), color=(0, 127, 255))
    processor = VaeImageProcessor(vae_scale_factor=1)

    pixels = processor.preprocess([image], height=2, width=2)

    expected_channels = torch.tensor(
        [-1.0, 2.0 * 127.0 / 255.0 - 1.0, 1.0],
        dtype=torch.float32,
    )
    expected = expected_channels.view(1, 3, 1, 1).expand(1, 3, 2, 2)
    torch.testing.assert_close(pixels, expected, rtol=0, atol=1e-7)
