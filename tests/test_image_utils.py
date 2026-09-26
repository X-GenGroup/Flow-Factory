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

from flow_factory.models.configured_image_output import ConfiguredImageOutputCodec
from flow_factory.utils.image import require_decoded_rgb_image, require_finite_bchw_image


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
