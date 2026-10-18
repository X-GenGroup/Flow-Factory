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
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from flow_factory.contracts.media import PixelRange
from flow_factory.utils.base import hash_tensor
from flow_factory.utils.media import (
    pixels_to_uint8,
    pixels_to_unit,
    signed_unit_to_unit_pixels,
    unit_to_signed_unit_pixels,
)

# The next bfloat16 value above 1.0.
_BF16_OVERSHOOT = 1.0078125


def test_pixel_ranges_declare_closed_intervals() -> None:
    """Expose each declared interval as one closed numeric range."""
    assert PixelRange.UNIT.bounds == (0.0, 1.0)
    assert PixelRange.SIGNED_UNIT.bounds == (-1.0, 1.0)
    assert PixelRange.BYTE.bounds == (0.0, 255.0)


def test_pixels_to_unit_returns_in_range_tensors_untouched() -> None:
    """Hand canonical producers' tensors through without a copy or dtype change."""
    pixels = torch.rand(3, 4, 4, generator=torch.Generator().manual_seed(0)).bfloat16()

    assert pixels_to_unit(pixels, source="test") is pixels


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixels_to_unit_clamps_edge_drift_in_its_own_dtype(dtype: torch.dtype) -> None:
    """Clamp bfloat16-sized drift at both edges and leave every other pixel unchanged."""
    pixels = torch.full((3, 4, 4), 0.5, dtype=dtype)
    pixels[0, 0, 0] = _BF16_OVERSHOOT
    pixels[0, 0, 1] = -0.0078125

    unit = pixels_to_unit(pixels, source="test")

    assert unit.dtype == dtype
    assert float(unit[0, 0, 0]) == 1.0
    assert float(unit[0, 0, 1]) == 0.0
    assert torch.equal(unit[1:], pixels[1:])


def test_pixels_to_unit_converts_integer_bytes_by_dtype() -> None:
    """Divide integer pixels by 255 whatever their peak, in tensors and arrays alike."""
    expected = [0.0, 1.0 / 255.0, 1.0]

    tensor = pixels_to_unit(torch.tensor([0, 1, 255], dtype=torch.uint8), source="test")
    array = pixels_to_unit(np.array([0, 1, 255], dtype=np.uint8), source="test")

    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.tensor(expected))
    assert array.dtype == np.float32
    np.testing.assert_allclose(array, np.array(expected, dtype=np.float32))


@pytest.mark.parametrize(
    ("value_range", "values", "expected"),
    [
        (PixelRange.SIGNED_UNIT, [-1.0, 0.0, 1.0], [0.0, 0.5, 1.0]),
        (PixelRange.BYTE, [0.0, 51.0, 255.0], [0.0, 0.2, 1.0]),
    ],
)
def test_pixels_to_unit_maps_declared_float_intervals(
    value_range: PixelRange,
    values: list[float],
    expected: list[float],
) -> None:
    """Map a declared floating interval onto [0, 1]."""
    unit = pixels_to_unit(torch.tensor(values), source="test", value_range=value_range)

    torch.testing.assert_close(unit, torch.tensor(expected))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_pixel_conversions_reject_non_finite_pixels(value: float) -> None:
    """Refuse non-finite pixels in both conversion directions and both containers."""
    tensor = torch.full((3, 4, 4), 0.5)
    tensor[1, 2, 3] = value

    for convert in (pixels_to_unit, pixels_to_uint8):
        for pixels in (tensor, tensor.numpy()):
            with pytest.raises(ValueError, match="received 1 non-finite values"):
                convert(pixels, source="test")


def test_pixel_conversion_errors_name_the_owner_and_offending_extrema() -> None:
    """Report the owner, interval, extrema, and shape of [-1, 1] pixels given as unit pixels."""
    signed = torch.linspace(-1.0, 1.0, 12).reshape(3, 2, 2)

    with pytest.raises(ValueError) as error:
        pixels_to_unit(signed, source="T2ISample.image")

    message = str(error.value)
    assert message.startswith("T2ISample.image expected unit pixels in [0, 1]")
    assert "min=-1, max=1" in message
    assert "shape=(3, 2, 2)" in message
    assert "signed_unit_to_unit_pixels()" in message


def test_pixel_conversions_reject_ambiguous_declarations_and_containers() -> None:
    """Refuse value ranges that are not PixelRange, masks as unit pixels, and non-arrays."""
    with pytest.raises(TypeError, match="value_range to be PixelRange or None"):
        pixels_to_uint8(torch.zeros(3), source="test", value_range="unit")
    with pytest.raises(TypeError, match="expected integer or floating pixels"):
        pixels_to_unit(torch.zeros(3, dtype=torch.bool), source="test")
    with pytest.raises(TypeError, match="expected a torch.Tensor or np.ndarray"):
        pixels_to_unit([0.5], source="test")
    with pytest.raises(ValueError, match="expected non-empty pixels"):
        pixels_to_uint8(torch.zeros(0), source="test")


def test_pixels_to_uint8_rounds_unit_pixels_and_keeps_masks() -> None:
    """Round unit pixels to the nearest byte and map boolean masks to 0/255."""
    unit = torch.tensor([0.0, 0.5, 0.6, 1.0])
    mask = np.array([False, True])

    assert pixels_to_uint8(unit, source="test").tolist() == [0, 128, 153, 255]
    assert pixels_to_uint8(mask, source="test").tolist() == [0, 255]


def test_vae_pixel_maps_match_diffusers_normalize_and_denormalize() -> None:
    """Reproduce Diffusers' VAE pixel maps exactly, including decoder-overshoot clamping."""
    signed = torch.linspace(-1.25, 1.25, 257)
    unit = torch.rand(1024, generator=torch.Generator().manual_seed(0))

    assert torch.equal(signed_unit_to_unit_pixels(signed), VaeImageProcessor.denormalize(signed))
    assert torch.equal(unit_to_signed_unit_pixels(unit), VaeImageProcessor.normalize(unit))
    with pytest.raises(TypeError, match="expected a floating torch.Tensor"):
        unit_to_signed_unit_pixels(torch.zeros(3, dtype=torch.uint8))


def test_unit_condition_cache_reproduces_the_processor_vae_input_bit_for_bit() -> None:
    """Storing [0, 1] condition pixels must leave the processor's VAE input unchanged."""
    processor = VaeImageProcessor(vae_scale_factor=8)
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image = Image.fromarray(np.stack([levels, levels.T, 255 - levels], axis=-1))
    random_unit = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(0))

    for vae_input in (
        processor.preprocess(image, height=16, width=16),
        processor.normalize(random_unit),
    ):
        cached = signed_unit_to_unit_pixels(vae_input)
        assert torch.equal(pixels_to_unit(cached, source="test"), cached)
        assert torch.equal(unit_to_signed_unit_pixels(cached), vae_input)


def test_hash_tensor_hashes_unit_pixels_like_their_bytes_without_range_detection() -> None:
    """Hash [0, 1] pixels like their bytes and keep signed pixels distinct from unit ones."""
    levels = torch.arange(256, dtype=torch.uint8)
    unit = levels.float() / 255.0

    assert hash_tensor(unit) == hash_tensor(levels)
    assert hash_tensor(unit + 1e-5) == hash_tensor(unit)
    assert hash_tensor(unit * 2.0 - 1.0) != hash_tensor(unit)
