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

"""Runtime validation for media representation contracts and declared pixel intervals.

Floating pixels carry no marker of their interval, and ``[0, 1]`` versus ``[-1, 1]`` (or a dark
``[0, 255]`` frame) cannot be told apart from values, so conversions here never infer it. Integer
pixels are ``[0, 255]`` bytes, floating pixels are ``[0, 1]`` unless the caller declares another
:class:`~flow_factory.contracts.media.PixelRange`, and ``[-1, 1]`` VAE pixels cross into and out of
that space only through :func:`signed_unit_to_unit_pixels` and :func:`unit_to_signed_unit_pixels`.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..contracts.media import (
    MediaContainer,
    MediaDevice,
    MediaDType,
    MediaLayout,
    MediaRepresentation,
    PixelRange,
)

_LAYOUT_RANK = {
    MediaLayout.HWC: 3,
    MediaLayout.FHWC: 4,
    MediaLayout.CHANNELS_SAMPLES: 2,
    MediaLayout.BCHW: 4,
    MediaLayout.BCFHW: 5,
}
_CHANNEL_AXIS = {
    MediaLayout.HWC: 2,
    MediaLayout.FHWC: 3,
    MediaLayout.CHANNELS_SAMPLES: 0,
    MediaLayout.BCHW: 1,
    MediaLayout.BCFHW: 1,
}
# Edge drift tolerated per interval width; covers bfloat16 rounding (1/128 near 1.0) and resampling.
_PIXEL_DRIFT_FRACTION = 1.0 / 64.0
_INTERVAL_HINT = (
    "floating pixels are read as [0, 1] unless their producer declares another interval "
    "(value_range=PixelRange.SIGNED_UNIT or PixelRange.BYTE; signed_unit_to_unit_pixels() maps "
    "[-1, 1] VAE pixels)"
)

PixelArray = Union[torch.Tensor, np.ndarray]


def pixels_to_uint8(
    data: PixelArray,
    *,
    source: str,
    value_range: Optional[PixelRange] = None,
) -> PixelArray:
    """Convert pixels from a declared interval to ``uint8`` bytes.

    Integer pixels are bytes, boolean pixels are masks, and floating pixels are ``[0, 1]`` unless
    ``value_range`` declares otherwise. Edge drift within 1/64 of the interval width is clamped;
    values further out raise instead of rescaling every other pixel.

    Args:
        data: Pixel tensor or array in any layout.
        source: User-facing owner included in validation errors.
        value_range: Interval of floating ``data``; ``None`` means :attr:`PixelRange.UNIT`.
            Integer ``data`` accepts only ``None`` or :attr:`PixelRange.BYTE`.

    Returns:
        ``uint8`` pixels in the input container type and shape; tensors keep their device.

    Raises:
        TypeError: If ``data`` is not a tensor or array, its dtype is not integer, boolean, or
            floating, or ``value_range`` is not a :class:`PixelRange`.
        ValueError: If ``data`` is empty or non-finite, leaves the declared interval beyond edge
            drift, or is integer while ``value_range`` declares a floating interval.
    """
    _require_pixel_container(data, source, value_range)
    if _is_bool(data):
        _require_byte_declaration(data, source, value_range)
        if isinstance(data, torch.Tensor):
            return data.to(torch.uint8) * 255
        return data.astype(np.uint8) * np.uint8(255)
    if not _is_floating(data):
        _require_byte_declaration(data, source, value_range)
        _require_byte_pixels(data, source)
        if isinstance(data, torch.Tensor):
            return data.to(torch.uint8, copy=True)
        return data.astype(np.uint8)
    value_range = value_range or PixelRange.UNIT
    _require_float_pixels(data, source, value_range)
    lower, upper = value_range.bounds
    if isinstance(data, torch.Tensor):
        values = data if data.dtype in (torch.float32, torch.float64) else data.float()
        scaled = (values.clamp(lower, upper) - lower) * (255.0 / (upper - lower))
        return scaled.round().to(torch.uint8)
    values = data if data.dtype in (np.float32, np.float64) else data.astype(np.float32)
    scaled = (np.clip(values, lower, upper) - lower) * (255.0 / (upper - lower))
    return np.round(scaled).astype(np.uint8)


def pixels_to_unit(
    data: PixelArray,
    *,
    source: str,
    value_range: Optional[PixelRange] = None,
) -> PixelArray:
    """Convert pixels from a declared interval to floating ``[0, 1]``.

    Integer pixels are bytes and become ``float32``. Floating pixels are ``[0, 1]`` unless
    ``value_range`` declares otherwise; edge drift within 1/64 of the interval width is clamped
    and values further out raise. In-range unit tensors are returned as-is, keeping dtype,
    device, and storage; arrays are returned as ``float32``.

    Args:
        data: Pixel tensor or array in any layout.
        source: User-facing owner included in validation errors.
        value_range: Interval of floating ``data``; ``None`` means :attr:`PixelRange.UNIT`.
            Integer ``data`` accepts only ``None`` or :attr:`PixelRange.BYTE`.

    Returns:
        Floating ``[0, 1]`` pixels in the input container type and shape.

    Raises:
        TypeError: If ``data`` is not a tensor or array, its dtype is not integer or floating,
            or ``value_range`` is not a :class:`PixelRange`.
        ValueError: If ``data`` is empty or non-finite, leaves the declared interval beyond edge
            drift, or is integer while ``value_range`` declares a floating interval.
    """
    _require_pixel_container(data, source, value_range)
    if _is_bool(data):
        raise TypeError(
            f"{source} expected integer or floating pixels, received dtype={data.dtype}"
        )
    if not _is_floating(data):
        _require_byte_declaration(data, source, value_range)
        _require_byte_pixels(data, source)
        if isinstance(data, torch.Tensor):
            return data.to(torch.float32) / 255.0
        return data.astype(np.float32) / np.float32(255.0)
    value_range = value_range or PixelRange.UNIT
    low, high = _require_float_pixels(data, source, value_range)
    lower, upper = value_range.bounds
    if isinstance(data, np.ndarray):
        data = data.astype(np.float32, copy=False)
    if value_range is PixelRange.UNIT:
        if low >= lower and high <= upper:
            return data
        return data.clip(lower, upper)
    return (data.clip(lower, upper) - lower) / (upper - lower)


def signed_unit_to_unit_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Map ``[-1, 1]`` VAE pixels to ``[0, 1]`` like Diffusers ``VaeImageProcessor.denormalize``.

    Decoder overshoot is clamped. For processor-normalized inputs the round trip through
    :func:`unit_to_signed_unit_pixels` reproduces the VAE input bit for bit.
    """
    _require_floating_tensor(pixels, "signed_unit_to_unit_pixels")
    return (pixels * 0.5 + 0.5).clamp(0, 1)


def unit_to_signed_unit_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Map ``[0, 1]`` pixels to ``[-1, 1]`` VAE input like Diffusers ``normalize``.

    Nothing is clamped, so pixels outside ``[0, 1]`` stay outside ``[-1, 1]``.
    """
    _require_floating_tensor(pixels, "unit_to_signed_unit_pixels")
    return 2.0 * pixels - 1.0


def _require_floating_tensor(pixels: Any, source: str) -> None:
    if not isinstance(pixels, torch.Tensor) or not pixels.is_floating_point():
        raise TypeError(
            f"{source} expected a floating torch.Tensor, received {type(pixels).__name__}"
            f" with dtype={getattr(pixels, 'dtype', None)}"
        )


def _require_pixel_container(data: Any, source: str, value_range: Any) -> None:
    if value_range is not None and not isinstance(value_range, PixelRange):
        raise TypeError(
            f"{source} expected value_range to be PixelRange or None, "
            f"received {type(value_range).__name__}: {value_range!r}"
        )
    if isinstance(data, torch.Tensor):
        if data.is_complex():
            raise TypeError(f"{source} expected real-valued pixels, received dtype={data.dtype}")
        size = data.numel()
    elif isinstance(data, np.ndarray):
        if not (
            data.dtype == np.bool_
            or np.issubdtype(data.dtype, np.integer)
            or np.issubdtype(data.dtype, np.floating)
        ):
            raise TypeError(
                f"{source} expected integer, boolean, or floating pixels, received dtype={data.dtype}"
            )
        size = data.size
    else:
        raise TypeError(
            f"{source} expected a torch.Tensor or np.ndarray of pixels, "
            f"received {type(data).__name__}"
        )
    if size == 0:
        raise ValueError(f"{source} expected non-empty pixels, received shape {tuple(data.shape)}")


def _is_bool(data: PixelArray) -> bool:
    if isinstance(data, torch.Tensor):
        return data.dtype == torch.bool
    return data.dtype == np.bool_


def _is_floating(data: PixelArray) -> bool:
    if isinstance(data, torch.Tensor):
        return data.is_floating_point()
    return bool(np.issubdtype(data.dtype, np.floating))


def _require_byte_declaration(
    data: PixelArray, source: str, value_range: Optional[PixelRange]
) -> None:
    if value_range not in (None, PixelRange.BYTE):
        raise ValueError(
            f"{source} received {data.dtype} pixels, which hold [0, 255] bytes, but "
            f"value_range={value_range.value!r} declares a floating interval"
        )


def _require_byte_pixels(data: PixelArray, source: str) -> None:
    low, high = _pixel_extrema(data)
    if low < 0 or high > 255:
        raise ValueError(
            f"{source} expected integer pixels in [0, 255], received min={low:g}, max={high:g} "
            f"for dtype={data.dtype} shape={tuple(data.shape)}"
        )


def _require_float_pixels(
    data: PixelArray, source: str, value_range: PixelRange
) -> Tuple[float, float]:
    low, high = _pixel_extrema(data)
    if not (math.isfinite(low) and math.isfinite(high)):
        if isinstance(data, torch.Tensor):
            non_finite = int((~torch.isfinite(data)).sum())
        else:
            non_finite = int((~np.isfinite(data)).sum())
        raise ValueError(
            f"{source} expected finite pixels, received {non_finite} non-finite values "
            f"in dtype={data.dtype} shape={tuple(data.shape)}"
        )
    lower, upper = value_range.bounds
    margin = (upper - lower) * _PIXEL_DRIFT_FRACTION
    if low < lower - margin or high > upper + margin:
        raise ValueError(
            f"{source} expected {value_range.value} pixels in [{lower:g}, {upper:g}] "
            f"(edge drift up to {margin:g} is clamped), received min={low:.6g}, max={high:.6g} "
            f"for dtype={data.dtype} shape={tuple(data.shape)}; {_INTERVAL_HINT}"
        )
    return low, high


def _pixel_extrema(data: PixelArray) -> Tuple[float, float]:
    if isinstance(data, torch.Tensor):
        low, high = torch.aminmax(data)
        return float(low), float(high)
    return float(data.min()), float(data.max())


def require_media_payload(
    payload: Any,
    *,
    representation: MediaRepresentation,
    source: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Require a payload to satisfy one common physical media contract.

    Args:
        payload: Candidate PIL image, NumPy array, or torch tensor.
        representation: Dependency-neutral container/layout/dtype/range declaration.
        source: User-facing owner included in validation errors.
        expected_shape: Optional exact shape in the declared axis order.

    Returns:
        The original validated payload without copying or conversion.
    """
    if type(representation) is not MediaRepresentation:
        raise TypeError(
            "expected representation to be MediaRepresentation, "
            f"received {type(representation).__name__}: {representation!r}"
        )
    if type(source) is not str or not source:
        raise ValueError(f"expected source to be a non-empty string, received {source!r}")
    if expected_shape is not None:
        _validate_expected_shape(expected_shape, representation, source)

    if representation.container is MediaContainer.PIL_IMAGE:
        return _require_pil_payload(payload, representation, source, expected_shape)
    if representation.container is MediaContainer.NUMPY_ARRAY:
        return _require_numpy_payload(payload, representation, source, expected_shape)
    return _require_torch_payload(payload, representation, source, expected_shape)


def _require_pil_payload(
    payload: Any,
    representation: MediaRepresentation,
    source: str,
    expected_shape: Optional[Tuple[int, ...]],
) -> Image.Image:
    if not isinstance(payload, Image.Image):
        raise TypeError(f"{source} expected an RGB PIL.Image, received {type(payload).__name__}")
    if representation.channels != 3 or payload.mode != "RGB":
        raise ValueError(f"{source} expected RGB mode, received {payload.mode!r}")
    shape = (payload.height, payload.width, 3)
    _validate_shape(shape, representation, source, expected_shape)
    _validate_pil_values(payload, representation, source)
    return payload


def _require_numpy_payload(
    payload: Any,
    representation: MediaRepresentation,
    source: str,
    expected_shape: Optional[Tuple[int, ...]],
) -> np.ndarray:
    if not isinstance(payload, np.ndarray):
        raise TypeError(f"{source} expected a NumPy array, received {type(payload).__name__}")
    _validate_numpy_dtype(payload, representation, source)
    _validate_shape(tuple(payload.shape), representation, source, expected_shape)
    if representation.requires_contiguous and not payload.flags.c_contiguous:
        raise ValueError(f"{source} expected a C-contiguous array")
    _validate_numpy_values(payload, representation, source)
    return payload


def _require_torch_payload(
    payload: Any,
    representation: MediaRepresentation,
    source: str,
    expected_shape: Optional[Tuple[int, ...]],
) -> torch.Tensor:
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"{source} expected a torch.Tensor, received {type(payload).__name__}")
    _validate_torch_dtype(payload, representation, source)
    _validate_shape(tuple(payload.shape), representation, source, expected_shape)
    if representation.device is MediaDevice.CPU and payload.device.type != "cpu":
        raise ValueError(f"{source} expected a CPU tensor, received {payload.device}")
    if representation.requires_detached and (payload.requires_grad or payload.grad_fn is not None):
        raise ValueError(
            f"{source} expected a detached no-grad tensor, received "
            f"requires_grad={payload.requires_grad}, grad_fn={payload.grad_fn}"
        )
    if representation.requires_contiguous and not payload.is_contiguous():
        raise ValueError(f"{source} expected a contiguous tensor")
    _validate_torch_values(payload, representation, source)
    return payload


def _validate_expected_shape(
    expected_shape: Tuple[int, ...],
    representation: MediaRepresentation,
    source: str,
) -> None:
    if type(expected_shape) is not tuple:
        raise TypeError(
            f"{source} expected expected_shape to be tuple, "
            f"received {type(expected_shape).__name__}: {expected_shape!r}"
        )
    expected_rank = _LAYOUT_RANK[representation.layout]
    if len(expected_shape) != expected_rank:
        raise ValueError(
            f"{source} expected_shape rank must match {representation.layout.value.upper()} "
            f"rank {expected_rank}, received {expected_shape}"
        )
    for index, size in enumerate(expected_shape):
        if type(size) is not int or size <= 0:
            raise ValueError(
                f"{source} expected positive integer expected_shape[{index}], received {size!r}"
            )


def _validate_shape(
    shape: Tuple[int, ...],
    representation: MediaRepresentation,
    source: str,
    expected_shape: Optional[Tuple[int, ...]],
) -> None:
    layout_name = representation.layout.value.upper()
    expected_rank = _LAYOUT_RANK[representation.layout]
    if len(shape) != expected_rank:
        raise ValueError(
            f"{source} expected {layout_name} rank {expected_rank}, received shape {shape}"
        )
    if any(size <= 0 for size in shape):
        raise ValueError(f"{source} expected positive {layout_name} dimensions, received {shape}")
    if representation.channels is not None:
        channel_axis = _CHANNEL_AXIS[representation.layout]
        if shape[channel_axis] != representation.channels:
            raise ValueError(
                f"{source} expected {representation.channels} channels in {layout_name}, "
                f"received shape {shape}"
            )
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"{source} expected {layout_name} shape {expected_shape}, received {shape}"
        )


def _validate_numpy_dtype(
    payload: np.ndarray,
    representation: MediaRepresentation,
    source: str,
) -> None:
    if representation.dtype is MediaDType.UINT8:
        valid = payload.dtype == np.uint8
        expected = "uint8"
    elif representation.dtype is MediaDType.FLOAT32:
        valid = payload.dtype == np.float32
        expected = "float32"
    else:
        valid = np.issubdtype(payload.dtype, np.floating)
        expected = "floating"
    if not valid:
        raise TypeError(f"{source} expected dtype {expected}, received {payload.dtype}")


def _validate_torch_dtype(
    payload: torch.Tensor,
    representation: MediaRepresentation,
    source: str,
) -> None:
    if representation.dtype is MediaDType.UINT8:
        valid = payload.dtype is torch.uint8
        expected = "uint8"
    elif representation.dtype is MediaDType.FLOAT32:
        valid = payload.dtype is torch.float32
        expected = "float32"
    else:
        valid = payload.is_floating_point()
        expected = "floating"
    if not valid:
        raise TypeError(f"{source} expected dtype {expected}, received {payload.dtype}")


def _validate_numpy_values(
    payload: np.ndarray,
    representation: MediaRepresentation,
    source: str,
) -> None:
    value_range = representation.value_range
    if (
        representation.dtype is MediaDType.UINT8
        and (value_range.minimum is None or value_range.minimum <= 0.0)
        and (value_range.maximum is None or value_range.maximum >= 255.0)
    ):
        return
    if value_range.finite and not bool(np.isfinite(payload).all()):
        raise ValueError(f"{source} contains non-finite values")
    if value_range.minimum is None and value_range.maximum is None:
        return
    minimum = float(payload.min())
    maximum = float(payload.max())
    if value_range.minimum is not None and minimum < value_range.minimum:
        raise ValueError(
            f"{source} expected values >= {value_range.minimum}, received minimum {minimum}"
        )
    if value_range.maximum is not None and maximum > value_range.maximum:
        raise ValueError(
            f"{source} expected values <= {value_range.maximum}, received maximum {maximum}"
        )


def _validate_pil_values(
    payload: Image.Image,
    representation: MediaRepresentation,
    source: str,
) -> None:
    value_range = representation.value_range
    if (value_range.minimum is None or value_range.minimum <= 0.0) and (
        value_range.maximum is None or value_range.maximum >= 255.0
    ):
        return
    extrema = payload.getextrema()
    minimum = float(min(channel[0] for channel in extrema))
    maximum = float(max(channel[1] for channel in extrema))
    if value_range.minimum is not None and minimum < value_range.minimum:
        raise ValueError(
            f"{source} expected values >= {value_range.minimum}, received minimum {minimum}"
        )
    if value_range.maximum is not None and maximum > value_range.maximum:
        raise ValueError(
            f"{source} expected values <= {value_range.maximum}, received maximum {maximum}"
        )


def _validate_torch_values(
    payload: torch.Tensor,
    representation: MediaRepresentation,
    source: str,
) -> None:
    value_range = representation.value_range
    if (
        representation.dtype is MediaDType.UINT8
        and (value_range.minimum is None or value_range.minimum <= 0.0)
        and (value_range.maximum is None or value_range.maximum >= 255.0)
    ):
        return
    if value_range.finite and not bool(torch.isfinite(payload).all()):
        raise ValueError(f"{source} contains non-finite values")
    if value_range.minimum is not None:
        minimum = float(payload.amin())
        if minimum < value_range.minimum:
            raise ValueError(
                f"{source} expected values >= {value_range.minimum}, received minimum {minimum}"
            )
    if value_range.maximum is not None:
        maximum = float(payload.amax())
        if maximum > value_range.maximum:
            raise ValueError(
                f"{source} expected values <= {value_range.maximum}, received maximum {maximum}"
            )


__all__ = [
    "pixels_to_uint8",
    "pixels_to_unit",
    "require_media_payload",
    "signed_unit_to_unit_pixels",
    "unit_to_signed_unit_pixels",
]
