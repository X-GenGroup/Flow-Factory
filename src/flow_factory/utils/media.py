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

"""Runtime validation for dependency-neutral media representation contracts."""

from __future__ import annotations

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
_FLOAT_PIXEL_INTERVALS = ((0.0, 1.0), (-1.0, 1.0))
_BYTE_SCALE_MIN_PEAK = 2.0


def normalize_pixels_to_uint8(
    data: Union[torch.Tensor, np.ndarray],
    *,
    source: str,
) -> Union[torch.Tensor, np.ndarray]:
    """Convert presentation pixels to ``uint8`` bytes without letting edge values pick the range.

    Integer arrays already hold ``[0, 255]`` bytes and boolean arrays are masks. A floating array
    is read as ``[0, 1]``, ``[-1, 1]``, or ``[0, 255]`` from its extrema, tolerating edge drift of
    1/64 of the interval width and clamping into the selected interval, so one drifted pixel cannot
    reinterpret every other pixel. A peak just above the unit interval but below two is ambiguous:
    as bytes it would be an almost black frame, and as unit pixels it signals a broken producer.

    Args:
        data: Pixel tensor or array in any layout.
        source: User-facing owner included in validation errors.

    Returns:
        ``uint8`` pixels in the input container type and shape; tensors keep their device.

    Raises:
        TypeError: If ``data`` is not a tensor or array, or its dtype is not integer, boolean, or
            floating.
        ValueError: If ``data`` is empty, holds non-finite values, or its extrema match no
            supported interval.
    """
    if isinstance(data, torch.Tensor):
        if data.numel() == 0:
            raise ValueError(
                f"{source} expected non-empty pixels, received shape {tuple(data.shape)}"
            )
        if data.dtype == torch.bool:
            return data.to(torch.uint8) * 255
        if data.is_complex():
            raise TypeError(f"{source} expected real-valued pixels, received dtype={data.dtype}")
        if not data.is_floating_point():
            _validate_byte_extrema(
                int(data.min().item()), int(data.max().item()), source, data.dtype, data.shape
            )
            return data.to(torch.uint8, copy=True)
        finite = torch.isfinite(data)
        if not bool(finite.all()):
            raise ValueError(
                f"{source} expected finite pixels, received {int((~finite).sum().item())} "
                f"non-finite values in dtype={data.dtype} shape={tuple(data.shape)}"
            )
        values = data if data.dtype in (torch.float32, torch.float64) else data.float()
        lower, upper = _pixel_interval(
            float(values.min().item()), float(values.max().item()), source, data.dtype, data.shape
        )
        scaled = (values.clamp(lower, upper) - lower) * (255.0 / (upper - lower))
        return scaled.round().to(torch.uint8)
    if isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError(f"{source} expected non-empty pixels, received shape {data.shape}")
        if data.dtype == np.bool_:
            return data.astype(np.uint8) * np.uint8(255)
        if np.issubdtype(data.dtype, np.integer):
            _validate_byte_extrema(int(data.min()), int(data.max()), source, data.dtype, data.shape)
            return data.astype(np.uint8)
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError(
                f"{source} expected integer, boolean, or floating pixels, received dtype={data.dtype}"
            )
        finite = np.isfinite(data)
        if not finite.all():
            raise ValueError(
                f"{source} expected finite pixels, received {int((~finite).sum())} "
                f"non-finite values in dtype={data.dtype} shape={data.shape}"
            )
        values = data if data.dtype in (np.float32, np.float64) else data.astype(np.float32)
        lower, upper = _pixel_interval(
            float(values.min()), float(values.max()), source, data.dtype, data.shape
        )
        scaled = (np.clip(values, lower, upper) - lower) * (255.0 / (upper - lower))
        return np.round(scaled).astype(np.uint8)
    raise TypeError(
        f"{source} expected a torch.Tensor or np.ndarray of pixels, received {type(data).__name__}"
    )


def _validate_byte_extrema(low: int, high: int, source: str, dtype: Any, shape: Any) -> None:
    if low < 0 or high > 255:
        raise ValueError(
            f"{source} expected integer pixels in [0, 255], received min={low}, max={high} "
            f"for dtype={dtype} shape={tuple(shape)}"
        )


def _pixel_interval(
    low: float,
    high: float,
    source: str,
    dtype: Any,
    shape: Any,
) -> Tuple[float, float]:
    for lower, upper in _FLOAT_PIXEL_INTERVALS:
        margin = (upper - lower) * _PIXEL_DRIFT_FRACTION
        if low >= lower - margin and high <= upper + margin:
            return lower, upper
    margin = 255.0 * _PIXEL_DRIFT_FRACTION
    if high >= _BYTE_SCALE_MIN_PEAK and low >= -margin and high <= 255.0 + margin:
        return 0.0, 255.0
    raise ValueError(
        f"{source} expected floating pixels in [0, 1], [-1, 1], or [0, 255] (peak >= "
        f"{_BYTE_SCALE_MIN_PEAK:g}), received min={low:.6g}, max={high:.6g} for dtype={dtype} "
        f"shape={tuple(shape)}"
    )


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


__all__ = ["normalize_pixels_to_uint8", "require_media_payload"]
