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

from typing import Any, Optional, Tuple

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


__all__ = ["require_media_payload"]
