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

"""TensorBoard media conversion preserves aspect ratio like the other loggers."""

import importlib
import sys
import types
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from flow_factory.logger.formatting import LogImage, LogVideo


def _tensorboard_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the TensorBoard logger with a stub writer so ``tensorboard`` stays optional."""
    writer_module = types.ModuleType("torch.utils.tensorboard")
    writer_module.SummaryWriter = object
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", writer_module)
    monkeypatch.delitem(sys.modules, "flow_factory.logger.tensorboard", raising=False)
    return importlib.import_module("flow_factory.logger.tensorboard")


def test_image_resize_with_only_height_preserves_aspect_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match ``LogImage.get_value`` when a table passes only the target height."""
    module = _tensorboard_module(monkeypatch)
    image = LogImage(Image.new("RGB", (200, 100)))

    kind, array, _ = module.TensorboardLogger._convert_to_platform(None, image, height=50)

    assert kind == "image"
    assert array.shape == (50, 100, 3)


def test_video_resize_with_only_width_preserves_aspect_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resize every video frame with the shared aspect-ratio rule."""
    module = _tensorboard_module(monkeypatch)
    video = LogVideo(np.zeros((2, 40, 80, 3), dtype=np.uint8))

    kind, array, _, _ = module.TensorboardLogger._convert_to_platform(None, video, width=40)

    assert kind == "video"
    assert array.shape == (1, 2, 3, 20, 40)
