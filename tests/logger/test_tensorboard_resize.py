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
    frames = np.zeros((2, 40, 80, 3), dtype=np.uint8)
    frames[1] = 200  # the GIF writer merges identical consecutive frames
    video = LogVideo(frames)
    writer = object.__new__(module.TensorboardLogger)
    writer.media_max_bytes = None

    kind, path, _ = writer._convert_to_platform(video, width=40)

    assert kind == "video"
    with Image.open(path) as gif:
        assert gif.format == "GIF" and gif.size == (40, 20) and gif.n_frames == 2
    video.cleanup()


def test_video_is_written_as_a_gif_image_summary_without_moviepy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write the encoded GIF bytes directly; SummaryWriter.add_video would need moviepy."""
    module = _tensorboard_module(monkeypatch)

    class Summary:
        def __init__(self, value):
            self.value = value

        class Value:
            def __init__(self, tag, image):
                self.tag, self.image = tag, image

        class Image:
            def __init__(self, **fields):
                self.fields = fields

    proto = types.ModuleType("tensorboard.compat.proto.summary_pb2")
    proto.Summary = Summary
    monkeypatch.setitem(sys.modules, "tensorboard.compat.proto.summary_pb2", proto)
    written = []
    writer = object.__new__(module.TensorboardLogger)
    writer.media_max_bytes = None
    writer.platform = types.SimpleNamespace(
        file_writer=types.SimpleNamespace(add_summary=lambda s, step: written.append((s, step))),
        close=lambda: None,
    )
    video = LogVideo(np.zeros((3, 12, 20, 3), dtype=np.uint8))

    writer._log_single("samples/0", writer._convert_to_platform(video), 5)

    ((summary, step),) = written
    (value,) = summary.value
    fields = value.image.fields
    assert step == 5 and value.tag == "samples/0"
    assert (fields["height"], fields["width"], fields["colorspace"]) == (12, 20, 3)
    assert fields["encoded_image_string"][:6] in (b"GIF87a", b"GIF89a")
    video.cleanup()
