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

# src/flow_factory/logger/tensorboard.py
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from .abc import Logger
from .formatting import LogImage, LogTable, LogVideo, _compute_resize_dims


class TensorboardLogger(Logger):
    # TensorBoard renders video summaries as animated GIF images.
    video_format = "gif"

    def _init_platform(self):
        # Store under <log_args.save_dir>/tensorboard/<run_name>
        log_dir = (
            getattr(self.config.log_args, "save_dir", "")
            + f"/tensorboard/{self.config.log_args.run_name}"
        )
        self.platform = SummaryWriter(log_dir=log_dir)
        # Log hyperparameters
        self.platform.add_text("config", str(self.config.to_dict()))

    def _convert_to_platform(
        self, value: Any, height: Optional[int] = None, width: Optional[int] = None
    ) -> Any:
        """Convert to a tensorboard-compatible ``(kind, data, *extras)`` tuple."""
        if isinstance(value, LogImage):
            # TensorBoard expects HWC uint8 numpy or CHW float tensor
            img = value.get_pil()
            if height or width:
                h, w = value.get_size()
                new_h, new_w = _compute_resize_dims(h, w, height, width)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            arr = np.array(img.convert("RGB"))  # HWC
            return ("image", arr, value.caption)

        if isinstance(value, LogVideo):
            # Encode the GIF here rather than through SummaryWriter.add_video, which needs moviepy.
            path = self._video_path(value, height, width)
            return None if path is None else ("video", path, value.caption)

        if isinstance(value, LogTable):
            # No native table support - flatten items; `_log_single` logs each one separately
            h = height or value.target_height
            items = []
            for row in value.rows:
                for item in row:
                    if item is not None:
                        items.append(self._convert_to_platform(item, height=h))
            return ("table", items, value.columns)

        return ("scalar", value, None)

    def _log_impl(self, data: Dict, step: int):
        for key, value in data.items():
            self._log_single(key, value, step)

    def _log_single(self, key: str, value: Any, step: int):
        """Log a single converted value."""
        if isinstance(value, list):
            for i, v in enumerate(value):
                self._log_single(f"{key}/{i}", v, step)
            return

        if not isinstance(value, tuple):
            # Raw scalar
            if isinstance(value, (int, float)):
                self.platform.add_scalar(key, value, step)
            return

        dtype, *args = value
        if dtype == "scalar" and isinstance(args[0], (int, float)):
            self.platform.add_scalar(key, args[0], step)
        elif dtype == "image":
            self.platform.add_image(key, args[0], step, dataformats="HWC")
        elif dtype == "video":
            from tensorboard.compat.proto.summary_pb2 import Summary

            with open(args[0], "rb") as f:
                encoded = f.read()
            width, height = Image.open(args[0]).size
            image = Summary.Image(
                height=height, width=width, colorspace=3, encoded_image_string=encoded
            )
            self.platform.file_writer.add_summary(
                Summary(value=[Summary.Value(tag=key, image=image)]), step
            )
        elif dtype == "table":
            # Log table items individually
            items, columns = args
            for i, item in enumerate(items):
                col_idx = i % len(columns)
                row_idx = i // len(columns)
                self._log_single(f"{key}/{columns[col_idx]}/{row_idx}", item, step)

    def _finish_platform(self) -> None:
        self.platform.close()

    def __del__(self):
        if hasattr(self, "platform") and not getattr(self, "_closed", False):
            self.platform.close()
