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
from torch.utils.tensorboard import SummaryWriter
from .abc import Logger
from .formatting import LogImage, LogVideo, LogTable
from PIL import Image


class TensorboardLogger(Logger):
    def _init_platform(self):
        # Store in log_args / tensorboard
        log_dir = getattr(self.config.log_args, 'save_dir', "") + f"/tensorboard/{self.config.run_name}"
        self.platform = SummaryWriter(log_dir=log_dir)
        # Log hyperparameters
        self.platform.add_text("config", str(self.config.to_dict()))

    def _convert_to_platform(
        self, 
        value: Any, 
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """Convert to tensorboard-compatible format (returns tuple of (type, data))."""
        if isinstance(value, LogImage):
            # TensorBoard expects HWC uint8 numpy or CHW float tensor
            img = value.get_pil()
            if height or width:
                h, w = value.get_size()
                new_h, new_w = height or h, width or w
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            arr = np.array(img.convert('RGB'))  # HWC
            return ('image', arr, value.caption)
        
        if isinstance(value, LogVideo):
            # TensorBoard expects NTCHW format, N=1 for single video
            arr = value.get_numpy()  # THWC
            if height or width:
                h, w = arr.shape[1], arr.shape[2]
                new_h, new_w = height or h, width or w
                resized = [np.array(Image.fromarray(f).resize((new_w, new_h), Image.Resampling.LANCZOS)) for f in arr]
                arr = np.stack(resized)
            # THWC -> 1TCHW, normalize to [0,1]
            vid = np.transpose(arr, (0, 3, 1, 2))[np.newaxis] / 255.0
            return ('video', vid, value.caption, value.fps)
        
        if isinstance(value, LogTable):
            # No native table support - log as grid of images/videos
            h = height or value.target_height
            items = []
            for row in value.rows:
                for item in row:
                    if item is not None:
                        items.append(self._convert_to_platform(item, height=h))
            return ('table', items, value.columns)
        
        return ('scalar', value, None)

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
        if dtype == 'scalar' and isinstance(args[0], (int, float)):
            self.platform.add_scalar(key, args[0], step)
        elif dtype == 'image':
            self.platform.add_image(key, args[0], step, dataformats='HWC')
        elif dtype == 'video':
            self.platform.add_video(key, args[0], step, fps=args[2])
        elif dtype == 'table':
            # Log table items individually
            items, columns = args
            for i, item in enumerate(items):
                col_idx = i % len(columns)
                row_idx = i // len(columns)
                self._log_single(f"{key}/{columns[col_idx]}/{row_idx}", item, step)

    def __del__(self):
        if hasattr(self, 'platform'):
            self.platform.close()
