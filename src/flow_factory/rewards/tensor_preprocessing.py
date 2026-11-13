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

"""Differentiable CLIP image geometry using processor-declared normalization."""

import torch
import torch.nn.functional as F


def clip_image_preprocess(image: torch.Tensor, processor) -> torch.Tensor:
    """Resize shortest side, center crop and normalize floating BCHW RGB pixels.

    Input values are in [0, 1]. Bicubic overshoot is retained, matching the
    tensor transform convention used in reward-gradient recipes. PIL scoring
    additionally quantizes pixels and therefore has a separate parity tolerance.
    """
    if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError("differentiable image scoring requires floating BCHW RGB pixels")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("differentiable image pixels must be finite and in [0, 1]")
    pixels = image.float()
    if processor.do_resize:
        size = processor.size
        if "shortest_edge" not in size:
            raise ValueError(f"CLIP resize requires shortest_edge, received {size!r}")
        height, width = pixels.shape[-2:]
        edge = size["shortest_edge"]
        target = (
            (edge, int(edge * width / height))
            if height <= width
            else (int(edge * height / width), edge)
        )
        pixels = F.interpolate(
            pixels, size=target, mode="bicubic", align_corners=False, antialias=True
        )
    if processor.do_center_crop:
        height, width = processor.crop_size["height"], processor.crop_size["width"]
        pad_h, pad_w = max(0, height - pixels.shape[-2]), max(0, width - pixels.shape[-1])
        pixels = F.pad(pixels, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        top, left = (pixels.shape[-2] - height) // 2, (pixels.shape[-1] - width) // 2
        pixels = pixels[..., top : top + height, left : left + width]
    if processor.do_normalize:
        mean = pixels.new_tensor(processor.image_mean)[None, :, None, None]
        std = pixels.new_tensor(processor.image_std)[None, :, None, None]
        pixels = (pixels - mean) / std
    return pixels
