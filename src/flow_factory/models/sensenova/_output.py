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

"""Pixel-space output-state codec for SenseNova-U1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...contracts import MediaType
from ...samples import LatentState
from ...utils.image import require_decoded_rgb_image, require_finite_bchw_image
from ..output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)


@dataclass(frozen=True, slots=True)
class SenseNovaPixelOutputCodec:
    """Encode configured target images directly as normalized pixel states."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("transformer",)

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Resize decoded RGB targets and map ``[0, 255]`` to ``[-1, 1]``."""
        del condition, generator
        height, width = self.adapter._configured_output_image_geometry()
        arrays = []
        for sample_index, candidate in enumerate(media_batch):
            if len(candidate) != 1:
                raise ValueError(
                    "SenseNova output codec expected one image per sample, "
                    f"received {len(candidate)} for sample {sample_index}"
                )
            payload = require_decoded_rgb_image(
                candidate[0].payload,
                source=f"SenseNova output codec sample {sample_index}",
            )
            resized = payload.resize(
                (width, height),
                resample=Image.Resampling.BICUBIC,
            )
            arrays.append(np.asarray(resized, dtype=np.float32))

        pixels = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2)
        pixels = pixels.div(127.5).sub(1.0)
        require_finite_bchw_image(
            pixels,
            source="SenseNova normalized target pixels",
            batch_size=len(arrays),
            height=height,
            width=width,
        )
        model_dtype = getattr(self.adapter.transformer, "dtype", None)
        if model_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "SenseNova output codec expected transformer dtype in "
                f"(float16, bfloat16, float32), received {model_dtype!r}"
            )
        pixels = pixels.to(device=self.adapter.device, dtype=model_dtype)
        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=height,
                    width=width,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": pixels}),
            forward_context={},
            decode_context={"height": height, "width": width},
            geometry_signatures=tuple(signature for _ in arrays),
        )


__all__ = ["SenseNovaPixelOutputCodec"]
