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

"""Expose an area budget while retaining Diffusers' FLUX.3 reference policies."""

from typing import Optional, Tuple, Union

from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
from PIL import Image


class _AreaBoundedFlux3ImageProcessor(Flux3ImageProcessor):
    """Override only sizing budgets; native preprocessing owns fill/crop and pixels."""

    def __init__(self, source: Flux3ImageProcessor, max_area: int):
        super().__init__(
            vae_scale_factor=source.config.vae_scale_factor,
            vae_latent_channels=source.config.vae_latent_channels,
        )
        self.max_area = max_area

    def _resize_if_canvas_exceeds_area(
        self, image: Image.Image, target_area: int = 4 * 1024 * 1024, max_side: int = 4096
    ) -> Image.Image:
        return super()._resize_if_canvas_exceeds_area(image, self.max_area, max_side)

    def _resize_if_exceeds_area(
        self, image: Image.Image, target_area: int = 1024 * 1024
    ) -> Image.Image:
        return super()._resize_if_exceeds_area(image, self.max_area)


def condition_image_processor(
    source: Flux3ImageProcessor,
    size: Optional[Union[int, Tuple[int, int]]],
    *,
    single_reference: bool,
) -> Flux3ImageProcessor:
    """Return native defaults or an independent processor with a user area budget.

    An integer caps area at size squared; a pair caps it at height times width,
    without forcing aspect ratio. The native single-reference canvas is at least
    64 pixels per side. Reject smaller budgets before its sizing loop can stall.
    """
    if size is None:
        return source
    dimensions = (size, size) if isinstance(size, int) else size
    if (
        not isinstance(dimensions, (tuple, list))
        or len(dimensions) != 2
        or any(type(value) is not int or value <= 0 for value in dimensions)
    ):
        raise ValueError(
            f"condition_image_size must be a positive integer or (height, width), received {size!r}"
        )
    area = dimensions[0] * dimensions[1]
    minimum = (
        max(64, source.config.vae_scale_factor)
        if single_reference
        else source.config.vae_scale_factor
    )
    if area < minimum * minimum:
        raise ValueError(
            f"condition_image_size={size!r} gives area={area}; this reference layout requires area>={minimum * minimum}"
        )
    return _AreaBoundedFlux3ImageProcessor(source, area)
