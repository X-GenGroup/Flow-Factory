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

"""FLUX.2-dev caption upsampling with its Mistral3 text encoder."""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from ..abc import BasePromptEnhancer
from ._diffusers import load_text_only_pipeline

FLUX2_DEFAULT_TEMPERATURE = 0.15


class Flux2PromptEnhancer(BasePromptEnhancer):
    """Rewrite prompts with ``Flux2Pipeline.upsample_prompt``.

    The upstream method selects the T2I or I2I system message from the first image of
    the whole batch, so prompt-only and image-conditioned rows are upsampled in separate
    calls. Each call samples the whole batch at once from the batch-seeded RNG, so a
    row's output also depends on its batch. Only ``generation.temperature`` is accepted;
    the system messages are fixed.
    """

    uses_images = True
    supported_generation_keys = frozenset({"temperature"})

    def load(self) -> None:
        from diffusers import Flux2Pipeline

        if self.args.system_prompt is not None:
            raise ValueError(
                "the flux2 enhancer uses the fixed FLUX.2 upsampling system messages; "
                f"system_prompt must be null, received {self.args.system_prompt!r}"
            )
        self.pipeline = load_text_only_pipeline(
            Flux2Pipeline,
            self.args.model_name_or_path,
            keep=("text_encoder", "tokenizer"),
            dtype=self.args.torch_dtype,
        )
        # upsample_prompt pads every prompt to 2048 tokens before generating; with right
        # padding the first new token is predicted from a pad position.
        self.pipeline.tokenizer.tokenizer.padding_side = "left"
        self.pipeline.text_encoder.to(self.device)

    def enhance(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        seeds: List[int],
    ) -> List[str]:
        del seeds
        temperature = self.args.generation.get("temperature", FLUX2_DEFAULT_TEMPERATURE)
        enhanced: List[Optional[str]] = [None] * len(prompts)
        text_rows = [row for row, row_images in enumerate(images) if not row_images]
        image_rows = [row for row, row_images in enumerate(images) if row_images]
        for rows, with_images in ((text_rows, False), (image_rows, True)):
            if not rows:
                continue
            outputs = self.pipeline.upsample_prompt(
                prompt=[prompts[row] for row in rows],
                images=[images[row] for row in rows] if with_images else None,
                temperature=temperature,
                device=self.device,
            )
            for row, text in zip(rows, outputs):
                enhanced[row] = text
        return enhanced
