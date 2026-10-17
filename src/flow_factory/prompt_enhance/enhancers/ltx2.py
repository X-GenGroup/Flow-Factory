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

"""LTX-2 prompt enhancement with its Gemma text encoder or dedicated enhancer."""

from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from ..abc import BasePromptEnhancer
from ._diffusers import load_text_only_pipeline

_LTX2_TEXT_COMPONENTS = ("text_encoder", "tokenizer", "processor", "prompt_enhancer")


class LTX2PromptEnhancer(BasePromptEnhancer):
    """Rewrite prompts with ``LTX2Pipeline.enhance_prompt``.

    LTX-2.0/2.3 generate with the Gemma3 text encoder; LTX-2.5 checkpoints ship a
    dedicated Gemma4 ``prompt_enhancer`` that greedy-decodes by default. A row with
    reference images is enhanced as image-to-video from its first image. A non-empty
    ``generation`` mapping replaces the upstream generation kwargs, except
    ``max_new_tokens``, which is passed separately.
    """

    uses_images = True
    supported_generation_keys = frozenset(
        {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "no_repeat_ngram_size",
        }
    )

    def load(self) -> None:
        from diffusers import LTX2Pipeline

        self.pipeline = load_text_only_pipeline(
            LTX2Pipeline,
            self.args.model_name_or_path,
            keep=_LTX2_TEXT_COMPONENTS,
            dtype=self.args.torch_dtype,
        )
        if self.pipeline.processor is None:
            raise ValueError(
                f"LTX2 prompt enhancement requires the 'processor' component, which "
                f"{self.args.model_name_or_path!r} does not provide"
            )
        generator = (
            self.pipeline.prompt_enhancer
            if self.pipeline.prompt_enhancer is not None
            else self.pipeline.text_encoder
        )
        generator.to(self.device)

    def system_prompt_for(self, with_image: bool) -> str:
        """Return the configured system prompt or the upstream default for this mode."""
        if self.args.system_prompt is not None:
            return self.args.system_prompt
        from diffusers.pipelines.ltx2 import utils as ltx2_utils

        if self.pipeline.prompt_enhancer is not None:
            return (
                ltx2_utils.LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
                if with_image
                else ltx2_utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
            )
        return (
            ltx2_utils.I2V_DEFAULT_SYSTEM_PROMPT
            if with_image
            else ltx2_utils.T2V_DEFAULT_SYSTEM_PROMPT
        )

    def enhance(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        seeds: List[int],
    ) -> List[str]:
        generation: Dict[str, Any] = dict(self.args.generation)
        max_new_tokens = generation.pop("max_new_tokens", None)
        enhanced = []
        for prompt, row_images, seed in zip(prompts, images, seeds):
            image = row_images[0] if row_images else None
            outputs = self.pipeline.enhance_prompt(
                prompt=prompt,
                system_prompt=self.system_prompt_for(image is not None),
                max_new_tokens=max_new_tokens,
                seed=seed,
                generation_kwargs=generation or None,
                device=self.device,
                image=image,
            )
            enhanced.append(outputs[0])
        return enhanced

    def describe(self) -> Dict[str, Any]:
        return {"dedicated_prompt_enhancer": self.pipeline.prompt_enhancer is not None}
