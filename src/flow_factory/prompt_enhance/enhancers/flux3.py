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

"""FLUX.3 prompt upsampling with ``black-forest-labs/flux-3-image-klein-pup``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from ...utils.base import isolated_rng
from ..abc import BasePromptEnhancer

FLUX3_SYSTEM_PROMPTS_FILENAME = "system_prompts.json"
FLUX3_DEFAULT_GENERATION = {"max_new_tokens": 3584, "do_sample": False}


class Flux3PromptEnhancer(BasePromptEnhancer):
    """Rewrite prompts into full scene captions with the FLUX.3 prompt upsampler.

    Messages follow the upsampler's published recipe: the ``t2i`` system prompt for
    prompt-only rows, and the ``i2i`` system prompt with the reference images in order
    for image-conditioned rows. The default decoding is greedy, so more than one variant
    per prompt requires ``generation.do_sample: true``.
    """

    uses_images = True
    supported_generation_keys = frozenset(
        {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty"}
    )

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        model_name_or_path = self.args.model_name_or_path
        if not model_name_or_path:
            raise ValueError("the flux3 enhancer requires model_name_or_path")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=self.args.torch_dtype,
        ).to(self.device)
        self.model.eval()
        self.system_prompts = self._load_system_prompts(model_name_or_path)

    @staticmethod
    def _load_system_prompts(model_name_or_path: str) -> Dict[str, str]:
        local_path = Path(model_name_or_path) / FLUX3_SYSTEM_PROMPTS_FILENAME
        if local_path.is_file():
            path = local_path
        else:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(model_name_or_path, FLUX3_SYSTEM_PROMPTS_FILENAME))
        system_prompts = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted({"t2i", "i2i"} - set(system_prompts))
        if missing:
            raise ValueError(f"expected 't2i' and 'i2i' in {path}, missing {missing}")
        return system_prompts

    def build_messages(self, prompt: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Build the upsampler chat messages of one prompt."""
        if not images:
            system_prompt = (
                self.system_prompts["t2i"]
                if self.args.system_prompt is None
                else self.args.system_prompt
            )
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        system_prompt = (
            self.system_prompts["i2i"]
            if self.args.system_prompt is None
            else self.args.system_prompt
        )
        text = (
            "Reference images follow in order (the first image, the second image, ...).\n"
            f"User request: {prompt}\nOutput the caption."
        )
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    @torch.no_grad()
    def enhance(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        seeds: List[int],
    ) -> List[str]:
        generation = {**FLUX3_DEFAULT_GENERATION, **self.args.generation}
        enhanced = []
        for prompt, row_images, seed in zip(prompts, images, seeds):
            inputs = self.processor.apply_chat_template(
                self.build_messages(prompt, row_images),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.device)
            with isolated_rng(seed):
                output_ids = self.model.generate(**inputs, **generation)
            new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            enhanced.append(self.processor.decode(new_tokens, skip_special_tokens=True).strip())
        return enhanced

    def describe(self) -> Dict[str, Any]:
        return {
            "system_prompts_sha256": {
                key: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for key, value in sorted(self.system_prompts.items())
            }
        }
