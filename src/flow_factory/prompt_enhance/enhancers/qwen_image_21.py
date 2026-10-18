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

"""Qwen-Image-2.1 prompt rewriting with ``Qwen/Qwen-Image-2.1-PE-T2I`` and ``-PE-I2I``."""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from ...utils.base import isolated_rng
from ...utils.logger_utils import setup_logger
from ..abc import BasePromptEnhancer
from ..args import PromptEnhanceArguments

logger = setup_logger(__name__)

QWEN_IMAGE_21_SYSTEM_PROMPT_FILENAME = "system_prompt.txt"
# The rewriters were trained on source images of at most this many pixels.
QWEN_IMAGE_21_IMAGE_MAX_PIXELS = 1024 * 1024
# A sampled rewrite occasionally loops until max_new_tokens; it gets this many attempts in total.
QWEN_IMAGE_21_MAX_ATTEMPTS = 3
_OFFICIAL_SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Subtract ``penalty`` once from the logit of every token generated so far.

    This is vLLM's ``presence_penalty``: unlike ``repetition_penalty`` it is additive and
    leaves prompt tokens unpenalized.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_length :]
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
        return scores - self.penalty * seen


def downscale_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Lanczos-downscale ``image`` to at most ``max_pixels`` pixels, keeping its aspect ratio."""
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / (width * height)) ** 0.5
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS
    )


def _json_object_spans(text: str) -> List[str]:
    """Return the balanced top-level ``{...}`` spans of ``text``, ignoring braces in strings."""
    spans: List[str] = []
    depth, start = 0, 0
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : index + 1])
    return spans


def parse_rewritten_prompt(generated: str) -> Tuple[Optional[str], str]:
    """Extract ``rewritten_prompt`` from a decoded ``<thinking></think><JSON answer>`` generation.

    The answer's balanced ``{...}`` spans are tried last first; the first one that parses to an
    object with a non-empty ``rewritten_prompt`` wins.

    Args:
        generated: Decoded generated text; the chat template already opened the thinking block.

    Returns:
        ``(rewritten_prompt, "")``, or ``(None, reason)`` when the thinking block never closes or
        no span carries a non-empty ``rewritten_prompt``.
    """
    _, closed, answer = generated.partition("</think>")
    if not closed:
        return None, (
            "the generation ended inside its thinking block, usually because it reached "
            f"max_new_tokens; generation tail: {generated[-300:]!r}"
        )
    for span in reversed(_json_object_spans(answer)):
        try:
            candidate = json.loads(span)
        except json.JSONDecodeError:
            continue
        # Some checkpoints emit the key misspelled as ``rewrited_prompt``.
        rewritten = candidate.get("rewritten_prompt", candidate.get("rewrited_prompt"))
        if isinstance(rewritten, str) and rewritten.strip():
            return rewritten.strip(), ""
    return None, (
        "expected a JSON object holding a non-empty 'rewritten_prompt' after '</think>', "
        f"received {answer[-500:]!r}"
    )


class _QwenImage21PromptEnhancer(BasePromptEnhancer):
    """Shared recipe of the Qwen-Image-2.1 T2I and I2I prompt rewriters.

    Both are Qwen3.5-VL checkpoints that think before answering with a JSON object; the
    enhanced prompt is its ``rewritten_prompt``. The answer's ``wh_ratio`` / ``ratio_follow``
    canvas recommendation is not recorded, because training renders at the configured
    resolution. The default system prompt is the checkpoint's ``system_prompt.txt``, and the
    default ``generation`` is the official sampling profile of the task, so the variants of a
    prompt differ. ``presence_penalty`` follows vLLM semantics.

    A sampled rewrite that ends inside its thinking block or answers without a parseable
    ``rewritten_prompt`` is resampled from the continuing RNG stream of its seed, so the result
    still depends only on that seed; after ``QWEN_IMAGE_21_MAX_ATTEMPTS`` attempts the run fails.
    Greedy decoding gets one attempt.
    """

    uses_images = True
    supported_generation_keys = frozenset(
        {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "presence_penalty"}
    )
    registry_key: ClassVar[str]
    default_generation: ClassVar[Dict[str, Any]]

    def __init__(self, args: PromptEnhanceArguments, device: torch.device) -> None:
        super().__init__(args, device)
        penalty = self.generation["presence_penalty"]
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
            raise TypeError(
                "expected generation.presence_penalty to be a number, "
                f"received {type(penalty).__name__}: {penalty!r}"
            )

    @property
    def generation(self) -> Dict[str, Any]:
        """Return the official profile overridden by the configured ``generation``."""
        return {**self.default_generation, **self.args.generation}

    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        model_name_or_path = self.args.model_name_or_path
        if not model_name_or_path:
            raise ValueError(f"the {self.registry_key} enhancer requires model_name_or_path")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=self.args.torch_dtype,
        ).to(self.device)
        self.model.eval()
        self.system_prompt = (
            self._load_system_prompt(model_name_or_path)
            if self.args.system_prompt is None
            else self.args.system_prompt
        )
        # The checkpoints' generation_config stops only at <|endoftext|>, while assistant
        # turns end with the tokenizer's <|im_end|>.
        configured_eos = self.model.generation_config.eos_token_id
        configured_eos = [configured_eos] if isinstance(configured_eos, int) else configured_eos
        self.eos_token_ids = sorted(
            {*(configured_eos or ()), self.processor.tokenizer.eos_token_id}
        )

    @staticmethod
    def _load_system_prompt(model_name_or_path: str) -> str:
        model_dir = Path(model_name_or_path)
        if model_dir.is_dir():
            path = model_dir / QWEN_IMAGE_21_SYSTEM_PROMPT_FILENAME
            if not path.is_file():
                raise FileNotFoundError(
                    f"expected the rewriter's system prompt at {path}; set system_prompt to "
                    "supply one"
                )
        else:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(model_name_or_path, QWEN_IMAGE_21_SYSTEM_PROMPT_FILENAME))
        return path.read_text(encoding="utf-8").strip()

    @abstractmethod
    def prepare_images(self, prompt: str, images: List[Image.Image]) -> List[Image.Image]:
        """Validate one row's reference images and return the images the rewriter sees."""

    def build_messages(self, prompt: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Build the rewriter chat messages of one prompt; images precede the text, in order."""
        content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]

    @torch.no_grad()
    def enhance(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        seeds: List[int],
    ) -> List[str]:
        generation = self.generation
        presence_penalty = generation.pop("presence_penalty")
        max_attempts = QWEN_IMAGE_21_MAX_ATTEMPTS if generation["do_sample"] else 1
        enhanced = []
        for prompt, row_images, seed in zip(prompts, images, seeds):
            messages = self.build_messages(prompt, self.prepare_images(prompt, row_images))
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=True,
            ).to(self.device)
            prompt_length = inputs["input_ids"].shape[1]
            penalties = (
                [PresencePenaltyLogitsProcessor(presence_penalty, prompt_length)]
                if presence_penalty
                else []
            )
            with isolated_rng(seed):
                for attempt in range(1, max_attempts + 1):
                    output_ids = self.model.generate(
                        **inputs,
                        **generation,
                        logits_processor=LogitsProcessorList(penalties),
                        eos_token_id=self.eos_token_ids,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                    )
                    generated = self.processor.decode(
                        output_ids[0, prompt_length:], skip_special_tokens=True
                    )
                    rewritten, failure = parse_rewritten_prompt(generated)
                    if rewritten is not None:
                        break
                    logger.warning(
                        "Rewrite attempt %d/%d of prompt %r (seed %d) failed: %s",
                        attempt,
                        max_attempts,
                        prompt,
                        seed,
                        failure,
                    )
            if rewritten is None:
                raise ValueError(
                    f"the {self.registry_key} rewrite of prompt {prompt!r} (seed {seed}) failed "
                    f"all {max_attempts} attempt(s); last failure: {failure}"
                )
            enhanced.append(rewritten)
        return enhanced

    def describe(self) -> Dict[str, Any]:
        return {
            "system_prompt_sha256": hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest(),
            "generation": self.generation,
        }


class QwenImage21T2IPromptEnhancer(_QwenImage21PromptEnhancer):
    """Expand prompt-only rows into detailed scene descriptions with the T2I rewriter.

    A row with reference images fails; use ``qwen-image-2.1-i2i`` for image editing.
    """

    registry_key = "qwen-image-2.1-t2i"
    default_generation = {"max_new_tokens": 16256, **_OFFICIAL_SAMPLING, "presence_penalty": 1.5}

    def prepare_images(self, prompt: str, images: List[Image.Image]) -> List[Image.Image]:
        if images:
            raise ValueError(
                f"the qwen-image-2.1-t2i enhancer rewrites prompt-only rows, but prompt "
                f"{prompt!r} has {len(images)} reference image(s); use qwen-image-2.1-i2i"
            )
        return []


class QwenImage21I2IPromptEnhancer(_QwenImage21PromptEnhancer):
    """Rewrite image-editing instructions with the I2I rewriter.

    Every row needs at least one reference image. Images are downscaled to at most
    ``QWEN_IMAGE_21_IMAGE_MAX_PIXELS`` pixels and passed in row order, which the rewriter
    addresses as ``<image1>``, ``<image2>``, ... when a row has several.
    """

    registry_key = "qwen-image-2.1-i2i"
    default_generation = {"max_new_tokens": 24000, **_OFFICIAL_SAMPLING, "presence_penalty": 0.0}

    def prepare_images(self, prompt: str, images: List[Image.Image]) -> List[Image.Image]:
        if not images:
            raise ValueError(
                f"the qwen-image-2.1-i2i enhancer rewrites image-editing rows, but prompt "
                f"{prompt!r} has no reference image; use qwen-image-2.1-t2i"
            )
        return [downscale_to_max_pixels(image, QWEN_IMAGE_21_IMAGE_MAX_PIXELS) for image in images]
