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

"""Prompt enhancement through an OpenAI-compatible chat-completions server."""

from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union

import requests
from PIL import Image

from ..abc import BasePromptEnhancer

_GENERATION_TO_REQUEST = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_new_tokens": "max_tokens",
}


def image_data_url(image: Image.Image) -> str:
    """Encode one RGB image as a PNG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAICompatiblePromptEnhancer(BasePromptEnhancer):
    """Send each prompt to ``{base_url}/chat/completions`` of a vLLM or OpenAI-style server.

    ``model_name_or_path`` is the served model name and ``system_prompt`` is required.
    ``extra_kwargs`` configures ``base_url`` (required), ``api_key_env``
    (default ``OPENAI_API_KEY``; unset keys send no ``Authorization`` header),
    ``timeout`` in seconds, and ``max_concurrency``. Each request carries its row's seed.
    """

    uses_images = True
    supported_generation_keys = frozenset(_GENERATION_TO_REQUEST)

    def load(self) -> None:
        extra = self.args.extra_kwargs
        base_url = extra.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(
                "the openai-compatible enhancer requires extra_kwargs.base_url, "
                f"received {base_url!r}"
            )
        if not self.args.model_name_or_path:
            raise ValueError(
                "the openai-compatible enhancer requires model_name_or_path as the served "
                "model name"
            )
        if self.args.system_prompt is None:
            raise ValueError("the openai-compatible enhancer requires system_prompt")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = os.environ.get(extra.get("api_key_env", "OPENAI_API_KEY"))
        self.timeout = float(extra.get("timeout", 300.0))
        self.max_concurrency = int(extra.get("max_concurrency", 8))

    def build_payload(self, prompt: str, images: List[Image.Image], seed: int) -> Dict[str, Any]:
        """Build the chat-completions request body of one prompt."""
        user_content: Union[str, List[Dict[str, Any]]] = prompt
        if images:
            user_content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": image_data_url(image)}}
                for image in images
            ]
        payload: Dict[str, Any] = {
            "model": self.args.model_name_or_path,
            "messages": [
                {"role": "system", "content": self.args.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "seed": seed,
        }
        for key, value in self.args.generation.items():
            payload[_GENERATION_TO_REQUEST[key]] = value
        return payload

    def _request(self, prompt: str, images: List[Image.Image], seed: int) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.endpoint,
            json=self.build_payload(prompt, images, seed),
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError(
                f"expected string message content from {self.endpoint}, "
                f"received {type(content).__name__}: {content!r}"
            )
        return content.strip()

    def enhance(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        seeds: List[int],
    ) -> List[str]:
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(prompts))) as pool:
            futures = [
                pool.submit(self._request, prompt, row_images, seed)
                for prompt, row_images, seed in zip(prompts, images, seeds)
            ]
            return [future.result() for future in futures]
