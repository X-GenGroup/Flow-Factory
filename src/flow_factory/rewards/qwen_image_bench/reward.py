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

# src/flow_factory/rewards/qwen_image_bench/reward.py
"""Qwen-Image-Bench reward: remote VLM judge ("Q-Judger") over OpenAI-compatible HTTP.

Mirrors the deployment pattern of ``rational_rewards_t2i``: the heavy judge (a
fine-tuned ~27B Qwen3-VL) is served separately (e.g. ``vllm serve``); training
calls it over an OpenAI-compatible API. For each (prompt, image) pair the judge
scores facets on a 3-level hierarchy (L1 -> L2 -> L3), each facet rated 0/1/2/N/A,
mapped 0->0, 1->60, 2->100, aggregated L3->L2->L1->total (0-100), then normalized
to a reward in [0, 1].

Which L1 dimensions are scored is resolved per sample:
  - Faithful mode: parse ``dims_en`` from ``sample.metadata`` (the per-prompt facet
    list shipped by the Qwen-Image-Bench dataset).
  - Fallback mode: a fixed, configurable ``dimensions`` list (used when a sample
    carries no ``dims_en``, e.g. a generic prompt-only dataset).

Config (via ``RewardArguments`` extra_kwargs / YAML keys):
    api_base_url (str): OpenAI-compatible base URL, e.g. ``http://localhost:8000/v1``.
    api_key (str): default ``EMPTY``.
    vlm_model (str): served model id (must equal vLLM ``--served-model-name``);
        default ``Qwen-Image-Bench``.
    call_strategy (str): ``per_dimension`` (faithful, one call per L1 dim) or
        ``single_call`` (cheaper, all dims in one call). Default ``per_dimension``.
    dimensions (list[str]): fallback L1 dims when a sample has no ``dims_en``.
        Default all five.
    score_dimension (str): ``total`` or one L1 dim name. Default ``total``.
    max_concurrent (int): max simultaneous requests. Default 64.
    max_retries (int): transport retries per call. Default 5.
    timeout (float): per-request timeout (s). Default 300.
    max_tokens (int): generation cap (judge uses thinking). Default 4096.
    temperature (float): default 0.0.
    top_k (int): default 1 (via ``extra_body``).
    repetition_penalty (float): default 1.05 (via ``extra_body``).
    enable_thinking (bool): Qwen3 thinking mode (via ``extra_body``). Default True.
    seed (int): default 42.
    max_image_size (int): resize to a ``max_image_size`` square when the longest side
        exceeds it before sending. Default 1024.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from PIL import Image

from ...hparams import RewardArguments
from ...utils.image import pil_image_to_base64
from ..abc import PointwiseRewardModel, RewardModelOutput
from .checklists import (
    DIM_TO_CHECKLIST,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    parse_dims_by_level1,
)
from .score_utils import (
    aggregate_total_score,
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class _OpenAITransportError(Exception):
    """Placeholder so retry except-clauses stay valid when openai is absent.

    Replaced by the real openai transport exception types at import time. When
    ``openai`` is not installed, ``__init__`` raises a clear ImportError before
    any request is issued, so this placeholder is never actually caught.
    """


# Optional dependency (constraint #22 sanctioned exception (a)). ``__init__``
# raises a clear ImportError if these are unavailable.
try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AsyncOpenAI,
        RateLimitError,
    )
except ImportError:
    AsyncOpenAI = None
    APIConnectionError = APITimeoutError = RateLimitError = _OpenAITransportError

ALL_L1_DIMENSIONS: Tuple[str, ...] = tuple(DIM_TO_CHECKLIST.keys())

# Single-call rubric: present every selected L1 dimension's checklist in one
# request and ask for an L1-keyed JSON (avoids the L3-name collisions that a
# flat L2->L3 object would have across dimensions, e.g. "Composition").
_SINGLE_CALL_TEMPLATE = """\
# Text Prompt Used to Generate the Image
{prompt}

# Generated Image
<image>

# Evaluation Dimensions and Checklists
{all_checklists}

# Scoring Rules
- 0 (Fail): Clear defect present. Would noticeably reduce image quality.
- 1 (Pass): No defect. Meets baseline expectations.
- 2 (Excel): Exceptionally executed. Only when concrete excellence is observable.
- N/A: This criterion does not apply to this image/prompt.

# Output Format
Respond with a valid JSON object only (no markdown code blocks), keyed by the
level-1 dimension name shown above:
{{
  "<L1 dimension>": {{
    "<L2 sub-dimension>": {{
      "<L3 facet>": {{"score": 0|1|2}}
    }}
  }}
}}"""


class QwenImageBenchRewardModel(PointwiseRewardModel):
    """Pointwise T2I reward via the remote Qwen-Image-Bench judge.

    See module docstring for the scoring procedure and configuration keys.
    """

    required_fields: Tuple[str, ...] = ("prompt", "image")
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        if AsyncOpenAI is None:
            raise ImportError(
                "QwenImageBenchRewardModel requires the `openai` package. "
                "Install with: pip install openai"
            )
        self._async_openai_cls = AsyncOpenAI

        extra = config.extra_kwargs
        self.api_base_url: str = extra.get("api_base_url", "http://localhost:8000/v1")
        self.api_key: str = extra.get("api_key", "EMPTY")
        self.vlm_model: str = extra.get("vlm_model", "Qwen-Image-Bench")
        self.max_concurrent: int = int(extra.get("max_concurrent", 64))
        self.max_retries: int = int(extra.get("max_retries", 5))
        self.timeout: float = float(extra.get("timeout", 300.0))
        self.max_tokens: int = int(extra.get("max_tokens", 4096))
        self.temperature: float = float(extra.get("temperature", 0.0))
        self.top_k: int = int(extra.get("top_k", 1))
        self.repetition_penalty: float = float(extra.get("repetition_penalty", 1.05))
        self.enable_thinking: bool = bool(extra.get("enable_thinking", True))
        self.seed: int = int(extra.get("seed", 42))
        self.max_image_size: int = int(extra.get("max_image_size", 1024))

        self.call_strategy: str = extra.get("call_strategy", "per_dimension")
        if self.call_strategy not in ("per_dimension", "single_call"):
            raise ValueError(
                f"expected call_strategy in ('per_dimension', 'single_call'), "
                f"got {self.call_strategy!r}"
            )

        raw_dims = extra.get("dimensions", list(ALL_L1_DIMENSIONS))
        if not isinstance(raw_dims, (list, tuple)) or not raw_dims:
            raise TypeError(
                f"expected non-empty list/tuple for dimensions, got "
                f"{type(raw_dims).__name__}: {raw_dims!r}"
            )
        unknown = [d for d in raw_dims if d not in DIM_TO_CHECKLIST]
        if unknown:
            raise ValueError(f"unknown dimension(s) {unknown!r}; allowed: {list(DIM_TO_CHECKLIST)}")
        self.default_dimensions: Tuple[str, ...] = tuple(raw_dims)

        self.score_dimension: str = extra.get("score_dimension", "total")
        if self.score_dimension != "total" and self.score_dimension not in DIM_TO_CHECKLIST:
            raise ValueError(
                f"expected score_dimension == 'total' or one of {list(DIM_TO_CHECKLIST)}, "
                f"got {self.score_dimension!r}"
            )

    # ============================== Public API ==============================

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        metadata: Optional[List[str]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Score a batch of generated images with the Qwen-Image-Bench judge.

        Args:
            prompt: Text prompts (batch_size,).
            image: Generated PIL images (batch_size,). Falls back to the first
                video frame when ``image`` is None and ``video`` is provided.
            video: Optional generated videos (list of frame lists).
            metadata: Optional per-sample JSON strings; if a sample's metadata
                contains ``dims_en``, that per-prompt facet list selects the L1
                dimensions (faithful mode), otherwise ``dimensions`` is used.

        Returns:
            RewardModelOutput with per-sample rewards in [0, 1], shape (batch_size,).
        """
        if image is None and video is not None:
            image = [frames[0] for frames in video]
        if image is None:
            raise ValueError("QwenImageBenchRewardModel requires 'image' or 'video' input.")
        if len(prompt) != len(image):
            raise ValueError(
                f"expected len(prompt)==len(image), got {len(prompt)} prompts "
                f"and {len(image)} images"
            )

        l1_dims_per_sample = [
            self._resolve_l1_dims(metadata[i] if metadata is not None else None)
            for i in range(len(prompt))
        ]

        # Prepare + base64-encode once per image here (sync, no event loop yet) so
        # the async judge calls below are pure I/O and never block the loop.
        image_data_urls = [
            pil_image_to_base64(self._prepare_image(img), format="PNG") for img in image
        ]

        scores = asyncio.run(self._run_batch(prompt, image_data_urls, l1_dims_per_sample))
        rewards = torch.tensor(scores, dtype=torch.float32, device=self.device)
        return RewardModelOutput(rewards=rewards, extra_info={})

    # ============================== Dimension resolution ==============================

    def _resolve_l1_dims(self, meta: Optional[str]) -> List[str]:
        """Pick the L1 dimensions to score for one sample.

        Returns the L1 dims declared by the sample's ``dims_en`` when present
        (faithful mode), otherwise the configured fallback ``dimensions``.
        """
        if meta is not None:
            parsed_meta = json.loads(meta) if isinstance(meta, str) else meta
            if not isinstance(parsed_meta, dict):
                raise TypeError(
                    f"expected metadata to decode to a dict, got "
                    f"{type(parsed_meta).__name__}: {parsed_meta!r}"
                )
            dims_en = parsed_meta.get("dims_en")
            if dims_en:
                l1_dims = [d for d in parse_dims_by_level1(dims_en) if d in DIM_TO_CHECKLIST]
                if l1_dims:
                    return l1_dims
        return list(self.default_dimensions)

    # ============================== Async scoring ==============================

    async def _run_batch(
        self,
        prompts: List[str],
        image_data_urls: List[str],
        l1_dims_per_sample: List[List[str]],
    ) -> List[float]:
        """Create a loop-local client + semaphore, then score the batch.

        ``AsyncOpenAI`` and ``asyncio.Semaphore`` are event-loop-bound, so they
        must be created inside this per-call ``asyncio.run`` loop and threaded
        through the call chain -- never cached on ``self`` (caching reuses them
        across loops and raises "bound to a different event loop"). The semaphore
        is per call: ``max_concurrent`` caps in-flight judge requests per batch,
        so with ``async_reward`` and ``num_workers`` > 1 the effective server
        concurrency is ``num_workers * max_concurrent``.
        """
        async with self._async_openai_cls(
            base_url=self.api_base_url, api_key=self.api_key
        ) as client:
            semaphore = asyncio.Semaphore(max(1, self.max_concurrent))
            return await self._async_score_batch(
                client, semaphore, prompts, image_data_urls, l1_dims_per_sample
            )

    async def _async_score_batch(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        prompts: List[str],
        image_data_urls: List[str],
        l1_dims_per_sample: List[List[str]],
    ) -> List[float]:
        """Score every image concurrently; returns per-image rewards in [0, 1]."""
        tasks = [
            self._score_single_image(client, semaphore, p, url, dims)
            for p, url, dims in zip(prompts, image_data_urls, l1_dims_per_sample)
        ]
        return list(await asyncio.gather(*tasks))

    async def _score_single_image(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        prompt: str,
        image_data_url: str,
        l1_dims: List[str],
    ) -> float:
        """Run the judge for one image over its L1 dims and aggregate to [0, 1]."""
        # image already prepared + base64-encoded by __call__
        if self.call_strategy == "per_dimension":
            outputs = await asyncio.gather(
                *[
                    self._call_judge(
                        client,
                        semaphore,
                        USER_PROMPT_TEMPLATE.format(
                            prompt=prompt,
                            level1_dim=l1,
                            format_checklist=DIM_TO_CHECKLIST[l1],
                        ),
                        image_data_url,
                    )
                    for l1 in l1_dims
                ]
            )
            dim_results: Dict[str, dict] = {}
            for l1, output in zip(l1_dims, outputs):
                parsed = self._parse_dimension(output, l1)
                if parsed is not None:
                    dim_results[l1] = parsed
            return self._aggregate(dim_results)

        # single_call
        all_checklists = "\n\n".join(f"## {l1}\n{DIM_TO_CHECKLIST[l1]}" for l1 in l1_dims)
        output = await self._call_judge(
            client,
            semaphore,
            _SINGLE_CALL_TEMPLATE.format(prompt=prompt, all_checklists=all_checklists),
            image_data_url,
        )
        return self._aggregate(self._parse_single_call(output, l1_dims))

    async def _call_judge(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        user_text: str,
        image_data_url: str,
    ) -> Optional[str]:
        """Issue one judge request with retries; returns raw text or None on failure."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_content(user_text, image_data_url)},
        ]
        extra_body = {
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                async with semaphore:
                    completion = await client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                        timeout=self.timeout,
                        extra_body=extra_body,
                    )
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                asyncio.TimeoutError,
            ) as e:
                last_err = e
                logger.warning(
                    "Qwen-Image-Bench judge transport error (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt + 1 >= self.max_retries:
                    break
                await asyncio.sleep(2**attempt)
                continue

            content = completion.choices[0].message.content
            if content is None or not str(content).strip():
                logger.warning("Qwen-Image-Bench judge returned empty content.")
                return None
            return str(content)

        logger.warning(
            "Qwen-Image-Bench judge failed after %s attempt(s); last error: %s",
            self.max_retries,
            last_err,
        )
        return None

    # ============================== Parsing / aggregation ==============================

    @staticmethod
    def _parse_dimension(output: Optional[str], l1_dim: str) -> Optional[dict]:
        """Parse one L1 dimension's judge reply into a ``compute_dimension_score`` result."""
        if output is None:
            return None
        score_json = extract_json_from_response(output)
        if score_json is None:
            return None
        return compute_dimension_score(fix_score_json(score_json, l1_dim))

    @staticmethod
    def _parse_single_call(output: Optional[str], l1_dims: List[str]) -> Dict[str, dict]:
        """Parse a single L1-keyed judge reply into per-dimension score results."""
        if output is None:
            return {}
        top = extract_json_from_response(output)
        if not isinstance(top, dict):
            return {}
        dim_results: Dict[str, dict] = {}
        for l1 in l1_dims:
            sub = top.get(l1)
            if isinstance(sub, dict):
                dim_results[l1] = compute_dimension_score(fix_score_json(sub, l1))
        return dim_results

    def _aggregate(self, dim_results: Dict[str, dict]) -> float:
        """Reduce per-dimension results to a single reward in [0, 1].

        Returns 0.0 (with a warning) when the judge produced no usable score, so
        a single bad/parse-failed reply does not crash a long RL run -- the same
        documented-intentional degradation as ``rational_rewards_t2i``.
        """
        if not dim_results:
            logger.warning("Qwen-Image-Bench: no parseable judge output; reward=0.0")
            return 0.0

        if self.score_dimension == "total":
            total = aggregate_total_score(dim_results)
            if total is None:
                logger.warning("Qwen-Image-Bench: total score is None; reward=0.0")
                return 0.0
            return max(0.0, min(1.0, total / 100.0))

        result = dim_results.get(self.score_dimension)
        if result is None or result.get("level1_score") is None:
            logger.warning(
                "Qwen-Image-Bench: score_dimension %r unavailable; reward=0.0",
                self.score_dimension,
            )
            return 0.0
        return max(0.0, min(1.0, result["level1_score"] / 100.0))

    # ============================== Helpers ==============================

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Match the judge's input: RGB, downscaled to a square ``max_image_size``.

        Upstream Qwen-Image-Bench resizes to a square ``max_image_size`` when the
        longest side exceeds it; replicated here so inputs stay in-distribution.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        if max(image.size) > self.max_image_size:
            image = image.resize(
                (self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS
            )
        return image

    @staticmethod
    def _build_content(user_text: str, image_data_url: str) -> List[dict]:
        """Build OpenAI multimodal content, placing the image at the ``<image>`` marker."""
        before, _, after = user_text.partition("<image>")
        content: List[dict] = [{"type": "text", "text": before}]
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        if after:
            content.append({"type": "text", "text": after})
        return content
