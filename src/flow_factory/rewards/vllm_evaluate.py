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

# src/flow_factory/rewards/vllm_evaluate.py
"""
A Simple VLM Evaluate Reward Model.

Evaluates image quality by querying a VLM (e.g., Qwen3-VL) via
OpenAI-compatible API with a simple Yes/No question, then extracting
P(Yes) / (P(Yes) + P(No)) from the VLM's logprobs as the reward.

Usage in YAML config:
    rewards:
      - name: "vllm_evaluate"
        reward_model: "vllm_evaluate"
        batch_size: 8
        # Extra kwargs passed via config.extra_kwargs:
        api_base_url: "http://localhost:8000/v1"
        api_key: "EMPTY"
        vlm_model: "Qwen3-VL"
        max_concurrent: 100
        max_retries: 10
        timeout: 60
        top_logprobs: 20
"""
from __future__ import annotations

import hashlib
import asyncio
import logging
from typing import Optional, List

import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64

logger = logging.getLogger(__name__)

# Suppress verbose HTTP/retry logs from openai client and httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# =====================================================================
# Helper functions
# =====================================================================

def _get_yes_cond_prob(completion, canonicalize: bool = False) -> float:
    """
    Extract P(Yes) / (P(Yes) + P(No)) from a VLM completion's logprobs.

    Args:
        completion: OpenAI-compatible ChatCompletion response.
        canonicalize: If True, aggregate probabilities for all case
            variations of "yes" and "no" (e.g., "Yes", "yes", "YES").

    Returns:
        Conditional probability of "Yes". Returns 0.0 on failure.
    """
    if completion is None:
        return 0.0

    logprobs = completion.choices[0].logprobs
    if not logprobs:
        return 0.0

    if not canonicalize:
        token_logprobs = {
            t.token: t.logprob for t in logprobs.content[0].top_logprobs
        }
        yes_logprob = token_logprobs.get('Yes', float('-inf'))
        no_logprob = token_logprobs.get('No', float('-inf'))

        if yes_logprob == float('-inf') and no_logprob == float('-inf'):
            return 0.0

        diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
        return torch.sigmoid(diff).item()
    else:
        # Aggregate all case variations
        token_probs = {
            t.token: np.exp(t.logprob) for t in logprobs.content[0].top_logprobs
        }
        tokens = np.array(list(token_probs.keys()))
        probs = np.array(list(token_probs.values()), dtype=np.float64)
        tokens_stripped = np.array([token.strip().lower() for token in tokens])

        yes_prob_sum = probs[tokens_stripped == "yes"].sum()
        no_prob_sum = probs[tokens_stripped == "no"].sum()
        total = yes_prob_sum + no_prob_sum

        if total == 0.0:
            return 0.0
        return float(yes_prob_sum / total)


# =====================================================================
# VLMEvaluateRewardModel
# =====================================================================

class VLMEvaluateRewardModel(PointwiseRewardModel):
    """
    VLM-based image evaluation reward model.

    For each image, the model sends it to a VLM with a comprehensive
    quality assessment prompt covering naturalness, artifacts, aesthetic
    appeal, detail/clarity, and overall coherence. The VLM is asked to
    answer Yes/No, and the reward is P(Yes) / (P(Yes) + P(No)) derived
    from the VLM's logprobs.

    Extra kwargs (passed via YAML config):
        api_base_url (str): Base URL for the OpenAI-compatible API.
            Default: "http://localhost:8000/v1"
        api_key (str): API key. Default: "EMPTY"
        vlm_model (str): VLM model name. Default: "Qwen3-VL"
        max_concurrent (int): Max concurrent API requests. Default: 100
        max_retries (int): Max retries per API call. Default: 10
        timeout (int): Timeout in seconds per API call. Default: 60
        top_logprobs (int): Number of top logprobs to request. Default: 20
        canonicalize (bool): Aggregate Yes/No case variants. Default: False
        max_cache_size (int): Max LRU cache entries. Default: 1024
    """

    required_fields = ("prompt", "image")
    use_tensor_inputs = False

    # Comprehensive evaluation prompt covering multiple quality dimensions,
    # while still requiring a Yes/No answer for logprob-based scoring.
    EVALUATE_PROMPT = (
        "You are an expert image quality assessor. "
        "Evaluate this AI-generated image by considering ALL of the following criteria:\n"
        "1. Naturalness: Does the scene look realistic with correct perspective, shadows, and lighting?\n"
        "2. Artifacts: Is the image free from distortions, blurriness, watermarks, "
        "deformed faces, unusual body parts, or unharmonized subjects?\n"
        "3. Aesthetic Appeal: Does the image exhibit pleasing composition, "
        "color harmony, and visual balance?\n"
        "4. Detail & Clarity: Are textures, edges, and fine details rendered "
        "sharply and coherently without noise or smearing?\n"
        "5. Overall Coherence: Is the image semantically consistent, with all "
        "elements logically fitting together in a unified scene?\n\n"
        "Considering all the above criteria holistically, is this a high-quality image? "
        "Answer Yes or No."
    )

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "VLMEvaluateRewardModel requires the `openai` package. "
                "Install it with: pip install openai"
            )

        # Read extra kwargs with defaults
        self.api_base_url = config.extra_kwargs.get("api_base_url", "http://localhost:8000/v1")
        self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
        self.vlm_model = config.extra_kwargs.get("vlm_model", "Qwen3-VL")
        self.max_concurrent = config.extra_kwargs.get("max_concurrent", 100)
        self.max_retries = config.extra_kwargs.get("max_retries", 10)
        self.timeout = config.extra_kwargs.get("timeout", 60)
        self.top_logprobs = config.extra_kwargs.get("top_logprobs", 20)
        self.canonicalize = config.extra_kwargs.get("canonicalize", False)
        self.max_cache_size = config.extra_kwargs.get("max_cache_size", 1024)

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key,
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # Simple FIFO cache: img_hash -> score
        self._cache: dict[str, float] = {}

    def _add_to_cache(self, key: str, value: float):
        """Add entry to cache with FIFO eviction."""
        if len(self._cache) >= self.max_cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """
        Compute VLM evaluation rewards for a batch of images.

        Args:
            prompt: List of text prompts (not used for evaluation question,
                but kept for interface compatibility).
            image: List of generated images.
            video: Not used; falls back to first frame if image is None.

        Returns:
            RewardModelOutput with per-sample scores in [0, 1].
        """
        # Handle video input (use first frame)
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided")

        assert len(prompt) == len(image), (
            f"Mismatch: {len(prompt)} prompts vs {len(image)} images"
        )

        # Run async scoring
        rewards = asyncio.run(self._async_score_batch(image))

        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32),
            extra_info={},
        )

    async def _async_score_batch(
        self,
        images: List[Image.Image],
    ) -> List[float]:
        """Score all images in the batch concurrently."""
        tasks = [self._score_single(img) for img in images]
        return list(await asyncio.gather(*tasks))

    async def _score_single(
        self,
        image: Image.Image,
    ) -> float:
        """
        Query the VLM for a single image and return P(Yes|Yes,No).

        Uses caching to avoid redundant API calls and retries on failure.
        """
        cache_key = hashlib.md5(image.tobytes()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": self.EVALUATE_PROMPT},
                ],
            }
        ]

        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=0.0,
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=self.top_logprobs,
                        timeout=self.timeout,
                    )

                score = _get_yes_cond_prob(completion, canonicalize=self.canonicalize)
                self._add_to_cache(cache_key, score)
                return score

            except Exception as e:
                logger.warning(
                    f"VLM API error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        # All retries exhausted, return default score (do not cache failures)
        return 0.0
