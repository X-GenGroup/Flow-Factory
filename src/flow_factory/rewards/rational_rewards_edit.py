# Copyright 2026 Jayce-Ping, Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rational Rewards for image editing: VLM rubric over OpenAI-compatible HTTP.

Uses source (condition) and edited images, parses four aspect scores from the
structured reply, averages selected aspects, maps mean from [1, 4] to [0, 1].
See the Rational Rewards (image edit) section in ``guidance/rewards.md``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from .rational_rewards_t2i import (
    _clip_vlm_text_for_log,
    aggregate_aspect_scores,
    extract_numeric_score,
)
from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

SUPPORTED_ASPECTS: Tuple[str, ...] = (
    "text_faithfulness",
    "image_faithfulness",
    "physical_quality",
    "text_rendering",
)

RATIONAL_EDIT_SYSTEM_PROMPT = (
    "You are an expert image editing evaluator. Your task is to evaluate the quality of an edited "
    "image based on a source image and a user instruction. Afterwards, you need to suggest how to "
    "refine the original user request to produce better image edits (if any)."
)

COMMON_TASK_GUIDELINE = """To do this, you must first assess the image on four critical aspects, provide justifications and absolute scores in 1-4 scale.

### Critical Aspects & Scoring Rubric
**1. Text Faithfulness** (How accurately does the output follow the instruction?)
- **4 (Full match):** All key elements (objects, colors, actions) are represented exactly as described. No hallucinations or unrequested changes.
- **3 (Minor mismatch):** Most key elements are present, but minor details are missing, incorrect, or slightly inaccurate.
- **2 (Some mismatch):** Some key elements are missing, altered, or interpreted incorrectly.
- **1 (Major deviations):** Key elements are completely missing, altered, or contradicted. Instruction is ignored.

**2. Image Faithfulness** (How well are the non-edited parts and key input elements preserved?)
- **4 (Uses input fully):** All relevant elements from the input (background, style, lighting, identity) are accurately preserved or transformed as instructed.
- **3 (Minor mismatch):** Most relevant elements are preserved, but a few aspects (e.g., background details, lighting consistency) are missing or incorrectly handled.
- **2 (Partial mismatch):** Some elements are carried over, but key aspects of the original image are lost or distorted.
- **1 (Fails to use input):** Key elements of the input image are ignored, misinterpreted, or destroyed.

**3. Physical and Visual Quality** (Technical errors, composition, realism, and physics)
- **4 (No noticeable flaws):** The image is physically plausible (correct lighting, shadows, geometry, anatomy). No visible artifacts (seams, blurring, noise).
- **3 (Minor flaws):** Small inaccuracies that are noticeable but not strongly disruptive (e.g., slight lighting mismatch, minor texture issues).
- **2 (Some flaws):** Clear physical or visual errors that disrupt the image (e.g., incorrect perspective, "floating" objects, wrong shadow direction, obvious seams).
- **1 (Severe flaws):** Major physical/visual errors (e.g., impossible geometry, distorted anatomy, garbled objects, severe artifacts).

**4. Text Rendering** (Only if the instruction involves generating text)
- **4 (Full match):** Text is correct, legible, and integrated well.
- **3 (Mostly match):** Minor misspellings or inconsistent capitalization.
- **2 (Partial match):** Major misspellings or distorted text.
- **1 (Major deviations):** Text is unreadable, severely distorted, or missing. (Use N/A if no text generation is required).

### Scoring Methodology (CRITICAL)
During assessment for each aspect, recall the initial user request, source image and the scoring rubrics of the aspect, provide scores with detailed justifications for each image and reflect fine-grained preferences.
1. **Anchor:** Have a global inspection based on the user request and the resulting generation. Determine the rough integer score level (1, 2, 3, or 4) according to the definitions provided .
2. **Justify and Adjust:** Do careful visual analysis and identify specific flaws in generation. Justify the score with concrete evidence and scoring logic. Fine-tune this anchor score into a float value. Add small increments for exceptional execution or deduct points for specific flaws.
   - *Example:* deduct points from 4.0 for slight flaws if the assessed dimension is close to satisfaction. add increments from 1.0 or 2.0 based on severity of flaws.

Afterwards, try to construct a refined user request that helps the visual generation model to produce better image edits.
Think of the weaknesses identified in the judgement, then map them to instruction details and apply specific fixes.
Provide a final new user request that enrich the initial user request.

Output your evaluation in the following format:
# User Request Analysis
[ understanding the user request, try to analyze or decompose the user request deeper. Think of what the request might imply or what needs to be inferred to successfully execute the request. ]
# Detailed Judgement
1. Text Faithfulness:
## Justification: [ Analysis of the user request and the assessment of the resulting generation. How it comes to a final score. ]
## Score: [ float score ]
2. Image Faithfulness:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score ]
3. Physical and Visual Quality:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score ]
4. Text Rendering:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score or N/A ]
# Summary: [ Summary of the evaluation ]

# User Request Refinement:
## Refinement Comments: [Specific suggestions for improving the user request]
## Refined Request: [The improved, more specific user request for editing like a standard user instruction]"""


def _extract_score_from_block(block_text: str) -> Optional[Union[float, str]]:
    for line in block_text.split("\n"):
        stripped = line.strip()
        match = re.search(r"(?:##\s*)?Score\s*:\s*(.+)$", stripped, re.IGNORECASE)
        if not match:
            continue
        raw_val = match.group(1).strip()
        try:
            return extract_numeric_score(raw_val)
        except (TypeError, ValueError):
            continue
    return None


def parse_scores_from_detailed_judgement_edit(
    detailed_judgement: str,
) -> Dict[str, Optional[Union[float, str]]]:
    """
    Parse four aspect scores from the ``# Detailed Judgement`` section (edit rubric).
    """
    result: Dict[str, Optional[Union[float, str]]] = {
        "text_faithfulness": None,
        "image_faithfulness": None,
        "physical_quality": None,
        "text_rendering": None,
    }

    content_body = detailed_judgement
    if "# Summary:" in detailed_judgement:
        parts = detailed_judgement.split("# Summary:")
        if len(parts) > 1:
            content_body = parts[0]

    lines = content_body.split("\n")
    section_blocks: Dict[str, str] = {}
    current_section: Optional[str] = None
    current_block: List[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("1.") and "Text Faithfulness" in stripped:
            if current_section:
                section_blocks[current_section] = "\n".join(current_block)
            current_section = "text_faithfulness"
            current_block = [raw_line]
        elif stripped.startswith("2.") and "Image Faithfulness" in stripped:
            if current_section:
                section_blocks[current_section] = "\n".join(current_block)
            current_section = "image_faithfulness"
            current_block = [raw_line]
        elif stripped.startswith("3.") and "Physical and Visual Quality" in stripped:
            if current_section:
                section_blocks[current_section] = "\n".join(current_block)
            current_section = "physical_quality"
            current_block = [raw_line]
        elif stripped.startswith("4.") and "Text Rendering" in stripped:
            if current_section:
                section_blocks[current_section] = "\n".join(current_block)
            current_section = "text_rendering"
            current_block = [raw_line]
        elif current_section:
            current_block.append(raw_line)

    if current_section:
        section_blocks[current_section] = "\n".join(current_block)

    if not section_blocks:
        h1 = "Text Faithfulness:"
        h2 = "Image Faithfulness:"
        h3 = "Physical and Visual Quality:"
        h4 = "Text Rendering:"
        if h1 in content_body:
            _, _, rest = content_body.partition(h1)
            block_tf, _, rest = rest.partition(h2) if h2 in rest else (rest, "", "")
            block_if, _, rest = rest.partition(h3) if h3 in rest else (rest, "", "")
            block_pq, _, rest = rest.partition(h4) if h4 in rest else (rest, "", "")
            block_tr = rest
            section_blocks = {
                "text_faithfulness": block_tf,
                "image_faithfulness": block_if,
                "physical_quality": block_pq,
                "text_rendering": block_tr,
            }

    for key, block_text in section_blocks.items():
        extracted = _extract_score_from_block(block_text)
        if extracted is not None:
            result[key] = extracted

    return result


def build_scoring_user_content_edit(
    prompt: str,
    source_image_data_url: str,
    edited_image_data_url: str,
) -> List[dict]:
    # Assemble text and images explicitly so user ``prompt`` may contain ``<image>`` without
    # breaking a sentinel-based split of the full rubric.
    head = f"User Instruction: {prompt}\nYou are provided with two images:\n1. Source Image "
    between_images = "\n2. Edited Image "
    after_images = (
        "\n\nGive your analysis and judgement following guidelines in the system prompt. \n\n"
        + COMMON_TASK_GUIDELINE
    )
    return [
        {"type": "text", "text": head},
        {"type": "image_url", "image_url": {"url": source_image_data_url}},
        {"type": "text", "text": between_images},
        {"type": "image_url", "image_url": {"url": edited_image_data_url}},
        {"type": "text", "text": after_images},
    ]


def build_scoring_messages_edit(
    prompt: str,
    source_image_data_url: str,
    edited_image_data_url: str,
) -> List[dict]:
    return [
        {"role": "system", "content": RATIONAL_EDIT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_scoring_user_content_edit(
                prompt, source_image_data_url, edited_image_data_url
            ),
        },
    ]


def _first_condition_image(cond: Union[Image.Image, List[Image.Image]]) -> Image.Image:
    if isinstance(cond, list):
        if not cond:
            raise ValueError("condition_images entry is an empty list; need at least one source image")
        first = cond[0]
        if not isinstance(first, Image.Image):
            raise TypeError(
                f"expected PIL.Image.Image inside condition_images list, got {type(first).__name__}"
            )
        return first
    if isinstance(cond, Image.Image):
        return cond
    raise TypeError(
        f"expected PIL.Image.Image or list of PIL images for condition_images element, "
        f"got {type(cond).__name__}"
    )


class RationalRewardsEditRewardModel(PointwiseRewardModel):
    """
    Pointwise image-edit reward via remote VLM (OpenAI-compatible chat completions).

    ``extra_kwargs`` match ``RationalRewardsT2IRewardModel`` (``api_base_url``, ``api_key``,
    ``vlm_model`` — default ``RationalRewards-8B-Edit`` (match vLLM ``--served-model-name``), concurrency, retries, timeout,
    generation params) plus optional
    ``aspects`` (subset of the four supported aspect keys).
    """

    required_fields = ("prompt", "image", "condition_images")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "RationalRewardsEditRewardModel requires the `openai` package. "
                "Install with: pip install openai"
            ) from e

        self.api_base_url = config.extra_kwargs.get("api_base_url", "http://localhost:8000/v1")
        self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
        self.vlm_model = config.extra_kwargs.get("vlm_model", "RationalRewards-8B-Edit")
        self.max_concurrent = int(config.extra_kwargs.get("max_concurrent", 8))
        self.max_retries = int(config.extra_kwargs.get("max_retries", 5))
        self.timeout = float(config.extra_kwargs.get("timeout", 180.0))
        self.temperature = float(config.extra_kwargs.get("temperature", 0.1))
        self.max_tokens = int(config.extra_kwargs.get("max_tokens", 2048))

        raw_aspects = config.extra_kwargs.get("aspects")
        if raw_aspects is None:
            self.aspects: Tuple[str, ...] = SUPPORTED_ASPECTS
        else:
            if not isinstance(raw_aspects, (list, tuple)) or not raw_aspects:
                raise TypeError(
                    f"expected non-empty list/tuple for aspects, got {type(raw_aspects).__name__}: "
                    f"{raw_aspects!r}"
                )
            self.aspects = tuple(str(a) for a in raw_aspects)
        unknown = [a for a in self.aspects if a not in SUPPORTED_ASPECTS]
        if unknown:
            raise ValueError(
                f"unsupported aspect(s) {unknown!r}; allowed: {list(SUPPORTED_ASPECTS)}"
            )

        self.client = AsyncOpenAI(base_url=self.api_base_url, api_key=self.api_key)
        self.semaphore = asyncio.Semaphore(max(1, self.max_concurrent))

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
        if image is None and video is not None:
            image = [frames[0] for frames in video]
        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided for RationalRewardsEdit")
        if condition_images is None:
            raise ValueError("condition_images (source) is required for RationalRewardsEdit")
        if len(prompt) != len(image) or len(prompt) != len(condition_images):
            raise ValueError(
                f"expected len(prompt)==len(image)==len(condition_images), got "
                f"{len(prompt)}, {len(image)}, {len(condition_images)}"
            )

        source_images = [_first_condition_image(c) for c in condition_images]
        scores = asyncio.run(self._async_score_batch(prompt, source_images, image))
        rewards = torch.tensor(scores, dtype=torch.float32, device=self.device)
        return RewardModelOutput(rewards=rewards, extra_info={})

    async def _async_score_batch(
        self,
        prompts: List[str],
        sources: List[Image.Image],
        edited: List[Image.Image],
    ) -> List[float]:
        tasks = [self._score_single(p, s, e) for p, s, e in zip(prompts, sources, edited)]
        return list(await asyncio.gather(*tasks))

    async def _score_single(self, prompt: str, source: Image.Image, edited: Image.Image) -> float:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        source_url = pil_image_to_base64(source, format="PNG")
        edited_url = pil_image_to_base64(edited, format="PNG")
        messages = build_scoring_messages_edit(prompt, source_url, edited_url)

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
            except (APIConnectionError, APITimeoutError, RateLimitError, asyncio.TimeoutError) as e:
                last_err = e
                logger.warning(
                    "RationalRewardsEdit API transport error (attempt %s/%s): %s",
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
                logger.warning(
                    "RationalRewardsEdit VLM returned empty assistant content; using reward 0.0"
                )
                return 0.0
            text = str(content)
            try:
                parsed = parse_scores_from_detailed_judgement_edit(text)
                return aggregate_aspect_scores(
                    parsed,
                    self.aspects,
                    supported_aspects=SUPPORTED_ASPECTS,
                )
            except (TypeError, ValueError) as e:
                logger.warning(
                    "RationalRewardsEdit failed to parse or aggregate VLM response; using reward 0.0: %s. "
                    "Response (truncated): %s",
                    e,
                    _clip_vlm_text_for_log(text),
                )
                return 0.0

        logger.warning(
            "RationalRewardsEdit HTTP request failed after %s attempt(s); using reward 0.0. Last error: %s",
            self.max_retries,
            last_err,
        )
        return 0.0
