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

# src/flow_factory/rewards/ocr.py
"""OCR Reward Model using PP-OCRv5.

Some instructions for installation on CUDA 12.9:
```bash
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddleocr
pip install python-Levenshtein
# Install the project baseline PyTorch stack and its audio codec runtime
pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchcodec==0.10.0 \
  --index-url https://download.pytorch.org/whl/cu129
# Maybe you will need this:
yum install -y mesa-libGL glib2
```
For other versions of CUDA, please refer to the official documentation of PaddleOCR.
"""

import json
import re
from typing import Any, Optional

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import *
from ..utils.logger_utils import setup_logger
from .abc import GroupwiseRewardModel, PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__)
_QUOTED_TEXT = re.compile(r'["“](.*?)["”]')

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError("paddleocr is required for OCR reward. Install with: pip install paddleocr")

try:
    from Levenshtein import distance
except ImportError:
    raise ImportError(
        "python-Levenshtein is required for OCR reward. Install with: pip install python-Levenshtein"
    )


class OCRRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "image", "metadata")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        device_index = self.accelerator.local_process_index
        use_cuda = "cuda" in str(self.device)
        # The CPU Paddle build in the Flow-Factory environment cannot execute
        # PaddleOCR's oneDNN PIR graph (ConvertPirAttribute2RuntimeAttribute).
        # Keep oneDNN disabled for CPU rewards; CUDA builds do not use this path.
        self.model = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=f"gpu:{device_index}" if use_cuda else "cpu",
            enable_mkldnn=use_cuda,
        )

    def _compute_scores_batch(
        self,
        prompt: list[str],
        image: list[Image.Image],
        metadata: Optional[list[str]] = None,
    ) -> list[float]:
        """Compute mean target-text fidelity for each image."""
        if len(prompt) != len(image) or (metadata is not None and len(prompt) != len(metadata)):
            raise ValueError(
                "expected equal OCR batch lengths for prompt and image, with optional "
                "metadata matching that length; "
                f"received prompt={len(prompt)}, image={len(image)}, "
                f"metadata={None if metadata is None else len(metadata)}"
            )

        rewards: list[float] = []
        for sample_index, (sample_prompt, img) in enumerate(zip(prompt, image)):
            if isinstance(img, Image.Image):
                img = np.array(img)
            sample_metadata = None if metadata is None else metadata[sample_index]
            targets = self._targets_for_sample(sample_prompt, sample_metadata, sample_index)

            result = self.model.predict(img)
            rec_texts: list[str] = []
            for res in result:
                rec_texts.extend(res["rec_texts"])

            target_scores = [self._target_similarity(target, rec_texts) for target in targets]
            rewards.append(float(np.mean(target_scores)))

        return rewards

    @classmethod
    def _targets_for_sample(
        cls,
        prompt: str,
        metadata: Optional[str],
        sample_index: int,
    ) -> list[str]:
        """Prefer explicit metadata; otherwise extract quoted text from the prompt."""
        if metadata is not None:
            parsed = cls._parse_metadata(metadata, sample_index)
            if "visible_texts" in parsed:
                return cls._targets_from_metadata(metadata, sample_index)
        return cls._targets_from_prompt(prompt, sample_index)

    @staticmethod
    def _targets_from_prompt(prompt: str, sample_index: int) -> list[str]:
        """Extract nonempty straight- or curly-quoted OCR targets."""
        if not isinstance(prompt, str):
            raise TypeError(
                f"expected string prompt for OCR sample {sample_index}, "
                f"received {type(prompt).__name__}: {prompt!r}"
            )
        targets = [target.strip() for target in _QUOTED_TEXT.findall(prompt) if target.strip()]
        if not targets:
            raise ValueError(
                f"expected quoted OCR target in prompt for sample {sample_index}, "
                f"received {prompt!r}"
            )
        return targets

    @staticmethod
    def _targets_from_metadata(metadata: str, sample_index: int) -> list[str]:
        parsed = OCRRewardModel._parse_metadata(metadata, sample_index)
        targets: Any = parsed.get("visible_texts")
        if isinstance(targets, str):
            targets = json.loads(targets)
        if not isinstance(targets, list) or any(
            not isinstance(target, str) or not target.strip() for target in targets
        ):
            raise ValueError(
                f"expected nonempty metadata.visible_texts list[str] for OCR sample "
                f"{sample_index}, received {targets!r}"
            )
        if not targets:
            raise ValueError(
                f"expected at least one metadata.visible_texts target for OCR sample "
                f"{sample_index}, received an empty list"
            )
        return targets

    @staticmethod
    def _parse_metadata(metadata: str, sample_index: int) -> dict[str, Any]:
        """Parse one metadata object with sample-local diagnostics."""
        if not isinstance(metadata, str):
            raise TypeError(
                f"expected JSON string metadata for OCR sample {sample_index}, "
                f"received {type(metadata).__name__}: {metadata!r}"
            )
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"expected JSON object metadata for OCR sample {sample_index}, "
                f"received invalid JSON: {metadata!r}"
            ) from error
        if not isinstance(parsed, dict):
            raise TypeError(
                f"expected metadata object for OCR sample {sample_index}, "
                f"received {type(parsed).__name__}: {parsed!r}"
            )
        return parsed

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(character for character in text.lower() if not character.isspace())

    @classmethod
    def _target_similarity(cls, target: str, recognized_parts: list[str]) -> float:
        normalized_target = cls._normalize_text(target)
        normalized_parts = [
            normalized for part in recognized_parts if (normalized := cls._normalize_text(part))
        ]
        if not normalized_parts:
            return 0.0

        recognized_text = "".join(normalized_parts)
        if normalized_target in recognized_text:
            return 1.0

        candidates = [*normalized_parts, recognized_text]
        normalized_distance = min(
            distance(normalized_target, candidate) / max(len(normalized_target), len(candidate))
            for candidate in candidates
        )
        return max(0.0, 1.0 - normalized_distance)

    def _compute_video_scores(
        self,
        prompt: list[str],
        video: list[list[Image.Image]],
        metadata: Optional[list[str]],
        batch_size: int,
    ) -> torch.Tensor:
        """Compute mean OCR score across all frames for each video.

        Uses flat-reconstruct strategy to handle variable frame counts
        while maintaining efficient batched computation.
        """
        # Flatten: expand prompts and images per frame count
        frame_counts = [len(clip) for clip in video]
        flat_images = [frame for clip in video for frame in clip]
        flat_prompts = [p for p, n in zip(prompt, frame_counts) for _ in range(n)]
        flat_metadata = (
            None
            if metadata is None
            else [m for m, n in zip(metadata, frame_counts) for _ in range(n)]
        )

        # Batched score computation
        all_scores = []
        for i in range(0, len(flat_images), batch_size):
            batch_scores = self._compute_scores_batch(
                flat_prompts[i : i + batch_size],
                flat_images[i : i + batch_size],
                None if flat_metadata is None else flat_metadata[i : i + batch_size],
            )
            all_scores.append(torch.tensor(batch_scores, dtype=torch.float32))
        flat_scores = torch.cat(all_scores, dim=0)

        # Reconstruct: mean pooling per video
        scores = flat_scores.split(frame_counts)
        scores = torch.stack([s.mean() for s in scores])
        return scores

    @torch.no_grad()
    def __call__(
        self,
        prompt: list[str],
        image: Optional[list[Image.Image]] = None,
        video: Optional[list[list[Image.Image]]] = None,
        metadata: Optional[list[str]] = None,
    ) -> RewardModelOutput:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if image is not None and video is not None:
            raise ValueError("Only one of image or video can be provided.")
        if image is None and video is None:
            raise ValueError("OCR reward requires image or video input, received neither.")
        batch_size = getattr(self.config, "batch_size", len(prompt))

        if video is not None:
            scores = self._compute_video_scores(prompt, video, metadata, batch_size)
        else:
            scores = self._compute_scores_batch(prompt, image, metadata)

        return RewardModelOutput(rewards=scores, extra_info={})


def download_model():
    ocr = PaddleOCR(
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False
    )
    logger.info("PaddleOCR initialized successfully")


if __name__ == "__main__":
    download_model()
