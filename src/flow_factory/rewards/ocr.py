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
"""
OCR Reward Model using PP-OCRv5.
Some instructions for installation on CUDA 12.9:
```bash
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddleocr
pip install python-Levenshtein
# Install torch2.8.0 and it will update nvcc toolkits automatically
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
# Maybe you will need this:
yum install -y mesa-libGL glib2
```
For other versions of CUDA, please refer to the official documentation of PaddleOCR.
"""
from typing import Optional
from accelerate import Accelerator
from PIL import Image
import torch
import numpy as np

from .abc import PointwiseRewardModel, GroupwiseRewardModel, RewardModelOutput
from ..hparams import *
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError("paddleocr is required for OCR reward. Install with: pip install paddleocr")

try:
    from Levenshtein import distance
except ImportError:
    raise ImportError("python-Levenshtein is required for OCR reward. Install with: pip install python-Levenshtein")

class OCRRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "image", "video")
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        device_index = self.accelerator.local_process_index

        # Initialize PP-OCRv5 reader with new API
        self.model = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=f"gpu:{device_index}" if "cuda" in str(self.device) else "cpu"
        )

    def _compute_scores_batch(
        self,
        prompt: list[str],
        image: list[Image.Image],
    ) -> torch.Tensor:
        """Compute OCR reward for a batch of image-prompt pairs."""
        rewards = []
        for img, p in zip(image, prompt):
            # Convert image format to np.ndarray
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition using PP-OCRv5 predict API
                result = self.model.predict(img)
                # Extract recognized text from PP-OCRv5 result
                recognized_text = ''
                for res in result:
                    recognized_text += ''.join(res['rec_texts'])

                recognized_text = recognized_text.replace(' ', '').lower()
                p = p.replace(' ', '').lower()
                if p in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, p)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(p):
                    dist = len(p)

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                logger.error(f"OCR processing failed: {str(e)}")
                dist = len(p)  # Maximum penalty
            
            reward = 1 - dist / (len(p))
            rewards.append(reward)

        return rewards

    def _compute_video_scores(
        self,
        prompt: list[str],
        video: list[list[Image.Image]],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute mean PickScore across all frames for each video.
        
        Uses flat-reconstruct strategy to handle variable frame counts
        while maintaining efficient batched computation.
        """
        # Flatten: expand prompts and images per frame count
        frame_counts = [len(clip) for clip in video]
        flat_images = [frame for clip in video for frame in clip]
        flat_prompts = [p for p, n in zip(prompt, frame_counts) for _ in range(n)]
        
        # Batched score computation
        all_scores = []
        for i in range(0, len(flat_images), batch_size):
            batch_scores = self._compute_scores_batch(
                flat_prompts[i:i + batch_size],
                flat_images[i:i + batch_size],
            )
            all_scores.append(batch_scores)
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
    ) -> RewardModelOutput:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if image is not None and video is not None:
            raise ValueError("Only one of image or video can be provided.")
        
        batch_size = getattr(self.config, 'batch_size', len(prompt))
        
        if video is not None:
            scores = self._compute_video_scores(prompt, video, batch_size)
        else:
            scores = self._compute_scores_batch(prompt, image)
        
        return RewardModelOutput(rewards=scores, extra_info={})

def download_model():
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    logger.info('PaddleOCR initialized successfully')

if __name__ == "__main__":
    download_model()