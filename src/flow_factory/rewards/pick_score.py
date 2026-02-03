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

# src/flow_factory/rewards/pick_score.py
from typing import Optional
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

from .abc import PointwiseRewardModel, GroupwiseRewardModel, RewardModelOutput
from ..hparams import *


class PickScoreRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "image", "video")
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    def _compute_scores_batch(
        self,
        prompt: list[str],
        image: list[Image.Image],
    ) -> torch.Tensor:
        """Compute PickScore for a batch of image-prompt pairs."""
        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs * image_embs).sum(dim=-1)
        return scores

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
        
        # Normalize to 0-1 range
        scores = scores / 26
        
        return RewardModelOutput(rewards=scores, extra_info={})


class PickScoreRankRewardModel(GroupwiseRewardModel):
    """Ranking-based reward model using PickScore with video support."""
    
    required_fields = ("prompt", "image", "video")
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    def _compute_scores_batch(
        self,
        prompt: list[str],
        image: list[Image.Image],
    ) -> torch.Tensor:
        """Compute PickScore for a batch of image-prompt pairs."""
        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs * image_embs).sum(dim=-1)
        return scores

    def _compute_video_scores(
        self,
        prompt: list[str],
        video: list[list[Image.Image]],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute mean PickScore across all frames for each video.
        
        Uses flat-reconstruct strategy for efficient batched computation
        with variable-length frame sequences.
        """
        frame_counts = [len(clip) for clip in video]
        flat_images = [frame for clip in video for frame in clip]
        flat_prompts = [p for p, n in zip(prompt, frame_counts) for _ in range(n)]
        
        all_scores = []
        for i in range(0, len(flat_images), batch_size):
            batch_scores = self._compute_scores_batch(
                flat_prompts[i:i + batch_size],
                flat_images[i:i + batch_size],
            )
            all_scores.append(batch_scores)
        flat_scores = torch.cat(all_scores, dim=0)
        
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
        group_size = len(prompt)
        batch_size = self.config.batch_size
        
        if video is not None:
            raw_scores = self._compute_video_scores(prompt, video, batch_size)
        else:
            all_scores = []
            for i in range(0, group_size, batch_size):
                batch_scores = self._compute_scores_batch(
                    prompt[i:i + batch_size],
                    image[i:i + batch_size],
                )
                all_scores.append(batch_scores)
            raw_scores = torch.cat(all_scores, dim=0)
        
        # Rank-based rewards: (0, 1, ..., n-1) / n
        ranks = raw_scores.argsort().argsort()
        rewards = ranks.float() / group_size
        
        return RewardModelOutput(rewards=rewards, extra_info={})


def download_model():
    scorer = PickScoreRewardModel(RewardArguments(device='cpu'), accelerator=None)

if __name__ == "__main__":
    download_model()