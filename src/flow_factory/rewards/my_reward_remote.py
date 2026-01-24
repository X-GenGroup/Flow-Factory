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

# src/flow_factory/rewards/my_reward_remote.py

"""
Remote Reward Model Client

Communicates with an external reward server via HTTP.
Enables using reward models in isolated environments with different dependencies.

Usage:
    rewards:
      - name: "my_reward"
        reward_model: "flow_factory.rewards.remote.RemotePointwiseRewardModel"
        server_url: "http://localhost:8000"
        batch_size: 16
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List, Optional, Tuple, Union

import requests
import torch
from PIL import Image

from flow_factory.rewards.abc import (
    GroupwiseRewardModel,
    PointwiseRewardModel,
    RewardModelOutput,
)
from flow_factory.hparams import RewardArguments
from accelerate import Accelerator

logger = logging.getLogger(__name__)


# ======================== Serialization Helpers ========================

def _image_to_b64(img: Union[Image.Image, torch.Tensor]) -> str:
    """Convert PIL Image or Tensor to base64 string."""
    if isinstance(img, torch.Tensor):
        # (C, H, W) in [0, 1] -> PIL
        arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _video_to_b64(video: Union[List[Image.Image], torch.Tensor]) -> List[str]:
    """Convert video frames to list of base64 strings."""
    if isinstance(video, torch.Tensor):
        # (T, C, H, W) -> list of frames
        return [_image_to_b64(frame) for frame in video]
    return [_image_to_b64(frame) for frame in video]


def _build_payload(
    prompt: List[str],
    image: Optional[List] = None,
    video: Optional[List] = None,
    condition_images: Optional[List[List]] = None,
    condition_videos: Optional[List[List]] = None,
) -> dict:
    """Build JSON-serializable request payload."""
    return {
        "prompt": prompt,
        "image": [_image_to_b64(img) for img in image] if image else None,
        "video": [_video_to_b64(v) for v in video] if video else None,
        "condition_images": [
            [_image_to_b64(img) for img in imgs] for imgs in condition_images
        ] if condition_images else None,
        "condition_videos": [
            [_video_to_b64(v) for v in vs] for vs in condition_videos
        ] if condition_videos else None,
    }


# ======================== Remote Client ========================

class RemoteRewardClient:
    """HTTP client for communicating with a Reward Server."""

    def __init__(self, server_url: str, timeout: float = 60.0, retries: int = 3):
        self.url = server_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()

    def health_check(self) -> bool:
        """Check server availability."""
        try:
            r = self._session.get(f"{self.url}/health", timeout=5.0)
            return r.ok
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def compute(
        self,
        prompt: List[str],
        image: Optional[List] = None,
        video: Optional[List] = None,
        condition_images: Optional[List[List]] = None,
        condition_videos: Optional[List[List]] = None,
    ) -> List[float]:
        """Send compute request and return rewards."""
        payload = _build_payload(
            prompt, image, video, condition_images, condition_videos
        )

        for attempt in range(self.retries):
            try:
                r = self._session.post(
                    f"{self.url}/compute", json=payload, timeout=self.timeout
                )
                r.raise_for_status()
                data = r.json()
                if data.get("error"):
                    raise RuntimeError(f"Server error: {data['error']}")
                return data["rewards"]
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Failed to connect to {self.url}: {e}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")

        raise RuntimeError("Unreachable")


# ======================== Reward Model Wrappers ========================

class RemotePointwiseRewardModel(PointwiseRewardModel):
    """
    Pointwise reward model that delegates computation to a remote server.

    Config Example:
        rewards:
          - name: "remote_aesthetic"
            reward_model: "flow_factory.rewards.remote.RemotePointwiseRewardModel"
            server_url: "http://localhost:8000"
            batch_size: 16
            timeout: 60.0        # optional
            retry_attempts: 3    # optional
    """

    required_fields: Tuple[str, ...] = ("prompt", "image") # Corresponds to the expected input kwargs
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        server_url = getattr(config, "server_url", None)
        if not server_url:
            raise ValueError("server_url is required for RemotePointwiseRewardModel")

        self.client = RemoteRewardClient(
            server_url=server_url,
            timeout=getattr(config, "timeout", 60.0),
            retries=getattr(config, "retry_attempts", 3),
        )

        if not self.client.health_check():
            raise RuntimeError(f"Cannot connect to reward server at {server_url}")
        logger.info(f"Connected to reward server at {server_url}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None, # If supporting video input, add 'video' to required_fields
        condition_images: Optional[List[List[Image.Image]]] = None, # If supporting conditional images, add 'condition_images' to required_fields
        condition_videos: Optional[List[List[List[Image.Image]]]] = None, # If supporting conditional videos, add 'condition_videos' to required_fields
    ) -> RewardModelOutput:
        rewards = self.client.compute(
            prompt=prompt,
            image=image,
            video=video,
            condition_images=condition_images,
            condition_videos=condition_videos,
        )
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device)
        )


class RemoteGroupwiseRewardModel(GroupwiseRewardModel):
    """
    Groupwise reward model that delegates computation to a remote server.

    Config Example:
        rewards:
          - name: "remote_ranking"
            reward_model: "flow_factory.rewards.remote.RemoteGroupwiseRewardModel"
            server_url: "http://localhost:8000"
            batch_size: 16
    """

    required_fields: Tuple[str, ...] = ("prompt", "image") # Corresponds to the expected input kwargs
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        server_url = getattr(config, "server_url", None)
        if not server_url:
            raise ValueError("server_url is required for RemoteGroupwiseRewardModel")

        self.client = RemoteRewardClient(
            server_url=server_url,
            timeout=getattr(config, "timeout", 120.0),
            retries=getattr(config, "retry_attempts", 3),
        )

        if not self.client.health_check():
            raise RuntimeError(f"Cannot connect to reward server at {server_url}")
        logger.info(f"Connected to reward server at {server_url}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None, # If supporting video input, add 'video' to required_fields
        condition_images: Optional[List[List[Image.Image]]] = None, # If supporting conditional images, add 'condition_images' to required_fields
        condition_videos: Optional[List[List[List[Image.Image]]]] = None, # If supporting conditional videos, add 'condition_videos' to required_fields
    ) -> RewardModelOutput:
        rewards = self.client.compute(
            prompt=prompt,
            image=image,
            video=video,
            condition_images=condition_images,
            condition_videos=condition_videos,
        )
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device)
        )