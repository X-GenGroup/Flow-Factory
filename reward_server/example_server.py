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

# reward_server/example_server.py
"""
Reward Server Template

A standalone server for computing rewards in an isolated environment.
Copy this file and implement `compute_reward()` with your logic.

Dependencies:
    >>> pip install fastapi uvicorn pillow

Usage:
    >>> python example_server.py --port 8000

Training Config:
    rewards:
      - name: "my_reward"
        reward_model: "flow_factory.rewards.remote.RemotePointwiseRewardModel"
        server_url: "http://localhost:8000"
        batch_size: 16
"""

import argparse
import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================== Deserialization Helpers ========================

def _b64_to_image(b64: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _parse_request(data: dict) -> dict:
    """Parse incoming request, converting base64 to PIL Images."""
    result = {"prompt": data.get("prompt"), "extra": data.get("extra") or {}}

    if data.get("image"):
        result["image"] = [_b64_to_image(b) for b in data["image"]]

    if data.get("video"):
        result["video"] = [
            [_b64_to_image(f) for f in frames] for frames in data["video"]
        ]

    if data.get("condition_images"):
        result["condition_images"] = [
            [_b64_to_image(b) for b in imgs] for imgs in data["condition_images"]
        ]

    if data.get("condition_videos"):
        result["condition_videos"] = [
            [[_b64_to_image(f) for f in v] for v in vs]
            for vs in data["condition_videos"]
        ]

    return result


# ======================== Base Reward Server ========================

class RewardServer(ABC):
    """
    Base class for reward computation servers.

    Subclass and implement `compute_reward()` to create your own server.

    Example:
        class MyServer(RewardServer):
            def __init__(self, model_path: str, **kwargs):
                super().__init__(**kwargs)
                self.model = load_model(model_path)

            def compute_reward(self, prompt, image=None, **kwargs):
                return [self.model.score(p, i) for p, i in zip(prompt, image)]

        if __name__ == "__main__":
            MyServer(model_path="./model").run()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port

    @abstractmethod
    def compute_reward(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
    ) -> List[float]:
        """
        Compute rewards for a batch of samples. Override this method.

        Args:
            prompt: Text prompts
            image: Generated images (PIL)
            video: Generated videos (list of PIL frames)
            condition_images: Condition images
            condition_videos: Condition videos

        Returns:
            List of reward scores (float), one per sample
        """
        pass

    def run(self):
        """Start the FastAPI server."""
        app = FastAPI(title="Reward Server")

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/compute")
        async def compute(request: Dict[str, Any]):
            try:
                data = _parse_request(request)
                rewards = self.compute_reward(
                    prompt=data.get("prompt"),
                    image=data.get("image"),
                    video=data.get("video"),
                    condition_images=data.get("condition_images"),
                    condition_videos=data.get("condition_videos"),
                )
                return {"rewards": rewards}
            except Exception as e:
                logger.exception("Compute error")
                raise HTTPException(status_code=500, detail=str(e))

        logger.info(f"Starting server at http://{self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)


# ======================== Example Implementation ========================

class MyRewardServer(RewardServer):
    """
    Example reward server. Replace with your own logic.

    To use a real model:
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model = YourModel.from_pretrained("model-name")

        def compute_reward(self, prompt, image=None, **kwargs):
            return self.model.score(prompt, image)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model here
        # self.model = ...
        logger.info("MyRewardServer initialized (placeholder)")

    def compute_reward(
        self,
        prompt: List[str], # A batch or a group of prompts
        image: Optional[List[Image.Image]] = None, # Corresponding images
        video: Optional[List[List[Image.Image]]] = None, # Corresponding videos
        condition_images: Optional[List[List[Image.Image]]] = None, # Corresponding condition images
        condition_videos: Optional[List[List[List[Image.Image]]]] = None, # Corresponding condition videos
    ) -> List[float]:
        # ===== Replace with your reward logic =====
        # Example: return random scores
        import random
        return [random.random() for _ in prompt]


# ======================== Main ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = MyRewardServer(host=args.host, port=args.port)
    server.run()