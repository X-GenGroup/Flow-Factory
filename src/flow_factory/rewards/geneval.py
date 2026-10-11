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

# src/flow_factory/rewards/geneval.py
"""GenEval Reward Model — compositional T2I evaluation via object detection.

Evaluates generated images on:
- Object presence and counting
- Color accuracy (via CLIP zero-shot classification)
- Spatial relationships (above/below/left/right)
- Object exclusion (penalize unwanted objects)

Based on the GenEval benchmark (https://github.com/djghosh13/geneval).

Note:
    Install dependencies with ``bash scripts/install_geneval_deps.sh``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]

DEFAULT_DETECTION_THRESHOLD = 0.3
DEFAULT_COUNTING_THRESHOLD = 0.9
DEFAULT_MAX_OBJECTS = 16

# Default paths (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OBJECT_NAMES_PATH = str(_PROJECT_ROOT / "dataset" / "geneval" / "object_names.txt")

# Mask2Former Swin-S config/checkpoint (mmdet 3.x model zoo).
# Names follow the mmdet 3.x convention: `8xb2-lsj-50e` (mmdet 2.x used `lsj_8x2_50e`).
DEFAULT_DETECTOR_CONFIG = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco"
DEFAULT_DETECTOR_CHECKPOINT = (
    "https://download.openmmlab.com/mmdetection/v3.0/mask2former/"
    "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/"
    "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
)
DEFAULT_CLIP_MODEL = "ViT-L-14"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _load_object_names(path: str) -> List[str]:
    """Load COCO class names from a text file (one name per line)."""
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def _check_position(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
    relation: str,
) -> bool:
    """Check if bbox_b satisfies spatial relation relative to bbox_a.

    Args:
        bbox_a: Reference object bounding box [x1, y1, x2, y2].
        bbox_b: Target object bounding box [x1, y1, x2, y2].
        relation: One of 'above', 'below', 'left of', 'right of'.

    Returns:
        True if the spatial relationship is satisfied.
    """
    center_a = ((bbox_a[0] + bbox_a[2]) / 2, (bbox_a[1] + bbox_a[3]) / 2)
    center_b = ((bbox_b[0] + bbox_b[2]) / 2, (bbox_b[1] + bbox_b[3]) / 2)

    if relation == "above":
        return center_b[1] < center_a[1]
    elif relation == "below":
        return center_b[1] > center_a[1]
    elif relation == "left of":
        return center_b[0] < center_a[0]
    elif relation == "right of":
        return center_b[0] > center_a[0]
    else:
        logger.warning(f"Unknown spatial relation: {relation!r}")
        return False


# ---------------------------------------------------------------------------
# GenEval Reward Model
# ---------------------------------------------------------------------------


class GenEvalRewardModel(PointwiseRewardModel):
    """Compositional T2I reward using Mask2Former detection + CLIP color classification.

    Evaluates generated images against structured metadata specifying required
    objects (with optional count, color, and spatial constraints) and excluded objects.

    Configuration (via RewardArguments extra_kwargs):
        detector_config: mmdet config name or path (default: Mask2Former Swin-S)
        detector_checkpoint: checkpoint URL or path (default: mmdet model zoo)
        clip_model: open_clip model name (default: 'ViT-L-14')
        object_names_path: path to COCO class names file
        detection_threshold: confidence threshold for detection (default: 0.3)
        counting_threshold: stricter threshold for counting (default: 0.9)
        max_objects: maximum detections counted per class (default: 16)
    """

    required_fields: Tuple[str, ...] = ("image", "prompt", "metadata")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        if self.device.type != "cuda":
            raise ValueError(
                "GenEval requires CUDA (Mask2Former uses CUDA-only ops). "
                "Set `device: cuda` in your reward config."
            )

        # Extract config params (extra_kwargs from YAML config)
        self._detection_threshold = getattr(
            config, "detection_threshold", DEFAULT_DETECTION_THRESHOLD
        )
        self._counting_threshold = getattr(config, "counting_threshold", DEFAULT_COUNTING_THRESHOLD)
        self._max_objects = getattr(config, "max_objects", DEFAULT_MAX_OBJECTS)

        object_names_path = getattr(config, "object_names_path", DEFAULT_OBJECT_NAMES_PATH)
        self._object_names = _load_object_names(object_names_path)
        self._name_to_idx = {name: idx for idx, name in enumerate(self._object_names)}

        # Load Mask2Former detector
        self._init_detector(config)

        # Load CLIP for color classification
        self._init_clip(config)

        logger.info(
            f"GenEvalRewardModel initialized: "
            f"det_thresh={self._detection_threshold}, "
            f"count_thresh={self._counting_threshold}, "
            f"{len(self._object_names)} object classes."
        )

    def _init_detector(self, config: RewardArguments) -> None:
        """Initialize Mask2Former instance segmentation model."""
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError:
            raise ImportError(
                "mmdet is required for GenEval reward. "
                "Install with: bash scripts/install_geneval_deps.sh"
            )

        self._inference_detector = inference_detector

        detector_config = getattr(config, "detector_config", DEFAULT_DETECTOR_CONFIG)
        detector_checkpoint = getattr(config, "detector_checkpoint", DEFAULT_DETECTOR_CHECKPOINT)

        # If config is a short name (no path separator, no .py), resolve from
        # mmdet's bundled model zoo configs. mmdet 3.x ships configs under
        # `<mmdet>/.mim/configs/...` (via the mim editable install hooks).
        if (
            not os.path.exists(detector_config)
            and not detector_config.endswith(".py")
            and "/" not in detector_config
        ):
            resolved = self._resolve_mmdet_short_config(detector_config)
            if resolved is None:
                raise FileNotFoundError(
                    f"Could not resolve mmdet config '{detector_config}'. "
                    f"Pass an absolute path via `detector_config:` in your YAML, "
                    f"or ensure mmdet was installed with its bundled model zoo "
                    f"(`<mmdet>/.mim/configs/...`)."
                )
            logger.info(f"Resolved mmdet config: {detector_config} -> {resolved}")
            detector_config = resolved

        device_str = f"cuda:{self.accelerator.local_process_index}"
        self._detector = init_detector(
            detector_config,
            detector_checkpoint,
            device=device_str,
        )
        logger.info(f"Mask2Former loaded on {device_str}.")

    @staticmethod
    def _resolve_mmdet_short_config(short_name: str) -> Optional[str]:
        """Locate `<short_name>.py` inside mmdet's bundled `.mim/configs` tree.

        Returns the absolute path on success, or None if not found.
        """
        import mmdet

        mmdet_root = Path(mmdet.__file__).parent
        # Standard layout for `mim install mmdet` / pip wheel:
        #   <mmdet>/.mim/configs/<algo>/<short_name>.py
        candidate_roots = [
            mmdet_root / ".mim" / "configs",
            mmdet_root.parent / "configs",  # legacy / source install
        ]
        target = f"{short_name}.py"
        for root in candidate_roots:
            if not root.is_dir():
                continue
            for path in root.rglob(target):
                return str(path)
        return None

    def _init_clip(self, config: RewardArguments) -> None:
        """Initialize CLIP model for zero-shot color classification."""
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for GenEval color classification. "
                "Install with: bash scripts/install_geneval_deps.sh"
            )

        clip_model_name = getattr(config, "clip_model", DEFAULT_CLIP_MODEL)
        device_str = f"cuda:{self.accelerator.local_process_index}"

        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained="openai", device=device_str
        )
        self._clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        self._clip_model.eval()

        # Per-class color text embeddings, filled lazily by `_get_color_text_features`
        self._color_text_features: Dict[str, torch.Tensor] = {}
        logger.info(f"CLIP {clip_model_name} loaded for color classification.")

    @torch.no_grad()
    def _get_color_text_features(self, classname: str) -> torch.Tensor:
        """Get or compute cached color text embeddings for a given class.

        Returns:
            Tensor of shape (num_colors, embed_dim) with L2-normalized features.
        """
        if classname not in self._color_text_features:
            prompts = [f"a photo of a {c} {classname}" for c in COLORS]
            tokens = self._clip_tokenizer(prompts).to(self._clip_model.visual.proj.device)
            features = self._clip_model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
            self._color_text_features[classname] = features
        return self._color_text_features[classname]

    @torch.no_grad()
    def _classify_color(
        self,
        image: Image.Image,
        bbox: np.ndarray,
        classname: str,
    ) -> str:
        """Classify the color of a detected object region via CLIP zero-shot.

        Args:
            image: Full PIL image.
            bbox: Bounding box [x1, y1, x2, y2].
            classname: Object class name for prompt construction.

        Returns:
            Predicted color string.
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        crop = image.crop((x1, y1, x2, y2))
        if crop.width < 1 or crop.height < 1:
            return "unknown"

        device = self._clip_model.visual.proj.device
        img_tensor = self._clip_preprocess(crop).unsqueeze(0).to(device)
        img_features = self._clip_model.encode_image(img_tensor)
        img_features = F.normalize(img_features, dim=-1)

        text_features = self._get_color_text_features(classname)
        similarity = (img_features @ text_features.T).squeeze(0)
        color_idx = similarity.argmax().item()
        return COLORS[color_idx]

    @torch.no_grad()
    def _detect_objects(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run Mask2Former detection on an image.

        Returns:
            bboxes: (N, 4) array of [x1, y1, x2, y2]
            labels: (N,) array of class indices
            scores: (N,) array of confidence scores
        """
        img_array = np.array(image)
        result = self._inference_detector(self._detector, img_array)

        # Extract predictions from mmdet 3.x DetDataSample
        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        scores = pred.scores.cpu().numpy()

        # Filter by base detection threshold
        mask = scores >= self._detection_threshold
        return bboxes[mask], labels[mask], scores[mask]

    def _evaluate_single(
        self,
        image: Image.Image,
        include: List[Dict[str, Any]],
        exclude: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Evaluate a single image against its GenEval metadata.

        Args:
            image: Generated PIL image.
            include: List of required object specs with class/count/color/position.
            exclude: Optional list of objects that should not appear.

        Returns:
            Reward score in [0, 1].
        """
        bboxes, labels, scores = self._detect_objects(image)

        sub_rewards: List[float] = []

        for req in include:
            classname = req["class"]
            expected_count = req.get("count", 1)
            required_color = req.get("color", None)
            position_spec = req.get("position", None)

            class_idx = self._name_to_idx.get(classname)
            if class_idx is None:
                logger.warning(f"Unknown class '{classname}' in GenEval metadata.")
                sub_rewards.append(0.0)
                continue

            # Find detections of this class
            class_mask = labels == class_idx
            class_bboxes = bboxes[class_mask]
            class_scores = scores[class_mask]

            # For counting, use stricter threshold
            if expected_count > 1 or (
                exclude and any(e.get("class") == classname for e in exclude)
            ):
                count_mask = class_scores >= self._counting_threshold
                count_bboxes = class_bboxes[count_mask]
            else:
                count_bboxes = class_bboxes

            # Limit max objects
            if len(count_bboxes) > self._max_objects:
                count_bboxes = count_bboxes[: self._max_objects]

            found_count = len(count_bboxes)

            # Count reward
            count_reward = max(0.0, 1.0 - abs(expected_count - found_count) / expected_count)

            if required_color and found_count > 0:
                # Color reward: check how many detected objects match the color
                colored_count = 0
                for bbox in count_bboxes[:expected_count]:
                    predicted_color = self._classify_color(image, bbox, classname)
                    if predicted_color == required_color:
                        colored_count += 1
                color_reward = max(0.0, 1.0 - abs(expected_count - colored_count) / expected_count)
                sub_rewards.append(min(count_reward, color_reward))

            elif position_spec and found_count > 0:
                # Position reward: check spatial relation
                relation, ref_group_idx = position_spec
                # Find the reference object (from a previous include entry)
                if ref_group_idx < len(include):
                    ref_classname = include[ref_group_idx]["class"]
                    ref_class_idx = self._name_to_idx.get(ref_classname)
                    if ref_class_idx is not None:
                        ref_mask = labels == ref_class_idx
                        ref_bboxes = bboxes[ref_mask]
                        if len(ref_bboxes) > 0 and len(count_bboxes) > 0:
                            pos_satisfied = _check_position(
                                ref_bboxes[0], count_bboxes[0], relation
                            )
                            sub_rewards.append(count_reward if pos_satisfied else 0.0)
                        else:
                            sub_rewards.append(0.0)
                    else:
                        sub_rewards.append(0.0)
                else:
                    sub_rewards.append(count_reward)
            else:
                sub_rewards.append(count_reward)

        # Exclude penalties
        if exclude:
            for exc in exclude:
                classname = exc["class"]
                max_allowed = exc.get("count", 0)
                class_idx = self._name_to_idx.get(classname)
                if class_idx is None:
                    continue
                class_mask = labels == class_idx
                class_scores_exc = scores[class_mask]
                found = int((class_scores_exc >= self._counting_threshold).sum())
                if found > max_allowed:
                    excess = found - max_allowed
                    penalty = max(0.0, 1.0 - excess / max(max_allowed, 1))
                    sub_rewards.append(penalty)

        if not sub_rewards:
            return 0.0

        return sum(sub_rewards) / len(sub_rewards)

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        metadata: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RewardModelOutput:
        """Compute GenEval rewards for a batch of images.

        Args:
            prompt: Text prompts (batch_size,).
            image: Generated PIL images (batch_size,).
            video: Generated videos; the first frame is scored when ``image`` is None.
            metadata: Per-sample metadata JSON strings (batch_size,).
                Each JSON object should contain ``include`` (required),
                and optionally ``exclude`` and ``tag``.

        Returns:
            RewardModelOutput with rewards in [0, 1].
        """
        if image is None and video is not None:
            image = [v[0] for v in video]

        if image is None:
            raise ValueError("GenEval reward requires image input.")
        if metadata is None:
            raise ValueError(
                "GenEval reward requires 'metadata' containing 'include'. "
                "Ensure dataset JSONL contains 'include' field."
            )

        if not isinstance(prompt, list):
            prompt = [prompt]

        batch_size = len(image)
        rewards = []
        tags: Optional[List[str]] = None

        with torch.amp.autocast("cuda", enabled=False):
            for i in range(batch_size):
                meta = metadata[i] if metadata else "{}"
                if isinstance(meta, str):
                    meta = json.loads(meta)
                inc = meta.get("include", [])
                exc = meta.get("exclude", None)
                tag = meta.get("tag", None)
                if isinstance(inc, str):
                    inc = json.loads(inc)
                if isinstance(exc, str):
                    exc = json.loads(exc) or None
                if tag is not None:
                    if tags is None:
                        tags = []
                    tags.append(tag)
                reward = self._evaluate_single(image[i], inc, exc)
                rewards.append(reward)

        extra_info = {}
        if tags is not None:
            extra_info["tags"] = tags

        return RewardModelOutput(rewards=rewards, extra_info=extra_info)
