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

# Human Preference Score v2 — https://github.com/tgxs002/HPSv2
#
# The PyPI `hpsv2` package pins protobuf<4, which conflicts with flow-factory's
# protobuf>=6. Install it without its dependencies after flow-factory:
#     uv pip install hpsv2 --no-deps
# Runtime works with protobuf 6+; only the import metadata is affected.
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import huggingface_hub
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__, rank_zero_only=True)

_BPE_URLS = (
    "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
    "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz",
)


def _ensure_hpsv2_openclip_bpe_vocab() -> None:
    """Install the OpenCLIP BPE vocab that PyPI ``hpsv2`` wheels often omit.

    open_clip needs the vocab at import time.
    """
    try:
        import hpsv2
    except ImportError:
        return

    bpe = (
        Path(hpsv2.__file__).resolve().parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    )
    if bpe.is_file():
        return

    bpe.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[BaseException] = None
    for url in _BPE_URLS:
        try:
            logger.info("HPSv2: missing %s — downloading BPE vocab from %s", bpe.name, url)
            req = Request(url, headers={"User-Agent": "Flow-Factory/1.0 (HPSv2 BPE fix)"})
            with urlopen(req, timeout=120) as resp:
                bpe.write_bytes(resp.read())
            return
        except (OSError, URLError, TimeoutError) as e:
            last = e
            if bpe.is_file():
                bpe.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not install OpenCLIP BPE vocab at {bpe} (auto-download failed: {last!r}). "
        "Download bpe_simple_vocab_16e6.txt.gz from the OpenAI CLIP repo and place it next to "
        "hpsv2/src/open_clip/tokenizer.py, or reinstall hpsv2 from source."
    ) from last


def _import_hpsv2() -> Tuple[object, object, dict]:
    _ensure_hpsv2_openclip_bpe_vocab()
    try:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map
    except ImportError as e:
        raise ImportError(
            "HPSv2 reward requires the `hpsv2` package. The PyPI `hpsv2` metadata pins protobuf<4, "
            "which conflicts with flow-factory's protobuf>=6; install after flow-factory with:\n"
            "  uv pip install hpsv2 --no-deps\n"
            "Runtime works with protobuf 6+; see https://github.com/tgxs002/HPSv2"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "HPSv2 open_clip BPE file is still missing after auto-download. "
            "Place bpe_simple_vocab_16e6.txt.gz under hpsv2/src/open_clip/ (see _ensure_hpsv2_openclip_bpe_vocab)."
        ) from e
    return create_model_and_transforms, get_tokenizer, hps_version_map


class HPSv2RewardModel(PointwiseRewardModel):
    """Pointwise HPS v2 scores for image/video-text pairs (official open_clip + HPS checkpoint).

    Scores match the diagonal of ``image_features @ text_features.T`` as in
    ``hpsv2.img_score`` / DanceGRPO reference implementations.

    Configuration (via RewardArguments extra_kwargs):
        hps_version: HPS checkpoint version (default "v2.1").
        open_clip_pretrained: OpenCLIP pretrained tag (default "laion2B-s32B-b79K").
        checkpoint_path: optional local HPS checkpoint path (skips HF download).
    """

    required_fields: Tuple[str, ...] = ("prompt", "image", "video")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        create_model_and_transforms, get_tokenizer, hps_version_map = _import_hpsv2()

        extras = config.extra_kwargs or {}
        hps_ver = extras.get("hps_version", "v2.1")
        if hps_ver not in hps_version_map:
            raise ValueError(
                f"unknown hps_version {hps_ver!r}; expected one of {sorted(hps_version_map)}"
            )
        self._hps_version = hps_ver

        open_clip_pretrained = extras.get("open_clip_pretrained", "laion2B-s32B-b79K")
        ckpt = extras.get("checkpoint_path")

        # Un-gated download on every rank: Hugging Face Hub's per-blob file lock
        # serializes concurrent fetches, so exactly one rank transfers bytes while
        # the rest block on the lock and then read the warm cache. This avoids the
        # is_local_main_process gate + barrier, whose failure mode deadlocks the
        # siblings (mirrors models/abc.py `_resolve_checkpoint_path`).
        if ckpt is None:
            ckpt = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_ver])
        model, _preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            open_clip_pretrained,
            precision="amp",
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        if accelerator is not None:
            accelerator.wait_for_everyone()

        try:
            checkpoint = torch.load(ckpt, map_location=self.device, weights_only=False)
        except TypeError:
            # Older torch without weights_only kwarg.
            checkpoint = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.to(self.device).eval()
        self.preprocess_val = preprocess_val
        self.tokenizer = get_tokenizer("ViT-H-14")

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    def _compute_scores_batch(self, prompt: List[str], image: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.preprocess_val(im) for im in image]).to(
            device=self.device, non_blocking=True
        )
        text_tok = self.tokenizer(prompt).to(device=self.device, non_blocking=True)
        with torch.no_grad(), self._autocast():
            outputs = self.model(imgs, text_tok)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            scores = torch.diagonal(logits_per_image)
        return scores.float()

    def _compute_video_scores(
        self,
        prompt: List[str],
        video: List[List[Image.Image]],
        batch_size: int,
    ) -> torch.Tensor:
        frame_counts = [len(clip) for clip in video]
        flat_images = [frame for clip in video for frame in clip]
        flat_prompts = [p for p, n in zip(prompt, frame_counts) for _ in range(n)]
        all_scores: List[torch.Tensor] = []
        for i in range(0, len(flat_images), batch_size):
            batch_scores = self._compute_scores_batch(
                flat_prompts[i : i + batch_size],
                flat_images[i : i + batch_size],
            )
            all_scores.append(batch_scores)
        flat_scores = torch.cat(all_scores, dim=0)
        split = flat_scores.split(frame_counts)
        return torch.stack([s.mean() for s in split])

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
    ) -> RewardModelOutput:
        """Compute per-sample HPS v2 scores.

        Args:
            prompt: Text prompts (batch_size,).
            image: Generated PIL images (batch_size,); mutually exclusive with ``video``.
            video: Generated clips as frame lists (batch_size,); scored frame-wise then mean-pooled.

        Returns:
            RewardModelOutput with per-sample HPS v2 scores.
        """
        if not isinstance(prompt, list):
            prompt = [prompt]
        if image is not None and video is not None:
            raise ValueError("Only one of image or video can be provided.")
        if image is None and video is None:
            raise ValueError("HPSv2 reward requires either image or video input.")

        batch_size = getattr(self.config, "batch_size", len(prompt))

        if video is not None:
            if len(video) != len(prompt):
                raise ValueError(
                    f"video/prompt length mismatch: {len(video)} clips vs {len(prompt)} prompts."
                )
            scores = self._compute_video_scores(prompt, video, batch_size)
        else:
            if len(image) != len(prompt):
                raise ValueError(
                    f"image/prompt length mismatch: {len(image)} images vs {len(prompt)} prompts."
                )
            chunks: List[torch.Tensor] = []
            for i in range(0, len(prompt), batch_size):
                chunks.append(
                    self._compute_scores_batch(
                        prompt[i : i + batch_size],
                        image[i : i + batch_size],
                    )
                )
            scores = torch.cat(chunks, dim=0)

        return RewardModelOutput(rewards=scores, extra_info={"hps_version": self._hps_version})
