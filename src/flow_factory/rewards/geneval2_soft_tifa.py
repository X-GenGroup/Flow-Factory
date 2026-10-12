# Copyright: Meta Platforms, Inc. and affiliates (GenEval2 evaluation logic)
# Copyright 2026 Jayce-Ping (Flow-Factory integration)
#
# GenEval2 benchmark and Soft-TIFA are from https://github.com/facebookresearch/GenEval2
# (CC BY-NC 4.0). This module adapts the official `soft_tifa` scoring for Flow-Factory
# PointwiseRewardModel; it does not import GenEval2's evaluation.py (top-level model load).

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from scipy.stats import gmean as _gmean_scores
except ImportError:  # optional; `pip install -e ".[geneval2]"` for exact GM parity

    def _gmean_scores(xs: List[float]) -> float:
        xs = [max(x, 1e-300) for x in xs]
        return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__, rank_zero_only=True)

VqaAtom = Tuple[str, str]


def _hf_single_gpu_device_map(device: torch.device) -> Dict[str, Any]:
    """Map the full model to one device (no CPU/disk offload).

    The result uses the Hugging Face ``device_map`` format.
    """
    if device.type == "cuda":
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        return {"": idx}
    if device.type == "cpu":
        return {"": "cpu"}
    return {"": device}


def _resolve_geneval2_device_map(
    device_map: Optional[Union[str, Dict[str, Any]]],
    reward_device: torch.device,
) -> Optional[Union[str, Dict[str, Any]]]:
    if device_map == "auto":
        return "auto"
    if device_map is None:
        return _hf_single_gpu_device_map(reward_device)
    if isinstance(device_map, dict):
        return device_map
    raise ValueError(
        "geneval2_soft_tifa ``device_map`` must be omitted (pin to reward device), "
        "'auto', or an explicit Hugging Face device_map dict. "
        f"Got: {device_map!r}"
    )


def _load_prompt_vqa_map_from_jsonl_paths(paths: List[Path]) -> Dict[str, List[Any]]:
    """Build ``prompt -> vqa_list`` from GenEval2-style JSONL files.

    Args:
        paths: GenEval2 JSONL files; each line must have ``prompt`` and ``vqa_list``.
            ``vqa_list`` may be a native list or a JSON string (GenEval-style
            stringified metadata); both are normalized to a list.

    Returns:
        Mapping from prompt to its ``vqa_list`` (always a list).

    Raises:
        ValueError: If a prompt appears twice with different ``vqa_list`` payloads.
    """
    out: Dict[str, List[Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"GenEval2 benchmark JSONL not found: {path}")
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt")
                if prompt is None:
                    raise ValueError(f"{path}:{lineno}: missing 'prompt'")
                vqa = obj.get("vqa_list")
                if vqa is None:
                    raise ValueError(f"{path}:{lineno}: missing 'vqa_list'")
                if isinstance(vqa, str):
                    vqa = json.loads(vqa)
                if prompt in out and out[prompt] != vqa:
                    raise ValueError(
                        f"Duplicate prompt with different vqa_list: {prompt!r} "
                        f"(first seen elsewhere, conflict at {path}:{lineno})"
                    )
                out[prompt] = vqa
    return out


def _resolve_benchmark_jsonl_paths(extras: dict) -> List[Path]:
    """Collect GenEval2 JSONL paths from ``benchmark_jsonl`` and/or ``data_path`` extras.

    Args:
        extras: RewardArguments.extra_kwargs. ``benchmark_jsonl`` is a single .jsonl;
            ``data_path`` is a .jsonl file or a directory holding train.jsonl / test.jsonl.

    Returns:
        Deduplicated, order-preserving list of JSONL paths.
    """
    paths: List[Path] = []
    bj = extras.get("benchmark_jsonl")
    if bj:
        paths.append(Path(bj).expanduser())

    dp = extras.get("data_path")
    if dp:
        p = Path(dp).expanduser()
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            for name in ("train.jsonl", "test.jsonl"):
                child = p / name
                if child.is_file():
                    paths.append(child)
        else:
            raise FileNotFoundError(f"data_path is not a file or directory: {p}")

    seen = set()
    uniq: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _vqa_from_metadata(metadata_entry: Optional[Union[str, Dict[str, Any]]]) -> Optional[List[Any]]:
    """Extract a ``vqa_list`` from a per-sample metadata payload, if present.

    Args:
        metadata_entry: The sample's ``metadata`` (JSON string or already-parsed dict),
            as delivered by Flow-Factory's reward pipeline from dataset JSONL columns.

    Returns:
        The parsed ``vqa_list`` (a list), or None when unavailable.
    """
    if metadata_entry is None:
        return None
    meta = metadata_entry
    if isinstance(meta, str):
        meta = json.loads(meta)
    if not isinstance(meta, dict):
        return None
    vqa = meta.get("vqa_list")
    if isinstance(vqa, str):
        vqa = json.loads(vqa)
    return vqa if isinstance(vqa, list) else None


def _return_numeric_string(number: str) -> str:
    match number:
        case "one":
            return "1"
        case "two":
            return "2"
        case "three":
            return "3"
        case "four":
            return "4"
        case "five":
            return "5"
        case "six":
            return "6"
        case "seven":
            return "7"
        case "eight":
            return "8"
        case "nine":
            return "9"
        case "ten":
            return "10"
    return "other"


def _construct_message_with_image(text: str, image_ref: Union[str, Image.Image]) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": text},
            ],
        }
    ]


class GenEval2SoftTIFARewardModel(PointwiseRewardModel):
    """Soft-TIFA score (GenEval2) as a pointwise reward.

    Qwen3-VL answers each VQA atom; the per-atom soft-match probabilities are
    aggregated with AM or GM (matching GenEval2 ``evaluation.py``).

    ``vqa_list`` is resolved per sample, in priority order:
      1. an explicit ``vqa_list`` kwarg (if the pipeline passes one),
      2. the sample's ``metadata`` JSON (Flow-Factory packs dataset JSONL columns into
         ``metadata``),
      3. a JSONL lookup keyed by ``prompt`` (``benchmark_jsonl`` / ``data_path`` extras).

    GenEval2 benchmark prompts are unique. Very slow: one short generation per VQA atom per image.

    YAML extras (RewardArguments.extra_kwargs):
        aggregation: "am" | "gm" (default "gm", matching Soft-TIFA GM benchmark reporting)
        model_name: Hugging Face id (default "Qwen/Qwen3-VL-8B-Instruct")
        device_map: omitted/null pins the model to ``self.device``; "auto" uses HF heuristics;
            or pass an explicit device_map dict.
        benchmark_jsonl: optional path to one GenEval2-style .jsonl (prompt + vqa_list per line)
        data_path: optional .jsonl, or a directory with train.jsonl and/or test.jsonl
            (all lines merged into one prompt -> vqa_list map).
    """

    required_fields: Tuple[str, ...] = ("prompt", "image", "metadata")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        extras = config.extra_kwargs or {}
        self._model_name = extras.get("model_name", "Qwen/Qwen3-VL-8B-Instruct")
        self._device_map = extras.get("device_map", None)
        agg = str(extras.get("aggregation", "gm")).lower()
        if agg not in ("am", "gm"):
            raise ValueError(f"aggregation must be 'am' or 'gm', got {agg!r}")
        self._aggregation = agg

        jsonl_paths = _resolve_benchmark_jsonl_paths(extras)
        self._prompt_to_vqa: Optional[Dict[str, List[Any]]] = None
        if jsonl_paths:
            self._prompt_to_vqa = _load_prompt_vqa_map_from_jsonl_paths(jsonl_paths)
            logger.info(
                "GenEval2 Soft-TIFA: loaded %d prompt->vqa_list entries from %s",
                len(self._prompt_to_vqa),
                [str(p) for p in jsonl_paths],
            )

        # Cache the first token id of each fixed answer variant; encoding is pure
        # function of the string, so this avoids re-tokenizing per VQA atom.
        self._answer_token_id: Dict[str, int] = {}

        resolved_device_map = _resolve_geneval2_device_map(self._device_map, self.device)

        # Un-gated load on every rank: Hugging Face Hub's per-blob file lock
        # serializes concurrent downloads, so only one rank transfers bytes while
        # the rest block then read the warm cache. This avoids the
        # is_local_main_process gate + barrier whose failure mode deadlocks the
        # siblings (mirrors models/abc.py `_resolve_checkpoint_path`).
        self._processor = AutoProcessor.from_pretrained(self._model_name, torch_dtype="auto")
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_name,
            dtype="auto",
            device_map=resolved_device_map,
        )
        if accelerator is not None:
            accelerator.wait_for_everyone()

        self.qwen_model.eval()
        self.model = self.qwen_model

    def _model_device(self) -> torch.device:
        if hasattr(self.qwen_model, "device"):
            return self.qwen_model.device
        return next(self.qwen_model.parameters()).device

    def _first_token_id(self, answer: str) -> int:
        """Return (and cache) the first token id of a fixed answer variant."""
        token_id = self._answer_token_id.get(answer)
        if token_id is None:
            token_id = self._processor.tokenizer.encode(answer)[0]
            self._answer_token_id[answer] = token_id
        return token_id

    def _answer_probability(
        self,
        text: str,
        image_ref: Union[str, Image.Image],
        answer_list: Sequence[str],
    ) -> float:
        """Total next-token probability mass over the accepted answer variants.

        Args:
            text: The VQA question (already suffixed with the one-word instruction).
            image_ref: The image to condition on (PIL image or path).
            answer_list: Accepted answer surface forms; duplicates are summed with
                multiplicity to match the official Soft-TIFA scoring.

        Returns:
            Summed softmax probability of the first generated token over ``answer_list``.
        """
        messages = _construct_message_with_image(text, image_ref)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model_device())
        with torch.inference_mode():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        token_ids = torch.as_tensor(
            [self._first_token_id(a) for a in answer_list], device=probs.device
        )
        return probs[0, token_ids].sum().item()

    def _soft_tifa(
        self,
        vqa_list: List[VqaAtom],
        image_ref: Union[str, Image.Image],
    ) -> List[float]:
        score_list: List[float] = []
        for question, answer in vqa_list:
            if question.startswith("How many"):
                answer_list = [
                    answer,
                    answer.capitalize(),
                    " " + answer,
                    " " + answer.capitalize(),
                    _return_numeric_string(answer),
                    " " + _return_numeric_string(answer),
                ]
            else:
                answer_list = ["Yes", "yes", " yes", " Yes"]
            score_list.append(
                self._answer_probability(
                    "{} Answer in one word.".format(question),
                    image_ref,
                    answer_list,
                )
            )
        return score_list

    def _aggregate(self, score_list: List[float]) -> float:
        if self._aggregation == "gm":
            return float(_gmean_scores(score_list))
        return float(sum(score_list) / len(score_list))

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        metadata: Optional[List[str]] = None,
        vqa_list: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> RewardModelOutput:
        """Compute Soft-TIFA rewards for a batch of generated images.

        Args:
            prompt: Text prompts (batch_size,).
            image: Generated PIL images (batch_size,); video is not supported.
            metadata: Per-sample metadata JSON strings (batch_size,); may carry ``vqa_list``.
            vqa_list: Optional explicit per-sample VQA atom lists (batch_size,).

        Returns:
            RewardModelOutput with per-sample Soft-TIFA scores in [0, 1].
        """
        if video is not None:
            raise ValueError("GenEval2SoftTIFARewardModel supports image only.")
        if image is None:
            raise ValueError("image is required.")
        if len(image) != len(prompt):
            raise ValueError(
                f"image/prompt length mismatch: {len(image)} images vs {len(prompt)} prompts."
            )

        rewards: List[float] = []
        for i in range(len(prompt)):
            vqa: Any = None
            if vqa_list is not None:
                vqa = vqa_list[i]
            if (vqa is None or not isinstance(vqa, list)) and metadata is not None:
                vqa = _vqa_from_metadata(metadata[i])
            if (vqa is None or not isinstance(vqa, list)) and self._prompt_to_vqa is not None:
                vqa = self._prompt_to_vqa.get(prompt[i])
            if not isinstance(vqa, list) or len(vqa) == 0:
                raise ValueError(
                    "Could not resolve a non-empty vqa_list for Soft-TIFA: provide per-sample "
                    "vqa_list, ensure the dataset JSONL has a 'vqa_list' column (delivered via "
                    "'metadata'), or set extra_kwargs benchmark_jsonl / data_path to a GenEval2 "
                    "JSONL (or dataset directory with train.jsonl/test.jsonl). "
                    f"Got {vqa!r} for prompt[{i}]={prompt[i]!r}."
                )
            score_list = self._soft_tifa(vqa, image[i])
            rewards.append(self._aggregate(score_list))

        return RewardModelOutput(rewards=torch.tensor(rewards, dtype=torch.float32))
