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

# src/flow_factory/utils/checkpoint.py
"""Utility functions for handling checkpoint management."""

import glob
import json
import os
import re
from typing import Dict, List, Literal, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def mapping_lora_state_dict(
    state_dict: Dict[str, torch.Tensor], adapter_name: str = "default"
) -> Dict[str, torch.Tensor]:
    """Map LoRA state_dict keys to PeftModel format.

    Prefixes keys with ``base_model.model.`` when missing and inserts ``adapter_name`` into
    LoRA weight keys: ``xxx.lora_A.weight`` -> ``base_model.model.xxx.lora_A.default.weight``.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("base_model.model"):
            key = "base_model.model." + key
        if "lora_A.weight" in key or "lora_B.weight" in key:
            new_key = key.replace("lora_A.weight", f"lora_A.{adapter_name}.weight").replace(
                "lora_B.weight", f"lora_B.{adapter_name}.weight"
            )
            new_state_dict[new_key] = value
        else:
            # Keep other keys as-is
            new_state_dict[key] = value
    return new_state_dict


# ================================ Config Inference ================================
def infer_lora_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer LoRA rank from state dict.

    Args:
        state_dict: LoRA state dictionary

    Returns:
        Inferred rank value

    Raises:
        ValueError: If no lora_A/lora_B weights found
    """
    # Try lora_A first (shape: [rank, in_features])
    for key, tensor in state_dict.items():
        if "lora_A" in key and "weight" in key:
            return tensor.shape[0]

    # Fallback to lora_B (shape: [out_features, rank])
    for key, tensor in state_dict.items():
        if "lora_B" in key and "weight" in key:
            return tensor.shape[1]

    raise ValueError("Cannot infer rank: no lora_A or lora_B weights found")


def infer_lora_alpha(
    state_dict: Dict[str, torch.Tensor], default_rank: Optional[int] = None
) -> int:
    """Infer LoRA alpha from state dict, defaulting to rank.

    Args:
        state_dict: LoRA state dictionary
        default_rank: Fallback if alpha not found (uses inferred rank if None)

    Returns:
        Inferred or default alpha value
    """
    for key, tensor in state_dict.items():
        if "lora_alpha" in key.lower() or "scaling" in key.lower():
            return int(tensor.item())

    return default_rank or infer_lora_rank(state_dict)


def infer_lora_config(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Infer both rank and alpha from state dict.

    Args:
        state_dict: LoRA state dictionary

    Returns:
        Tuple of (rank, alpha)
    """
    rank = infer_lora_rank(state_dict)
    alpha = infer_lora_alpha(state_dict, default_rank=rank)
    return rank, alpha


def infer_target_modules(
    state_dict: Dict[str, torch.Tensor],
    prefix: Optional[str] = None,
) -> List[str]:
    """Infer full module paths from state dict (for precise LoRA targeting).

    Args:
        state_dict: LoRA state dictionary
        prefix: Optional prefix to strip from paths; when None, a known prefix
            (``transformer``, ``unet``, ``text_encoder``, ``base_model.model``) is detected
            from the first key

    Returns:
        Sorted list of full module paths
    """
    # Auto-detect prefix
    if prefix is None:
        first_key = next(iter(state_dict.keys()), "")
        for p in ("transformer.", "unet.", "text_encoder.", "base_model.model."):
            if first_key.startswith(p):
                prefix = p.rstrip(".")
                break

    prefix_pattern = f"^(?:{re.escape(prefix)}\\.)?" if prefix else "^"
    module_pattern = re.compile(prefix_pattern + r"(.*)\.lora_[AB](?:\.[^.]+)?\.weight$")

    target_modules = set()
    for key in state_dict.keys():
        match = module_pattern.match(key)
        if match:
            target_modules.add(match.group(1))

    return sorted(target_modules)


# ================================ Hugging Face Hub ================================
HF_PATH_PREFIX = "hf://"


def parse_hf_checkpoint_path(path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse a Hugging Face checkpoint path spec into ``(repo_id, subfolder, revision)``.

    Accepts both bare and ``hf://``-prefixed specs:
      - ``owner/repo``                          -> (``owner/repo``,  None,           None)
      - ``hf://owner/repo``                     -> (``owner/repo``,  None,           None)
      - ``owner/repo/sub/dir``                  -> (``owner/repo``,  ``sub/dir``,    None)
      - ``owner/repo@v1.0``                     -> (``owner/repo``,  None,           ``v1.0``)
      - ``hf://owner/repo/sub/dir@v1.0``        -> (``owner/repo``,  ``sub/dir``,    ``v1.0``)

    Args:
        path: A bare or ``hf://``-prefixed checkpoint spec.

    Returns:
        Tuple of (repo_id, subfolder, revision); subfolder and revision are ``None`` when absent.

    Raises:
        TypeError: If ``path`` is not a string.
        ValueError: If the spec lacks the ``owner/repo`` form (at minimum two path segments),
            has an empty or ``/``-containing revision, or contains a ``.``, ``..``, or
            backslash-containing segment.
    """
    if not isinstance(path, str):
        raise TypeError(f"expected str for path, got {type(path).__name__}: {path!r}")

    spec = path[len(HF_PATH_PREFIX) :] if path.startswith(HF_PATH_PREFIX) else path

    # Split off optional @revision (revision token cannot contain '/' or '@').
    revision: Optional[str] = None
    if "@" in spec:
        spec, revision = spec.rsplit("@", 1)
        if not revision or "/" in revision:
            raise ValueError(
                f"invalid revision in HF checkpoint path: {path!r} "
                f"(expected 'owner/repo[/subfolder][@revision]', got revision={revision!r})"
            )

    parts = [p for p in spec.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"invalid HF checkpoint path: {path!r} "
            f"(expected at least 'owner/repo', got {len(parts)} non-empty segments)"
        )

    # Reject path-traversal segments. Without this, a spec like
    # 'owner/repo/..' would resolve via os.path.join to a directory outside
    # the snapshot root and let downstream loaders read from unintended
    # locations. Backslashes are rejected to block Windows-style traversal.
    for seg in parts:
        if seg in (".", "..") or "\\" in seg:
            raise ValueError(
                f"invalid segment {seg!r} in HF checkpoint path: {path!r} "
                f"('.', '..', and backslashes are not allowed)"
            )

    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, subfolder, revision


def download_hf_checkpoint(
    repo_id: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download a Hugging Face checkpoint snapshot and return the local directory path.

    Thin wrapper over ``huggingface_hub.snapshot_download``. When ``subfolder`` is
    provided, restricts the download to that subtree via ``allow_patterns`` and
    returns the path joined with the subfolder so the caller receives the directory
    layout that the existing local-checkpoint loaders expect.

    Authentication is taken from the standard ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
    environment variables (and the local ``~/.cache/huggingface/token`` cache). For
    multi-node training the token must be available on every node.

    Args:
        repo_id: HF repository identifier in ``owner/repo`` form.
        subfolder: Optional subdirectory within the repo to fetch.
        revision: Optional git revision (branch, tag, or commit SHA).

    Returns:
        Absolute local directory path containing the snapshot (with ``subfolder`` appended
        when set).

    Raises:
        ValueError: If ``repo_id`` is not in ``owner/repo`` form.
        FileNotFoundError: If the snapshot does not contain ``subfolder``.
    """
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise ValueError(f"expected 'owner/repo' for repo_id, got {repo_id!r}")

    allow_patterns: Optional[List[str]] = None
    if subfolder:
        # Match the subfolder itself plus everything beneath it.
        allow_patterns = [f"{subfolder}/*", f"{subfolder}/**"]

    local_root = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
    )

    if subfolder:
        local_path = os.path.join(local_root, subfolder)
        if not os.path.isdir(local_path):
            raise FileNotFoundError(
                f"HF snapshot for repo_id={repo_id!r} (revision={revision!r}) did not "
                f"contain expected subfolder {subfolder!r}; downloaded root={local_root!r}"
            )
        return local_path

    return local_root
