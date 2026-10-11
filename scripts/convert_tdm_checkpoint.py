#!/usr/bin/env python3
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

"""Convert an official TDM / TDM-R1 LoRA release into a Flow-Factory checkpoint.

The upstream releases (``Luo-Yihong/TDM_sd3-5_lora``, ``Luo-Yihong/TDM-R1``, and the
per-run ``*_ema.ckpt`` files the training scripts write) do not ship parameter names.
They store ``{"decay": float, "ema_parameters": [Tensor, ...]}``, a plain list whose
order is whatever ``transformer.named_parameters()`` yielded for the trainable LoRA
weights of one adapter. Loading it therefore means rebuilding the identical PEFT
wrapper and zipping the two sequences together, which is what the upstream
``load_ema`` helper does inline and what this script does once, writing a named
``adapter_model.safetensors`` that ``PeftModel.from_pretrained`` -- and so
``model.resume_path`` -- can read.

Because the list is positional and every LoRA tensor in a given release has the same
pair of shapes, a wrong module order would load without complaint and train from
subtly wrong weights. The rebuild is checked against the release for count, for
per-tensor shape, and for the alternating ``lora_A``/``lora_B`` structure PEFT emits,
so a mismatched target-module set fails here instead of silently degrading a run.

Usage:
    python scripts/convert_tdm_checkpoint.py \\
        --checkpoint tdm_sd3-5_lora.ckpt \\
        --model stabilityai/stable-diffusion-3.5-medium \\
        --output /path/to/tdm_sd35_lora

    # then, in a training config:
    #   model:
    #     resume_path: /path/to/tdm_sd35_lora
    #     resume_type: lora
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

# The attention projections the official TDM and TDM-R1 recipes adapt. Flow-Factory's
# SD3.5 adapter defaults to this same set, so a run resumed from the converted weights
# sees the modules the release was trained on.
DEFAULT_TARGET_MODULES: Tuple[str, ...] = (
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
)


def read_release_parameters(checkpoint_path: str) -> List[torch.Tensor]:
    """Read the positional EMA parameter list out of an official release.

    Args:
        checkpoint_path: Path to the ``.ckpt`` file.

    Returns:
        The release's LoRA tensors in their stored order.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        TypeError: If the file is not the expected mapping of tensors.
        ValueError: If the parameter list is missing or empty.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"expected a TDM checkpoint file at {checkpoint_path!r}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            "expected the TDM checkpoint to hold a dict with an 'ema_parameters' entry, "
            f"received {type(payload).__name__} from {checkpoint_path!r}"
        )
    if "ema_parameters" not in payload:
        raise ValueError(
            "expected key 'ema_parameters' in the TDM checkpoint, received keys "
            f"{sorted(payload)} from {checkpoint_path!r}"
        )
    parameters = payload["ema_parameters"]
    if not isinstance(parameters, (list, tuple)) or not parameters:
        raise ValueError(
            "expected a non-empty list of tensors under 'ema_parameters', received "
            f"{type(parameters).__name__} of length "
            f"{len(parameters) if hasattr(parameters, '__len__') else 'unknown'}"
        )
    for index, tensor in enumerate(parameters):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "expected every entry of 'ema_parameters' to be a torch.Tensor, received "
                f"{type(tensor).__name__} at index {index}"
            )
    return [tensor.detach().cpu() for tensor in parameters]


def build_reference_names(
    model_name_or_path: str,
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules: Sequence[str],
) -> List[Tuple[str, torch.Size]]:
    """Rebuild the release's PEFT wrapper to recover the names its list omitted.

    The transformer is instantiated from its config on the meta device: only module
    names and shapes are needed, and downloading multi-gigabyte weights to read them
    would make this script far slower than the conversion warrants.

    Args:
        model_name_or_path: Diffusers repo id or local path of the base model.
        lora_rank: LoRA rank the release was trained with.
        lora_alpha: LoRA alpha the release was trained with.
        target_modules: Modules the release adapted.

    Returns:
        Trainable LoRA parameter names with their shapes, in PEFT's own order.

    Raises:
        ValueError: If the rebuilt wrapper exposes no trainable LoRA parameters.
    """
    from accelerate import init_empty_weights
    from diffusers import SD3Transformer2DModel

    config = SD3Transformer2DModel.load_config(model_name_or_path, subfolder="transformer")
    with init_empty_weights():
        transformer = SD3Transformer2DModel.from_config(config)
        wrapped = get_peft_model(
            transformer,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=list(target_modules),
            ),
        )
    named = [
        (name, parameter.shape)
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    ]
    if not named:
        raise ValueError(
            f"expected trainable LoRA parameters after wrapping {model_name_or_path!r} with "
            f"target_modules={list(target_modules)}, received none; the target modules "
            "probably do not match this architecture"
        )
    return named


def align(
    release: Sequence[torch.Tensor],
    reference: Sequence[Tuple[str, torch.Size]],
) -> Dict[str, torch.Tensor]:
    """Name the release's tensors by position, refusing any layout that disagrees.

    Args:
        release: Tensors in the order the release stored them.
        reference: Names and shapes in the order PEFT produces them.

    Returns:
        Named state dict ready to be written as a PEFT adapter.

    Raises:
        ValueError: If the two sequences differ in length or in any tensor's shape.
    """
    if len(release) != len(reference):
        raise ValueError(
            f"expected the rebuilt LoRA to have the release's {len(release)} trainable "
            f"parameters, received {len(reference)}. The base model, lora_rank, or "
            "target_modules do not match the release."
        )
    state_dict: Dict[str, torch.Tensor] = {}
    for index, (tensor, (name, shape)) in enumerate(zip(release, reference)):
        if tensor.shape != shape:
            raise ValueError(
                f"expected shape {tuple(shape)} for {name!r} at position {index}, received "
                f"{tuple(tensor.shape)}. The rebuilt module order does not match the release."
            )
        # PEFT emits `base_model.model.<module>.lora_A.default.weight`; the saved adapter
        # drops the adapter name, which is how `from_pretrained` expects to read it back.
        state_dict[name.replace(".default.weight", ".weight")] = tensor
    lora_a = sum(1 for name in state_dict if ".lora_A." in name)
    lora_b = sum(1 for name in state_dict if ".lora_B." in name)
    if lora_a == 0 or lora_a != lora_b:
        raise ValueError(
            f"expected matching lora_A and lora_B counts, received {lora_a} and {lora_b}; "
            "the rebuilt wrapper does not look like a LoRA"
        )
    return state_dict


def write_adapter(
    output_dir: str,
    state_dict: Dict[str, torch.Tensor],
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules: Sequence[str],
) -> None:
    """Write a flat PEFT adapter directory.

    Args:
        output_dir: Directory to create.
        state_dict: Named LoRA tensors.
        lora_rank: LoRA rank to record.
        lora_alpha: LoRA alpha to record.
        target_modules: Modules to record.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_file(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA",
        "task_type": None,
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "inference_mode": False,
        "init_lora_weights": "gaussian",
        "target_modules": list(target_modules),
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as handle:
        json.dump(adapter_config, handle, indent=2)


def main() -> None:
    """Convert one official release into a Flow-Factory-readable adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Official .ckpt to convert.")
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Base model the release was trained on.",
    )
    parser.add_argument("--output", required=True, help="Directory to write the adapter to.")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=list(DEFAULT_TARGET_MODULES),
        help="Modules the release adapted.",
    )
    args = parser.parse_args()

    release = read_release_parameters(args.checkpoint)
    print(f"read {len(release)} tensors from {args.checkpoint}")
    reference = build_reference_names(
        args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    state_dict = align(release, reference)
    write_adapter(
        args.output,
        state_dict,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    print(f"wrote {len(state_dict)} named tensors to {args.output}")
    print("set model.resume_path to this directory with model.resume_type: lora")


if __name__ == "__main__":
    main()
