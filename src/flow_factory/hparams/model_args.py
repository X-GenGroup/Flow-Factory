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

import logging
import math

# src/flow_factory/hparams/model_args.py
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Optional, Union

import torch
import yaml

from .abc import ArgABC

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

DTypeName = Literal["fp32", "bf16", "fp16", "float16", "bfloat16", "float32"]
DTypeValue = Union[DTypeName, torch.dtype]
DTypePolicy = Optional[Union[DTypeValue, dict[str, Optional[DTypeValue]]]]


def _normalize_dtype_policy(
    value: DTypePolicy,
    *,
    field_name: str,
    inject_default_null: bool,
) -> DTypePolicy:
    """Normalize one scalar or selector-based dtype policy."""
    if isinstance(value, str):
        if value not in dtype_map:
            raise ValueError(
                f"expected model.{field_name} as a known dtype name, "
                f"received {value!r}; expected one of {tuple(dtype_map)}"
            )
        return dtype_map[value]
    if isinstance(value, Mapping):
        normalized_policy: dict[str, Optional[torch.dtype]] = {}
        for selector, configured_dtype in value.items():
            if not isinstance(selector, str) or not selector:
                raise TypeError(
                    f"expected every model.{field_name} selector to be a non-empty str, "
                    f"received {type(selector).__name__}: {selector!r}"
                )
            if configured_dtype is None:
                normalized_policy[selector] = None
            elif isinstance(configured_dtype, torch.dtype):
                normalized_policy[selector] = configured_dtype
            elif isinstance(configured_dtype, str):
                if configured_dtype not in dtype_map:
                    raise ValueError(
                        f"expected model.{field_name}[{selector!r}] as a known dtype "
                        f"name or null, received {configured_dtype!r}; expected one of "
                        f"{tuple(dtype_map)}"
                    )
                normalized_policy[selector] = dtype_map[configured_dtype]
            else:
                raise TypeError(
                    f"expected model.{field_name}[{selector!r}] as str, torch.dtype, "
                    f"or None, received {type(configured_dtype).__name__}: "
                    f"{configured_dtype!r}"
                )
        if inject_default_null:
            normalized_policy.setdefault("default", None)
        return normalized_policy
    if value is not None and not isinstance(value, torch.dtype):
        raise TypeError(
            f"expected model.{field_name} as a dtype, mapping, or None, "
            f"received {type(value).__name__}: {value!r}"
        )
    return value


def _serialize_dtype_policy(value: DTypePolicy) -> Any:
    """Serialize one normalized dtype policy for YAML output."""
    if isinstance(value, dict):
        return {
            selector: (None if configured_dtype is None else str(configured_dtype).split(".")[-1])
            for selector, configured_dtype in value.items()
        }
    if value is not None:
        return str(value).split(".")[-1]
    return None


@dataclass
class ModelArguments(ArgABC):
    r"""Arguments pertaining to model configuration."""

    model_name_or_path: str = field(
        default="black-forest-labs/FLUX.1-dev",
        metadata={
            "help": "Path to pre-trained model or model identifier from huggingface.co/models"
        },
    )

    finetune_type: Literal["full", "lora"] = field(
        default="full", metadata={"help": "Fine-tuning type. Options are ['full', 'lora']"}
    )

    trainable_parameters_dtype: DTypeValue = field(
        default="bfloat16",
        metadata={
            "help": "Torch dtype for all trainable parameters (`requires_grad=True`) -- i.e. the "
            "optimizer 'master weight' precision. (Renamed from the misnamed "
            "`master_weight_dtype`, which despite its name only ever set the *trainable* "
            "parameter dtype.)"
        },
    )
    component_load_dtypes: DTypePolicy = field(
        default=None,
        metadata={
            "help": "Optional pretrained-component load dtype policy. A scalar applies to every "
            "loadable component. A mapping supports `default`, component groups such as "
            "`transformers`, and concrete component names. `None` delegates to the adapter's "
            "model-specific manifest; an explicit null mapping value delegates that selector "
            "to its native loader."
        },
    )
    frozen_parameters_dtype: DTypePolicy = field(
        default=None,
        metadata={
            "help": "Frozen-parameter dtype policy. A scalar dtype applies to every frozen "
            "component for backward compatibility. A mapping supports `default`, component "
            "groups such as `transformers`, and concrete component names; concrete names "
            "override groups, which override `default`. `None` or `default: null` performs no "
            "post-load dtype mutation for unmatched components."
        },
    )

    target_components: Union[str, List[str]] = field(
        default="transformer",
        metadata={
            "help": "Which components to fine-tune. Options are like ['transformer', 'transformer_2', ['transformer', 'transformer_2']]"
        },
    )
    target_modules: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "Which layers to fine-tune. Options are like ['all',  'default', 'to_q', ['to_q', 'to_k', 'to_v']]"
        },
    )

    model_type: Literal[
        "sd3-5",
        "flux1",
        "flux1-kontext",
        "flux2",
        "flux2-klein",
        "qwen-image",
        "qwen-image-edit-plus",
        "qwen-image-2.1",
        "z-image",
        "wan2_t2v",
        "wan2_i2v",
        "bagel",
        "sensenova",
        "ltx2_t2av",
        "ltx2_i2av",
        "minimax-h3-t2va",
        "minimax-h3-fl2va",
        "minimax-h3-ref2va",
    ] = field(
        default="flux1",
        metadata={
            "help": "Registered model adapter key (see models/registry.py), or a custom 'pkg.module.Adapter' python path."
        },
    )

    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters."},
    )

    lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": "Alpha scaling factor for LoRA adapters. Default to `2 * lora_rank` if None."
        },
    )

    resume_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Resume from checkpoint. Accepts either a local directory or a "
            "Hugging Face repo spec ('owner/repo[/subfolder][@revision]', or "
            "explicit 'hf://owner/repo[/subfolder][@revision]'). When a local "
            "path doesn't exist, falls back to Hugging Face Hub download. "
            "Multi-node: HF_TOKEN must be set on every node; downloads happen "
            "once per node; consider HF_HUB_ENABLE_HF_TRANSFER=1 for large "
            "checkpoints to avoid NCCL watchdog timeouts."
        },
    )

    resume_type: Optional[Literal["lora", "full", "state"]] = field(
        default=None,
        metadata={
            "help": "Type of checkpoint to load from resume_path. "
            "'lora': Load LoRA adapters only. "
            "'full': Load full model weights. "
            "'state': Load full training state (model + optimizer). "
            "If None, auto-detect based on finetune_type."
        },
    )

    def __post_init__(self):
        if "attn_backend" in self.extra_kwargs:
            raise ValueError(
                "`model.attn_backend` has been removed. Attention-backend selection now lives in "
                "the acceleration layer as a `shared` accelerator. Replace it with:\n"
                "  acceleration:\n"
                "    shared:\n"
                "      - name: attention_backend\n"
                f"        params: {{ backend: {self.extra_kwargs['attn_backend']!r} }}\n"
                "See guidance/acceleration.md. (Bagel forces flash_attention_2 at load and ignores "
                "this knob — just drop the line.)"
            )

        if isinstance(self.trainable_parameters_dtype, str):
            self.trainable_parameters_dtype = dtype_map[self.trainable_parameters_dtype]
        self.component_load_dtypes = _normalize_dtype_policy(
            self.component_load_dtypes,
            field_name="component_load_dtypes",
            inject_default_null=False,
        )
        self.frozen_parameters_dtype = _normalize_dtype_policy(
            self.frozen_parameters_dtype,
            field_name="frozen_parameters_dtype",
            inject_default_null=True,
        )

        # Normalize target_components to list
        if isinstance(self.target_components, str):
            self.target_components = [self.target_components]

        if isinstance(self.target_modules, str):
            if self.target_modules not in ["all", "default"]:
                self.target_modules = [self.target_modules]

        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_rank

        self.resume_path = (
            os.path.expanduser(self.resume_path) if self.resume_path is not None else None
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["trainable_parameters_dtype"] = str(self.trainable_parameters_dtype).split(".")[-1]
        d["component_load_dtypes"] = _serialize_dtype_policy(self.component_load_dtypes)
        d["frozen_parameters_dtype"] = _serialize_dtype_policy(self.frozen_parameters_dtype)
        return d

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()
