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

"""Configuration for one ``ff-prompt-enhance`` run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import yaml

from ..hparams.abc import ArgABC
from ..hparams.model_args import dtype_map


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"expected {name} to be a positive int, " f"received {type(value).__name__}: {value!r}"
        )


@dataclass(kw_only=True)
class PromptEnhanceArguments(ArgABC):
    """Arguments for rewriting a prompt dataset with one enhancer model.

    Attributes:
        enhancer: Registered enhancer key (``flux2``, ``ltx2``, ``flux3``,
            ``qwen-image-2.1-t2i``, ``qwen-image-2.1-i2i``, ``openai-compatible``) or a
            ``package.module.Class`` python path.
        input_dir: Source dataset directory holding ``{split}.jsonl`` or ``{split}.txt``.
        output_dir: Destination directory of the enhanced dataset.
        model_name_or_path: Enhancer model repo, local path, or served model name.
        splits: Splits to enhance; ``None`` selects every ``train``/``test`` split present.
        num_variants: Enhanced prompts written per source prompt.
        batch_size: Prompts handed to the enhancer per call.
        seed: Base of the per-row generation seeds.
        dtype: Load dtype of the enhancer model.
        system_prompt: Overrides the enhancer's default system prompt when set.
        generation: Enhancer-specific generation arguments.
        num_processes, num_machines, machine_rank, main_process_ip, main_process_port,
        config_file, mixed_precision: Launcher settings read by ``ff-prompt-enhance``.
    """

    enhancer: str
    input_dir: str
    output_dir: str
    model_name_or_path: Optional[str] = None
    splits: Optional[List[str]] = None
    num_variants: int = 1
    batch_size: int = 8
    seed: int = 42
    dtype: str = "bfloat16"
    system_prompt: Optional[str] = None
    generation: Dict[str, Any] = field(default_factory=dict)
    num_processes: Optional[int] = None
    num_machines: Optional[int] = None
    machine_rank: Optional[int] = None
    main_process_ip: Optional[str] = None
    main_process_port: Optional[int] = None
    config_file: Optional[str] = None
    mixed_precision: str = "no"

    def __post_init__(self) -> None:
        if not isinstance(self.enhancer, str) or not self.enhancer:
            raise ValueError(f"expected a non-empty enhancer, received {self.enhancer!r}")
        _require_positive_int(self.num_variants, "num_variants")
        _require_positive_int(self.batch_size, "batch_size")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"expected seed to be a non-negative int, received {self.seed!r}")
        if self.dtype not in dtype_map:
            raise ValueError(f"expected dtype in {tuple(dtype_map)}, received {self.dtype!r}")
        if not isinstance(self.generation, dict):
            raise TypeError(
                "expected generation to be a mapping, "
                f"received {type(self.generation).__name__}: {self.generation!r}"
            )
        self.input_dir = os.path.expanduser(self.input_dir)
        self.output_dir = os.path.expanduser(self.output_dir)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the resolved enhancer load dtype."""
        return dtype_map[self.dtype]

    @classmethod
    def load_from_yaml(cls, yaml_file: str) -> "PromptEnhanceArguments":
        """Load prompt-enhancement arguments from a YAML file."""
        with open(yaml_file, "r", encoding="utf-8") as f:
            args_dict = yaml.safe_load(f)
        if not isinstance(args_dict, dict):
            raise TypeError(
                f"expected a YAML mapping in {yaml_file}, received {type(args_dict).__name__}"
            )
        return cls.from_dict(args_dict)
