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

# src/flow_factory/hparams/model_args.py
import os
import math
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union, List
from .abc import ArgABC
import logging

import torch

dtype_map = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,    
    'fp32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}

@dataclass
class ModelArguments(ArgABC):
    r"""Arguments pertaining to model configuration."""

    model_name_or_path: str = field(
        default="black-forest-labs/FLUX.1-dev",
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"},
    )

    finetune_type : Literal['full', 'lora'] = field(
        default='full',
        metadata={"help": "Fine-tuning type. Options are ['full', 'lora']"}
    )

    master_weight_dtype : Union[Literal['fp32', 'bf16', 'fp16', 'float16', 'bfloat16', 'float32'], torch.dtype] = field(
        default='bfloat16',
        metadata={'help': "The dtype of master weight for full-parameter traing."}
    )

    target_components : Union[str, List[str]] = field(
        default='transformer',
        metadata={"help": "Which components to fine-tune. Options are like ['transformer', 'transformer_2', ['transformer', 'transformer_2']]"}
    )
    target_modules : Union[str, List[str]] = field(
        default='all',
        metadata={"help": "Which layers to fine-tune. Options are like ['all',  'default', 'to_q', ['to_q', 'to_k', 'to_v']]"}
    )

    model_type: Literal["sd3", "flux1", "flux1-kontext", 'flux2', 'qwenimage', 'qwenimage-edit', 'z-image'] = field(
        default="flux1",
        metadata={"help": "Type of model to use."},
    )

    lora_rank : int = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters."},
    )

    lora_alpha : Optional[int] = field(
        default=None,
        metadata={"help": "Alpha scaling factor for LoRA adapters. Default to `2 * lora_rank` if None."},
    )

    resume_path : Optional[str] = field(
        default=None,
        metadata={"help": "Resume from checkpoint directory."}
    )

    resume_training_state : bool = field(
        default=False,
        metadata={"help": "Whether to resume training state, only effective when resume_path is a directory with full checkpoint."}
    )

    def __post_init__(self):        
        if isinstance(self.master_weight_dtype, str):
            self.master_weight_dtype = dtype_map[self.master_weight_dtype]

        # Normalize target_components to list
        if isinstance(self.target_components, str):
            self.target_components = [self.target_components]


        if isinstance(self.target_modules, str):
            if self.target_modules not in ['all', 'default']:
                self.target_modules = [self.target_modules]

        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_rank

        self.resume_path = os.path.expanduser(self.resume_path) if self.resume_path is not None else None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d['master_weight_dtype'] = str(self.master_weight_dtype).split('.')[-1]
        return d

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()