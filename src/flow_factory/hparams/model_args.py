# src/flow_factory/hparams/model_args.py
import os
import math
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union, List
from .abc import ArgABC
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    r"""Arguments pertaining to model configuration."""

    model_name_or_path: str = field(
        default="black-forest-labs/FLUX.1-dev",
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"},
    )

    finetune_type : Literal['full', 'lora'] = field(
        default='full',
        metadata={"help": "Fine-tuning type. Options are ['full', 'lora']"}
    )

    target_modules : Union[str, List[str]] = field(
        default='all',
        metadata={"help": "Which layers to fine-tune. Options are like ['all',  'default', 'to_q']"}
    )

    resume_path : Optional[str] = field(
        default=None,
        metadata={"help": "Resume from checkpoint directory."}
    )

    model_type: Literal["sd3", "flux1", "flux1-kontext", 'flux2', 'qwenimage', 'qwenimage-edit'] = field(
        default="flux1",
        metadata={"help": "Type of model to use."},
    )

    lora_rank : int = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters."},
    )

    lora_alpha : int = field(
        default=16,
        metadata={"help": "Alpha scaling factor for LoRA adapters."},
    )

    def __post_init__(self):
        if isinstance(self.target_modules, str):
            if self.target_modules not in ['all', 'default']:
                self.target_modules = [self.target_modules]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()