# src/flow_factory/hparams/reward_args.py
import os
import math
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Type, Union
import logging
import torch

from .abc import ArgABC

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


dtype_map = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,    
    'fp32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}

@dataclass
class RewardArguments(ArgABC):
    r"""Arguments pertaining to reward configuration."""

    reward_model : str = field(
        default='PickScore',
        metadata={"help": "The path or name of the reward model to use. You can specify 'PickScore' to use the default PickScore model. Or /path/to/your/model:class_name to use your own reward model."},
    )

    dtype: Union[Literal['float16', 'bfloat16', 'float32'], torch.dtype] = field(
        default='bfloat16',
        metadata={"help": "The data type for the reward model."},
        repr=False,
    )

    device: Union[Literal['cpu', 'cuda'], torch.device] = field(
        default='cuda',
        metadata={"help": "The device to load the reward model on."},
        repr=False,
    )

    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for reward model inference."},
    )

    def __post_init__(self):

        if isinstance(self.dtype, str):
            self.dtype = dtype_map[self.dtype]

        if isinstance(self.device, str):
            self.device = torch.device(self.device)        

    def to_dict(self) -> dict[str, Any]:
        # Use super().to_dict() here as well
        d = super().to_dict()
        d['dtype'] = str(self.dtype).split('.')[-1]
        d['device'] = str(self.device)
        return d

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()