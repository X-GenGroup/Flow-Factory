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

# src/flow_factory/hparams/args.py
"""
Main arguments class that encapsulates all configurations.

Supports loading from YAML files with nested structure.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional
import yaml
from datetime import datetime

from .abc import ArgABC
from .data_args import DataArguments
from .model_args import ModelArguments
from .scheduler_args import SchedulerArguments
from .training_args import TrainingArguments, EvaluationArguments
from .reward_args import RewardArguments, MultiRewardArguments
from .log_args import LogArguments


@dataclass
class Arguments(ArgABC):
    """
    Main arguments class encapsulating all configurations.
    """
    
    launcher: Literal['accelerate'] = field(
        default='accelerate',
        metadata={"help": "Distributed launcher to use."},
    )
    config_file: str | None = field(
        default=None,
        metadata={"help": "Path to distributed configuration file."},
    )
    num_processes: int = field(
        default=1,
        metadata={"help": "Number of processes for distributed training."},
    )
    main_process_port: int = field(
        default=29500,
        metadata={"help": "Main process port for distributed training."},
    )
    mixed_precision: Optional[Literal['no', 'fp16', 'bf16']] = field(
        default='bf16',
        metadata={"help": "Mixed precision setting for training."},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the training run."},
    )
    project: str = field(
        default='Flow-Factory',
        metadata={"help": "Project name for logging platforms."},
    )
    logging_backend: Optional[Literal['wandb', 'swanlab', 'none']] = field(
        default=None,
        metadata={"help": "Logging backend to use."},
    )
    
    # Nested argument groups
    data_args: DataArguments = field(
        default_factory=DataArguments,
        metadata={"help": "Arguments for data configuration."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={"help": "Arguments for model configuration."},
    )
    scheduler_args: SchedulerArguments = field(
        default_factory=SchedulerArguments,
        metadata={"help": "Arguments for scheduler configuration."},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Arguments for training configuration."},
    )
    eval_args: EvaluationArguments = field(
        default_factory=EvaluationArguments,
        metadata={"help": "Arguments for evaluation configuration."},
    )
    log_args: LogArguments = field(
        default_factory=LogArguments,
        metadata={"help": "Arguments for logging configuration."},
    )
    reward_args: MultiRewardArguments = field(
        default_factory=MultiRewardArguments,
        metadata={"help": "Arguments for multiple reward configurations."},
    )
    eval_reward_args: Optional[MultiRewardArguments] = field(
        default=None,
        metadata={"help": "Arguments for multiple evaluation reward configurations."},
    )

    def __post_init__(self):
        if self.run_name is None:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.model_args.model_type}_{self.model_args.finetune_type}_{time_stamp}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, ArgABC):
                # Remove '_args' suffix for nested configs
                key = f.name.replace('_args', '')
                result[key] = value.to_dict()
            else:
                result[f.name] = value

        extras = result.pop("extra_kwargs", {})
        result.update(extras)
        return result

    @classmethod
    def from_dict(cls, args_dict: dict[str, Any]) -> Arguments:
        """Create Arguments instance from dictionary."""

        # 1. Nested arguments map
        # Define which keys in the YAML correspond to which nested dataclasses
        nested_map = {
            'data': ('data_args', DataArguments),
            'model': ('model_args', ModelArguments),
            'scheduler': ('scheduler_args', SchedulerArguments),
            'train': ('training_args', TrainingArguments),
            'eval': ('eval_args', EvaluationArguments),
            'log': ('log_args', LogArguments),
            'rewards': ('reward_args', MultiRewardArguments),
            'eval_rewards': ('eval_reward_args', MultiRewardArguments),
        }

        # 2. Build init kwargs
        init_kwargs = {}
        extras = {} # To collect unknown top-level keys
        
        # Get all valid field names for Arguments (including 'extra_kwargs' from base)
        valid_field_names = {f.name for f in fields(cls)}

        for k, v in args_dict.items():
            # Case A: It is a nested config block (e.g., "data": {...})
            if k in nested_map:
                arg_name, arg_cls = nested_map[k]
                # Use the nested class's from_dict to handle its own kwargs
                # For `MultiRewardArguments`, from_dict can handle list/dict - to handle multi/single reward configs
                init_kwargs[arg_name] = arg_cls.from_dict(v)
            
            # Case B: It is a known top-level field (e.g., "run_name", "launcher")
            elif k in valid_field_names:
                init_kwargs[k] = v
            
            # Case C: It is unknown -> send to extra_kwargs bucket
            else:
                extras[k] = v

        # 3. Handle explicit 'extra_kwargs' if present in YAML and merge
        if "extra_kwargs" in init_kwargs:
            extras.update(init_kwargs["extra_kwargs"])
        
        init_kwargs["extra_kwargs"] = extras
        
        return cls(**init_kwargs)

    @classmethod
    def load_from_yaml(cls, yaml_file: str) -> Arguments:
        """
        Load Arguments from a YAML configuration file.
        Example: args = Arguments.load_from_yaml("config.yaml")
        """
        with open(yaml_file, 'r', encoding='utf-8') as f:
            args_dict = yaml.safe_load(f)
        
        return cls.from_dict(args_dict)
    
    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()