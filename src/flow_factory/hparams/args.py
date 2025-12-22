# src/flow_factory/hparams/args.py
"""
Main arguments class that encapsulates all configurations.
Supports loading from YAML files with nested structure.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional
import yaml
from datetime import datetime

from .abc import ArgABC
from .data_args import DataArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments
from .reward_args import RewardArguments


@dataclass
class Arguments(ArgABC):
    """Main arguments class encapsulating all configurations."""
    launcher : Literal['accelerate'] = field(
        default='accelerate',
        metadata={"help": "Distributed launcher to use."},
    )
    config_file: str | None = field(
        default=None,
        metadata={"help": "Path to distributed configuration file (e.g., multi_gpu / deepspeed config)."},
    )
    num_processes : int = field(
        default=1,
        metadata={"help": "Number of processes for distributed training."},
    )
    main_process_port : int = field(
        default=29500,
        metadata={"help": "Main process port for distributed training."},
    )
    mixed_precision : Optional[Literal['no', 'fp16', 'bf16']] = field(
        default='bf16',
        metadata={"help": "Mixed precision setting for training."},
    )
    run_name : Optional[str] = field(
        default=None,
        metadata={"help": "Name of the training run. Defaults to a timestamp."},
    )
    project : str = field(
        default='Flow-Factory',
        metadata={"help": "Project name for logging platforms."},
    )
    logging_backend : Optional[Literal['wandb', 'swanlab', 'none']] = field(
        default=None,
        metadata={"help": "Logging backend to use."},
    )
    data_args: DataArguments = field(
        default_factory=DataArguments,
        metadata={"help": "Arguments for data configuration."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={"help": "Arguments for model configuration."},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Arguments for training configuration."},
    )
    reward_args: RewardArguments = field(
        default_factory=RewardArguments,
        metadata={"help": "Arguments for reward model configuration."},
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
            if isinstance(value, ArgABC):
                # Remove '_args' suffix for nested configs
                key = f.name.replace('_args', '')
                result[key] = value.to_dict()
            else:
                result[f.name] = value
        
        return result

    @classmethod
    def from_dict(cls, args_dict: dict[str, Any]) -> Arguments:
        """Create Arguments instance from dictionary."""
        
        # 1. Nested arguments
        nested_args = {
            'data_args': DataArguments(**args_dict.get('data', {})),
            'model_args': ModelArguments(**args_dict.get('model', {})),
            'training_args': TrainingArguments(**args_dict.get('train', {})),
            'reward_args': RewardArguments(**args_dict.get('reward', {})),
        }
        
        # 2. Automatically get all dataclass field names
        # fields(cls) returns all field objects defined in the class
        valid_field_names = {f.name for f in fields(cls)}
        
        # 3. Exclude nested fields that have been manually handled
        # This prevents duplicate or conflicting arguments
        target_keys = valid_field_names - nested_args.keys()
        
        # 4. Automatically filter top_level_args
        # Logic: iterate over the input dictionary, keep keys that exist in the class definition and are not nested parameters
        top_level_args = {
            k: v for k, v in args_dict.items() 
            if k in target_keys
        }
        
        return cls(**top_level_args, **nested_args)

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