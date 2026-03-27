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
from .training_args import TrainingArguments, EvaluationArguments, get_training_args_class
from .reward_args import RewardArguments, MultiRewardArguments
from .log_args import LogArguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


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
        if self.log_args.run_name is None:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_args.run_name = f"{self.model_args.model_type}_{self.model_args.finetune_type}_{self.training_args.trainer_type}_{time_stamp}"

        self._resolve_scheduler_sde_defaults()

        # Adjust gradient accumulation for per-timestep losses
        num_train_timesteps = self.training_args.get_num_train_timesteps(self)
        self.training_args.gradient_accumulation_steps *= num_train_timesteps

        self._resolve_async_reward()

    def _resolve_async_reward(self) -> None:
        """Detect if any reward model uses async_reward and adjust sampler constraints."""
        import math
        from .training_args import get_world_size

        all_configs = list(self.reward_args or [])
        if self.eval_reward_args:
            all_configs += list(self.eval_reward_args)

        self._has_async_rewards = any(getattr(cfg, 'async_reward', False) for cfg in all_configs)

        if self._has_async_rewards:
            # Auto-adjust `unique_sample_num` to ensure it is divisible by `num_replicas` for GroupContiguousSampler.
            world_size = get_world_size()
            ta = self.training_args
            sample_num_per_iteration = world_size * ta.per_device_batch_size
            old_step = (sample_num_per_iteration * ta.gradient_step_per_epoch) // math.gcd(ta.group_size, sample_num_per_iteration)
            step = math.lcm(old_step, world_size)
            new_m = (ta.unique_sample_num_per_epoch + step - 1) // step * step
            if new_m != ta.unique_sample_num_per_epoch:
                logger.warning(
                    f"Async reward detected. Adjusted `unique_sample_num` from {ta.unique_sample_num_per_epoch} to {new_m} "
                    f"to ensure `unique_sample_num` is divisible by `num_replicas` ({world_size}) for GroupContiguousSampler."
                )
                ta.unique_sample_num_per_epoch = new_m
                ta.num_batches_per_epoch = (new_m * ta.group_size) // sample_num_per_iteration
                ta.gradient_accumulation_steps = max(1, ta.num_batches_per_epoch // ta.gradient_step_per_epoch)

    def _resolve_scheduler_sde_defaults(self) -> None:
        """Fill `sde_steps` / `num_sde_steps` when YAML uses null.

        Matches runtime SDE schedulers: default step indices are
        ``0 .. num_inference_steps-2`` (all steps except the last). When
        ``num_sde_steps`` is null, use the full resolved pool (same as the
        scheduler property default).
        """
        sched = self.scheduler_args
        n_inf = self.training_args.num_inference_steps
        if sched.sde_steps is None:
            sched.sde_steps = list(range(max(0, n_inf - 1)))
        if sched.num_sde_steps is None:
            sched.num_sde_steps = len(sched.sde_steps)
        if sched.num_sde_steps <= 0:
            raise ValueError(
                "scheduler.num_sde_steps must be positive after resolving nulls; "
                f"got num_sde_steps={sched.num_sde_steps!r}, sde_steps={sched.sde_steps!r}, "
                f"num_inference_steps={n_inf!r}."
            )

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

        # 1. Resolve TrainingArguments subclass based on trainer_type
        train_dict = args_dict.get('train', {})
        trainer_type = train_dict.get('trainer_type', 'grpo')
        training_args_cls = get_training_args_class(trainer_type)

        # 2. Nested arguments map
        nested_map = {
            'data': ('data_args', DataArguments),
            'model': ('model_args', ModelArguments),
            'scheduler': ('scheduler_args', SchedulerArguments),
            'train': ('training_args', training_args_cls),
            'eval': ('eval_args', EvaluationArguments),
            'log': ('log_args', LogArguments),
            'rewards': ('reward_args', MultiRewardArguments),
            'eval_rewards': ('eval_reward_args', MultiRewardArguments),
        }

        # 3. Build init kwargs
        init_kwargs = {}
        extras = {}
        
        valid_field_names = {f.name for f in fields(cls)}

        for k, v in args_dict.items():
            if k in nested_map:
                arg_name, arg_cls = nested_map[k]
                init_kwargs[arg_name] = arg_cls.from_dict(v)
            
            elif k in valid_field_names:
                init_kwargs[k] = v
            
            else:
                extras[k] = v

        # 4. Handle explicit 'extra_kwargs' if present in YAML and merge
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