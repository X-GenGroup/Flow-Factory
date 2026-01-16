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

# src/flow_factory/hparams/reward_args.py
"""
Reward Model Arguments Configuration.

Supports both single reward and multi-reward configurations.
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Optional, List, Union, Dict
import torch

from .abc import ArgABC


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
    """
    Arguments pertaining to a single reward model configuration.
    
    Attributes:
        name: Unique identifier for this reward configuration. Used for logging,
            deduplication, and referencing in eval_reward_names.
        reward_model: The path or name of the reward model. Can be:
            - A registered name like 'PickScore'
            - A python path like 'my_package.rewards.CustomReward'
        weight: Weight for reward aggregation (reserved for future multi-reward aggregation).
        dtype: Data type for the reward model inference.
        device: Device to load the reward model on.
        batch_size: Batch size for reward model inference.
    
    Examples:
        >>> args = RewardArguments(name="aesthetic", reward_model="PickScore")
        >>> args = RewardArguments(
        ...     name="custom",
        ...     reward_model="my_package.rewards.ImageReward",
        ...     batch_size=32,
        ...     model_path="/path/to/model"  # goes to extra_kwargs
        ... )
    """

    name: str = field(
        default="default",
        metadata={"help": "Unique identifier for this reward configuration."},
    )

    reward_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path or name of the reward model to use. "
                    "You can specify 'PickScore' to use the registered PickScore model, "
                    "or /path/to/your/model:class_name to use your own reward model."
        },
    )

    weight: float = field(
        default=1.0,
        metadata={"help": "Weight for reward aggregation (reserved for future use)."},
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

    def get_identity_key(self) -> tuple:
        """
        Generate a unique identity key for deduplication.
        
        Two RewardArguments with the same identity key can share the same
        loaded model instance, even if they have different names or weights.
        
        Returns:
            A tuple that uniquely identifies the model configuration.
        """
        return (
            self.reward_model,
            str(self.dtype),
            str(self.device),
            tuple(sorted(self.extra_kwargs.items()))
        )

    def __hash__(self):
        return hash(self.get_identity_key())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with proper type serialization."""
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

    def __eq__(self, other):
        """
        Compare RewardArguments instances considering all fields including extra_kwargs.
        Handles torch.dtype and torch.device comparison properly.
        """
        if not isinstance(other, RewardArguments):
            return False
        
        core_fields = [
            'name',
            'reward_model',
            'dtype', 'device',
            'batch_size',
            'weight',
        ]
        # Compare core fields
        core_equal = all(
            getattr(self, f) == getattr(other, f)
            for f in core_fields
        )
        
        if not core_equal:
            return False
        
        # Compare extra_kwargs
        return self.extra_kwargs == other.extra_kwargs


@dataclass
class MultiRewardArguments(ArgABC):
    """
    Container for multiple reward model configurations.
    
    Supports iteration, indexing, and lookup by name. Provides a unified
    interface for managing multiple reward models in training and evaluation.
    
    Attributes:
        rewards: List of RewardArguments configurations.
    
    Examples:
        >>> multi_args = MultiRewardArguments(rewards=[
        ...     RewardArguments(name="aesthetic", reward_model="PickScore"),
        ...     RewardArguments(name="safety", reward_model="SafetyReward"),
        ... ])
        >>> multi_args.get_by_name("aesthetic")
        RewardArguments(name="aesthetic", ...)
        >>> len(multi_args)
        2
        >>> for args in multi_args:
        ...     print(args.name)
    
    YAML Configuration Example:
        ```yaml
        rewards:
          - name: "aesthetic"
            reward_model: "PickScore"
            weight: 1.0
            batch_size: 16
          
          - name: "text_align"
            reward_model: "CLIPScore"
            weight: 0.5
            batch_size: 32
        ```
    """

    reward_configs: List[RewardArguments] = field(default_factory=list)

    def get_by_name(self, name: str) -> Optional[RewardArguments]:
        """
        Retrieve a reward configuration by its unique name.
        
        Args:
            name: The name of the reward configuration to find.
        
        Returns:
            The matching RewardArguments, or None if not found.
        """
        return next((r for r in self.reward_configs if r.name == name), None)

    def get_names(self) -> List[str]:
        """
        Get all reward names.
        
        Returns:
            List of reward configuration names.
        """
        return [r.name for r in self.reward_configs]

    def __len__(self) -> int:
        """Return the number of reward configurations."""
        return len(self.reward_configs)

    def __iter__(self) -> Iterator[RewardArguments]:
        """Iterate over reward configurations."""
        return iter(self.reward_configs)

    def __getitem__(self, index: int) -> RewardArguments:
        """Get reward configuration by index."""
        return self.reward_configs[index]

    def __bool__(self) -> bool:
        """Return True if there are any reward configurations."""
        return len(self.reward_configs) > 0

    @classmethod
    def from_dict(cls, args_input: Union[Dict, List]) -> MultiRewardArguments:
        """
        Create MultiRewardArguments from a dictionary or list.
        
        Handles multiple input formats:
        - List format: [{...}, {...}]  (from YAML rewards: [...])
        - Dict format: {name: ..., reward_model: ...}  (single reward shorthand)
        
        Args:
            args_dict: List of reward configs or dict that indicates one single reward.
        
        Returns:
            MultiRewardArguments instance.
        """
        if isinstance(args_input, list):
            reward_configs = [RewardArguments.from_dict(r) for r in args_input]
            return cls(reward_configs=reward_configs)
        elif isinstance(args_input, dict):
            # Single reward dict format
            reward_configs = [RewardArguments.from_dict(args_input)]
            return cls(reward_configs=reward_configs)
        else:
            raise ValueError("Invalid input for MultiRewardArguments.from_dict")

    def to_list(self) -> List[dict[str, Any]]:
        """
        Convert to list of dictionaries for each reward configuration.
        
        Returns:
            List of reward configuration dictionaries.
        """
        return [r.to_dict() for r in self.reward_configs]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            f"reward_{i}": r.to_dict() for i, r in enumerate(self.reward_configs)
        }

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()