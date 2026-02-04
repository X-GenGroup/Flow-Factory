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

# src/flow_factory/hparams/log_args.py
import os
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
from .abc import ArgABC


@dataclass
class LogArguments(ArgABC):
    r"""Arguments pertaining to data input for training and evaluation."""

    save_dir: str = field(
        default='save',
        metadata={"help": "Directory to save logs and checkpoints. None for no saving."},
    )

    save_freq: int = field(
        default=10,
        metadata={"help": "Model saving frequency (in epochs). 0 for no saving."},
    )
    
    save_model_only : bool = field(
        default=True,
        metadata={"help": "Whether to save the model only, or the complete training state (model and optimizer)."}
    )

    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print detailed progress during training."},
    )

    def __post_init__(self):

        # Expand path to user's path
        self.save_dir = os.path.expanduser(self.save_dir)
        # If save_dir does not exist, create it
        os.makedirs(self.save_dir, exist_ok=True)


    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()