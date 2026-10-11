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

"""Training arguments registry and lookup functions."""

from __future__ import annotations

import importlib
from typing import Dict, Type

from ._base import TrainingArguments
from .awm import AWMTrainingArguments
from .crd import CRDTrainingArguments
from .dgpo import DGPOTrainingArguments
from .dmd2 import DMD2TrainingArguments
from .dpo import DPOTrainingArguments
from .dppo import DPPOTrainingArguments
from .grpo import GRPOTrainingArguments
from .nft import NFTTrainingArguments
from .offline_dpo import OfflineDPOTrainingArguments
from .opd import DiffusionOPDTrainingArguments
from .sft import SFTTrainingArguments
from .tdm import TDMTrainingArguments
from .tdm_r1 import TDMR1TrainingArguments

# ============================================================================
# Training Arguments Registry
# ============================================================================

_TRAINING_ARGS_REGISTRY: Dict[str, Type[TrainingArguments]] = {
    "grpo": GRPOTrainingArguments,
    "grpo-guard": GRPOTrainingArguments,
    "dppo": DPPOTrainingArguments,
    "nft": NFTTrainingArguments,
    "awm": AWMTrainingArguments,
    "dgpo": DGPOTrainingArguments,
    "dpo": DPOTrainingArguments,
    "crd": CRDTrainingArguments,
    "dmd2": DMD2TrainingArguments,
    "tdm": TDMTrainingArguments,
    "tdm-r1": TDMR1TrainingArguments,
    "diffusion-opd": DiffusionOPDTrainingArguments,
    "sft": SFTTrainingArguments,
    "offline-dpo": OfflineDPOTrainingArguments,
}


def get_training_args_class(identifier: str) -> Type[TrainingArguments]:
    """Resolve the TrainingArguments subclass for a given trainer type.

    Supports:
    1. Registry lookup: 'grpo' -> GRPOTrainingArguments
    2. Direct python path: 'my_package.hparams.CustomTrainingArgs' -> CustomTrainingArgs

    Raises:
        ImportError: If the identifier is not found in the registry and cannot be resolved
            as a python path to a TrainingArguments subclass.
    """
    identifier_lower = identifier.lower()

    if identifier_lower in _TRAINING_ARGS_REGISTRY:
        return _TRAINING_ARGS_REGISTRY[identifier_lower]

    # Try dynamic import (python path like 'my_package.args.CustomArgs')
    try:
        module_path, class_name = identifier.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if isinstance(cls, type) and issubclass(cls, TrainingArguments):
            return cls
        raise TypeError(
            f"'{identifier}' resolved to {cls}, which is not a TrainingArguments subclass."
        )
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        raise ImportError(
            f"Could not resolve TrainingArguments for trainer_type='{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered trainer: {list(_TRAINING_ARGS_REGISTRY.keys())}\n"
            f"  2. A valid python path to a TrainingArguments subclass\n"
            f"Error: {e}"
        ) from e


def list_registered_training_args() -> Dict[str, Type[TrainingArguments]]:
    """Get all registered training argument classes."""
    return _TRAINING_ARGS_REGISTRY.copy()
