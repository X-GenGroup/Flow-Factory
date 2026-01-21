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

# src/flow_factory/utils/checkpoint.py
"""
Utility functions for handling checkpoint management.
"""
import os
import re
import glob
import json
import torch
from typing import Dict, Optional, List, Tuple, Literal

from safetensors.torch import save_file, load_file

def mapping_lora_state_dict(
        state_dict: Dict[str, torch.Tensor],
        adapter_name: str = "default"
    ) -> Dict[str, torch.Tensor]:
    """
    Map LoRA state_dict keys to PeftModel format.
    Converts 'xxx.lora_A.weight' -> 'base_model.model.xxx.lora_A.default.weight'
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('base_model.model'):
            key = 'base_model.model.' + key
        if "lora_A.weight" in key or "lora_B.weight" in key:
            new_key = key.replace("lora_A.weight", f"lora_A.{adapter_name}.weight").replace("lora_B.weight", f"lora_B.{adapter_name}.weight")
            new_state_dict[new_key] = value
        else:
            # Keep other keys as-is
            new_state_dict[key] = value
    return new_state_dict


# ================================ Config Inference ================================
def infer_lora_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Infer LoRA rank from state dict.
    
    Args:
        state_dict: LoRA state dictionary
    
    Returns:
        Inferred rank value
    
    Raises:
        ValueError: If no lora_A/lora_B weights found
    """
    # Try lora_A first (shape: [rank, in_features])
    for key, tensor in state_dict.items():
        if "lora_A" in key and "weight" in key:
            return tensor.shape[0]
    
    # Fallback to lora_B (shape: [out_features, rank])
    for key, tensor in state_dict.items():
        if "lora_B" in key and "weight" in key:
            return tensor.shape[1]
    
    raise ValueError("Cannot infer rank: no lora_A or lora_B weights found")


def infer_lora_alpha(state_dict: Dict[str, torch.Tensor], default_rank: Optional[int] = None) -> int:
    """
    Infer LoRA alpha from state dict, defaulting to rank.
    
    Args:
        state_dict: LoRA state dictionary
        default_rank: Fallback if alpha not found (uses inferred rank if None)
    
    Returns:
        Inferred or default alpha value
    """
    for key, tensor in state_dict.items():
        if "lora_alpha" in key.lower() or "scaling" in key.lower():
            return int(tensor.item())
    
    return default_rank or infer_lora_rank(state_dict)


def infer_lora_config(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Infer both rank and alpha from state dict.
    
    Args:
        state_dict: LoRA state dictionary
    
    Returns:
        Tuple of (rank, alpha)
    """
    rank = infer_lora_rank(state_dict)
    alpha = infer_lora_alpha(state_dict, default_rank=rank)
    return rank, alpha


def infer_target_modules(
    state_dict: Dict[str, torch.Tensor],
    prefix: Optional[str] = None,
) -> List[str]:
    """
    Infer full module paths from state dict (for precise LoRA targeting).
    
    Args:
        state_dict: LoRA state dictionary
        prefix: Optional prefix to strip from paths
    
    Returns:
        Sorted list of full module paths
    """
    # Auto-detect prefix
    if prefix is None:
        first_key = next(iter(state_dict.keys()), "")
        for p in ("transformer.", "unet.", "text_encoder.", "base_model.model."):
            if first_key.startswith(p):
                prefix = p.rstrip(".")
                break

    prefix_pattern = f"^(?:{re.escape(prefix)}\\.)?" if prefix else "^"
    module_pattern = re.compile(prefix_pattern + r"(.*)\.lora_[AB](?:\.[^.]+)?\.weight$")
    
    target_modules = set()
    for key in state_dict.keys():
        match = module_pattern.match(key)
        if match:
            target_modules.add(match.group(1))
    
    return sorted(target_modules)