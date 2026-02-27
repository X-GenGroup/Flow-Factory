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

# src/flow_factory/utils/dist.py
from __future__ import annotations
from typing import List, Optional, Union
from contextlib import nullcontext

import torch
from torch import nn
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils.operations import gather_object

from ..samples import BaseSample
from .base import (
    is_tensor_list
)
from .logger_utils import setup_logger

logger = setup_logger(__name__)

# -----------------------------------Tensor Gathering Utils---------------------------------------
def all_gather_tensor_list(
    accelerator: Accelerator,
    tensor_list: List[torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Gather a list of tensors from all processes, each process has a list of tensors.
    Each tensor can have a different shape (e.g., (C, H, W)).

    Args:
        accelerator (`Accelerator`): Accelerator object
        tensor_list (`List[torch.Tensor]`): list of tensors to gather, each tensor can have different shape but same dimension,  for example, [(3, 64, 64), (3, 128, 128), ...]. Each list can have different length on different processes.
        dtype (`torch.dtype`, *optional*): dtype of the gathered tensors, if None, use the dtype of the first tensor in tensor_list
        device (`Union[str, torch.device]`, *optional*, defaults to `torch.device("cpu")`): device of the gathered tensors

    Returns:
        gathered_tensors (`List[torch.Tensor]`): tensors from all processes, concatenated in rank order
    
    NOTE:
        This function requires 3 times of communication:
        1. Gather **lengths of `tensor_list`** from all ranks
        2. Gather **shapes of each tensor** in `tensor_list` from all ranks
        3. Gather **all tensors** by flattening and concatenating
    """
    if not tensor_list:
        return []
    
    assert all(isinstance(t, torch.Tensor) for t in tensor_list), "All elements in tensor_list must be torch.Tensor"
    assert all(t.dim() == tensor_list[0].dim() for t in tensor_list), "All tensors must have the same number of dimensions"

    tensor_dim = tensor_list[0].dim()
    tensor_dtype = tensor_list[0].dtype if dtype is None else dtype
    device = torch.device(device)

    # Step 1: Gather lengths of tensor_list from all ranks
    local_length = torch.tensor([len(tensor_list)], device=accelerator.device, dtype=torch.long)
    gathered_lengths = [torch.zeros(1, dtype=torch.long, device=accelerator.device) for _ in range(accelerator.num_processes)]
    dist.all_gather(gathered_lengths, local_length)
    gathered_lengths = [int(length.item()) for length in gathered_lengths]

    # Step 2: Gather shapes of each tensor in tensor_list from all ranks
    local_shapes = torch.tensor([list(t.shape) for t in tensor_list], device=accelerator.device, dtype=torch.long)
    gathered_shapes = [
        torch.zeros((length, tensor_dim), dtype=torch.long, device=accelerator.device)
        for length in gathered_lengths
    ]
    dist.all_gather(gathered_shapes, local_shapes)
    gathered_shapes = [shapes.cpu() for shapes in gathered_shapes]  # Move to CPU to save some GPU memory

    # Compute the total length of flattened tensors for each rank, [rank0_total_length, rank1_total_length, ...]
    flat_lengths = [
        sum(int(shape.prod().item()) for shape in this_rank_shapes)
        for this_rank_shapes in gathered_shapes
    ]

    # Step 3: Gather all tensors by flattening and concatenating
    local_flat_tensor = torch.cat([t.flatten() for t in tensor_list], dim=0).to(device=accelerator.device, dtype=tensor_dtype)
    gathered_flat_tensors = [
        torch.zeros(length, dtype=tensor_dtype, device=accelerator.device)
        for length in flat_lengths
    ]
    dist.all_gather(gathered_flat_tensors, local_flat_tensor)
    gathered_flat_tensors = [t.cpu() for t in gathered_flat_tensors]  # Move to CPU to save some GPU memory

    # Step 4: Reconstruct the original tensors from gathered shapes and flattened tensors
    gathered_tensors = []
    for rank, (this_rank_shapes, this_rank_flat_tensor) in enumerate(zip(gathered_shapes, gathered_flat_tensors)):
        offset = 0
        for shape in this_rank_shapes:
            length = int(shape.prod().item())
            # Reshape and move to the specified device
            this_tensor = this_rank_flat_tensor[offset:offset+length].reshape(shape.tolist()).to(device)
            gathered_tensors.append(this_tensor)
            offset += length

    return gathered_tensors

def all_gather_nested_tensor_list(
    accelerator: Accelerator,
    nested_tensor_list: List[List[torch.Tensor]],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> List[List[torch.Tensor]]:
    """
    Gather a list of list of tensors from all processes, each process has a list of list of tensors.
    Each tensor can have a different shape (e.g., (C, H, W)).

    Args:
        accelerator (`Accelerator`): Accelerator object
        nested_tensor_list (`List[List[torch.Tensor]]`): list of list of tensors to gather, each tensor can have different shape but same dimension,  for example, [[(3, 64, 64), (3, 128, 128)], [(3, 32, 32)]]. Each inner list can have different length on different processes.
        dtype (`torch.dtype`, *optional*): dtype of the gathered tensors, if None, use the dtype of the first tensor in nested_tensor_list
        device (`Union[str, torch.device]`, *optional*, defaults to `torch.device("cpu")`): device of the gathered tensors
    Returns:
        gathered_nested_tensors (`List[List[torch.Tensor]]`): list of list of tensors from all processes, concatenated in rank order
    
    NOTE:
        This function requires 5 times of communication:
        1. Flatten the local nested structure into a single list of tensors for `all_gather_tensor_list` (3 times inside that function)
        2. Gather the structure information - **length of the list** and **the lengths of inner lists** (2 times)
        3. Reconstruct the nested structure using the gathered structure info
    """
    # 1. Flatten the local nested structure into a single list of tensors
    # [[t1, t2], [t3]] -> [t1, t2, t3]
    flat_tensor_list = [t for sublist in nested_tensor_list for t in sublist]

    # 2. Gather the flattened tensors
    # This gives us all the raw data: [t1_rank0, t2_rank0, t3_rank0, t4_rank1...]
    gathered_flat_tensors = all_gather_tensor_list(
        accelerator, 
        flat_tensor_list, 
        dtype=dtype, 
        device=device
    )

    # 3. Gather the structure information (lengths of inner lists) from all ranks
    local_structure = torch.tensor(
        [len(sublist) for sublist in nested_tensor_list], 
        dtype=torch.long,
        device=accelerator.device
    )

    # Gather the NUMBER of inner lists per rank (Scalar gather)
    # Rank 0 has 2 lists, Rank 1 has 5 lists -> We need to know this to allocate memory
    local_list_count = torch.tensor([local_structure.numel()], device=accelerator.device, dtype=torch.long) # rank 0: 2, rank 1: 3
    gathered_list_counts = [torch.zeros_like(local_list_count) for _ in range(dist.get_world_size())] # [_, _]
    dist.all_gather(gathered_list_counts, local_list_count) # gathered_list_counts = [2, 3]

    # Gather the actual structure tensors using the counts above
    gathered_structures = [
        torch.zeros(count.item(), dtype=torch.long, device=accelerator.device) 
        for count in gathered_list_counts
    ] # [tensor([0, 0]), tensor([0, 0, 0])]
    dist.all_gather(gathered_structures, local_structure) # gathered_structures = [tensor([2, 1]), tensor([1, 2, 1])]

    # 4. Reconstruct the nested structure using the gathered structure info
    gathered_nested_tensors = []
    flat_tensor_idx = 0

    # Iterate over every rank's structure
    for rank_structure in gathered_structures:
        # rank_structure is a tensor like [2, 1] meaning:
        # this rank had 2 inner lists, first with length 2, second with length 1
        for inner_list_len in rank_structure.tolist():
            length = int(inner_list_len)
            
            # Slice the big flat list to rebuild the inner list
            inner_list = gathered_flat_tensors[flat_tensor_idx : flat_tensor_idx + length]
            gathered_nested_tensors.append(inner_list)
            
            flat_tensor_idx += length

    assert flat_tensor_idx == len(gathered_flat_tensors), "Mismatch in reconstructed tensor count when rebuilding nested structure."

    return gathered_nested_tensors

# -----------------------------------Sample Utils---------------------------------------
def gather_samples(
        accelerator: Accelerator,
        samples: List[BaseSample],
        field_names: List[str],
        device: Union[str, torch.device]=torch.device("cpu")
    ) -> List[BaseSample]:
    """
    Gather a list of BaseSample from all processes.

    Args:
        accelerator (`Accelerator`): Accelerator object
        samples (`List[BaseSample]`): list of BaseSample to gather
        field_names (`List[str]`): list of field names to gather and concatenate
        device (`Union[str, torch.device]`, *optional*, defaults to `torch.device("cpu")`): device of the gathered samples
    Returns:
        gathered_samples (`List[BaseSample]`): samples from all processes, concatenated in rank order
    """
    if not samples:
        return []
    
    sample_cls = samples[0].__class__ # Assume all samples are of the same class
    device = torch.device(device)
    field_names = sorted(field_names) # Sort to make sure async
    d = {field_name: [] for field_name in field_names}

    for field_name in field_names:
        # Collect field values from all samples
        field_values = [getattr(sample, field_name) for sample in samples]
        if is_tensor_list(field_values):
            # Gather list of tensors, for fields like images, videos, etc.
            gathered_field_values = all_gather_tensor_list(
                accelerator=accelerator,
                tensor_list=field_values,
                device=device
            )
            d[field_name].extend(gathered_field_values)
        elif is_tensor_list(field_values[0]):
            # List[List[torch.Tensor]], for fields like condition_images, condition_videos, etc.
            gathered_field_values = all_gather_nested_tensor_list(
                accelerator=accelerator,
                nested_tensor_list=field_values,
                device=device
            )
        else:
            # Gather other objects using accelerate's gather_object
            gathered_field_values = gather_object(field_values)
            d[field_name].extend(gathered_field_values)

    # Reconstruct BaseSample objects
    gathered_samples = [
        sample_cls(**dict(zip(field_names, values)))
        for values in zip(*(d[field_name] for field_name in field_names))
    ]
    return gathered_samples