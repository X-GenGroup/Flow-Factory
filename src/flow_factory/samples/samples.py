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

# src/flow_factory/models/samples.py
from __future__ import annotations
import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal, Iterable, ClassVar
from dataclasses import dataclass, field, asdict, fields
import hashlib
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from ..utils.base import (
    standardize_image_batch,
    standardize_video_batch,
)

from diffusers.utils.import_utils import is_torch_available, is_torch_version

from ..utils.base import (
    ImageSingle,
    ImageBatch,
    VideoSingle,
    VideoBatch,
    hash_pil_image,
    hash_tensor,
    hash_pil_image_list,
    hash_tensor_list,
    is_tensor_list,
    standardize_image_batch,
    standardize_video_batch,
)
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


__all__ = [
    'BaseSample',
    'ImageConditionSample',
    'VideoConditionSample',
    'T2ISample',
    'T2VSample',
    'I2ISample',
    'I2VSample',
    'V2VSample',
]

@dataclass
class BaseSample:
    """
    Base output class for Adapter models.
    The tensors are without batch dimension.
    """
    _id_fields : ClassVar[frozenset[str]] = frozenset({'prompt', 'prompt_ids'})  # Fields used for unique_id computation

    # Fields that are shared across the batch
    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        'height', 'width', 'latent_index_map', 'log_prob_index_map'
    })

    # Denoiseing trajectory
    timesteps : Optional[torch.Tensor] = None # (T+1,)
    all_latents : Optional[torch.Tensor] = None # (num_steps, Seq_len, C)
    latent_index_map: Optional[torch.Tensor] = None   # (T+1,) LongTensor
    log_probs : Optional[torch.Tensor] = None # (num_steps,)
    log_prob_index_map: Optional[torch.Tensor] = None  # (T+1,) LongTensor
    # Output dimensions
    height : Optional[int] = None
    width : Optional[int] = None
    # Generated media
    image: Optional[ImageSingle] = None # PIL.Image | torch.Tensor | np.ndarray. This field will be convert to a tensor of shape (C, H, W) for canonicalization.
    video: Optional[VideoSingle] = None # List[Image.Image] | torch.Tensor | np.ndarray. This field will be convert to a tensor of shape (T, C, H, W) for canonicalization.
    # Prompt information
    prompt : Optional[str] = None
    prompt_ids : Optional[torch.Tensor] = None
    prompt_embeds : Optional[torch.Tensor] = None
    # Negative prompt information
    negative_prompt : Optional[str] = None
    negative_prompt_ids : Optional[torch.Tensor] = None
    negative_prompt_embeds : Optional[torch.Tensor] = None
    extra_kwargs : Dict[str, Any] = field(default_factory=dict)

    _unique_id: Optional[int] = field(default=None, repr=False, compare=False)

    def __init_subclass__(cls) -> None:
        """
        **Copied from diffusers.utils.outputs.BaseOutput.__init_subclass__**
        Register subclasses as PyTorch pytree nodes for DDP/FSDP compatibility.
        """
        super().__init_subclass__()
        if is_torch_available():
            import torch.utils._pytree as pytree
            
            def flatten(obj):
                """Flatten dataclass to (values, context)."""
                values = []
                keys = []
                for f in fields(obj):
                    keys.append(f.name)
                    values.append(getattr(obj, f.name))
                return values, keys
            
            def unflatten(values, keys):
                """Reconstruct dataclass from (values, context)."""
                return cls(**dict(zip(keys, values)))
            
            if is_torch_available() and is_torch_version("<", "2.2"):
                pytree._register_pytree_node(cls, flatten, unflatten)
            else:
                pytree.register_pytree_node(
                    cls, 
                    flatten, 
                    unflatten,
                    serialized_type_name=f"{cls.__module__}.{cls.__name__}"
                )

    def __post_init__(self):
        """Post-initialization processing."""
        # Standardize image field to tensor (C, H, W)
        if self.image is not None:
            # -> (1, C, H, W) -> (C, H, W)
            self.image = standardize_image_batch(self.image, 'pt')[0]
        
        # Standardize video field to tensor (T, C, H, W)
        if self.video is not None:
            # -> (1, T, C, H, W) -> (T, C, H, W)
            self.video = standardize_video_batch(self.video, 'pt')[0]
    
    @classmethod
    def shared_fields(cls) -> frozenset[str]:
        """Merge all _shared_fields from inheritance chain."""
        fields = set()
        for base in cls.__mro__[:-1]:  # Exclude object
            if hasattr(base, '_shared_fields'):
                fields.update(base._shared_fields)
        return frozenset(fields)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for memory tracking, excluding non-tensor fields."""
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        extra = result.pop('extra_kwargs', {})
        result.update(extra)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BaseSample:
        """Create instance from dict, putting unknown fields into extra_kwargs."""
        field_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in d.items() if k in field_names and k != 'extra_kwargs'}
        
        # Collect unknown fields
        extra = {k: v for k, v in d.items() if k not in field_names}
        
        # Merge with incoming extra_kwargs and check for conflicts
        incoming_extra = d.get('extra_kwargs', {})
        conflicting_keys = set(incoming_extra) & (field_names - {'extra_kwargs'})
        if conflicting_keys:
            raise ValueError(
                f"extra_kwargs contains reserved field names: {conflicting_keys}"
            )
        extra.update(incoming_extra)
        
        return cls(**known, extra_kwargs=extra)
    
    def __getattr__(self, key: str) -> Any:
        """Access attributes. Check extra_kwargs if not found."""
        try:
            extra = object.__getattribute__(self, 'extra_kwargs')
        except AttributeError:
            # extra_kwargs not yet initialized (during __init__)
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")
        
        if key in extra:
            return extra[key]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """Set attributes."""
        if key in type(self)._id_fields:
            object.__setattr__(self, '_unique_id', None) # Reset unique_id cache

        super().__setattr__(key, value)

    def keys(self):
        return self.to_dict().keys() # Keep consistent

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: sample['prompt']."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __iter__(self):
        """Allow iteration over keys (required for some mapping operations)."""
        return iter(self.keys())

    def short_rep(self) -> Dict[str, Any]:
        """Short representation for logging (replaces large tensors with shapes)."""
        def tensor_to_repr(v):
            if isinstance(v, torch.Tensor) and v.numel() > 16:
                return f"Tensor{tuple(v.shape)}"
            return v

        return {k: tensor_to_repr(v) for k, v in self.to_dict().items()}

    def to(self, device: Union[torch.device, str], depth : int = 1) -> BaseSample:
        """Move all tensor fields to specified device."""
        assert 0 <= depth <= 1, "Only depth 0 and 1 are supported."
        device = torch.device(device)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))
            elif depth == 1 and is_tensor_list(value):
                setattr(
                    self,
                    field.name,
                    [t.to(device) if isinstance(t, torch.Tensor) else t for t in value]
                )
            
        return self

    def compute_unique_id(self) -> int:
        """
        Compute a unique identifier for distributed grouping.
        Base implementation handles prompt.
        Subclasses can override to customize hash behavior.
        
        Returns:
            int: A 64-bit signed integer hash for tensor compatibility.
        """
        hasher = hashlib.sha256()
        
        # Hash prompt
        if self.prompt_ids is not None:
            hasher.update(self.prompt_ids.cpu().numpy().tobytes())
        elif self.prompt is not None:
            hasher.update(self.prompt.encode('utf-8'))

        # Convert to 64-bit signed integer
        return int.from_bytes(hasher.digest()[:8], byteorder='big', signed=True)

    @property
    def unique_id(self) -> int:
        """Get or compute the unique identifier."""
        if self._unique_id is None:
            self._unique_id = self.compute_unique_id()
        return self._unique_id
    
    def reset_unique_id(self):
        """Reset cached unique_id (call after modifying relevant fields)."""
        self._unique_id = None

    @classmethod
    def _stack_values(cls, key: str, values: List[Any]) -> Union[torch.Tensor, Dict, List, Any]:
        """
        Recursively stack values based on field configuration.
        
        Processing order:
            1. Shared fields → return first element only
            2. Stackable fields → attempt stacking (tensors/dicts)
            3. Other fields → return as list
        
        Args:
            key: Field name to determine stacking behavior
            values: List of values to stack
        
        Returns:
            - Any: If shared, returns first element
            - torch.Tensor: If stackable and all values are matching tensors
            - Dict: If stackable and all values are dicts (recursively stacked)
            - List: Otherwise
        """
        if not values:
            return values
        
        # All are None - return None
        if all(v is None for v in values):
            return None

        first = values[0]
        
        # 1. Shared fields - take first element only
        if key in cls.shared_fields():
            return first

        # 2. Tensor fields, try to stack
        # Stack tensors with matching shapes
        if isinstance(first, torch.Tensor):
            # Assume all tensors
            if all(v.shape == first.shape for v in values):
                return torch.stack(values)
            return values

        # 3. Recursively stack dictionaries
        if isinstance(first, dict):
            if all(isinstance(v, dict) for v in values):
                return {
                    k: cls._stack_values(k, [v[k] for v in values])
                    for k in first.keys()
                }
            return values
        
        # 3. Default - return as list
        return values

    @classmethod
    def stack(cls, samples: List[BaseSample]) -> Dict[str, Union[torch.Tensor, Dict, List, Any]]:
        """
        Stack BaseSample instances into batched structures.
        
        Field behavior controlled by class methods:
            - shared_fields(): Take first element only (shared across batch)
            - stackable_fields(): Stack tensors/dicts with matching structure
            - Other: Collect into lists
        
        Args:
            samples: List of BaseSample instances
        
        Returns:
            Dictionary with processed values per field
        
        Raises:
            ValueError: If samples list is empty
        """
        if not samples:
            raise ValueError("No samples to stack.")
        
        sample_cls = type(samples[0]) # Dynamically use the sample's class
        sample_dicts = [s.to_dict() for s in samples]
        
        return {
            key: sample_cls._stack_values(key, [d[key] for d in sample_dicts])
            for key in sample_dicts[0].keys()
        }


@dataclass
class ImageConditionSample(BaseSample):
    """Sample for tasks with image conditions."""
    _id_fields : ClassVar[frozenset[str]] = BaseSample._id_fields | frozenset({'condition_images'})

    condition_images : Optional[ImageBatch] = None # A list of (Image.Image | torch.Tensor | np.ndarray) or a batched tensor/array
    # `condition_images` will be convert to a list of tensors of shape (C, H, W) for canonicalization.

    def __post_init__(self):
        super().__post_init__()
        if self.condition_images is not None:
            # Standardize condition_images to List[torch.Tensor] of shape (C, H, W),
            # if the input `condition_images` is already a batched tensor/array, it will stay as a batched tensor.
            self.condition_images = standardize_image_batch(self.condition_images, 'pt')

    def compute_unique_id(self) -> int:
        """Hash prompt + condition_images."""
        hasher = hashlib.sha256()
        
        # 1. Hash prompt
        if self.prompt_ids is not None:
            hasher.update(self.prompt_ids.cpu().numpy().tobytes())
        elif self.prompt is not None:
            hasher.update(self.prompt.encode('utf-8'))
        
        # 2. Hash condition_images
        if self.condition_images is not None:
            cond_images = standardize_image_batch(
                self.condition_images,
                output_type='pil'
            )
            hasher.update(hash_pil_image_list(cond_images).encode())
        
        return int.from_bytes(hasher.digest()[:8], byteorder='big', signed=True)

@dataclass
class VideoConditionSample(BaseSample):
    """Sample for tasks with video conditions."""
    _id_fields : ClassVar[frozenset[str]] = BaseSample._id_fields | frozenset({'condition_videos'})

    condition_videos: Optional[VideoBatch] = None # A list of (List[Image.Image] | torch.Tensor | np.ndarray) or a batched tensor/array
    # `condition_videos` will be convert to a list of tensors of shape (T, C, H, W) for canonicalization.

    def __post_init__(self):
        super().__post_init__()
        if self.condition_videos is not None:
            # Standardize condition_videos to List[torch.Tensor] of shape (T, C, H, W),
            # if the input `condition_videos` is already a batched tensor/array, it will stay as a batched tensor.
            self.condition_videos = standardize_video_batch(self.condition_videos, 'pt')

    def compute_unique_id(self) -> int:
        """Hash prompt + condition_videos (sampling 4 evenly spaced frames)."""
        hasher = hashlib.sha256()
        
        # 1. Hash prompt
        if self.prompt_ids is not None:
            hasher.update(self.prompt_ids.cpu().numpy().tobytes())
        elif self.prompt is not None:
            hasher.update(self.prompt.encode('utf-8'))
        
        # 2. Hash condition_videos
        if self.condition_videos is not None:
            cond_videos = standardize_video_batch(
                self.condition_videos,
                output_type='pil'
            )
            for v in cond_videos:
                hasher.update(hash_pil_image_list(v).encode())
        
        return int.from_bytes(hasher.digest()[:8], byteorder='big', signed=True)

@dataclass
class T2ISample(BaseSample):
    """Text-to-Image sample output."""
    pass

@dataclass
class T2VSample(BaseSample):
    """Text-to-Video sample output."""
    pass

@dataclass
class I2ISample(ImageConditionSample):
    """Image-to-Image sample output."""
    pass

@dataclass
class I2VSample(ImageConditionSample):
    """Image-to-Video sample output."""
    pass

@dataclass
class V2VSample(VideoConditionSample):
    """Video-to-Video sample output."""
    pass