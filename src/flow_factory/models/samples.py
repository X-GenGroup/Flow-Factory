# src/flow_factory/models/samples.py
import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal, Iterable
from dataclasses import dataclass, field, asdict, fields
import hashlib
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from diffusers.utils.outputs import BaseOutput

from ..utils.base import hash_pil_image, hash_tensor, hash_pil_image_list, is_tensor_list
from ..utils.logger_utils import setup_logger


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
class BaseSample(BaseOutput):
    """
    Base output class for Adapter models.
    The tensors are without batch dimension.
    """
    # Denoiseing trajectory
    all_latents : torch.FloatTensor
    timesteps : torch.FloatTensor
    log_probs : Optional[torch.FloatTensor] = None
    # Output dimensions
    height : Optional[int] = None
    width : Optional[int] = None
    # Generated media
    image: Optional[Image.Image] = None
    video: Optional[List[Image.Image]] = None
    # Prompt information
    prompt : Optional[str] = None
    prompt_ids : Optional[torch.LongTensor] = None
    prompt_embeds : Optional[torch.FloatTensor] = None
    # Negative prompt information
    negative_prompt : Optional[str] = None
    negative_prompt_ids : Optional[torch.LongTensor] = None
    negative_prompt_embeds : Optional[torch.FloatTensor] = None
    extra_kwargs : Dict[str, Any] = field(default_factory=dict)

    _unique_id: Optional[int] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for memory tracking, excluding non-tensor fields."""
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        extra = result.pop('extra_kwargs', {})
        result.update(extra)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseSample":
        """Create instance from dict, putting unknown fields into extra_kwargs."""
        field_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in d.items() if k in field_names and k != 'extra_kwargs'}
        extra = {k: v for k, v in d.items() if k not in field_names}
        assert not (set(extra) & field_names), f"Key collision: {set(extra) & field_names} when creating BaseSample from dict."
        extra.update(d.get('extra_kwargs', {}))
        return cls(**known, extra_kwargs=extra)
    
    def __getattr__(self, key: str) -> Any:
        """Access attributes. Check extra_kwargs if not found."""
        extra = object.__getattribute__(self, 'extra_kwargs')
        if key in extra:
            return extra[key]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def short_rep(self) -> Dict[str, Any]:
        """Short representation for logging."""
        def long_tensor_to_shape(t : torch.Tensor):
            if isinstance(t, torch.Tensor) and t.numel() > 16:
                return t.shape
            else:
                return t

        d = self.to_dict()
        d = {k: long_tensor_to_shape(v) for k,v in d.items()}
        return d

    def to(self, device: Union[torch.device, str], depth : int = 1) -> "BaseSample":
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


@dataclass
class ImageConditionSample(BaseSample):
    condition_images : Optional[Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]] = None

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
            if isinstance(self.condition_images, Image.Image):
                hasher.update(hash_pil_image(self.condition_images, size=32).encode())
            elif isinstance(self.condition_images, list) and self.condition_images:
                # List[Image.Image]
                if isinstance(self.condition_images[0], Image.Image):
                    hasher.update(hash_pil_image_list(self.condition_images, size=32).encode())
            elif isinstance(self.condition_images, (torch.Tensor, np.ndarray)):
                tensor = self.condition_images
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                hasher.update(hash_tensor(tensor).encode())
        
        return int.from_bytes(hasher.digest()[:8], byteorder='big', signed=True)

@dataclass
class VideoConditionSample(BaseSample):
    """Sample for video editing tasks."""
    condition_videos: Optional[Union[List[Image.Image], List[List[Image.Image]], torch.Tensor, np.ndarray]] = None

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
            if isinstance(self.condition_videos, (torch.Tensor, np.ndarray)):
                tensor = self.condition_videos
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                hasher.update(hash_tensor(tensor).encode())
            elif isinstance(self.condition_videos, list) and self.condition_videos:
                # List[Image.Image] (single video) or List[List[Image.Image]] (multi video)
                frames = self.condition_videos
                # Flatten if nested list
                if isinstance(frames[0], list):
                    frames = [f for video in frames for f in video]
                
                # Sample 4 evenly spaced frames
                if frames and isinstance(frames[0], Image.Image):
                    n = len(frames)
                    if n >= 4:
                        indices = [i * (n - 1) // 3 for i in range(4)]
                    else:
                        indices = list(range(n))
                    sampled = [frames[i] for i in indices]
                    hasher.update(hash_pil_image_list(sampled, size=32).encode())
        
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