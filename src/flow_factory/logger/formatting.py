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

# src/flow_factory/logger/formatting.py
from __future__ import annotations

import os
import tempfile
import math

import torch
import numpy as np
from PIL import Image
import imageio
from typing import Any, Dict, List, Union, Optional, Tuple
from dataclasses import dataclass, is_dataclass, asdict, field
from ..models.samples import BaseSample, T2ISample, T2VSample, I2ISample, I2VSample, V2VSample
from ..utils.base import (
    numpy_to_pil_image,
    tensor_to_pil_image,
    tensor_list_to_pil_image,
    numpy_list_to_pil_image,
    is_video_frame_list,
    video_frames_to_numpy,
    video_frames_to_tensor,
    tensor_to_video_frames,
    numpy_to_video_frames,
)
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------- Helper Functions -------------------------------------------
def _compute_optimal_grid(n: int) -> Tuple[int, int]:
    """Compute optimal grid (rows, cols) for n images, preferring wider layouts."""
    if n <= 0:
        return (0, 0)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return (rows, cols)

def _concat_images_grid(images: List[Image.Image]) -> Image.Image:
    """Concatenate images into optimal grid layout."""
    if not images:
        raise ValueError("Empty image list")
    if len(images) == 1:
        return images[0]
    
    rows, cols = _compute_optimal_grid(len(images))
    
    # Resize all to match first image
    w, h = images[0].size
    resized = [img.resize((w, h), Image.Resampling.LANCZOS) if img.size != (w, h) else img for img in images]
    
    grid = Image.new('RGB', (cols * w, rows * h))
    for idx, img in enumerate(resized):
        grid.paste(img.convert('RGB'), ((idx % cols) * w, (idx // cols) * h))
    return grid

def _to_pil_list(images: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray, None]) -> List[Image.Image]:
    """Convert various image types to List[PIL.Image]."""
    if images is None:
        return []
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, torch.Tensor):
        return tensor_to_pil_image(images)
    if isinstance(images, np.ndarray):
        return numpy_to_pil_image(images)

    if isinstance(images, list):
        if isinstance(images[0], Image.Image):
            return images
        elif isinstance(images[0], torch.Tensor):
            return tensor_list_to_pil_image(images)
        elif isinstance(images[0], np.ndarray):
            return numpy_list_to_pil_image(images)

    return []

def _to_video_list(
    videos: Union[str, np.ndarray, torch.Tensor, List[Image.Image], List[Any], None]
) -> List[Union[str, np.ndarray, torch.Tensor, List[Image.Image]]]:
    """
    Convert various video types to List of videos.
    
    Handles the ambiguity where List[PIL.Image] represents a single video (frames),
    not multiple videos.
    
    Args:
        videos: Single video or list of videos. Supported formats:
            - str: Video file path (single video)
            - np.ndarray: Shape (T,H,W,C) or (B,T,H,W,C)
            - torch.Tensor: Shape (T,C,H,W) or (B,T,C,H,W)
            - List[PIL.Image]: Single video as frame list
            - List[Video]: Multiple videos
    
    Returns:
        List of videos, each in its original format.
    
    Example:
        >>> frames = [Image.new('RGB', (64, 64)) for _ in range(16)]
        >>> _to_video_list(frames)  # Single video -> wrapped in list
        [[PIL.Image, ...]]
        >>> _to_video_list([frames, frames])  # Multiple videos -> as-is
        [[PIL.Image, ...], [PIL.Image, ...]]
    """
    
    if videos is None:
        return []
    
    # Single video: str path
    if isinstance(videos, str):
        return [videos]
    
    # Single video: tensor (T,C,H,W) or batch (B,T,C,H,W)
    if isinstance(videos, torch.Tensor):
        if videos.ndim == 4:  # (T,C,H,W) single video
            return [videos]
        elif videos.ndim == 5:  # (B,T,C,H,W) batch -> split
            return list(videos.unbind(0))
    
    # Single video: numpy (T,H,W,C) or batch (B,T,H,W,C)
    if isinstance(videos, np.ndarray):
        if videos.ndim == 4:  # (T,H,W,C) single video
            return [videos]
        elif videos.ndim == 5:  # (B,T,H,W,C) batch -> split
            return [videos[i] for i in range(videos.shape[0])]
    
    # List types
    if isinstance(videos, list) and len(videos) > 0:
        # List[PIL.Image] = single video as frames
        if is_video_frame_list(videos):
            return [videos]
        # List[List[PIL.Image]] or List[tensor/ndarray/str] = multiple videos
        return videos
    
    return []

def _build_sample_caption(sample : BaseSample, max_length: Optional[int] = None) -> str:
    """Build caption from reward and prompt."""
    parts = []
    if 'rewards' in sample.extra_kwargs:
        rewards = sample.extra_kwargs['rewards']
        if isinstance(rewards, float):
            parts.append(f"{rewards:.2f}")
        elif isinstance(rewards, (list, tuple)) and rewards:
            if len(rewards) == 1:
                parts.append(f"{rewards[0]:.2f}")
            else:
                parts.append(", ".join(f"{r:.2f}" for r in rewards))
        elif isinstance(rewards, dict):
            if len(rewards) == 1:
                parts.append(f"{next(iter(rewards.values())):.2f}")
            else:
                parts.append(", ".join(f"{k}: {v:.2f}" for k, v in rewards.items()))
    if sample.prompt:
        parts.append(sample.prompt[:max_length] + "..." if (max_length is not None and len(sample.prompt) > max_length) else sample.prompt)
    return " | ".join(parts)

def _compute_resize_dims(
    orig_h: int, 
    orig_w: int, 
    target_h: Optional[int] = None, 
    target_w: Optional[int] = None
) -> Tuple[int, int]:
    """
    Compute resize dimensions while preserving aspect ratio.
    
    Args:
        orig_h: Original height.
        orig_w: Original width.
        target_h: Target height. If only this is specified, width is computed to preserve aspect ratio.
        target_w: Target width. If only this is specified, height is computed to preserve aspect ratio.
    
    Returns:
        Tuple of (new_height, new_width).
    
    Examples:
        >>> _compute_resize_dims(1080, 1920, target_h=540)  # -> (540, 960)
        >>> _compute_resize_dims(1080, 1920, target_w=960)  # -> (540, 960)
        >>> _compute_resize_dims(1080, 1920, 960, 540)      # -> (540, 960), exact resize
    """
    if target_h is None and target_w is None:
        return orig_h, orig_w
    if target_h and target_w:
        return target_h, target_w
    aspect = orig_w / orig_h
    if target_h:
        return target_h, int(target_h * aspect)
    return int(target_w / aspect), target_w


# ------------------------------------------- LogImage & LogVideo Classes -------------------------------------------

@dataclass
class LogImage:
    """
    Intermediate representation for an image with compression and resize support.
    
    Supports lazy loading, automatic compression to JPEG, and aspect-ratio-preserving resize.
    Temporary files are cached by (height, width) and cleaned up on exit or manual cleanup().
    
    Args:
        _value: Source image - can be file path, PIL Image, numpy array, or torch Tensor.
        caption: Optional caption for logging platforms.
        compress: Whether to compress output to JPEG (default: True).
        quality: JPEG quality when compress=True (default: 85).
    
    Example:
        >>> img = LogImage(tensor, caption="generated")
        >>> path = img.get_value(height=512)  # aspect-ratio preserved resize
        >>> img.cleanup()  # remove temp files
    """
    _value: Union[str, Image.Image, np.ndarray, torch.Tensor] = field(repr=False)
    _img: Optional[Image.Image] = field(default=None, init=False, repr=False)
    caption: Optional[str] = None
    compress: bool = True
    quality: int = 85
    _temp_paths: Dict[Tuple, str] = field(default_factory=dict, init=False, repr=False)
    
    @classmethod
    def to_pil(cls, value: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert various input types to PIL Image."""
        if isinstance(value, Image.Image):
            return value
        elif isinstance(value, torch.Tensor):
            return tensor_to_pil_image(value)[0]
        elif isinstance(value, np.ndarray):
            return numpy_to_pil_image(value)[0]
        elif isinstance(value, str) and os.path.exists(value):
            return Image.open(value).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(value)}")

    def get_pil(self) -> Image.Image:
        """Get PIL Image (lazily loaded and cached)."""
        if self._img is None:
            self._img = LogImage.to_pil(self._value)
        return self._img

    def get_size(self) -> Tuple[int, int]:
        """Get image dimensions as (height, width)."""
        return self.get_pil().size[1], self.get_pil().size[0]

    def get_value(
        self, 
        height: Optional[int] = None, 
        width: Optional[int] = None
    ) -> Union[str, Image.Image]:
        """
        Get image as compressed file path or PIL Image, optionally resized.
        
        Args:
            height: Target height. If only height is specified, width is computed to preserve aspect ratio.
            width: Target width. If only width is specified, height is computed to preserve aspect ratio.
        
        Returns:
            File path (str) if compress=True, otherwise PIL Image.
        
        Note:
            Results are cached by (height, width) tuple. Call cleanup() to remove temp files.
        """
        cache_key = (height, width)
        if cache_key in self._temp_paths:
            return self._temp_paths[cache_key]
        
        # If already a path with no resize needed, return as-is
        if isinstance(self._value, str) and height is None and width is None:
            return self._value
        
        img = self.get_pil()
        
        # Resize if dimensions specified
        if height is not None or width is not None:
            new_h, new_w  = _compute_resize_dims(img.size[1], img.size[0], height, width)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Save to temp file if compression enabled
        if self.compress:
            fd, path = tempfile.mkstemp(suffix='.jpg')
            try:
                with os.fdopen(fd, 'wb') as f:
                    img.convert('RGB').save(f, format='JPEG', quality=self.quality)
                self._temp_paths[cache_key] = path
            except Exception as e:
                if os.path.exists(path):
                    os.unlink(path)
                raise e
            return path

        return img

    @property
    def value(self) -> Union[str, Image.Image]:
        """Get image at original size (shorthand for get_value())."""
        return self.get_value()
    
    @value.setter
    def value(self, val: Union[str, Image.Image, np.ndarray, torch.Tensor]):
        """Set new source value and reset all cached state."""
        self.cleanup()
        self._value = val
        self._img = None
    
    def cleanup(self):
        """Remove all temporary files created by get_value()."""
        for path in self._temp_paths.values():
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        self._temp_paths.clear()

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()


@dataclass
class LogVideo:
    """
    Intermediate representation for a video with format conversion and resize support.
    
    Supports lazy loading, format conversion (mp4/gif), and aspect-ratio-preserving resize.
    Temporary files are cached by (format, height, width) and cleaned up on exit or manual cleanup().
    
    Args:
        _value: Source video - can be file path, numpy array (T,H,W,C), torch Tensor, or List[PIL.Image].
        caption: Optional caption for logging platforms.
        fps: Frames per second for output video (default: 8).
    
    Example:
        >>> vid = LogVideo(frames, caption="generated", fps=24)
        >>> path = vid.get_value('gif', height=256)  # convert to gif, resized
        >>> vid.cleanup()  # remove temp files
    """
    _value: Union[str, np.ndarray, torch.Tensor, List[Image.Image]] = field(repr=False)
    caption: Optional[str] = None
    fps: int = 8
    _temp_paths: Dict[Tuple, str] = field(default_factory=dict, init=False, repr=False)
    _arr: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def format(self) -> str:
        """Get source video format extension (without dot). Defaults to 'mp4' for non-file sources."""
        if isinstance(self._value, str):
            return os.path.splitext(self._value)[1].lstrip('.').lower() or 'mp4'
        return 'mp4'
    
    @classmethod
    def to_numpy(cls, value: Union[np.ndarray, torch.Tensor, List[Image.Image]]) -> np.ndarray:
        """
        Convert various input types to numpy array with shape (T, H, W, C), dtype uint8.
        
        Handles channel order conversion (TCHW -> THWC) and value normalization to [0, 255].
        """
        if isinstance(value, str):
            raise ValueError("Cannot convert path to numpy directly, use get_numpy() instead")
        
        if isinstance(value, list) and value and isinstance(value[0], Image.Image):
            frames = [np.array(img.convert('RGB')) for img in value]
            arr = np.stack(frames, axis=0)
        elif isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = value
        
        # Convert TCHW -> THWC if channels-first
        if arr.ndim == 4 and arr.shape[1] in (1, 3, 4) and arr.shape[1] < arr.shape[2]:
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 3:
            arr = arr[..., np.newaxis]
        
        # Normalize to uint8
        if arr.dtype != np.uint8:
            arr = ((arr * 255) if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
        return arr
    
    def get_numpy(self) -> np.ndarray:
        """Get video as numpy array (T, H, W, C), lazily loaded and cached."""
        if self._arr is None:
            if isinstance(self._value, str):
                frames = imageio.mimread(self._value)
                self._arr = np.stack(frames, axis=0)
            else:
                self._arr = self.to_numpy(self._value)
        return self._arr

    def get_size(self) -> Tuple[int, int]:
        """Get video frame dimensions as (height, width)."""
        arr = self.get_numpy()
        return arr.shape[1], arr.shape[2]

    def get_value(
        self, 
        format: str = 'mp4', 
        height: Optional[int] = None, 
        width: Optional[int] = None
    ) -> str:
        """
        Get video file path in specified format, optionally resized.
        
        Args:
            format: Output format, either 'mp4' or 'gif' (default: 'mp4').
            height: Target height. If only height is specified, width is computed to preserve aspect ratio.
            width: Target width. If only width is specified, height is computed to preserve aspect ratio.
        
        Returns:
            Path to the video file (temporary file if conversion/resize needed).
        
        Note:
            Results are cached by (format, height, width) tuple. Call cleanup() to remove temp files.
        """
        format = format.lower().lstrip('.')
        cache_key = (format, height, width)
        
        if cache_key in self._temp_paths:
            return self._temp_paths[cache_key]
        
        # If source is file with matching format and no resize, return directly
        if isinstance(self._value, str) and self.format == format and height is None and width is None:
            return self._value
        
        arr = self.get_numpy()
        
        # Resize frames if dimensions specified
        if height is not None or width is not None:
            orig_h, orig_w = arr.shape[1], arr.shape[2]
            new_h, new_w = _compute_resize_dims(orig_h, orig_w, height, width)
            resized = []
            for frame in arr:
                img = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.LANCZOS)
                resized.append(np.array(img))
            arr = np.stack(resized, axis=0)
        
        # Write to temp file
        fd, path = tempfile.mkstemp(suffix=f'.{format}')
        try:
            os.close(fd)
            if format == 'gif':
                imageio.mimwrite(path, arr, fps=self.fps, format='GIF', loop=0)
            else:
                imageio.mimwrite(path, arr, fps=self.fps, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
            self._temp_paths[cache_key] = path
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            raise
        return path

    @property
    def value(self) -> str:
        """Get video as mp4 at original size (shorthand for get_value())."""
        return self.get_value('mp4')

    @value.setter
    def value(self, val: Union[str, np.ndarray, torch.Tensor, List[Image.Image]]):
        """Set new source value and reset all cached state."""
        self.cleanup()
        self._value = val
        self._arr = None

    def cleanup(self):
        """Remove all temporary files created by get_value()."""
        for path in self._temp_paths.values():
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        self._temp_paths.clear()

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

@dataclass
class LogTable:
    """
    Table structure for conditional generation logging: [cond_1, ..., cond_n, generation].
    
    Used for I2V (image-to-video) and V2V (video-to-video) samples where conditions 
    and generations should be displayed side-by-side with unified dimensions.
    
    Args:
        columns: Column names for the table.
        rows: List of rows, each containing LogImage/LogVideo items.
        target_height: Unified height for all items (derived from first generation).
    
    Example:
        >>> table = LogTable.from_i2v_samples(samples)
        >>> for row in table.rows:
        ...     for item in row:
        ...         path = item.get_value(height=table.target_height)
    """
    columns: List[str] = field(default_factory=list)
    rows: List[List[Optional[Union[LogImage, LogVideo]]]] = field(default_factory=list)
    target_height: Optional[int] = None
    
    @classmethod
    def from_i2v_samples(cls, samples: List[I2VSample]) -> Optional['LogTable']:
        """
        Build table from I2V samples: [condition_images...] -> video.
        
        Sets `target_height` from the **first valid generation's height** for unified display.
        """
        if not samples or not hasattr(samples[0], 'condition_images'):
            return None
        
        first_conds = _to_pil_list(samples[0].condition_images)
        n_conds = len(first_conds)
        columns = [f"condition_image_{i}" for i in range(n_conds)] + ["generation"]
        
        rows = []
        target_height = None
        
        for s in samples:
            # Skip samples with no generation
            if s.video is None:
                continue
            conds = _to_pil_list(s.condition_images)[:n_conds]
            
            caption = _build_sample_caption(s)
            gen_video = LogVideo(s.video, caption=caption)
            
            # Use first generation's height as target for unified display
            if target_height is None:
                target_height, _ = gen_video.get_size()
            
            # Build row with None padding for missing conditions
            cond_items: List[Optional[LogImage]] = [LogImage(c) for c in conds]
            row = cond_items + [None] * (n_conds - len(conds)) + [gen_video]
            rows.append(row)
        
        return cls(columns=columns, rows=rows, target_height=target_height) if rows else None
    
    @classmethod
    def from_v2v_samples(cls, samples: List[V2VSample]) -> Optional['LogTable']:
        """
        Build table from V2V samples: [condition_videos...] -> video.
        
        Sets `target_height` from the **first valid generation's height** for unified display.
        """
        if not samples or not hasattr(samples[0], 'condition_videos'):
            return None
        
        first_conds = _to_video_list(samples[0].condition_videos)
        n_conds = len(first_conds)
        columns = [f"condition_video_{i}" for i in range(n_conds)] + ["generation"]
        
        rows = []
        target_height = None
        
        for s in samples:
            if s.video is None:
                continue
            conds = _to_video_list(s.condition_videos)[:n_conds]
            
            caption = _build_sample_caption(s)
            gen_video = LogVideo(s.video, caption=caption)
            
            if target_height is None:
                target_height, _ = gen_video.get_size()
            
            # Build row with None padding for missing conditions
            cond_items: List[Optional[LogVideo]] = [LogVideo(c) for c in conds]
            row = cond_items + [None] * (n_conds - len(conds)) + [gen_video]
            rows.append(row)
        
        return cls(columns=columns, rows=rows, target_height=target_height) if rows else None
    
    def cleanup(self):
        """Remove all temporary files from contained LogImage/LogVideo items."""
        for row in self.rows:
            for item in row:
                if item is not None and hasattr(item, 'cleanup'):
                    item.cleanup()

# ----------------------------------- LogFormatter Class -----------------------------------
class LogFormatter:
    """
    Standardizes input dictionaries for logging.
    Rules:
    1. Strings -> Check path extension -> LogImage/LogVideo
    2. List[Number/Tensor/Array] -> Mean value (float)
    3. PIL Image -> LogImage
    """
    
    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    VID_EXTENSIONS = ('.mp4', '.gif', '.mov', '.avi', '.webm')

    @classmethod
    def format_dict(cls, data: Union[Dict, Any]) -> Dict[str, Any]:
        """Entry point: Converts a Dict or Dataclass (BaseSample) into a clean loggable dict."""
        if is_dataclass(data):
            # Shallow conversion is usually enough, but deep conversion ensures lists are accessible
            data = asdict(data)
            
        if not isinstance(data, dict):
            raise ValueError(f"LogFormatter expects a dict or dataclass, got {type(data)}")

        clean_data = {}
        for k, v in data.items():
            clean_data[k] = cls._process_value(v)
        
        return clean_data

    @classmethod
    def _process_sample_list(cls, samples: List[BaseSample]) -> Union[List[Union[LogImage, LogVideo]], LogTable]:
        """Dispatch to appropriate handler based on sample type."""
        # If there are inherit relationships, order matters - more specific types should come first
        sample_cls_to_handler = {
            V2VSample: cls._process_v2v_samples,
            I2VSample: cls._process_i2v_samples,
            I2ISample: cls._process_i2i_samples,
            T2VSample: cls._process_t2v_samples,
            T2ISample: cls._process_t2i_samples,
        }

        first_cls = type(samples[0])
        if not all(isinstance(s, first_cls) for s in samples):
            logger.warning("Mixed sample types detected; unexpected behavior may occur.")

        def get_handler(cls_type):
            for sample_cls in sample_cls_to_handler:
                if issubclass(cls_type, sample_cls):
                    return sample_cls_to_handler[sample_cls]
            return None

        handler = get_handler(first_cls) or cls._process_base_samples

        result = handler(samples)
        return result

    @classmethod
    def _process_base_samples(cls, samples: List[BaseSample]) -> List[Union[LogImage, LogVideo, None]]:
        """Handle basic sample with single generated image."""
        def _process_single_base_sample(s: BaseSample) -> Optional[Union[LogImage, LogVideo]]:
            if s.image is not None:
                return LogImage(s.image, caption=_build_sample_caption(s))
            elif s.video is not None:
                return LogVideo(s.video, caption=_build_sample_caption(s))
            return None
        
        results = [_process_single_base_sample(s) for s in samples]

        return results
    
    @classmethod
    def _process_t2i_samples(cls, samples: List[T2ISample]) -> List[Union[LogImage, None]]:
        """Handle text-to-image sample with generated image."""
        def _process_single_t2i_sample(sample: T2ISample) -> Optional[LogImage]:
            if sample.image is None:
                return None
            return LogImage(sample.image, caption=_build_sample_caption(sample))

        results = [_process_single_t2i_sample(s) for s in samples]
        return results
        
    @classmethod
    def _process_t2v_samples(cls, samples: List[T2VSample]) -> List[Union[LogVideo, None]]:
        """Handle text-to-video sample with generated video."""
        def _process_single_t2v_sample(sample: T2VSample) -> Optional[LogVideo]:
            if sample.video is None:
                return None
            return LogVideo(sample.video, caption=_build_sample_caption(sample))

        results = [_process_single_t2v_sample(s) for s in samples]
        return results
    
    @classmethod
    def _process_i2i_samples(cls, samples: List[I2ISample]) -> List[Union[LogImage, None]]:
        """Handle sample with condition images + generated image, concatenated in grid."""
        def _process_single_i2i_sample(sample: I2ISample) -> Optional[LogImage]:
            cond_imgs = _to_pil_list(sample.condition_images)
            gen_imgs = _to_pil_list(sample.image)
            all_imgs = cond_imgs + gen_imgs
            
            if not all_imgs:
                return None
        
            grid = _concat_images_grid(all_imgs) if len(all_imgs) > 1 else all_imgs[0]
            return LogImage(grid, caption=_build_sample_caption(sample))
        
        results = [_process_single_i2i_sample(s) for s in samples]
        return results
    
    @classmethod
    def _process_i2v_samples(cls, samples: List[I2VSample]) -> Union[LogTable, None]:
        """Handle sample with condition images + generated video, as LogTable."""
        table = LogTable.from_i2v_samples(samples)
        return table
    
    @classmethod
    def _process_v2v_samples(cls, samples: List[V2VSample]) -> Union[LogTable, None]:
        """Handle sample with condition videos + generated video, as LogTable."""
        table = LogTable.from_v2v_samples(samples)
        return table

    @classmethod
    def _process_value(cls, value: Any) -> Any:
        """Processes a single value according to the formatting rules."""
        # Rule 0: BaseSample or List of BaseSample
        if isinstance(value, BaseSample):
            value = [value]
        if cls._is_sample_collection(value):
            return cls._process_sample_list(value)

        # Rule 1: PIL Image
        if isinstance(value, Image.Image):
            return LogImage(value)

        # Rule 2: String paths
        if isinstance(value, str):
            if os.path.exists(value):
                ext = os.path.splitext(value)[1].lower()
                file_name = os.path.basename(value)
                if ext in cls.IMG_EXTENSIONS:
                    return LogImage(value, caption=file_name)
                if ext in cls.VID_EXTENSIONS:
                    return LogVideo(value, caption=file_name)
            # If string is not a path or file doesn't exist, log as string text
            return value

        # Rule 3: Lists / Arrays / Tensors (Aggregations)
        if cls._is_numerical_collection(value):
            return cls._compute_mean(value)

        # Handle single Tensors/Numpy arrays that aren't images
        if isinstance(value, (torch.Tensor, np.ndarray)):
             if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
                 return cls._compute_mean(value)

        return value

    @classmethod
    def _is_sample_collection(cls, value: Any) -> bool:
        """Checks if value is a list/tuple of BaseSample."""
        if isinstance(value, (list, tuple)):
            if len(value) == 0: return False
            first = value[0]
            return isinstance(first, BaseSample)
        return False

    @classmethod
    def _is_numerical_collection(cls, value: Any) -> bool:
        """Checks if value is a list/tuple of numbers, arrays, or tensors."""
        if isinstance(value, (list, tuple)):
            if len(value) == 0: return False
            first = value[0]
            return isinstance(first, (int, float, complex, np.number, torch.Tensor, np.ndarray))
        return False

    @classmethod
    def _compute_mean(cls, value: Union[List, torch.Tensor, np.ndarray]) -> float:
        """Detaches tensors, converts to float, and computes mean."""
        try:
            # Handle List of Tensors / Arrays
            if isinstance(value, (list, tuple)):
                if isinstance(value[0], torch.Tensor):
                    # Stack and mean
                    return torch.stack([v.detach().cpu().float() for v in value]).mean().item()
                elif isinstance(value[0], (np.ndarray, np.number)):
                    return float(np.mean(value))
                else:
                    # Simple python numbers
                    return float(sum(value) / len(value))
            
            # Handle Direct Tensor
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().float().mean().item()
            
            # Handle Direct Numpy
            if isinstance(value, np.ndarray):
                return float(value.mean())
                
        except Exception as e:
            # Fallback if computation fails
            print(f"Warning: Failed to compute mean for value. Error: {e}")
            return 0.0
            
        return float(value)