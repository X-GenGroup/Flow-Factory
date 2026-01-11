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
    video_frames_to_numpy,
    video_frames_to_tensor,
    tensor_to_video_frames,
    numpy_to_video_frames,
    tensor_list_to_pil_image,
    numpy_list_to_pil_image,
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

def _build_sample_caption(sample : BaseSample, max_length: Optional[int] = None) -> str:
    """Build caption from reward and prompt."""
    parts = []
    if 'reward' in sample.extra_kwargs:
        parts.append(f"{sample.extra_kwargs['reward']:.2f}")
    if sample.prompt:
        parts.append(sample.prompt[:max_length] + "..." if (max_length is not None and len(sample.prompt) > max_length) else sample.prompt)
    return " | ".join(parts)


# ------------------------------------------- LogImage & LogVideo Classes -------------------------------------------

@dataclass
class LogImage:
    """Intermediate representation for an Image with compression support."""
    _value: Union[str, Image.Image, np.ndarray, torch.Tensor] = field(repr=False)
    _img: Optional[Image.Image] = field(default=None, init=False, repr=False)
    caption: Optional[str] = None
    compress: bool = True
    quality: int = 85
    _temp_path: Optional[str] = field(default=None, init=False, repr=False)
    
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

    @property
    def value(self) -> Union[str, Image.Image]:
        """Get compressed .jpg file path or original value."""
        if self._temp_path:
            return self._temp_path
            
        # If already a path, return as-is
        if isinstance(self._value, str):
            return self._value
        
        # Convert to PIL Image
        if self._img is None:
            self._img = LogImage.to_pil(self._value)

        # Save to temporary file if compression enabled
        if self.compress:
            # Using mkstemp ensures the file exists and gives us control over closing it
            fd, path = tempfile.mkstemp(suffix='.jpg')
            try:
                with os.fdopen(fd, 'wb') as f:
                    self._img.convert('RGB').save(f, format='JPEG', quality=self.quality)
                self._temp_path = path
            except Exception as e:
                if os.path.exists(path):
                    os.unlink(path)
                raise e
            return self._temp_path

        return self._img
    
    @value.setter
    def value(self, val: Union[str, Image.Image, np.ndarray, torch.Tensor]):
        """Set the value and reset all cached state."""
        self.cleanup()  # Clean up existing temp files before replacing
        self._value = val
        self._img = None
        self._temp_path = None
    
    def cleanup(self):
        """Remove temporary file if created."""
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.unlink(self._temp_path)
            finally:
                self._temp_path = None

    def __del__(self):
        """Destructor to prevent storage leaks if cleanup is forgotten."""
        self.cleanup()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit to ensure cleanup."""
        self.cleanup()

    def __enter__(self):
        """Context manager enter."""
        return self


@dataclass
class LogVideo:
    """Intermediate representation for a Video with format conversion support."""
    _value: Union[str, np.ndarray, torch.Tensor, List[Image.Image]] = field(repr=False)
    caption: Optional[str] = None
    fps: int = 8
    _temp_paths: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    @property
    def format(self) -> str:
        """Get video format extension (without dot)."""
        if isinstance(self._value, str):
            return os.path.splitext(self._value)[1].lstrip('.').lower() or 'mp4'
        return 'mp4'  # defaults to `mp4`
    
    @classmethod
    def to_numpy(cls, value: Union[np.ndarray, torch.Tensor, List[Image.Image]]) -> np.ndarray:
        """Convert to numpy array (T, H, W, C), uint8."""
        if isinstance(value, str):
            raise ValueError("Cannot convert path to numpy")
        
        # Handle List[PIL.Image]
        if isinstance(value, list) and value and isinstance(value[0], Image.Image):
            frames = [np.array(img.convert('RGB')) for img in value]
            arr = np.stack(frames, axis=0)
        elif isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = value
        # (T, C, H, W) -> (T, H, W, C) if channels-first
        if arr.ndim == 4 and arr.shape[1] in (1, 3, 4) and arr.shape[1] < arr.shape[2]:
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 3:
            arr = arr[..., np.newaxis]
        
        # Normalize to uint8
        if arr.dtype != np.uint8:
            arr = ((arr * 255) if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
        return arr
    
    def get_value(self, format: str = 'mp4') -> str:
        """Get video file path in specified format (converts if needed)."""
        format = format.lower().lstrip('.')
        
        # Check cache
        if format in self._temp_paths:
            return self._temp_paths[format]
        
        # If source is a file with matching format, return directly
        if isinstance(self._value, str):
            if self.format == format:
                return self._value
            arr = imageio.mimread(self._value)
            arr = np.stack(arr, axis=0)
        else:
            arr = self.to_numpy(self._value)
        
        # Write to temp file
        fd, path = tempfile.mkstemp(suffix=f'.{format}')
        try:
            os.close(fd)
            if format == 'gif':
                imageio.mimwrite(
                    path,
                    arr,
                    fps=self.fps,
                    format='GIF',
                    loop=0
                )
            else:  # mp4
                imageio.mimwrite(
                    path,
                    arr,
                    fps=self.fps,
                    format='FFMPEG',
                    codec='libx264',
                    pixelformat='yuv420p'
                )
            self._temp_paths[format] = path
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            raise
        return path

    @property
    def value(self) -> str:
        """Default: get mp4 path."""
        return self.get_value('mp4')

    @value.setter
    def value(self, val: Union[str, np.ndarray, torch.Tensor, List[Image.Image]]):
        self.cleanup()
        self._value = val

    def cleanup(self):
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
    """Table for [cond_1, cond_2, ..., cond_n, generation] logging."""
    columns: List[str] = field(default_factory=list)
    rows: List[List[Union[LogImage, LogVideo]]] = field(default_factory=list)
    
    @classmethod
    def from_i2v_samples(cls, samples: List[I2VSample]) -> Optional[LogTable]:
        """Build table from I2V samples: condition_images -> video."""
        if not samples or not hasattr(samples[0], 'condition_images'):
            return None
        
        # Get column count from first valid sample
        first_conds = _to_pil_list(samples[0].condition_images)
        n_conds = len(first_conds)
        columns = [f"condition_image_{i}" for i in range(n_conds)] + ["generation"]
        
        rows = []
        for s in samples:
            if s.video is None:
                continue
            conds = _to_pil_list(s.condition_images)[:n_conds]
            # Pad if needed
            while len(conds) < n_conds:
                conds.append(conds[-1] if conds else Image.new('RGB', (64, 64)))
            
            caption = _build_sample_caption(s)
            row = [LogImage(c) for c in conds] + [LogVideo(s.video, caption=caption)]
            rows.append(row)
        
        return cls(columns=columns, rows=rows) if rows else None
    
    @classmethod
    def from_v2v_samples(cls, samples: List[V2VSample]) -> Optional[LogTable]:
        """Build table from V2V samples: condition_videos -> video."""
        if not samples or not hasattr(samples[0], 'condition_videos'):
            return None
        
        first_conds = samples[0].condition_videos
        n_conds = len(first_conds) if isinstance(first_conds, list) else 1
        columns = [f"condition_video_{i}" for i in range(n_conds)] + ["generation"]
        
        rows = []
        for s in samples:
            if s.video is None:
                continue
            conds = s.condition_videos if isinstance(s.condition_videos, list) else [s.condition_videos]
            conds = conds[:n_conds]
            
            caption = _build_sample_caption(s)
            row = [LogVideo(c) for c in conds] + [LogVideo(s.video, caption=caption)]
            rows.append(row)
        
        return cls(columns=columns, rows=rows) if rows else None
    
    def cleanup(self):
        for row in self.rows:
            for item in row:
                if hasattr(item, 'cleanup'):
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