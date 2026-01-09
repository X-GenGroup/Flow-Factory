# src/flow_factory/logger/formatting.py
import os
import tempfile
import math

import torch
import numpy as np
from PIL import Image
import imageio
from typing import Any, Dict, List, Union, Optional, Tuple
from dataclasses import dataclass, is_dataclass, asdict, field
from ..models.samples import BaseSample, ImageConditionSample, VideoConditionSample
from ..utils.base import numpy_to_pil_image, tensor_to_pil_image


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
    if isinstance(images, list) and images and isinstance(images[0], Image.Image):
        return images
    if isinstance(images, torch.Tensor):
        return tensor_to_pil_image(images)
    if isinstance(images, np.ndarray):
        return numpy_to_pil_image(images)
    return []


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
    _temp_path: Optional[str] = field(default=None, init=False, repr=False)

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

    @property
    def value(self) -> str:
        """Get video file path (converts tensor/array/frames to mp4)."""
        if self._temp_path:
            return self._temp_path
        if isinstance(self._value, str):
            return self._value
        
        arr = self.to_numpy(self._value)
        fd, path = tempfile.mkstemp(suffix='.mp4')
        try:
            os.close(fd)
            imageio.mimwrite(path, arr, fps=self.fps, codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
            self._temp_path = path
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            raise
        return self._temp_path

    @value.setter
    def value(self, val: Union[str, np.ndarray, torch.Tensor, List[Image.Image]]):
        self.cleanup()
        self._value = val
        self._temp_path = None

    def cleanup(self):
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.unlink(self._temp_path)
            finally:
                self._temp_path = None

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

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
    def _build_caption(cls, sample: BaseSample, max_length : Optional[int] = None) -> str:
        """Build caption from reward and prompt."""
        parts = []
        if 'reward' in sample.extra_kwargs:
            parts.append(f"{sample.extra_kwargs['reward']:.2f}")
        if sample.prompt:
            parts.append(sample.prompt[:max_length] + "..." if (max_length is not None and len(sample.prompt) > max_length) else sample.prompt)
        return " | ".join(parts)

    @classmethod
    def _process_sample(cls, samples: List[BaseSample]) -> List[Union[LogImage, LogVideo]]:
        """Dispatch to appropriate handler based on sample type."""
        results = []
        for sample in samples:
            if isinstance(sample, VideoConditionSample):
                result = cls._process_video_condition_sample(sample)
            elif isinstance(sample, ImageConditionSample):
                result = cls._process_image_condition_sample(sample)
            else:
                result = cls._process_base_sample(sample)
            if result:
                results.append(result)
        return results
    
    @classmethod
    def _process_base_sample(cls, sample: BaseSample) -> Optional[Union[LogImage, LogVideo]]:
        """Handle basic sample with single generated image."""
        if sample.image is not None:
            return LogImage(sample.image, caption=cls._build_caption(sample))
        elif sample.video is not None:
            return LogVideo(sample.video, caption=cls._build_caption(sample))

        return None
        
    @classmethod
    def _process_image_condition_sample(cls, sample: ImageConditionSample) -> Optional[LogImage]:
        """Handle sample with condition images + generated image, concatenated in grid."""
        cond_imgs = _to_pil_list(sample.condition_images)
        gen_imgs = _to_pil_list(sample.image)
        all_imgs = cond_imgs + gen_imgs
        
        if not all_imgs:
            return None
        
        grid = _concat_images_grid(all_imgs) if len(all_imgs) > 1 else all_imgs[0]
        return LogImage(grid, caption=cls._build_caption(sample))
    
    @classmethod
    def _process_video_condition_sample(cls, sample: VideoConditionSample) -> Optional[LogVideo]:
        """Handle video condition sample (placeholder for future extension)."""
        # TODO: Implement video grid concatenation
        return None
    
    @classmethod
    def _process_value(cls, value: Any) -> Any:
        """Processes a single value according to the formatting rules."""
        # Rule 0: BaseSample or List of BaseSample
        if isinstance(value, BaseSample):
            value = [value]
        if cls._is_sample_collection(value):
            return cls._process_sample(value)

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