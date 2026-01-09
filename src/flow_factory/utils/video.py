# src/flow_factory/utils/video.py
"""
Video utility functions for converting between PIL frames, torch Tensors, and NumPy arrays.

Supported formats:
    - VideoFrames: List[PIL.Image] - Single video as list of frames
    - VideoFramesBatch: List[List[PIL.Image]] - Batch of videos
    - torch.Tensor: (T, C, H, W) for single video, (B, T, C, H, W) for batch
    - np.ndarray: (T, H, W, C) for single video, (B, T, H, W, C) for batch

Value ranges:
    - [0, 255]: Standard uint8 format
    - [0, 1]: Normalized float format (PyTorch convention)
    - [-1, 1]: Normalized float format (diffusion models)
"""

from typing import List, Union, Any
from PIL import Image
import torch
import numpy as np


# ----------------------------------- Type Aliases --------------------------------------

VideoFrames = List[Image.Image]
"""Type alias for a single video represented as a list of PIL Images."""

VideoFramesBatch = List[List[Image.Image]]
"""Type alias for a batch of videos, each represented as a list of PIL Images."""


__all__ = [
    # Type aliases
    'VideoFrames',
    'VideoFramesBatch',
    # Type checks
    'is_video_frame_list',
    'is_video_frame_batch_list',
    # Validation
    'is_valid_video',
    'is_valid_video_list',
    'is_valid_video_batch',
    # Tensor/NumPy -> Frames
    'tensor_to_video_frames',
    'numpy_to_video_frames',
    'tensor_list_to_video_frames',
    'numpy_list_to_video_frames',
    # Frames -> Tensor/NumPy
    'video_frames_to_tensor',
    'video_frames_to_numpy',
]


# ----------------------------------- Type Check --------------------------------------

def is_video_frame_list(frames: List[Any]) -> bool:
    """
    Check if the input is a list of PIL Images representing video frames.
    
    Args:
        frames: List to check.
    
    Returns:
        bool: True if all elements are PIL Images and list is non-empty, False otherwise.
    
    Example:
        >>> frames = [Image.new('RGB', (64, 64)) for _ in range(10)]
        >>> is_video_frame_list(frames)
        True
        >>> is_video_frame_list([])
        False
    """
    return isinstance(frames, list) and len(frames) > 0 and all(isinstance(f, Image.Image) for f in frames)


def is_video_frame_batch_list(frame_batches: VideoFramesBatch) -> bool:
    """
    Check if the input is a list of lists of PIL Images (batch of videos).
    
    Args:
        frame_batches: List of lists to check.
    
    Returns:
        bool: True if all sublists are valid video frame lists, False otherwise.
    
    Example:
        >>> batch = [[Image.new('RGB', (64, 64)) for _ in range(10)] for _ in range(4)]
        >>> is_video_frame_batch_list(batch)
        True
    """
    return (
        isinstance(frame_batches, list) and 
        len(frame_batches) > 0 and 
        all(is_video_frame_list(batch) for batch in frame_batches)
    )


# ----------------------------------- Validation --------------------------------------

def is_valid_video(video: Union[VideoFrames, torch.Tensor, np.ndarray]) -> bool:
    """
    Check if the input is a valid video type.
    
    Args:
        video: Input video in one of the supported formats.
    
    Returns:
        bool: True if valid video type:
            - VideoFrames (List[PIL.Image]): Non-empty list of PIL Images
            - torch.Tensor: Shape (T, C, H, W) where C in {1, 3, 4}
            - np.ndarray: Shape (T, H, W, C) where C in {1, 3, 4}
    
    Example:
        >>> # Tensor video
        >>> video_tensor = torch.rand(16, 3, 256, 256)  # 16 frames
        >>> is_valid_video(video_tensor)
        True
        
        >>> # NumPy video
        >>> video_array = np.random.randint(0, 256, (16, 256, 256, 3), dtype=np.uint8)
        >>> is_valid_video(video_array)
        True
        
        >>> # PIL frame list
        >>> frames = [Image.new('RGB', (256, 256)) for _ in range(16)]
        >>> is_valid_video(frames)
        True
    """
    if isinstance(video, list):
        return is_video_frame_list(video)
    
    if isinstance(video, torch.Tensor):
        if video.ndim != 4:
            return False
        t, c, h, w = video.shape
        return t > 0 and c in (1, 3, 4) and h > 0 and w > 0
    
    if isinstance(video, np.ndarray):
        if video.ndim != 4:
            return False
        t, h, w, c = video.shape
        return t > 0 and h > 0 and w > 0 and c in (1, 3, 4)
    
    return False


def is_valid_video_list(videos: List[Union[VideoFrames, torch.Tensor, np.ndarray]]) -> bool:
    """
    Check if the input is a valid list of videos.
    
    Args:
        videos: List of videos to check.
    
    Returns:
        bool: True if valid video list:
            - Non-empty list
            - All elements are valid videos
            - All elements are of the same type
    
    Example:
        >>> videos = [torch.rand(16, 3, 64, 64) for _ in range(4)]
        >>> is_valid_video_list(videos)
        True
    """
    if not isinstance(videos, list) or len(videos) == 0:
        return False
    
    first_type = type(videos[0])
    if not all(isinstance(v, first_type) for v in videos):
        return False
    
    return all(is_valid_video(v) for v in videos)


def is_valid_video_batch(videos: Union[VideoFramesBatch, torch.Tensor, np.ndarray]) -> bool:
    """
    Check if the input is a valid batch of videos.
    
    Args:
        videos: Input video batch.
    
    Returns:
        bool: True if valid video batch:
            - VideoFramesBatch (List[List[PIL.Image]]): Batch of frame lists
            - torch.Tensor: Shape (B, T, C, H, W) where C in {1, 3, 4}
            - np.ndarray: Shape (B, T, H, W, C) where C in {1, 3, 4}
    
    Example:
        >>> # Batched tensor
        >>> batch_tensor = torch.rand(4, 16, 3, 256, 256)  # 4 videos, 16 frames each
        >>> is_valid_video_batch(batch_tensor)
        True
        
        >>> # Batched numpy
        >>> batch_array = np.random.randint(0, 256, (4, 16, 256, 256, 3), dtype=np.uint8)
        >>> is_valid_video_batch(batch_array)
        True
    """
    if isinstance(videos, list):
        return is_video_frame_batch_list(videos)
    
    if isinstance(videos, torch.Tensor):
        if videos.ndim != 5:
            return False
        b, t, c, h, w = videos.shape
        return b > 0 and t > 0 and c in (1, 3, 4) and h > 0 and w > 0
    
    if isinstance(videos, np.ndarray):
        if videos.ndim != 5:
            return False
        b, t, h, w, c = videos.shape
        return b > 0 and t > 0 and h > 0 and w > 0 and c in (1, 3, 4)
    
    return False


# ----------------------------------- Normalization --------------------------------------

def _normalize_video_to_uint8(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Detect value range and normalize video data to [0, 255] uint8.
    
    Args:
        data: Input tensor or array with values in one of three ranges:
            - [0, 255]: Standard uint8 format (common in NumPy/PIL)
            - [0, 1]: Normalized float format (common in PyTorch)
            - [-1, 1]: Normalized float format (common in diffusion models)
    
    Returns:
        Data normalized to [0, 255] and converted to uint8 dtype.
        Returns torch.Tensor if input is tensor, np.ndarray if input is array.
    
    Note:
        Range detection logic:
            - If min < 0 and values in [-1, 1]: treated as [-1, 1] range
            - Elif max <= 1.0: treated as [0, 1] range
            - Else: treated as [0, 255] range (no scaling applied)
    """
    is_tensor = isinstance(data, torch.Tensor)
    
    min_val = data.min().item() if is_tensor else data.min()
    max_val = data.max().item() if is_tensor else data.max()
    
    if min_val >= -1.0 and max_val <= 1.0 and min_val < 0:
        # [-1, 1] -> [0, 255]
        data = (data + 1) / 2 * 255
    elif max_val <= 1.0:
        # [0, 1] -> [0, 255]
        data = data * 255
    # else: already [0, 255], no scaling needed
    
    if is_tensor:
        return data.round().clamp(0, 255).to(torch.uint8)
    return np.clip(np.round(data), 0, 255).astype(np.uint8)


# ----------------------------------- Tensor/NumPy -> Frames --------------------------------------

def tensor_to_video_frames(tensor: torch.Tensor) -> Union[VideoFrames, VideoFramesBatch]:
    """
    Convert a torch Tensor to video frames (PIL Images).
    
    Args:
        tensor: Video tensor of shape (T, C, H, W) or (B, T, C, H, W).
            Supported value ranges:
                - [0, 1]: Standard normalized tensor format
                - [-1, 1]: Normalized tensor format (e.g., from diffusion models)
    
    Returns:
        - If input is 4D (T, C, H, W): VideoFrames (List of T PIL Images)
        - If input is 5D (B, T, C, H, W): VideoFramesBatch (List of B videos)
    
    Raises:
        ValueError: If tensor is not 4D or 5D.
    
    Example:
        >>> # Single video
        >>> video_tensor = torch.rand(16, 3, 256, 256)
        >>> frames = tensor_to_video_frames(video_tensor)
        >>> len(frames)
        16
        
        >>> # Batch of videos
        >>> batch_tensor = torch.rand(4, 16, 3, 256, 256)
        >>> videos = tensor_to_video_frames(batch_tensor)
        >>> len(videos), len(videos[0])
        (4, 16)
    """
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    
    # (B, T, C, H, W) -> (B, T, H, W, C)
    tensor = _normalize_video_to_uint8(tensor).cpu().numpy()
    tensor = tensor.transpose(0, 1, 3, 4, 2)
    
    if tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    
    result = [[Image.fromarray(frame) for frame in video] for video in tensor]
    return result[0]


def numpy_to_video_frames(array: np.ndarray) -> Union[VideoFrames, VideoFramesBatch]:
    """
    Convert a NumPy array to video frames (PIL Images).
    
    Args:
        array: Video array of shape (T, H, W, C) / (T, C, H, W) or (B, T, H, W, C) / (B, T, C, H, W).
            Supported value ranges:
                - [0, 255]: Standard uint8 format
                - [0, 1]: Normalized float format
                - [-1, 1]: Normalized float format (e.g., from diffusion models)
    
    Returns:
        - If input is 4D: VideoFrames (List of T PIL Images)
        - If input is 5D: VideoFramesBatch (List of B videos)
    
    Raises:
        ValueError: If array is not 4D or 5D.
    
    Note:
        Channel dimension detection: If shape[2] (for 5D) or shape[1] (for 4D) is in {1, 3, 4}
        and smaller than the next dimension, the array is assumed to be channel-first and
        will be transposed to channel-last.
    
    Example:
        >>> # Single video (THWC)
        >>> video_array = np.random.rand(16, 256, 256, 3).astype(np.float32)
        >>> frames = numpy_to_video_frames(video_array)
        >>> len(frames)
        16
        
        >>> # Batch of videos
        >>> batch_array = np.random.randint(0, 256, (4, 16, 256, 256, 3), dtype=np.uint8)
        >>> videos = numpy_to_video_frames(batch_array)
        >>> len(videos), len(videos[0])
        (4, 16)
    """
    if array.ndim == 4:
        array = array[np.newaxis, ...]
    
    array = _normalize_video_to_uint8(array)
    
    # BTCHW -> BTHWC if channel dim detected
    if array.shape[2] in (1, 3, 4) and array.shape[2] < array.shape[3]:
        array = array.transpose(0, 1, 3, 4, 2)
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    result = [[Image.fromarray(frame) for frame in video] for video in array]
    return result[0]


def tensor_list_to_video_frames(tensor_list: List[torch.Tensor]) -> VideoFramesBatch:
    """
    Convert a list of torch Tensors to video frame lists.
    
    This function handles tensors with potentially different shapes by processing
    them individually when necessary, or batch-processing when all shapes match.
    
    Args:
        tensor_list: List of video tensors, each of shape (T, C, H, W) or (1, T, C, H, W).
            Each tensor can have different T, H, W dimensions.
            Supported value ranges: [0, 1] or [-1, 1].
    
    Returns:
        VideoFramesBatch: List of videos, each a list of PIL Images.
    
    Note:
        - Tensors with shape (1, T, C, H, W) are automatically squeezed to (T, C, H, W).
        - If all tensors have the same shape, they are stacked and batch-processed
          for efficiency.
        - If tensors have different shapes, they are processed individually.
    
    Example:
        >>> # Same shape (batch-processed)
        >>> tensors = [torch.rand(16, 3, 256, 256) for _ in range(4)]
        >>> videos = tensor_list_to_video_frames(tensors)
        >>> len(videos), len(videos[0])
        (4, 16)
        
        >>> # Different shapes (processed individually)
        >>> tensors = [torch.rand(16, 3, 256, 256), torch.rand(24, 3, 512, 512)]
        >>> videos = tensor_list_to_video_frames(tensors)
        >>> len(videos[0]), len(videos[1])
        (16, 24)
    """
    if not tensor_list:
        return []
    
    # Squeeze batch dim if present
    squeezed = [t.squeeze(0) if t.ndim == 5 and t.shape[0] == 1 else t for t in tensor_list]
    
    # Uniform shape -> batch process
    if all(t.shape == squeezed[0].shape for t in squeezed):
        return tensor_to_video_frames(torch.stack(squeezed, dim=0))
    
    # Variable shape -> process individually (returns VideoFrames for each 4D tensor)
    return [tensor_to_video_frames(t) for t in squeezed]


def numpy_list_to_video_frames(numpy_list: List[np.ndarray]) -> VideoFramesBatch:
    """
    Convert a list of NumPy arrays to video frame lists.
    
    This function handles arrays with potentially different shapes by processing
    them individually when necessary, or batch-processing when all shapes match.
    
    Args:
        numpy_list: List of video arrays, each of shape (T, H, W, C) or (T, C, H, W).
            Each array can have different T, H, W dimensions.
            Supported value ranges: [0, 255], [0, 1], or [-1, 1].
    
    Returns:
        VideoFramesBatch: List of videos, each a list of PIL Images.
    
    Note:
        - Arrays with shape (1, T, H, W, C) are automatically squeezed.
        - If all arrays have the same shape, they are stacked and batch-processed
          for efficiency.
        - If arrays have different shapes, they are processed individually.
    
    Example:
        >>> # Same shape (batch-processed)
        >>> arrays = [np.random.rand(16, 256, 256, 3) for _ in range(4)]
        >>> videos = numpy_list_to_video_frames(arrays)
        >>> len(videos), len(videos[0])
        (4, 16)
        
        >>> # Different shapes (processed individually)
        >>> arrays = [np.random.rand(16, 256, 256, 3), np.random.rand(24, 512, 512, 3)]
        >>> videos = numpy_list_to_video_frames(arrays)
        >>> len(videos[0]), len(videos[1])
        (16, 24)
    """
    if not numpy_list:
        return []
    
    # Squeeze batch dim if present
    squeezed = [arr.squeeze(0) if arr.ndim == 5 and arr.shape[0] == 1 else arr for arr in numpy_list]
    
    # Uniform shape -> batch process
    if all(arr.shape == squeezed[0].shape for arr in squeezed):
        return numpy_to_video_frames(np.stack(squeezed, axis=0))
    
    # Variable shape -> process individually
    return [numpy_to_video_frames(arr) for arr in squeezed]


# ----------------------------------- Frames -> Tensor/NumPy --------------------------------------

def video_frames_to_tensor(
    frames: Union[VideoFrames, VideoFramesBatch]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Convert PIL frames to torch Tensor(s).
    
    Args:
        frames: Single video (VideoFrames) or batch (VideoFramesBatch).
            All frames within a video should have the same dimensions.
    
    Returns:
        - If VideoFrames (List[PIL]): Tensor of shape (T, C, H, W) with values in [0, 1]
        - If VideoFramesBatch (List[List[PIL]]):
            - Same shape videos: Tensor of shape (B, T, C, H, W)
            - Variable shape: List of Tensors, each (T, C, H, W)
    
    Raises:
        ValueError: If frames list is empty or contains empty video.
    
    Example:
        >>> # Single video
        >>> frames = [Image.new('RGB', (64, 64)) for _ in range(16)]
        >>> tensor = video_frames_to_tensor(frames)
        >>> tensor.shape
        torch.Size([16, 3, 64, 64])
        
        >>> # Batch of videos (same shape)
        >>> batch = [[Image.new('RGB', (64, 64)) for _ in range(16)] for _ in range(4)]
        >>> tensor = video_frames_to_tensor(batch)
        >>> tensor.shape
        torch.Size([4, 16, 3, 64, 64])
        
        >>> # Batch of videos (variable shape)
        >>> batch = [
        ...     [Image.new('RGB', (64, 64)) for _ in range(16)],
        ...     [Image.new('RGB', (128, 128)) for _ in range(24)]
        ... ]
        >>> tensors = video_frames_to_tensor(batch)
        >>> tensors[0].shape, tensors[1].shape
        (torch.Size([16, 3, 64, 64]), torch.Size([24, 3, 128, 128]))
    """
    if not frames:
        raise ValueError("Empty frame list")
    
    # Single video: List[PIL]
    if isinstance(frames[0], Image.Image):
        arrays = [np.array(img.convert('RGB')).astype(np.float32) / 255.0 for img in frames]
        tensors = [torch.from_numpy(arr).permute(2, 0, 1) for arr in arrays]
        return torch.stack(tensors, dim=0)
    
    # Batch: List[List[PIL]]
    if isinstance(frames[0], list) and not frames[0]:
        raise ValueError("Empty video in batch")
    
    converted = [video_frames_to_tensor(v) for v in frames]
    
    # Stack if uniform shape
    if all(t.shape == converted[0].shape for t in converted):
        return torch.stack(converted, dim=0)
    return converted


def video_frames_to_numpy(
    frames: Union[VideoFrames, VideoFramesBatch]
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Convert PIL frames to NumPy array(s).
    
    Args:
        frames: Single video (VideoFrames) or batch (VideoFramesBatch).
            All frames within a video should have the same dimensions.
    
    Returns:
        - If VideoFrames (List[PIL]): Array of shape (T, H, W, C) with uint8 dtype
        - If VideoFramesBatch (List[List[PIL]]):
            - Same shape videos: Array of shape (B, T, H, W, C)
            - Variable shape: List of Arrays, each (T, H, W, C)
    
    Raises:
        ValueError: If frames list is empty or contains empty video.
    
    Example:
        >>> # Single video
        >>> frames = [Image.new('RGB', (64, 64)) for _ in range(16)]
        >>> array = video_frames_to_numpy(frames)
        >>> array.shape
        (16, 64, 64, 3)
        
        >>> # Batch of videos (same shape)
        >>> batch = [[Image.new('RGB', (64, 64)) for _ in range(16)] for _ in range(4)]
        >>> array = video_frames_to_numpy(batch)
        >>> array.shape
        (4, 16, 64, 64, 3)
        
        >>> # Batch of videos (variable shape)
        >>> batch = [
        ...     [Image.new('RGB', (64, 64)) for _ in range(16)],
        ...     [Image.new('RGB', (128, 128)) for _ in range(24)]
        ... ]
        >>> arrays = video_frames_to_numpy(batch)
        >>> arrays[0].shape, arrays[1].shape
        ((16, 64, 64, 3), (24, 128, 128, 3))
    """
    if not frames:
        raise ValueError("Empty frame list")
    
    # Single video: List[PIL]
    if isinstance(frames[0], Image.Image):
        return np.stack([np.array(img.convert('RGB')) for img in frames], axis=0)
    
    # Batch: List[List[PIL]]
    if isinstance(frames[0], list) and not frames[0]:
        raise ValueError("Empty video in batch")
    
    converted = [video_frames_to_numpy(v) for v in frames]
    
    # Stack if uniform shape
    if all(arr.shape == converted[0].shape for arr in converted):
        return np.stack(converted, axis=0)
    return converted