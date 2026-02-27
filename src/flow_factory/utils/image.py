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

# src/flow_factory/utils/image.py
"""
Image utility functions for converting between PIL Images, torch Tensors, and NumPy arrays.

Type Hierarchy:
    ImageSingle
        ├─ PIL.Image.Image              Single image
        ├─ torch.Tensor (C, H, W)       Single image tensor
        └─ np.ndarray (H, W, C)         Single image array
    
    ImageBatch                          Batch of images
        ├─ torch.Tensor (N, C, H, W)    Stacked batch
        ├─ np.ndarray (N, H, W, C)      Stacked batch
        └─ List[ImageSingle]            List of images (variable shapes allowed)
    
    MultiImageBatch                     Multiple images per sample (e.g., conditioning images)
        ├─ torch.Tensor (B, N, C, H, W) Uniform batch: B samples, N images each
        ├─ np.ndarray (B, N, H, W, C)   Uniform batch: B samples, N images each
        └─ List[ImageBatch]             Ragged batch: variable N per sample

Tensor/Array Conventions:
    - torch.Tensor: Channel-first (C, H, W) or (N, C, H, W)
    - np.ndarray: Channel-last (H, W, C) or (N, H, W, C)
    - Channels: C ∈ {1, 3, 4} for grayscale, RGB, RGBA

Value Ranges:
    - [0, 255]: Standard uint8 format (NumPy/PIL convention)
    - [0, 1]: Normalized float format (PyTorch convention)
    - [-1, 1]: Normalized float format (diffusion model convention)

Main Functions:
    Type Validation:
        - is_image(), is_image_list(), is_image_batch(), is_multi_image_batch()
    
    Conversions:
        - tensor_to_pil_image(), numpy_to_pil_image()
        - pil_image_to_tensor(), pil_image_to_numpy()
    
    Standardization:
        - standardize_image_batch(): Unified conversion to pil/np/pt formats
        - normalize_to_uint8(): Auto-detect range and normalize to [0, 255]

Examples:
    >>> # Single image to batch
    >>> img = Image.new('RGB', (256, 256))
    >>> batch = standardize_image_batch(img, output_type='pt')
    >>> batch.shape
    torch.Size([1, 3, 256, 256])
    
    >>> # Multi-image batch (conditioning)
    >>> cond_images = torch.rand(4, 3, 3, 512, 512)  # 4 samples, 3 conditions each
    >>> is_multi_image_batch(cond_images)
    True
"""

import base64
from io import BytesIO
from typing import List, Union, Any, Literal

from PIL import Image
import torch
import numpy as np


# ----------------------------------- Type Aliases --------------------------------------

ImageSingle = Union[Image.Image, torch.Tensor, np.ndarray]
"""Type alias for a single image in various formats."""

ImageList = List[ImageSingle]
"""Type alias for a list of PIL Images."""

ImageBatch = Union[
    np.ndarray, # (N, H, W, C)
    torch.Tensor, # (N, C, H, W)
    List[torch.Tensor], # List of (C, H, W)
    List[np.ndarray], # List of (H, W, C)
    List[Image.Image], # List of PIL Images
]
"""Type alias for a batch of image lists."""

MultiImageBatch = Union[
    List[ImageBatch], # List of batches (ragged)
    torch.Tensor, # (B, N, C, H, W)
    np.ndarray, # (B, N, H, W, C)
]
"""Type alias for a list of image batches."""


__all__ = [
    # Type aliases
    'ImageSingle',
    'ImageList',
    'ImageBatch',
    'MultiImageBatch',
    # Type checks
    'is_pil_image_list',
    'is_pil_image_batch_list',
    # Validation
    'is_image',
    'is_image_list',
    'is_image_batch',
    'is_multi_image_batch',
    # Tensor/NumPy -> PIL
    'tensor_to_pil_image',
    'numpy_to_pil_image',
    'tensor_list_to_pil_image',
    'numpy_list_to_pil_image',
    # PIL -> Tensor/NumPy/Base64
    'pil_image_to_tensor',
    'pil_image_to_numpy',
    'pil_image_to_base64',
    # Normalization
    'normalize_to_uint8',
    'standardize_image_batch',
]


# ----------------------------------- Type Check --------------------------------------

def is_pil_image_list(image_list: List[Any]) -> bool:
    """
    Check if the input is a list of PIL Images.
    
    Args:
        image_list: List to check.
    
    Returns:
        bool: True if all elements are PIL Images and list is non-empty, False otherwise.
    
    Example:
        >>> images = [Image.new('RGB', (64, 64)) for _ in range(4)]
        >>> is_pil_image_list(images)
        True
        >>> is_pil_image_list([])
        False
    """
    return isinstance(image_list, list) and len(image_list) > 0 and all(isinstance(img, Image.Image) for img in image_list)


def is_pil_image_batch_list(image_batch_list: MultiImageBatch) -> bool:
    """
    Check if the input is a list of lists of PIL Images (batch of image lists).
    
    Args:
        image_batch_list: List of lists to check.
    
    Returns:
        bool: True if all sublists are valid image lists, False otherwise.
    
    Example:
        >>> batch = [[Image.new('RGB', (64, 64)) for _ in range(3)] for _ in range(4)]
        >>> is_pil_image_batch_list(batch)
        True
    """
    return (
        isinstance(image_batch_list, list) and
        len(image_batch_list) > 0 and
        all(is_pil_image_list(batch) for batch in image_batch_list)
    )


# ----------------------------------- Validation --------------------------------------

def is_image(image: Any) -> bool:
    """
    Check if the input is a valid single image.
    Corresponds to type `ImageSingle`.
    
    Args:
        image: Input image in one of the supported formats.
    
    Returns:
        bool: True if valid image type:
            - PIL.Image: Valid PIL Image with positive dimensions
            - torch.Tensor: Shape (C, H, W) or (1, C, H, W) where C in {1, 3, 4}
            - np.ndarray: Shape (H, W, C) or (1, H, W, C) where C in {1, 3, 4}
    
    Example:
        >>> is_image(Image.new('RGB', (64, 64)))
        True
        >>> is_image(torch.rand(3, 256, 256))
        True
        >>> is_image(np.random.rand(256, 256, 3))
        True
    """
    if isinstance(image, Image.Image):
        return image.size[0] > 0 and image.size[1] > 0
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            c, h, w = image.shape
            return c in (1, 3, 4) and h > 0 and w > 0
        elif image.ndim == 4:
            b, c, h, w = image.shape
            return b == 1 and c in (1, 3, 4) and h > 0 and w > 0
        return False
    
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            h, w, c = image.shape
            return h > 0 and w > 0 and c in (1, 3, 4)
        elif image.ndim == 4:
            b, h, w, c = image.shape
            return b == 1 and h > 0 and w > 0 and c in (1, 3, 4)
        return False
    
    return False


def is_image_list(images: Any) -> bool:
    """
    Check if the input is a valid list of images.
    Corresponds to type List[ImageSingle].
    
    Args:
        images: List of images to check.
    
    Returns:
        bool: True if valid image list:
            - Non-empty list
            - All elements are valid images
            - All elements are of the same type
    
    Example:
        >>> images = [torch.rand(3, 64, 64) for _ in range(4)]
        >>> is_image_list(images)
        True
    """
    if not isinstance(images, list) or len(images) == 0:
        return False
    
    first_type = type(images[0])
    if not all(isinstance(img, first_type) for img in images):
        return False
    
    return all(is_image(img) for img in images)


def is_image_batch(images: Any) -> bool:
    """
    Check if the input is a valid batch of images.
    Corresponds to type `ImageBatch`.
    
    Args:
        images: Input image batch.
    
    Returns:
        bool: True if valid image batch:
            - ImageList (List[ImageSingle])
            - List[torch.Tensor] where each tensor is (C, H, W) or (1, C, H, W)
            - List[np.ndarray] where each array is (H, W, C) or (1, H, W, C)
            - torch.Tensor with shape (N, C, H, W)
            - np.ndarray with shape (N, H, W, C)
    
    Example:
        >>> is_image_batch(torch.rand(4, 3, 256, 256))
        True
        >>> is_image_batch(np.random.rand(4, 256, 256, 3))
        True
    """
    
    # 4D Tensor: (N, C, H, W)
    if isinstance(images, torch.Tensor):
        if images.ndim != 4:
            return False
        b, c, h, w = images.shape
        return b > 0 and c in (1, 3, 4) and h > 0 and w > 0
    
    # 4D NumPy: (N, H, W, C)
    if isinstance(images, np.ndarray):
        if images.ndim != 4:
            return False
        b, h, w, c = images.shape
        return b > 0 and h > 0 and w > 0 and c in (1, 3, 4)
    
    # List[ImageSingle]
    return is_image_list(images)


def is_multi_image_batch(image_batches: Any) -> bool:
    """
    Check if the input is a valid batch of multiple images. Useful for batch input of multiple conditioning images per sample.
    Corresponds to type `MultiImageBatch`.
    
    Supported formats:
        - List[ImageBatch]: Ragged batches (different sizes allowed)
        - torch.Tensor: Shape (B, N, C, H, W) - uniform batches
        - np.ndarray: Shape (B, N, H, W, C) - uniform batches
    """
    # 5D Tensor: (B, N, C, H, W)
    if isinstance(image_batches, torch.Tensor):
        if image_batches.ndim != 5:
            return False
        b, n, c, h, w = image_batches.shape
        return b > 0 and n > 0 and c in (1, 3, 4) and h > 0 and w > 0
    
    # 5D NumPy: (B, N, H, W, C)
    if isinstance(image_batches, np.ndarray):
        if image_batches.ndim != 5:
            return False
        b, n, h, w, c = image_batches.shape
        return b > 0 and n > 0 and h > 0 and w > 0 and c in (1, 3, 4)
    
    # List[ImageBatch]
    if not isinstance(image_batches, list) or len(image_batches) == 0: # If None, here will return False
        return False
    
    return all(is_image_batch(batch) for batch in image_batches)


# ----------------------------------- Normalization --------------------------------------

def normalize_to_uint8(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Detect value range and normalize to [0, 255] uint8.
    
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


# ----------------------------------- Tensor/NumPy -> PIL --------------------------------------

def tensor_to_pil_image(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert a torch Tensor to PIL Image(s).
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (N, C, H, W).
            Supported value ranges:
                - [0, 1]: Standard normalized tensor format
                - [-1, 1]: Normalized tensor format (e.g., from diffusion models)
    
    Returns:
        - If input is 3D (C, H, W): Single-ton list containing one PIL Image
        - If input is 4D (N, C, H, W): List of N PIL Images
    
    Raises:
        ValueError: If tensor is not 3D or 4D.
    
    Example:
        >>> # Single image
        >>> img_tensor = torch.rand(3, 256, 256)
        >>> pil_image = tensor_to_pil_image(img_tensor)[0] # Take the first and the only element
        >>> isinstance(pil_image, Image.Image)
        True
        
        >>> # Batch of images
        >>> batch_tensor = torch.rand(4, 3, 256, 256)
        >>> pil_images = tensor_to_pil_image(batch_tensor)
        >>> len(pil_images)
        4
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    # (N, C, H, W) -> (N, H, W, C)
    tensor = normalize_to_uint8(tensor).cpu().numpy()
    tensor = tensor.transpose(0, 2, 3, 1)
    
    if tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    
    result = [Image.fromarray(img) for img in tensor]
    return result


def numpy_to_pil_image(array: np.ndarray) -> List[Image.Image]:
    """
    Convert a NumPy array to PIL Image(s).
    
    Args:
        array: Image array of shape (H, W, C) / (C, H, W) or (N, H, W, C) / (N, C, H, W).
            Supported value ranges:
                - [0, 255]: Standard uint8 format
                - [0, 1]: Normalized float format
                - [-1, 1]: Normalized float format (e.g., from diffusion models)
    
    Returns:
        - If input is 3D: Single-ton list containing one PIL Image
        - If input is 4D: List of N PIL Images
    
    Raises:
        ValueError: If array is not 3D or 4D.
    
    Note:
        Channel dimension detection: If the suspected channel dimension is in {1, 3, 4}
        and smaller than spatial dimensions, the array is assumed to be channel-first
        and will be transposed to channel-last.
    
    Example:
        >>> # Single image (HWC)
        >>> img_array = np.random.rand(256, 256, 3).astype(np.float32)
        >>> pil_image = numpy_to_pil_image(img_array)[0] # Take the first and the only element
        >>> isinstance(pil_image, Image.Image)
        True
        
        >>> # Batch of images (NCHW)
        >>> batch_array = np.random.randint(0, 256, (4, 3, 256, 256), dtype=np.uint8)
        >>> pil_images = numpy_to_pil_image(batch_array)
        >>> len(pil_images)
        4
    """
    if array.ndim == 3:
        array = array[np.newaxis, ...]
    
    array = normalize_to_uint8(array)
    
    # NCHW -> NHWC if channel dim detected
    if array.shape[1] in (1, 3, 4) and array.shape[1] < array.shape[2]:
        array = array.transpose(0, 2, 3, 1)
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    result = [Image.fromarray(img) for img in array]
    return result


def tensor_list_to_pil_image(tensor_list: List[torch.Tensor]) -> List[Image.Image]:
    """
    Convert a list of torch Tensors to PIL Images.
    
    This function handles tensors with potentially different shapes by processing
    them individually when necessary, or batch-processing when all shapes match.
    
    Args:
        tensor_list: List of image tensors, each of shape (C, H, W) or (1, C, H, W).
            Each tensor can have different H, W dimensions.
            Supported value ranges: [0, 1] or [-1, 1].
    
    Returns:
        List[Image.Image]: List of PIL Images, one per input tensor.
    
    Note:
        - Tensors with shape (1, C, H, W) are automatically squeezed to (C, H, W).
        - If all tensors have the same shape, they are stacked and batch-processed
          for efficiency.
        - If tensors have different shapes, they are processed individually.
    
    Example:
        >>> # Same shape (batch-processed)
        >>> tensors = [torch.rand(3, 256, 256) for _ in range(4)]
        >>> pil_images = tensor_list_to_pil_image(tensors)
        >>> len(pil_images)
        4
        
        >>> # Different shapes (processed individually)
        >>> tensors = [torch.rand(3, 256, 256), torch.rand(3, 512, 512)]
        >>> pil_images = tensor_list_to_pil_image(tensors)
        >>> len(pil_images)
        2
    """
    if not tensor_list:
        return []
    
    # Squeeze batch dim if present
    squeezed = [t.squeeze(0) if t.ndim == 4 and t.shape[0] == 1 else t for t in tensor_list]
    
    # Uniform shape -> batch process
    if all(t.shape == squeezed[0].shape for t in squeezed):
        return tensor_to_pil_image(torch.stack(squeezed, dim=0))
    
    # Variable shape -> process individually (returns single Image for each 3D tensor)
    return [tensor_to_pil_image(t)[0] for t in squeezed]


def numpy_list_to_pil_image(numpy_list: List[np.ndarray]) -> List[Image.Image]:
    """
    Convert a list of NumPy arrays to PIL Images.
    
    This function handles arrays with potentially different shapes by processing
    them individually when necessary, or batch-processing when all shapes match.
    
    Args:
        numpy_list: List of image arrays, each of shape (H, W, C) or (C, H, W).
            Each array can have different H, W dimensions.
            Supported value ranges: [0, 255], [0, 1], or [-1, 1].
    
    Returns:
        List[Image.Image]: List of PIL Images, one per input array.
    
    Note:
        - Arrays with shape (1, H, W, C) are automatically squeezed.
        - If all arrays have the same shape, they are stacked and batch-processed
          for efficiency.
        - If arrays have different shapes, they are processed individually.
    
    Example:
        >>> # Same shape (batch-processed)
        >>> arrays = [np.random.rand(256, 256, 3) for _ in range(4)]
        >>> pil_images = numpy_list_to_pil_image(arrays)
        >>> len(pil_images)
        4
        
        >>> # Different shapes (processed individually)
        >>> arrays = [np.random.rand(256, 256, 3), np.random.rand(512, 512, 3)]
        >>> pil_images = numpy_list_to_pil_image(arrays)
        >>> len(pil_images)
        2
    """
    if not numpy_list:
        return []
    
    # Squeeze batch dim if present
    squeezed = [arr.squeeze(0) if arr.ndim == 4 and arr.shape[0] == 1 else arr for arr in numpy_list]
    
    # Uniform shape -> batch process
    if all(arr.shape == squeezed[0].shape for arr in squeezed):
        return numpy_to_pil_image(np.stack(squeezed, axis=0))
    
    # Variable shape -> process individually
    return [numpy_to_pil_image(arr)[0] for arr in squeezed]


# ----------------------------------- PIL -> Tensor/NumPy/Base64 --------------------------------------

def pil_image_to_tensor(
    images: Union[Image.Image, List[Image.Image]]
) -> torch.Tensor:
    """
    Convert PIL Image(s) to torch Tensor.
    
    Args:
        images: Single PIL Image or List of PIL Images.
            All images should have the same dimensions for batch processing.
    
    Returns:
        torch.Tensor: Image tensor with values in [0, 1].
            - If single Image: Shape (1, C, H, W)
            - If List of PIL Images: Shape (N, C, H, W)
    
    Raises:
        ValueError: If images is empty.
    
    Note:
        - Grayscale images are converted to RGB by duplicating channels.
        - RGBA images have their alpha channel discarded.
    
    Example:
        >>> # Single image
        >>> img = Image.new('RGB', (256, 256))
        >>> tensor = pil_image_to_tensor(img)
        >>> tensor.shape
        torch.Size([1, 3, 256, 256])
        
        >>> # Multiple images
        >>> images = [Image.new('RGB', (256, 256)) for _ in range(4)]
        >>> tensor = pil_image_to_tensor(images)
        >>> tensor.shape
        torch.Size([4, 3, 256, 256])
    """
    if isinstance(images, Image.Image):
        images = [images]
    
    if not images:
        raise ValueError("Empty image list")
    
    tensors = []
    for img in images:
        img_array = np.array(img).astype(np.float32) / 255.0
        if img_array.ndim == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        tensors.append(torch.from_numpy(img_array).permute(2, 0, 1))  # HWC -> CHW
    
    return torch.stack(tensors, dim=0)


def pil_image_to_numpy(
    images: Union[Image.Image, List[Image.Image]]
) -> np.ndarray:
    """
    Convert PIL Image(s) to NumPy array.
    
    Args:
        images: Single PIL Image or List of PIL Images.
            All images should have the same dimensions for batch processing.
    
    Returns:
        np.ndarray: Image array with uint8 dtype.
            - If single Image: Shape (1, H, W, C)
            - If List of PIL Images: Shape (N, H, W, C)
    
    Raises:
        ValueError: If images is empty.
    
    Example:
        >>> # Single image
        >>> img = Image.new('RGB', (256, 256))
        >>> array = pil_image_to_numpy(img)
        >>> array.shape
        (1, 256, 256, 3)
        
        >>> # Multiple images
        >>> images = [Image.new('RGB', (256, 256)) for _ in range(4)]
        >>> array = pil_image_to_numpy(images)
        >>> array.shape
        (4, 256, 256, 3)
    """
    if isinstance(images, Image.Image):
        images = [images]
    
    if not images:
        raise ValueError("Empty image list")
    
    return np.stack([np.array(img.convert('RGB')) for img in images], axis=0)


def pil_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert a PIL Image to a base64-encoded string.
    
    Args:
        image: PIL Image object.
        format: Image format, e.g., "JPEG", "PNG".
    
    Returns:
        str: Base64-encoded data URL string.
    
    Example:
        >>> img = Image.new('RGB', (64, 64), color='red')
        >>> b64 = pil_image_to_base64(img)
        >>> b64.startswith('data:image/jpeg;base64,')
        True
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


# ----------------------------------- Standardization --------------------------------------
def standardize_image_batch(
    images: Union[ImageSingle, ImageBatch],
    output_type: Literal['pil', 'np', 'pt'] = 'pil',
) -> ImageBatch:
    """
    Standardize input image(s) (single or batch) to the desired output format.
    
    Supports automatic promotion of single images to batch format and preserves
    input structure: stacked input -> stacked output, list input -> list output.
    
    Args:
        images: Input image(s) in any supported format:
            - Single: PIL.Image, torch.Tensor (C,H,W), np.ndarray (H,W,C)
            - Batch: List[PIL.Image], torch.Tensor (N,C,H,W), np.ndarray (N,H,W,C)
            - List[torch.Tensor] or List[np.ndarray] (variable shapes allowed)
        output_type: Target format.
            - 'pil': List[PIL.Image]
            - 'np': np.ndarray (N,H,W,C) uint8 if stacked, else List[np.ndarray]
            - 'pt': torch.Tensor (N,C,H,W) float32 [0,1] if stacked, else List[torch.Tensor]
    
    Returns:
        ImageBatch: Standardized batch in the requested format.
    
    Raises:
        ValueError: If input type is unsupported.
    
    Example:
        >>> # Single image -> batch
        >>> img = Image.new('RGB', (64, 64))
        >>> batch = standardize_image_batch(img, 'pt')
        >>> batch.shape
        torch.Size([1, 3, 64, 64])
        
        >>> # Tensor batch -> PIL
        >>> tensor = torch.rand(4, 3, 256, 256)
        >>> pil_images = standardize_image_batch(tensor, 'pil')
        >>> len(pil_images)
        4
        
        >>> # Variable-shape list preserved as list
        >>> tensors = [torch.rand(3, 64, 64), torch.rand(3, 128, 128)]
        >>> arrays = standardize_image_batch(tensors, 'np')
        >>> isinstance(arrays, list)
        True
    """
    # Single image -> list
    if isinstance(images, Image.Image):
        images = [images]
    # Single tensor/array -> batch
    if isinstance(images, (torch.Tensor, np.ndarray)) and images.ndim == 3:
        images = images.unsqueeze(0) if isinstance(images, torch.Tensor) else images[np.newaxis, ...]
    
    # Tensor (N, C, H, W)
    if isinstance(images, torch.Tensor):
        if output_type == 'pil':
            return tensor_to_pil_image(images)
        elif output_type == 'np':
            return normalize_to_uint8(images).cpu().numpy().transpose(0, 2, 3, 1)
        return images  # pt
    # NumPy array (N, H, W, C)
    elif isinstance(images, np.ndarray):
        if output_type == 'pil':
            return numpy_to_pil_image(images)
        elif output_type == 'pt':
            # NHWC -> NCHW, [0,255] -> [0,1]
            arr = images.astype(np.float32) / 255.0 if images.max() > 1.0 else images
            return torch.from_numpy(arr.transpose(0, 3, 1, 2))
        return images  # np

    # List of images
    elif isinstance(images, list):
        # List[torch.Tensor]
        if isinstance(images[0], torch.Tensor):
            if output_type == 'pil':
                return tensor_list_to_pil_image(images)
            elif output_type == 'np':
                return [normalize_to_uint8(t).cpu().numpy().transpose(1, 2, 0) for t in images]
            return images  # pt
        # List[np.ndarray]
        elif isinstance(images[0], np.ndarray):
            if output_type == 'pil':
                return numpy_list_to_pil_image(images)
            elif output_type == 'pt':
                return [
                    torch.from_numpy(
                        arr.astype(np.float32) / 255.0 if arr.max() > 1.0 else arr.astype(np.float32)
                    ).permute(2, 0, 1)
                    for arr in images
                ]
            return images  # np
        # List[PIL.Image]
        elif isinstance(images[0], Image.Image):
            if output_type == 'np':
                return pil_image_to_numpy(images)
            elif output_type == 'pt':
                return pil_image_to_tensor(images)
            return images  # pil

    raise ValueError(f'Unsupported image input type: {type(images)}')