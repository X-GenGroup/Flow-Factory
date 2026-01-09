# src/flow_factory/utils/image.py
"""
Image utility functions for converting between PIL Images, torch Tensors, and NumPy arrays.

Supported formats:
    - ImageList: List[PIL.Image] - List of images
    - ImageBatch: List[List[PIL.Image]] - Batch of image lists
    - torch.Tensor: (C, H, W) for single image, (N, C, H, W) for batch
    - np.ndarray: (H, W, C) for single image, (N, H, W, C) for batch

Value ranges:
    - [0, 255]: Standard uint8 format
    - [0, 1]: Normalized float format (PyTorch convention)
    - [-1, 1]: Normalized float format (diffusion models)
"""

import base64
from io import BytesIO
from typing import List, Union, Any

from PIL import Image
import torch
import numpy as np


# ----------------------------------- Type Aliases --------------------------------------

ImageList = List[Image.Image]
"""Type alias for a list of PIL Images."""

ImageBatch = List[List[Image.Image]]
"""Type alias for a batch of image lists."""


__all__ = [
    # Type aliases
    'ImageList',
    'ImageBatch',
    # Type checks
    'is_pil_image_list',
    'is_pil_image_batch_list',
    # Validation
    'is_valid_image',
    'is_valid_image_list',
    'is_valid_image_batch',
    'is_valid_image_batch_list',
    # Tensor/NumPy -> PIL
    'tensor_to_pil_image',
    'numpy_to_pil_image',
    'tensor_list_to_pil_image',
    'numpy_list_to_pil_image',
    # PIL -> Tensor/NumPy/Base64
    'pil_image_to_tensor',
    'pil_image_to_numpy',
    'pil_image_to_base64',
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


def is_pil_image_batch_list(image_batch_list: ImageBatch) -> bool:
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

def is_valid_image(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> bool:
    """
    Check if the input is a valid image type.
    
    Args:
        image: Input image in one of the supported formats.
    
    Returns:
        bool: True if valid image type:
            - PIL.Image: Valid PIL Image with positive dimensions
            - torch.Tensor: Shape (C, H, W) or (N, C, H, W) where C in {1, 3, 4}
            - np.ndarray: Shape (H, W, C) or (N, H, W, C) where C in {1, 3, 4}
    
    Example:
        >>> is_valid_image(Image.new('RGB', (64, 64)))
        True
        >>> is_valid_image(torch.rand(3, 256, 256))
        True
        >>> is_valid_image(np.random.rand(256, 256, 3))
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
            return b > 0 and c in (1, 3, 4) and h > 0 and w > 0
        return False
    
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            h, w, c = image.shape
            return h > 0 and w > 0 and c in (1, 3, 4)
        elif image.ndim == 4:
            b, h, w, c = image.shape
            return b > 0 and h > 0 and w > 0 and c in (1, 3, 4)
        return False
    
    return False


def is_valid_image_list(images: List[Union[Image.Image, torch.Tensor, np.ndarray]]) -> bool:
    """
    Check if the input is a valid list of images.
    
    Args:
        images: List of images to check.
    
    Returns:
        bool: True if valid image list:
            - Non-empty list
            - All elements are valid images
            - All elements are of the same type
    
    Example:
        >>> images = [torch.rand(3, 64, 64) for _ in range(4)]
        >>> is_valid_image_list(images)
        True
    """
    if not isinstance(images, list) or len(images) == 0:
        return False
    
    first_type = type(images[0])
    if not all(isinstance(img, first_type) for img in images):
        return False
    
    return all(is_valid_image(img) for img in images)


def is_valid_image_batch(
    images: Union[ImageList, List[torch.Tensor], List[np.ndarray], torch.Tensor, np.ndarray]
) -> bool:
    """
    Check if the input is a valid batch of images.
    
    Args:
        images: Input image batch.
    
    Returns:
        bool: True if valid image batch:
            - ImageList (List[PIL.Image])
            - List[torch.Tensor] where each tensor is (C, H, W)
            - List[np.ndarray] where each array is (H, W, C)
            - torch.Tensor with shape (N, C, H, W)
            - np.ndarray with shape (N, H, W, C)
    
    Example:
        >>> is_valid_image_batch(torch.rand(4, 3, 256, 256))
        True
        >>> is_valid_image_batch(np.random.rand(4, 256, 256, 3))
        True
    """
    if isinstance(images, list):
        return is_valid_image_list(images)
    
    if isinstance(images, torch.Tensor):
        if images.ndim != 4:
            return False
        b, c, h, w = images.shape
        return b > 0 and c in (1, 3, 4) and h > 0 and w > 0
    
    if isinstance(images, np.ndarray):
        if images.ndim != 4:
            return False
        b, h, w, c = images.shape
        return b > 0 and h > 0 and w > 0 and c in (1, 3, 4)
    
    return False


def is_valid_image_batch_list(image_batches: ImageBatch) -> bool:
    """
    Check if the input is a valid batch of image lists.
    
    Args:
        image_batches: Batch of image lists, e.g., [[img1, img2], [img3], [img4, img5, img6]].
    
    Returns:
        bool: True if valid:
            - Outer list is non-empty
            - Each inner element is either a valid image list or an empty list
    
    Note:
        Empty inner lists are allowed (some samples may have no images).
    
    Example:
        >>> batch = [[Image.new('RGB', (64, 64))], [], [Image.new('RGB', (64, 64)) for _ in range(3)]]
        >>> is_valid_image_batch_list(batch)
        True
    """
    if not isinstance(image_batches, list) or len(image_batches) == 0:
        return False
    
    for batch in image_batches:
        if not isinstance(batch, list):
            return False
        if len(batch) > 0 and not is_valid_image_list(batch):
            return False
    
    return True


# ----------------------------------- Normalization --------------------------------------

def _normalize_to_uint8(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
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

def tensor_to_pil_image(tensor: torch.Tensor) -> Union[Image.Image, ImageList]:
    """
    Convert a torch Tensor to PIL Image(s).
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (N, C, H, W).
            Supported value ranges:
                - [0, 1]: Standard normalized tensor format
                - [-1, 1]: Normalized tensor format (e.g., from diffusion models)
    
    Returns:
        - If input is 3D (C, H, W): Single PIL Image
        - If input is 4D (N, C, H, W): ImageList (List of N PIL Images)
    
    Raises:
        ValueError: If tensor is not 3D or 4D.
    
    Example:
        >>> # Single image
        >>> img_tensor = torch.rand(3, 256, 256)
        >>> pil_image = tensor_to_pil_image(img_tensor)
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
    tensor = _normalize_to_uint8(tensor).cpu().numpy()
    tensor = tensor.transpose(0, 2, 3, 1)
    
    if tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    
    result = [Image.fromarray(img) for img in tensor]
    return result[0]


def numpy_to_pil_image(array: np.ndarray) -> Union[Image.Image, ImageList]:
    """
    Convert a NumPy array to PIL Image(s).
    
    Args:
        array: Image array of shape (H, W, C) / (C, H, W) or (N, H, W, C) / (N, C, H, W).
            Supported value ranges:
                - [0, 255]: Standard uint8 format
                - [0, 1]: Normalized float format
                - [-1, 1]: Normalized float format (e.g., from diffusion models)
    
    Returns:
        - If input is 3D: Single PIL Image
        - If input is 4D: ImageList (List of N PIL Images)
    
    Raises:
        ValueError: If array is not 3D or 4D.
    
    Note:
        Channel dimension detection: If the suspected channel dimension is in {1, 3, 4}
        and smaller than spatial dimensions, the array is assumed to be channel-first
        and will be transposed to channel-last.
    
    Example:
        >>> # Single image (HWC)
        >>> img_array = np.random.rand(256, 256, 3).astype(np.float32)
        >>> pil_image = numpy_to_pil_image(img_array)
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
    
    array = _normalize_to_uint8(array)
    
    # NCHW -> NHWC if channel dim detected
    if array.shape[1] in (1, 3, 4) and array.shape[1] < array.shape[2]:
        array = array.transpose(0, 2, 3, 1)
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    result = [Image.fromarray(img) for img in array]
    return result[0]


def tensor_list_to_pil_image(tensor_list: List[torch.Tensor]) -> ImageList:
    """
    Convert a list of torch Tensors to PIL Images.
    
    This function handles tensors with potentially different shapes by processing
    them individually when necessary, or batch-processing when all shapes match.
    
    Args:
        tensor_list: List of image tensors, each of shape (C, H, W) or (1, C, H, W).
            Each tensor can have different H, W dimensions.
            Supported value ranges: [0, 1] or [-1, 1].
    
    Returns:
        ImageList: List of PIL Images, one per input tensor.
    
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
    return [tensor_to_pil_image(t) for t in squeezed]


def numpy_list_to_pil_image(numpy_list: List[np.ndarray]) -> ImageList:
    """
    Convert a list of NumPy arrays to PIL Images.
    
    This function handles arrays with potentially different shapes by processing
    them individually when necessary, or batch-processing when all shapes match.
    
    Args:
        numpy_list: List of image arrays, each of shape (H, W, C) or (C, H, W).
            Each array can have different H, W dimensions.
            Supported value ranges: [0, 255], [0, 1], or [-1, 1].
    
    Returns:
        ImageList: List of PIL Images, one per input array.
    
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
    return [numpy_to_pil_image(arr) for arr in squeezed]


# ----------------------------------- PIL -> Tensor/NumPy/Base64 --------------------------------------

def pil_image_to_tensor(
    images: Union[Image.Image, ImageList]
) -> torch.Tensor:
    """
    Convert PIL Image(s) to torch Tensor.
    
    Args:
        images: Single PIL Image or ImageList.
            All images should have the same dimensions for batch processing.
    
    Returns:
        torch.Tensor: Image tensor with values in [0, 1].
            - If single Image: Shape (1, C, H, W)
            - If ImageList: Shape (N, C, H, W)
    
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
    images: Union[Image.Image, ImageList]
) -> np.ndarray:
    """
    Convert PIL Image(s) to NumPy array.
    
    Args:
        images: Single PIL Image or ImageList.
            All images should have the same dimensions for batch processing.
    
    Returns:
        np.ndarray: Image array with uint8 dtype.
            - If single Image: Shape (1, H, W, C)
            - If ImageList: Shape (N, H, W, C)
    
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