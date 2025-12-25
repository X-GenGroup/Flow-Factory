# src/flow_factory/data_utils/dataset.py
import os
import inspect
import hashlib

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from PIL import Image
from collections import defaultdict
from typing import Optional, Dict, Any, Callable, List, Protocol, Union
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

class TextEncodeCallable(Protocol):
    def __call__(self, prompt: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]:
        ...

class ImageEncodeCallable(Protocol):
    def __call__(self, image: Union[Image.Image, List[Image.Image]], **kwargs: Any) -> Dict[str, Any]:
        ...

class VideoEncodeCallable(Protocol):
    def __call__(self, video: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]:
        """
        Args:
            video: Path(s) to video file(s)
        Returns:
            Dict with encoded video tensors, typically:
            - 'video': torch.Tensor of shape (T, C, H, W) or (C, T, H, W)
            - 'num_frames': int
        """
        ...

class PreprocessCallable(Protocol):
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]],
        image: Optional[Union[Image.Image, List[Image.Image]]],
        video: Optional[Union[str, List[str]]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        ...

class GeneralDataset(Dataset):
    @staticmethod
    def check_exists(dataset_dir: str, split: str) -> bool:
        dataset_dir = os.path.expanduser(dataset_dir)
        jsonl_path = os.path.join(dataset_dir, f"{split}.jsonl")
        txt_path = os.path.join(dataset_dir, f"{split}.txt")
        return os.path.exists(jsonl_path) or os.path.exists(txt_path)

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        cache_dir="~/.cache/flow_factory/datasets",
        enable_preprocess=True,
        force_reprocess=False,
        preprocessing_batch_size=16,
        max_dataset_size: Optional[int] = None,
        preprocess_func: Optional[PreprocessCallable] = None,
        **kwargs
    ):
        super().__init__()
        self.data_root = os.path.expanduser(dataset_dir)
        cache_dir = os.path.expanduser(cache_dir)
        
        # Detect file format (jsonl priority, then txt)
        jsonl_path = os.path.join(self.data_root, f"{split}.jsonl")
        txt_path = os.path.join(self.data_root, f"{split}.txt")
        
        if os.path.exists(jsonl_path):
            raw_dataset = load_dataset("json", data_files=jsonl_path, split="train")
            self.image_dir = os.path.join(self.data_root, "images")
            self.video_dir = os.path.join(self.data_root, "videos")
        elif os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            raw_dataset = HFDataset.from_dict({"prompt": prompts})
            self.image_dir = None
            self.video_dir = None
            logger.info(f"Loaded {len(prompts)} prompts from {txt_path}")
        else:
            raise FileNotFoundError(f"Could not find {jsonl_path} or {txt_path}")
        
        if max_dataset_size is not None and len(raw_dataset) > max_dataset_size:
            raw_dataset = raw_dataset.select(range(max_dataset_size))
            logger.info(f"Dataset size limited to {max_dataset_size} samples.")
    
        if enable_preprocess:
            self._preprocess_func = preprocess_func
            
            os.makedirs(cache_dir, exist_ok=True)
            funcs_hash = _compute_encode_funcs_hash(preprocess_func)
            fingerprint = (
                f"cache_{os.path.basename(self.data_root)}_{split}_"
                f"cutoff{max_dataset_size if max_dataset_size else 'full'}_"
                f"{funcs_hash}"
            )
            
            self.processed_dataset = raw_dataset.map(
                self._preprocess_batch,
                batched=True,
                batch_size=preprocessing_batch_size,
                fn_kwargs={
                    "image_dir": self.image_dir,
                    "video_dir": self.video_dir,
                },
                remove_columns=raw_dataset.column_names,
                new_fingerprint=fingerprint,
                desc="Pre-processing dataset",
                load_from_cache_file=not force_reprocess,
            )
            
            try:
                self.processed_dataset.set_format(type="torch", columns=self.processed_dataset.column_names)
            except Exception:
                pass

        else:
            self._text_encode_func = None
            self._image_encode_func = None
            self._video_encode_func = None
            self.processed_dataset = raw_dataset

    def _preprocess_batch(
        self,
        batch: Dict[str, Any],
        image_dir: Optional[str],
        video_dir: Optional[str],
    ) -> Dict[str, Any]:
        assert self._preprocess_func is not None, "Preprocess function must be provided for preprocessing."
    
        
        # 1. Prepare prompt inputs
        prompt = batch["prompt"]
        negative_prompt = batch.get("negative_prompt", None)
        prompt_args = {'prompt': prompt} if negative_prompt is None else {'prompt': prompt, 'negative_prompt': negative_prompt}
        
        # 2. Prepare image inputs (only when image_dir exists and batch has images)
        if 'image' in batch:
            # Rename 'image' key to 'images' for consistency
            batch['images'] = batch.pop('image')

        image_args : Dict[
            str,
            Optional[List[List[Image.Image]]]
        ] = {'images': []}
        if image_dir is not None and "images" in batch:
            img_paths_list = batch["images"]
            image_args['images'] = []
            for img_paths in img_paths_list:
                if not img_paths:
                    # img_paths is [] or None
                    image_args['images'].append([])
                else:
                    if isinstance(img_paths, str):
                        img_paths = [img_paths]
                    
                    image_args['images'].append([
                        Image.open(os.path.join(image_dir, img_path)).convert("RGB") 
                        for img_path in img_paths
                    ])
        else:
            image_args['images'] = None


        # 3. Prepare video inputs (only when video_dir exists and batch has videos)
        if 'video' in batch:
            # Rename 'video' key to 'videos' for consistency
            batch['videos'] = batch.pop('video')

        video_args : Dict[
            str,
            Optional[List[List[str]]]
        ] = {'videos': []}
        if video_dir is not None and "videos" in batch:
            video_paths_list = batch["videos"]
            video_args['videos'] = []
            for video_paths in video_paths_list:
                if not video_paths:
                    # video_paths is [] or None
                    video_args['videos'].append([])
                else:
                    if isinstance(video_paths, str):
                        video_paths = [video_paths]
                    
                    video_args['videos'].append([
                        os.path.join(video_dir, video_path) 
                        for video_path in video_paths
                    ])
        else:
            video_args['videos'] = None

        # 4. Merge results
        input_args = {**prompt_args, **image_args, **video_args}
        preprocess_res = self._preprocess_func(**input_args)

        # Warn if there are overlapping keys
        # Latter keys override former keys if any overlap
        key_intersection = set(batch.keys()).intersection(set(preprocess_res.keys()))
        if key_intersection:
            logger.warning(
                f"Preprocess function returned keys that overlap with original batch: {key_intersection}. "
                f"Latter keys will override former keys."
            )

        final_res = {}
        for k, v in preprocess_res.items():
            if isinstance(v, torch.Tensor):
                # Case A: Dense Batch Tensor (e.g. Qwen prompt embeddings)
                # Move entire batch to CPU first (faster than moving slices), then unbind
                v_cpu = v.cpu()
                final_res[k] = list(torch.unbind(v_cpu, dim=0))
            elif isinstance(v, list):
                # Case B: Ragged List (e.g. Flux image latents of varying sizes)
                # Check contents and move tensors to CPU if found
                final_res[k] = [
                    x.cpu() if isinstance(x, torch.Tensor) else x 
                    for x in v
                ]
            else:
                # Case C: Other types (None, int, etc)
                final_res[k] = v

        return {**batch, **final_res}

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        collated_batch = {}
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]
            
            # Check if values are tensors
            if isinstance(values[0], torch.Tensor):
                # Check if all tensors have the same shape
                shapes = [v.shape for v in values]
                if all(s == shapes[0] for s in shapes):
                    collated_batch[key] = torch.stack(values, dim=0)
                else:
                    # Shapes differ (e.g., different number of conditions), keep as List[Tensor]
                    collated_batch[key] = values
            else:
                # For int, str, list and etc...
                collated_batch[key] = values

        return collated_batch


def _compute_function_hash(func: Optional[Callable], digits: int = 16) -> str:
    """
    Compute a stable hash value for a callable function to use in cache fingerprints.
    
    Strategy (fallback chain):
    1. Use function source code (most accurate)
    2. Fall back to module path + function name (for compatibility)
    3. Final fallback to object ID (unstable but always works)
    
    Args:
        func: The callable to compute hash for, can be None
    
    Returns:
        A 16-character hexadecimal hash string
    
    Examples:
        >>> def my_func(x): return x * 2
        >>> hash1 = _compute_function_hash(my_func)
        >>> hash2 = _compute_function_hash(None)
        >>> hash1 != hash2
        True
    """
    _MAX_DIGITS = 32
    digits = min(digits, _MAX_DIGITS)
    if func is None:
        return "none" * 4  # "nonenonenoneone" - stable identifier for None
    
    try:
        # Method 1: Get function source code (most reliable)
        source = inspect.getsource(func)
        # Remove whitespace differences to avoid spurious cache misses
        source = "".join(source.split())
        return hashlib.md5(source.encode()).hexdigest()[:digits]
    except (TypeError, OSError):
        # Method 2: Use module path + function name (fallback)
        try:
            module = inspect.getmodule(func)
            module_name = module.__name__ if module else "unknown"
            func_name = getattr(func, '__qualname__', getattr(func, '__name__', 'anonymous'))
            signature = f"{module_name}.{func_name}"
            return hashlib.md5(signature.encode()).hexdigest()[:digits]
        except:
            # Method 3: Final fallback - use object ID (not stable across runs but prevents crashes)
            logger.warning(f"Could not compute stable hash for {func}, using id() fallback")
            return hashlib.md5(str(id(func)).encode()).hexdigest()[:digits]


def _compute_encode_funcs_hash(*funcs: Optional[Callable], digits: int = 16) -> str:
    """
    Compute a joint hash value for multiple encoding functions.
    
    This ensures that cache is invalidated when any preprocessing logic changes,
    while allowing cache reuse when logic remains the same.
    
    Args:
        *funcs: Variable number of callables (encoding functions)
                Can include None values which will be handled gracefully
    
    Returns:
        A 16-character hexadecimal hash string representing the joint hash
    
    Examples:
        >>> hash1 = _compute_encode_funcs_hash(text_enc, image_enc, None)
        >>> hash2 = _compute_encode_funcs_hash(text_enc, image_enc, video_enc)
        >>> hash1 != hash2  # Different because third function changed
        True
        
        >>> # Same functions produce same hash
        >>> hash3 = _compute_encode_funcs_hash(text_enc, image_enc, None)
        >>> hash1 == hash3
        True
    """
    _MAX_DIGITS = 32
    digits = min(digits, _MAX_DIGITS)
    # Compute individual hashes for each function
    individual_hashes = [_compute_function_hash(func) for func in funcs]
    
    # Combine hashes with labels for clarity in debugging
    combined_parts = [f"func{i}:{hash_val}" for i, hash_val in enumerate(individual_hashes)]
    combined = "|".join(combined_parts)
    
    # Generate final joint hash
    joint_hash = hashlib.md5(combined.encode()).hexdigest()[:digits]
    
    return joint_hash