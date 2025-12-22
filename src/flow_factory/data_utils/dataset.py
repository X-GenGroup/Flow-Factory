# src/flow_factory/data_utils/dataset.py
import os
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
        preprocessing_batch_size=16,
        max_dataset_size: Optional[int] = None,
        text_encode_func: Optional[TextEncodeCallable] = None,
        image_encode_func: Optional[ImageEncodeCallable] = None,
        video_encode_func: Optional[VideoEncodeCallable] = None,
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
            self._text_encode_func = text_encode_func
            self._image_encode_func = image_encode_func
            self._video_encode_func = video_encode_func
            
            os.makedirs(cache_dir, exist_ok=True)
            fingerprint = (
                f"{os.path.basename(self.data_root)}_{split}_"
                f"cutoff{max_dataset_size if max_dataset_size else 'full'}_v1"
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
                load_from_cache_file=True,
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
        
        prompt = batch["prompt"]
        negative_prompt = batch.get("negative_prompt", None)
        
        # 1. Process Text
        assert self._text_encode_func is not None, "Text encode function must be provided to process prompt."
        prompt_args = {'prompt': prompt} if negative_prompt is None else {'prompt': prompt, 'negative_prompt': negative_prompt}
        prompt_res = self._text_encode_func(**prompt_args)
        
        # 2. Process Images (only when image_dir exists and batch has images)
        collated_image_res = defaultdict(list)
        if image_dir is not None and "images" in batch:
            img_paths_list = batch["images"]
            for img_paths in img_paths_list:
                image = []
                for img_path in img_paths:
                    with Image.open(os.path.join(image_dir, img_path)) as img:
                        image.append(img.convert("RGB"))
                if len(image) > 0:
                    assert self._image_encode_func is not None, "Image encode function must be provided to process image."
                    encoded_single_sample = self._image_encode_func(image)
                    for k, v in encoded_single_sample.items():
                        collated_image_res[k].append(v)

        # 3. Process Videos (only when video_dir exists and batch has videos)
        collated_video_res = defaultdict(list)
        if video_dir is not None and "videos" in batch:
            video_paths_list = batch["videos"]
            for video_paths in video_paths_list:
                video = []
                for video_path in video_paths:
                    video.append(os.path.join(video_dir, video_path))
                if len(video) > 0:
                    assert self._video_encode_func is not None, "Video encode function must be provided to process video."
                    encoded_single_sample = self._video_encode_func(video)
                    for k, v in encoded_single_sample.items():
                        collated_video_res[k].append(v)

        # 4. Merge results
        prompt_res = {
            k: (list(torch.unbind(v)) if isinstance(v, torch.Tensor) else v) 
            for k, v in prompt_res.items()
        }

        return {**batch, **prompt_res, **collated_image_res, **collated_video_res}

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to handle batching of pre-processed samples.
        """
        if not batch:
            return {}

        collated_batch = {}
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated_batch[key] = torch.stack(values)
                except:
                    collated_batch[key] = values
            else:
                collated_batch[key] = values

        return collated_batch