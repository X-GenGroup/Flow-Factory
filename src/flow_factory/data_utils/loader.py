# src/flow_factory/data/loader.py
from typing import Union, Tuple, Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from .dataset import GeneralDataset
from .sampler import DistributedKRepeatSampler
from ..hparams import *
from ..data_utils.dataset import TextEncodeCallable, ImageEncodeCallable

def get_dataloader(
    data_args : DataArguments,
    training_args : TrainingArguments,
    text_encode_func : Optional[TextEncodeCallable] = None,
    image_encode_func : Optional[ImageEncodeCallable] = None,
    **kwargs,
) -> Tuple[DataLoader, Union[DataLoader, None]]:
    """
    Factory to create the DDP/FSDP compatible DataLoader.
    """

    # 1. Initialize Dataset (Now handles tokenization internally)
    dataset = GeneralDataset(
        dataset_dir=data_args.dataset,
        split="train",
        enable_preprocess=data_args.enable_preprocess,
        preprocessing_batch_size=data_args.preprocessing_batch_size,
        text_encode_func=text_encode_func,
        image_encode_func=image_encode_func
    )
    if GeneralDataset.check_exists(data_args.dataset, "test"):
        test_dataset = GeneralDataset(
            dataset_dir=data_args.dataset,
            split="test",
            enable_preprocess=data_args.enable_preprocess,
            preprocessing_batch_size=data_args.preprocessing_batch_size,
            text_encode_func=text_encode_func,
            image_encode_func=image_encode_func
        )

    # 2. Determine Distributed Context
    if dist.is_initialized():
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
    else:
        num_replicas = 1
        rank = 0

    # 3. Initialize GRPO Sampler (Logic Unchanged)
    sampler = DistributedKRepeatSampler(
        dataset=dataset,
        batch_size=training_args.per_device_batch_size,
        group_size=training_args.group_size,
        unique_sample_num=training_args.unique_sample_num_per_epoch,
        num_replicas=num_replicas,
        rank=rank,
        seed=training_args.seed
    )
    
    # 4. Build DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=data_args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=GeneralDataset.collate_fn,
    )
    if GeneralDataset.check_exists(data_args.dataset, "test"):
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=training_args.eval_args.per_device_batch_size,
            shuffle=False,
            num_workers=data_args.dataloader_num_workers,
            collate_fn=GeneralDataset.collate_fn,
        )
    else:
        test_dataloader = None

    return dataloader, test_dataloader