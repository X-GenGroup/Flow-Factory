# src/flow_factory/data_utils/loader.py
import os
from typing import Union, Tuple, Optional
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator
from .dataset import GeneralDataset
from .sampler import DistributedKRepeatSampler
from ..hparams import *
from ..data_utils.dataset import TextEncodeCallable, ImageEncodeCallable, VideoEncodeCallable, PreprocessCallable
from ..utils.base import filter_kwargs

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_dataloader(
    config : Arguments,
    accelerator : Accelerator,
    preprocess_func : Optional[PreprocessCallable] = None,
    **kwargs,
) -> Tuple[DataLoader, Union[DataLoader, None]]:
    """
    Factory to create the DDP/FSDP compatible DataLoader.
    """
    data_args = config.data_args
    training_args = config.training_args
    eval_args = config.eval_args

    # 1. Initialize Dataset (Now handles tokenization internally)
    dataset_init_kwargs = {
        "preprocess_func": preprocess_func,
    }
    dataset_init_kwargs.update(filter_kwargs(GeneralDataset.__init__, **data_args.to_dict()))
    # Only the main process handles preprocessing and caching
    dataset_init_kwargs['force_reprocess'] = data_args.force_reprocess and accelerator.is_main_process
    dataset = GeneralDataset(
        split="train",
        **dataset_init_kwargs,
    )
    if GeneralDataset.check_exists(data_args.dataset, "test"):
        test_dataset = GeneralDataset(
            split="test",
            **dataset_init_kwargs,
        )

    # 3. Initialize GRPO Sampler (Logic Unchanged)
    sampler = DistributedKRepeatSampler(
        dataset=dataset,
        batch_size=training_args.per_device_batch_size,
        group_size=training_args.group_size,
        unique_sample_num=training_args.unique_sample_num_per_epoch,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
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
            batch_size=eval_args.per_device_batch_size,
            shuffle=False,
            num_workers=data_args.dataloader_num_workers,
            collate_fn=GeneralDataset.collate_fn,
        )
    else:
        test_dataloader = None

    return dataloader, test_dataloader