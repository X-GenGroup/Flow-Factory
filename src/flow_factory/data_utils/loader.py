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

# src/flow_factory/data_utils/loader.py
import os
import shutil
from typing import Union, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import concatenate_datasets, load_from_disk
from .dataset import GeneralDataset
from .sampler import DistributedKRepeatSampler
from ..hparams import Arguments
from ..data_utils.dataset import PreprocessCallable
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _create_or_load_dataset(
    split: str,
    accelerator: Accelerator,
    base_kwargs: dict,
    enable_distributed: bool,
) -> GeneralDataset:
    """
    Create or load preprocessed dataset with optional distributed sharding.
    
    Workflow:
        1. Compute cache path without creating dataset
        2. If merged cache exists → load directly (fast path)
        3. Otherwise:
           a. Single-process: preprocess directly
           b. Multi-process: shard → preprocess → merge → load
    
    Args:
        split: Dataset split ('train', 'test', etc.)
        accelerator: Accelerator for distributed coordination
        base_kwargs: Base arguments for GeneralDataset
        enable_distributed: Whether to use distributed preprocessing
        
    Returns:
        GeneralDataset instance (fully preprocessed and ready for training)
    """
    # Setup shard parameters
    kwargs = base_kwargs.copy()
    if enable_distributed:
        kwargs['num_shards'] = accelerator.num_processes
        kwargs['shard_index'] = accelerator.process_index
    else:
        kwargs['num_shards'] = None
        kwargs['shard_index'] = None
    
    # Compute cache path WITHOUT creating dataset (avoids unnecessary preprocessing)
    merged_cache_path = GeneralDataset.compute_cache_path(
        dataset_dir=kwargs['dataset_dir'],
        split=split,
        cache_dir=kwargs.get('cache_dir', '~/.cache/flow_factory/datasets'),
        max_dataset_size=kwargs.get('max_dataset_size'),
        preprocess_func=kwargs.get('preprocess_func'),
        preprocess_kwargs=kwargs.get('preprocess_kwargs'),
        extra_hash_strs=kwargs.get('extra_hash_strs', []),
    )
    
    # Fast path: merged cache already exists
    if os.path.exists(merged_cache_path) and not base_kwargs.get('force_reprocess', False):
        if accelerator.is_main_process:
            logger.info(f"Loading {split} dataset from merged cache: {merged_cache_path}")
        return GeneralDataset.load_merged(merged_cache_path)
    
    # Single-process path: direct preprocessing
    if not enable_distributed:
        logger.info(f"Preprocessing {split} dataset (single process)")
        return GeneralDataset(split=split, **kwargs)
    
    # Distributed path: shard → merge → load
    logger.info(f"Preprocessing {split} dataset shard {kwargs['shard_index']}/{kwargs['num_shards']}")
    dataset = GeneralDataset(split=split, **kwargs)
    
    # Step 1: Save shard to disk
    shard_path = os.path.join(
        dataset.cache_dir,
        f"{os.path.basename(merged_cache_path)}_shard{kwargs['shard_index']}"
    )
    dataset.save_shard(shard_path)
    accelerator.wait_for_everyone()
    
    # Step 2: Main process merges all shards
    if accelerator.is_main_process:
        logger.info(f"Merging {kwargs['num_shards']} shards for {split} split")
        shard_paths = []
        shards = []
        for i in range(kwargs['num_shards']):
            shard_path_i = os.path.join(
                dataset.cache_dir, 
                f"{os.path.basename(merged_cache_path)}_shard{i}"
            )
            shard_paths.append(shard_path_i)
            shards.append(load_from_disk(shard_path_i))
        
        merged = concatenate_datasets(shards)
        merged.save_to_disk(merged_cache_path)
        logger.info(f"Merged {split} dataset saved to {merged_cache_path}")
        
        # Step 3: Clean up shard caches
        for shard_path_i in shard_paths:
            if os.path.exists(shard_path_i):
                shutil.rmtree(shard_path_i)
        logger.info(f"Cleaned up {len(shard_paths)} shard caches")
    
    accelerator.wait_for_everyone()
    
    # Step 4: All processes load merged dataset
    return GeneralDataset.load_merged(merged_cache_path)

def get_dataloader(
    config: Arguments,
    accelerator: Accelerator,
    preprocess_func: Optional[PreprocessCallable] = None,
    **kwargs,
) -> Tuple[DataLoader, Union[DataLoader, None]]:
    """
    Factory to create DDP/FSDP compatible DataLoader with distributed preprocessing.
    
    Features:
        - Automatic distributed preprocessing across multiple GPUs
        - Intelligent caching (reuses preprocessed data on subsequent runs)
        - Supports both train and test splits
        - Custom sampler for GRPO-style grouped sampling
    
    Args:
        config: Configuration object containing all arguments
        accelerator: Accelerator for distributed training
        preprocess_func: Function to preprocess batches
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
        test_dataloader is None if test split doesn't exist
    """
    data_args = config.data_args
    training_args = config.training_args
    eval_args = config.eval_args

    # Determine if distributed preprocessing is needed
    enable_distributed = accelerator.num_processes > 1 and data_args.enable_preprocess

    # Common dataset kwargs
    base_kwargs = {
        "preprocess_func": preprocess_func,
        "preprocess_kwargs": filter_kwargs(preprocess_func, **data_args) if preprocess_func else None, # Preprocess kwargs
        'extra_hash_strs': [config.model_args.model_type, config.model_args.model_name_or_path], # Use model info to differentiate caches
    }
    base_kwargs.update(filter_kwargs(GeneralDataset.__init__, **data_args))
    base_kwargs['force_reprocess'] = data_args.force_reprocess

    # === CREATE/LOAD TRAIN DATASET ===
    train_preprocess_kwargs = base_kwargs.get('preprocess_kwargs', {}).copy()
    train_preprocess_kwargs.update(
        {
            'is_train': True,
            **training_args,
        }
    )
    train_preprocess_kwargs = filter_kwargs(preprocess_func, **train_preprocess_kwargs)
    dataset = _create_or_load_dataset(
        split="train",
        accelerator=accelerator,
        base_kwargs={**base_kwargs, 'preprocess_kwargs': train_preprocess_kwargs},
        enable_distributed=enable_distributed,
    )

    # === CREATE TRAIN DATALOADER ===
    sampler = DistributedKRepeatSampler(
        dataset=dataset,
        batch_size=training_args.per_device_batch_size,
        group_size=training_args.group_size,
        unique_sample_num=training_args.unique_sample_num_per_epoch,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=training_args.seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=data_args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=GeneralDataset.collate_fn,
    )

    # === CREATE/LOAD TEST DATASET ===
    test_dataloader = None
    if GeneralDataset.check_exists(data_args.dataset, "test"):
        test_preprocess_kwargs = base_kwargs.get('preprocess_kwargs', {}).copy()
        test_preprocess_kwargs.update(
            {
                'is_train': False,
                **eval_args,
            }
        )
        test_preprocess_kwargs = filter_kwargs(preprocess_func, **test_preprocess_kwargs)
        test_dataset = _create_or_load_dataset(
            split="test",
            accelerator=accelerator,
            base_kwargs={**base_kwargs, 'preprocess_kwargs': test_preprocess_kwargs},
            enable_distributed=enable_distributed,
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_args.per_device_batch_size,
            shuffle=False,
            num_workers=data_args.dataloader_num_workers,
            collate_fn=GeneralDataset.collate_fn,
        )

    return dataloader, test_dataloader