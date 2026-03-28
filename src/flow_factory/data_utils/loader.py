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
from typing import Union, Tuple, Optional, Literal
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import concatenate_datasets, load_from_disk
from .dataset import GeneralDataset
from .sampler_loader import get_data_sampler
from ..hparams import Arguments
from ..data_utils.dataset import PreprocessCallable
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=False)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _get_local_process_info(accelerator: Accelerator):
    """
    Get local_rank and local_world_size within the current node.
    Prefers environment variables set by torchrun / accelerate launch,
    falls back to accelerator attributes.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    # If LOCAL_WORLD_SIZE is not set but we have multiple processes, try to infer
    if local_world_size == 1 and accelerator.num_processes > 1:
        num_machines = int(os.environ.get("NUM_MACHINES", os.environ.get("NNODES", 1)))
        local_world_size = accelerator.num_processes // num_machines
    return local_rank, local_world_size


def _create_or_load_dataset(
    split: str,
    accelerator: Accelerator,
    base_kwargs: dict,
    enable_distributed: bool,
    preprocess_parallelism: Literal["global", "local"] = "global",
) -> GeneralDataset:
    """
    Create or load preprocessed dataset with optional distributed sharding.
    
    Workflow:
        1. Compute cache path without creating dataset
        2. If merged cache exists → load directly (fast path)
        3. Otherwise:
           a. Single-process: preprocess directly
           b. Multi-process: shard → preprocess → merge → load
    
    For 'local' parallelism, each node independently preprocesses and merges
    shards using only global barriers (no node-local process groups needed).
    
    Args:
        split: Dataset split ('train', 'test', etc.)
        accelerator: Accelerator for distributed coordination
        base_kwargs: Base arguments for GeneralDataset
        enable_distributed: Whether to use distributed preprocessing
        preprocess_parallelism: 'global' for cross-node parallelism (requires shared FS),
                                'local' for per-node parallelism (no shared FS required)
        
    Returns:
        GeneralDataset instance (fully preprocessed and ready for training)
    """
    # Setup shard parameters based on parallelism mode
    kwargs = base_kwargs.copy()
    if enable_distributed:
        if preprocess_parallelism == "local":
            # Local mode: each node's processes independently shard and preprocess
            local_rank, local_world_size = _get_local_process_info(accelerator)
            kwargs['num_shards'] = local_world_size
            kwargs['shard_index'] = local_rank
        else:
            # Global mode: all processes across nodes split the workload
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
        if accelerator.is_local_main_process:
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

    # Step 2: Merge shards and save to disk
    accelerator.wait_for_everyone() # Sync point: ensure all shards are written before merging
    if preprocess_parallelism == "local":
        # ---- Local parallelism using global barriers only ----
        local_rank, local_world_size = _get_local_process_info(accelerator)

        # local_rank == 0 on each node merges that node's shards
        if accelerator.is_local_main_process:
            logger.info(f"[Local] Merging {local_world_size} shards for {split} split on this node")
            shard_paths = []
            shards = []
            for i in range(local_world_size):
                shard_path_i = os.path.join(
                    dataset.cache_dir,
                    f"{os.path.basename(merged_cache_path)}_shard{i}"
                )
                shard_paths.append(shard_path_i)
                shards.append(load_from_disk(shard_path_i))

            merged = concatenate_datasets(shards)
            merged.save_to_disk(merged_cache_path)
            logger.info(f"[Local] Merged {split} dataset saved to {merged_cache_path}")

            # Clean up shard caches
            for shard_path_i in shard_paths:
                if os.path.exists(shard_path_i):
                    shutil.rmtree(shard_path_i)
            logger.info(f"[Local] Cleaned up {len(shard_paths)} shard caches")
    else:
        # ---- Global parallelism: cross-node sync and merge ----
        # Only global main process (rank 0) performs the merge to avoid redundant work and ensure consistency
        if accelerator.is_main_process:
            logger.info(f"[Global] Merging {kwargs['num_shards']} shards for {split} split")
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
            logger.info(f"[Global] Merged {split} dataset saved to {merged_cache_path}")

            # Step 3: Clean up shard caches
            for shard_path_i in shard_paths:
                if os.path.exists(shard_path_i):
                    shutil.rmtree(shard_path_i)
            logger.info(f"[Global] Cleaned up {len(shard_paths)} shard caches")

    # Global barrier: ensure merge is complete on all nodes before anyone loads
    accelerator.wait_for_everyone()

    # Final step: All processes load merged dataset
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
    preprocess_parallelism = getattr(data_args, 'preprocess_parallelism', 'local')

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
        preprocess_parallelism=preprocess_parallelism,
    )

    # === CREATE TRAIN DATALOADER ===
    sampler = get_data_sampler(
        dataset=dataset,
        config=config,
        accelerator=accelerator,
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
            preprocess_parallelism=preprocess_parallelism,
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_args.per_device_batch_size,
            shuffle=False,
            num_workers=data_args.dataloader_num_workers,
            collate_fn=GeneralDataset.collate_fn,
        )

    return dataloader, test_dataloader
