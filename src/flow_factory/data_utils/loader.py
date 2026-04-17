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
import json
import os
import shutil
from typing import Literal, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..data_utils.dataset import PreprocessCallable
from ..hparams import Arguments
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger
from .dataset import GeneralDataset
from .sampler_loader import get_data_sampler

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
    """Create or load preprocessed dataset with optional distributed sharding.

    Each rank writes its preprocessed Arrow shard exactly once via
    ``Dataset.map(cache_file_name=...)`` straight into the final cache directory.
    The consolidator (``local_main`` for ``"local"``, global rank 0 for ``"global"``,
    the lone process for single-process) then writes ``state.json`` and
    ``dataset_info.json`` referencing those per-rank Arrow files and atomically
    renames the build directory ``{merged_cache_path}.tmp`` to ``merged_cache_path``.
    No shard data is re-copied.

    Args:
        split: Dataset split (``"train"``, ``"test"``, ...).
        accelerator: Accelerator used for cross-rank synchronization.
        base_kwargs: Base arguments forwarded to ``GeneralDataset``.
        enable_distributed: ``True`` when more than one process needs to share work.
        preprocess_parallelism: ``"global"`` for cross-node parallelism (shared FS
            required); ``"local"`` for per-node parallelism (no shared FS required).

    Returns:
        Fully preprocessed ``GeneralDataset`` ready for training.
    """
    kwargs = base_kwargs.copy()

    if not kwargs.get("enable_preprocess", True):
        logger.info(
            f"Loading {split} dataset without preprocessing (enable_preprocess=False); "
            f"skipping consolidate pipeline"
        )
        return GeneralDataset(split=split, **kwargs)

    if enable_distributed:
        if preprocess_parallelism == "local":
            local_rank, local_world_size = _get_local_process_info(accelerator)
            kwargs["num_shards"] = local_world_size
            kwargs["shard_index"] = local_rank
        else:
            kwargs["num_shards"] = accelerator.num_processes
            kwargs["shard_index"] = accelerator.process_index
    else:
        kwargs["num_shards"] = 1
        kwargs["shard_index"] = 0

    merged_cache_path = GeneralDataset.compute_cache_path(
        dataset_dir=kwargs["dataset_dir"],
        split=split,
        cache_dir=kwargs.get("cache_dir", "~/.cache/flow_factory/datasets"),
        max_dataset_size=kwargs.get("max_dataset_size"),
        preprocess_func=kwargs.get("preprocess_func"),
        preprocess_kwargs=kwargs.get("preprocess_kwargs"),
        extra_hash_strs=kwargs.get("extra_hash_strs", []),
    )

    if os.path.exists(merged_cache_path) and not base_kwargs.get("force_reprocess", False):
        if accelerator.is_local_main_process:
            logger.info(f"Loading {split} dataset from merged cache: {merged_cache_path}")
        return GeneralDataset.load_merged(merged_cache_path)

    shard_idx = kwargs["shard_index"]
    num_shards = kwargs["num_shards"]
    merged_fp = os.path.basename(merged_cache_path)
    shard_fp = f"{merged_fp}_shard{shard_idx}of{num_shards - 1}"

    build_dir = merged_cache_path + ".tmp"
    sentinel = os.path.join(build_dir, "_build_meta.json")

    def _meta_matches() -> bool:
        if not os.path.isfile(sentinel):
            return False
        try:
            with open(sentinel) as f:
                return json.load(f).get("num_shards") == num_shards
        except (json.JSONDecodeError, OSError):
            # Sentinel was corrupted (e.g., previous run crashed mid-write).
            # Treat as stale so the orchestrator wipes and recreates the build dir,
            # matching the existing "missing -> return False -> wipe" semantics.
            return False

    # Single owner for build-dir prep + final consolidation: rank-0 globally for
    # "global" mode (all nodes share the FS) and per-node local main for "local"
    # mode. Using is_orchestrator in both places prevents multi-node races on
    # shutil.rmtree / sentinel writes when the FS is shared.
    if not enable_distributed:
        is_orchestrator = True
    elif preprocess_parallelism == "local":
        is_orchestrator = accelerator.is_local_main_process
    else:
        is_orchestrator = accelerator.is_main_process

    # 1. Orchestrator prepares (or wipes-then-prepares) the build dir. The wipe only
    #    fires when num_shards changed since the last attempt; otherwise per-rank
    #    Arrow files written before a previous crash are reused via HF's
    #    load_from_cache_file path.
    if is_orchestrator:
        if os.path.exists(build_dir) and not _meta_matches():
            logger.warning(f"Wiping stale build dir {build_dir} (num_shards changed)")
            shutil.rmtree(build_dir)
        os.makedirs(build_dir, exist_ok=True)
        if not os.path.isfile(sentinel):
            with open(sentinel, "w") as f:
                json.dump({"num_shards": num_shards}, f)
    if enable_distributed:
        accelerator.wait_for_everyone()

    # 2. Per-rank Arrow file. Basename is byte-equivalent to today's HF auto-cache
    #    name; the rank_*_of_N subdir prevents cross-config collisions if a stale
    #    .tmp directory survives a launch-config change between runs.
    part_arrow_path = os.path.join(
        build_dir,
        "_parts",
        f"rank_{shard_idx:05d}_of_{num_shards:05d}",
        f"cache-{shard_fp}.arrow",
    )
    kwargs["target_arrow_path"] = part_arrow_path

    logger.info(
        f"Preprocessing {split} dataset shard {shard_idx}/{num_shards - 1} -> {part_arrow_path}"
    )
    _ = GeneralDataset(split=split, **kwargs)

    if enable_distributed:
        accelerator.wait_for_everyone()

    # 3. Consolidate: write top-level state.json + dataset_info.json (no row data
    #    copied) and atomically rename .tmp -> merged_cache_path.
    if is_orchestrator:
        all_parts = [
            os.path.join(
                build_dir,
                "_parts",
                f"rank_{i:05d}_of_{num_shards:05d}",
                f"cache-{merged_fp}_shard{i}of{num_shards - 1}.arrow",
            )
            for i in range(num_shards)
        ]
        GeneralDataset.consolidate_parts(merged_cache_path, all_parts, split=split)
        mode_label = preprocess_parallelism if enable_distributed else "single"
        logger.info(
            f"[{mode_label}] Consolidated {num_shards} part(s) for {split} split "
            f"-> {merged_cache_path}"
        )

    if enable_distributed:
        accelerator.wait_for_everyone()
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
