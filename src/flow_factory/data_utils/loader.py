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
from typing import Dict, List, Literal, Optional, Tuple, Union

from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..contracts.sampler import get_sampler_layout_contract
from ..data_utils.dataset import PreprocessCallable
from ..hparams import Arguments
from ..hparams.dataset_args import DatasetArguments
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger
from .dataset import GeneralDataset
from .multi_source import MultiSourceTrainDataLoader, WeightedSourceBatchScheduler
from .sampler_loader import get_data_sampler

logger = setup_logger(__name__, rank_zero_only=False)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        cache_dir=kwargs["cache_dir"],
        max_dataset_size=kwargs.get("max_dataset_size"),
        preprocess_func=kwargs.get("preprocess_func"),
        preprocess_kwargs=kwargs.get("preprocess_kwargs"),
        extra_hash_strs=kwargs.get("extra_hash_strs", []),
        source_hash_override=kwargs.get("source_hash_override"),
    )

    if os.path.exists(merged_cache_path) and not base_kwargs.get("force_reprocess", False):
        if accelerator.is_local_main_process:
            logger.info(f"Loading {split} dataset from merged cache: {merged_cache_path}")
        return GeneralDataset.load_merged(merged_cache_path)

    shard_idx = kwargs["shard_index"]
    num_shards = kwargs["num_shards"]

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

    # Pick the single owner of build-dir prep + final consolidation per case:
    #   - non-distributed:  the lone process.
    #   - "global" mode:    rank-0 globally. Required when the cache_dir lives
    #                       on a shared FS visible to every node — a single
    #                       orchestrator eliminates the cross-node race on
    #                       shutil.rmtree and sentinel writes.
    #   - "local"  mode:    per-node local main. ASSUMES cache_dir is on
    #                       node-local storage (each node has its own copy of
    #                       the build dir). Pointing "local" mode at a shared
    #                       FS WILL race across node-local mains and corrupt
    #                       the build dir; that configuration is unsupported.
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
    #    .tmp directory survives a launch-config change between runs. Layout is
    #    owned by GeneralDataset so the writer and the consolidator cannot drift.
    part_arrow_path = GeneralDataset.build_part_arrow_path(merged_cache_path, shard_idx, num_shards)
    kwargs["target_arrow_path"] = part_arrow_path

    logger.info(
        f"Preprocessing {split} dataset shard {shard_idx:04d}/{num_shards - 1:04d} "
        f"-> {part_arrow_path}"
    )
    _ = GeneralDataset(split=split, **kwargs)

    if enable_distributed:
        accelerator.wait_for_everyone()

    # 3. Consolidate: write top-level state.json + dataset_info.json (no row data
    #    copied) and atomically rename .tmp -> merged_cache_path. A single call;
    #    consolidate_parts iterates the per-rank layout itself via
    #    GeneralDataset.build_part_arrow_path.
    if is_orchestrator:
        GeneralDataset.consolidate_parts(merged_cache_path, num_shards, split=split)
        mode_label = preprocess_parallelism if enable_distributed else "single"
        logger.info(
            f"[{mode_label}] Consolidated {num_shards} part(s) for {split} split "
            f"-> {merged_cache_path}"
        )

    if enable_distributed:
        accelerator.wait_for_everyone()
    return GeneralDataset.load_merged(merged_cache_path)


def get_train_dataloader(
    config: Arguments,
    accelerator: Accelerator,
    preprocess_func: Optional[PreprocessCallable] = None,
    **kwargs,
) -> Tuple[
    Union[DataLoader, "MultiSourceTrainDataLoader", None],
    Dict[str, DataLoader],
]:
    """Factory for the training DataLoader(s).

    Returns a 2-tuple ``(train_loader, train_loaders_by_source)``:

    * ``train_loader`` — either a plain ``torch.utils.data.DataLoader``
      (legacy single-source) or a :class:`MultiSourceTrainDataLoader`
      (multi-source).  Both expose ``__iter__`` / ``__len__`` /
      ``set_epoch`` so trainers don't have to branch.  ``None`` only
      when ``data.datasets`` is set but no entry has ``train: enabled``
      (eval-only run; trainers should respect that).
    * ``train_loaders_by_source`` — ``Dict[str, DataLoader]`` keyed
      by training-dataset name in multi-source mode; empty ``{}`` in
      legacy mode.  Exposed publicly for the future DiffusionOPD
      trainer (which iterates per-source independently).

    The eval / test path is fully owned by :func:`get_eval_dataloaders`;
    callers requesting an eval loader must invoke it explicitly.

    Args:
        config: Full ``Arguments`` configuration object.
        accelerator: Accelerator for distributed preprocessing & sampling.
        preprocess_func: Adapter's batch-preprocessing function.
        **kwargs: Reserved for future use (currently ignored).
    """
    data_args = config.data_args
    training_args = config.training_args

    enable_distributed = accelerator.num_processes > 1 and data_args.enable_preprocess
    preprocess_parallelism = getattr(data_args, "preprocess_parallelism", "local")

    # Common dataset kwargs (shared across legacy / multi-source paths).
    base_kwargs = {
        "preprocess_func": preprocess_func,
        "preprocess_kwargs": (
            filter_kwargs(preprocess_func, **data_args) if preprocess_func else None
        ),
        "extra_hash_strs": [
            config.model_args.model_type,
            config.model_args.model_name_or_path,
        ],
    }
    base_kwargs.update(filter_kwargs(GeneralDataset.__init__, **data_args))
    base_kwargs["force_reprocess"] = data_args.force_reprocess

    # Train preprocess kwargs (algorithm-aware guidance scale, etc.).
    train_preprocess_kwargs = (base_kwargs.get("preprocess_kwargs") or {}).copy()
    train_preprocess_kwargs.update({"is_train": True, **training_args})
    train_preprocess_kwargs["guidance_scale"] = training_args.get_preprocess_guidance_scale()
    train_preprocess_kwargs = (
        filter_kwargs(preprocess_func, **train_preprocess_kwargs)
        if preprocess_func
        else train_preprocess_kwargs
    )

    # ------------------------------------------------------------------
    # Train path: go through the per-source loader builder.  All configs
    # use `data.datasets` (the unified schema).  For a single training
    # source we hand back the underlying plain DataLoader to keep batches
    # lean (no per-batch __source__ injection).
    # ------------------------------------------------------------------
    train_loader: Union[DataLoader, "MultiSourceTrainDataLoader", None]
    train_loaders_by_source: Dict[str, DataLoader] = {}

    training_specs = data_args.training_datasets  # property
    if not training_specs:
        # `data.datasets` declares no training-eligible entry -> eval-only
        # run. Trainers that need a train loop must respect this.
        train_loader = None
    else:
        per_source_loaders = _load_per_source_train_dataloaders(
            training_datasets=training_specs,
            config=config,
            accelerator=accelerator,
            base_kwargs=base_kwargs,
            train_preprocess_kwargs=train_preprocess_kwargs,
            enable_distributed=enable_distributed,
            preprocess_parallelism=preprocess_parallelism,
        )
        train_loaders_by_source = per_source_loaders

        if len(per_source_loaders) == 1:
            # Single training source (whether the user wrote it as a
            # legacy `data.dataset_dir` or as a 1-entry `data.datasets`):
            # skip the wrapper and hand back the underlying DataLoader so
            # batches don't carry __source__ / __source_id__ keys. Reward
            # gate then falls through to the legacy "applies to all"
            # behavior; metric keys, cache fingerprints, and sample
            # schemas stay byte-identical to the pre-refactor flow.
            train_loader = next(iter(per_source_loaders.values()))
        else:
            num_batches_per_source = {
                name: loader.batch_sampler.num_batches_per_epoch  # type: ignore[union-attr]
                for name, loader in per_source_loaders.items()
            }
            total = sum(num_batches_per_source.values())
            if total != training_args.num_batches_per_epoch:
                # Caught by alignment math — but log clearly if it ever drifts.
                logger.warning(
                    f"Multi-source partition produced {total} batches/epoch but "
                    f"training_args.num_batches_per_epoch = "
                    f"{training_args.num_batches_per_epoch}. "
                    "This indicates a partitioning bug; "
                    "tqdm and gradient accumulation will use the dataloader's actual length."
                )
            batches_per_source_block = 1
            if training_args.reward_optimization_overlap:
                sampler_layout = get_sampler_layout_contract(data_args.sampler_type)
                batches_per_source_block = sampler_layout.global_group_window_batches(
                    num_replicas=accelerator.num_processes,
                    per_device_batch_size=training_args.per_device_batch_size,
                    group_size=training_args.group_size,
                )
            scheduler = WeightedSourceBatchScheduler(
                num_batches_per_source=num_batches_per_source,
                seed=training_args.seed,
                batches_per_block=batches_per_source_block,
            )
            train_loader = MultiSourceTrainDataLoader(
                per_source_loaders,
                scheduler,
                source_name_to_id=data_args.source_name_to_id,
                batch_size=training_args.per_device_batch_size,
            )

    # The eval / test path is fully owned by `get_eval_dataloaders`;
    # callers requesting an eval loader must invoke it explicitly.

    return train_loader, train_loaders_by_source


def _load_per_source_train_dataloaders(
    *,
    training_datasets: List[DatasetArguments],
    config: Arguments,
    accelerator: Accelerator,
    base_kwargs: dict,
    train_preprocess_kwargs: dict,
    enable_distributed: bool,
    preprocess_parallelism: Literal["global", "local"],
) -> Dict[str, DataLoader]:
    """Build one DataLoader per declared training source.

    Reads the per-source aligned ``M_i`` from
    ``DatasetTrainSpec.unique_sample_num_per_epoch`` (set by
    ``Arguments._align_unique_sample_num``).  Each per-source DataLoader
    is fingerprinted with ``train_source:{name}`` so caches don't
    collide across sources that share a ``dataset_dir`` with different
    overrides.

    Sanity checks (raised here rather than in alignment so we have
    dataset lengths available):

    * ``M_i <= len(per_source_dataset)`` — otherwise raise with
      actionable advice.
    * Sum of per-source batch counts equals
      ``training_args.num_batches_per_epoch`` (asserted by caller).
    """
    out: Dict[str, DataLoader] = {}
    for d in training_datasets:
        spec = d.train
        if spec is None:
            raise RuntimeError(
                f"Internal error: dataset '{d.name}' passed to "
                "_load_per_source_train_dataloaders with train=None. "
                "The is_training_source filter should have excluded it."
            )

        # Per-source media-root + dataset_dir overrides.
        per_kwargs = dict(base_kwargs)
        per_kwargs.update(d.get_dataset_overrides())
        per_kwargs["force_reprocess"] = config.data_args.force_reprocess

        # Cache fingerprint includes the source name so two sources sharing
        # a dataset_dir with different overrides get separate caches.
        extra = list(base_kwargs.get("extra_hash_strs", []))
        extra.append(f"train_source:{d.name}")
        per_kwargs["extra_hash_strs"] = extra

        # Per-source max_dataset_size override (DataArguments default
        # acts as fallback via base_kwargs.update).
        if spec.max_dataset_size is not None:
            per_kwargs["max_dataset_size"] = spec.max_dataset_size

        dataset = _create_or_load_dataset(
            split=spec.split,
            accelerator=accelerator,
            base_kwargs={**per_kwargs, "preprocess_kwargs": train_preprocess_kwargs},
            enable_distributed=enable_distributed,
            preprocess_parallelism=preprocess_parallelism,
        )

        M_i = spec.unique_sample_num_per_epoch
        if M_i is None:
            raise RuntimeError(
                f"Internal error: per-source unique_sample_num_per_epoch "
                f"is missing for source '{d.name}'. "
                f"Did `Arguments._align_batch_geometry` run?"
            )
        if M_i > len(dataset):
            raise ValueError(
                f"Training dataset '{d.name}': aligned per-source "
                f"unique_sample_num_per_epoch (M_i = {M_i}) exceeds dataset "
                f"size ({len(dataset)}). Either lower this source's "
                f"`train.weight`, lower `train.unique_sample_num_per_epoch`, "
                f"or grow the dataset."
            )

        sampler = get_data_sampler(
            dataset=dataset,
            sampler_type=config.data_args.sampler_type,
            batch_size=config.training_args.per_device_batch_size,
            group_size=config.training_args.group_size,
            unique_sample_num=M_i,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=config.training_args.seed,
        )
        out[d.name] = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.data_args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=GeneralDataset.collate_fn,
        )
        # Mirror the resolved per-source `num_batches_per_epoch` onto
        # the spec so `print(config)` shows it. The `M_i` writeback
        # already happened in `Arguments._align_unique_sample_num`.
        spec.num_batches_per_epoch = sampler.num_batches_per_epoch  # type: ignore[union-attr]
        logger.info(
            f"Multi-source: built DataLoader for '{d.name}' "
            f"(dataset_size={len(dataset)}, "
            f"unique_sample_num_per_epoch={M_i}, "
            f"num_batches_per_epoch={sampler.num_batches_per_epoch})"  # type: ignore[union-attr]
        )

    return out


def get_eval_dataloaders(
    eval_datasets: List[DatasetArguments],
    config: Arguments,
    accelerator: Accelerator,
    preprocess_func: Optional[PreprocessCallable] = None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for multiple evaluation datasets.

    Each dataset is independently preprocessed and cached using its own
    ``dataset_dir`` and a unique cache fingerprint (includes the eval dataset
    name to prevent collisions).

    Args:
        eval_datasets: List of evaluation-eligible dataset configurations
            (typically ``config.data_args.eval_datasets`` — the property
            returning ``[d for d in data.datasets if d.is_eval_source]``).
        config: Full configuration object (for model info, data args, eval args).
        accelerator: Accelerator for distributed preprocessing.
        preprocess_func: Model adapter's preprocessing function.

    Returns:
        Dict mapping eval dataset name → DataLoader, ready for evaluation.
    """
    data_args = config.data_args
    eval_args = config.eval_args

    enable_distributed = accelerator.num_processes > 1 and data_args.enable_preprocess
    preprocess_parallelism = getattr(data_args, "preprocess_parallelism", "local")

    eval_dataloaders: Dict[str, DataLoader] = {}

    for ed in eval_datasets:
        # Each entry is a DatasetArguments; its eval-only block carries
        # the split / size / sampling overrides.
        spec = ed.eval
        if spec is None or not spec.enabled:
            # Defensive: caller should have already filtered via the
            # `is_eval_source` property, but keep this safety net.
            continue

        # Per-dataset eval preprocess kwargs: merge this dataset's eval
        # overrides (notably `guidance_scale`) with the shared EvaluationArguments
        # via the same `get_merged_eval_kwargs` used by `BaseTrainer.evaluate`.
        # This keeps Stage 1 (encode_prompt) consistent with Stage 2 (inference):
        # a dataset evaluated at guidance_scale > 1.0 must cache negative prompt
        # embeds during preprocessing, otherwise CFG is silently disabled at eval.
        per_preprocess_kwargs = None
        if preprocess_func:
            merged_eval = spec.get_merged_eval_kwargs(eval_args)
            per_preprocess_kwargs = filter_kwargs(preprocess_func, **data_args).copy()
            per_preprocess_kwargs.update({"is_train": False, **merged_eval})
            per_preprocess_kwargs = filter_kwargs(preprocess_func, **per_preprocess_kwargs)

        # Check that the split file exists
        if not GeneralDataset.check_exists(ed.dataset_dir, spec.split):
            logger.warning(
                f"Eval dataset '{ed.name}': split '{spec.split}' not found in "
                f"'{ed.dataset_dir}', skipping."
            )
            continue

        # Start with filter_kwargs from data_args (same pattern as get_dataloader)
        base_kwargs = {
            "preprocess_func": preprocess_func,
            "preprocess_kwargs": per_preprocess_kwargs,
            "extra_hash_strs": [
                config.model_args.model_type,
                config.model_args.model_name_or_path,
                f"eval_{ed.name}",
            ],
        }
        base_kwargs.update(filter_kwargs(GeneralDataset.__init__, **data_args))

        # Override dataset_dir, per-dataset media-root overrides, and the
        # eval-spec-level max_dataset_size (if set).
        base_kwargs.update(ed.get_dataset_overrides())
        base_kwargs["force_reprocess"] = data_args.force_reprocess
        if spec.max_dataset_size is not None:
            base_kwargs["max_dataset_size"] = spec.max_dataset_size

        # Create/load dataset
        dataset = _create_or_load_dataset(
            split=spec.split,
            accelerator=accelerator,
            base_kwargs=base_kwargs,
            enable_distributed=enable_distributed,
            preprocess_parallelism=preprocess_parallelism,
        )

        # Create DataLoader
        eval_dataloaders[ed.name] = DataLoader(
            dataset,
            batch_size=eval_args.per_device_batch_size,
            shuffle=False,
            num_workers=data_args.dataloader_num_workers,
            collate_fn=GeneralDataset.collate_fn,
        )

    return eval_dataloaders
