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

"""Finite offline dataloaders using PyTorch's official distribution semantics."""

from __future__ import annotations

from typing import Sequence, Union

from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from .offline_dataset import OfflineCollator, OfflineDataset
from .sampler import OfflineConditionSampler

OfflineDatasetCollection = Union[OfflineDataset, Sequence[OfflineDataset]]


def build_offline_dataloader(
    datasets: OfflineDatasetCollection,
    *,
    source_weights: Sequence[int | float],
    batch_size: int,
    num_replicas: int,
    rank: int,
    gradient_accumulation_steps: int,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    sampler_drop_last: bool = False,
    batch_drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """Build one finite, already-distributed offline epoch loader.

    Every source is concatenated once through a :class:`ConcatDataset`. Weighted
    replacement would make "one complete dataloader traversal" stop meaning one
    data epoch, so all source weights must be explicitly equal to one. The returned
    loader is not passed through any Accelerator preparation; its official
    :class:`DistributedSampler` already owns rank sharding. With
    ``sampler_drop_last=False``, PyTorch may repeat tail indices so every rank has
    the same length; that standard sampler behavior remains part of the traversal.

    The execution driver, not this builder, calls ``sampler.set_epoch`` before
    each traversal.

    Args:
        datasets: One offline dataset or an ordered sequence of sources.
        source_weights: One explicit unit weight per source.
        batch_size: Per-rank DataLoader batch size.
        num_replicas: Total distributed process count, including one-process runs.
        rank: Current global process rank.
        gradient_accumulation_steps: Complete batch windows required per epoch.
        num_workers: CPU DataLoader worker count.
        shuffle: Whether the official sampler shuffles each epoch.
        seed: Shared sampler seed.
        sampler_drop_last: Whether the official sampler drops the dataset tail
            that cannot be divided evenly across ranks.
        batch_drop_last: Whether DataLoader drops a rank-local incomplete batch.
        pin_memory: Whether DataLoader tensors use pinned host memory. Defaults
            to ``False``, which is safe for macOS and MPS.

    Returns:
        A finite DataLoader with ``DistributedSampler`` at ``loader.sampler``.

    Raises:
        TypeError: If an argument has the wrong runtime type.
        ValueError: If source contracts or batch geometry are invalid.
    """
    offline_datasets = _normalize_datasets(datasets)
    _validate_source_weights(source_weights, source_count=len(offline_datasets))
    _require_positive_int(batch_size, "batch_size")
    _require_positive_int(num_replicas, "num_replicas")
    _require_non_negative_int(rank, "rank")
    if rank >= num_replicas:
        raise ValueError(
            f"rank must satisfy 0 <= rank < num_replicas, got "
            f"rank={rank}, num_replicas={num_replicas}"
        )
    _require_positive_int(gradient_accumulation_steps, "gradient_accumulation_steps")
    _require_non_negative_int(num_workers, "num_workers")
    _require_int(seed, "seed")
    _require_bool(shuffle, "shuffle")
    _require_bool(sampler_drop_last, "sampler_drop_last")
    _require_bool(batch_drop_last, "batch_drop_last")
    _require_bool(pin_memory, "pin_memory")

    supervision_type = offline_datasets[0].supervision_type
    for source_index, dataset in enumerate(offline_datasets[1:], start=1):
        if dataset.supervision_type != supervision_type:
            raise ValueError(
                "offline sources must share one supervision_type, "
                f"source 0 has {supervision_type!r} while source {source_index} "
                f"has {dataset.supervision_type!r}"
            )
    _validate_unique_source_identities(offline_datasets)

    concatenated = ConcatDataset(offline_datasets)
    sampler_type = (
        OfflineConditionSampler
        if any(ds.sft_conditions is not None for ds in offline_datasets)
        else DistributedSampler
    )
    sampler = sampler_type(
        concatenated,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=sampler_drop_last,
    )
    loader = DataLoader(
        concatenated,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=batch_drop_last,
        collate_fn=OfflineCollator(supervision_type),
    )

    num_batches = len(loader)
    if num_batches == 0:
        raise ValueError(
            "offline dataloader is empty on this rank; adjust dataset size, "
            "batch_size, num_replicas, sampler_drop_last, or batch_drop_last"
        )
    if num_batches % gradient_accumulation_steps != 0:
        raise ValueError(
            f"offline dataloader yields {num_batches} batches on rank {rank}, which is not "
            f"divisible by gradient_accumulation_steps={gradient_accumulation_steps}. "
            "Offline training does not add batches solely to close or implicitly flush a "
            "partial gradient-accumulation window; adjust dataset size, batch_size, num_replicas, "
            "sampler_drop_last, batch_drop_last, or gradient_accumulation_steps explicitly."
        )
    return loader


def _normalize_datasets(datasets: OfflineDatasetCollection) -> tuple[OfflineDataset, ...]:
    if isinstance(datasets, OfflineDataset):
        normalized = (datasets,)
    else:
        if isinstance(datasets, (str, bytes)):
            raise TypeError("datasets must be OfflineDataset instances, not a string")
        try:
            normalized = tuple(datasets)
        except TypeError as exc:
            raise TypeError(
                "datasets must be one OfflineDataset or a sequence of OfflineDataset instances"
            ) from exc
    if not normalized:
        raise ValueError("offline dataloader requires at least one OfflineDataset source")
    for source_index, dataset in enumerate(normalized):
        if not isinstance(dataset, OfflineDataset):
            raise TypeError(
                f"offline source {source_index} must be OfflineDataset, "
                f"got {type(dataset).__name__}"
            )
    return normalized


def _validate_source_weights(
    source_weights: Sequence[int | float],
    *,
    source_count: int,
) -> None:
    if isinstance(source_weights, (str, bytes)):
        raise TypeError("source_weights must be a numeric sequence")
    try:
        weights = tuple(source_weights)
    except TypeError as exc:
        raise TypeError("source_weights must be a numeric sequence") from exc
    if len(weights) != source_count:
        raise ValueError(
            f"source_weights must contain one entry per offline source, "
            f"got {len(weights)} weights for {source_count} sources"
        )
    for source_index, weight in enumerate(weights):
        if type(weight) not in (int, float):
            raise TypeError(
                f"offline source weight {source_index} must be int or float, "
                f"got {type(weight).__name__}: {weight!r}"
            )
        if weight != 1:
            raise ValueError(
                f"offline source weight {source_index} must equal 1, got {weight!r}; "
                "weighted replacement is incompatible with a full-dataloader data epoch"
            )


def _validate_unique_source_identities(datasets: Sequence[OfflineDataset]) -> None:
    source_names: dict[str, int] = {}
    source_ids: dict[int, int] = {}
    for source_index, dataset in enumerate(datasets):
        previous_name_index = source_names.get(dataset.source_name)
        if previous_name_index is not None:
            raise ValueError(
                f"offline source_name {dataset.source_name!r} is duplicated at source indices "
                f"{previous_name_index} and {source_index}; source names must be unique for "
                "provenance and metric routing"
            )
        previous_id_index = source_ids.get(dataset.source_id)
        if previous_id_index is not None:
            raise ValueError(
                f"offline source_id {dataset.source_id} is duplicated at source indices "
                f"{previous_id_index} and {source_index}; source ids must be unique for "
                "provenance and metric routing"
            )
        source_names[dataset.source_name] = source_index
        source_ids[dataset.source_id] = source_index


def _require_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be int, got {type(value).__name__}: {value!r}")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be int, got {type(value).__name__}: {value!r}")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")


def _require_int(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be int, got {type(value).__name__}: {value!r}")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be bool, got {type(value).__name__}: {value!r}")


__all__ = ["build_offline_dataloader"]
