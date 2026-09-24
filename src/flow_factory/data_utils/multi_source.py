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

# src/flow_factory/data_utils/multi_source.py
"""Multi-source train DataLoader components.

Houses the weighted, deterministic cross-source scheduler and the wrapper
DataLoader that drives one underlying DataLoader per training source. These
are composed by ``data_utils.loader.get_train_dataloader`` when more than one
training source is declared.
"""

import hashlib
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader

_SCHEDULE_SEED_DOMAIN = "flow_factory.multi_source_schedule.v1"


def _stable_schedule_seed(seed: int, epoch: int) -> int:
    """Derive a process-independent seed for one source-schedule epoch."""
    payload = f"{_SCHEDULE_SEED_DOMAIN}:{seed}:{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


class WeightedSourceBatchScheduler:
    """Build a deterministic shared-across-ranks source schedule.

    Source names are repeated in ``batches_per_block`` chunks and those chunks
    are shuffled under a ``torch.Generator`` seeded by a stable digest of
    ``seed`` and ``epoch``. The default one-batch block preserves the legacy
    schedule exactly. Reward-overlap loaders use the sampler's smallest
    group-complete window so cross-source interleaving cannot split a group.
    All ranks see the same list every epoch.

    The input dict's iteration order is **ignored** — sources are processed
    in ``sorted(name)`` order so the generated schedule is byte-identical
    across runs and across rank-zero re-runs (no insertion-order
    dependence).  Combined with the seed, this yields total reproducibility.

    Why a list, not a stream:

    - We need ``__len__`` for ``tqdm`` and exact-length validation.
    - Mid-epoch checkpointing can record an integer step index and resume
      from there.
    - The "effective num_batches_per_epoch" is checked against
      ``training_args.num_batches_per_epoch`` once at build time.
    """

    def __init__(
        self,
        num_batches_per_source: Dict[str, int],
        seed: int,
        batches_per_block: int = 1,
    ) -> None:
        if type(batches_per_block) is not int or batches_per_block < 1:
            raise ValueError(
                f"batches_per_block must be a positive integer, received {batches_per_block!r}"
            )
        incompatible = {
            name: count
            for name, count in num_batches_per_source.items()
            if count % batches_per_block
        }
        if incompatible:
            raise ValueError(
                "each source batch count must close complete scheduling blocks: "
                f"batches_per_block={batches_per_block}, incompatible={incompatible!r}"
            )
        self._counts: Dict[str, int] = dict(num_batches_per_source)
        self._seed = int(seed)
        self._batches_per_block = batches_per_block
        self._epoch = 0
        self._schedule: List[str] = []
        self._build()

    def _build(self) -> None:
        """Materialise the per-epoch shuffled name sequence."""
        blocks: List[str] = []
        for name in sorted(self._counts.keys()):
            blocks.extend([name] * (self._counts[name] // self._batches_per_block))

        if not blocks:
            self._schedule = []
            return

        g = torch.Generator()
        g.manual_seed(_stable_schedule_seed(self._seed, self._epoch))
        perm = torch.randperm(len(blocks), generator=g).tolist()
        self._schedule = [
            blocks[index] for index in perm for _batch_in_block in range(self._batches_per_block)
        ]

    def __iter__(self) -> Iterator[str]:
        return iter(self._schedule)

    def __len__(self) -> int:
        return len(self._schedule)

    def set_epoch(self, epoch: int) -> None:
        """Reseed the shuffle for the given epoch."""
        self._epoch = int(epoch)
        self._build()


class MultiSourceTrainDataLoader:
    """Iterate per-source DataLoaders in a weighted, shuffled order.

    Wraps a ``Dict[str, DataLoader]`` plus a
    :class:`WeightedSourceBatchScheduler`.  Each yielded batch dict is
    augmented with ``__source__: List[str]`` of length ``B`` (homogeneous
    in this PR; the per-sample shape leaves room for future PRs that
    might interleave within a batch without code changes).

    Key contracts (consumed by ``BaseTrainer``):

    - ``__len__`` == ``num_batches_per_epoch`` (so existing
      ``tqdm(range(num_batches_per_epoch))`` keeps working unchanged).
    - ``set_epoch(epoch)`` reseeds the schedule AND propagates to every
      per-source ``batch_sampler.set_epoch(epoch)``, then drops cached
      iters so the next ``__iter__()`` starts fresh.
    - ``dataloaders_by_source`` exposes the underlying per-source dict
      so the future ``DiffusionOPDTrainer`` can drive its own balanced
      per-teacher sampling without going through the global scheduler.
    """

    def __init__(
        self,
        dataloaders_by_source: Dict[str, DataLoader],
        scheduler: WeightedSourceBatchScheduler,
        source_name_to_id: Optional[Dict[str, int]] = None,
        batch_size: Optional[int] = None,
    ):
        self._loaders_by_source = dataloaders_by_source
        self._scheduler = scheduler
        # Optional name -> id mapping; when present, every batch carries
        # both `__source__` (str, for logs/debugging) and `__source_id__`
        # (int, for hot-path gate + cross-rank gather). Resolved by
        # `Arguments._assign_source_ids` and read here from
        # `data_args.source_name_to_id`. None means "id form not configured" —
        # the str form alone is emitted (legacy behavior).
        self._source_name_to_id = source_name_to_id or {}
        # Explicit batch size when known (item 7's exact-divisibility
        # geometry guarantees every batch is exactly `per_device_batch_size`).
        # When None, fall back to the per-batch heuristic in `_infer_batch_size`.
        self._batch_size = batch_size
        self._iters: Dict[str, Iterator] = {}

    @property
    def batch_size(self) -> Optional[int]:
        """Expose the fixed batch size expected by Accelerate/DeepSpeed."""
        return self._batch_size

    @property
    def dataloaders_by_source(self) -> Dict[str, DataLoader]:
        """Public access for OPD-style consumers."""
        return self._loaders_by_source

    def _ensure_iters(self) -> None:
        """Lazily refresh per-source iterators (after set_epoch / first use)."""
        for name, loader in self._loaders_by_source.items():
            if name not in self._iters:
                self._iters[name] = iter(loader)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._ensure_iters()
        for src in self._scheduler:
            batch = next(self._iters[src])

            B = self._batch_size if self._batch_size is not None else self._infer_batch_size(batch)
            batch = dict(batch)
            batch["__source__"] = [src] * B
            # Emit the small-int form too when the registry is configured,
            # so `_inject_batch_metadata` populates `BaseSample.source_id`
            # for hot-path comparisons. Falls back gracefully when not set.
            if self._source_name_to_id:
                batch["__source_id__"] = [self._source_name_to_id[src]] * B
            yield batch

    def __len__(self) -> int:
        return len(self._scheduler)

    def set_epoch(self, epoch: int) -> None:
        """Reseed the schedule and propagate to per-source samplers."""
        self._scheduler.set_epoch(epoch)
        for loader in self._loaders_by_source.values():
            sampler = getattr(loader, "batch_sampler", None) or getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        # Force fresh iters next epoch.
        self._iters.clear()

    @staticmethod
    def _infer_batch_size(batch: Dict[str, Any]) -> int:
        """Best-effort batch-size inference from a dataloader batch dict.

        Prefers ``prompt`` (length-bearing list of strings used by every
        adapter), falls back to the first length-bearing value found.
        """
        if "prompt" in batch and hasattr(batch["prompt"], "__len__"):
            return len(batch["prompt"])
        for v in batch.values():
            if hasattr(v, "__len__"):
                try:
                    return len(v)
                except TypeError:
                    continue
        return 1
