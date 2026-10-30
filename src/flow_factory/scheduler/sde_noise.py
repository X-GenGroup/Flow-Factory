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

"""Per-sample SDE transition seeds for rollout.

The inference ``generator`` seeds only ``x_T``. Each ``scheduler.step`` draws
its transition noise from a fresh generator keyed by
``(seed, step_index, source_id, sampling_sample_id)``. The key uses the sample
id, not the group id, so members of one group do not share transition noise.
``source_id`` keeps two datasets that reuse the same sample id apart. A rerun
matches.
"""

from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Sequence

import torch

from ..data_utils.sampling_plan import SAMPLING_SAMPLE_ID_COLUMN
from ..samples import LEGACY_SOURCE_ID
from ..trainers.common.group_noise import stable_text_id
from ..utils.base import create_generator

SEED_TAG_SDE_STEP = 5

_PINNED_SDE_SAMPLE: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "flow_factory_pinned_sde_sample",
    default=None,
)


def _as_row_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int for {name}, got {type(value).__name__}: {value!r}")
    return value


def rollout_sde_sample_ids(batch: Any) -> List[int]:
    """Return one SDE sample id per prompt row.

    Planned training batches already carry ``__sampling_sample_id__``. Eval and
    other unplanned batches use a stable hash of the prompt so different prompts
    stay apart and a rerun matches.
    """
    prompts = batch.get("prompt") if isinstance(batch, dict) else None
    prompt_count = len(prompts) if isinstance(prompts, (list, tuple)) else None
    sample_ids = batch.get(SAMPLING_SAMPLE_ID_COLUMN) if isinstance(batch, dict) else None
    if isinstance(sample_ids, (list, tuple)):
        if prompt_count is None or len(sample_ids) != prompt_count:
            received = len(sample_ids)
            raise ValueError(
                "rollout SDE noise requires one sampling_sample_id per prompt row; "
                f"prompts={prompt_count}, received {received}"
            )
        return [
            _as_row_int(sample_id, f"sampling_sample_id[{index}]")
            for index, sample_id in enumerate(sample_ids)
        ]
    if prompt_count is None:
        received = None if prompts is None else type(prompts).__name__
        raise ValueError(
            "rollout SDE noise requires planned sampling_sample_id values or a prompt "
            f"list, received prompts={received}"
        )
    resolved: List[int] = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt:
            raise TypeError(
                "expected a non-empty str prompt to seed rollout SDE noise at "
                f"row {index}, got {type(prompt).__name__}: {prompt!r}"
            )
        resolved.append(stable_text_id(prompt))
    return resolved


def rollout_sde_source_ids(batch: Any, count: int) -> List[int]:
    """Return one source id per prompt row.

    Missing ``__source_id__`` uses the legacy single-source id. A present column
    must line up with the prompt batch so two datasets cannot share a default.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"expected a positive prompt count, received {count!r}")
    source_ids = batch.get("__source_id__") if isinstance(batch, dict) else None
    if source_ids is None:
        return [LEGACY_SOURCE_ID] * count
    if not isinstance(source_ids, (list, tuple)) or len(source_ids) != count:
        received = None if not isinstance(source_ids, (list, tuple)) else len(source_ids)
        raise ValueError(
            "rollout SDE noise requires one __source_id__ per prompt row; "
            f"prompts={count}, received {received}"
        )
    return [
        _as_row_int(source_id, f"__source_id__[{index}]")
        for index, source_id in enumerate(source_ids)
    ]


def sde_step_generators(
    base_seed: int,
    step_index: int,
    sample_ids: Sequence[int],
    source_ids: Optional[Sequence[int]] = None,
) -> List[torch.Generator]:
    """Build one fresh CPU generator per sample for a single SDE transition.

    The key is ``(base_seed, step_index, source_id, sample_id)``, not the group
    id, so members of one group do not reuse each other's transition noise.
    ``source_id`` keeps the same sample id from two datasets apart. A missing
    source list uses the legacy single-source id. A new generator is built for
    every call; nothing is carried across steps.
    """
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise TypeError(
            "expected a non-negative int for step_index, "
            f"got {type(step_index).__name__}: {step_index!r}"
        )
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError(
            f"expected int for base_seed, got {type(base_seed).__name__}: {base_seed!r}"
        )
    if source_ids is None:
        resolved_sources = [LEGACY_SOURCE_ID] * len(sample_ids)
    elif len(source_ids) != len(sample_ids):
        raise ValueError(
            "SDE step generators require one source id per sample id; "
            f"sample_ids={len(sample_ids)}, source_ids={len(source_ids)}"
        )
    else:
        resolved_sources = [
            _as_row_int(source_id, f"source_id[{index}]")
            for index, source_id in enumerate(source_ids)
        ]
    generators: List[torch.Generator] = []
    for index, sample_id in enumerate(sample_ids):
        generators.append(
            create_generator(
                int(base_seed),
                step_index,
                resolved_sources[index],
                _as_row_int(sample_id, f"sampling_sample_id[{index}]"),
                SEED_TAG_SDE_STEP,
            )
        )
    return generators


class _SDEStepCursor:
    """Map each ``scheduler.step`` call onto ``(step_index, sample_ids)``.

    Batched calls cover the inference batch in order and share one step index.
    A pinned sample index marks a loop that finishes every step of one sample
    before the next sample.
    """

    def __init__(self, sample_ids: Sequence[int], source_ids: Sequence[int]) -> None:
        if len(source_ids) != len(sample_ids):
            raise ValueError(
                "SDE cursor requires one source id per sample id; "
                f"sample_ids={len(sample_ids)}, source_ids={len(source_ids)}"
            )
        self.sample_ids = tuple(sample_ids)
        self.source_ids = tuple(source_ids)
        self.next_step = [0] * len(self.sample_ids)
        self.row = 0

    def take(self, batch_size: int) -> tuple[int, List[int], List[int]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                f"expected a positive batch size for an SDE step, received {batch_size!r}"
            )
        pinned = _PINNED_SDE_SAMPLE.get()
        if pinned is not None:
            if batch_size != 1:
                raise ValueError(
                    "a pinned SDE sample expects scheduler.step batch size 1, "
                    f"received {batch_size}"
                )
            if not 0 <= pinned < len(self.sample_ids):
                raise IndexError(
                    f"pinned SDE sample index {pinned} is outside the batch of "
                    f"{len(self.sample_ids)} samples"
                )
            step_index = self.next_step[pinned]
            self.next_step[pinned] += 1
            return step_index, [self.sample_ids[pinned]], [self.source_ids[pinned]]
        if self.row + batch_size > len(self.sample_ids):
            raise ValueError(
                "SDE step batch does not line up with the inference batch: "
                f"row={self.row}, batch_size={batch_size}, samples={len(self.sample_ids)}"
            )
        rows = range(self.row, self.row + batch_size)
        step_indexes = {self.next_step[row] for row in rows}
        if len(step_indexes) != 1:
            raise ValueError(
                "one scheduler.step call mixes samples that have already taken "
                f"different numbers of SDE steps: {sorted(step_indexes)}"
            )
        step_index = step_indexes.pop()
        for row in rows:
            self.next_step[row] += 1
        self.row += batch_size
        if self.row == len(self.sample_ids):
            self.row = 0
        return (
            step_index,
            [self.sample_ids[row] for row in rows],
            [self.source_ids[row] for row in rows],
        )


@contextmanager
def bind_sde_sample(index: int) -> Iterator[None]:
    """Mark a loop that runs every denoising step of one sample before the next.

    The index is into the inference batch, not a group id.
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise TypeError(
            f"expected a non-negative int SDE sample index, got {type(index).__name__}: {index!r}"
        )
    token = _PINNED_SDE_SAMPLE.set(index)
    try:
        yield
    finally:
        _PINNED_SDE_SAMPLE.reset(token)


def _adapter_schedulers(adapter: Any) -> List[Any]:
    group = getattr(adapter, "scheduler_group", None)
    if group is None:
        scheduler = getattr(adapter, "scheduler", None)
        if scheduler is None or not callable(getattr(scheduler, "step", None)):
            raise TypeError(
                f"expected {type(adapter).__name__} to own a scheduler_group or scheduler.step"
            )
        return [scheduler]
    schedulers: List[Any] = []
    seen: set[int] = set()
    for name in group:
        scheduler = group[name]
        if id(scheduler) in seen:
            continue
        seen.add(id(scheduler))
        schedulers.append(scheduler)
    if not schedulers:
        raise TypeError(f"expected {type(adapter).__name__} to own at least one scheduler")
    return schedulers


@contextmanager
def seeded_sde_transitions(
    adapter: Any,
    *,
    base_seed: int,
    sample_ids: Sequence[int],
    source_ids: Optional[Sequence[int]] = None,
) -> Iterator[None]:
    """Install per-sample SDE generators on every scheduler for one rollout call.

    Restores ``step`` on exit. A step that already passes ``next_latents`` is a
    replay and is left unchanged. A step that already passes ``generator`` is
    rejected so the inference generator cannot be reused as transition noise.
    """
    if source_ids is None:
        source_ids = [LEGACY_SOURCE_ID] * len(sample_ids)
    cursor = _SDEStepCursor(sample_ids, source_ids)
    restored: List[tuple[Any, Callable[..., Any]]] = []
    for scheduler in _adapter_schedulers(adapter):
        original = scheduler.step

        def step(*args: Any, _original: Callable[..., Any] = original, **kwargs: Any) -> Any:
            bound = inspect.signature(_original).bind(*args, **kwargs)
            bound.apply_defaults()
            if bound.arguments.get("next_latents") is not None:
                return _original(*args, **kwargs)
            if bound.arguments.get("generator") is not None:
                raise RuntimeError(
                    "seeded SDE transitions found a scheduler.step generator. "
                    "The inference generator must not be forwarded into SDE steps "
                    "(constraint #12b)."
                )
            latents = bound.arguments["latents"]
            if not isinstance(latents, torch.Tensor) or latents.ndim < 1:
                raise TypeError(
                    "seeded SDE transitions expected scheduler.step latents to be a "
                    f"batched tensor, received {type(latents).__name__}"
                )
            step_index, row_ids, row_sources = cursor.take(int(latents.shape[0]))
            bound.arguments["generator"] = sde_step_generators(
                base_seed, step_index, row_ids, row_sources
            )
            return _original(*bound.args, **bound.kwargs)

        scheduler.step = step
        restored.append((scheduler, original))
    try:
        yield
        if cursor.row != 0:
            raise RuntimeError(
                "seeded SDE transitions ended in the middle of the inference batch: "
                f"row={cursor.row}, samples={len(cursor.sample_ids)}"
            )
    finally:
        for scheduler, original in reversed(restored):
            scheduler.step = original
