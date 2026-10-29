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

The inference ``generator`` seeds only ``x_T``. Transition noise, when
``train.sde_step_noise`` is ``sample``, is a fresh generator per sample per
``scheduler.step`` call. The key includes the sample id, not the group id.
"""

from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Sequence

import torch

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


def sde_step_generators(
    base_seed: int,
    step_index: int,
    sample_ids: Sequence[int],
) -> List[torch.Generator]:
    """Build one fresh CPU generator per sample for a single SDE transition.

    The key is ``(base_seed, step_index, sample_id)``, not the group id, so
    members of one group do not reuse each other's transition noise. A new
    generator is built for every call; nothing is carried across steps.
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
    generators: List[torch.Generator] = []
    for index, sample_id in enumerate(sample_ids):
        generators.append(
            create_generator(
                int(base_seed),
                step_index,
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

    def __init__(self, sample_ids: Sequence[int]) -> None:
        self.sample_ids = tuple(sample_ids)
        self.next_step = [0] * len(self.sample_ids)
        self.row = 0

    def take(self, batch_size: int) -> tuple[int, List[int]]:
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
            return step_index, [self.sample_ids[pinned]]
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
        return step_index, [self.sample_ids[row] for row in rows]


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
    adapter: Any, *, base_seed: int, sample_ids: Sequence[int]
) -> Iterator[None]:
    """Install per-sample SDE generators on every scheduler for one rollout call.

    Restores ``step`` on exit. A step that already passes ``next_latents`` is a
    replay and is left unchanged. A step that already passes ``generator`` is
    rejected so the inference generator cannot be reused as transition noise.
    """
    cursor = _SDEStepCursor(sample_ids)
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
                    "sde_step_noise='sample' found a scheduler.step generator. "
                    "The inference generator must not be forwarded into SDE steps "
                    "(constraint #12b)."
                )
            latents = bound.arguments["latents"]
            if not isinstance(latents, torch.Tensor) or latents.ndim < 1:
                raise TypeError(
                    "sde_step_noise='sample' expected scheduler.step latents to be a "
                    f"batched tensor, received {type(latents).__name__}"
                )
            step_index, row_ids = cursor.take(int(latents.shape[0]))
            bound.arguments["generator"] = sde_step_generators(base_seed, step_index, row_ids)
            return _original(*bound.args, **bound.kwargs)

        scheduler.step = step
        restored.append((scheduler, original))
    try:
        yield
        if cursor.row != 0:
            raise RuntimeError(
                "sde_step_noise='sample' ended in the middle of the inference batch: "
                f"row={cursor.row}, samples={len(cursor.sample_ids)}"
            )
    finally:
        for scheduler, original in reversed(restored):
            scheduler.step = original
