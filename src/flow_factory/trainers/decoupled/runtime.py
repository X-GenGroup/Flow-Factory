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

"""Shared batch and timestep iteration for decoupled forward-process trainers."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import tqdm as tqdm_

from ...samples import BaseSample, LatentState, StackedSampleBatch
from ..common.state_validation import state_batch_size

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

StepT = TypeVar("StepT")


@dataclass(frozen=True)
class DecoupledReplayBatch(Generic[StepT]):
    """One device batch and the sampling-policy states precomputed for it."""

    batch: StackedSampleBatch
    clean_state: LatentState
    steps: Sequence[StepT]


def iter_decoupled_replay_batches(
    trainer: Any,
    samples: list[BaseSample],
    inner_epoch: int,
    precompute_steps: Callable[[StackedSampleBatch, LatentState, int], Sequence[StepT]],
) -> Iterator[DecoupledReplayBatch[StepT]]:
    """Yield per-batch forward-process states without changing RNG or phase order."""
    per_device_batch_size = trainer.training_args.per_device_batch_size
    num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
    shuffled_samples = trainer._order_samples_for_optimize(samples, inner_epoch)

    for batch in tqdm(
        trainer._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
        total=num_batches,
        desc=f"Epoch {trainer.epoch} Training",
        position=0,
        disable=not trainer.show_progress_bar,
    ):
        clean_state = trainer.adapter.get_terminal_state(batch)
        batch_size = state_batch_size(trainer, clean_state, "terminal clean state")
        steps = precompute_steps(batch, clean_state, batch_size)
        if len(steps) != trainer.num_train_timesteps:
            raise ValueError(
                f"expected {trainer.num_train_timesteps} precomputed timestep(s) for "
                f"{type(trainer).__name__} at inner_epoch={inner_epoch}, received {len(steps)}"
            )
        trainer.adapter.train()
        yield DecoupledReplayBatch(
            batch=batch,
            clean_state=clean_state,
            steps=steps,
        )


def iter_decoupled_steps(
    trainer: Any,
    steps: Sequence[StepT],
) -> Iterator[tuple[int, StepT]]:
    """Yield precomputed steps with the standard nested progress display."""
    if len(steps) != trainer.num_train_timesteps:
        raise ValueError(
            f"expected {trainer.num_train_timesteps} precomputed timestep(s) for "
            f"{type(trainer).__name__}, received {len(steps)}"
        )
    for step_index in tqdm(
        range(trainer.num_train_timesteps),
        desc=f"Epoch {trainer.epoch} Timestep",
        position=1,
        leave=False,
        disable=not trainer.show_progress_bar,
    ):
        yield step_index, steps[step_index]
