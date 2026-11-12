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

"""Acquisition drivers and progress state for the unified training kernel."""

from dataclasses import dataclass, replace
from typing import Any, Protocol

from torch.utils.data import DataLoader, DistributedSampler

from ..contracts.execution import (
    DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT,
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
    ExecutionContract,
    FeedbackMode,
)


@dataclass(frozen=True)
class TrainingProgress:
    """Track optimizer updates independently from acquisition cycles."""

    optimizer_step: int = 0
    rollout_iteration: int = 0
    data_epoch: int = 0

    def __post_init__(self) -> None:
        """Reject invalid counter state at construction."""
        _require_non_negative_int(self.optimizer_step, "optimizer_step")
        _require_non_negative_int(self.rollout_iteration, "rollout_iteration")
        _require_non_negative_int(self.data_epoch, "data_epoch")

    def cycle_index(self, acquisition: AcquisitionMode) -> int:
        """Return the completed-cycle count for an acquisition mode.

        Args:
            acquisition: Algorithm acquisition mode.

        Returns:
            Completed rollout iterations or complete dataset epochs.
        """
        _require_acquisition(acquisition)
        if acquisition is AcquisitionMode.GENERATION:
            return self.rollout_iteration
        return self.data_epoch

    def advance_acquisition(
        self,
        acquisition: AcquisitionMode,
        *,
        completed: bool,
    ) -> "TrainingProgress":
        """Advance one acquisition cycle only after it completed successfully.

        Args:
            acquisition: Algorithm acquisition mode.
            completed: Whether generation completed or the finite loader was exhausted.

        Returns:
            New progress with exactly one acquisition counter advanced.
        """
        _require_acquisition(acquisition)
        if type(completed) is not bool:
            raise TypeError(
                f"expected completed to be bool, received "
                f"{type(completed).__name__}: {completed!r}"
            )
        if not completed:
            raise RuntimeError(
                f"cannot advance {acquisition.value!r} because its acquisition cycle "
                "did not complete"
            )
        if acquisition is AcquisitionMode.GENERATION:
            return replace(self, rollout_iteration=self.rollout_iteration + 1)
        return replace(self, data_epoch=self.data_epoch + 1)

    def advance_optimizer_step(self, count: int = 1) -> "TrainingProgress":
        """Advance optimizer progress independently of acquisition cycles.

        Args:
            count: Positive number of completed optimizer updates.

        Returns:
            New progress with ``optimizer_step`` advanced by ``count``.
        """
        _require_positive_int(count, "count")
        return replace(self, optimizer_step=self.optimizer_step + count)


class AcquisitionHost(Protocol):
    """Define trainer hooks consumed by acquisition drivers."""

    dataloader: DataLoader

    def set_trajectory_seed(self, seed: int) -> None:
        """Set the seed for the next generated acquisition."""
        ...

    def run_generation_acquisition(self) -> None:
        """Generate examples and execute their algorithm-specific update."""
        ...

    def train_on_dataset_batch(self, batch: Any) -> None:
        """Execute the declared stages for one acquired dataset batch."""
        ...


class AcquisitionDriver(Protocol):
    """Define the runtime interface shared by acquisition strategies."""

    def prepare_cycle(
        self,
        host: AcquisitionHost,
        progress: TrainingProgress,
        *,
        seed: int,
    ) -> None:
        """Prepare one acquisition cycle before periodic boundaries."""
        ...

    def run_cycle(
        self,
        host: AcquisitionHost,
        progress: TrainingProgress,
    ) -> None:
        """Run one acquisition cycle without mutating progress."""
        ...


class GenerationAcquisitionDriver:
    """Acquire one generated example collection and train on it."""

    def prepare_cycle(
        self,
        host: AcquisitionHost,
        progress: TrainingProgress,
        *,
        seed: int,
    ) -> None:
        """Seed one generation cycle from its completed-iteration count.

        Args:
            host: Trainer hooks consumed by this driver.
            progress: Immutable progress at cycle start.
            seed: Base training seed.
        """
        _require_progress(progress)
        _require_int(seed, "seed")
        host.set_trajectory_seed(seed + progress.rollout_iteration)

    def run_cycle(
        self,
        host: AcquisitionHost,
        progress: TrainingProgress,
    ) -> None:
        """Generate and consume one acquisition without advancing progress.

        Args:
            host: Trainer hooks consumed by this driver.
            progress: Immutable progress at cycle start.
        """
        _require_progress(progress)
        host.run_generation_acquisition()


class DatasetAcquisitionDriver:
    """Acquire every batch in one finite distributed dataset epoch."""

    def prepare_cycle(
        self,
        host: AcquisitionHost,
        progress: TrainingProgress,
        *,
        seed: int,
    ) -> None:
        """Validate a finite official distributed loader before side effects.

        Args:
            host: Trainer whose dataloader supplies offline examples.
            progress: Immutable progress at cycle start.
            seed: Common interface value; dataset shuffling uses sampler epochs.
        """
        _require_progress(progress)
        _require_int(seed, "seed")
        _require_distributed_sampler(host)

    def run_cycle(
        self,
        host: AcquisitionHost,
        progress: TrainingProgress,
    ) -> None:
        """Exhaust exactly one dataloader traversal without advancing progress.

        Args:
            host: Trainer whose dataloader supplies offline examples.
            progress: Immutable progress at cycle start.

        Note:
            Batch or optimization exceptions propagate. The caller advances
            ``data_epoch`` only after this method returns normally.
        """
        _require_progress(progress)
        sampler = _require_distributed_sampler(host)
        sampler.set_epoch(progress.data_epoch)
        for batch in host.dataloader:
            host.train_on_dataset_batch(batch)


def build_acquisition_driver(contract: ExecutionContract) -> AcquisitionDriver:
    """Build the acquisition strategy declared by an execution contract.

    Args:
        contract: Typed algorithm execution contract.

    Returns:
        Generation or dataset acquisition driver.
    """
    if not isinstance(contract, ExecutionContract):
        raise TypeError(
            "expected contract to be ExecutionContract, received "
            f"{type(contract).__name__}: {contract!r}"
        )
    if contract.acquisition is AcquisitionMode.GENERATION:
        return GenerationAcquisitionDriver()
    return DatasetAcquisitionDriver()


def _require_distributed_sampler(host: AcquisitionHost) -> DistributedSampler:
    """Return the official sampler required by dataset acquisition."""
    dataloader = getattr(host, "dataloader", None)
    sampler = getattr(dataloader, "sampler", None)
    if not isinstance(sampler, DistributedSampler):
        sampler_name = type(sampler).__name__ if sampler is not None else "None"
        raise TypeError(
            "expected offline dataloader.sampler to be "
            "torch.utils.data.DistributedSampler, received "
            f"{sampler_name}; use DistributedSampler even when num_replicas=1"
        )
    return sampler


def _require_acquisition(value: object) -> None:
    """Require a typed acquisition mode."""
    if not isinstance(value, AcquisitionMode):
        raise TypeError(
            "expected acquisition to be AcquisitionMode, received "
            f"{type(value).__name__}: {value!r}"
        )


def _require_non_negative_int(value: object, field_name: str) -> None:
    """Require a non-negative integer counter without accepting booleans."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received " f"{type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"expected {field_name} >= 0, received {value}")


def _require_positive_int(value: object, field_name: str) -> None:
    """Require a positive integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received " f"{type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(f"expected {field_name} >= 1, received {value}")


def _require_int(value: object, field_name: str) -> None:
    """Require an integer without accepting booleans."""
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received " f"{type(value).__name__}: {value!r}"
        )


def _require_progress(progress: object) -> None:
    """Require immutable typed progress at a driver boundary."""
    if not isinstance(progress, TrainingProgress):
        raise TypeError(
            "expected progress to be TrainingProgress, received "
            f"{type(progress).__name__}: {progress!r}"
        )


__all__ = [
    "DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT",
    "AcquisitionDriver",
    "AcquisitionHost",
    "AcquisitionMode",
    "DatasetAcquisitionDriver",
    "ExecutionContract",
    "FeedbackMode",
    "GenerationAcquisitionDriver",
    "OFFLINE_EXECUTION_CONTRACT",
    "ONLINE_EXECUTION_CONTRACT",
    "ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT",
    "TrainingProgress",
    "build_acquisition_driver",
]
