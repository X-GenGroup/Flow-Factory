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

"""Optimize-loop sample ordering for batch-invariant and pack-dependent adapters."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import pytest
import torch

from flow_factory.samples import AcquisitionManifest
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.utils.base import create_generator

SEED, EPOCH = 42, 3


def _samples(num_packs: int, pack_size: int) -> List[SimpleNamespace]:
    return [
        SimpleNamespace(unique_id=pack, name=f"pack{pack}-item{item}")
        for pack in range(num_packs)
        for item in range(pack_size)
    ]


def _packs_manifest(samples: List[Any], pack_size: int) -> AcquisitionManifest:
    return AcquisitionManifest.from_samples(
        samples,
        rollout_batch_object_ids=[
            [id(sample) for sample in samples[start : start + pack_size]]
            for start in range(0, len(samples), pack_size)
        ],
    )


def _trainer(
    *,
    pack_dependent: bool,
    shuffle: bool = True,
    per_device_batch_size: int = 2,
    manifest: Optional[AcquisitionManifest] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        training_args=SimpleNamespace(
            shuffle_samples=shuffle, seed=SEED, per_device_batch_size=per_device_batch_size
        ),
        epoch=EPOCH,
        adapter=SimpleNamespace(requires_preserved_replay_batch_composition=pack_dependent),
        _rollout_acquisition_manifest=manifest,
    )


def _names(samples: List[Any]) -> List[str]:
    return [sample.name for sample in samples]


@pytest.mark.parametrize("pack_dependent", [False, True])
def test_shuffle_samples_false_keeps_the_rollout_order(pack_dependent: bool) -> None:
    samples = _samples(num_packs=4, pack_size=2)
    trainer = _trainer(pack_dependent=pack_dependent, shuffle=False)

    assert BaseTrainer._order_samples_for_optimize(trainer, samples, 0) == samples


def test_batch_invariant_adapters_shuffle_individual_samples() -> None:
    samples = _samples(num_packs=6, pack_size=2)
    trainer = _trainer(pack_dependent=False)

    ordered = BaseTrainer._order_samples_for_optimize(trainer, samples, inner_epoch=1)

    permutation = torch.randperm(len(samples), generator=create_generator(SEED, EPOCH, 1))
    assert ordered == [samples[index] for index in permutation]
    assert ordered != samples


def test_pack_dependent_adapters_shuffle_whole_rollout_packs() -> None:
    pack_size = 2
    samples = _samples(num_packs=6, pack_size=pack_size)
    trainer = _trainer(pack_dependent=True, manifest=_packs_manifest(samples, pack_size))

    ordered = BaseTrainer._order_samples_for_optimize(trainer, samples, inner_epoch=1)

    micro_batches = [
        _names(ordered[start : start + pack_size]) for start in range(0, len(ordered), pack_size)
    ]
    rollout_packs = [
        _names(samples[start : start + pack_size]) for start in range(0, len(samples), pack_size)
    ]
    pack_permutation = torch.randperm(
        len(rollout_packs), generator=create_generator(SEED, EPOCH, 1)
    )
    assert micro_batches == [rollout_packs[index] for index in pack_permutation]
    assert micro_batches != rollout_packs
    assert BaseTrainer._order_samples_for_optimize(trainer, samples, inner_epoch=1) == ordered


def test_single_sample_packs_shuffle_like_individual_samples() -> None:
    samples = _samples(num_packs=8, pack_size=1)
    pack_dependent = _trainer(
        pack_dependent=True, per_device_batch_size=1, manifest=_packs_manifest(samples, 1)
    )
    batch_invariant = _trainer(pack_dependent=False, per_device_batch_size=1)

    assert BaseTrainer._order_samples_for_optimize(
        pack_dependent, samples, 2
    ) == BaseTrainer._order_samples_for_optimize(batch_invariant, samples, 2)


def test_pack_shuffling_requires_matching_rollout_packs() -> None:
    samples = _samples(num_packs=3, pack_size=2)

    with pytest.raises(RuntimeError, match="optimize-loop pack shuffling .* rollout micro-batch"):
        BaseTrainer._order_samples_for_optimize(_trainer(pack_dependent=True), samples, 0)

    uneven = _trainer(
        pack_dependent=True,
        per_device_batch_size=2,
        manifest=_packs_manifest(samples, pack_size=3),
    )
    with pytest.raises(ValueError, match=r"train.per_device_batch_size=2; invalid_sizes=\(3, 3\)"):
        BaseTrainer._order_samples_for_optimize(uneven, samples, 0)

    reordered = _trainer(pack_dependent=True, manifest=_packs_manifest(samples, pack_size=2))
    with pytest.raises(ValueError, match="samples changed after its manifest was created"):
        BaseTrainer._order_samples_for_optimize(reordered, samples[::-1], 0)


class _GenerationHarness:
    def __init__(self, pack_dependent: bool) -> None:
        self.adapter = SimpleNamespace(
            rollout=lambda: None, requires_preserved_replay_batch_composition=pack_dependent
        )
        self.training_args = SimpleNamespace(
            reward_optimization_overlap=False, num_batches_per_epoch=2
        )
        self.dataloader = [{"pack": 0}, {"pack": 1}]
        self.epoch = 0
        self.show_progress_bar = False
        self.train_dataloaders_by_source = {"default": self.dataloader}

    @contextlib.contextmanager
    def _rollout_acceleration(self) -> Iterator[None]:
        yield

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        yield

    def sample_batch(self, batch: dict, **_: Any) -> List[SimpleNamespace]:
        return [SimpleNamespace(unique_id=batch["pack"], source=None) for _ in range(2)]


@pytest.mark.parametrize("pack_dependent", [False, True])
def test_generation_records_rollout_packs_for_pack_dependent_adapters(
    pack_dependent: bool,
) -> None:
    harness = _GenerationHarness(pack_dependent)

    samples = BaseTrainer.generate_samples(harness)

    assert harness._reward_overlap_acquisition_manifest is None
    if pack_dependent:
        assert harness._rollout_acquisition_manifest.rollout_batches == ((0, 1), (2, 3))
        harness._rollout_acquisition_manifest.validate_samples(samples)
    else:
        assert harness._rollout_acquisition_manifest is None
