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

"""Batch geometry counts data-parallel replicas, not launched ranks.

Every divisibility rule in ``Arguments`` is written against the number of
replicas that tile the batch. Without tensor parallelism that is the launched
world size, so the two were interchangeable and the code read the launched
world. Under tensor parallelism the ranks of one group consume the *same*
sample, so a group is one replica and the world is ``replicas * tp_size``.

Reading the launched world there inflates the global batch by ``tp_size``,
which makes ``num_batches_per_epoch`` that factor too small and -- in ``auto``
mode -- ``gradient_accumulation_steps`` with it. That silently breaks the
one-optimizer-update-per-epoch geometry the alignment exists to produce, and it
does so without any error, which is why it is pinned here.

The central assertion is an equivalence: a run at ``world=8, tp=2`` has the
same four replicas as a run at ``world=4, tp=1``, so it must derive the same
batch geometry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from flow_factory.hparams import Arguments


def _parse(
    world_size: int,
    tensor_parallel_size: Optional[int],
    monkeypatch: pytest.MonkeyPatch,
    **train: Any,
) -> Arguments:
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    config: Dict[str, Any] = {
        "train": {
            "trainer_type": "grpo",
            "num_inference_steps": 4,
            "unique_sample_num_per_epoch": 16,
            "group_size": 4,
            "per_device_batch_size": 1,
            **train,
        },
        "rewards": [{"name": "score", "reward_model": "clip"}],
    }
    if tensor_parallel_size is not None:
        config["parallelism"] = {"tp_size": tensor_parallel_size}
    return Arguments.from_dict(config)


def test_a_group_of_ranks_counts_as_one_replica(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eight ranks in groups of two tile the batch as four replicas."""
    config = _parse(8, 2, monkeypatch)

    assert config.data_parallel_world_size() == 4


def test_absent_parallelism_block_counts_every_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the block every rank is its own replica, as before."""
    config = _parse(8, None, monkeypatch)

    assert config.data_parallel_world_size() == 8


def test_tensor_parallelism_does_not_change_the_derived_batch_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``world=8, tp=2`` derives the same geometry as ``world=4, tp=1``.

    Tensor parallelism splits each replica's model across more ranks; it does
    not add replicas. So every quantity derived from the replica count has to be
    unchanged, and in particular the optimizer still steps the same number of
    times per epoch.
    """
    # Read beside each parse: `data_parallel_world_size` consults `WORLD_SIZE`
    # on every call, and the next `_parse` rewrites it.
    composed = _parse(8, 2, monkeypatch)
    composed_replicas = composed.data_parallel_world_size()
    plain = _parse(4, None, monkeypatch)
    plain_replicas = plain.data_parallel_world_size()

    assert composed_replicas == plain_replicas == 4
    assert (
        composed.training_args.unique_sample_num_per_epoch
        == plain.training_args.unique_sample_num_per_epoch
    )
    assert composed.training_args.num_batches_per_epoch == (
        plain.training_args.num_batches_per_epoch
    )
    assert composed.training_args.gradient_accumulation_steps == (
        plain.training_args.gradient_accumulation_steps
    )


def test_counting_launched_ranks_would_have_shrunk_the_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show the error the equivalence above rules out.

    A run at ``world=8`` with no tensor parallelism genuinely has eight
    replicas, so it accumulates over half as many microbatches as four replicas
    do. That is the value ``world=8, tp=2`` would have taken if the geometry had
    counted ranks, and it is not the same number -- so the equivalence test
    above is not vacuous.
    """
    composed = _parse(8, 2, monkeypatch)
    eight_replicas = _parse(8, None, monkeypatch)

    assert eight_replicas.training_args.num_batches_per_epoch == (
        composed.training_args.num_batches_per_epoch // 2
    )


@pytest.mark.parametrize("trainer_type", ["grpo", "sft"])
def test_an_unhostable_layout_is_rejected_for_every_acquisition_mode(
    monkeypatch: pytest.MonkeyPatch, trainer_type: str
) -> None:
    """Dataset acquisition is checked too, though it skips batch alignment.

    ``sft`` returns from ``__post_init__`` before the geometry is aligned, so a
    check placed inside the alignment path would accept a layout no launch can
    satisfy and fail much later, inside mesh construction.
    """
    monkeypatch.setenv("WORLD_SIZE", "6")
    config: Dict[str, Any] = {
        "train": {"trainer_type": trainer_type, "gradient_accumulation_steps": 1},
        "parallelism": {"tp_size": 4},
    }
    if trainer_type == "grpo":
        config["train"].update(
            num_inference_steps=4,
            unique_sample_num_per_epoch=16,
            group_size=4,
            per_device_batch_size=1,
        )
        config["train"].pop("gradient_accumulation_steps")
        config["rewards"] = [{"name": "score", "reward_model": "clip"}]

    # The layout check has to precede the acquisition-specific validation, or
    # this would surface as a complaint about the (deliberately absent) dataset.
    with pytest.raises(ValueError, match="must divide the launched world size 6"):
        Arguments.from_dict(config)
