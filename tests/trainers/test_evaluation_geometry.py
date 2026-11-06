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

"""Evaluation keeps padded model collectives but scores each dataset row once."""

import math
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler


def _loader(total, batch_size, replicas, rank, **overrides):
    dataset = list(range(total))
    sampler = BatchSampler(SequentialSampler(dataset), batch_size, drop_last=False)
    if replicas > 1:
        sampler = BatchSamplerShard(
            sampler, num_processes=replicas, process_index=rank, **overrides
        )
    return DataLoader(dataset, batch_sampler=sampler)


@pytest.mark.parametrize("total", [0, 1, 3, 9])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("replicas", [1, 2, 4])
def test_official_eval_padding_never_becomes_a_scored_row(monkeypatch, total, batch_size, replicas):
    from flow_factory.trainers.common import evaluation

    real_rows, batches_per_rank = [], []
    for rank in range(replicas):
        monkeypatch.setattr(evaluation, "data_parallel_geometry", lambda _: (replicas, rank))
        loader = _loader(total, batch_size, replicas, rank)
        plan = evaluation.EvaluationBatchPlan.from_loader(loader, object())
        batches = list(loader)
        batches_per_rank.append(len(batches))
        for index, rows in enumerate(batches):
            real_rows.extend(rows[: plan.valid_count(index)].tolist())
    assert sorted(real_rows) == list(range(total))
    assert len(set(batches_per_rank)) == 1


@pytest.mark.parametrize("override", [{"even_batches": False}, {"split_batches": True}])
def test_eval_rejects_unqualified_shard_geometry_before_generation(monkeypatch, override):
    from flow_factory.trainers.common import evaluation

    monkeypatch.setattr(evaluation, "data_parallel_geometry", lambda _: (2, 0))
    loader = _loader(5, 2, 2, 0, **override)
    with pytest.raises(ValueError, match="evaluation"):
        evaluation.EvaluationBatchPlan.from_loader(loader, object())


def test_eval_rejects_physical_world_sharding_of_logical_samples(monkeypatch):
    from flow_factory.trainers.common import evaluation

    monkeypatch.setattr(evaluation, "data_parallel_geometry", lambda _: (2, 0))
    loader = _loader(7, 1, 4, 0)
    with pytest.raises(ValueError, match="evaluation"):
        evaluation.EvaluationBatchPlan.from_loader(loader, object())


def test_eval_requires_the_existing_sequential_loader(monkeypatch):
    from flow_factory.trainers.common import evaluation

    monkeypatch.setattr(evaluation, "data_parallel_geometry", lambda _: (1, 0))
    dataset = list(range(5))
    loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=2)
    with pytest.raises(ValueError, match="evaluation"):
        evaluation.EvaluationBatchPlan.from_loader(loader, object())


@pytest.mark.parametrize("values", [[], [3.0], [-2.0, 1.0, 7.0], [1e12, 1e12 + 2, 1e12 + 4]])
def test_eval_summary_matches_independent_float64_statistics(values):
    from flow_factory.trainers.common.evaluation import summarize_evaluation_rewards

    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1)
    tensor = torch.tensor(values, dtype=torch.float64)
    result = summarize_evaluation_rewards(accelerator, {"score": tensor})["score"]
    assert result["count"] == len(values)
    if not values:
        assert all(math.isnan(result[key]) for key in ("mean", "std", "min", "max"))
    else:
        assert result["mean"] == pytest.approx(sum(values) / len(values), rel=0, abs=1e-12)
        assert result["std"] == pytest.approx(float(tensor.std(correction=0)), rel=1e-12)
        assert result["min"] == min(values)
        assert result["max"] == max(values)
