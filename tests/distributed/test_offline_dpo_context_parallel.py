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

"""Offline preference CP admission, pair calculus and native continuation."""

import pytest
import torch

from flow_factory.trainers.common.context_parallel import (
    context_parallel_training_capability,
)
from flow_factory.trainers.offline.offline_dpo import OfflineDPOTrainer


def test_offline_dpo_declares_its_own_full_output_capability():
    capability = context_parallel_training_capability(OfflineDPOTrainer)
    assert capability.loss_gradient == "full_output"
    assert capability.checkpoint_boundary == "drained_acquisition"

    class UnqualifiedPreferenceTrainer(OfflineDPOTrainer):
        pass

    with pytest.raises(ValueError, match="does not explicitly declare"):
        context_parallel_training_capability(UnqualifiedPreferenceTrainer)


@pytest.mark.parametrize("model,mode,tp_size", [("flux1", "full", 1), ("wan2_t2v", "lora", 2)])
def test_cuda_offline_dpo_cp_pairs_updates_and_resume(tmp_path, model, mode, tp_size):
    if torch.cuda.device_count() < 4 * tp_size:
        pytest.skip(f"requires {4 * tp_size} local CUDA ranks")
    from context_parallel_offline_runner import qualify_offline

    qualify_offline(tmp_path, model=model, mode=mode, tp_size=tp_size)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_real_offline_objective_matches_pair_derivative_oracle(monkeypatch, cp_size):
    from offline_dpo_observations import assert_preference_rows, observe_preferences

    from flow_factory.trainers.common import context_parallel
    from tests.trainers import test_offline_trainers as fixture

    trainer, adapter, _ = fixture._trainer(OfflineDPOTrainer, [False, True])
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    original = trainer.accelerator.backward
    trainer.accelerator.backward = lambda loss: original(loss / 2)
    with observe_preferences(trainer) as rows:
        trainer.optimize_batch(fixture._batch("preference"))
        trainer.optimize_batch(fixture._batch("preference"))
    assert len(rows) == 4
    assert_preference_rows(rows, cp_size=cp_size)
    assert adapter.policy_weight.grad is not None
    assert torch.isfinite(adapter.policy_weight.grad)


@pytest.mark.parametrize("corruption", ["cp_factor", "arm", "reference_gradient", "noise"])
def test_preference_oracle_rejects_corruption(monkeypatch, corruption):
    import copy

    from offline_dpo_observations import assert_preference_rows, observe_preferences

    from flow_factory.trainers.common import context_parallel
    from tests.trainers import test_offline_trainers as fixture

    trainer, _, _ = fixture._trainer(OfflineDPOTrainer, [True])
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    original = trainer.accelerator.backward
    trainer.accelerator.backward = lambda loss: original(loss / 2)
    with observe_preferences(trainer) as observations:
        trainer.optimize_batch(fixture._batch("preference"))
    rows = copy.deepcopy(observations)
    row = rows[0]
    if corruption == "cp_factor":
        row["terms"][0]["gradient"].mul_(0.5)
    elif corruption == "arm":
        row["terms"][0], row["terms"][1] = row["terms"][1], row["terms"][0]
    elif corruption == "reference_gradient":
        row["terms"][2]["requires_grad"] = True
    else:
        next(iter(row["noising"][1]["noise"]["components"].values())).add_(1)
    with pytest.raises(AssertionError):
        assert_preference_rows(rows, cp_size=2)
