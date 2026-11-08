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

"""CRD immutable prepass identities, failure atomicity and snapshot cadence."""

import copy
from types import SimpleNamespace

import pytest
import torch

from flow_factory.trainers.rl.crd import CRDTrainer
from tests.trainers.test_dgpo_crd_production_paths import (
    _adapter,
    _crd_trainer,
    _samples,
    _with_snapshot_storage,
)


@pytest.mark.parametrize("corruption", ["swap", "source", "sample", "missing", "shape", "extra"])
def test_overlap_rejects_late_invalid_feedback_before_any_attachment(corruption):
    trainer = _crd_trainer(_adapter(), per_device_batch_size=1)
    samples = _samples([1, 1, 2, 2], [0.1, 0.2, 0.3, 0.4])
    for index, sample in enumerate(samples):
        sample.source_id, sample.sampling_sample_id = 7, index
    prepared = trainer._precompute_crd_batches(samples)
    before = [batch.batch["advantage"].clone() for batch in prepared]
    if corruption == "swap":
        samples[2], samples[3] = samples[3], samples[2]
    elif corruption in ("source", "sample"):
        setattr(samples[-1], "source_id" if corruption == "source" else "sampling_sample_id", 99)
    elif corruption == "missing":
        del samples[-1].extra_kwargs["advantage"]
    elif corruption == "shape":
        samples[-1].extra_kwargs["advantage"] = torch.ones(2)
    else:
        samples.append(copy.deepcopy(samples[-1]))
    trainer.optimize = lambda _: pytest.fail("invalid feedback reached optimize")
    with pytest.raises((RuntimeError, ValueError)):
        trainer._optimize_reward_overlap_tile(SimpleNamespace(tile_id=0), samples, {0: prepared})
    for expected, batch in zip(before, prepared, strict=True):
        assert torch.equal(expected, batch.batch["advantage"])
    assert not hasattr(trainer, "_crd_overlap_precomputed_batches")


def test_overlap_override_is_cleaned_after_optimizer_failure():
    trainer = _crd_trainer(_adapter())
    samples = _samples([1, 1], [0.1, 0.2])
    prepared = trainer._precompute_crd_batches(samples)

    def fail(_):
        assert trainer._crd_overlap_precomputed_batches is prepared
        raise RuntimeError("optimizer failed")

    trainer.optimize = fail
    with pytest.raises(RuntimeError, match="optimizer failed"):
        trainer._optimize_reward_overlap_tile(SimpleNamespace(tile_id=0), samples, {0: prepared})
    assert not hasattr(trainer, "_crd_overlap_precomputed_batches")


@pytest.mark.parametrize("use_old", [False, True])
@pytest.mark.parametrize("inner_epochs", [1, 2])
def test_pass_inputs_values_and_every_sample_time_survive_earlier_updates(use_old, inner_epochs):
    adapter = _adapter()
    trainer = _crd_trainer(
        adapter,
        use_old_for_loss=use_old,
        per_device_batch_size=1,
        num_train_timesteps=2,
        num_inner_epochs=inner_epochs,
        kl_cfg=4.5,
    )
    samples = _samples([1, 1, 2, 2], [0.1, 0.2, 0.3, 0.4])
    for index, sample in enumerate(samples):
        sample.source_id, sample.sampling_sample_id = 7, index
    records, expected, rebuilt = [], {}, []
    original_precompute, original_rebuild = (
        trainer._precompute_old_velocities,
        trainer._rebuild_noised_state,
    )

    def precompute(batch, *args):
        result = original_precompute(batch, *args)
        for step in result.steps:
            expected[id(step)] = copy.deepcopy((step.times, step.noise, step.old_velocity))
        records.append(result)
        return result

    def rebuild(clean, step, **kwargs):
        times, noise, old = expected[id(step)]
        for name in noise.components:
            assert torch.equal(noise.components[name], step.noise.components[name])
            assert torch.equal(old.components[name], step.old_velocity.components[name])
            assert torch.equal(times.timestep[name], step.times.timestep[name])
        result = original_rebuild(clean, step, **kwargs)
        rebuilt.append(id(step))
        return result

    trainer._precompute_old_velocities, trainer._rebuild_noised_state = precompute, rebuild
    trainer.optimize(samples)
    assert len(records) == 4 * inner_epochs and len(rebuilt) == 8 * inner_epochs
    assert len(set(rebuilt)) == len(rebuilt)
    scopes = [call["scope"] for call in adapter.forward_calls]
    for start in range(0, len(scopes), 24):
        assert scopes[start : start + 8] == [CRDTrainer._OLD_PARAMS_NAME if use_old else "ref"] * 8
    old_calls = [call for index, call in enumerate(adapter.forward_calls) if index % 24 < 8]
    policy_calls = [call for call in adapter.forward_calls if call["scope"] == "policy"]
    for old, policy in zip(old_calls, policy_calls, strict=True):
        assert torch.equal(old["latents"], policy["latents"])
        assert torch.equal(old["t"], policy["t"])
    assert all(
        call["guidance_scale"] == 4.5
        for index, call in enumerate(adapter.forward_calls)
        if index % 24 >= 8 and call["scope"] == "ref"
    )


def test_distinct_snapshot_schedules_and_noop_copy_ema_branches():
    adapter = _adapter()
    copied = _with_snapshot_storage(adapter)
    trainer = _crd_trainer(adapter)
    trainer.old_model_decay, trainer.sampling_model_decay = "0-0.25-0.05-0.6", "0-0.1-0.03-0.5"
    for step in (2, 4):
        trainer.step = step
        before = {name: value[0].clone() for name, value in adapter.named_snapshots.items()}
        with torch.no_grad():
            adapter.weight.add_(0.1)
        trainer._after_acquisition_cycle()
        for name, decay in (
            (trainer._OLD_PARAMS_NAME, min(0.25 + step * 0.05, 0.6)),
            (trainer._SAMPLING_PARAMS_NAME, min(0.1 + step * 0.03, 0.5)),
        ):
            torch.testing.assert_close(
                adapter.named_snapshots[name][0],
                before[name] * decay + adapter.weight.detach() * (1 - decay),
            )
    before = adapter.named_snapshots[trainer._OLD_PARAMS_NAME][0].clone()
    trainer._blend_named_params(trainer._OLD_PARAMS_NAME, 1)
    assert torch.equal(before, adapter.named_snapshots[trainer._OLD_PARAMS_NAME][0])
    trainer._blend_named_params(trainer._SAMPLING_PARAMS_NAME, 0)
    assert copied == [trainer._SAMPLING_PARAMS_NAME]
