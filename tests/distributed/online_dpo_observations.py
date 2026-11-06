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

"""Observe online pair ownership and reuse the independently checked preference calculus."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from native_observation_utils import cpu
from offline_dpo_observations import assert_preference_rows

from flow_factory.trainers.rl import dpo as implementation


def identity(sample):
    return (sample.source, int(sample.sampling_sample_id))


def expected_pairs(samples):
    """Independent complete-group expectation, including source and stable tie ordering."""
    groups = {}
    for sample in samples:
        key = (sample.source_id, sample.unique_id)
        groups.setdefault(key, []).append(sample)
    result = []
    for members in groups.values():
        if len(members) < 2:
            continue
        chosen = max(members, key=lambda sample: float(sample["advantage"]))
        rejected = min(members, key=lambda sample: float(sample["advantage"]))
        result.append((identity(chosen), identity(rejected)))
    return sorted(result, key=repr)


def training_term_identities(samples):
    return [chosen for chosen, rejected in expected_pairs(samples)]


@contextmanager
def observe_objectives(trainer):
    rows, forwards, errors = [], [], []
    forward = implementation.forward_velocity_state
    mse, preference, pair = (
        trainer._arm_velocity_error,
        trainer._preference_loss,
        trainer._form_pairs,
    )

    def observed_forward(owner, batch, state, times, *, source, **kwargs):
        from test_context_parallel_training import _compare

        before_state, before_times = cpu(state), cpu(times)
        result = forward(owner, batch, state, times, source=source, **kwargs)
        _compare(cpu(state), before_state, exact=True)
        _compare(cpu(times), before_times, exact=True)
        forwards.append(
            {
                "source": source,
                "ids": [identity(s) for s in batch.samples],
                "times": cpu(times),
                "state": cpu(state),
            }
        )
        return result

    def observed_mse(prediction, noised):
        result = mse(prediction, noised)
        assert len(prediction.components) == 1 and prediction.active_masks is None
        tensor = next(iter(prediction.components.values()))
        row = {
            "prediction": cpu(tensor),
            "target": cpu(next(iter(noised.target_velocity.components.values()))),
            "mse": cpu(result),
            "requires_grad": tensor.requires_grad,
            "noise": cpu(noised.noise),
            "state": cpu(noised.state),
        }
        if tensor.requires_grad:
            tensor.register_hook(lambda gradient: row.update(gradient=cpu(gradient)))
        errors.append(row)
        return result

    def observed_preference(*args):
        loss, metrics = preference(*args)
        assert len(forwards) == 4 and len(errors) == 4
        rows.append(
            {
                "terms": list(errors),
                "forwards": list(forwards),
                "noising": [
                    {
                        "times": forwards[i]["times"],
                        "noise": errors[i]["noise"],
                        "state": errors[i]["state"],
                    }
                    for i in (0, 1)
                ],
                "loss": cpu(loss),
                "beta": trainer.training_args.beta,
                "gas": trainer.accelerator.gradient_accumulation_steps,
            }
        )
        forwards.clear()
        errors.clear()
        return loss, metrics

    def observed_pairs(samples):
        pairs, metrics = pair(samples)
        if trainer.advantage_processor.group_on_same_rank:
            actual = sorted([(identity(a), identity(b)) for a, b in pairs], key=repr)
            assert actual == expected_pairs(samples)
            assert (
                metrics["train/dpo_num_pairs"]
                == len(pairs) * trainer.group_coordinator.dp_world_size
            )
        return pairs, metrics

    with (
        patch.object(implementation, "forward_velocity_state", observed_forward),
        patch.object(trainer, "_arm_velocity_error", observed_mse),
        patch.object(trainer, "_preference_loss", observed_preference),
        patch.object(trainer, "_form_pairs", observed_pairs),
    ):
        yield rows
    assert not forwards and not errors


def assert_objectives(rows, cp_size):
    from test_context_parallel_training import _compare

    assert rows
    for row in rows:
        # Online DPO accumulates each timestep directly; GAS owns its normalization.
        assert_preference_rows([row], cp_size=cp_size, gas=row["gas"], timesteps=1)
        forwards = row["forwards"]
        assert [f["source"] for f in forwards] == [
            "policy chosen",
            "policy rejected",
            "reference chosen",
            "reference rejected",
        ]
        assert forwards[0]["ids"] == forwards[2]["ids"]
        assert forwards[1]["ids"] == forwards[3]["ids"]
        assert forwards[0]["ids"] != forwards[1]["ids"]
        for arm in (0, 1):
            _compare(forwards[arm]["state"], forwards[arm + 2]["state"], exact=True)
            _compare(forwards[arm]["times"], forwards[arm + 2]["times"], exact=True)
            _compare(row["terms"][arm]["noise"], row["terms"][arm + 2]["noise"], exact=True)


@contextmanager
def observe_training_terms(trainer, terms):
    forward = implementation.forward_velocity_state

    def observed_forward(owner, batch, state, times, *, source, **kwargs):
        if source == "policy chosen":
            assert len(batch.samples) == 1
            terms.append(
                (
                    identity(batch.samples[0]),
                    float(next(iter(times.timestep.values())).reshape(-1)[0]),
                    trainer.accelerator.gradient_accumulation_steps,
                )
            )
        return forward(owner, batch, state, times, source=source, **kwargs)

    with patch.object(implementation, "forward_velocity_state", observed_forward):
        yield


def audit_samples(trainer, samples):
    rows = []
    for batch in trainer._iter_prefetched_batches(samples, 1):
        state = trainer.adapter.get_terminal_state(batch)
        assert all(torch.isfinite(value).all() for value in state.components.values())
        rows.append((identity(batch.samples[0]), 0))
    return rows
