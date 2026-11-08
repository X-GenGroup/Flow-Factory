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


"""Passive AWM matching, snapshot and noised-input observations."""

from contextlib import contextmanager

import torch
from awm_objective_reference import assert_objective
from native_observation_utils import cpu

from flow_factory.trainers.rl import awm


def _input_values(state, times):
    """Snapshot operands before any model query can mutate them in place."""
    return {
        "state": cpu(state.components),
        "masks": cpu(state.active_masks or {}),
        "times": {
            key: cpu(getattr(times, key))
            for key in ("timestep", "next_timestep", "sigma", "next_sigma")
        },
    }


def _assert_input_values(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), "AWM input fields changed"
        for key in expected:
            _assert_input_values(expected[key], actual[key])
    elif isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype and torch.equal(
            expected, actual
        ), "AWM noised input changed in place"
    else:
        assert expected == actual, "AWM input metadata changed"


@contextmanager
def observe_objectives(trainer):
    originals = {
        name: getattr(trainer, name)
        for name in (
            "_precompute_old_log_probs",
            "_matching_log_prob",
            "_velocity_kl",
            "_backward_full_output_loss",
        )
    }
    forward = awm.forward_velocity_state
    records, prepared, old_values = [], {}, {}
    current = {}

    def precompute(batch, clean, size, *args, **kwargs):
        old_values.clear()
        steps = originals["_precompute_old_log_probs"](batch, clean, size, *args, **kwargs)
        prepared.clear()
        for step in steps:
            assert not step.log_prob.requires_grad
            prepared[id(step.noised.state)] = (
                batch,
                clean,
                step,
                *old_values[id(step.noised.state)],
            )
        return steps

    def observed_forward(owner, batch, state, times, *, source, **kwargs):
        if owner is not trainer:
            return forward(owner, batch, state, times, source=source, **kwargs)
        if source != "sampling policy":
            expected_batch, clean, step, old, expected_inputs = prepared[id(state)]
            assert batch is expected_batch and times is step.times and state is step.noised.state
        else:
            expected_inputs = _input_values(state, times)
        _assert_input_values(expected_inputs, _input_values(state, times))
        result = forward(owner, batch, state, times, source=source, **kwargs)
        _assert_input_values(expected_inputs, _input_values(state, times))
        if source == "sampling policy":
            assert not torch.is_grad_enabled()
            assert all(not value.requires_grad for value in result.components.values())
            old_values[id(state)] = (cpu(result.components), expected_inputs)
        elif source == "policy":
            current.clear()
            current.update(batch=batch, clean=clean, step=step, old=old)
        elif source in ("reference", "EMA"):
            assert not torch.is_grad_enabled()
            assert all(not value.requires_grad for value in result.components.values())
            current["other"] = "reference" if source == "reference" else "ema"
            records[-1][current["other"]] = cpu(result.components)
        return result

    def matching(velocity, noised, times):
        result = originals["_matching_log_prob"](velocity, noised, times)
        if torch.is_grad_enabled():
            assert noised is current["step"].noised and times is current["step"].times
            args = trainer.training_args
            row = {
                "new": cpu(velocity.components),
                "old": current["old"],
                "target": cpu(noised.target_velocity.components),
                "state": cpu(noised.state.components),
                "noise": cpu(noised.noise.components),
                "sigma": cpu(times.sigma),
                "masks": cpu(noised.state.active_masks or {}),
                "advantage": cpu(current["batch"]["advantage"]),
                "adv_clip_range": args.adv_clip_range,
                "clip_range": args.clip_range,
                "weighting": trainer.weighting,
                "ghuber_power": trainer.ghuber_power,
                "kl_beta": trainer.kl_beta,
                "ema_kl_beta": trainer.ema_kl_beta,
                "cp_size": trainer.config.parallelism_args.cp_size,
                "gas": trainer.accelerator.gradient_accumulation_steps,
                "off_policy": trainer.off_policy,
                "old_log_prob": cpu(current["step"].log_prob),
                "log_prob": cpu(result),
                "same_noised_inputs": True,
                "gradients": {},
                "kl_dtypes": {},
            }
            for name, value in velocity.components.items():
                assert value.requires_grad
                value.register_hook(
                    lambda gradient, name=name, row=row: row["gradients"].__setitem__(
                        name, cpu(gradient)
                    )
                )
            records.append(row)
        return result

    def velocity_kl(*args, **kwargs):
        value = originals["_velocity_kl"](*args, **kwargs)
        records[-1]["kl_dtypes"][current["other"]] = value.dtype
        return value

    def backward(loss, **kwargs):
        records[-1]["loss"] = cpu(loss)
        value = originals["_backward_full_output_loss"](loss, **kwargs)
        records[-1]["oracle"] = assert_objective(records[-1])
        return value

    trainer._precompute_old_log_probs = precompute
    trainer._matching_log_prob = matching
    trainer._velocity_kl = velocity_kl
    trainer._backward_full_output_loss = backward
    awm.forward_velocity_state = observed_forward
    try:
        yield records
    finally:
        awm.forward_velocity_state = forward
        for name, original in originals.items():
            setattr(trainer, name, original)


def assert_objectives(records, cp_size):
    assert records
    for row in records:
        assert row["cp_size"] == cp_size and row["same_noised_inputs"]
        assert "oracle" in row
        assert_objective(row)
    assert any(
        bool((gradient.abs() > 4 * row["oracle"]["budgets"][name]).any())
        for row in records
        for name, gradient in row["oracle"]["gradients"].items()
    )
    for other, coefficient in (("reference", "kl_beta"), ("ema", "ema_kl_beta")):
        if any(row[coefficient] for row in records):
            assert any(
                not torch.equal(row["new"][name], row[other][name])
                for row in records
                if row[coefficient]
                for name in row["new"]
            ), other
    if any(row["off_policy"] for row in records):
        assert any(
            not torch.equal(row["new"][name], row["old"][name])
            for row in records
            for name in row["new"]
        )


@contextmanager
def observe_training_terms(trainer, terms):
    precompute, matching = trainer._precompute_old_log_probs, trainer._matching_log_prob
    identity = None

    def observed_precompute(batch, *args, **kwargs):
        nonlocal identity
        assert len(batch.samples) == 1
        sample = batch.samples[0]
        identity = (sample.source, int(sample.sampling_sample_id))
        return precompute(batch, *args, **kwargs)

    def observed_matching(velocity, noised, times):
        if torch.is_grad_enabled():
            terms.append(
                (
                    identity,
                    float(times.timestep["latent"].reshape(-1)[0]),
                    trainer.accelerator.gradient_accumulation_steps,
                )
            )
        return matching(velocity, noised, times)

    trainer._precompute_old_log_probs = observed_precompute
    trainer._matching_log_prob = observed_matching
    try:
        yield
    finally:
        trainer._precompute_old_log_probs, trainer._matching_log_prob = precompute, matching


def audit_samples(trainer, samples):
    """AWM consumes terminal states; matching passes reuse precomputed inputs."""
    rows = []
    for batch in trainer._iter_prefetched_batches(samples, 1):
        state = trainer.adapter.get_terminal_state(batch)
        assert all(torch.isfinite(value).all() for value in state.components.values())
        sample = batch.samples[0]
        rows.append(((sample.source, int(sample.sampling_sample_id)), 0))
    return rows
