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


"""Passive observations of NFT's sampling-policy and full-output objective paths."""

from contextlib import contextmanager

import torch
from native_observation_utils import cpu
from nft_objective_reference import assert_objective

from flow_factory.trainers.rl import nft


@contextmanager
def observe_objectives(trainer):
    """Bind each real policy/reference query to its precomputed noised state."""
    originals = {
        name: getattr(trainer, name)
        for name in (
            "_precompute_sampling_policy_steps",
            "_matching_losses",
            "_velocity_kl",
            "_backward_full_output_loss",
        )
    }
    forward = nft.forward_velocity_state
    records, prepared = [], {}
    current = {}

    def precompute(batch, clean, size, *args, **kwargs):
        steps = originals["_precompute_sampling_policy_steps"](batch, clean, size, *args, **kwargs)
        prepared.clear()
        for step in steps:
            prepared[id(step.noised.state)] = (batch, clean, step)
            assert all(not value.requires_grad for value in step.velocity.components.values())
        return steps

    def observed_forward(owner, batch, state, times, *, source, **kwargs):
        if owner is not trainer:
            return forward(owner, batch, state, times, source=source, **kwargs)
        if source in ("policy", "reference"):
            expected_batch, clean, step = prepared[id(state)]
            assert batch is expected_batch and times is step.times
            assert state is step.noised.state
        result = forward(owner, batch, state, times, source=source, **kwargs)
        if source == "policy":
            current.clear()
            current.update(batch=batch, step=step, clean=clean)
        elif source == "reference":
            assert not torch.is_grad_enabled()
            records[-1]["reference"] = cpu(result.components)
        return result

    def matching(clean, noised, times, new, old):
        assert noised is current["step"].noised and times is current["step"].times
        assert old is current["step"].velocity and clean is current["clean"]
        args = trainer.training_args
        row = {
            "clean": cpu(clean.components),
            "state": cpu(noised.state.components),
            "sigma": cpu(times.sigma),
            "new": cpu(new.components),
            "old": cpu(old.components),
            "noise": cpu(noised.noise.components),
            "masks": cpu(noised.state.active_masks or {}),
            "velocity_direction": trainer.adapter.flow_velocity_direction,
            "advantage": cpu(current["batch"]["advantage"]),
            "adv_clip_range": args.adv_clip_range,
            "nft_beta": trainer.nft_beta,
            "kl_beta": args.kl_beta,
            "cp_size": trainer.config.parallelism_args.cp_size,
            "gas": trainer.accelerator.gradient_accumulation_steps,
            "off_policy": trainer.off_policy,
            "gradients": {},
            "same_noised_inputs": True,
        }
        for name, value in new.components.items():
            assert value.requires_grad
            value.register_hook(
                lambda gradient, name=name, row=row: row["gradients"].__setitem__(
                    name, cpu(gradient)
                )
            )
        records.append(row)
        return originals["_matching_losses"](clean, noised, times, new, old)

    def velocity_kl(*args, **kwargs):
        result = originals["_velocity_kl"](*args, **kwargs)
        # CUDA autocast promotes the squared KL residual to FP32 even when the
        # exposed velocity and its subtraction remain BF16.
        records[-1]["kl_dtype"] = result.dtype
        return result

    def backward(loss, **kwargs):
        records[-1]["loss"] = cpu(loss)
        returned = originals["_backward_full_output_loss"](loss, **kwargs)
        records[-1]["oracle"] = assert_objective(records[-1])
        return returned

    trainer._precompute_sampling_policy_steps = precompute
    trainer._matching_losses = matching
    trainer._velocity_kl = velocity_kl
    trainer._backward_full_output_loss = backward
    nft.forward_velocity_state = observed_forward
    try:
        yield records
    finally:
        nft.forward_velocity_state = forward
        for name, original in originals.items():
            setattr(trainer, name, original)


def assert_objectives(records, cp_size):
    assert records
    for row in records:
        assert row["cp_size"] == cp_size and row["same_noised_inputs"]
        assert "oracle" in row
        assert_objective(row)
    assert any(
        bool((row["oracle"]["gradients"][name].abs() > 4 * budget).any())
        for row in records
        for name, budget in row["oracle"]["budgets"].items()
    )


@contextmanager
def observe_training_terms(trainer, terms):
    """Record the actual current-policy terms, not retained transition log-probs."""
    precompute, matching = trainer._precompute_sampling_policy_steps, trainer._matching_losses
    identity = None

    def observed_precompute(batch, *args, **kwargs):
        nonlocal identity
        assert len(batch.samples) == 1
        sample = batch.samples[0]
        identity = (sample.source, int(sample.sampling_sample_id))
        return precompute(batch, *args, **kwargs)

    def observed_matching(clean, noised, times, new, old):
        terms.append(
            (
                identity,
                float(times.timestep["latent"].reshape(-1)[0]),
                trainer.accelerator.gradient_accumulation_steps,
            )
        )
        return matching(clean, noised, times, new, old)

    trainer._precompute_sampling_policy_steps = observed_precompute
    trainer._matching_losses = observed_matching
    try:
        yield
    finally:
        trainer._precompute_sampling_policy_steps, trainer._matching_losses = precompute, matching


def audit_samples(trainer, samples):
    """NFT consumes one terminal state per sample and does not require transition replay."""
    rows = []
    for batch in trainer._iter_prefetched_batches(samples, 1):
        state = trainer.adapter.get_terminal_state(batch)
        assert all(torch.isfinite(value).all() for value in state.components.values())
        sample = batch.samples[0]
        rows.append(((sample.source, int(sample.sampling_sample_id)), 0))
    return rows
