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

"""Passive CRD two-pass, logical-centering and named-snapshot observations."""

from collections import Counter
from contextlib import contextmanager

import torch
import torch.distributed as dist
from crd_objective_reference import assert_objective
from native_observation_utils import cpu

from flow_factory.trainers.rl import crd


def _equal(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), "CRD input fields changed"
        for key in expected:
            _equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _equal(left, right)
    elif isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype and torch.equal(
            expected, actual
        ), "CRD input value changed"
    else:
        assert expected == actual


def _conditioning(batch):
    return cpu({key: value for key, value in batch.items() if key != "advantage"})


def _same_parameters(left, right):
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )


@contextmanager
def observe_objectives(trainer):
    names = (
        "_precompute_crd_batches",
        "_precompute_old_velocities",
        "_rebuild_noised_state",
        "_implicit_reward",
        "_compute_crd_loss",
        "_velocity_kl",
        "_backward_full_output_loss",
        "_blend_named_params",
        "sampling_context",
    )
    originals = {name: getattr(trainer, name) for name in names}
    forward = crd.forward_velocity_state
    add_noise = trainer.adapter.add_forward_process_noise
    records, prepared, active, old_queries = [], {}, {}, []
    lifecycle = dict(prepasses=[], blends=[], sampling=[], initial_step=trainer.step)
    frozen = cpu(trainer.adapter._ref_ema.ema_parameters)
    cycle_snapshots = {}
    expected_counts, consumed = Counter(), Counter()

    def snapshots():
        return {
            name: cpu(trainer.adapter.get_named_parameters(name))
            for name in trainer.runtime_child_names
        }

    def precompute(samples, *args, **kwargs):
        active["pass_one_complete"] = False
        before = trainer.step
        cycle_snapshots.clear()
        cycle_snapshots.update(snapshots())
        result = originals["_precompute_crd_batches"](samples, *args, **kwargs)
        assert trainer.step == before
        _equal(cycle_snapshots, snapshots())
        lifecycle["prepasses"].append(
            dict(
                cycle=trainer.cycle_index,
                step=before,
                batches=len(result),
                terms=sum(len(batch.steps) for batch in result),
            )
        )
        active["pass_one_complete"] = True
        return result

    def observed_noise(clean, times, **kwargs):
        result = add_noise(clean, times, **kwargs)
        active["prepass_noised"] = dict(
            clean=cpu(clean), target=cpu(result.target_velocity), noise=cpu(result.noise)
        )
        return result

    def precompute_batch(batch, *args, **kwargs):
        start = len(old_queries)
        conditioning = _conditioning(batch)
        result = originals["_precompute_old_velocities"](batch, *args, **kwargs)
        queries = old_queries[start:]
        assert len(result.steps) == len(queries) == trainer.num_train_timesteps
        for index, (step, query) in enumerate(zip(result.steps, queries, strict=True)):
            identity = tuple(
                (sample.source_id, int(sample.sampling_sample_id)) for sample in batch.samples
            )
            key = (trainer.cycle_index, identity, index)
            expected_counts[key] += trainer.training_args.num_inner_epochs
            prepared[id(step)] = dict(
                batch=batch,
                conditioning=conditioning,
                times=cpu(step.times),
                noise=cpu(step.noise),
                old=cpu(step.old_velocity),
                query=query,
                key=key,
                initial_step=trainer.step,
            )
            _equal(query["times"], cpu(step.times))
            _equal(query["old"], cpu(step.old_velocity))
        return result

    def rebuild(clean, step, **kwargs):
        record = prepared[id(step)]
        assert active.get("pass_one_complete")
        _equal(record["conditioning"], _conditioning(record["batch"]))
        _equal(record["times"], cpu(step.times))
        _equal(record["noise"], cpu(step.noise))
        _equal(record["old"], cpu(step.old_velocity))
        _equal(cycle_snapshots, snapshots())
        _equal(record["query"]["clean"], cpu(clean))
        result = originals["_rebuild_noised_state"](clean, step, **kwargs)
        _equal(record["query"]["target"], cpu(result.target_velocity))
        _equal(record["query"]["noise"], cpu(result.noise))
        _equal(record["query"]["state"], cpu(result.state))
        active.update(prepared=record, noised=result, step=step)
        consumed[record["key"]] += 1
        return result

    def observed_forward(owner, batch, state, times, *, source, **kwargs):
        if owner is not trainer:
            return forward(owner, batch, state, times, source=source, **kwargs)
        before = dict(state=cpu(state), times=cpu(times), conditioning=_conditioning(batch))
        if source == "old policy":
            assert not torch.is_grad_enabled()
            if trainer.use_old_for_loss:
                _equal(
                    cpu(trainer.adapter.get_named_parameters(trainer._OLD_PARAMS_NAME)),
                    cpu(trainer.adapter._ema_tracked_parameters()),
                )
        else:
            record = active["prepared"]
            assert batch is record["batch"] and times is active["step"].times
            _equal(record["query"]["state"], cpu(state))
        if source == "reference":
            _equal(frozen, cpu(trainer.adapter._ema_tracked_parameters()))
        result = forward(owner, batch, state, times, source=source, **kwargs)
        _equal(before, dict(state=cpu(state), times=cpu(times), conditioning=_conditioning(batch)))
        if source == "old policy":
            old_queries.append(
                dict(
                    state=cpu(state), times=cpu(times), old=cpu(result), **active["prepass_noised"]
                )
            )
        elif source == "policy":
            active["velocity"] = result
        elif source == "reference":
            assert not torch.is_grad_enabled()
            active["reference"] = result
            assert kwargs.get("guidance_scale") == trainer.kl_cfg
        if source != "policy":
            assert all(not value.requires_grad for value in result.components.values())
        return result

    def implicit(velocity, old, noised):
        result = originals["_implicit_reward"](velocity, old, noised)
        args, batch = trainer.training_args, active["prepared"]["batch"]
        adv = batch["advantage"].detach().clamp(*args.adv_clip_range)
        normalized = (adv / max(args.adv_clip_range) / 2 + 0.5).clamp(0, 1)
        row = dict(
            new=cpu(velocity.components),
            old=cpu(old.components),
            reference=cpu(active["reference"].components),
            target=cpu(noised.target_velocity.components),
            noise=cpu(noised.noise.components),
            masks=cpu(noised.state.active_masks or {}),
            advantage=cpu(normalized),
            identities=active["prepared"]["key"],
            adaptive_logp=trainer.adaptive_logp,
            weight_temp=trainer.weight_temp,
            crd_loss_type=trainer.crd_loss_type,
            crd_beta=trainer.crd_beta,
            adv_clip_max=args.adv_clip_range[1],
            reward_adaptive_kl=trainer.reward_adaptive_kl,
            kl_beta=trainer.kl_beta,
            kl_dtype=torch.float32,
            cp_size=trainer.config.parallelism_args.cp_size,
            gas=trainer.accelerator.gradient_accumulation_steps,
            gradients={},
            implicit_reward=cpu(result),
            same_noised_inputs=True,
        )
        for name, value in velocity.components.items():
            assert value.requires_grad
            value.register_hook(
                lambda gradient, name=name, row=row: row["gradients"].__setitem__(
                    name, cpu(gradient)
                )
            )
        _equal(frozen, cpu(trainer.adapter._ref_ema.ema_parameters))
        records.append(row)
        return result

    def centered(**kwargs):
        row = records[-1]
        payload = {key: value for key, value in row.items() if key != "gradients"}
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
        degree = trainer.config.parallelism_args.cp_size * trainer.config.parallelism_args.tp_size
        logical = gathered[dist.get_rank() % degree :: degree]
        expected_adv = torch.cat([peer["advantage"] for peer in logical])
        expected_reward = torch.cat([peer["implicit_reward"] for peer in logical])
        _equal(expected_adv, cpu(kwargs["adv_cur"]))
        _equal(expected_reward, cpu(kwargs["r_theta_gathered"]))
        row.update(
            logical_rows=logical, logical_replicas=len(logical), launched_ranks=len(gathered)
        )
        return originals["_compute_crd_loss"](**kwargs)

    def kl(velocity, reference, noised):
        result = originals["_velocity_kl"](velocity, reference, noised)
        if velocity is active["velocity"] and reference is active["reference"]:
            records[-1].update(kl_dtype=result.dtype, kl_div=cpu(result))
        return result

    def backward(loss, **kwargs):
        records[-1]["loss"] = cpu(loss)
        result = originals["_backward_full_output_loss"](loss, **kwargs)
        records[-1]["oracle"] = assert_objective(records[-1])
        return result

    def blend(name, decay):
        before, live = cpu(trainer.adapter.get_named_parameters(name)), cpu(
            trainer.adapter._ema_tracked_parameters()
        )
        schedule = (
            trainer.old_model_decay
            if name == trainer._OLD_PARAMS_NAME
            else trainer.sampling_model_decay
        )
        start, initial, slope, end = map(float, schedule.split("-"))
        expected_decay = (
            initial if trainer.step < start else min(initial + (trainer.step - start) * slope, end)
        )
        assert decay == expected_decay and expected_counts == consumed
        result = originals["_blend_named_params"](name, decay)
        after = cpu(trainer.adapter.get_named_parameters(name))
        maximum_error = 0.0
        for old, current, actual in zip(before, live, after, strict=True):
            expected = old.double() * decay + current.double() * (1 - decay)
            bound = (
                4 * torch.finfo(actual.dtype).eps * (old.double().abs() + current.double().abs())
                + torch.finfo(actual.dtype).tiny
            )
            error = (actual.double() - expected).abs()
            assert bool((error <= bound).all())
            maximum_error = max(maximum_error, float(error.max()))
        lifecycle["blends"].append(
            dict(
                name=name,
                step=trainer.step,
                decay=decay,
                changed=not _same_parameters(before, after),
                error=maximum_error,
            )
        )
        return result

    @contextmanager
    def sampling():
        before = cpu(trainer.adapter._ema_tracked_parameters())
        desired = cpu(trainer.adapter.get_named_parameters(trainer._SAMPLING_PARAMS_NAME))
        with originals["sampling_context"]():
            _equal(desired, cpu(trainer.adapter._ema_tracked_parameters()))
            yield
            _equal(desired, cpu(trainer.adapter._ema_tracked_parameters()))
        _equal(before, cpu(trainer.adapter._ema_tracked_parameters()))
        lifecycle["sampling"].append(
            dict(step=trainer.step, differs=not _same_parameters(before, desired))
        )

    replacements = dict(
        _precompute_crd_batches=precompute,
        _precompute_old_velocities=precompute_batch,
        _rebuild_noised_state=rebuild,
        _implicit_reward=implicit,
        _compute_crd_loss=centered,
        _velocity_kl=kl,
        _backward_full_output_loss=backward,
        _blend_named_params=blend,
        sampling_context=sampling,
    )
    for name, value in replacements.items():
        setattr(trainer, name, value)
    crd.forward_velocity_state = observed_forward
    trainer.adapter.add_forward_process_noise = observed_noise
    try:
        yield records
    finally:
        crd.forward_velocity_state = forward
        trainer.adapter.add_forward_process_noise = add_noise
        for name, value in originals.items():
            setattr(trainer, name, value)
        if records:
            lifecycle.update(
                expected=dict(expected_counts), consumed=dict(consumed), final_step=trainer.step
            )
            records[-1]["lifecycle"] = lifecycle


def assert_objectives(records, cp_size):
    assert records
    for row in records:
        assert row["cp_size"] == cp_size and row["same_noised_inputs"]
        assert row["logical_replicas"] == 2 and row["launched_ranks"] == 4
        assert_objective(row)
    assert any(bool((row["oracle"]["upstream"].abs() > 1e-8).any()) for row in records)
    assert any(float(row["oracle"]["kl_loss"]) > 1e-9 for row in records)
    lifecycle = records[-1]["lifecycle"]
    assert lifecycle["expected"] == lifecycle["consumed"]
    assert len(lifecycle["blends"]) == 2 * len(lifecycle["prepasses"])
    assert {event["name"] for event in lifecycle["blends"]} == {"_crd_old", "_crd_sampling"}
    assert all(event["changed"] for event in lifecycle["blends"])
    assert len({event["decay"] for event in lifecycle["blends"]}) > 1
    if len(lifecycle["prepasses"]) > 1:
        assert any(event["differs"] for event in lifecycle["sampling"])


@contextmanager
def observe_training_terms(trainer, terms):
    original = trainer._rebuild_noised_state
    forward = crd.forward_velocity_state
    # Rebuild identifies the step; the following policy query owns its sample.
    active_times = None

    def rebuild(clean, step, **kwargs):
        nonlocal active_times
        active_times = step.times
        return original(clean, step, **kwargs)

    def observed_forward(owner, batch, state, times, *, source, **kwargs):
        if owner is trainer and source == "policy":
            assert times is active_times and len(batch.samples) == 1
            sample = batch.samples[0]
            terms.append(
                (
                    (sample.source, int(sample.sampling_sample_id)),
                    float(times.timestep["latent"].reshape(-1)[0]),
                    trainer.accelerator.gradient_accumulation_steps,
                )
            )
        return forward(owner, batch, state, times, source=source, **kwargs)

    trainer._rebuild_noised_state = rebuild
    crd.forward_velocity_state = observed_forward
    try:
        yield
    finally:
        trainer._rebuild_noised_state = original
        crd.forward_velocity_state = forward


def audit_samples(trainer, samples):
    rows = []
    for batch in trainer._iter_prefetched_batches(samples, 1):
        state = trainer.adapter.get_terminal_state(batch)
        assert all(torch.isfinite(value).all() for value in state.components.values())
        sample = batch.samples[0]
        rows.append(((sample.source, int(sample.sampling_sample_id)), 0))
    return rows
