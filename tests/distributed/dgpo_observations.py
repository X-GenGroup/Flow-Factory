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


"""Passive native DGPO objective, logical-group and fast-reference observations."""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from dgpo_objective_reference import assert_local_preference, assert_objective
from native_observation_utils import cpu

from flow_factory.trainers.rl import dgpo


def _equal_parameters(left, right):
    assert len(left) == len(right)
    return all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        for a, b in zip(left, right, strict=True)
    )


@contextmanager
def observe_objectives(trainer):
    names = (
        "_prep_training_batch",
        "_forward_velocities",
        "_compute_group_dgpo_loss",
        "_compute_dsm_loss",
        "_reduce_group_sums",
        "_maybe_clip_dsm",
        "_velocity_kl",
        "_backward_full_output_loss",
        "_update_ema_ref",
        "sampling_context",
    )
    originals = {name: getattr(trainer, name) for name in names}
    forward = dgpo.forward_velocity_state
    records, active = [], {}
    lifecycle = {"sampling": [], "ema_ref_updates": [], "initial_step": trainer.step}
    frozen = cpu(trainer.adapter._ref_ema.ema_parameters)

    def prep(tb):
        prepared = originals["_prep_training_batch"](tb)
        active["prepared"] = prepared
        return prepared

    def observed_forward(owner, batch, state, times, *, source, **kwargs):
        from test_context_parallel_training import _compare

        if owner is not trainer:
            return forward(owner, batch, state, times, source=source, **kwargs)
        assert batch is active["batch"] and state is active["noised"].state
        assert times is active["times"]
        if source != "policy":
            assert not torch.is_grad_enabled()
        _compare(cpu(state), active["state_values"], exact=True)
        _compare(cpu(times), active["time_values"], exact=True)
        result = forward(owner, batch, state, times, source=source, **kwargs)
        _compare(cpu(state), active["state_values"], exact=True)
        _compare(cpu(times), active["time_values"], exact=True)
        return result

    def velocities(batch, times, noised):
        active.update(
            batch=batch,
            times=times,
            noised=noised,
            state_values=cpu(noised.state),
            time_values=cpu(times),
        )
        returned = originals["_forward_velocities"](batch, times, noised)
        active["velocities"] = returned
        prepared, args = active["prepared"], trainer.training_args
        assert returned["ref_dgpo_v"] is returned["old_v"]
        row = {
            "new": cpu(returned["model_v"].components),
            "old": cpu(returned["old_v"].components),
            "reference": cpu(returned["ref_v"].components),
            "target": cpu(noised.target_velocity.components),
            "noise": cpu(noised.noise.components),
            "masks": cpu(noised.state.active_masks or {}),
            "advantage": cpu(prepared["adv"]),
            "group_keys": [(sample.source, int(sample.unique_id)) for sample in batch.samples],
            "members": [int(sample.sampling_group_member_id) for sample in batch.samples],
            "group_size": args.group_size,
            "dpo_beta": trainer.dpo_beta,
            "use_ema_ref": trainer.use_ema_ref,
            "clip_dsm": trainer.clip_dsm,
            "clip_kl": trainer.clip_kl,
            "clip_range": args.clip_range,
            "kl_beta": trainer.kl_beta,
            "kl_dtype": torch.float32,
            "cp_size": trainer.config.parallelism_args.cp_size,
            "gas": trainer.accelerator.gradient_accumulation_steps,
            "gradients": {},
            "dsm_endpoints": {},
            "same_noised_inputs": True,
        }
        for name, value in returned["model_v"].components.items():
            assert value.requires_grad
            value.register_hook(
                lambda gradient, name=name, row=row: row["gradients"].__setitem__(
                    name, cpu(gradient)
                )
            )
        assert all(
            not value.requires_grad
            for key in ("old_v", "ref_v")
            for value in returned[key].components.values()
        )
        assert _equal_parameters(frozen, cpu(trainer.adapter._ref_ema.ema_parameters))
        records.append(row)
        return returned

    def dsm(target_v, pred_v, noised):
        result = originals["_compute_dsm_loss"](target_v, pred_v, noised)
        for output, name in (("model_v", "new"), ("old_v", "old"), ("ref_v", "reference")):
            if pred_v is active["velocities"][output]:
                previous = records[-1]["dsm_endpoints"].get(name)
                if previous is not None:
                    torch.testing.assert_close(cpu(result), previous, rtol=0, atol=0)
                records[-1]["dsm_endpoints"][name] = cpu(result)
        return result

    def clipped(*args, **kwargs):
        returned = originals["_maybe_clip_dsm"](*args, **kwargs)
        records[-1]["clipped"] = cpu(returned[0])
        return returned

    def group_loss(*args, **kwargs):
        row = records[-1]
        info = kwargs["group_info"]
        row["local_group_indices"] = cpu(info["local_group_indices"])
        return originals["_compute_group_dgpo_loss"](*args, **kwargs)

    def group_sums(local):
        row = records[-1]
        payload = {key: value for key, value in row.items() if key != "gradients"}
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
        degree = trainer.config.parallelism_args.cp_size * trainer.config.parallelism_args.tp_size
        logical = gathered[::degree]
        for start in range(0, len(gathered), degree):
            representative = gathered[start]
            for peer in gathered[start + 1 : start + degree]:
                assert peer["group_keys"] == representative["group_keys"]
                assert peer["members"] == representative["members"]
                for name in row["new"]:
                    for field in ("new", "old", "reference", "target", "noise"):
                        torch.testing.assert_close(
                            peer[field][name], representative[field][name], rtol=0, atol=0
                        )
        keys = sorted({tuple(key) for peer in logical for key in peer["group_keys"]})
        members = {key: [] for key in keys}
        noises = {}
        for peer in logical:
            for index, (key, member) in enumerate(
                zip(peer["group_keys"], peer["members"], strict=True)
            ):
                key = tuple(key)
                members[key].append(member)
                for name, noise in peer["noise"].items():
                    if (key, name) in noises:
                        torch.testing.assert_close(
                            noise[index], noises[(key, name)], rtol=0, atol=0
                        )
                    noises[(key, name)] = noise[index]
        assert all(sorted(values) == list(range(row["group_size"])) for values in members.values())
        row.update(logical_rows=logical, dense_group_keys=keys, local_sums=cpu(local))
        assert_local_preference(row, cpu(local), keys)
        result = originals["_reduce_group_sums"](local)
        assert not result.requires_grad
        row.update(
            logical_rows=logical,
            dense_group_keys=keys,
            global_sums=cpu(result),
            logical_replicas=len(logical),
            launched_ranks=len(gathered),
        )
        return result

    def kl(*args, **kwargs):
        result = originals["_velocity_kl"](*args, **kwargs)
        records[-1]["kl_dtype"] = result.dtype
        return result

    def backward(loss, **kwargs):
        row = records[-1]
        row["loss"] = cpu(loss)
        result = originals["_backward_full_output_loss"](loss, **kwargs)
        row["oracle"] = assert_objective(row)
        return result

    def update(step):
        before = cpu(trainer.adapter.get_named_parameters("ema_ref"))
        live = cpu(trainer.adapter._ema_tracked_parameters())
        result = originals["_update_ema_ref"](step)
        after = cpu(trainer.adapter.get_named_parameters("ema_ref"))
        decay = min(trainer.ema_ref_max_decay, trainer.ema_ref_ramp_rate * step)
        maximum_error = 0.0
        for old, current, actual in zip(before, live, after, strict=True):
            expected = old.double() * decay + current.double() * (1 - decay)
            bound = (
                4 * torch.finfo(actual.dtype).eps * (old.double().abs() + current.double().abs())
                + torch.finfo(actual.dtype).tiny
            )
            error = (actual.double() - expected).abs()
            assert bool((error <= bound).all())
            maximum_error = max(maximum_error, float(error.max()) if error.numel() else 0.0)
        lifecycle["ema_ref_updates"].append(
            {
                "step": step,
                "decay": decay,
                "maximum_error": maximum_error,
                "changed": not _equal_parameters(before, after),
            }
        )
        assert _equal_parameters(frozen, cpu(trainer.adapter._ref_ema.ema_parameters))
        return result

    @contextmanager
    def sampling():
        # Read the same prepared registry owner used by the real EMA swap.
        # Fresh component.parameters() can expose retained FSDP2 full/BF16 views
        # after a forward; observing those must not trigger backend writes.
        before = cpu(trainer.adapter._ema_tracked_parameters())
        switched = trainer.step > trainer.switch_ema_ref
        scope = "ema_ref" if switched else "ema" if trainer.off_policy else "live"
        desired = (
            cpu(trainer.adapter.get_named_parameters("ema_ref"))
            if switched
            else cpu(trainer.adapter.ema_wrapper.ema_parameters) if trainer.off_policy else before
        )
        with originals["sampling_context"]():
            assert _equal_parameters(cpu(trainer.adapter._ema_tracked_parameters()), desired)
            try:
                yield
            finally:
                assert _equal_parameters(cpu(trainer.adapter._ema_tracked_parameters()), desired)
        assert _equal_parameters(cpu(trainer.adapter._ema_tracked_parameters()), before)
        lifecycle["sampling"].append(
            {
                "step": trainer.step,
                "scope": scope,
                "snapshot_differs_from_live": not _equal_parameters(before, desired),
            }
        )

    replacements = {
        "_prep_training_batch": prep,
        "_forward_velocities": velocities,
        "_compute_group_dgpo_loss": group_loss,
        "_compute_dsm_loss": dsm,
        "_reduce_group_sums": group_sums,
        "_maybe_clip_dsm": clipped,
        "_velocity_kl": kl,
        "_backward_full_output_loss": backward,
        "_update_ema_ref": update,
        "sampling_context": sampling,
    }
    for name, replacement in replacements.items():
        setattr(trainer, name, replacement)
    dgpo.forward_velocity_state = observed_forward
    try:
        yield records
    finally:
        dgpo.forward_velocity_state = forward
        for name, original in originals.items():
            setattr(trainer, name, original)
        if records:
            lifecycle["final_step"] = trainer.step
            records[-1]["lifecycle"] = lifecycle


def assert_objectives(records, cp_size):
    assert records
    for row in records:
        assert row["cp_size"] == cp_size and row["same_noised_inputs"]
        assert row["logical_replicas"] == 2 and row["launched_ranks"] == 8
        assert_objective(row)
    assert any(
        not torch.equal(row["old"][name], row["reference"][name])
        for row in records
        for name in row["old"]
    )
    lifecycle = records[-1]["lifecycle"]
    updates = lifecycle["ema_ref_updates"]
    start = lifecycle["initial_step"]
    assert [event["step"] for event in updates] == list(range(start, lifecycle["final_step"]))
    assert any(event["changed"] for event in updates)
    assert any(
        event["scope"] == "ema_ref" and event["snapshot_differs_from_live"]
        for event in lifecycle["sampling"]
    )
    if start == 0:
        assert lifecycle["sampling"][0]["scope"] == "ema"


@contextmanager
def observe_training_terms(trainer, terms):
    prep, forward = trainer._prep_training_batch, trainer._forward_velocities
    identity = None

    def observed_prep(tb):
        nonlocal identity
        prepared = prep(tb)
        (sample,) = prepared["batch"].samples
        identity = (sample.source, int(sample.sampling_sample_id))
        return prepared

    def observed_forward(batch, times, noised):
        terms.append(
            (
                identity,
                float(times.timestep["latent"].reshape(-1)[0]),
                trainer.accelerator.gradient_accumulation_steps,
            )
        )
        return forward(batch, times, noised)

    trainer._prep_training_batch, trainer._forward_velocities = observed_prep, observed_forward
    try:
        yield
    finally:
        trainer._prep_training_batch, trainer._forward_velocities = prep, forward


def audit_samples(trainer, samples):
    rows = []
    for batch in trainer._iter_prefetched_batches(samples, 1):
        state = trainer.adapter.get_terminal_state(batch)
        assert all(torch.isfinite(value).all() for value in state.components.values())
        (sample,) = batch.samples
        rows.append(((sample.source, int(sample.sampling_sample_id)), 0))
    return rows
