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

"""Passive TDM boundary, objective and backward observations."""

from contextlib import contextmanager
from functools import wraps

import torch
from native_observation_utils import cpu
from tdm_objective_reference import assert_conditional_renoise, assert_objective, assert_projection
from test_context_parallel_training import _compare


@contextmanager
def observe_objectives(trainer, records):
    from flow_factory.trainers.distillation import tdm

    fake_loss, generator_loss = tdm.tdm_fake_loss, tdm.tdm_generator_loss
    replay = trainer._replay_generator_prediction
    sample_query = trainer._sample_score_query_times
    renoise = tdm.tdm_conditional_renoise
    query = tdm.query_score_velocity
    backward = trainer._backward_full_output_loss
    projection = trainer.adapter.project_velocity_to_clean_state
    active = {}

    def observe_loss(original, kind):
        @wraps(original)
        def wrapped(adapter, live, *targets, **kwargs):
            row = {
                "kind": kind,
                "live": cpu(live.components),
                "masks": cpu(live.active_masks),
                "gas": trainer.accelerator.gradient_accumulation_steps,
                "gradients": {},
                "boundary": active["boundary"],
                "sample_ids": active["sample_ids"],
                "cycle": trainer.cycle_index,
            }
            if kind == "fake":
                row.update(
                    target=cpu(targets[0].components),
                    sigma=cpu(kwargs["sigma"]),
                    importance=cpu(kwargs["importance"]),
                    snr_gamma=kwargs["snr_gamma"],
                )
                assert not kwargs["importance"].requires_grad
            else:
                row.update(
                    real=cpu(targets[0].components),
                    fake=cpu(targets[1].components),
                    use_huber=kwargs["use_huber"],
                    huber_c=kwargs["huber_c"],
                )
            assert all(
                not value.requires_grad
                for target in targets
                for value in target.components.values()
            )
            for name, value in live.components.items():
                assert value.requires_grad
                value.register_hook(
                    lambda gradient, name=name: row["gradients"].update({name: cpu(gradient)})
                )
            loss = original(adapter, live, *targets, **kwargs)
            row["loss"] = cpu(loss)
            records["objectives"].append(row)
            active["objective"] = row
            return loss

        return wrapped

    @wraps(replay)
    def observed_replay(unit):
        active.update(
            boundary=unit.boundary_index,
            sample_ids=tuple(int(s.sampling_sample_id) for s in unit.samples),
        )
        active["in_replay"] = True
        try:
            batch, clean, noise = replay(unit)
        finally:
            active["in_replay"] = False
        projected = active["replay_projection"]
        for name, value in noise.components.items():
            expected = cpu(clean.components[name]).float() + projected["velocity"][name].float()
            _compare(cpu(value), expected, exact=True)
        step = trainer.adapter.get_replay_step(batch, unit.boundary_index - 1)
        row = {
            "cycle": trainer.cycle_index,
            "boundary": unit.boundary_index,
            "ids": active["sample_ids"],
            "stored_times": cpu(step.times),
            "clean": cpu(clean),
            "noise": cpu(noise),
            "live": torch.is_grad_enabled(),
        }
        records.setdefault("replays", []).append(row)
        assert all(not value.requires_grad for value in noise.components.values())
        return batch, clean, noise

    @wraps(sample_query)
    def observed_query(unit, batch):
        result = sample_query(unit, batch)
        primary = trainer.adapter.trajectory_component_order[0]
        assert torch.all(result.sigma[primary] > unit.mid_times.sigma[primary])
        assert torch.all(result.sigma[primary] < 1.0)
        records.setdefault("queries", []).append(
            {"cycle": trainer.cycle_index, "boundary": unit.boundary_index, "times": cpu(result)}
        )
        return result

    @wraps(renoise)
    def observed_renoise(*args, **kwargs):
        generator = torch.Generator(device=kwargs["generator"].device)
        generator.set_state(kwargs["generator"].get_state())
        fresh = {
            name: torch.randn(value.shape, generator=generator, dtype=torch.float32)
            for name, value in args[1].components.items()
        }
        result = renoise(*args, **kwargs)
        _compare(generator.get_state(), kwargs["generator"].get_state(), exact=True)
        row = {
            "cycle": trainer.cycle_index,
            "clean": cpu(args[1].components),
            "implied": cpu(args[2].components),
            "masks": cpu(args[1].active_masks),
            "fresh": fresh,
            "mid": cpu(kwargs["mid_times"].sigma),
            "target_sigma": cpu(kwargs["target_times"].sigma),
            "importance_clip": kwargs["importance_clip"],
            "noise": cpu(result[0].noise.components),
            "state": cpu(result[0].state.components),
            "velocity_target": cpu(result[0].target_velocity.components),
            "importance": cpu(result[1]),
        }
        assert_conditional_renoise(row)
        records.setdefault("renoising", []).append(row)
        assert all(
            not value.requires_grad for state in args[1:3] for value in state.components.values()
        )
        active["noised"] = cpu(result[0].state)
        return result

    @wraps(projection)
    def observed_projection(state, times, velocity):
        result = projection(state, times, velocity)
        row = {
            "state": cpu(state.components),
            "sigma": cpu(times.sigma),
            "velocity": cpu(velocity.components),
            "clean": cpu(result.components),
        }
        assert_projection(row)
        if active.get("in_replay"):
            active["replay_projection"] = row
        records.setdefault("projections", []).append(row)
        return result

    @wraps(query)
    def observed_score(*args, **kwargs):
        before = cpu(args[2])
        _compare(before, active["noised"], exact=True)
        result = query(*args, **kwargs)
        _compare(cpu(args[2]), before, exact=True)
        assert all(not value.requires_grad for value in result.components.values())
        records.setdefault("score_queries", []).append(
            {
                "cycle": trainer.cycle_index,
                "role": kwargs["role_name"],
                "state": before,
                "times": cpu(args[3]),
                "kwargs": cpu(kwargs["forward_kwargs"]),
            }
        )
        return result

    @wraps(backward)
    def observed_backward(loss, **kwargs):
        row = active.get("objective")
        result = backward(loss, **kwargs)
        # A derived algorithm can add terms after the TDM anchor. Its observer
        # owns the complete derivative rather than treating the anchor as total.
        if row is not None and trainer.training_args.trainer_type == "tdm":
            try:
                assert_objective(row, trainer.config.parallelism_args.cp_size)
            except AssertionError:
                import os
                from pathlib import Path

                directory = Path(os.environ.get("FF_TEST_FAILURE_DIR", ".scratch/tdm-failures"))
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(
                    row, directory / f"objective-rank-{trainer.accelerator.process_index}.pt"
                )
                raise
        return result

    tdm.tdm_fake_loss = observe_loss(fake_loss, "fake")
    tdm.tdm_generator_loss = observe_loss(generator_loss, "generator")
    trainer._replay_generator_prediction = observed_replay
    trainer._sample_score_query_times = observed_query
    tdm.tdm_conditional_renoise = observed_renoise
    tdm.query_score_velocity = observed_score
    trainer._backward_full_output_loss = observed_backward
    trainer.adapter.project_velocity_to_clean_state = observed_projection
    try:
        yield
    finally:
        tdm.tdm_fake_loss, tdm.tdm_generator_loss = fake_loss, generator_loss
        trainer._replay_generator_prediction, trainer._sample_score_query_times = (
            replay,
            sample_query,
        )
        tdm.tdm_conditional_renoise, tdm.query_score_velocity = renoise, query
        trainer._backward_full_output_loss = backward
        trainer.adapter.project_velocity_to_clean_state = projection
