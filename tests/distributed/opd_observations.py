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

"""Passive OPD replay, teacher-query and target-cache ownership observations."""

from contextlib import contextmanager
from functools import wraps

import torch
from native_observation_utils import cpu
from opd_objective_reference import assert_objective, assert_ode_projection
from test_context_parallel_training import _compare

from flow_factory.trainers.distillation.opd import trainer as opd


def _identity(sample):
    return sample.source, int(sample.sampling_sample_id)


def _replay(replay, batch):
    return cpu(
        {
            "replay": replay,
            "conditioning": {
                key: batch.get(key)
                for key in ("prompt_embeds", "negative_prompt_embeds", "guidance_scale")
            },
        }
    )


@contextmanager
def observe_objectives(trainer, records):
    """Observe actual calls, restoring every wrapper after training."""
    forward_step = trainer._forward_step
    scheduler_step = trainer.adapter.scheduler_group.primary.step
    precompute, distill = trainer._precompute_teacher_targets, trainer._distill
    named = trainer.adapter.use_named_parameters
    reducer, loss_fn = (
        trainer.adapter.reduce_latent_values,
        opd.compute_structured_distillation_loss,
    )
    backward = trainer._backward_full_output_loss
    metric_reduce = opd.reduce_across_replicas
    active, queries = {}, {}
    frozen = {
        name: cpu(info.ema_wrapper.ema_parameters)
        for name, info in trainer.adapter._named_parameters.items()
    }
    routing = {
        source: (teacher.name, teacher.guidance_scale)
        for teacher in trainer.training_args.teachers
        for source in teacher.applicable_datasets
    }

    def frozen_teachers():
        for name, info in trainer.adapter._named_parameters.items():
            _compare(cpu(info.ema_wrapper.ema_parameters), frozen[name], exact=True)

    @contextmanager
    def observed_named(name):
        before = cpu(trainer.adapter.get_trainable_parameters())
        with named(name):
            info = trainer.adapter._named_parameters[name]
            live = cpu(trainer.adapter._get_component_parameters(info.target_components))
            _compare(live, frozen[name], exact=True)
            assert any(not torch.equal(a, b) for a, b in zip(live, before, strict=True))
            active["teacher"] = name
            try:
                yield
            finally:
                active.pop("teacher")
        _compare(cpu(trainer.adapter.get_trainable_parameters()), before, exact=True)
        frozen_teachers()

    @wraps(scheduler_step)
    def observed_scheduler_step(*args, **kwargs):
        result = scheduler_step(*args, **kwargs)
        if active.get("forward_step"):
            active["scheduler"] = {
                "velocity": cpu(kwargs["velocity"]),
                "state": cpu(kwargs["latents"]),
            }
            if kwargs["velocity"].requires_grad:
                row = active["scheduler"]
                kwargs["velocity"].register_hook(
                    lambda gradient: row.update(velocity_gradient=cpu(gradient))
                )
        return result

    def observed_step(batch, step_index, **kwargs):
        identity = tuple(_identity(sample) for sample in batch.samples)
        replay_before = trainer.adapter.get_replay_step(batch, step_index)
        before = _replay(replay_before, batch)
        active["forward_step"] = True
        try:
            replay, target, output = forward_step(batch, step_index, **kwargs)
        finally:
            active["forward_step"] = False
        _compare(_replay(replay, batch), before, exact=True)
        assert tuple(target.components) == ("latent",)
        row = active["scheduler"]
        row.update(
            {
                "identity": identity,
                "step": step_index,
                "inputs": before,
                "guidance_scale": kwargs["guidance_scale"],
                "timestep": cpu(replay.times.timestep["latent"]),
                "next_timestep": cpu(replay.times.next_timestep["latent"]),
                "target": cpu(target.components["latent"]),
            }
        )
        assert_ode_projection(row)
        if "teacher" in active:
            assert not torch.is_grad_enabled()
            expected_name, expected_guidance = routing[batch.samples[0].source]
            assert active["teacher"] == expected_name
            assert kwargs["guidance_scale"] == expected_guidance
            assert all(sample.source == batch.samples[0].source for sample in batch.samples)
            row["teacher_name"] = expected_name
            queries[(identity, step_index)] = row
            records["cache_events"].append(row)
        else:
            teacher = queries[(identity, step_index)]
            _compare(before, teacher["inputs"], exact=True)
            assert kwargs["guidance_scale"] == trainer.training_args.guidance_scale
            assert not torch.equal(row["target"], teacher["target"])
            active["student"] = row
            active["expected_teacher"] = teacher["target"]
            records["replay_inputs"].append(before)
        return replay, target, output

    def observed_precompute(samples, steps):
        assert not queries
        records["samples"].append([_identity(sample) for sample in samples])
        returned = precompute(samples, steps)
        assert len(queries) == len(samples) * len(steps)
        for sample in samples:
            cached = sample.extra_kwargs["teacher_target"]["latent"]
            assert not cached.requires_grad
            for index, step in enumerate(steps):
                query = queries[((_identity(sample),), int(step))]
                _compare(cpu(cached[index]), query["target"][0], exact=True)
        return returned

    def observed_distill(samples, steps):
        cached = cpu([sample.extra_kwargs["teacher_target"] for sample in samples])
        returned = distill(samples, steps)
        _compare(
            cpu([sample.extra_kwargs["teacher_target"] for sample in samples]), cached, exact=True
        )
        frozen_teachers()
        queries.clear()
        return returned

    def observed_reduce(*args, **kwargs):
        result = reducer(*args, **kwargs)
        if active.get("loss_row") is not None:
            active["loss_row"]["reductions"].append(cpu(result))
        return result

    def observed_loss(adapter, **kwargs):
        student, teacher = (
            kwargs["student_target"].components["latent"],
            kwargs["teacher_target"].components["latent"],
        )
        _compare(cpu(teacher), active["expected_teacher"], exact=True)
        row = {
            "student": cpu(student),
            "teacher": cpu(teacher),
            "teacher_detached": not teacher.requires_grad,
            "self_normalize": kwargs["self_normalize"],
            "gas": trainer.accelerator.gradient_accumulation_steps,
            "reductions": [],
            "source": active["student"]["identity"][0][0],
            "projection": active["student"],
        }
        active["loss_row"] = row
        student.register_hook(lambda gradient: row.update(target_gradient=cpu(gradient)))
        result = loss_fn(adapter, **kwargs)
        active["loss_row"] = None
        active["objective"] = row
        records["objectives"].append(row)
        return result

    def observed_backward(loss, **kwargs):
        row = active["objective"]
        row["loss"] = cpu(loss)
        result = backward(loss, **kwargs)
        assert_objective(row, trainer.config.parallelism_args.cp_size)
        frozen_teachers()
        return result

    def observed_metric_reduce(accelerator, tensor, reduction):
        result = metric_reduce(accelerator, tensor, reduction)
        assert reduction == "sum" and tensor.shape == (2, len(trainer._teacher_names))
        records["packed_metrics"].append({"local": cpu(tensor), "global": cpu(result)})
        return result

    trainer.adapter.use_named_parameters = observed_named
    trainer.adapter.scheduler_group.primary.step = observed_scheduler_step
    trainer._forward_step = observed_step
    trainer._precompute_teacher_targets = observed_precompute
    trainer._distill = observed_distill
    trainer.adapter.reduce_latent_values = observed_reduce
    opd.compute_structured_distillation_loss = observed_loss
    trainer._backward_full_output_loss = observed_backward
    opd.reduce_across_replicas = observed_metric_reduce
    try:
        yield
        assert not queries
    finally:
        trainer.adapter.use_named_parameters = named
        trainer.adapter.scheduler_group.primary.step = scheduler_step
        trainer._forward_step = forward_step
        trainer._precompute_teacher_targets = precompute
        trainer._distill = distill
        trainer.adapter.reduce_latent_values = reducer
        opd.compute_structured_distillation_loss = loss_fn
        trainer._backward_full_output_loss = backward
        opd.reduce_across_replicas = metric_reduce
