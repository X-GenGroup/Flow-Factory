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

"""Read-only native DMD2 inputs and derivative endpoints for qualification."""

import dataclasses
from contextlib import contextmanager
from functools import wraps

import torch


def cpu(value):
    """Detach only local tensor values; never materialize a DTensor collectively."""
    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            value = value.to_local()
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value):
        return {field.name: cpu(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def instrument(trainer, captures):
    import flow_factory.trainers.distillation.dmd2 as dmd2

    current = [None]
    events = captures["microbatches"]
    adapter = trainer.adapter
    handles = []

    def begin(original, role):
        @wraps(original)
        def wrapped(batch):
            event = {
                "index": len(events),
                "role": role,
                "stage": "loss",
                "sample_ids": [
                    {
                        key: getattr(sample, key)
                        for key in (
                            "source_id",
                            "sampling_group_id",
                            "sampling_group_member_id",
                            "sampling_sample_id",
                        )
                    }
                    for sample in batch.samples
                ],
                "modules": [],
                "scores": [],
            }
            event["active_root"] = "members." + adapter.component_variant_registry.resolve_route(
                role, "transformer"
            )
            events.append(event)
            old, current[0] = current[0], event
            try:
                result = original(batch)
                event["loss"] = cpu(result)
                return result
            finally:
                current[0] = old

        return wrapped

    trainer._fake_replay_loss = begin(trainer._fake_replay_loss, "fake")
    trainer._generator_replay_loss = begin(trainer._generator_replay_loss, "generator")
    replay = trainer._replay_generator_clean_prediction

    @wraps(replay)
    def observed_replay(batch, boundary_index):
        event = current[0]
        if event is None:
            return replay(batch, boundary_index)
        event["boundary_index"] = boundary_index
        event["replay_step"] = cpu(adapter.get_replay_step(batch, boundary_index - 1))
        prior, event["stage"] = event["stage"], "generator_clean_replay"
        try:
            result = replay(batch, boundary_index)
            event["clean_state"] = cpu(result)
            return result
        finally:
            event["stage"] = prior

    trainer._replay_generator_clean_prediction = observed_replay

    sample_times = trainer._sample_perturbation_times

    @wraps(sample_times)
    def observed_times(*args, **kwargs):
        result = sample_times(*args, **kwargs)
        if current[0] is not None:
            current[0]["perturbation_times"] = cpu(result)
        return result

    trainer._sample_perturbation_times = observed_times

    add_noise = adapter.add_forward_process_noise

    @wraps(add_noise)
    def observed_noise(*args, **kwargs):
        result = add_noise(*args, **kwargs)
        if current[0] is not None:
            current[0]["noised"] = cpu(result)
        return result

    adapter.add_forward_process_noise = observed_noise

    score = dmd2.query_score_velocity

    @wraps(score)
    def observed_score(*args, **kwargs):
        event = current[0]
        if event is None:
            return score(*args, **kwargs)
        row = {
            "role": kwargs["role_name"],
            "state": cpu(args[2]),
            "times": cpu(args[3]),
            "forward_kwargs": cpu(kwargs["forward_kwargs"]),
        }
        event["scores"].append(row)
        prior, event["stage"] = event["stage"], "score/" + row["role"]
        try:
            result = score(*args, **kwargs)
            row["velocity"] = cpu(result)
            return result
        finally:
            event["stage"] = prior

    dmd2.query_score_velocity = observed_score

    def observe_objective(original, kind):
        @wraps(original)
        def wrapped(adapter_arg, *states, **kwargs):
            event = current[0]
            if event is None:
                return original(adapter_arg, *states, **kwargs)
            names = (
                ("gen", "real", "fake")
                if kind == "generator"
                else ("predicted_velocity", "target_velocity")
            )
            row = {
                "kind": kind,
                **{name: cpu(state) for name, state in zip(names, states)},
                "kwargs": cpu(kwargs),
                "upstream": {},
            }
            event["objective"] = row
            for name, tensor in states[0].components.items():
                if tensor.requires_grad:
                    tensor.register_hook(
                        lambda gradient, name=name, row=row: row["upstream"].__setitem__(
                            name, cpu(gradient)
                        )
                    )
            loss = original(adapter_arg, *states, **kwargs)
            row["loss"] = cpu(loss)
            return loss

        return wrapped

    dmd2.flow_matching_loss = observe_objective(dmd2.flow_matching_loss, "fake")
    dmd2.dmd_generator_loss = observe_objective(dmd2.dmd_generator_loss, "generator")

    roots = {"members." + name for name in adapter.component_variant_registry.bundle_members()}
    for name, module in trainer.model_bundle.named_modules():
        if name not in roots:
            continue

        def hook(module, args, kwargs, output, name=name):
            event = current[0]
            if event is None:
                return
            row = {
                "name": name,
                "stage": event["stage"],
                "args": cpu(args),
                "kwargs": cpu(kwargs),
                "output": cpu(output),
            }
            tensor_output = output[0] if isinstance(output, tuple) and len(output) == 1 else output
            if isinstance(tensor_output, torch.Tensor):
                row["output_requires_grad"] = tensor_output.requires_grad
                if tensor_output.requires_grad:
                    tensor_output.register_hook(
                        lambda gradient, row=row: row.__setitem__("dy", cpu(gradient))
                    )
            event["modules"].append(row)

        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    captures["module_hooks"] = len(handles)
    captures["metadata"] = {
        "cp_size": trainer.config.parallelism_args.cp_size,
        "gas": trainer.training_args.gradient_accumulation_steps,
        "torch": torch.__version__,
    }
    # Keep handles alive until training, all backward hooks, and state observations end.
    captures["_handles"] = handles


@contextmanager
def observe_dmd2_objectives(trainer):
    """Observe actual inputs/upstreams without extra forwards or collectives."""
    if trainer.training_args.trainer_type != "dmd2":
        yield []
        return
    import flow_factory.trainers.distillation.dmd2 as dmd2

    originals = {
        name: getattr(dmd2, name)
        for name in ("query_score_velocity", "flow_matching_loss", "dmd_generator_loss")
    }
    captured = {"microbatches": []}
    try:
        instrument(trainer, captured)
        yield captured["microbatches"]
    finally:
        for handle in captured.pop("_handles", []):
            handle.remove()
        for name, original in originals.items():
            setattr(dmd2, name, original)
