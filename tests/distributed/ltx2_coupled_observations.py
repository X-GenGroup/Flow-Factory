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


"""Independent joint GRPO oracle at LTX's two native scheduler graph endpoints."""

import math
from contextlib import contextmanager
from functools import wraps

import torch
from native_observation_utils import cpu


def objective_reference(row):
    """Differentiate the joint active-DOF likelihood without production loss helpers."""
    components = row["components"]
    assert tuple(components) == ("video", "audio")
    total = sum(item["mean"][0].numel() for item in components.values())
    log_probs = {}
    for name, item in components.items():
        mean, following = item["mean"].double(), item["following"].double()
        scale = item["std"].double() * (-item["dt"].double()).sqrt()
        assert bool((scale > 0).all())
        log_probs[name] = (
            (
                -0.5 * ((following - mean) / scale).square()
                - scale.log()
                - 0.5 * math.log(2 * math.pi)
            )
            .flatten(1)
            .mean(1)
        )
        torch.testing.assert_close(item["log_prob"].double(), log_probs[name], rtol=2e-5, atol=1e-6)
    joint = sum(
        log_probs[name] * item["mean"][0].numel() / total for name, item in components.items()
    )
    torch.testing.assert_close(row["log_prob"].double(), joint, rtol=2e-5, atol=1e-6)
    # Use the observed FP32 likelihood for the same finite-precision PPO branch;
    # its value was independently reconstructed immediately above.
    ratio = (row["log_prob"].double() - row["old_log_prob"].double()).exp()
    advantage = row["advantage"].double().clamp(*row["adv_clip_range"])
    lower, upper = (1 + value for value in row["clip_range"])
    keep = ~((advantage > 0) & (ratio > upper) | (advantage < 0) & (ratio < lower))
    loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(lower, upper)).mean()
    normalization = row["cp_size"] / row["gas"]
    unit = torch.finfo(torch.float32).eps / 2
    operations = 128 + 8 * total
    assert operations * unit < 1
    gamma = operations * unit / (1 - operations * unit)
    result = {
        "loss": loss,
        "ratio": ratio,
        "keep": keep,
        "joint_log_prob": joint,
        "active_elements": total,
        "components": {},
    }
    for name, item in components.items():
        mean, following = item["mean"].double(), item["following"].double()
        batch = mean.shape[0]
        shape = (batch,) + (1,) * (mean.ndim - 1)
        std, dt, sigma = (item[key].double().reshape(shape) for key in ("std", "dt", "sigma"))
        coefficient = (-advantage * ratio * keep / batch).reshape(shape)
        mean_gradient = coefficient * (following - mean) / (total * std.square() * (-dt))
        jacobian = (1 + std.square() * (1 - sigma) / (2 * sigma)) * dt
        velocity_gradient = mean_gradient * jacobian
        result["components"][name] = {
            "mean_gradient": mean_gradient * normalization,
            "velocity_gradient": velocity_gradient * normalization,
            "mean_budget": gamma * mean_gradient.abs() * normalization
            + torch.finfo(torch.float32).tiny,
            "velocity_budget": gamma * velocity_gradient.abs() * normalization
            + torch.finfo(torch.float32).tiny,
        }
    return result


def assert_objective(row):
    reference = objective_reference(row)
    torch.testing.assert_close(row["loss"].double(), reference["loss"], rtol=2e-5, atol=1e-7)
    for name, item in row["components"].items():
        for field in ("mean", "velocity"):
            actual = item[field + "_gradient"].double()
            expected = reference["components"][name][field + "_gradient"]
            budget = reference["components"][name][field + "_budget"]
            assert actual.shape == expected.shape
            assert bool(((actual - expected).abs() <= budget).all()), (
                name,
                field,
                (actual - expected).abs().max(),
                budget.max(),
            )
            if bool((expected.abs() > 4 * budget).any()):
                for wrong in (torch.zeros_like(actual), -actual, actual / 2, actual * 2):
                    assert bool(((wrong - expected).abs() > budget).any()), (
                        name,
                        field,
                        "vacuous derivative",
                    )
    return reference


@contextmanager
def observe_objectives(trainer):
    """Capture original scheduler tensors before I2AV repacks its immutable frame."""
    assert trainer.training_args.trainer_type == "grpo" and trainer.training_args.kl_beta == 0
    assert trainer.adapter.trajectory_component_order == ("video", "audio")
    schedulers = {"video": trainer.adapter.scheduler, "audio": trainer.adapter.audio_scheduler}
    assert all(s.dynamics_type == "Flow-SDE" for s in schedulers.values())
    originals = {name: s.step for name, s in schedulers.items()}
    replay = trainer._replay_forward
    backward = trainer._backward_full_output_loss
    current = None
    records = []

    def capture(name, original):
        @wraps(original)
        def step(*args, **kwargs):
            output = original(*args, **kwargs)
            if current is not None:
                assert name not in current["components"]
                item = {
                    "mean": cpu(output.next_latents_mean),
                    "velocity": cpu(output.velocity),
                    "following": cpu(kwargs["next_latents"]),
                    "log_prob": cpu(output.log_prob),
                    "std": cpu(output.std_dev_t),
                    "dt": cpu(output.dt),
                    "sigma": cpu(kwargs["timestep"]) / 1000,
                }
                current["components"][name] = item
                for field, tensor in (
                    ("mean", output.next_latents_mean),
                    ("velocity", output.velocity),
                ):
                    assert tensor.requires_grad
                    tensor.register_hook(
                        lambda gradient, field=field, item=item: item.__setitem__(
                            field + "_gradient", cpu(gradient)
                        )
                    )
            return output

        return step

    for name, scheduler in schedulers.items():
        scheduler.step = capture(name, originals[name])

    @wraps(replay)
    def observed_replay(batch, replayed, return_fields):
        nonlocal current
        if "advantage" not in batch or batch["advantage"] is None:
            return replay(batch, replayed, return_fields)
        args = trainer.training_args
        current = {
            "algorithm": "grpo",
            "cycle": trainer.cycle_index,
            "step": trainer.step,
            "advantage": cpu(batch["advantage"]),
            "adv_clip_range": args.adv_clip_range,
            "clip_range": args.clip_range,
            "old_log_prob": cpu(replayed.log_prob),
            "cp_size": trainer.config.parallelism_args.cp_size,
            "gas": args.gradient_accumulation_steps,
            "components": {},
        }
        fields = tuple(
            dict.fromkeys((*return_fields, "velocity", "next_latents_mean", "std_dev_t", "dt"))
        )
        try:
            output = replay(batch, replayed, fields)
        finally:
            row = current
            current = None
        row["log_prob"] = cpu(output.log_prob)
        assert tuple(row["components"]) == ("video", "audio")
        for name, item in row["components"].items():
            for field, public in (
                ("mean", output.next_state_mean.components[name]),
                ("velocity", output.velocity.components[name]),
            ):
                if item[field].ndim == 5:
                    public = trainer.adapter.pipeline._unpack_latents(public, 3, 2, 2, 1, 1)[
                        :, :, 1:
                    ]
                torch.testing.assert_close(cpu(public), item[field], rtol=0, atol=0)
            if name == "video" and item["mean"].ndim == 5:
                assert item["mean"].shape == (1, 128, 2, 2, 2)
                active = replayed.state.active_masks["video"]
                assert torch.count_nonzero(active) == 8
        records.append(row)
        return output

    @wraps(backward)
    def observed_backward(loss, **kwargs):
        row = records[-1]
        row["loss"] = cpu(loss)
        value = backward(loss, **kwargs)
        row["oracle"] = assert_objective(row)
        return value

    trainer._replay_forward = observed_replay
    trainer._backward_full_output_loss = observed_backward
    try:
        yield records
    finally:
        for name, scheduler in schedulers.items():
            scheduler.step = originals[name]
        trainer._replay_forward = replay
        trainer._backward_full_output_loss = backward


def assert_objectives(records, cp_size):
    assert records
    for row in records:
        assert row["cp_size"] == cp_size and "oracle" in row
        assert_objective(row)
    for name in ("video", "audio"):
        assert any(
            bool(
                (
                    row["oracle"]["components"][name]["velocity_gradient"].abs()
                    > 4 * row["oracle"]["components"][name]["velocity_budget"]
                ).any()
            )
            for row in records
        ), name
