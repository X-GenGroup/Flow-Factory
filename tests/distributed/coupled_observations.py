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


"""Observe native coupled objectives without changing forwards or gradients."""

from contextlib import contextmanager

import torch
from coupled_objective_reference import assert_objective
from native_observation_utils import cpu


@contextmanager
def observe_coupled_objectives(trainer):
    """Check actual dense Flow-SDE objective upstreams before CP gather backward."""
    algorithm = trainer.training_args.trainer_type
    assert algorithm in ("grpo", "grpo-guard", "dppo"), algorithm
    assert trainer.adapter.trajectory_component_order == ("latent",)
    assert trainer.adapter.scheduler.dynamics_type == "Flow-SDE"
    originals = {
        name: getattr(trainer, name)
        for name in ("_replay_forward", "_reference_forward", "_backward_full_output_loss")
    }
    records = []

    def replay_forward(batch, replay, return_fields):
        # The complete-loop runner audits retained replay before feedback exists.
        if "advantage" not in batch or batch["advantage"] is None:
            return originals["_replay_forward"](batch, replay, return_fields)
        fields = tuple(
            dict.fromkeys((*return_fields, "velocity", "next_latents_mean", "std_dev_t", "dt"))
        )
        output = originals["_replay_forward"](batch, replay, fields)
        positions = (
            (batch["timesteps"] == replay.times.timestep["latent"].reshape(-1, 1))
            .all(dim=0)
            .nonzero()
            .flatten()
        )
        assert positions.numel() == 1
        index = positions.item()
        args = trainer.training_args
        row = {
            "algorithm": algorithm,
            "cycle": trainer.cycle_index,
            "step": trainer.step,
            "timestep_index": index,
            "advantage": cpu(batch["advantage"]),
            "adv_clip_range": args.adv_clip_range,
            "clip_range": getattr(args, "clip_range", None),
            "kl_type": args.kl_type,
            "kl_beta": args.kl_beta,
            "kl_mask_type": getattr(args, "kl_mask_type", None),
            "kl_mask_threshold": getattr(args, "kl_mask_threshold", None),
            "mean": cpu(output.next_state_mean.components["latent"]),
            "velocity": cpu(output.velocity.components["latent"]),
            "following": cpu(replay.next_state.components["latent"]),
            "log_prob": cpu(output.log_prob),
            "old_log_prob": cpu(replay.log_prob),
            "std": cpu(output.std_dev_t["latent"]),
            "dt": cpu(output.dt["latent"]),
            # Native Euler Flow-SDE replays use the public 0..1000 clock.
            "sigma": cpu(replay.times.timestep["latent"]) / 1000.0,
            "cp_size": trainer.config.parallelism_args.cp_size,
            "gas": args.gradient_accumulation_steps,
        }
        if algorithm == "grpo-guard" or (algorithm == "dppo" and args.kl_mask_type == "x-based"):
            row["old_mean"] = cpu(
                trainer.adapter.get_replay_callback(batch, index, "next_latents_mean").components[
                    "latent"
                ]
            )
        elif algorithm == "dppo":
            row["old_velocity"] = cpu(
                trainer.adapter.get_replay_callback(batch, index, "velocity").components["latent"]
            )
        records.append(row)
        for key, tensor in (
            ("mean", output.next_state_mean.components["latent"]),
            ("velocity", output.velocity.components["latent"]),
        ):
            assert tensor.requires_grad
            tensor.register_hook(
                lambda gradient, key=key, row=row: row.__setitem__(key + "_gradient", cpu(gradient))
            )
        return output

    def reference_forward(*args, **kwargs):
        output = originals["_reference_forward"](*args, **kwargs)
        state = (
            output.velocity
            if trainer.training_args.kl_type == "v-based"
            else output.next_state_mean
        )
        assert not state.components["latent"].requires_grad
        records[-1]["reference"] = cpu(state.components["latent"])
        records[-1]["reference_overrides"] = cpu(kwargs)
        return output

    def backward(loss, **kwargs):
        row = records[-1]
        row["loss"] = cpu(loss)
        result = originals["_backward_full_output_loss"](loss, **kwargs)
        row["oracle"] = assert_objective(row)
        return result

    trainer._replay_forward = replay_forward
    trainer._reference_forward = reference_forward
    trainer._backward_full_output_loss = backward
    try:
        yield records
    finally:
        for name, original in originals.items():
            setattr(trainer, name, original)


def assert_objectives(records, cp_size):
    """Require checked real backwards, including nonzero policy signal across a run.

    Branch and boundary coverage is supplied by deterministic CPU controls. A
    native schedule may legitimately encounter only one branch or zero derivative
    for an individual masked sample; the complete run must update from nonzero
    independently checked derivatives.
    """
    assert records
    for row in records:
        assert row["cp_size"] == cp_size
        assert "oracle" in row
        assert_objective(row)
    assert any(
        bool(
            (row["oracle"]["velocity_gradient"].abs() > 4 * row["oracle"]["velocity_budget"]).any()
        )
        for row in records
    ), "the native objective run must contain a nonzero derivative"
