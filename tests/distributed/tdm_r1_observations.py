# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Passive native R1 objective, logical operands and slow-snapshot observations."""

from contextlib import contextmanager
from functools import wraps

import torch
import torch.distributed as dist
from native_observation_utils import cpu
from tdm_objective_reference import assert_objective, objective_reference
from tdm_observations import observe_objectives as observe_tdm
from tdm_r1_objective_reference import assert_surrogate, direction_reference
from test_context_parallel_training import _compare

from flow_factory.samples import group_identity_rows
from flow_factory.trainers.distillation import tdm_r1 as r1
from flow_factory.utils.dist import data_parallel_geometry, data_parallel_process_group


@contextmanager
def observe_objectives(trainer, records):
    """Keep all native forwards, updates and targets intact while recording proofs."""
    originals = {
        name: getattr(trainer, name)
        for name in (
            "_surrogate_boundary_loss",
            "_generator_boundary_loss",
            "_boundary_preference_values",
            "_diffusion_density_values",
            "_backward_full_output_loss",
            "_generator_score_terms",
            "_guidance_reward_direction",
            "_surrogate_reward_direction",
        )
    }
    snapshot, update_snapshot = (
        trainer.adapter.use_variant_snapshot,
        trainer.adapter.update_variant_snapshot,
    )
    revised, query = r1.revised_x0_loss, r1.query_score_velocity
    active = {}
    records.update(
        surrogate_objectives=[],
        generator_rewards=[],
        snapshot_updates=[],
        snapshot_queries=[],
        reward_queries=[],
    )

    @wraps(originals["_surrogate_boundary_loss"])
    def surrogate(unit):
        row = dict(
            cycle=trainer.cycle_index,
            boundary_index=unit.boundary_index,
            num_steps=2,
            group_keys=group_identity_rows(unit.samples),
            advantage=torch.tensor([float(s.extra_kwargs["advantage"]) for s in unit.samples]),
            group_size=2,
            preference_beta=trainer.training_args.surrogate_preference_beta,
            reference_beta=trainer.training_args.surrogate_reference_beta,
            reference_threshold=trainer.training_args.surrogate_reference_threshold,
            clip_range=trainer.training_args.surrogate_clip_range,
            time_weighting=True,
            cp_size=trainer.config.parallelism_args.cp_size,
            gas=4,
            dsm_endpoints={},
            gradients={},
        )
        active["surrogate"] = row
        loss = originals["_surrogate_boundary_loss"](unit)
        active.pop("surrogate")
        row["loss"] = cpu(loss)
        group = data_parallel_process_group(trainer.accelerator)
        peers = [None] * data_parallel_geometry(trainer.accelerator)[0]
        dist.all_gather_object(peers, cpu(row), group=group)
        row["logical_rows"] = peers
        records["surrogate_objectives"].append(row)
        active["pending_surrogate"] = row
        return loss

    @wraps(originals["_boundary_preference_values"])
    def preference(unit, *, trainable_role, context=None):
        row = active.get("surrogate")
        if row is not None:
            operands = cpu(context)
            if "score_context" in row:
                _compare(row["score_context"], operands, exact=True)
            else:
                row["score_context"] = operands
            active["density_calls"] = 0
        return originals["_boundary_preference_values"](
            unit, trainable_role=trainable_role, context=context
        )

    @wraps(originals["_diffusion_density_values"])
    def density(target, prediction, *, state):
        result = originals["_diffusion_density_values"](target, prediction, state=state)
        row = active.get("surrogate")
        if row is not None:
            first = active["density_calls"] == 0
            kind = ("slow" if active.get("snapshot") else "new") if first else "reference"
            active["density_calls"] += 1
            if "target" in row:
                _compare(row["target"], cpu(target.components), exact=True)
            row["target"], row["masks"] = cpu(target.components), cpu(state.active_masks or {})
            if kind in row:
                _compare(row[kind], cpu(prediction.components), exact=True)
            row[kind], row["dsm_endpoints"][kind] = cpu(prediction.components), cpu(result)
            if kind == "new":
                assert result.requires_grad
                result.register_hook(lambda grad: row.update(upstream=cpu(grad)))
                for name, value in prediction.components.items():
                    value.register_hook(
                        lambda grad, name=name: row["gradients"].update({name: cpu(grad)})
                    )
            else:
                assert not result.requires_grad
        return result

    @contextmanager
    def use_snapshot(name):
        before = cpu(trainer.adapter.variant_parameters("surrogate"))
        expected = cpu(trainer.adapter.get_variant_snapshot(name))
        with snapshot(name):
            active["snapshot"] = True
            _compare(cpu(trainer.adapter.variant_parameters("surrogate")), expected, exact=True)
            try:
                yield
            finally:
                active.pop("snapshot")
        _compare(cpu(trainer.adapter.variant_parameters("surrogate")), before, exact=True)
        records["snapshot_queries"].append({"cycle": trainer.cycle_index, "restored": True})

    def update(name, decay):
        before = cpu(trainer.adapter.get_variant_snapshot(name))
        live = cpu(trainer.adapter.variant_parameters("surrogate"))
        expected_decay = min(0.5, 0.2 + 0.001 * trainer.step)
        assert decay == expected_decay
        result = update_snapshot(name, decay)
        after = cpu(trainer.adapter.get_variant_snapshot(name))
        for old, value, actual in zip(before, live, after, strict=True):
            expected = (old.double() * expected_decay + value.double() * (1 - expected_decay)).to(
                actual.dtype
            )
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        records["snapshot_updates"].append(
            {
                "cycle": trainer.cycle_index,
                "step": trainer.step,
                "decay": decay,
                "before": before,
                "live": live,
                "after": after,
            }
        )
        return result

    def generator(unit):
        active["rewards"] = []
        loss = originals["_generator_boundary_loss"](unit)
        anchor = records["objectives"][-1]
        assert anchor["kind"] == "generator"
        rewards = active.pop("rewards")
        assert len(rewards) == 2
        row = {
            "cycle": trainer.cycle_index,
            "boundary": unit.boundary_index,
            "loss": cpu(loss),
            "anchor": anchor,
            "rewards": rewards,
        }
        records["generator_rewards"].append(row)
        active["pending_generator"] = row
        return loss

    def score_terms(unit):
        terms = originals["_generator_score_terms"](unit)
        active["terms"] = terms
        return terms

    @wraps(query)
    def observed_query(*args, **kwargs):
        result = query(*args, **kwargs)
        if "rewards" in active:
            terms = active["terms"]
            _compare(cpu(args[2]), cpu(terms.noised.state), exact=True)
            _compare(cpu(args[3]), cpu(terms.times), exact=True)
            assert all(not value.requires_grad for value in result.components.values())
            assert (kwargs["role_name"], kwargs["forward_kwargs"]["guidance_scale"]) == active[
                "expected_query"
            ]
            records["reward_queries"].append(
                {
                    "cycle": trainer.cycle_index,
                    "role": kwargs["role_name"],
                    "guidance_scale": kwargs["forward_kwargs"]["guidance_scale"],
                }
            )
            active["query_output"] = result
        return result

    def guidance(batch, terms):
        active["expected_query"] = ("reference", 1.0)
        result = originals["_guidance_reward_direction"](batch, terms)
        assert result is not None
        projected = trainer.adapter.project_velocity_to_clean_state(
            terms.noised.state, terms.times, active["query_output"]
        )
        for name, value in result.components.items():
            expected = 0.7 * (
                terms.x0_real.components[name].float() - projected.components[name].float()
            )
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        active["expected_direction"] = cpu(result.components)
        active["expected_reference"] = cpu(terms.x0_real.components)
        return result

    def learned(batch, terms):
        active["expected_query"] = ("surrogate", 2.0)
        result = originals["_surrogate_reward_direction"](batch, terms)
        projected = trainer.adapter.project_velocity_to_clean_state(
            terms.noised.state, terms.times, active["query_output"]
        )
        for name, value in result[0].components.items():
            torch.testing.assert_close(
                value,
                projected.components[name].float() - terms.x0_real.components[name].float(),
                rtol=0,
                atol=0,
            )
        _compare(cpu(result[1].components), cpu(projected.components), exact=True)
        active["expected_direction"], active["expected_reference"] = cpu(result[0].components), cpu(
            result[1].components
        )
        return result

    def revised_loss(adapter, live, direction, reference, **kwargs):
        _compare(cpu(direction.components), active["expected_direction"], exact=True)
        _compare(cpu(reference.components), active["expected_reference"], exact=True)
        assert all(
            not value.requires_grad
            for state in (direction, reference)
            for value in state.components.values()
        )
        row = {
            "live": cpu(live.components),
            "direction": cpu(direction.components),
            "reference": cpu(reference.components),
            "masks": cpu(live.active_masks),
            **kwargs,
        }
        result = revised(adapter, live, direction, reference, **kwargs)
        row["loss"] = cpu(result)
        expected, _ = direction_reference(row)
        torch.testing.assert_close(row["loss"].double(), expected, rtol=8e-6, atol=1e-7)
        active["rewards"].append(row)
        return result

    # Install inside the TDM observer so its complete fake/replay/renoise evidence remains active.
    with observe_tdm(trainer, records):
        backward = trainer._backward_full_output_loss

        def observed_backward(loss, **kwargs):
            result = backward(loss, **kwargs)
            if "pending_surrogate" in active:
                assert_surrogate(active.pop("pending_surrogate"))
            elif "pending_generator" in active:
                row = active.pop("pending_generator")
                expected_loss, slopes = objective_reference(row["anchor"])
                for coefficient, reward in zip((0.3, 0.7), row["rewards"], strict=True):
                    scalar, gradient = direction_reference(reward)
                    expected_loss += coefficient * scalar
                    for name, value in gradient.items():
                        slopes[name] += coefficient * value
                torch.testing.assert_close(
                    row["loss"].double(), expected_loss, rtol=8e-6, atol=1e-7
                )
                for name, gradient in row["anchor"]["gradients"].items():
                    expected = (slopes[name] * trainer.config.parallelism_args.cp_size / 4).to(
                        gradient.dtype
                    )
                    torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-8)
            else:
                assert_objective(records["objectives"][-1], trainer.config.parallelism_args.cp_size)
            return result

        (
            trainer._surrogate_boundary_loss,
            trainer._boundary_preference_values,
            trainer._diffusion_density_values,
        ) = (surrogate, preference, density)
        trainer._generator_boundary_loss, trainer._generator_score_terms = generator, score_terms
        trainer._guidance_reward_direction, trainer._surrogate_reward_direction = guidance, learned
        trainer._backward_full_output_loss = observed_backward
        trainer.adapter.use_variant_snapshot, trainer.adapter.update_variant_snapshot = (
            use_snapshot,
            update,
        )
        r1.revised_x0_loss, r1.query_score_velocity = revised_loss, observed_query
        try:
            yield
        finally:
            for name, original in originals.items():
                setattr(trainer, name, original)
            trainer.adapter.use_variant_snapshot, trainer.adapter.update_variant_snapshot = (
                snapshot,
                update_snapshot,
            )
            r1.revised_x0_loss, r1.query_score_velocity = revised, query
