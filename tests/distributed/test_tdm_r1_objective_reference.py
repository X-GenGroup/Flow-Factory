# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Actual R1 surrogate path against independently derived masked DSM derivatives."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from tdm_r1_objective_reference import assert_surrogate, surrogate_reference

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import LatentState
from flow_factory.trainers.common import context_parallel
from flow_factory.trainers.common.context_parallel import full_output_context_parallel_loss
from flow_factory.trainers.distillation.group_preference import GroupPreferenceBatch
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer


class Adapter(BaseAdapter):
    trajectory_component_order = ("image", "audio")

    def load_pipeline(self):
        raise NotImplementedError

    def decode_latents(self, *args, **kwargs):
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def _row(indices, *, clip_range, threshold, time_weighting, cp, gas):
    target = {"image": torch.zeros(4, 3), "audio": torch.zeros(4, 2)}
    masks = {
        "image": torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype=torch.bool),
        "audio": torch.tensor([[1, 0], [1, 1], [0, 1], [1, 1]], dtype=torch.bool),
    }

    def predictions(values):
        return {
            name: torch.sqrt(torch.tensor(values)).unsqueeze(1).expand_as(value).clone()
            for name, value in target.items()
        }

    full = dict(
        target=target,
        masks=masks,
        new=predictions([0.6, 1.4, 0.97, 0.9]),
        reference=predictions([1.0] * 4),
        slow=predictions([1.0, 1.0, 0.99, 0.94]),
        advantage=torch.tensor([1.0, -0.7, 0.2, -1.3]),
        group_keys=[[-1, 10], [-1, 10], [-1, 20], [-1, 20]],
    )
    row = {
        key: (
            {name: value[indices].clone() for name, value in values.items()}
            if isinstance(values, dict)
            else (
                values[indices].clone()
                if isinstance(values, torch.Tensor)
                else [values[i] for i in indices]
            )
        )
        for key, values in full.items()
    }
    row.update(
        preference_beta=2.3,
        reference_beta=0.17,
        clip_range=clip_range,
        reference_threshold=threshold,
        boundary_index=2,
        num_steps=2,
        time_weighting=time_weighting,
        group_size=2,
        cp_size=cp,
        gas=gas,
    )
    return row


@pytest.mark.parametrize("layout", ["rank_local", "equal_share", "packed"])
@pytest.mark.parametrize(
    "clip_range,threshold", [(0.1, 0.05), (0.0, 0.05), (0.1, 0.0), (0.0, 10.0)]
)
@pytest.mark.parametrize("cp,gas,time_weighting", [(1, 1, False), (2, 4, True)])
def test_full_surrogate_objective_and_raw_derivatives(
    monkeypatch, layout, clip_range, threshold, cp, gas, time_weighting
):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp, 0))
    indices = (
        [0, 1, 2, 3] if layout == "rank_local" else [0, 2] if layout == "equal_share" else [0, 1]
    )
    row = _row(
        indices,
        clip_range=clip_range,
        threshold=threshold,
        time_weighting=time_weighting,
        cp=cp,
        gas=gas,
    )
    peer = None
    if layout != "rank_local":
        peer = _row(
            [i for i in range(4) if i not in indices],
            clip_range=clip_range,
            threshold=threshold,
            time_weighting=time_weighting,
            cp=cp,
            gas=gas,
        )
        row["logical_rows"] = [row, peer]
    trainer = object.__new__(TDMR1Trainer)
    trainer.adapter = object.__new__(Adapter)
    trainer.training_args = SimpleNamespace(
        surrogate_preference_beta=row["preference_beta"],
        surrogate_reference_beta=row["reference_beta"],
        surrogate_clip_range=clip_range,
        surrogate_reference_threshold=threshold,
        use_time_weighting=time_weighting,
        num_inference_steps=2,
    )
    trainer.accelerator = SimpleNamespace(parallelism_config=SimpleNamespace(cp_size=cp))
    names = trainer.adapter.trajectory_component_order
    state = LatentState(row["target"], active_masks=row["masks"])
    target = LatentState(row["target"])
    live = LatentState({name: row["new"][name].clone().requires_grad_() for name in names})
    reference = LatentState(row["reference"])
    slow = LatentState(row["slow"])
    values = trainer._diffusion_density_values(target, live, state=state)
    values.retain_grad()
    ref_values = trainer._diffusion_density_values(target, reference, state=state)
    slow_values = trainer._diffusion_density_values(target, slow, state=state)
    row["dsm_endpoints"] = dict(new=values.detach(), reference=ref_values, slow=slow_values)
    context = object()
    in_snapshot = False

    @contextmanager
    def snapshot(name):
        nonlocal in_snapshot
        assert name == "tdm_r1_slow_surrogate"
        in_snapshot = True
        try:
            yield
        finally:
            in_snapshot = False

    trainer.adapter.use_variant_snapshot = snapshot
    trainer._boundary_score_context = lambda unit: context

    def preference_values(unit, *, trainable_role, context=None):
        assert trainable_role == "surrogate" and context is not None
        return (slow_values, ref_values) if in_snapshot else (values, ref_values)

    trainer._boundary_preference_values = preference_values
    group_indices = torch.tensor([0 if key[-1] == 10 else 1 for key in row["group_keys"]])
    batch = GroupPreferenceBatch(
        group_indices,
        2,
        2,
        row["advantage"],
        reduce_across_ranks=peer is not None,
        allow_sparse_local_groups=layout == "packed",
    )
    trainer._group_preference_batch = lambda unit, actual: batch

    def reduce_sum(local):
        if peer is None:
            raise AssertionError("rank-local preference may not communicate")
        peer_target = LatentState(peer["target"], active_masks=peer["masks"])
        peer_live = trainer._diffusion_density_values(
            peer_target, LatentState(peer["new"]), state=peer_target
        )
        peer_reference = trainer._diffusion_density_values(
            peer_target, LatentState(peer["reference"]), state=peer_target
        )
        expected = torch.zeros_like(local)
        for key, advantage, delta in zip(
            peer["group_keys"], peer["advantage"], peer_live - peer_reference, strict=True
        ):
            expected[0 if key[-1] == 10 else 1] += advantage * row["preference_beta"] * delta / 2
        return local + expected

    trainer.group_coordinator = SimpleNamespace(reduce_sum=reduce_sum)
    loss = trainer._surrogate_boundary_loss(SimpleNamespace(boundary_index=2))
    (full_output_context_parallel_loss(loss, trainer.accelerator) / gas).backward()
    row.update(
        loss=loss.detach(),
        upstream=values.grad,
        gradients={name: value.grad for name, value in live.components.items()},
    )
    expected = assert_surrogate(row)
    assert not in_snapshot
    assert any(torch.count_nonzero(value) for value in row["gradients"].values())
    if clip_range and threshold == 0.05:
        assert expected["trust"].any()
    # Missing anchor or an extra preference-beta derivative must be distinguishable.
    altered = dict(row, reference_beta=0.0)
    wrong = surrogate_reference(altered)
    assert not torch.allclose(wrong["upstream"], expected["upstream"], atol=1e-7)
    assert not torch.allclose(
        expected["upstream"] * row["preference_beta"], expected["upstream"], atol=1e-7
    )


@pytest.mark.parametrize("huber", [False, True])
@pytest.mark.parametrize("guidance", [False, True])
def test_complete_generator_three_directions_keep_anchor_and_detached_normalizers(
    monkeypatch, huber, guidance
):
    from tdm_objective_reference import objective_reference
    from tdm_r1_objective_reference import direction_reference

    from flow_factory.trainers.distillation.distribution_matching import tdm_generator_loss

    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer = object.__new__(TDMR1Trainer)
    trainer.adapter = object.__new__(Adapter)
    trainer.training_args = SimpleNamespace(tdm_weight=0.3, use_huber=huber, huber_c=0.1)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    live = LatentState(
        {
            "image": torch.tensor([[0.8, -0.4, 1.2], [0.2, 0.7, -0.9]], requires_grad=True),
            "audio": torch.tensor([[0.5, -1.1], [1.4, 0.3]], requires_grad=True),
        },
        active_masks={
            "image": torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool),
            "audio": torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
        },
    )
    real = LatentState(
        {name: value.detach() * 0.3 + 0.7 for name, value in live.components.items()},
        active_masks=live.active_masks,
    )
    fake = LatentState(
        {name: value.detach() * 0.6 - 0.2 for name, value in live.components.items()},
        active_masks=live.active_masks,
    )
    conditional = LatentState(
        {name: value.detach() * -0.4 + 0.2 for name, value in live.components.items()},
        active_masks=live.active_masks,
    )
    learned = LatentState(
        {name: value.detach() * 0.4 - 0.6 for name, value in live.components.items()},
        active_masks=live.active_masks,
    )
    cfg_direction = LatentState(
        {
            name: 0.7 * (real.components[name] - conditional.components[name])
            for name in live.component_names
        },
        active_masks=live.active_masks,
    )
    learned_direction = LatentState(
        {name: learned.components[name] - real.components[name] for name in live.component_names},
        active_masks=live.active_masks,
    )
    anchor = tdm_generator_loss(trainer.adapter, live, real, fake, use_huber=huber, huber_c=0.1)
    terms = SimpleNamespace(loss=anchor, boundary_state=live, x0_real=real)
    trainer._generator_score_terms = lambda unit: terms
    trainer._stack_replay_unit = lambda samples: None
    trainer._guidance_reward_direction = lambda batch, terms: cfg_direction if guidance else None
    trainer._surrogate_reward_direction = lambda batch, terms: (learned_direction, learned)
    actual = trainer._generator_boundary_loss(SimpleNamespace(samples=()))
    (full_output_context_parallel_loss(actual, trainer.accelerator) / 4).backward()
    row = dict(
        kind="generator",
        live={k: v.detach() for k, v in live.components.items()},
        real=real.components,
        fake=fake.components,
        masks=live.active_masks,
        use_huber=huber,
        huber_c=0.1,
    )
    expected, slopes = objective_reference(row)
    anchor_only = expected.clone()
    for coefficient, direction, reference in (
        (0.3 if guidance else 0.0, cfg_direction, real),
        (0.7, learned_direction, learned),
    ):
        scalar, derivatives = direction_reference(
            {**row, "direction": direction.components, "reference": reference.components}
        )
        expected += coefficient * scalar
        for name, value in derivatives.items():
            slopes[name] += coefficient * value
    torch.testing.assert_close(actual.double(), expected, rtol=8e-6, atol=1e-7)
    assert not torch.isclose(expected, anchor_only, atol=1e-5)
    for name, value in live.components.items():
        torch.testing.assert_close(value.grad.double(), slopes[name] / 2, rtol=8e-6, atol=1e-7)
        assert not torch.allclose(value.grad.double(), slopes[name], atol=1e-5)


def test_reward_direction_reference_retains_public_fp32_near_zero_huber_cancellation():
    from tdm_r1_objective_reference import direction_reference

    from flow_factory.trainers.distillation.distribution_matching import revised_x0_loss

    adapter = object.__new__(Adapter)
    live = LatentState(
        {
            "image": torch.ones(1, 3, requires_grad=True),
            "audio": torch.ones(1, 2, requires_grad=True),
        }
    )
    direction = LatentState(
        {name: torch.full_like(value, 1e-6) for name, value in live.components.items()}
    )
    reference = LatentState(
        {name: torch.zeros_like(value) for name, value in live.components.items()}
    )
    actual = revised_x0_loss(adapter, live, direction, reference, use_huber=True, huber_c=0.1)
    actual.backward()
    expected, derivatives = direction_reference(
        dict(
            live={k: v.detach() for k, v in live.components.items()},
            direction=direction.components,
            reference=reference.components,
            use_huber=True,
            huber_c=0.1,
        )
    )
    assert actual.item() == expected.item() == 0.0
    for name, value in live.components.items():
        torch.testing.assert_close(value.grad.double(), derivatives[name], rtol=3e-6, atol=1e-12)
        assert torch.count_nonzero(value.grad)
