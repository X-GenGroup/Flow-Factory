# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""R1 query guidance and concrete CP admission contracts."""

from types import SimpleNamespace

import pytest
import torch

from flow_factory.models.trajectory_bridge.dispatch import build_forward_state_kwargs
from flow_factory.samples import LatentState
from flow_factory.trainers.common.context_parallel import ContextParallelTrainingCapability
from flow_factory.trainers.distillation import tdm_r1 as module
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer


@pytest.mark.parametrize("key", ["guidance_scale", "cfg_text_scale"])
@pytest.mark.parametrize("student,batch_scale", [(1.0, None), (2.0, None), (2.0, 4.0), (2.0, 1.0)])
def test_guidance_direction_queries_actual_conditional_endpoint(
    monkeypatch, key, student, batch_scale
):
    trainer = object.__new__(TDMR1Trainer)
    trainer.training_args = SimpleNamespace(
        guidance_scale=student,
        real_guidance_scale=3.0,
        get_reference_guidance_scale=lambda: 3.0,
        cfg_reward_scale=3.5,
        extra_kwargs={"other_input": 9},
    )
    trainer.adapter = SimpleNamespace(
        trajectory_component_order=("latent",),
        reference_guidance_kwargs=lambda scale: {key: scale},
        project_velocity_to_clean_state=lambda state, times, velocity: velocity,
    )
    trainer.autocast = lambda: None
    batch = {} if batch_scale is None else {key: batch_scale}
    original = dict(batch)
    effective = 3.0 if batch_scale is None else batch_scale
    state = LatentState({"latent": torch.zeros(1, 2)})
    terms = SimpleNamespace(
        noised=SimpleNamespace(state=state),
        times=object(),
        boundary_state=state,
        x0_real=LatentState({"latent": torch.full((1, 2), 2 + 5 * effective)}),
    )
    calls = []

    def query(adapter, actual_batch, actual_state, times, **kwargs):
        resolved = build_forward_state_kwargs(adapter, actual_batch, kwargs["forward_kwargs"])
        assert actual_batch is batch and actual_state is state and times is terms.times
        assert kwargs["role_name"] == "reference" and resolved["other_input"] == 9
        calls.append(resolved[key])
        return LatentState({"latent": torch.full((1, 2), 2 + 5 * resolved[key])})

    monkeypatch.setattr(module, "query_score_velocity", query)
    actual = trainer._guidance_reward_direction(batch, terms)
    assert batch == original
    if effective <= 1:
        assert actual is None and not calls
    else:
        assert calls == [1.0]
        torch.testing.assert_close(actual.components["latent"], torch.full((1, 2), 17.5))
        assert not actual.components["latent"].requires_grad
        if student != 1 and batch_scale is None:
            wrong = 3.5 * (effective - student) * 5 / (effective - 1)
            assert wrong != pytest.approx(17.5)


def test_guidance_direction_rejects_ambiguous_effective_scales():
    trainer = object.__new__(TDMR1Trainer)
    trainer.training_args = SimpleNamespace(
        guidance_scale=1.0,
        real_guidance_scale=3.0,
        get_reference_guidance_scale=lambda: 3.0,
        cfg_reward_scale=3.5,
        extra_kwargs={},
    )
    trainer.adapter = SimpleNamespace(
        reference_guidance_kwargs=lambda scale: {"image_cfg": scale, "audio_cfg": scale}
    )
    with pytest.raises(ValueError, match="consistent finite"):
        trainer._guidance_reward_direction({"image_cfg": 2.0, "audio_cfg": 3.0}, None)


def test_r1_declares_its_own_full_output_cp_contract():
    assert (
        TDMR1Trainer.__dict__["context_parallel_training_capability"]
        == ContextParallelTrainingCapability()
    )


@pytest.mark.parametrize("key", ["guidance_scale", "cfg_text_scale"])
def test_cfg_free_reference_keeps_no_guidance_reward_without_mapped_override(key):
    trainer = object.__new__(TDMR1Trainer)
    trainer.training_args = SimpleNamespace(
        guidance_scale=1.0,
        real_guidance_scale=None,
        get_reference_guidance_scale=lambda: 1.0,
        cfg_reward_scale=3.5,
        extra_kwargs={},
    )
    trainer.adapter = SimpleNamespace(reference_guidance_kwargs=lambda scale: {key: scale})
    assert trainer._guidance_reward_direction({}, None) is None


def test_nontrivial_unresolved_mapped_reference_guidance_fails_before_query():
    trainer = object.__new__(TDMR1Trainer)
    trainer.training_args = SimpleNamespace(
        guidance_scale=2.0,
        real_guidance_scale=None,
        get_reference_guidance_scale=lambda: 2.0,
        cfg_reward_scale=3.5,
        extra_kwargs={},
    )
    trainer.adapter = SimpleNamespace(
        reference_guidance_kwargs=lambda scale: {"cfg_text_scale": scale}
    )
    with pytest.raises(ValueError, match="cannot infer"):
        trainer._guidance_reward_direction({}, None)


class _RoutedScoreAdapter:
    """Real registry/reference scopes with distinct scalar full-weight roles."""

    from flow_factory.models.abc import BaseAdapter

    _require_variant_registry = BaseAdapter._require_variant_registry
    use_component_variant = BaseAdapter.use_component_variant
    use_ref_parameters = BaseAdapter.use_ref_parameters
    _ema_tracked_parameters = BaseAdapter._ema_tracked_parameters
    trainable_component_names = ("transformer",)
    trajectory_component_order = ("latent",)
    target_module_map = {"transformer": "all"}

    def __init__(self):
        from flow_factory.ema import EMAModuleWrapper
        from flow_factory.models.variants import ComponentVariantRegistry, ComponentVariantSpec

        self.model_args = SimpleNamespace(finetune_type="full")
        self.module = torch.nn.Linear(1, 1, bias=False)
        registry = self.component_variant_registry = ComponentVariantRegistry(self)
        for name in ("generator", "surrogate"):
            registry.declare(
                ComponentVariantSpec(
                    name=name,
                    storage_mode="full",
                    component_routes={
                        "transformer": "transformer" if name == "generator" else "surrogate"
                    },
                )
            )
        registry.materialize(("generator", "surrogate"))
        self.base = registry.parameters("generator")[0]
        self.surrogate = registry.parameters("surrogate")[0]
        with torch.no_grad():
            self.base.fill_(2)
            self.surrogate.fill_(5)
        self._ref_ema = EMAModuleWrapper([self.base])
        registry.add_snapshot("surrogate", "slow")
        with torch.no_grad():
            self.base.fill_(3)
            self.surrogate.fill_(7)
        self.fail_reference = False
        self.seen = []

    def get_component(self, name):
        return self.module

    def forward_state(self, *, state, **kwargs):
        registry = self.component_variant_registry
        route = registry.resolve_route(registry.active_variant, "transformer")
        self.seen.append(
            (registry.active_variant, float(registry.bundle_members()[route].weight.detach()))
        )
        if self.fail_reference and not torch.is_grad_enabled():
            raise RuntimeError("reference forward failed")
        return SimpleNamespace(
            velocity=LatentState(
                {"latent": registry.bundle_members()[route](state.components["latent"])}
            )
        )

    def reduce_latent_values(self, values, *, state):
        return values["latent"].flatten(1).mean(1)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_reference_score_uses_base_snapshot_inside_live_or_slow_surrogate(slow, fail):
    from contextlib import nullcontext

    adapter = _RoutedScoreAdapter()
    registry = adapter.component_variant_registry
    trainer = object.__new__(TDMR1Trainer)
    trainer.adapter = adapter
    trainer.autocast = nullcontext
    trainer._replay_forward_kwargs = trainer._reference_forward_kwargs = lambda batch: {}
    state = LatentState({"latent": torch.ones(1, 1)})
    noised = SimpleNamespace(
        state=state, target_velocity=LatentState({"latent": torch.zeros(1, 1)})
    )
    before = [
        (id(p), p.detach().clone(), p.requires_grad)
        for name in registry.variant_names
        for p in registry.parameters(name)
    ]
    with adapter.use_component_variant("surrogate"):
        with registry.use_snapshot("slow") if slow else nullcontext():
            adapter.fail_reference = fail
            if fail:
                with pytest.raises(RuntimeError, match="reference forward failed"):
                    trainer._score_boundary_values({}, noised, object(), "surrogate")
            else:
                live, reference = trainer._score_boundary_values({}, noised, object(), "surrogate")
                torch.testing.assert_close(live, torch.tensor([25.0 if slow else 49.0]))
                torch.testing.assert_close(reference, torch.tensor([4.0]))
                assert live.requires_grad and not reference.requires_grad
                live.sum().backward()
                assert adapter.surrogate.grad.item() == (10.0 if slow else 14.0)
                assert adapter.base.grad is None
            assert registry.active_variant == "surrogate"
            assert adapter.surrogate.item() == (5.0 if slow else 7.0)
            assert adapter.base.item() == 3.0
    assert adapter.seen == [("surrogate", 5.0 if slow else 7.0), ("generator", 2.0)]
    assert registry.active_variant == "generator"
    after = [p for name in registry.variant_names for p in registry.parameters(name)]
    for (identity, value, trainable), actual in zip(before, after, strict=True):
        assert identity == id(actual) and actual.requires_grad == trainable
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
