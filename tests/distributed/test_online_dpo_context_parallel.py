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

"""Concrete online pair ownership, calculus and CP lifecycle witnesses."""

import copy
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from online_dpo_observations import (
    assert_objectives,
    expected_pairs,
    observe_objectives,
    training_term_identities,
)

from flow_factory.samples import BaseSample, LatentState, MultiModalStepOutput
from flow_factory.trainers.common import context_parallel
from flow_factory.trainers.rl.dpo import DPOTrainer
from tests.trainers.test_decoupled_trainer_parity import _adapter, _dpo_trainer


def sample(source, sample_id, advantage, group=7):
    return BaseSample(
        source=f"source-{source}",
        source_id=source,
        prompt="same prompt",
        prompt_ids=torch.tensor([7]),
        sampling_group_id=group,
        sampling_sample_id=sample_id,
        sampling_group_member_id=sample_id % 2,
        timesteps=torch.tensor([1000.0, 500.0]),
        all_latents=(torch.arange(6).reshape(1, 2, 3).float() / 8 + sample_id / 3),
        latent_index_map=torch.tensor([-1, -1, 0]),
        prompt_embeds=torch.tensor([source + 0.5]),
        extra_kwargs={"advantage": torch.tensor(float(advantage))},
    )


def fixture(cp_size=2, gas=2):
    adapter = _adapter()
    trainer = _dpo_trainer(adapter, beta=3.0)
    trainer.training_args.update(
        per_device_batch_size=1,
        num_inner_epochs=1,
        shuffle_samples=False,
        weighting_scheme="uniform",
        time_shift=1.0,
        timestep_range=(0.1, 0.8),
    )
    trainer.epoch = trainer.step = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.logger = None
    trainer.config = SimpleNamespace(parallelism_args=SimpleNamespace(cp_size=cp_size))
    trainer.accumulate_gradients = nullcontext
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        sync_gradients=False,
        gradient_accumulation_steps=gas,
        backward=lambda loss: (loss / gas).backward(),
    )
    trainer.group_coordinator = SimpleNamespace(
        dp_world_size=1, reduce_sum_across_replicas=lambda x: x
    )
    trainer.advantage_processor = SimpleNamespace(group_on_same_rank=True)
    trainer.log_data = lambda *a, **k: None
    trainer._iter_prefetched_batches = lambda samples, size: (
        BaseSample.stack(samples[i : i + size]) for i in range(0, len(samples), size)
    )
    parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.6, 6).reshape(1, 2, 3))
    active = "policy"
    scopes = []

    @contextmanager
    def reference():
        nonlocal active
        active = "reference"
        scopes.append(active)
        try:
            yield
        finally:
            active = "policy"

    adapter.use_ref_parameters = reference

    def forward_state(**kwargs):
        theta = parameter if active == "policy" else parameter.detach() - 0.23
        value = theta + 0.17 * kwargs["state"].components["latent"]
        return MultiModalStepOutput(velocity=LatentState({"latent": value}))

    adapter.forward_state = forward_state
    samples = [sample(0, 0, -1), sample(0, 1, 1), sample(1, 2, 2), sample(1, 3, -2)]
    return trainer, parameter, samples, scopes


def test_online_dpo_explicit_capability_and_subclass_rejection():
    assert (
        context_parallel.context_parallel_training_capability(DPOTrainer).loss_gradient
        == "full_output"
    )

    class Unqualified(DPOTrainer):
        pass

    with pytest.raises(ValueError, match="does not explicitly declare"):
        context_parallel.context_parallel_training_capability(Unqualified)


@pytest.mark.parametrize("cp_size,gas", [(1, 1), (2, 2), (2, 4)])
def test_real_online_preference_matches_independent_derivatives(monkeypatch, cp_size, gas):
    trainer, parameter, samples, scopes = fixture(cp_size, gas)
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    assert len(rows) == 4 and trainer.adapter.noise_calls == 4
    assert len(scopes) == 4
    assert_objectives(rows, cp_size)
    assert parameter.grad is not None and torch.count_nonzero(parameter.grad)
    assert len({tuple(r["forwards"][0]["ids"]) for r in rows}) == 2


@pytest.mark.parametrize(
    "corruption",
    ["cp_factor", "sign", "zero", "arm", "reference_gradient", "noise", "conditioning"],
)
def test_online_preference_oracle_rejects_corruption(monkeypatch, corruption):
    trainer, _, samples, _ = fixture()
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    rows = copy.deepcopy(rows)
    row = rows[0]
    if corruption in {"cp_factor", "sign", "zero"}:
        row["terms"][0]["gradient"].mul_({"cp_factor": 0.5, "sign": -1, "zero": 0}[corruption])
    elif corruption == "arm":
        row["terms"][0], row["terms"][1] = row["terms"][1], row["terms"][0]
    elif corruption == "reference_gradient":
        row["terms"][2]["requires_grad"] = True
    elif corruption == "noise":
        next(iter(row["noising"][1]["noise"]["components"].values())).add_(1)
    else:
        row["forwards"][3]["ids"] = row["forwards"][2]["ids"]
    with pytest.raises(AssertionError):
        assert_objectives(rows, 2)


def test_pair_selection_sources_ties_singletons_and_order():
    samples = [
        sample(0, 0, 3),
        sample(1, 2, 8),
        sample(0, 1, -2),
        sample(1, 3, -7),
        sample(2, 4, 0),
    ]
    pairs = DPOTrainer._form_pairs_from_advantages(samples)
    assert [(a.sampling_sample_id, b.sampling_sample_id) for a, b in pairs] == [(0, 1), (2, 3)]
    assert expected_pairs(samples) == [
        (("source-0", 0), ("source-0", 1)),
        (("source-1", 2), ("source-1", 3)),
    ]
    ties = [sample(0, 0, 1), sample(0, 1, 1)]
    a, b = DPOTrainer._form_pairs_from_advantages(ties)[0]
    assert a is b is ties[0]  # Existing deterministic first-occurrence tie convention.
    assert DPOTrainer._form_pairs_from_advantages([]) == []


def test_empty_pair_failure_precedes_forward_and_restores_overlap_flag():
    trainer, _, _, _ = fixture()
    with pytest.raises(RuntimeError, match="no valid chosen/rejected pairs"):
        trainer._optimize_reward_overlap_tile(None, [sample(0, 0, 1)], None)
    assert not hasattr(trainer, "_dpo_reward_overlap_tile_active")
    assert trainer.adapter.noise_calls == 0


@pytest.mark.parametrize("pairwise", [False, True])
@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "wrong_id",
        "duplicate_time",
        "missing_term",
        "double_term",
        "wrong_gas",
        "duplicate_expected_id",
    ],
)
def test_tile_accounting_keeps_sample_and_pair_terms_strict(pairwise, corruption):
    from context_parallel_overlap_runner import assert_tile_training_terms

    samples = [sample(0, 0, -1), sample(0, 1, 1)]
    ids = (
        training_term_identities(samples)
        if pairwise
        else [(s.source, s.sampling_sample_id) for s in samples]
    )
    gas = 2 * len(ids)
    terms = [(key, time, gas) for key in ids for time in (200.0, 700.0)]
    losses = torch.zeros(gas)
    if corruption == "wrong_id":
        terms[0] = (("wrong", 999), 200.0, gas)
    if corruption == "duplicate_time":
        terms[1] = (terms[1][0], 200.0, gas)
    if corruption == "missing_term":
        terms.pop()
    if corruption == "double_term":
        terms.append(terms[0])
    if corruption == "wrong_gas":
        gas *= 2
    if corruption == "duplicate_expected_id":
        ids.append(ids[0])
    if corruption is None:
        assert_tile_training_terms(terms, identities=ids, gas=gas, losses=losses)
    else:
        with pytest.raises(AssertionError):
            assert_tile_training_terms(terms, identities=ids, gas=gas, losses=losses)


def test_cuda_flux_online_dpo_ordered_lora(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "flux_online_dpo_fixture",
            "algorithm": "dpo",
            "model_family": "flux",
            "training_mode": "lora",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "cp_size": 2,
            "tp_size": 1,
        },
    )


@pytest.mark.parametrize("sampler", ["rank_local", "global_batch", "subgroup_tile"])
def test_shared_recipe_preserves_explicit_sampler_geometry(tmp_path, sampler, monkeypatch):
    monkeypatch.setattr("flow_factory.hparams.args.get_world_size", lambda: 4)
    monkeypatch.setattr("flow_factory.hparams.training_args._base.get_world_size", lambda: 4)
    import flux_coupled_fixture
    from context_parallel_overlap_runner import make_overlap_config

    options = {
        "algorithm": "grpo-guard",
        "training_mode": "lora",
        "scheduling_mode": "ordered",
        "dp_size": 2,
        "cp_size": 2,
        "tp_size": 1,
        "sampler_type": sampler,
        "subgroup_size": 2 if sampler == "subgroup_tile" else None,
    }
    base = flux_coupled_fixture.make_config(tmp_path, tmp_path, options)
    result = make_overlap_config(base, tmp_path, tmp_path, options)
    assert result.data_args.sampler_type == sampler
    assert result.training_args.group_size == (4 if sampler == "subgroup_tile" else 2)


def test_shared_recipe_rejects_unknown_sampler_before_config_construction(tmp_path):
    from context_parallel_overlap_runner import make_overlap_config

    with pytest.raises(ValueError, match="Unsupported native sampler_type"):
        make_overlap_config(None, tmp_path, tmp_path, {"sampler_type": "typo"})


def test_pair_recipe_preserves_default_sampler_and_rejects_cross_rank_overlap(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("flow_factory.hparams.args.get_world_size", lambda: 4)
    monkeypatch.setattr("flow_factory.hparams.training_args._base.get_world_size", lambda: 4)
    import flux_online_dpo_fixture
    from context_parallel_overlap_runner import make_overlap_config

    options = {
        "algorithm": "dpo",
        "training_mode": "lora",
        "scheduling_mode": "ordered",
        "dp_size": 2,
        "cp_size": 2,
        "tp_size": 1,
    }
    base = flux_online_dpo_fixture.make_config(tmp_path, tmp_path, options)
    result = make_overlap_config(base, tmp_path, tmp_path, options)
    assert result.data_args.sampler_type == "rank_local"
    assert result.training_args.beta == 3 and result.training_args.num_train_timesteps == 2
    DPOTrainer.validate_reward_optimization_overlap(result)
    result.data_args.sampler_type = "global_batch"
    with pytest.raises(ValueError, match="rank.local"):
        DPOTrainer.validate_reward_optimization_overlap(result)


def test_global_batch_advantage_oracle_counts_replicas_and_rejects_local_mislabeling():
    from context_parallel_overlap_runner import _expected_advantages

    rows = [
        [{"source": "a", "group": 7, "member": rank, "sample": rank, "score": float(rank * 2 + 1)}]
        for rank in range(2)
    ]
    gathered = [copy.deepcopy(rows[rank]) for rank in (0, 0, 1, 1)]
    kwargs = dict(model_degree=2, dp_size=2, group_size=2, subgroup_size=None)
    expected = _expected_advantages(gathered, **kwargs, sampler_type="global_batch")
    assert expected == {("a", 0): -1.0, ("a", 1): 1.0}
    with pytest.raises(AssertionError):
        _expected_advantages(gathered, **kwargs)
    wrong = copy.deepcopy(gathered)
    wrong[1][0]["score"] += 1
    with pytest.raises(AssertionError):
        _expected_advantages(wrong, **kwargs, sampler_type="global_batch")
    local = [[*rows[0], *rows[1]], [*rows[0], *rows[1]], [], []]
    with pytest.raises(AssertionError):
        _expected_advantages(local, **kwargs, sampler_type="global_batch")


@pytest.mark.parametrize("operand", ["state", "times"])
def test_online_pair_observer_rejects_in_place_forward_input_mutation(monkeypatch, operand):
    trainer, _, samples, _ = fixture()
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    original = trainer.adapter.forward_state

    def corrupt(**kwargs):
        result = original(**kwargs)
        value = (
            kwargs["state"].components["latent"]
            if operand == "state"
            else kwargs["times"].timestep["latent"]
        )
        value.add_(1)
        return result

    trainer.adapter.forward_state = corrupt
    with pytest.raises(AssertionError):
        with observe_objectives(trainer):
            trainer.optimize(samples)
