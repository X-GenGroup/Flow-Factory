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


"""Real DGPO owner checks against independent group and derivative expectations."""

import copy
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from dgpo_objective_reference import assert_objective, local_preference, objective_reference

from flow_factory.samples import LatentState, NoisedState
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.common import context_parallel
from flow_factory.trainers.rl.dgpo import DGPOTrainer
from tests.trainers.test_dgpo_crd_parity import _adapter, _dgpo_trainer


def _objective_case(
    monkeypatch,
    *,
    cp_size=2,
    gas=4,
    dtype=torch.float64,
    velocity_dtype=None,
    clip_dsm=True,
    clip_kl=True,
    use_ema_ref=True,
    kl_beta=0.03,
    rank=0,
    kl_compute_dtype=None,
    advantage_dtype=None,
):
    velocity_dtype = velocity_dtype or dtype
    values = torch.tensor([0.2, 1.0, 0.4, 1.2, 0.8, 0.3], dtype=velocity_dtype)[:, None]
    current = values * torch.tensor([[1.0, 0.5, -0.7, 3.0]], dtype=velocity_dtype)
    old = torch.tensor([1.0, 0.2, 0.4, 1.2, 0.7, 0.3], dtype=velocity_dtype)[
        :, None
    ] * torch.tensor([[1.0, 0.5, -0.7, 3.0]], dtype=velocity_dtype)
    reference = current + 0.13
    target = torch.zeros(6, 4, dtype=dtype)
    mask = torch.tensor([[True, True, True, False]] * 6)
    advantages = torch.tensor([1.0, -1.0, 0.8, -0.8, 0.6, -0.6], dtype=advantage_dtype or dtype)
    keys = [("a", 5), ("a", 5), ("b", 5), ("b", 5), ("a", 9), ("a", 9)]
    shared = dict(
        group_size=2,
        dpo_beta=4.0,
        use_ema_ref=use_ema_ref,
        clip_dsm=clip_dsm,
        clip_kl=clip_kl,
        clip_range=(-0.2, 0.2),
        kl_beta=kl_beta,
        cp_size=cp_size,
        gas=gas,
        kl_dtype=torch.promote_types(velocity_dtype, velocity_dtype),
    )
    rows = []
    for offset in (0, 1):
        rows.append(
            {
                **shared,
                "new": {"latent": current[offset::2].clone()},
                "old": {"latent": old[offset::2].clone()},
                "reference": {"latent": reference[offset::2].clone()},
                "target": {"latent": target[offset::2].clone()},
                "masks": {"latent": mask[offset::2].clone()},
                "advantage": advantages[offset::2].clone(),
                "group_keys": keys[offset::2],
            }
        )
    row = {**rows[rank], "logical_rows": rows}
    if kl_compute_dtype is not None:
        row["kl_dtype"] = kl_compute_dtype
    parameter = row["new"]["latent"].clone().requires_grad_()
    adapter = _adapter()
    trainer = _dgpo_trainer(
        adapter,
        clip_dsm=clip_dsm,
        clip_kl=clip_kl,
        use_ema_ref=use_ema_ref,
        kl_beta=kl_beta,
        dpo_beta=4.0,
    )
    for peer in rows:
        peer_noised = NoisedState(
            state=LatentState(
                {"latent": torch.zeros_like(peer["target"]["latent"])}, active_masks=peer["masks"]
            ),
            noise=LatentState(peer["target"]),
            target_velocity=LatentState(peer["target"]),
        )
        peer["dsm_endpoints"] = {
            name: trainer._compute_dsm_loss(
                peer_noised.target_velocity, LatentState(peer[name]), peer_noised
            ).detach()
            for name in ("new", "old", "reference")
        }
    row["dsm_endpoints"] = rows[rank]["dsm_endpoints"]
    trainer.accelerator = SimpleNamespace(
        sync_gradients=False,
        gradient_accumulation_steps=gas,
        backward=lambda loss: (loss / gas).backward(),
    )
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    if kl_compute_dtype is not None:
        # CPU reproduction of CUDA autocast's pow promotion, preserving the
        # native low-precision subtraction. GPU observations record the real dtype.
        def promoted_kl(model, reference, noised):
            values = {
                name: (value - reference.components[name]).to(kl_compute_dtype).square()
                for name, value in model.components.items()
            }
            return adapter.reduce_latent_values(values, state=noised.state)

        trainer._velocity_kl = promoted_kl
    ordered_keys = sorted(set(keys))
    indices = torch.tensor([ordered_keys.index(key) for key in row["group_keys"]])
    peer = rows[1 - rank]
    remote = torch.zeros(len(ordered_keys), dtype=torch.promote_types(dtype, advantages.dtype))
    for key, value in zip(peer["group_keys"], local_preference(peer), strict=True):
        remote[ordered_keys.index(key)] += value.to(remote.dtype)
    calls = []

    def reduce_sum(local):
        assert not local.requires_grad
        calls.append(local.detach().clone())
        return local + remote

    trainer.group_coordinator = SimpleNamespace(reduce_sum=reduce_sum)
    noised = NoisedState(
        state=LatentState({"latent": torch.zeros_like(target[rank::2])}, active_masks=row["masks"]),
        noise=LatentState({"latent": target[rank::2]}),
        target_velocity=LatentState(row["target"]),
    )
    model, old_state, reference_state = (
        LatentState({"latent": value})
        for value in (parameter, row["old"]["latent"], row["reference"]["latent"])
    )
    dsm = trainer._compute_dsm_loss(noised.target_velocity, model, noised)
    info = defaultdict(list)
    clipped, dsm = trainer._maybe_clip_dsm(
        dsm, old_state, noised.target_velocity, row["advantage"], info, noised
    )
    loss = trainer._compute_group_dgpo_loss(
        old_state if use_ema_ref else reference_state,
        noised.target_velocity,
        row["advantage"],
        {"local_group_indices": indices, "num_groups": len(ordered_keys)},
        dsm,
        noised,
    )
    trainer._apply_total_loss_and_backward(
        dgpo_loss=loss,
        model_v=model,
        ref_v=reference_state,
        should_clip=clipped,
        loss_info=info,
        noised=noised,
    )
    row.update(
        loss=info["loss"][0],
        gradients={"latent": parameter.grad},
        global_sums=calls[0] + remote,
        dense_group_keys=ordered_keys,
    )
    if clipped is not None:
        row["clipped"] = clipped
    return row


@pytest.mark.parametrize("cp_size,gas", [(1, 1), (2, 4)])
@pytest.mark.parametrize(
    "clip_dsm,clip_kl", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("use_ema_ref", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_dgpo_real_group_objective_matches_detached_derivative(
    monkeypatch, cp_size, gas, clip_dsm, clip_kl, use_ema_ref, rank
):
    row = _objective_case(
        monkeypatch,
        cp_size=cp_size,
        gas=gas,
        clip_dsm=clip_dsm,
        clip_kl=clip_kl,
        use_ema_ref=use_ema_ref,
        rank=rank,
    )
    expected = assert_objective(row)
    assert set(expected["group_sums"]) == {("a", 5), ("a", 9), ("b", 5)}
    assert row["gradients"]["latent"][:, 3].count_nonzero() == 0
    if clip_dsm or clip_kl:
        assert expected["clipped"].any() and not expected["clipped"].all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dgpo_oracle_rejects_wrong_cp_and_gradient_sign(monkeypatch, dtype):
    row = _objective_case(monkeypatch, dtype=dtype, clip_dsm=False, clip_kl=False)
    assert_objective(row)
    for factor in (0.0, 0.5, 2.0, -1.0):
        broken = copy.deepcopy(row)
        broken["gradients"]["latent"] *= factor
        with pytest.raises(AssertionError):
            assert_objective(broken)


def test_dgpo_oracle_rejects_duplicate_model_peer_contributions(monkeypatch):
    row = _objective_case(monkeypatch, clip_dsm=False, clip_kl=False)
    broken = copy.deepcopy(row)
    broken["logical_rows"] *= 4
    with pytest.raises(AssertionError):
        objective_reference(broken)


def test_dgpo_group_weight_is_not_sigmoid_of_each_replica(monkeypatch):
    row = _objective_case(monkeypatch, clip_dsm=False, clip_kl=False)
    expected = objective_reference(row)
    local = local_preference(row)
    assert not torch.allclose(torch.sigmoid(local), expected["weights"])
    broken = copy.deepcopy(row)
    broken["loss"] += 0.25
    with pytest.raises(AssertionError):
        assert_objective(broken)


def test_dgpo_context_parallel_declaration_is_concrete():
    capability = context_parallel.context_parallel_training_capability(DGPOTrainer)
    assert capability.loss_gradient == "full_output"
    assert capability.checkpoint_boundary == "drained_acquisition"

    class Derived(DGPOTrainer):
        pass

    with pytest.raises(ValueError, match="explicitly declare"):
        context_parallel.context_parallel_training_capability(Derived)


@pytest.mark.parametrize(
    "clip_dsm,clip_kl,use_ema_ref",
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True)],
)
def test_dgpo_runtime_child_declared_only_when_consumed(clip_dsm, clip_kl, use_ema_ref):
    trainer = object.__new__(DGPOTrainer)
    trainer.training_args = SimpleNamespace(
        clip_dsm=clip_dsm, clip_kl=clip_kl, use_ema_ref=use_ema_ref
    )
    assert trainer._algorithm_runtime_child_names() == (
        ("ema_ref",) if clip_dsm or clip_kl or use_ema_ref else ()
    )


@pytest.mark.parametrize(
    "step,off_policy,required,expected",
    [
        (0, False, True, "policy"),
        (0, True, True, "ema"),
        (1, False, True, "policy"),
        (2, False, True, "ema_ref"),
        (2, True, True, "ema_ref"),
        (2, True, False, "ema"),
    ],
)
def test_dgpo_sampling_switch_is_strict_and_restores_on_exception(
    step, off_policy, required, expected
):
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    trainer.step, trainer.switch_ema_ref = step, 1
    trainer.off_policy, trainer._requires_ema_ref = off_policy, required
    with pytest.raises(RuntimeError, match="sampling interruption"):
        with trainer.sampling_context():
            assert adapter.active_scope == expected
            raise RuntimeError("sampling interruption")
    assert adapter.active_scope == "policy"


@pytest.mark.parametrize("mode", ["consume", "leftover", "raise"])
def test_dgpo_overlap_cache_consumption_and_failure_cleanup(monkeypatch, mode):
    trainer = _dgpo_trainer(_adapter())
    cached = object()
    trainer._reward_overlap_group_infos_for_tile = lambda _: [cached]

    def optimize(owner, tile, samples, context):
        assert owner._dgpo_reward_overlap_group_infos == [cached]
        if mode == "consume":
            owner._dgpo_reward_overlap_group_infos.pop()
        if mode == "raise":
            raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(BaseTrainer, "_optimize_reward_overlap_tile", optimize)
    if mode == "consume":
        trainer._optimize_reward_overlap_tile(SimpleNamespace(tile_id=3), [], None)
    else:
        with pytest.raises(
            RuntimeError, match="did not consume" if mode == "leftover" else "injected"
        ):
            trainer._optimize_reward_overlap_tile(SimpleNamespace(tile_id=3), [], None)
    assert not hasattr(trainer, "_dgpo_reward_overlap_group_infos")


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("clip_dsm,clip_kl", [(False, False), (True, True)])
def test_dgpo_bf16_velocity_fp32_target_and_kl_cast_boundaries(
    monkeypatch, rank, clip_dsm, clip_kl
):
    row = _objective_case(
        monkeypatch,
        dtype=torch.float32,
        velocity_dtype=torch.bfloat16,
        kl_compute_dtype=torch.float32,
        rank=rank,
        clip_dsm=clip_dsm,
        clip_kl=clip_kl,
    )
    assert_objective(row)


def test_dgpo_native_recipe_preserves_global_groups_and_state_contract(monkeypatch, tmp_path):
    from context_parallel_overlap_runner import make_overlap_config
    from wan_dgpo_fixture import make_config

    from flow_factory.models.registry import get_model_adapter_class
    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config

    monkeypatch.setenv("WORLD_SIZE", "8")
    options = dict(
        dp_size=2,
        cp_size=2,
        tp_size=2,
        training_mode="full",
        algorithm="dgpo",
        scheduling_mode="ordered",
        sampler_type="global_batch",
    )
    config = make_overlap_config(
        make_config(tmp_path, tmp_path / "results", options),
        tmp_path,
        tmp_path / "results",
        options,
    )
    DGPOTrainer.validate_reward_optimization_overlap(config)
    validate_context_parallel_config(
        config, get_model_adapter_class(config.model_args.model_type), DGPOTrainer
    )
    train = config.training_args
    assert config.data_args.sampler_type == "global_batch"
    assert train.group_size == 2 and train.gradient_accumulation_steps == 4
    assert train.num_train_timesteps == 2 and train.use_shared_noise
    assert train.use_ema_ref and train.clip_dsm and train.clip_kl and train.kl_beta > 0
    assert train.off_policy and train.switch_ema_ref == 0
    assert (
        config.parallelism_args.sequence_parallel and config.scheduler_args.dynamics_type == "ODE"
    )


def test_dgpo_observer_restores_methods_and_module_forward_after_failure():
    from dgpo_observations import observe_objectives

    from flow_factory.trainers.rl import dgpo

    trainer = _dgpo_trainer(_adapter())
    trainer.adapter._ref_ema = SimpleNamespace(ema_parameters=[torch.tensor([1.0])])
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
    with pytest.raises(RuntimeError, match="injected"):
        with observe_objectives(trainer):
            raise RuntimeError("injected")
    assert dgpo.forward_velocity_state is forward
    assert originals == {name: getattr(trainer, name) for name in names}


@pytest.mark.parametrize("kl_type", ["x-based", "unknown", None])
@pytest.mark.parametrize("kl_beta", [0.0, 0.03])
@pytest.mark.parametrize("parsed", [False, True])
def test_dgpo_rejects_unsupported_kl_before_trainer_materialization(kl_type, kl_beta, parsed):
    from flow_factory.hparams import Arguments, DGPOTrainingArguments

    with pytest.raises(ValueError, match="DGPO requires kl_type='v-based'"):
        if parsed:
            Arguments.from_dict(
                {"train": {"trainer_type": "dgpo", "kl_type": kl_type, "kl_beta": kl_beta}}
            )
        else:
            DGPOTrainingArguments(kl_type=kl_type, kl_beta=kl_beta)


def test_dgpo_v_based_kl_retains_requested_identity():
    from flow_factory.hparams import DGPOTrainingArguments

    args = DGPOTrainingArguments(kl_type="v-based", kl_beta=0.03)
    assert args.to_dict()["kl_type"] == "v-based" and args.kl_beta == 0.03


@pytest.mark.parametrize("scope", ["ema", "ema_ref"])
@pytest.mark.parametrize("corruption", [None, "value", "order", "shape", "dtype"])
def test_dgpo_sampling_observer_tracks_prepared_owner_not_transient_views(scope, corruption):
    from dgpo_observations import observe_objectives

    from flow_factory.ema.ema import EMAModuleWrapper
    from flow_factory.models.abc import BaseAdapter
    from flow_factory.models.variants import ComponentVariantRegistry, ComponentVariantSpec

    adapter = _adapter()
    registry = ComponentVariantRegistry(SimpleNamespace(trainable_component_names=["transformer"]))
    registry.declare(ComponentVariantSpec("policy", "full", {"transformer": "transformer"}))
    canonical = [
        torch.nn.Parameter(torch.tensor([1.125, 2.25])),
        torch.nn.Parameter(torch.tensor([[3.5]])),
    ]
    for name, parameter in zip(("block.weight", "block.bias"), canonical, strict=True):
        registry.register_parameter("policy", "transformer", name, parameter)
    adapter.component_variant_registry = registry
    exposed = list(canonical)
    adapter.get_trainable_parameters = lambda: exposed
    adapter.ema_wrapper = EMAModuleWrapper(canonical, decay=0.9)
    for snapshot in adapter.ema_wrapper.ema_parameters:
        snapshot.add_(0.5)
    adapter._ref_ema = SimpleNamespace(ema_parameters=[item.detach().clone() for item in canonical])
    # Use the real prepared-owner EMA swap; only the backend's transient public
    # component views are simulated below, without changing canonical ownership.
    adapter.use_ema_parameters = BaseAdapter.use_ema_parameters.__get__(adapter)
    adapter.use_named_parameters = BaseAdapter.use_named_parameters.__get__(adapter)
    adapter._named_parameters = {
        "ema_ref": SimpleNamespace(
            target_components=["transformer"], ema_wrapper=adapter.ema_wrapper
        )
    }
    adapter.has_component = lambda name: name == "transformer"
    adapter._require_component = lambda name: SimpleNamespace(parameters=lambda: canonical)
    trainer = _dgpo_trainer(adapter)
    trainer.switch_ema_ref, trainer.off_policy = 1, True
    trainer.step = 2 if scope == "ema_ref" else 0
    initial = [item.detach().clone() for item in canonical]
    records = registry.parameter_records("policy")
    assert [(record.component_name, record.parameter_name) for record in records] == [
        ("transformer", "block.weight"),
        ("transformer", "block.bias"),
    ]
    assert all(
        actual is expected
        for actual, expected in zip(adapter._ema_tracked_parameters(), canonical, strict=True)
    )

    def sample():
        with observe_objectives(trainer), trainer.sampling_context():
            for actual, expected in zip(canonical, adapter.ema_wrapper.ema_parameters, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_dtype=True)
            # A retained full materialization may have a different shape, dtype,
            # object identity and requires_grad membership from the local shard.
            exposed[:] = [torch.zeros(4, dtype=torch.bfloat16)]
            assert exposed[0].shape != canonical[0].shape and not exposed[0].requires_grad
            if corruption == "value":
                canonical[0].data.add_(1)
            elif corruption == "order":
                adapter._ema_tracked_parameters = lambda: list(reversed(canonical))
            elif corruption == "shape":
                adapter._ema_tracked_parameters = lambda: [canonical[0].reshape(1, 2), canonical[1]]
            elif corruption == "dtype":
                adapter._ema_tracked_parameters = lambda: [item.double() for item in canonical]

    if corruption is None:
        sample()
    else:
        with pytest.raises(AssertionError):
            sample()
    for actual, expected in zip(canonical, initial, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_dtype=True)
    assert registry.parameter_records("policy") == records


@pytest.mark.parametrize("source", ["old policy", "policy", "reference"])
@pytest.mark.parametrize("operand", ["state", "sigma", "timestep"])
def test_dgpo_observer_rejects_in_place_forward_input_mutation(monkeypatch, source, operand):
    from dgpo_observations import observe_objectives

    from flow_factory.trainers.rl import dgpo
    from tests.trainers.test_dgpo_crd_production_paths import _adapter, _dgpo_trainer, _samples

    trainer = _dgpo_trainer(_adapter(), use_ema_ref=True, kl_beta=0.03)
    trainer.adapter._ref_ema = SimpleNamespace(ema_parameters=[torch.tensor([1.0])])
    original = dgpo.forward_velocity_state
    observed = []

    def corrupt(owner, batch, state, times, **kwargs):
        result = original(owner, batch, state, times, **kwargs)
        observed.append(kwargs["source"])
        if kwargs["source"] == source:
            values = state.components if operand == "state" else getattr(times, operand)
            value = next(iter(values.values()))
            value[(0,) * value.ndim].add_(0.125)
        return result

    monkeypatch.setattr(dgpo, "forward_velocity_state", corrupt)
    with pytest.raises(AssertionError):
        with observe_objectives(trainer):
            trainer.optimize(_samples([4, 4], [1.0, 2.0]))
    assert observed[-1] == source


@pytest.mark.parametrize("initial_step", [0, 4])
def test_dgpo_observer_rejects_missing_final_ema_ref_update(monkeypatch, initial_step):
    from dgpo_observations import assert_objectives, observe_objectives

    row = _objective_case(monkeypatch, clip_dsm=False, clip_kl=False)
    row.update(same_noised_inputs=True, logical_replicas=2, launched_ranks=8)
    final_step = initial_step + 3
    row["lifecycle"] = {
        "initial_step": initial_step,
        "final_step": final_step,
        "ema_ref_updates": [
            {"step": step, "changed": True} for step in range(initial_step, final_step)
        ],
        "sampling": [
            {"scope": "ema", "snapshot_differs_from_live": False},
            {"scope": "ema_ref", "snapshot_differs_from_live": True},
        ],
    }
    assert_objectives([row], 2)
    broken = copy.deepcopy(row)
    broken["lifecycle"]["ema_ref_updates"].pop()
    with pytest.raises(AssertionError):
        assert_objectives([broken], 2)

    trainer = _dgpo_trainer(_adapter())
    trainer.step = initial_step
    trainer.adapter._ref_ema = SimpleNamespace(ema_parameters=[torch.tensor([1.0])])
    with observe_objectives(trainer) as observed:
        observed.append({})
        trainer.step = final_step
    assert observed[-1]["lifecycle"]["initial_step"] == initial_step
    assert observed[-1]["lifecycle"]["final_step"] == final_step


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("clipped", [False, True])
def test_dgpo_fp64_advantage_preserves_public_operand_promotions(monkeypatch, rank, clipped):
    row = _objective_case(
        monkeypatch,
        dtype=torch.float32,
        velocity_dtype=torch.bfloat16,
        advantage_dtype=torch.float64,
        kl_compute_dtype=torch.float32,
        rank=rank,
        clip_dsm=clipped,
        clip_kl=clipped,
    )
    assert row["advantage"].dtype == torch.float64
    assert row["dsm_endpoints"]["new"].dtype == torch.float32
    assert row["loss"].dtype == torch.float64
    assert_objective(row)


@pytest.mark.parametrize("operand", ["new", "old"])
@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0, -1.0])
def test_dgpo_dsm_operand_proof_rejects_wrong_reduction(monkeypatch, operand, factor):
    row = _objective_case(monkeypatch, dtype=torch.float32, clip_dsm=False, clip_kl=False)
    row["dsm_endpoints"][operand] *= factor
    with pytest.raises(AssertionError):
        assert_objective(row)


def test_dgpo_cancellation_uses_independently_proved_dsm_operands():
    from dgpo_objective_reference import assert_local_preference

    generator = torch.Generator().manual_seed(0)
    old = torch.randn(1, 128, generator=generator).bfloat16()
    current = (old.float() + 0.003 * torch.randn(1, 128, generator=generator)).bfloat16()
    target = torch.randn(1, 128, generator=generator)
    row = dict(
        new={"latent": current},
        old={"latent": old},
        target={"latent": target},
        masks={},
        advantage=torch.tensor([1.0], dtype=torch.float64),
        dpo_beta=4.0,
        group_size=2,
        use_ema_ref=True,
        group_keys=[("source", 9)],
        local_group_indices=torch.tensor([0]),
    )
    trainer = _dgpo_trainer(_adapter(), use_ema_ref=True, dpo_beta=4.0)
    noised = NoisedState(
        state=LatentState({"latent": torch.zeros_like(target)}),
        noise=LatentState(row["target"]),
        target_velocity=LatentState(row["target"]),
    )
    endpoints = {
        name: trainer._compute_dsm_loss(noised.target_velocity, LatentState(row[name]), noised)
        for name in ("new", "old")
    }
    actual = trainer._compute_per_sample_preference(
        endpoints["new"],
        LatentState(row["old"]),
        noised.target_velocity,
        row["advantage"],
        noised,
    )
    # The old delta-relative check fails on a reproducible, ordinary FP32 sum
    # although each public DSM endpoint differs by at most its reduction error.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual.double(), local_preference(row), rtol=2e-5, atol=1e-7)
    row["dsm_endpoints"] = endpoints
    torch.testing.assert_close(actual.double(), local_preference(row), rtol=0, atol=0)
    assert_local_preference(row, actual, row["group_keys"])
    assert actual.abs().min() > 1e-4
    for factor in (0.0, 0.5, 2.0, -1.0):
        with pytest.raises(AssertionError):
            assert_local_preference(row, actual * factor, row["group_keys"])


def test_dgpo_local_sum_proof_preserves_sparse_group_slots(monkeypatch):
    from dgpo_objective_reference import assert_local_preference

    row = _objective_case(monkeypatch, clip_dsm=False, clip_kl=False)
    keys = sorted(set(row["group_keys"]))
    dense = [("absent", -1), *keys]
    row["local_group_indices"] = torch.tensor([dense.index(key) for key in row["group_keys"]])
    actual = torch.zeros(len(dense), dtype=torch.float64)
    for index, value in zip(row["local_group_indices"], local_preference(row), strict=True):
        actual[index] += value
    assert_local_preference(row, actual, dense)
    for mutation in ("zero_slot", "index", "length"):
        broken = copy.deepcopy(row)
        observed = actual.clone()
        if mutation == "zero_slot":
            observed[0] = 1e-10
        elif mutation == "index":
            broken["local_group_indices"][0] = 0
        else:
            observed = observed[1:]
        with pytest.raises(AssertionError):
            assert_local_preference(broken, observed, dense)
