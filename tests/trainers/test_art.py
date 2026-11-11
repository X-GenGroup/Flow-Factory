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

"""Independent ART numerical controls and production argument routing."""

import math
from dataclasses import fields

import pytest
import torch

from flow_factory.hparams import Arguments, ARTTrainingArguments
from flow_factory.hparams.training_args import get_training_args_class
from flow_factory.models.abc import BaseAdapter
from flow_factory.trainers.registry import get_trainer_class
from flow_factory.trainers.rl.art import DiffusionARTTrainer
from flow_factory.trainers.rl.art_objective import art_objective, endpoint_guidance


def config_dict(**train):
    return {
        "model": {"model_type": "sd3_5", "model_name_or_path": "/tmp/art-contract"},
        "train": {
            "trainer_type": "art",
            "unique_sample_num_per_epoch": 2,
            "group_size": 2,
            "num_inference_steps": 3,
            **train,
        },
        "scheduler": {"dynamics_type": "ODE"},
        "log": {"save_dir": ".scratch/art/argument-saves"},
        "rewards": [{"name": "score", "reward_model": "clip"}],
    }


def config(**train):
    return Arguments.from_dict(config_dict(**train))


def objective(v, old, g, s=None, dt=None, mask=None, threshold=1e-5):
    n = v.shape[0]
    return art_objective(
        v,
        old,
        g,
        torch.full((n,), 0.5) if s is None else s,
        torch.full((n,), -0.1) if dt is None else dt,
        guidance_scale=1.0,
        kl_threshold=threshold,
        reference_noise_level=0.8,
        epsilon=1e-5,
        active_mask=mask,
    )


def test_registry_and_two_update_geometry(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    args = config()
    assert get_trainer_class("ART") is DiffusionARTTrainer
    assert get_training_args_class("art") is ARTTrainingArguments
    assert args.data_args.sampler_type == "group_contiguous"
    assert args.training_args.num_batches_per_epoch == 2
    assert args.training_args.gradient_accumulation_steps == 3
    assert args.training_args.ema_decay == 0
    DiffusionARTTrainer.validate_training_arguments_contract(args.training_args)
    values = config_dict()
    values["train"] = {
        f.name: getattr(args.training_args, f.name) for f in fields(args.training_args) if f.init
    }
    roundtrip = Arguments.from_dict(values)
    DiffusionARTTrainer.validate_training_arguments_contract(roundtrip.training_args)


@pytest.mark.parametrize(
    "values",
    [
        {"group_size": 1},
        {"art_guidance_scale": float("nan")},
        {"art_kl_threshold": -1},
        {"art_reference_noise_level": 0},
        {"kl_beta": -1},
        {"kl_beta": float("inf")},
        {"kl_type": "x-based"},
    ],
)
def test_invalid_arguments(values):
    with pytest.raises(ValueError):
        config(**values)


@pytest.mark.parametrize(
    "layout", ["global_random", "global_batch", "global_tile", "subgroup_tile", "contiguous_shard"]
)
def test_cross_rank_groups_rejected(layout):
    values = config_dict()
    values["data"] = {"sampler_type": layout}
    with pytest.raises(ValueError, match="sampler"):
        Arguments.from_dict(values)


def test_alias_and_explicit_accumulation():
    values = config_dict()
    values["data"] = {"sampler_type": "rank_local"}
    parsed = Arguments.from_dict(values)
    DiffusionARTTrainer.validate_training_arguments_contract(parsed.training_args)
    parsed.training_args.gradient_accumulation_steps += 1
    DiffusionARTTrainer.validate_training_arguments_contract(parsed.training_args)


@pytest.mark.parametrize("updates,epochs", [(1, 1), (3, 2), (4, 3)])
def test_configurable_update_geometry(monkeypatch, updates, epochs):
    monkeypatch.setenv("WORLD_SIZE", "2")
    args = config(
        gradient_step_per_epoch=updates,
        num_inner_epochs=epochs,
        global_std=True,
        latent_storage_dtype="bf16",
    )
    train = args.training_args
    DiffusionARTTrainer.validate_training_arguments_contract(train)
    assert train.num_inner_epochs == epochs
    assert (
        train.num_batches_per_epoch
        * train.get_num_train_timesteps(args)
        // train.gradient_accumulation_steps
        == updates
    )


def test_custom_advantage_reducer_is_owned_by_shared_feedback():
    from flow_factory.contracts.feedback import resolve_feedback_reducer_contract

    def reducer(rewards, **kwargs):
        return rewards

    args = ARTTrainingArguments(advantage_aggregation=reducer, global_std=True)
    contract = resolve_feedback_reducer_contract(
        args.advantage_aggregation, global_std=args.global_std
    )
    assert contract.combination_order == "custom"


def test_zero_guidance_and_mask_threshold_are_valid_ablations():
    args = config(art_guidance_scale=0, art_kl_threshold=0)
    assert args.training_args.art_guidance_scale == 0
    assert args.training_args.art_kl_threshold == 0


@pytest.mark.parametrize("accumulation,epochs", [(6, 2), (2, 3), (4, 2)])
def test_multiple_replay_epochs_match_explicit_sgd_reference(accumulation, epochs):
    from collections import defaultdict
    from copy import deepcopy

    from test_full_output_backward import _art_trainer

    trainer, group, (weight,) = _art_trainer()
    samples = []
    for group_id in range(3):
        members = deepcopy(group)
        for sample in members:
            sample.sampling_group_id = group_id
        samples.extend(members)
    trainer.training_args.num_batches_per_epoch = len(samples)
    trainer.training_args.num_inner_epochs = epochs
    trainer.training_args.gradient_accumulation_steps = accumulation
    trainer.accelerator.gradient_accumulation_steps = accumulation
    trainer.training_args.art_kl_threshold = 1e6
    optimizer = torch.optim.SGD([weight], lr=0.05)
    actual = []
    orders = []

    def order(rows, inner_epoch):
        orders.append(inner_epoch)
        return rows

    def step(metrics):
        actual.append(weight.grad.detach().clone())
        optimizer.step()
        optimizer.zero_grad()
        return defaultdict(list)

    trainer._order_samples_for_optimize = order
    trainer._apply_optimizer_step = step
    trainer.optimize(samples)
    assert orders == list(range(epochs))
    assert len(actual) == len(samples) * epochs // accumulation
    reference = torch.tensor(0.1, requires_grad=True)
    reference_optimizer = torch.optim.SGD([reference], lr=0.05)
    # An independently written loop consumes the same immutable rollout target,
    # including a window crossing the inner-epoch boundary for accumulation=4.
    direction = torch.tensor([[1.0, 2.0]]) / torch.tensor(2.5).sqrt()
    expected = []
    for i in range(len(samples) * epochs):
        loss = 0.5 * (reference.expand(1, 2) + direction).square().mean() / accumulation
        loss.backward()
        if (i + 1) % accumulation == 0:
            expected.append(reference.grad.detach().clone())
            reference_optimizer.step()
            reference_optimizer.zero_grad()
    torch.testing.assert_close(weight, reference)
    for observed, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, wanted)


def test_multicomponent_admitted_before_loading():
    DiffusionARTTrainer.validate_adapter_class_execution_contract(BaseAdapter)
    DiffusionARTTrainer.validate_adapter_class_execution_contract(
        type("JointAdapter", (BaseAdapter,), {"trajectory_component_order": ("video", "audio")})
    )


def test_ode_and_rollout_parity_gates():
    args = config()
    args.scheduler_args.dynamics_type = "Flow-SDE"
    with pytest.raises(ValueError, match="ODE"):
        args.training_args.validate_configuration(args)
    args.scheduler_args.dynamics_type = "ODE"
    args.acceleration_args.rollout = [object()]
    with pytest.raises(ValueError, match="rollout"):
        args.training_args.validate_configuration(args)


def test_covariance_matches_independent_scalar_reference():
    x = torch.tensor([[1.0, 8.0, -3.0], [7.0, 2.0, 6.0], [4.0, 1.0, -4.0], [-2.0, 5.0, 9.0]])
    r = torch.tensor([0.5, 2.0, -1.0, 1.0])
    result = endpoint_guidance(x, r, epsilon=1e-5)
    centered = [float(y) - math.fsum(r.tolist()) / 4 for y in r]
    covariance = [math.fsum(centered[k] * float(x[k, j]) for k in range(4)) / 3 for j in range(3)]
    norm = math.sqrt(math.fsum(v * v for v in covariance) / 3)
    torch.testing.assert_close(result, torch.tensor([[c / norm for c in covariance]]))
    # The whole prompt group, not arbitrary optimizer chunks, owns this statistic.
    chunks = [endpoint_guidance(x[i : i + 2], r[i : i + 2], epsilon=1e-5) for i in (0, 2)]
    assert not torch.allclose(result, torch.stack(chunks).mean(0))
    torch.testing.assert_close(
        result, endpoint_guidance(x[[3, 1, 0, 2]], r[[3, 1, 0, 2]], epsilon=1e-5)
    )


@pytest.mark.parametrize("r", [torch.ones(4), torch.full((4,), 0.1), torch.zeros(4)])
def test_constant_reward_is_exactly_zero(r):
    assert (
        torch.count_nonzero(endpoint_guidance(torch.arange(12.0).reshape(4, 3), r, epsilon=1e-5))
        == 0
    )


def test_masks_exclude_fixed_coordinates():
    x = torch.tensor([[0.0, 1000.0], [2.0, -1000.0]])
    mask = torch.tensor([[True, False], [True, False]])
    direction = endpoint_guidance(x, torch.tensor([-1.0, 1.0]), epsilon=1e-5, active_mask=mask)
    torch.testing.assert_close(direction, torch.tensor([[1.0, 0.0]]))
    v = torch.zeros(1, 2, requires_grad=True)
    terms = objective(v, v.detach().clone(), direction, mask=mask[:1])
    terms["per_sample_loss"].sum().backward()
    torch.testing.assert_close(v.grad, torch.tensor([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="share"):
        endpoint_guidance(
            x,
            torch.tensor([-1.0, 1.0]),
            epsilon=1e-5,
            active_mask=torch.tensor([[True, False], [True, True]]),
        )


def test_target_sign_and_frozen_anchor_gradient():
    v = torch.zeros(1, 3, requires_grad=True)
    old = torch.zeros_like(v, requires_grad=True)
    g = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
    terms = objective(v, old, g)
    terms["per_sample_loss"].sum().backward()
    torch.testing.assert_close(v.grad, torch.tensor([[1.0, 2.0, -1.0]]) / 3)
    assert old.grad is None and g.grad is None
    assert torch.all(terms["keep"])
    # Decreasing sigma means a negative velocity update moves x0 toward high reward.
    torch.testing.assert_close(terms["target"], -g.detach())


def test_directional_trust_region_keeps_corrective_moves():
    v = torch.tensor([[-0.5], [-2.0], [0.5]], requires_grad=True)
    old, g = torch.zeros_like(v), torch.ones_like(v)
    terms = objective(v, old, g)
    assert terms["keep"].tolist() == [False, True, True]
    terms["per_sample_loss"].sum().backward()
    torch.testing.assert_close(v.grad, torch.tensor([[0.0], [-1.0], [1.5]]))


def test_virtual_sde_initial_endpoint_and_interior():
    v = torch.tensor([[0.3, 0.1], [0.3, 0.1]])
    terms = objective(
        v,
        torch.zeros_like(v),
        torch.ones_like(v),
        s=torch.tensor([1.0, 0.5]),
        dt=torch.tensor([-0.2, -0.1]),
    )
    energy = (0.3**2 + 0.1**2) / 2
    expected = torch.tensor(
        [
            0.2**2 * energy / (2 * 0.8**2),
            (0.1 * (1 + 0.5 * 0.8**2)) ** 2 * energy / (2 * 0.8**2 * 0.1),
        ]
    )
    torch.testing.assert_close(terms["kl"], expected)


def test_two_updates_have_restoration_and_differ_from_accumulation():
    p = torch.nn.Parameter(torch.tensor([[0.0]]))
    old = p.detach().clone()
    optim = torch.optim.SGD([p], lr=0.2)
    gradients = []
    for _ in range(2):
        terms = objective(p, old, torch.ones_like(p), threshold=1e6)
        terms["per_sample_loss"].sum().backward()
        gradients.append(p.grad.clone())
        optim.step()
        optim.zero_grad()
    torch.testing.assert_close(gradients[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(gradients[1], torch.tensor([[0.8]]))
    torch.testing.assert_close(p, torch.tensor([[-0.36]]))
    assert not torch.allclose(p, torch.tensor([[-0.4]]))


def test_planned_group_identity_and_incomplete_groups():
    from flow_factory.samples import BaseSample
    from flow_factory.trainers.rl.art import _group_key

    a = BaseSample(prompt="same text", source_id=1, sampling_group_id=7, sampling_group_member_id=0)
    b = BaseSample(prompt="same text", source_id=2, sampling_group_id=7, sampling_group_member_id=0)
    assert _group_key(a) != _group_key(b)
    # Single-source planned loaders may retain the canonical legacy source namespace.
    legacy = BaseSample(sampling_group_id=7, sampling_group_member_id=0)
    assert _group_key(legacy).source_id == -1
    with pytest.raises(ValueError, match="planned"):
        _group_key(BaseSample(prompt="same text"))
    trainer = DiffusionARTTrainer.__new__(DiffusionARTTrainer)
    trainer.training_args = config().training_args
    with pytest.raises(ValueError, match="one member"):
        trainer._group_guidance([a])
    with pytest.raises(ValueError, match="one member"):
        trainer._group_guidance([a, a])


@pytest.mark.parametrize("name", ["sd3-5", "flux1", "wan2_t2v", "ltx2_t2av", "minimax-h3-t2va"])
def test_adapter_admission_is_component_based(name):
    from flow_factory.models.registry import get_model_adapter_class

    cls = get_model_adapter_class(name)
    DiffusionARTTrainer.validate_adapter_class_execution_contract(cls)


def test_group_guidance_releases_each_temporary_cpu_trajectory():
    import weakref
    from types import SimpleNamespace

    from flow_factory.samples import BaseSample, LatentState

    trainer = DiffusionARTTrainer.__new__(DiffusionARTTrainer)
    trainer.training_args = config(group_size=4).training_args
    trajectories = []

    def terminal(batch):
        # Deliberately return a view into a larger temporary trajectory. A group
        # aggregation must not keep all K underlying storages alive at once.
        assert not any(reference() is not None for reference in trajectories)
        member = batch.samples[0].sampling_group_member_id
        storage = torch.full((16, 2), float(member))
        trajectories.append(weakref.ref(storage))
        return LatentState({"latent": storage[-1:]})

    trainer.adapter = SimpleNamespace(get_terminal_state=terminal)
    samples = [
        BaseSample(
            sampling_group_id=3,
            sampling_group_member_id=i,
            extra_kwargs={"advantage": torch.tensor(float(i))},
        )
        for i in range(4)
    ]
    guidance = trainer._group_guidance(samples)
    assert len(guidance) == 1
    torch.testing.assert_close(next(iter(guidance.values())), torch.ones(1, 2))
    assert not any(reference() is not None for reference in trajectories)


def test_official_pickscore_geometry(monkeypatch, tmp_path):
    from pathlib import Path

    import yaml

    values = yaml.safe_load(
        (Path(__file__).parents[2] / "examples/art/lora/sd3_5/pickscore_8gpu.yaml").read_text()
    )
    values["log"]["save_dir"] = str(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = Arguments.from_dict(values)
    DiffusionARTTrainer.validate_training_arguments_contract(args.training_args)
    assert args.training_args.num_batches_per_epoch == 16
    assert args.training_args.gradient_accumulation_steps == 80
    assert args.training_args.max_epochs * args.training_args.gradient_step_per_epoch == 1000
    assert args.eval_args.eval_freq * 2 == 50
    assert args.model_args.lora_rank == 32
    assert args.model_args.lora_alpha == 64


def test_fully_frozen_component_has_zero_loss_and_gradient():
    v = torch.ones(2, 3, requires_grad=True)
    mask = torch.zeros_like(v, dtype=torch.bool)
    direction = endpoint_guidance(v, torch.tensor([-1.0, 1.0]), epsilon=1e-5, active_mask=mask)
    terms = objective(v, torch.zeros_like(v), direction.expand_as(v), mask=mask)
    terms["per_sample_loss"].sum().backward()
    assert torch.equal(v.grad, torch.zeros_like(v))
    assert not terms["keep"].any()


@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_art_stream_accumulates_across_group_tiles(accumulation):
    from collections import defaultdict
    from copy import deepcopy

    from test_full_output_backward import _art_trainer

    def run(streaming):
        trainer, group, (weight,) = _art_trainer()
        samples = group + deepcopy(group)
        for sample in samples[2:]:
            sample.sampling_group_id = 1
        trainer.training_args.num_batches_per_epoch = 4
        trainer.accelerator.gradient_accumulation_steps = accumulation
        trainer.training_args.gradient_accumulation_steps = accumulation
        optimizer = torch.optim.SGD([weight], lr=0.05)
        updates = []

        def step(metrics):
            updates.append(weight.grad.detach().clone())
            optimizer.step()
            optimizer.zero_grad()
            return defaultdict(list)

        trainer._apply_optimizer_step = step
        if streaming:
            context = trainer._prepare_reward_optimization_overlap(samples, None)
            trainer._optimize_reward_overlap_tile(None, samples[:2], context)
            assert len(updates) == 2 // accumulation
            trainer._optimize_reward_overlap_tile(None, samples[2:], context)
            trainer._finalize_reward_optimization_overlap(context, samples)
        else:
            trainer.optimize(samples)
        return weight.detach(), updates

    ordinary, ordinary_grads = run(False)
    streamed, streamed_grads = run(True)
    torch.testing.assert_close(ordinary, streamed, rtol=0, atol=0)
    assert len(streamed_grads) == 4 // accumulation
    for a, b in zip(ordinary_grads, streamed_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_joint_components_use_own_direction_time_and_equal_weight():
    from collections import defaultdict
    from types import SimpleNamespace

    from test_full_output_backward import _art_trainer

    from flow_factory.samples import LatentState

    trainer, samples, (weight,) = _art_trainer()
    adapter = trainer.adapter
    adapter.trajectory_component_order = ("video", "audio")
    adapter.get_terminal_state = lambda batch: LatentState(
        {
            "video": batch["all_latents"][:, -1],
            "audio": -batch["all_latents"][:, -1, :1],
        }
    )
    state = LatentState({"video": torch.ones(1, 2), "audio": torch.ones(1, 1)})
    times = SimpleNamespace(
        timestep={"video": torch.tensor([500.0]), "audio": torch.tensor([250.0])}
    )
    adapter.get_replay_step = lambda batch, i: SimpleNamespace(
        state=state, next_state=state, times=times
    )
    adapter.get_replay_callback = lambda batch, i, field: LatentState(
        {k: torch.zeros_like(v) for k, v in state.components.items()}
    )
    adapter.forward_state = lambda **kwargs: SimpleNamespace(
        velocity=LatentState({k: weight.expand_as(v) for k, v in state.components.items()}),
        dt={"video": torch.tensor([-0.5]), "audio": torch.tensor([-0.25])},
    )
    actual = []
    trainer._apply_optimizer_step = lambda metrics: actual.extend(metrics["loss"]) or defaultdict(
        list
    )
    trainer.optimize(samples)
    video_g = endpoint_guidance(
        torch.tensor([[-1.0, -2.0], [1.0, 2.0]]), torch.tensor([-1.0, 1.0]), epsilon=1e-5
    )
    audio_g = torch.tensor([[-1.0]])
    v = objective(
        weight.expand(1, 2),
        torch.zeros(1, 2),
        video_g,
        s=torch.tensor([0.5]),
        dt=torch.tensor([-0.5]),
    )
    a = objective(
        weight.expand(1, 1),
        torch.zeros(1, 1),
        audio_g,
        s=torch.tensor([0.25]),
        dt=torch.tensor([-0.25]),
    )
    expected = (v["per_sample_loss"] + a["per_sample_loss"]) / 2
    assert len(actual) == 2
    for loss in actual:
        torch.testing.assert_close(loss, expected.mean())


def test_reference_velocity_mse_is_opt_in():
    assert config().training_args.kl_beta == 0
    assert not config().training_args.requires_ref_model
    assert config(kl_beta=1e-3).training_args.requires_ref_model


def test_reference_velocity_mse_is_unweighted_unmasked_and_equal_per_component():
    from collections import defaultdict
    from contextlib import contextmanager
    from types import SimpleNamespace

    from test_full_output_backward import _art_trainer

    from flow_factory.samples import LatentState

    trainer, samples, (weight,) = _art_trainer()
    trainer.training_args.kl_beta = 0.5
    adapter = trainer.adapter
    adapter.trajectory_component_order = ("video", "audio")
    # Equal rewards give zero guidance: ART keeps no sample, so only the reference term trains.
    for sample in samples:
        sample.extra_kwargs["advantage"] = torch.tensor(1.0)
    adapter.get_terminal_state = lambda batch: LatentState(
        {"video": batch["all_latents"][:, -1], "audio": batch["all_latents"][:, -1, :1]}
    )
    state = LatentState({"video": torch.ones(1, 2), "audio": torch.ones(1, 1)})
    times = SimpleNamespace(
        timestep={"video": torch.tensor([500.0]), "audio": torch.tensor([250.0])}
    )
    adapter.get_replay_step = lambda batch, i: SimpleNamespace(
        state=state, next_state=state, times=times
    )
    adapter.get_replay_callback = lambda batch, i, field: LatentState(
        {k: torch.zeros_like(v) for k, v in state.components.items()}
    )
    reference = {"video": torch.tensor([[0.3, -0.1]]), "audio": torch.tensor([[0.7]])}
    in_reference, reference_grad_enabled = [False], []

    @contextmanager
    def use_ref_parameters():
        in_reference[0] = True
        try:
            yield
        finally:
            in_reference[0] = False

    def forward_state(**kwargs):
        if in_reference[0]:
            reference_grad_enabled.append(torch.is_grad_enabled())
            assert kwargs["return_fields"] == ("velocity",)
            return SimpleNamespace(velocity=LatentState(dict(reference)))
        return SimpleNamespace(
            velocity=LatentState({k: weight.expand_as(v) for k, v in state.components.items()}),
            dt={"video": torch.tensor([-0.5]), "audio": torch.tensor([-0.25])},
        )

    adapter.use_ref_parameters = use_ref_parameters
    adapter.forward_state = forward_state
    updates = []
    trainer._apply_optimizer_step = lambda metrics: updates.append(
        {k: list(v) for k, v in metrics.items()}
    ) or defaultdict(list)
    trainer.optimize(samples)

    def reference_term(w):
        video = (w - reference["video"]).square().mean()
        audio = (w - reference["audio"]).square().mean()
        return (video + audio) / 2

    expected_kl = reference_term(weight.detach())
    assert reference_grad_enabled == [False, False]
    assert len(updates) == 2
    for metrics in updates:
        torch.testing.assert_close(metrics["art_keep_fraction/video"][0], torch.tensor(0.0))
        torch.testing.assert_close(metrics["kl_div"][0], expected_kl)
        torch.testing.assert_close(metrics["loss"][0], 0.5 * expected_kl)
    # The optimizer step is stubbed, so both microbatch gradients accumulate.
    expected_grad = 2 * torch.autograd.grad(0.5 * reference_term(weight), weight)[0]
    torch.testing.assert_close(weight.grad, expected_grad)


def test_stream_geometry_accumulates_across_complete_group_tiles(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = config(
        unique_sample_num_per_epoch=64,
        group_size=18,
        per_device_batch_size=9,
        num_inference_steps=10,
    )
    geometry = DiffusionARTTrainer.reward_optimization_overlap_geometry(args)
    assert geometry.accumulation_scope == "acquisition"
    assert geometry.optimizer_terms_per_batch == 10
    assert args.training_args.gradient_accumulation_steps == 80


def test_h3_dataward_velocity_is_normalized_before_art_loss():
    from types import SimpleNamespace

    from flow_factory.scheduler.minimax_h3 import MiniMaxH3SDEScheduler

    trainer = object.__new__(DiffusionARTTrainer)
    trainer.adapter = SimpleNamespace(scheduler=MiniMaxH3SDEScheduler(dynamics_type="ODE"))
    trainer._validate_paradigm_dynamics()
    assert trainer._art_velocity_sign == -1
    dataward = torch.tensor([[0.2, -0.3]], requires_grad=True)
    old = torch.tensor([[0.1, -0.2]])
    direction = torch.tensor([[1.0, 1.0]])
    actual = objective(-dataward, -old, direction, threshold=1e6)["per_sample_loss"]
    # In H3's native convention the target is old + guidance.
    expected = 0.5 * (old + direction - dataward).square().mean(1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), dataward, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), dataward)[0],
    )
