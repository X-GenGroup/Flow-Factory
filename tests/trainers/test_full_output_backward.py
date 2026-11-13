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

"""CPU algebra and production-path evidence, not qualification of new CP trainers."""

import copy
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator

from flow_factory.hparams import TDMR1TrainingArguments
from flow_factory.hparams.training_args._registry import list_registered_training_args
from flow_factory.samples import LatentState, MultiModalStepOutput
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.common import context_parallel
from flow_factory.trainers.distillation.distillation_runtime import pop_distillation_metrics
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer
from flow_factory.trainers.registry import get_trainer_class, list_registered_trainers
from flow_factory.trainers.role_optimization import OptimizationRole, RoleOptimizationCoordinator
from flow_factory.utils.group_coordinator import GroupCoordinator
from tests.models.minimax_h3 import test_no_weight_production_acceptance as h3
from tests.trainers import test_dgpo_crd_production_paths as grouped
from tests.trainers import test_multirole_production_paths as dmd
from tests.trainers import test_offline_trainers as offline
from tests.trainers import test_opd_production_paths as opd
from tests.trainers import test_tdm as tdm


def _online_trainer(name, monkeypatch):
    """Use real coupled/matching/pair objectives and the tiny native H3 adapter."""
    cls = get_trainer_class(name)
    adapter = h3._production_adapter()
    coupled = name in {"grpo", "grpo-guard", "dppo"}
    fields = {"grpo-guard": ("next_latents_mean",), "dppo": ("velocity",)}.get(name, ())
    sample = h3._run_production_inference(
        monkeypatch,
        trajectory_indices="all" if coupled else [-1],
        adapter=adapter,
        callback_fields=fields,
    )
    sample.extra_kwargs["advantage"] = torch.tensor(1.0)
    kwargs = {}
    if name == "dppo":
        kwargs.update(kl_mask_type="v-based", kl_guidance_scale=1.0, kl_mask_threshold=1e6)
    if name == "dpo":
        kwargs.update(beta=1.0, weighting_scheme="uniform")
    trainer, _ = h3._production_trainer(cls, adapter, **kwargs)
    samples = [sample]
    if not coupled:
        trainer.time_sampling_strategy = "discrete"
        trainer.time_shift = 1.0
        trainer.num_train_timesteps = 2
        trainer.timestep_range = (0.0, 1.0)
        trainer.off_policy = False
        trainer.kl_type = "v-based"
    if name == "nft":
        trainer.nft_beta = 0.5
    if name == "awm":
        trainer.weighting = "Uniform"
        trainer.ghuber_power = 0.25
        trainer.kl_beta = trainer.ema_kl_beta = 0.0
    if name == "dpo":
        rejected = copy.deepcopy(sample)
        rejected.prompt_embeds = rejected.prompt_embeds + 10.0
        rejected.extra_kwargs["advantage"] = torch.tensor(-1.0)
        samples.append(rejected)
        trainer.training_args.update(timestep_range=(0.0, 1.0), time_shift=1.0)
        trainer.advantage_processor = SimpleNamespace(group_on_same_rank=True)
        adapter.training_args = SimpleNamespace(requires_ref_model=True, ref_param_device="cpu")
        adapter._init_ref_parameters()
    return trainer, samples, tuple(trainer.model_bundle.parameters())


class _R1Adapter(tdm.ObjectiveTDMAdapter):
    """Add a live surrogate and its slow snapshot to the existing two-component fixture."""

    @contextmanager
    def use_component_variant(self, name):
        assert name in {"generator", "fake", "surrogate"}
        previous = self.active_role
        self.active_role = name
        try:
            yield
        finally:
            self.active_role = previous

    def has_variant_snapshot(self, name):
        return hasattr(self, "slow_surrogate")

    def declare_variant_snapshot(self, role, name):
        assert role == "surrogate"
        self.slow_surrogate = self.bundle.surrogate.detach().clone()

    def update_variant_snapshot(self, name, decay):
        self.slow_surrogate.lerp_(self.bundle.surrogate.detach(), 1 - decay)

    @contextmanager
    def use_variant_snapshot(self, name):
        previous = self.active_role
        self.active_role = "slow_surrogate"
        try:
            yield
        finally:
            self.active_role = previous

    def _forward_state(self, **kwargs):
        if self.active_role not in {"surrogate", "slow_surrogate"}:
            return super()._forward_state(**kwargs)
        value = self.bundle.surrogate if self.active_role == "surrogate" else self.slow_surrogate
        state = kwargs["state"]
        return MultiModalStepOutput(
            velocity=LatentState(
                {name: value.expand_as(tensor) for name, tensor in state.components.items()},
                active_masks=state.active_masks,
            )
        )


def _r1_trainer():
    """Exercise actual fake, surrogate and generator objective implementations."""
    template, _, bundle = tdm._objective_trainer()
    trainer = object.__new__(TDMR1Trainer)
    trainer.__dict__.update(template.__dict__)
    bundle.surrogate = torch.nn.Parameter(torch.tensor(0.1))
    trainer.adapter = _R1Adapter(bundle)
    trainer.training_args = TDMR1TrainingArguments(
        num_inference_steps=2,
        per_device_batch_size=2,
        group_size=2,
        gradient_accumulation_steps=2,
        use_huber=False,
        cfg_reward_scale=0.0,
        surrogate_clip_range=0.2,
        replay_rtol=0,
        replay_atol=0,
    )
    config = tdm._role_config("surrogate", 0.03)
    trainer.optimizer.add_param_group({"params": [bundle.surrogate], "role_name": "surrogate"})
    trainer.optimization_roles["surrogate"] = OptimizationRole(config, (bundle.surrogate,), (2,))
    trainer.role_optimization = RoleOptimizationCoordinator(
        trainer.accelerator, bundle, trainer.optimizer, trainer.optimization_roles
    )
    trainer.config = SimpleNamespace(
        optimizer_args=SimpleNamespace(get_by_name=lambda name: None),
        data_args=SimpleNamespace(sampler_type="group_contiguous"),
    )
    trainer.group_coordinator = GroupCoordinator(
        trainer.accelerator, sampler_type="group_contiguous"
    )
    samples = [tdm._sample(sampling_sample_id=i) for i in range(2)]
    for i, sample in enumerate(samples):
        sample.source_id = 0
        sample.sampling_group_id = 7
        sample.sampling_group_member_id = i
        sample.extra_kwargs["advantage"] = torch.tensor(1.0 if i == 0 else -1.0)
    return trainer, samples, tuple(bundle.parameters())


def _art_trainer():
    """Exercise the production ART group/replay/optimizer path with a scalar policy."""
    from flow_factory.hparams import ARTTrainingArguments
    from flow_factory.samples import BaseSample, ComponentTimes, LatentState

    cls = get_trainer_class("art")
    trainer = object.__new__(cls)
    weight = torch.nn.Parameter(torch.tensor(0.1))
    trainer.model_bundle = torch.nn.ParameterList([weight])
    trainer.accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    trainer.training_args = ARTTrainingArguments(
        unique_sample_num_per_epoch=1,
        group_size=2,
        num_inference_steps=1,
        gradient_accumulation_steps=1,
        shuffle_samples=False,
    )
    trainer.training_args.num_batches_per_epoch = 2
    trainer.autocast = nullcontext
    trainer.accumulate_gradients = lambda: trainer.accelerator.accumulate(trainer.model_bundle)
    trainer._iter_prefetched_batches = lambda samples, size: (
        BaseSample.stack([s]) for s in samples
    )
    state = LatentState({"latent": torch.ones(1, 2)})
    replay = SimpleNamespace(
        state=state,
        next_state=state,
        times=ComponentTimes(
            timestep={"latent": torch.tensor([500.0])},
            next_timestep={"latent": torch.tensor([0.0])},
        ),
    )
    trainer.adapter = SimpleNamespace(
        trajectory_component_order=("latent",),
        get_train_step_indices=lambda: torch.tensor([0]),
        train=lambda: None,
        get_terminal_state=lambda batch: LatentState({"latent": batch["all_latents"][:, -1]}),
        get_replay_step=lambda batch, i: replay,
        get_replay_callback=lambda batch, i, field: LatentState({"latent": torch.zeros(1, 2)}),
        forward_state=lambda **kwargs: SimpleNamespace(
            velocity=LatentState({"latent": weight.expand(1, 2)}),
            dt={"latent": torch.tensor([-0.5])},
        ),
    )
    samples = [
        BaseSample(
            all_latents=torch.tensor([[0.0, 0.0], [float(sign), float(sign * 2)]]),
            sampling_group_id=0,
            sampling_group_member_id=i,
            extra_kwargs={"advantage": torch.tensor(float(sign))},
        )
        for i, sign in enumerate((-1, 1))
    ]
    return trainer, samples, (weight,)


def _refl_trainer():
    """Exercise ReFL's real objective through the common full-output backward owner."""
    from flow_factory.hparams import ReFLTrainingArguments
    from flow_factory.rewards.differentiable import DifferentiableRewardOutput

    template, samples, parameters = _art_trainer()
    trainer = object.__new__(get_trainer_class("refl"))
    trainer.__dict__.update(template.__dict__)
    trainer.training_args = ReFLTrainingArguments(
        unique_sample_num_per_epoch=1,
        group_size=2,
        num_inference_steps=1,
        gradient_accumulation_steps=1,
        shuffle_samples=False,
    )
    trainer.config = SimpleNamespace(data_args=SimpleNamespace(source_id_to_name=["tiny"]))
    for index, sample in enumerate(samples):
        sample.source_id = 0
        sample.sampling_sample_id = index
        sample.prompt = "a bird"
    trainer.adapter.build_training_component_times = lambda *a, **kw: SimpleNamespace(
        sigma={"latent": torch.tensor([0.5])}
    )
    trainer.adapter.project_velocity_to_clean_state = lambda state, times, velocity: LatentState(
        {"latent": state.components["latent"] - 0.5 * velocity.components["latent"]}
    )
    trainer.adapter.decode_differentiable_state = lambda state, **kw: {
        "image": state.components["latent"].sigmoid()
    }
    trainer.differentiable_feedback = lambda **kw: DifferentiableRewardOutput(
        scores={"pixel": kw["image"].square().mean(1)}, weights={"pixel": torch.ones(1)}
    )
    return trainer, samples, parameters


def _case(name, monkeypatch):
    """Build real objective entry points; new registry entries must supply a case."""
    metrics = defaultdict(list)
    if name in {"sft", "offline-dpo"}:
        trainer, adapter, windows = offline._trainer(get_trainer_class(name), [False, True])
        batch = offline._batch("demonstration" if name == "sft" else "preference")

        def run():
            trainer.optimize_batch(batch)
            trainer.optimize_batch(batch)
            metrics["policy"].extend(windows[0]["loss"])

        return trainer, (adapter.policy_weight,), run, metrics, 1
    if name in {"grpo", "grpo-guard", "dppo", "nft", "awm", "dpo"}:
        trainer, samples, parameters = _online_trainer(name, monkeypatch)
    elif name == "refl":
        trainer, samples, parameters = _refl_trainer()
    elif name == "art":
        trainer, samples, parameters = _art_trainer()
    elif name in {"dgpo", "crd"}:
        adapter = grouped._adapter()
        factory = grouped._dgpo_trainer if name == "dgpo" else grouped._crd_trainer
        trainer = factory(adapter, num_train_timesteps=2, kl_beta=0.01)
        samples = grouped._samples([4, 4], [1.0, 2.0])
        parameters = (adapter.weight,)
    elif name == "diffusion-opd":
        adapter = opd._adapter()
        trainer = opd._two_teacher_trainer(adapter)
        samples = opd._mixed_source_samples()
        parameters = (adapter.weight,)
        trainer.log_data = lambda data, **kwargs: metrics["policy"].append(
            data["train/distill_loss"]
        )
        return trainer, parameters, lambda: trainer.optimize(samples), metrics, 1
    elif name in {"dmd2", "tdm", "tdm-r1"}:
        if name == "dmd2":
            trainer, _, bundle = dmd._real_objective_trainer()
            trainer.accelerator.gradient_accumulation_steps = 2
            samples = [dmd._trajectory_unit(sampling_sample_id=i) for i in range(2)]
            run_phases = lambda: (trainer._fake_phase(samples), trainer._generator_phase(samples))
            parameters = tuple(bundle.parameters())
        elif name == "tdm":
            trainer, _, bundle = tdm._objective_trainer()
            units = trainer._build_boundary_units([tdm._sample(sampling_sample_id=0)])
            run_phases = lambda: (trainer._fake_phase(units), trainer._generator_phase(units))
            parameters = tuple(bundle.parameters())
        else:
            trainer, samples, parameters = _r1_trainer()
            run_phases = lambda: trainer.optimize(samples)
        # Keep replay fixed between phases while still exercising real step/zero/counters.
        for group in trainer.optimizer.param_groups:
            group["lr"] = 0.0

        def run():
            run_phases()
            values = pop_distillation_metrics(trainer)
            for role in trainer.optimization_roles:
                metrics[role].append(values[f"train/{role}_loss"])

        return trainer, parameters, run, metrics, 2
    else:
        raise AssertionError(f"registered trainer {name!r} has no full-output production case")

    def finish(loss_info):
        metrics["policy"].extend(loss_info["loss"])
        return defaultdict(list)

    trainer._apply_optimizer_step = finish
    return trainer, parameters, lambda: trainer.optimize(samples), metrics, 1


def _observe_backward(monkeypatch, trainer, parameters, degree, gas):
    """Compare each production backward with a separately differentiated raw objective."""
    losses = defaultdict(list)
    expected = None
    backend_calls = 0
    owner_calls = 0
    original_owner = trainer._backward_full_output_loss
    original_backward = trainer.accelerator.backward

    def owner(loss, *, role=False):
        nonlocal expected, owner_calls
        assert expected is None, "nested full-output normalization"
        owner_calls += 1
        name = trainer.role_optimization.active_role_name if role else "policy"
        losses[name].append(loss.detach().clone())
        active = tuple(parameter for parameter in parameters if parameter.requires_grad)
        gradients = torch.autograd.grad(loss, active, allow_unused=True, retain_graph=True)
        expected = (loss.detach().clone(), active, gradients)
        original_owner(loss, role=role)
        assert expected is None, "full-output boundary did not dispatch backward"

    def backward(loss):
        nonlocal expected, backend_calls
        assert expected is not None, "backward bypassed the full-output owner"
        backend_calls += 1
        raw_loss, active, gradients = expected
        torch.testing.assert_close(loss.detach(), raw_loss * degree)
        before = [None if p.grad is None else p.grad.detach().clone() for p in active]
        original_backward(loss)
        for parameter, old_gradient, gradient in zip(active, before, gradients):
            if gradient is None:
                if old_gradient is None:
                    assert parameter.grad is None
                else:
                    torch.testing.assert_close(parameter.grad, old_gradient)
            else:
                target = gradient * (degree / gas)
                if old_gradient is not None:
                    target = target + old_gradient
                torch.testing.assert_close(parameter.grad, target, rtol=2e-5, atol=2e-6)
        expected = None

    monkeypatch.setattr(trainer, "_backward_full_output_loss", owner)
    monkeypatch.setattr(trainer.accelerator, "backward", backward)

    def check():
        assert owner_calls == backend_calls >= 2
        return losses

    return check


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("name", sorted(list_registered_trainers()))
def test_registered_objective_paths_use_one_full_output_backward(monkeypatch, name, degree):
    """Every registry implementation reaches the owner with unscaled metrics and gradients."""
    assert set(list_registered_trainers()) == set(list_registered_training_args())
    torch.manual_seed(117)
    trainer, parameters, run, metrics, gas = _case(name, monkeypatch)
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (degree, 0))
    check = _observe_backward(monkeypatch, trainer, parameters, degree, gas)
    run()
    losses = check()
    assert losses.keys() == metrics.keys()
    for role, raw_losses in losses.items():
        assert sum(float(value) for value in metrics[role]) / len(metrics[role]) == pytest.approx(
            torch.stack(raw_losses).mean().item(), rel=2e-5, abs=2e-6
        )


@pytest.mark.parametrize("degree", [1, 2])
def test_tdm_r1_streamed_surrogate_uses_the_same_boundary(monkeypatch, degree):
    trainer, samples, parameters = _r1_trainer()
    trainer._ensure_slow_surrogate()
    units = trainer._build_boundary_units(samples)
    context = {"units_by_tile": {0: units[:1], 1: units[1:]}}
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (degree, 0))
    check = _observe_backward(monkeypatch, trainer, parameters, degree, gas=2)
    with trainer.role_optimization.phase("surrogate"):
        for tile_id in (0, 1):
            trainer._optimize_reward_overlap_tile(SimpleNamespace(tile_id=tile_id), [], context)
    losses = check()
    assert set(losses) == {"surrogate"}
    assert trainer.optimization_roles["surrogate"].step == 1
    assert trainer.optimization_roles["fake"].step == 0
    assert trainer.optimization_roles["generator"].step == 0
    metrics = pop_distillation_metrics(trainer)
    assert metrics["train/surrogate_loss"] == pytest.approx(
        torch.stack(losses["surrogate"]).mean().item()
    )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("role", [False, True])
def test_full_output_boundary_matches_independent_partitioned_gradient_and_update(
    monkeypatch, degree, role
):
    """Sum CP token-slice gradients, emulate FSDP averaging, and compare a dense oracle."""
    initial = torch.tensor([0.4, -0.2], dtype=torch.float64)
    inputs = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [0.5, 3.0], [2.0, -1.0]])
    targets = torch.tensor([0.3, -0.7, 1.0, 0.2])
    dense = torch.nn.Parameter(initial.clone())
    # Analytic derivative of the mean squared output, independent of the owner/autograd.
    residuals = inputs.double() @ initial - targets
    expected_gradient = 2 * (inputs.double() * residuals[:, None]).mean(dim=0)
    dense.grad = expected_gradient
    dense_optimizer = torch.optim.SGD([dense], lr=0.1)
    dense_optimizer.step()
    peer_gradients = []
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (degree, 0))
    for rank in range(degree):
        parameter = torch.nn.Parameter(initial.clone())
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        trainer = object.__new__(BaseTrainer)
        trainer.accelerator = accelerator
        bundle = torch.nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(
            [{"params": bundle.parameters(), "role_name": "policy"}], lr=0.1
        )
        config = tdm._role_config("policy", 0.1)
        coordinator = RoleOptimizationCoordinator(
            accelerator, bundle, optimizer, {"policy": OptimizationRole(config, (parameter,), (0,))}
        )
        trainer.role_optimization = coordinator
        # Keep complete output values, but only this CP rank's token slice has a gradient.
        with coordinator.phase("policy") if role else nullcontext():
            for start in (0, 2):
                x = inputs[start : start + 2].double()
                output = x @ parameter
                mask = torch.arange(2) % degree == rank
                gathered = torch.where(mask, output, output.detach())
                loss = (gathered - targets[start : start + 2]).square().mean()
                with coordinator.microbatch() if role else accelerator.accumulate(bundle):
                    trainer._backward_full_output_loss(loss, role=role)
                    if role:
                        # Capture before the existing role finish clears gradients.
                        if accelerator.sync_gradients:
                            peer_gradients.append(parameter.grad.detach().clone())
                        coordinator.finish_microbatch()
            if not role:
                peer_gradients.append(parameter.grad.detach().clone())
    actual_gradient = torch.stack(peer_gradients).mean(dim=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=1e-12)
    actual = torch.nn.Parameter(initial.clone())
    actual.grad = actual_gradient
    torch.optim.SGD([actual], lr=0.1).step()
    torch.testing.assert_close(actual, dense, rtol=0, atol=1e-12)
