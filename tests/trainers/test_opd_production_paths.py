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

"""Production-path coverage for the migrated DiffusionOPD trainer.

``test_opd_parity.py`` pins each helper in isolation. This module drives the
real ``optimize()`` entry point with a tiny trainable adapter, several teachers
and several micro-batches, so the global two-pass order, teacher swap scoping,
gradient ownership, cache round-trip, optimizer cadence, logging collectives and
the pre-migration numerical oracle all run through the production code path.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    ReplayStep,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerMixin, SDESchedulerOutput
from flow_factory.trainers.distillation.opd import DiffusionOPDTrainer
from flow_factory.trainers.distillation.opd.common import (
    compute_structured_distillation_loss,
    project_distillation_target_state,
)

# Per-scope additive bias, so a teacher swap is observable in the output and a
# forgotten swap cannot be mistaken for a correct one.
_SCOPE_BIAS: Dict[str, float] = {
    "student": 0.0,
    "teacher_a": 0.25,
    "teacher_b": -0.5,
}


class TrainingArgsFake(dict):
    """Mapping/attribute hybrid mirroring ``ArgABC`` unpacking behaviour."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class SchedulerFake:
    """Scheduler stub recording the seeds the trainer dispatches."""

    def __init__(self, dynamics_type: str = "ODE", noise_level: float = 0.7) -> None:
        self.dynamics_type = dynamics_type
        self.noise_level = noise_level
        self.train_timesteps = torch.tensor([0, 1])
        self.seeds: List[int] = []

    def step(self) -> None:
        """Provide scheduler compatibility."""

    def set_seed(self, seed: int) -> None:
        """Record the dispatched seed."""
        self.seeds.append(seed)

    get_kl_divergence_denominator = SDESchedulerMixin.get_kl_divergence_denominator


class TrainableAdapterFake(BaseAdapter):
    """One trainable scalar plus a per-scope bias wired into every output.

    ``velocity = latents * weight + bias(scope)`` and
    ``next_latents_mean = latents + velocity`` keep both target spaces
    differentiable w.r.t. a single parameter, while the scope bias makes every
    teacher swap visible in the returned tensors.
    """

    def load_pipeline(self) -> Any:
        """Return an unused pipeline fake."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Record the call and return the requested scope-dependent outputs.

        Only the fields named in ``return_kwargs`` are populated, matching the
        production adapters: an unrequested field must stay ``None`` so the
        trainer cannot silently depend on it.
        """
        latents = kwargs["latents"]
        velocity = self.velocity(latents, self.active_scope)
        broadcast_shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
        requested = tuple(kwargs.get("return_kwargs", ()))
        self.forward_calls.append(
            {
                "scope": self.active_scope,
                "latents": latents.detach().clone(),
                "t": kwargs["t"].detach().clone(),
                "t_next": kwargs["t_next"].detach().clone(),
                "guidance_scale": kwargs.get("guidance_scale"),
                "noise_level": kwargs.get("noise_level"),
                "return_kwargs": requested,
                "requires_grad": velocity.requires_grad,
            }
        )
        self.events.append(f"forward:{self.active_scope}")
        return SDESchedulerOutput(
            next_latents_mean=(latents + velocity) if "next_latents_mean" in requested else None,
            velocity=velocity if "velocity" in requested else None,
            std_dev_t=torch.full(broadcast_shape, 0.5) if "std_dev_t" in requested else None,
            dt=torch.full(broadcast_shape, -0.5) if "dt" in requested else None,
        )

    def velocity(self, latents: torch.Tensor, scope: str) -> torch.Tensor:
        """Model output for a scope; reused verbatim by the legacy oracle."""
        return latents * self.weight + _SCOPE_BIAS[scope]

    def get_trainable_parameters(self) -> List[torch.Tensor]:
        """Expose the single trainable scalar."""
        return [self.weight]

    def train(self, mode: bool = True) -> None:
        """Record the train-mode switch."""
        self.events.append("mode:train" if mode else "mode:eval")

    def ema_step(self, step: int) -> None:
        """Record the sampling-EMA advance."""
        self.events.append(f"ema_step:{step}")

    @contextmanager
    def use_named_parameters(self, name: str) -> Iterator[None]:
        """Record a named teacher-snapshot scope."""
        previous = self.active_scope
        self.active_scope = name
        self.events.append(f"enter:{name}")
        try:
            yield
        finally:
            self.active_scope = previous
            self.events.append(f"exit:{name}")


def _adapter(weight: float = 0.7, dynamics_type: str = "ODE") -> TrainableAdapterFake:
    adapter = object.__new__(TrainableAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake(dynamics_type))
    adapter.scheduler_group = adapter.build_scheduler_group()
    adapter.weight = torch.nn.Parameter(torch.tensor(weight))
    adapter.active_scope = "student"
    adapter.forward_calls = []
    adapter.events = []
    return adapter


class SnapshotModuleFake(torch.nn.Module):
    """One-parameter component the real named-parameter snapshots operate on."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


class SnapshotAdapterFake(TrainableAdapterFake):
    """Teacher swaps go through the real ``BaseAdapter`` named-parameter machinery.

    ``velocity`` reads the live parameter instead of a scope label, so a missing,
    leaked or unrestored swap changes the produced tensors rather than only a
    bookkeeping string.
    """

    @property
    def weight(self) -> torch.nn.Parameter:
        """Expose the live component parameter."""
        return self.snapshot_module.weight

    def has_component(self, name: str) -> bool:
        """Declare the single snapshot-backed component without a real runtime."""
        return name == "snapshot_module"

    def get_component(self, name: str) -> torch.nn.Module:
        """Resolve the single snapshot-backed component."""
        if name != "snapshot_module":
            raise KeyError(f"expected component 'snapshot_module', received {name!r}")
        return self.snapshot_module

    def get_trainable_parameters(self) -> List[torch.Tensor]:
        """Expose the live component parameter."""
        return [self.snapshot_module.weight]

    def velocity(self, latents: torch.Tensor, scope: str) -> torch.Tensor:
        """Model output driven purely by the currently installed weight."""
        return latents * self.weight

    @contextmanager
    def use_named_parameters(self, name: str) -> Iterator[None]:
        """Record the scope around the real snapshot swap."""
        previous = self.active_scope
        self.active_scope = name
        self.events.append(f"enter:{name}")
        try:
            with BaseAdapter.use_named_parameters(self, name):
                self.observed_weights.append((name, float(self.weight)))
                yield
        finally:
            self.active_scope = previous
            self.events.append(f"exit:{name}")


def _snapshot_adapter(
    student_weight: float, teacher_weights: Dict[str, float]
) -> SnapshotAdapterFake:
    """Adapter with one real named-parameter snapshot per teacher."""
    adapter = object.__new__(SnapshotAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    adapter.snapshot_module = SnapshotModuleFake(student_weight)
    adapter.target_module_map = {"snapshot_module": ["weight"]}
    adapter._named_parameters = {}
    adapter.active_scope = "student"
    adapter.forward_calls = []
    adapter.events = []
    adapter.observed_weights = []
    for name, value in teacher_weights.items():
        with torch.no_grad():
            adapter.snapshot_module.weight.fill_(value)
        adapter.add_named_parameters(name, target_components=["snapshot_module"], device="cpu")
    with torch.no_grad():
        adapter.snapshot_module.weight.fill_(student_weight)
    return adapter


class TwoComponentAdapterFake(BaseAdapter):
    """Adapter declaring a heterogeneous video/audio stochastic contract."""

    trajectory_component_order = ("video", "audio")

    def load_pipeline(self) -> Any:
        """Return an unused pipeline fake."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Unused: this adapter is driven through the structured surface."""
        raise NotImplementedError

    def train(self, mode: bool = True) -> None:
        """Record the train-mode switch."""
        self.events.append("mode:train" if mode else "mode:eval")

    def ema_step(self, step: int) -> None:
        """Record the sampling-EMA advance."""
        self.events.append(f"ema_step:{step}")


def _two_component_adapter(
    video_dynamics: str = "Flow-SDE", audio_dynamics: str = "Flow-SDE"
) -> TwoComponentAdapterFake:
    adapter = object.__new__(TwoComponentAdapterFake)
    video = SchedulerFake(video_dynamics)
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake(audio_dynamics)},
        primary_name="video",
    )
    adapter.events = []
    adapter.forward_calls = []
    return adapter


class OptimizerFake:
    """Optimizer recording its call order into the shared event log."""

    def __init__(self, events: List[str]) -> None:
        self.events = events

    def step(self) -> None:
        """Record the optimizer step."""
        self.events.append("optimizer:step")

    def zero_grad(self) -> None:
        """Record the gradient reset."""
        self.events.append("optimizer:zero_grad")


class AcceleratorFake:
    """Single-process accelerator recording backward / clip / reduce calls."""

    def __init__(self, events: List[str], num_processes: int = 1) -> None:
        self.events = events
        self.device = torch.device("cpu")
        self.num_processes = num_processes
        self.sync_gradients = True
        self.is_main_process = True
        self.is_local_main_process = False
        self.losses: List[torch.Tensor] = []
        self.reduced: List[Tuple[str, torch.Tensor]] = []

    @contextmanager
    def accumulate(self, model: Any) -> Iterator[None]:
        """Record the accumulation window."""
        self.events.append("accumulate:enter")
        try:
            yield
        finally:
            self.events.append("accumulate:exit")

    def backward(self, loss: torch.Tensor) -> None:
        """Record the loss and run the real backward."""
        self.events.append("backward")
        self.losses.append(loss.detach().clone())
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        """Record the clip call without touching the gradients."""
        self.events.append("clip_grad_norm")
        return torch.tensor(1.5)

    def reduce(self, tensor: Any, reduction: str = "sum") -> Any:
        """Return the local value; a no-op in single-process contexts."""
        self.events.append(f"reduce:{reduction}")
        self.reduced.append((reduction, tensor.detach().clone()))
        return tensor


class PeerSumAcceleratorFake(AcceleratorFake):
    """Two-rank accelerator adding a fixed peer contribution to sum reductions."""

    def __init__(self, events: List[str], peer_packed: torch.Tensor) -> None:
        super().__init__(events, num_processes=2)
        self.peer_packed = peer_packed

    def reduce(self, tensor: Any, reduction: str = "sum") -> Any:
        """Sum the rank-local per-teacher packs with a recorded peer rank."""
        self.events.append(f"reduce:{reduction}")
        self.reduced.append((reduction, tensor.detach().clone()))
        return tensor + self.peer_packed


class LoggerFake:
    """Logging backend recording every payload the trainer emits."""

    def __init__(self, events: List[str]) -> None:
        self.events = events
        self.payloads: List[Tuple[int, Dict[str, Any]]] = []

    def log_data(self, data: Dict[str, Any], step: int) -> None:
        """Record one logged payload."""
        self.events.append("log")
        self.payloads.append((step, dict(data)))


def _samples(sources: List[Optional[str]], values: List[float], num_steps: int) -> List[BaseSample]:
    """Rollout samples whose stored trajectory covers ``num_steps`` transitions."""
    timesteps = torch.linspace(1000.0, 1000.0 / (num_steps + 1), num_steps)
    return [
        BaseSample(
            timesteps=timesteps.clone(),
            all_latents=torch.stack(
                [
                    torch.tensor([value + position, value - position, value * 0.5 + position])
                    for position in range(num_steps + 1)
                ]
            ),
            latent_index_map=torch.arange(num_steps + 1),
            prompt_embeds=torch.tensor([value]),
            source=source,
        )
        for source, value in zip(sources, values)
    ]


def _trainer(
    adapter: TrainableAdapterFake,
    *,
    teacher_names: List[str],
    source_to_teacher: Dict[str, int],
    teacher_gs: Optional[List[float]] = None,
    events: Optional[List[str]] = None,
    accelerator: Optional[AcceleratorFake] = None,
    loss_target: str = "xt",
    self_normalize: bool = False,
    is_sde: bool = False,
    num_inference_steps: int = 2,
    per_device_batch_size: int = 2,
    num_inner_epochs: int = 1,
) -> DiffusionOPDTrainer:
    """Assemble a DiffusionOPD trainer whose ``optimize()`` runs end to end."""
    events = events if events is not None else adapter.events
    trainer = object.__new__(DiffusionOPDTrainer)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(
        seed=5,
        guidance_scale=3.0,
        max_grad_norm=1.0,
        per_device_batch_size=per_device_batch_size,
        num_inner_epochs=num_inner_epochs,
        num_inference_steps=num_inference_steps,
        timestep_range=(0.0, 1.0),
        loss_target=loss_target,
        self_normalize=self_normalize,
        offload_samples_to_cpu=False,
        shuffle_samples=False,
        max_epochs=1,
    )
    trainer.autocast = nullcontext
    trainer.accelerator = accelerator if accelerator is not None else AcceleratorFake(events)
    trainer.optimizer = OptimizerFake(events)
    trainer.model_bundle = object()
    trainer.logger = LoggerFake(events)
    trainer.log_args = SimpleNamespace(verbose=False, save_freq=0, save_dir=None, run_name="run")
    trainer.epoch = 4
    trainer.step = 0
    trainer._is_sde = is_sde
    trainer._student_noise_level = (
        float(adapter.scheduler_group.primary.noise_level) if is_sde else 0.0
    )
    trainer._teacher_names = list(teacher_names)
    trainer._teacher_gs = list(teacher_gs) if teacher_gs is not None else [3.0] * len(teacher_names)
    trainer._source_to_teacher = dict(source_to_teacher)
    trainer._available_sources = set(source_to_teacher)
    trainer._teacher_target_store_device = torch.device("cpu")
    return trainer


def _two_teacher_trainer(adapter: TrainableAdapterFake, **kwargs: Any) -> DiffusionOPDTrainer:
    return _trainer(
        adapter,
        teacher_names=["teacher_a", "teacher_b"],
        source_to_teacher={"ds_a": 0, "ds_b": 1},
        **kwargs,
    )


def _mixed_source_samples(num_steps: int = 2) -> List[BaseSample]:
    """Four samples alternating sources, so every micro-batch is source-mixed."""
    return _samples(["ds_a", "ds_b", "ds_a", "ds_b"], [1.0, 2.0, 3.0, 4.0], num_steps)


# ============================ Global two-pass order ============================


def test_optimize_caches_every_teacher_target_before_any_student_forward() -> None:
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    trainer = _two_teacher_trainer(adapter, events=events, accelerator=AcceleratorFake(events))

    trainer.optimize(_mixed_source_samples())

    pass_one = [
        "enter:teacher_a",
        "forward:teacher_a",
        "forward:teacher_a",
        "exit:teacher_a",
        "enter:teacher_b",
        "forward:teacher_b",
        "forward:teacher_b",
        "exit:teacher_b",
    ]
    pass_two: List[str] = []
    for _ in range(4):  # 2 micro-batches x 2 distilled steps
        pass_two += [
            "accumulate:enter",
            "forward:student",
            "backward",
            "clip_grad_norm",
            "optimizer:step",
            "optimizer:zero_grad",
            "reduce:sum",
            "log",
            "accumulate:exit",
        ]
    assert events == ["mode:train"] + pass_one + ["mode:train"] + pass_two

    train_switches = [index for index, event in enumerate(events) if event == "mode:train"]
    last_teacher_forward = max(
        index for index, event in enumerate(events) if event.startswith("forward:teacher")
    )
    first_student_forward = min(
        index for index, event in enumerate(events) if event == "forward:student"
    )
    assert last_teacher_forward < train_switches[1] < first_student_forward


def test_optimize_swaps_each_teacher_exactly_once_per_epoch() -> None:
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    trainer = _two_teacher_trainer(
        adapter, events=events, accelerator=AcceleratorFake(events), per_device_batch_size=1
    )

    trainer.optimize(_mixed_source_samples())

    assert events.count("enter:teacher_a") == 1
    assert events.count("enter:teacher_b") == 1
    # Two samples per teacher at batch size 1 => two batches x two steps each.
    assert events.count("forward:teacher_a") == 4
    assert events.count("forward:teacher_b") == 4


def test_optimize_routes_each_teacher_guidance_scale_and_keeps_the_student_scale() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter, teacher_gs=[1.0, 7.5])

    trainer.optimize(_mixed_source_samples())

    scales = {call["scope"]: call["guidance_scale"] for call in adapter.forward_calls}
    assert scales == {"teacher_a": 1.0, "teacher_b": 7.5, "student": 3.0}


def test_optimize_gives_the_gradient_graph_to_the_student_forward_only() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)

    trainer.optimize(_mixed_source_samples())

    graph_owners = {call["scope"]: call["requires_grad"] for call in adapter.forward_calls}
    assert graph_owners == {"teacher_a": False, "teacher_b": False, "student": True}
    assert adapter.weight.grad is not None
    assert all(not loss.requires_grad for loss in trainer.accelerator.losses)


def test_optimize_keeps_the_optimizer_cadence_and_logging_step_sequence() -> None:
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    trainer = _two_teacher_trainer(adapter, events=events, accelerator=AcceleratorFake(events))

    trainer.optimize(_mixed_source_samples())

    for event in ("backward", "clip_grad_norm", "optimizer:step", "optimizer:zero_grad", "log"):
        assert events.count(event) == 4, event
    assert [step for step, _ in trainer.logger.payloads] == [0, 1, 2, 3]
    assert trainer.step == 4


def test_optimize_logs_a_per_teacher_and_overall_distillation_loss() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)

    trainer.optimize(_mixed_source_samples())

    _, payload = trainer.logger.payloads[0]
    assert set(payload) == {
        "train/distill_loss",
        "train/distill_loss_teacher_a",
        "train/distill_loss_teacher_b",
        "train/grad_norm",
    }


def test_optimize_reduces_a_fixed_shape_per_teacher_pack_once_per_step() -> None:
    """The collective must be collective-safe: identical shape on every rank."""
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)

    trainer.optimize(_mixed_source_samples())

    assert len(trainer.accelerator.reduced) == 4
    for reduction, packed in trainer.accelerator.reduced:
        assert reduction == "sum"
        assert packed.shape == (2, 2)


def test_optimize_uses_the_globally_reduced_counts_for_the_logged_means() -> None:
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    peer_packed = torch.tensor([[2.0, 6.0], [1.0, 3.0]])
    accelerator = PeerSumAcceleratorFake(events, peer_packed)
    trainer = _two_teacher_trainer(adapter, events=events, accelerator=accelerator)

    trainer.optimize(_mixed_source_samples())

    _, local_packed = accelerator.reduced[0]
    _, payload = trainer.logger.payloads[0]
    summed = local_packed + peer_packed
    assert torch.equal(payload["train/distill_loss_teacher_a"], summed[0][0] / summed[1][0])
    assert torch.equal(payload["train/distill_loss"], summed[0].sum() / summed[1].sum())


def test_optimize_matches_each_sample_to_its_own_routed_teacher_target() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)
    samples = _mixed_source_samples()

    trainer.optimize(samples)

    for sample, expected_scope in zip(samples, ["teacher_a", "teacher_b"] * 2):
        cache = sample.extra_kwargs["teacher_target"]
        assert tuple(cache) == ("latent",)
        stored = cache["latent"]
        assert stored.shape == (2, 3)
        latents = sample.all_latents[0]
        expected = latents + adapter.velocity(latents, expected_scope)
        assert torch.equal(stored[0], expected)


# ============================ Teacher cache validation ============================


def _cached_batch(trainer: DiffusionOPDTrainer, samples: List[BaseSample]) -> Any:
    train_steps = trainer._select_train_step_indices(
        trainer.training_args.num_inference_steps, trainer.training_args.timestep_range
    )
    trainer._precompute_teacher_targets(samples, train_steps)
    return BaseSample.stack(samples)


def test_teacher_cache_is_an_ordered_component_mapping_of_stacked_steps() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)
    samples = _mixed_source_samples()

    batch = _cached_batch(trainer, samples)
    cache = trainer._require_teacher_target_cache(batch, num_steps=2)

    assert tuple(cache) == ("latent",)
    assert cache["latent"].shape == (4, 2, 3)


def test_pass_two_rejects_a_missing_teacher_cache() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)
    batch = BaseSample.stack(_mixed_source_samples())

    with pytest.raises(
        ValueError,
        match=r"DiffusionOPDTrainer.*teacher_target.*component order \('latent',\)"
        r".*received None.*_precompute_teacher_targets",
    ):
        trainer._require_teacher_target_cache(batch, num_steps=2)


def test_pass_two_rejects_a_single_tensor_teacher_cache() -> None:
    """The pre-migration layout stored one stacked tensor; it must not be indexed."""
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)
    samples = _mixed_source_samples()
    for sample in samples:
        sample.extra_kwargs["teacher_target"] = torch.zeros(2, 3)
    batch = BaseSample.stack(samples)

    with pytest.raises(
        ValueError,
        match=r"teacher_target.*component order \('latent',\).*received Tensor",
    ):
        trainer._require_teacher_target_cache(batch, num_steps=2)


def test_pass_two_rejects_a_teacher_cache_in_the_wrong_component_order() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)
    samples = _mixed_source_samples()
    for sample in samples:
        sample.extra_kwargs["teacher_target"] = {"other": torch.zeros(2, 3)}
    batch = BaseSample.stack(samples)

    with pytest.raises(
        ValueError,
        match=r"teacher_target.*component order \('latent',\), received \('other',\)",
    ):
        trainer._require_teacher_target_cache(batch, num_steps=2)


def test_pass_two_rejects_a_teacher_cache_with_a_drifted_step_count() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)
    samples = _mixed_source_samples()
    batch = _cached_batch(trainer, samples)

    with pytest.raises(
        ValueError,
        match=r"teacher_target.*component 'latent'.*3 stored distillation steps.*\(4, 2, 3\)",
    ):
        trainer._require_teacher_target_cache(batch, num_steps=3)


def test_teacher_cache_is_stored_on_the_configured_offload_device() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)
    trainer._teacher_target_store_device = torch.device("cpu")
    samples = _mixed_source_samples()

    _cached_batch(trainer, samples)

    for sample in samples:
        assert sample.extra_kwargs["teacher_target"]["latent"].device == torch.device("cpu")


# ============================ Replay contract ============================


def test_student_and_teacher_replay_the_same_states_times_and_noise_level() -> None:
    adapter = _adapter(dynamics_type="Flow-SDE")
    trainer = _two_teacher_trainer(adapter, is_sde=True)

    trainer.optimize(_mixed_source_samples())

    teacher_a = [call for call in adapter.forward_calls if call["scope"] == "teacher_a"]
    student = [call for call in adapter.forward_calls if call["scope"] == "student"]
    # First micro-batch of pass 2 holds samples 0 and 1 (ds_a, ds_b); teacher_a
    # saw samples 0 and 2. Compare the shared row for the first distilled step.
    assert torch.equal(teacher_a[0]["latents"][0], student[0]["latents"][0])
    assert torch.equal(teacher_a[0]["t"], student[0]["t"])
    assert torch.equal(teacher_a[0]["t_next"], student[0]["t_next"])
    assert teacher_a[0]["noise_level"] == student[0]["noise_level"] == 0.7


def test_terminal_step_replays_the_legacy_zero_next_timestep() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)

    trainer.optimize(_mixed_source_samples())

    terminal = [call for call in adapter.forward_calls if call["scope"] == "student"][1]
    assert terminal["t_next"].tolist() == [0] * terminal["latents"].shape[0]


@pytest.mark.parametrize(
    "loss_target, expected",
    [("xt", ("next_latents_mean",)), ("v", ("velocity",)), ("x0", ("velocity",))],
)
def test_forward_requests_the_canonical_legacy_field_for_each_target(
    loss_target: str, expected: Tuple[str, ...]
) -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter, loss_target=loss_target)

    trainer.optimize(_mixed_source_samples())

    for call in adapter.forward_calls:
        assert call["return_kwargs"] == expected


def test_stochastic_student_pass_also_requests_the_transition_statistics() -> None:
    adapter = _adapter(dynamics_type="Flow-SDE")
    trainer = _two_teacher_trainer(adapter, is_sde=True)

    trainer.optimize(_mixed_source_samples())

    teacher_fields = {
        call["return_kwargs"] for call in adapter.forward_calls if call["scope"] != "student"
    }
    student_fields = {
        call["return_kwargs"] for call in adapter.forward_calls if call["scope"] == "student"
    }
    assert teacher_fields == {("next_latents_mean",)}
    assert student_fields == {("next_latents_mean", "std_dev_t", "dt")}


# ============================ Legacy numerical oracle ============================


def _single_step_samples() -> List[BaseSample]:
    return _samples([None, None], [1.0, 2.0], num_steps=1)


def _single_teacher_trainer(adapter: TrainableAdapterFake, **kwargs: Any) -> DiffusionOPDTrainer:
    return _trainer(
        adapter,
        teacher_names=["teacher_a"],
        source_to_teacher={"ds_a": 0},
        teacher_gs=[1.0],
        num_inference_steps=1,
        **kwargs,
    )


def _legacy_opd_step(
    adapter: TrainableAdapterFake,
    weight: torch.nn.Parameter,
    samples: List[BaseSample],
    *,
    loss_target: str,
    self_normalize: bool,
    is_sde: bool,
) -> torch.Tensor:
    """Pre-migration DiffusionOPD loss for one micro-batch and one distilled step.

    Written against raw tensors and the legacy index/reduction expressions, so it
    is an independent oracle rather than a second call into the migrated helpers.
    """
    batch = BaseSample.stack(samples)
    latent_index_map = batch["latent_index_map"]
    latents = batch["all_latents"][:, latent_index_map[0]]
    timestep = batch["timesteps"][:, 0]
    batch_size = latents.shape[0]

    def project(scope: str, grad: bool) -> torch.Tensor:
        with torch.enable_grad() if grad else torch.no_grad():
            velocity = latents * weight + _SCOPE_BIAS[scope]
            if loss_target == "xt":
                return latents + velocity
            if loss_target == "v":
                return velocity
            sigma = (timestep.float() / 1000.0).clamp(0.0, 1.0)
            sigma = sigma.view(-1, *([1] * (latents.ndim - 1)))
            return latents.float() - sigma * velocity.float()

    teacher_target = project("teacher_a", grad=False).detach()
    student_target = project("student", grad=True)

    error = student_target.float() - teacher_target.float()
    per_sample = error.square().flatten(1).mean(dim=1)
    if self_normalize:
        scale = error.abs().flatten(1).mean(dim=1).detach()
        per_sample = per_sample / (scale + 1e-8)
    if is_sde:
        std_dev_t = torch.full((batch_size,) + (1,) * (latents.ndim - 1), 0.5)
        dt = torch.full((batch_size,) + (1,) * (latents.ndim - 1), -0.5)
        denom = (std_dev_t.float() ** 2 * (-dt.float())).clamp_min(1e-8)
        per_sample = per_sample / denom.reshape(batch_size, -1).mean(dim=1)
    return per_sample.mean()


@pytest.mark.parametrize("loss_target", ["xt", "v", "x0"])
@pytest.mark.parametrize("self_normalize", [False, True])
def test_optimize_matches_the_legacy_ode_loss_and_gradient(
    loss_target: str, self_normalize: bool
) -> None:
    samples = _single_step_samples()

    adapter = _adapter()
    trainer = _single_teacher_trainer(
        adapter, loss_target=loss_target, self_normalize=self_normalize
    )
    trainer.optimize(samples)

    oracle_adapter = _adapter()
    legacy = _legacy_opd_step(
        oracle_adapter,
        oracle_adapter.weight,
        _single_step_samples(),
        loss_target=loss_target,
        self_normalize=self_normalize,
        is_sde=False,
    )
    legacy.backward()

    assert len(trainer.accelerator.losses) == 1
    assert torch.equal(trainer.accelerator.losses[0], legacy.detach())
    assert torch.equal(adapter.weight.grad, oracle_adapter.weight.grad)


@pytest.mark.parametrize("self_normalize", [False, True])
def test_optimize_matches_the_legacy_sde_loss_and_gradient(self_normalize: bool) -> None:
    samples = _single_step_samples()

    adapter = _adapter(dynamics_type="Flow-SDE")
    trainer = _single_teacher_trainer(adapter, self_normalize=self_normalize, is_sde=True)
    trainer.optimize(samples)

    oracle_adapter = _adapter(dynamics_type="Flow-SDE")
    legacy = _legacy_opd_step(
        oracle_adapter,
        oracle_adapter.weight,
        _single_step_samples(),
        loss_target="xt",
        self_normalize=self_normalize,
        is_sde=True,
    )
    legacy.backward()

    assert torch.equal(trainer.accelerator.losses[0], legacy.detach())
    assert torch.equal(adapter.weight.grad, oracle_adapter.weight.grad)


def test_stochastic_pass_uses_every_component_scheduler_denominator() -> None:
    adapter = _adapter(dynamics_type="Flow-SDE")
    trainer = _single_teacher_trainer(adapter, is_sde=True)
    batch = BaseSample.stack(_single_step_samples())
    replay, _, output = trainer._forward_step(
        batch, 0, guidance_scale=3.0, context="student", include_transition_stats=True
    )

    denominators = trainer._component_kl_denominators(output, replay, step_index=0)

    assert tuple(denominators) == ("latent",)
    assert torch.equal(denominators["latent"], torch.full((2,), 0.125))


def test_stochastic_pass_rejects_a_denominator_that_is_not_one_value_per_sample() -> None:
    adapter = _adapter(dynamics_type="Flow-SDE")
    trainer = _single_teacher_trainer(adapter, is_sde=True)
    batch = BaseSample.stack(_single_step_samples())
    replay, _, output = trainer._forward_step(
        batch, 0, guidance_scale=3.0, context="student", include_transition_stats=True
    )
    object.__setattr__(output, "std_dev_t", {"latent": torch.full((2, 3), 0.5)})
    object.__setattr__(output, "dt", {"latent": torch.full((2, 3), -0.5)})

    with pytest.raises(
        ValueError,
        match=r"DiffusionOPDTrainer.*step_index=0.*component 'latent'.*one value per sample"
        r".*\(2, 3\)",
    ):
        trainer._component_kl_denominators(output, replay, step_index=0)


def test_stochastic_pass_rejects_missing_transition_statistics() -> None:
    adapter = _adapter(dynamics_type="Flow-SDE")
    trainer = _single_teacher_trainer(adapter, is_sde=True)
    batch = BaseSample.stack(_single_step_samples())
    replay, _, output = trainer._forward_step(batch, 0, guidance_scale=3.0, context="student")

    with pytest.raises(
        ValueError,
        match=r"DiffusionOPDTrainer.*step_index=0.*std_dev_t.*component order \('latent',\)"
        r".*received None",
    ):
        trainer._component_kl_denominators(output, replay, step_index=0)


# ==================== Real named-parameter snapshot swaps ====================


def test_optimize_swaps_real_teacher_weights_and_restores_the_student() -> None:
    adapter = _snapshot_adapter(0.7, {"teacher_a": 2.0, "teacher_b": -3.0})
    trainer = _two_teacher_trainer(adapter)
    samples = _mixed_source_samples()

    trainer.optimize(samples)

    # Each context installed its own snapshot, and neither leaked into the other.
    assert adapter.observed_weights == [("teacher_a", 2.0), ("teacher_b", -3.0)]
    # The live student weight is back after both contexts closed.
    assert torch.equal(adapter.weight.detach(), torch.tensor(0.7))
    student_calls = [call for call in adapter.forward_calls if call["scope"] == "student"]
    assert all(call["requires_grad"] for call in student_calls)


def test_teacher_targets_are_cached_from_the_swapped_in_teacher_weights() -> None:
    adapter = _snapshot_adapter(0.7, {"teacher_a": 2.0, "teacher_b": -3.0})
    trainer = _two_teacher_trainer(adapter)
    samples = _mixed_source_samples()

    trainer.optimize(samples)

    for sample, teacher_weight in zip(samples, [2.0, -3.0] * 2):
        latents = sample.all_latents[0]
        stored = sample.extra_kwargs["teacher_target"]["latent"][0]
        assert torch.equal(stored, latents + latents * teacher_weight)


def test_every_real_teacher_swap_closes_before_the_first_student_forward() -> None:
    adapter = _snapshot_adapter(0.7, {"teacher_a": 2.0, "teacher_b": -3.0})
    events: List[str] = []
    adapter.events = events
    trainer = _two_teacher_trainer(adapter, events=events, accelerator=AcceleratorFake(events))

    trainer.optimize(_mixed_source_samples())

    last_exit = max(index for index, event in enumerate(events) if event.startswith("exit:"))
    first_student = min(index for index, event in enumerate(events) if event == "forward:student")
    assert last_exit < first_student
    assert [event for event in events if event.startswith(("enter:", "exit:"))] == [
        "enter:teacher_a",
        "exit:teacher_a",
        "enter:teacher_b",
        "exit:teacher_b",
    ]


# ============ Two-component stochastic path: denominators to backward ============


def _two_component_replay(video: torch.Tensor, audio: torch.Tensor) -> ReplayStep:
    return ReplayStep(
        state=LatentState({"video": video, "audio": audio}),
        next_state=LatentState({"video": video, "audio": audio}),
        times=ComponentTimes(
            timestep={"video": torch.full((2,), 700.0), "audio": torch.full((2,), 650.0)},
            next_timestep={"video": torch.full((2,), 300.0), "audio": torch.full((2,), 250.0)},
        ),
    )


def test_two_component_stochastic_path_backpropagates_each_component_denominator() -> None:
    """`_component_kl_denominators` -> projection -> structured loss -> backward."""
    torch.manual_seed(21)
    adapter = _two_component_adapter()
    trainer = _single_teacher_trainer(adapter, is_sde=True)
    weight = torch.nn.Parameter(torch.tensor(0.6))
    video, audio = torch.randn(2, 3, 4), torch.randn(2, 5)
    replay = _two_component_replay(video, audio)
    teacher = LatentState({"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)})
    output = MultiModalStepOutput(
        next_state_mean=LatentState({"video": video * weight, "audio": audio * weight}),
        # Distinct per-component statistics, and deliberately not powers of two:
        # one shared denominator, or dividing after the reduction, would be wrong.
        std_dev_t={"video": torch.full((2, 1, 1), 0.5), "audio": torch.full((2, 1), 0.25)},
        dt={"video": torch.full((2, 1, 1), -0.3), "audio": torch.full((2, 1), -0.7)},
    )

    denominators = trainer._component_kl_denominators(output, replay, step_index=1)
    student_target = project_distillation_target_state(
        adapter,
        loss_target="xt",
        state=replay.state,
        output=output,
        times=replay.times,
    )
    loss = compute_structured_distillation_loss(
        adapter,
        student_target=student_target,
        teacher_target=teacher,
        state=replay.state,
        self_normalize=False,
        denominators=denominators,
    ).mean()
    loss.backward()

    assert torch.equal(
        denominators["video"], torch.full((2, 1, 1), 0.5).square().mul(0.3).flatten()
    )
    assert torch.equal(denominators["audio"], torch.full((2, 1), 0.25).square().mul(0.7).flatten())

    oracle_weight = torch.nn.Parameter(torch.tensor(0.6))
    video_error = (video * oracle_weight - teacher.components["video"]).square()
    audio_error = (audio * oracle_weight - teacher.components["audio"]).square()
    oracle = (
        (video_error / denominators["video"].reshape(2, 1, 1)).flatten(1).sum(dim=1)
        + (audio_error / denominators["audio"].reshape(2, 1)).flatten(1).sum(dim=1)
    ) / 17
    oracle_loss = oracle.mean()
    oracle_loss.backward()

    assert torch.equal(loss.detach(), oracle_loss.detach())
    assert torch.equal(weight.grad, oracle_weight.grad)


class ConstantDenominatorSchedulerFake(SchedulerFake):
    """Scheduler returning a fixed per-sample denominator."""

    def __init__(self, dynamics_type: str, values: List[float]) -> None:
        super().__init__(dynamics_type)
        self.values = values

    def get_kl_divergence_denominator(
        self, std_dev_t: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Return the configured denominator, bypassing the variance arithmetic."""
        return torch.tensor(self.values)


@pytest.mark.parametrize(
    "values, reason",
    [
        pytest.param([float("nan"), 0.5], "finite", id="nan"),
        pytest.param([float("inf"), 0.5], "finite", id="inf"),
        pytest.param([0.0, 0.5], "strictly positive", id="zero"),
        pytest.param([-0.25, 0.5], "strictly positive", id="negative"),
    ],
)
def test_two_component_denominators_reject_invalid_values_with_step_context(
    values: List[float], reason: str
) -> None:
    """An unusable transition variance must name the step and the component."""
    adapter = _two_component_adapter()
    adapter.scheduler_group = SchedulerGroup(
        {
            "video": ConstantDenominatorSchedulerFake("Flow-SDE", values),
            "audio": SchedulerFake("Flow-SDE"),
        },
        primary_name="video",
    )
    trainer = _single_teacher_trainer(adapter, is_sde=True)
    replay = _two_component_replay(torch.zeros(2, 3), torch.zeros(2, 5))
    output = MultiModalStepOutput(
        next_state_mean=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
        std_dev_t={"video": torch.full((2, 1), 0.5), "audio": torch.full((2, 1), 0.25)},
        dt={"video": torch.full((2, 1), -0.5), "audio": torch.full((2, 1), -0.125)},
    )

    with pytest.raises(
        ValueError,
        match=rf"step_index=1.*component 'video'.*{reason}",
    ):
        trainer._component_kl_denominators(output, replay, step_index=1)


class ScalarDenominatorSchedulerFake(SchedulerFake):
    """Scheduler collapsing the transition variance to a batch-less scalar."""

    def get_kl_divergence_denominator(
        self, std_dev_t: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Return a 0-dim denominator shared by the whole batch."""
        return (std_dev_t.float() ** 2 * (-dt.float())).clamp_min(1e-8).mean()


def test_two_component_denominators_reject_a_scalar_without_a_batch_axis() -> None:
    """A 0-dim denominator must fail the contract, not crash on ``shape[0]``."""
    adapter = _two_component_adapter()
    adapter.scheduler_group = SchedulerGroup(
        {
            "video": ScalarDenominatorSchedulerFake("Flow-SDE"),
            "audio": SchedulerFake("Flow-SDE"),
        },
        primary_name="video",
    )
    trainer = _single_teacher_trainer(adapter, is_sde=True)
    replay = _two_component_replay(torch.zeros(2, 3), torch.zeros(2, 5))
    output = MultiModalStepOutput(
        next_state_mean=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
        std_dev_t={"video": torch.full((2, 1), 0.5), "audio": torch.full((2, 1), 0.25)},
        dt={"video": torch.full((2, 1), -0.5), "audio": torch.full((2, 1), -0.125)},
    )

    with pytest.raises(
        ValueError,
        match=r"step_index=1.*component 'video'.*one value per sample.*\(2,\).*\(\)",
    ):
        trainer._component_kl_denominators(output, replay, step_index=1)


# ==================== Teacher / student / step error context ====================


class DropVelocityAdapterFake(TrainableAdapterFake):
    """Adapter that omits the requested velocity for one scope only."""

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Return the normal output with the velocity stripped for ``drop_scope``."""
        output = super().forward(**kwargs)
        if self.active_scope == self.drop_scope:
            output.velocity = None
        return output


def _drop_velocity_adapter(drop_scope: str) -> DropVelocityAdapterFake:
    adapter = object.__new__(DropVelocityAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    adapter.weight = torch.nn.Parameter(torch.tensor(0.7))
    adapter.active_scope = "student"
    adapter.drop_scope = drop_scope
    adapter.forward_calls = []
    adapter.events = []
    return adapter


def test_teacher_projection_errors_name_the_teacher_and_the_step() -> None:
    adapter = _drop_velocity_adapter("teacher_b")
    trainer = _two_teacher_trainer(adapter, loss_target="v")

    with pytest.raises(
        ValueError,
        match=r"teacher 'teacher_b' pass at step_index=0.*velocity.*received None",
    ):
        trainer.optimize(_mixed_source_samples())


def test_student_projection_errors_name_the_student_pass_and_the_step() -> None:
    adapter = _drop_velocity_adapter("student")
    trainer = _two_teacher_trainer(adapter, loss_target="v")

    with pytest.raises(
        ValueError,
        match=r"student pass at step_index=0.*velocity.*received None",
    ):
        trainer.optimize(_mixed_source_samples())


# ============================ Lifecycle ============================


def test_start_seeds_every_component_scheduler_before_each_epoch() -> None:
    adapter = _two_component_adapter()
    trainer = _two_teacher_trainer(adapter)
    iterations = iter([True, False])
    trainer.should_continue_training = lambda: next(iterations)
    trainer.eval_args = SimpleNamespace(eval_freq=0)
    trainer.sample = lambda: []
    trainer.optimize = lambda samples: None

    trainer.start()

    seeds = {
        name: adapter.scheduler_group[name].seeds for name in adapter.trajectory_component_order
    }
    assert seeds == {"video": [9], "audio": [9]}
    assert f"ema_step:{4}" in adapter.events
    assert trainer.epoch == 5


def test_prepare_feedback_stays_a_no_op() -> None:
    trainer = _two_teacher_trainer(_adapter())

    assert trainer.prepare_feedback(_mixed_source_samples()) is None


def test_optimize_without_samples_leaves_the_step_counter_untouched() -> None:
    adapter = _adapter()
    trainer = _two_teacher_trainer(adapter)

    trainer.optimize([])

    assert trainer.step == 0
    assert adapter.forward_calls == []


def test_pass_two_projects_the_student_target_as_a_latent_state() -> None:
    adapter = _adapter()
    trainer = _single_teacher_trainer(adapter)
    batch = BaseSample.stack(_single_step_samples())

    replay, target, _ = trainer._forward_step(batch, 0, guidance_scale=3.0, context="student")

    assert isinstance(target, LatentState)
    assert target.component_names == ("latent",)
    latents = replay.state.components["latent"]
    assert torch.equal(target.components["latent"], latents + adapter.velocity(latents, "student"))
