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

"""Cover the distillation metric buffer that feeds DMD, TDM, and TDM-R1 logging."""

from __future__ import annotations

import random
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from flow_factory.data_utils.multi_source import (
    MultiSourceTrainDataLoader,
    WeightedSourceBatchScheduler,
)
from flow_factory.data_utils.sampler import DistributedKRepeatSampler
from flow_factory.trainers.distillation.distillation_runtime import (
    generate_one_rollout_batch,
    pop_distillation_metrics,
    record_distillation_metric,
    record_state_statistics,
    run_distillation_training_step,
    run_role_phase,
)
from flow_factory.trainers.execution import TrainingProgress


class SingleRankAccelerator:
    """Reduce over one rank, which leaves every buffered value unchanged."""

    device = torch.device("cpu")
    num_processes = 1
    process_index = 0

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        """Return the tensor untouched, as a one-rank group reduction would."""
        # Accelerate maps every reduction other than "max" onto a sum, so the buffer
        # may only ask for the two it really implements.
        if reduction not in {"sum", "max"}:
            raise ValueError(f"expected a sum or max reduction, received {reduction!r}")
        return tensor


class DivergentAccelerator(SingleRankAccelerator):
    """Simulate a peer rank that buffered a larger metric set."""

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        """Raise the group maximum above this rank's own signature."""
        if reduction == "max":
            return tensor + 1.0
        return tensor


@contextmanager
def _null_context() -> Iterator[None]:
    """Stand in for the trainer's sampling context."""
    yield


class _InfiniteEpochBatchSampler:
    """Expose the epoch controls used by the online grouped samplers."""

    num_batches_per_epoch = 3

    def __init__(self) -> None:
        self.epoch = 0
        self.set_epoch_calls: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.set_epoch_calls.append(epoch)


class _InfiniteGroupedLoader:
    """Yield finite epoch blocks through one never-ending iterator."""

    def __init__(self) -> None:
        self.batch_sampler = _InfiniteEpochBatchSampler()

    def __iter__(self) -> Iterator[tuple[int, int]]:
        while True:
            epoch = self.batch_sampler.epoch
            for batch_offset in range(self.batch_sampler.num_batches_per_epoch):
                yield epoch, batch_offset
            self.batch_sampler.epoch += 1


class _RandomizedCursorDataset(Dataset):
    """Expose parent-process RNG draws made by one real DataLoader fetch."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "index": index,
            "python_draw": random.random(),
            "numpy_draw": float(np.random.random()),
            "torch_draw": torch.rand(()),
        }


class _ValueDataset(Dataset):
    """Return deterministic dictionary rows accepted by the multi-source wrapper."""

    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"value": torch.tensor(self.values[index])}


def _rollout_cursor_trainer(
    progress: TrainingProgress,
    *,
    accumulation_steps: int = 2,
    losses_per_rollout: int = 1,
) -> SimpleNamespace:
    """Build the one-batch rollout surface around an infinite grouped loader."""
    return SimpleNamespace(
        progress=progress,
        training_args=SimpleNamespace(
            gradient_accumulation_steps=accumulation_steps,
            num_batches_per_epoch=3,
            get_num_train_timesteps=lambda config: losses_per_rollout,
        ),
        dataloader=_InfiniteGroupedLoader(),
        adapter=SimpleNamespace(rollout=lambda: None),
        _rollout_acceleration=_null_context,
        autocast=_null_context,
        sample_batch=lambda batch, **kwargs: [batch],
        _rollout_data_iter=None,
        _rollout_batches_consumed=None,
    )


def _real_rollout_cursor_trainer(
    progress: TrainingProgress,
    *,
    use_explicit_generator: bool,
) -> SimpleNamespace:
    """Build a real infinite grouped DataLoader with observable RNG fetches."""
    dataset = _RandomizedCursorDataset()
    sampler = DistributedKRepeatSampler(
        dataset,
        batch_size=1,
        group_size=1,
        unique_sample_num=3,
        num_replicas=1,
        rank=0,
        seed=17,
    )
    loader_generator = torch.Generator().manual_seed(91) if use_explicit_generator else None
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        generator=loader_generator,
    )

    def sample_batch(batch: Dict[str, torch.Tensor], **kwargs: Any) -> list[tuple]:
        del kwargs
        return [
            (
                int(batch["index"].item()),
                float(batch["python_draw"].item()),
                float(batch["numpy_draw"].item()),
                float(batch["torch_draw"].item()),
            )
        ]

    return SimpleNamespace(
        progress=progress,
        training_args=SimpleNamespace(
            gradient_accumulation_steps=2,
            num_batches_per_epoch=3,
            get_num_train_timesteps=lambda config: 1,
        ),
        dataloader=dataloader,
        adapter=SimpleNamespace(rollout=lambda: None),
        _rollout_acceleration=_null_context,
        autocast=_null_context,
        sample_batch=sample_batch,
        _rollout_data_iter=None,
        _rollout_batches_consumed=None,
    )


def _source_loader(
    values: list[int],
    *,
    seed: int,
    use_explicit_generator: bool = False,
) -> DataLoader:
    dataset = _ValueDataset(values)
    sampler = DistributedKRepeatSampler(
        dataset,
        batch_size=1,
        group_size=1,
        unique_sample_num=len(dataset),
        num_replicas=1,
        rank=0,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed + 1000) if use_explicit_generator else None
    return DataLoader(dataset, batch_sampler=sampler, generator=generator)


def _finite_multi_source_loader(
    *,
    use_explicit_generators: bool = False,
) -> MultiSourceTrainDataLoader:
    loaders = {
        "a": _source_loader(
            [10],
            seed=11,
            use_explicit_generator=use_explicit_generators,
        ),
        "b": _source_loader(
            [20, 21],
            seed=13,
            use_explicit_generator=use_explicit_generators,
        ),
    }
    return MultiSourceTrainDataLoader(
        loaders,
        WeightedSourceBatchScheduler({"a": 1, "b": 2}, seed=17),
        batch_size=1,
    )


def _finite_multi_source_trainer(
    progress: TrainingProgress,
    *,
    accumulation_steps: int,
) -> SimpleNamespace:
    def sample_batch(batch: Dict[str, Any], **kwargs: Any) -> list[tuple[str, int]]:
        del kwargs
        return [(batch["__source__"][0], int(batch["value"].item()))]

    return SimpleNamespace(
        progress=progress,
        training_args=SimpleNamespace(
            gradient_accumulation_steps=accumulation_steps,
            # DMD2 validates the global geometry before its per-source samplers
            # independently round their quotas. The finite wrapper length is the
            # authoritative result when those two values differ.
            num_batches_per_epoch=2,
            get_num_train_timesteps=lambda config: 1,
        ),
        dataloader=_finite_multi_source_loader(),
        adapter=SimpleNamespace(rollout=lambda: None),
        _rollout_acceleration=_null_context,
        autocast=_null_context,
        sample_batch=sample_batch,
        _rollout_data_iter=None,
        _rollout_batches_consumed=None,
    )


def _trainer(accelerator: Any = None) -> SimpleNamespace:
    """Build the minimal trainer surface the metric buffer touches."""
    return SimpleNamespace(accelerator=accelerator or SingleRankAccelerator())


def _state(**components: torch.Tensor) -> SimpleNamespace:
    """Build a latent state exposing only the components mapping."""
    return SimpleNamespace(components=components)


def test_repeated_records_are_averaged_not_overwritten() -> None:
    """A role phase records once per microbatch, so the log must be their mean."""
    trainer = _trainer()

    record_distillation_metric(trainer, "train/fake_loss", 1.0)
    record_distillation_metric(trainer, "train/fake_loss", 3.0)

    assert pop_distillation_metrics(trainer) == {"train/fake_loss": 2.0}


def test_popping_clears_the_buffer_so_epochs_do_not_bleed_together() -> None:
    """A metric left in the buffer would keep being averaged into later epochs."""
    trainer = _trainer()
    record_distillation_metric(trainer, "train/fake_loss", 1.0)

    pop_distillation_metrics(trainer)

    assert pop_distillation_metrics(trainer) == {}


def test_tensor_values_are_detached_from_the_graph() -> None:
    """Buffering a live loss tensor would pin its whole graph until the epoch ends."""
    trainer = _trainer()
    loss = (torch.ones(1, requires_grad=True) * 2).sum()

    record_distillation_metric(trainer, "train/generator_loss", loss)

    assert pop_distillation_metrics(trainer) == {"train/generator_loss": 2.0}


def test_state_statistics_pool_every_component() -> None:
    """A multi-component state must report one mean and std over all of its latents."""
    trainer = _trainer()

    record_state_statistics(
        trainer,
        "train/x0_gen",
        _state(image=torch.tensor([1.0, 3.0]), text=torch.tensor([5.0, 7.0])),
    )

    metrics = pop_distillation_metrics(trainer)
    assert metrics["train/x0_gen_mean"] == pytest.approx(4.0)
    # Population rather than sample std: the values are pooled from per-rank sums, where
    # a Bessel correction has no single N to apply.
    pooled = torch.tensor([1.0, 3.0, 5.0, 7.0])
    assert metrics["train/x0_gen_std"] == pytest.approx(float(pooled.std(unbiased=False)))


def test_diverging_metric_sets_raise_instead_of_deadlocking() -> None:
    """The reduction would hang on rank-dependent keys, which is undiagnosable at scale."""
    trainer = _trainer(DivergentAccelerator())
    record_distillation_metric(trainer, "train/fake_loss", 1.0)

    with pytest.raises(RuntimeError, match="same distillation metrics"):
        pop_distillation_metrics(trainer)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0", None])
def test_non_finite_scalars_are_rejected_at_the_call_site(value: Any) -> None:
    """A NaN loss silently averaged into the log hides the step that produced it."""
    with pytest.raises((TypeError, ValueError)):
        record_distillation_metric(_trainer(), "train/fake_loss", value)


def test_non_scalar_tensors_are_rejected() -> None:
    """Recording a per-sample tensor would silently mean-reduce a batch dimension."""
    with pytest.raises(ValueError, match="received a tensor of shape"):
        record_distillation_metric(_trainer(), "train/fake_loss", torch.ones(4))


def test_metrics_are_summed_across_ranks_before_averaging() -> None:
    """Two ranks recording different losses must log the mean of both, not one of them."""

    class TwoRankAccelerator(SingleRankAccelerator):
        num_processes = 2

        def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
            if reduction == "sum":
                # The peer rank buffered the same key twice with value 5.
                return tensor + torch.tensor([10.0, 2.0], dtype=torch.float64)
            return tensor

    trainer = _trainer(TwoRankAccelerator())
    record_distillation_metric(trainer, "train/fake_loss", 1.0)
    record_distillation_metric(trainer, "train/fake_loss", 3.0)

    metrics: Dict[str, float] = pop_distillation_metrics(trainer)
    assert metrics["train/fake_loss"] == pytest.approx((1.0 + 3.0 + 10.0) / 4.0)


def test_an_epoch_logs_what_its_roles_recorded() -> None:
    """Metrics recorded during optimize must reach the logger against the epoch's step."""
    logged: list = []

    def optimize(microbatches: Any) -> None:
        del microbatches
        record_distillation_metric(trainer, "train/generator_loss", 7.0)

    trainer = SimpleNamespace(
        accelerator=SingleRankAccelerator(),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=2,
            get_num_train_timesteps=lambda config: 1,
        ),
        config=SimpleNamespace(),
        epoch=3,
        step=11,
        show_progress_bar=False,
        sampling_context=lambda: _null_context(),
        sample=lambda: ["sample"],
        prepare_feedback=lambda samples: None,
        optimize=optimize,
        log_data=lambda data, step: logged.append((data, step)),
    )

    run_distillation_training_step(trainer)

    assert logged == [({"train/generator_loss": 7.0}, 11)]


def test_timestep_aligned_gas_samples_only_its_rollout_factor() -> None:
    sampled: list[str] = []
    optimized: list[list[list[str]]] = []
    trainer = SimpleNamespace(
        accelerator=SingleRankAccelerator(),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=8,
            get_num_train_timesteps=lambda config: 4,
        ),
        config=SimpleNamespace(),
        reward_buffer=None,
        epoch=0,
        step=0,
        show_progress_bar=False,
        sampling_context=_null_context,
        sample=lambda: sampled.append("sample") or ["sample"],
        prepare_feedback=lambda samples: None,
        optimize=lambda microbatches: optimized.append(microbatches),
        log_data=lambda data, step: None,
    )

    run_distillation_training_step(trainer)

    assert sampled == ["sample", "sample"]
    assert optimized == [[["sample"], ["sample"]]]


def test_role_variant_stays_active_through_backward_recomputation() -> None:
    events: list[tuple[str, str]] = []

    class Adapter:
        active = "generator"

        @contextmanager
        def use_component_variant(self, role_name: str) -> Iterator[None]:
            previous = self.active
            self.active = role_name
            try:
                yield
            finally:
                self.active = previous

    class Coordinator:
        roles = {"surrogate": SimpleNamespace(last_grad_norm=None)}

        @contextmanager
        def phase(self, role_name: str) -> Iterator[None]:
            events.append(("phase", role_name))
            yield

        @contextmanager
        def microbatch(self) -> Iterator[None]:
            yield

        def backward(self, loss: torch.Tensor) -> None:
            events.append(("backward", adapter.active))
            loss.backward()

    adapter = Adapter()
    trainer = SimpleNamespace(
        adapter=adapter,
        role_optimization=Coordinator(),
        accelerator=SingleRankAccelerator(),
        epoch=0,
        show_progress_bar=False,
        _finish_role_microbatch=lambda: None,
    )

    run_role_phase(
        trainer,
        "surrogate",
        [["sample"]],
        lambda batch: torch.ones((), requires_grad=True),
    )

    assert events == [("phase", "surrogate"), ("backward", "surrogate")]
    assert adapter.active == "generator"


def test_timestep_work_items_match_explicit_mean_gradient() -> None:
    events: list[tuple[str, float]] = []
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    reference = torch.nn.Parameter(torch.tensor(1.0))

    class Adapter:
        @contextmanager
        def use_component_variant(self, role_name: str) -> Iterator[None]:
            del role_name
            yield

    class Coordinator:
        roles = {"fake": SimpleNamespace(last_grad_norm=None)}

        @contextmanager
        def phase(self, role_name: str) -> Iterator[None]:
            del role_name
            yield

        @contextmanager
        def microbatch(self) -> Iterator[None]:
            yield

        def backward(self, loss: torch.Tensor) -> None:
            events.append(("backward", float(loss.detach())))
            (loss / 3).backward()

    def boundary_loss(coefficient: float) -> torch.Tensor:
        events.append(("forward", coefficient))
        return parameter * coefficient

    trainer = SimpleNamespace(
        adapter=Adapter(),
        role_optimization=Coordinator(),
        accelerator=SingleRankAccelerator(),
        epoch=0,
        show_progress_bar=False,
        _finish_role_microbatch=lambda: None,
    )

    run_role_phase(trainer, "fake", [1.0, 2.0, 3.0], boundary_loss)
    torch.stack([reference * value for value in (1.0, 2.0, 3.0)]).mean().backward()

    assert events == [
        ("forward", 1.0),
        ("backward", 1.0),
        ("forward", 2.0),
        ("backward", 2.0),
        ("forward", 3.0),
        ("backward", 3.0),
    ]
    assert parameter.grad is not None
    torch.testing.assert_close(parameter.grad, reference.grad)


def test_timestep_aligned_rollouts_finalize_one_shared_feedback_window() -> None:
    class RecordingBuffer:
        def __init__(self) -> None:
            self.samples: list = []
            self.clears = 0

        def clear(self) -> None:
            self.clears += 1
            self.samples = []

        def add_samples(self, samples: list) -> None:
            self.samples.extend(samples)

    buffer = RecordingBuffer()
    feedback: list[tuple[list, list]] = []
    trainer = SimpleNamespace(
        dataloader=[["prompt"]],
        adapter=SimpleNamespace(rollout=lambda: None),
        accelerator=SingleRankAccelerator(),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=2,
            get_num_train_timesteps=lambda config: 1,
        ),
        config=SimpleNamespace(),
        reward_buffer=buffer,
        epoch=0,
        step=0,
        show_progress_bar=False,
        sampling_context=_null_context,
        _rollout_acceleration=_null_context,
        autocast=_null_context,
        sample_batch=lambda batch, **kwargs: (
            kwargs["reward_buffer"].add_samples(["sample"]) or ["sample"]
        ),
        prepare_feedback=lambda samples: feedback.append((list(samples), list(buffer.samples))),
        optimize=lambda microbatches: None,
        log_data=lambda data, step: None,
    )
    trainer.sample = lambda: generate_one_rollout_batch(
        trainer,
        reward_buffer=buffer,
        algorithm_name="TDM-R1",
    )

    run_distillation_training_step(trainer)

    assert buffer.clears == 1
    assert buffer.samples == ["sample", "sample"]
    assert feedback == [
        (["sample", "sample"], ["sample", "sample"]),
    ]


def test_exact_resume_reconstructs_infinite_grouped_rollout_cursor() -> None:
    """A resumed distillation run must not restart its prompt sampler at batch zero."""
    uninterrupted = _rollout_cursor_trainer(TrainingProgress())
    observed = []
    for rollout_iteration in range(2):
        uninterrupted.progress = TrainingProgress(rollout_iteration=rollout_iteration)
        for _ in range(2):
            observed.extend(
                generate_one_rollout_batch(
                    uninterrupted,
                    reward_buffer=None,
                    algorithm_name="DMD2",
                )
            )
    uninterrupted.progress = TrainingProgress(rollout_iteration=2)
    expected_next = generate_one_rollout_batch(
        uninterrupted,
        reward_buffer=None,
        algorithm_name="DMD2",
    )

    resumed = _rollout_cursor_trainer(TrainingProgress(rollout_iteration=2))
    resumed_next = generate_one_rollout_batch(
        resumed,
        reward_buffer=None,
        algorithm_name="DMD2",
    )

    assert observed == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert expected_next == [(1, 1)]
    assert resumed_next == expected_next
    assert resumed.dataloader.batch_sampler.set_epoch_calls == [1]


def test_exact_resume_uses_rollout_factor_to_reconstruct_batch_count() -> None:
    """Boundary-aligned GAS must not make exact resume skip extra prompt batches."""
    resumed = _rollout_cursor_trainer(
        TrainingProgress(rollout_iteration=2),
        accumulation_steps=8,
        losses_per_rollout=4,
    )

    next_batch = generate_one_rollout_batch(
        resumed,
        reward_buffer=None,
        algorithm_name="TDM",
    )

    assert next_batch == [(1, 1)]
    assert resumed._rollout_batches_consumed == 5
    assert resumed.dataloader.batch_sampler.set_epoch_calls == [1]


@pytest.mark.parametrize("use_explicit_generator", [False, True])
def test_exact_resume_real_dataloader_preserves_rng_state(
    use_explicit_generator: bool,
) -> None:
    """Iterator reconstruction and skipped fetches must be RNG-neutral."""
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    uninterrupted = _real_rollout_cursor_trainer(
        TrainingProgress(),
        use_explicit_generator=use_explicit_generator,
    )
    explicit_generator = uninterrupted.dataloader.generator
    explicit_initial_state = (
        explicit_generator.get_state().clone() if explicit_generator is not None else None
    )
    for rollout_iteration in range(2):
        uninterrupted.progress = TrainingProgress(rollout_iteration=rollout_iteration)
        for _ in range(2):
            generate_one_rollout_batch(
                uninterrupted,
                reward_buffer=None,
                algorithm_name="DMD2",
            )
    uninterrupted.progress = TrainingProgress(rollout_iteration=2)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    expected_next = generate_one_rollout_batch(
        uninterrupted,
        reward_buffer=None,
        algorithm_name="DMD2",
    )

    resumed = _real_rollout_cursor_trainer(
        TrainingProgress(rollout_iteration=2),
        use_explicit_generator=use_explicit_generator,
    )
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    resumed_next = generate_one_rollout_batch(
        resumed,
        reward_buffer=None,
        algorithm_name="DMD2",
    )

    assert resumed_next == expected_next
    if explicit_initial_state is not None:
        assert torch.equal(uninterrupted.dataloader.generator.get_state(), explicit_initial_state)
        assert torch.equal(resumed.dataloader.generator.get_state(), explicit_initial_state)


def test_finite_multi_source_rollover_uses_actual_loader_length() -> None:
    """A declared-geometry drift must not skip the first batch of the next epoch."""
    trainer = _finite_multi_source_trainer(
        TrainingProgress(),
        accumulation_steps=1,
    )
    assert len(trainer.dataloader) == 3
    for _ in range(len(trainer.dataloader)):
        generate_one_rollout_batch(
            trainer,
            reward_buffer=None,
            algorithm_name="DMD2",
        )
    actual_next = generate_one_rollout_batch(
        trainer,
        reward_buffer=None,
        algorithm_name="DMD2",
    )

    reference = _finite_multi_source_loader()
    reference.set_epoch(1)
    expected_batch = next(iter(reference))
    expected_next = [(expected_batch["__source__"][0], int(expected_batch["value"].item()))]

    assert actual_next == expected_next
    assert trainer.dataloader._scheduler._epoch == 1


@pytest.mark.parametrize("use_explicit_generators", [False, True])
def test_finite_multi_source_lazy_iterators_are_rng_neutral_at_zero_offset(
    use_explicit_generators: bool,
) -> None:
    """The wrapper's lazy child iterators must initialize inside the RNG scope."""
    trainer = _finite_multi_source_trainer(
        TrainingProgress(),
        accumulation_steps=1,
    )
    if use_explicit_generators:
        trainer.dataloader = _finite_multi_source_loader(use_explicit_generators=True)
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    generator_states = {
        name: loader.generator.get_state().clone()
        for name, loader in trainer.dataloader.dataloaders_by_source.items()
        if loader.generator is not None
    }

    generate_one_rollout_batch(
        trainer,
        reward_buffer=None,
        algorithm_name="DMD2",
    )

    assert random.getstate() == python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy_state[1], numpy_state[1])
    assert actual_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    for name, expected_state in generator_states.items():
        actual_generator = trainer.dataloader.dataloaders_by_source[name].generator
        assert torch.equal(actual_generator.get_state(), expected_state)


def test_exact_resume_finite_multi_source_crosses_epoch_with_gas() -> None:
    """Progress times GAS maps through the finite wrapper's actual epoch size."""
    uninterrupted = _finite_multi_source_trainer(
        TrainingProgress(),
        accumulation_steps=2,
    )
    for rollout_iteration in range(2):
        uninterrupted.progress = TrainingProgress(rollout_iteration=rollout_iteration)
        for _ in range(2):
            generate_one_rollout_batch(
                uninterrupted,
                reward_buffer=None,
                algorithm_name="TDM",
            )
    uninterrupted.progress = TrainingProgress(rollout_iteration=2)
    expected_next = generate_one_rollout_batch(
        uninterrupted,
        reward_buffer=None,
        algorithm_name="TDM",
    )

    resumed = _finite_multi_source_trainer(
        TrainingProgress(rollout_iteration=2),
        accumulation_steps=2,
    )
    resumed_next = generate_one_rollout_batch(
        resumed,
        reward_buffer=None,
        algorithm_name="TDM",
    )

    assert resumed_next == expected_next
    assert resumed._rollout_batches_consumed == 5
    assert resumed.dataloader._scheduler._epoch == 1


def test_an_epoch_that_records_nothing_logs_nothing() -> None:
    """An empty log call would still stamp a step and clutter the run's history."""
    logged: list = []
    trainer = SimpleNamespace(
        accelerator=SingleRankAccelerator(),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            get_num_train_timesteps=lambda config: 1,
        ),
        config=SimpleNamespace(),
        epoch=0,
        step=0,
        show_progress_bar=False,
        sampling_context=lambda: _null_context(),
        sample=lambda: ["sample"],
        prepare_feedback=lambda samples: None,
        optimize=lambda microbatches: None,
        log_data=lambda data, step: logged.append((data, step)),
    )

    run_distillation_training_step(trainer)

    assert logged == []
