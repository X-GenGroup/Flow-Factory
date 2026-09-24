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

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    NoisedState,
    StackedSampleBatch,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerOutput
from flow_factory.trainers.abc import _RewardOverlapGroupInfo
from flow_factory.trainers.rl.crd import CRDTrainer
from flow_factory.trainers.rl.dgpo import _SEED_TAG_SHARED_NOISE, DGPOTrainer
from flow_factory.utils.base import create_generator, to_broadcast_tensor
from flow_factory.utils.noise_schedule import flow_match_sigma


class TrainingArgsFake(dict):
    """Mapping/attribute hybrid mirroring ``ArgABC`` unpacking behaviour."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class SchedulerFake:
    """Small scheduler-like object for adapter group construction."""

    def __init__(self) -> None:
        self.noise_level = 0.7
        self.train_timesteps = torch.tensor([0, 1])
        self.seeds: List[int] = []

    def step(self) -> None:
        """Provide scheduler compatibility."""

    def set_seed(self, seed: int) -> None:
        """Record the dispatched seed."""
        self.seeds.append(seed)


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter recording forwards, noising and parameter scopes."""

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
        """Record kwargs plus the active parameter scope and return a velocity."""
        self.forward_kwargs = kwargs
        self.forward_calls.append(kwargs)
        self.forward_scopes.append(self.active_scope)
        return SDESchedulerOutput(velocity=kwargs["latents"] + 3)

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> NoisedState:
        """Count the RNG-owning noising calls."""
        self.noise_calls += 1
        return super().add_forward_process_noise(clean_state, times, generator=generator)

    def apply_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        """Count the RNG-free noise applications."""
        self.apply_calls += 1
        return super().apply_forward_process_noise(clean_state, times, noise)

    def rollout(self, *args: Any, **kwargs: Any) -> None:
        """Record the rollout-mode switch."""
        self.modes.append("rollout")

    def train(self, mode: bool = True) -> None:
        """Record the train-mode switch."""
        self.modes.append("train" if mode else "eval")

    @contextmanager
    def _scope(self, name: str) -> Iterator[None]:
        previous = self.active_scope
        self.active_scope = name
        self.entered_scopes.append(name)
        try:
            yield
        finally:
            self.active_scope = previous

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        """Record the reference-parameter scope without touching real weights."""
        with self._scope("ref"):
            yield

    @contextmanager
    def use_ema_parameters(self) -> Iterator[None]:
        """Record the EMA-parameter scope without touching real weights."""
        with self._scope("ema"):
            yield

    @contextmanager
    def use_named_parameters(self, name: str) -> Iterator[None]:
        """Record a named-snapshot scope without touching real weights."""
        with self._scope(name):
            yield


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring the structured video/audio component contract."""

    trajectory_component_order = ("video", "audio")


class DynamicMaskAdapterFake(AdapterFake):
    """Adapter reducing only the positions the noised state marks active."""

    def _reduce_latent_values(
        self,
        values: Dict[str, torch.Tensor],
        *,
        active_numel: Optional[Dict[str, int]] = None,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Average the elements the per-sample state mask selects."""
        if state is None:
            raise ValueError("expected state context for DynamicMaskAdapterFake reduction")
        mask = state.components["latent"]
        values_flat = values["latent"].reshape(mask.shape[0], -1)
        return (values_flat * mask.to(values_flat.dtype)).sum(dim=1) / mask.sum(dim=1)


def _prepare(adapter: AdapterFake) -> AdapterFake:
    adapter.forward_calls = []
    adapter.forward_scopes = []
    adapter.entered_scopes = []
    adapter.active_scope = "policy"
    adapter.noise_calls = 0
    adapter.apply_calls = 0
    adapter.modes = []
    return adapter


def _adapter(cls: type = AdapterFake) -> AdapterFake:
    adapter = object.__new__(cls)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    return _prepare(adapter)


def _structured_adapter() -> StructuredAdapterFake:
    adapter = object.__new__(StructuredAdapterFake)
    video = SchedulerFake()
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake()}, primary_name="video"
    )
    return _prepare(adapter)


class AcceleratorFake(SimpleNamespace):
    """Single-process accelerator recording reductions and gathers."""

    def __init__(self) -> None:
        super().__init__(device=torch.device("cpu"), num_processes=1)

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Return the local tensor; a no-op in single-process contexts."""
        return tensor

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the local tensor; a no-op in single-process contexts."""
        return tensor


def _dgpo_trainer(
    adapter: BaseAdapter,
    *,
    use_shared_noise: bool = True,
    clip_dsm: bool = False,
    clip_kl: bool = False,
    use_ema_ref: bool = False,
    kl_beta: float = 0.0,
    kl_cfg: float = 1.0,
    dpo_beta: float = 100.0,
    group_size: int = 2,
    epoch: int = 3,
    num_train_timesteps: int = 2,
) -> DGPOTrainer:
    trainer = object.__new__(DGPOTrainer)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(
        seed=7,
        group_size=group_size,
        clip_range=(-0.2, 0.2),
        adv_clip_range=(-5.0, 5.0),
    )
    trainer.autocast = nullcontext
    trainer.accelerator = AcceleratorFake()
    trainer.epoch = epoch
    trainer.step = 0
    trainer.dpo_beta = dpo_beta
    trainer.use_shared_noise = use_shared_noise
    trainer.clip_dsm = clip_dsm
    trainer.clip_kl = clip_kl
    trainer.use_ema_ref = use_ema_ref
    trainer.kl_beta = kl_beta
    trainer.kl_cfg = kl_cfg
    trainer.kl_type = "v-based"
    trainer.num_train_timesteps = num_train_timesteps
    trainer._requires_ema_ref = clip_dsm or clip_kl or use_ema_ref
    return trainer


def _crd_trainer(
    adapter: BaseAdapter,
    *,
    adaptive_logp: bool = False,
    use_old_for_loss: bool = True,
    weight_temp: float = -1.0,
    crd_beta: float = 0.5,
    crd_loss_type: str = "mse",
    kl_beta: float = 0.01,
    kl_cfg: float = 1.0,
    reward_adaptive_kl: bool = False,
    num_train_timesteps: int = 2,
) -> CRDTrainer:
    trainer = object.__new__(CRDTrainer)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(seed=11, adv_clip_range=(-5.0, 5.0))
    trainer.autocast = nullcontext
    trainer.accelerator = AcceleratorFake()
    trainer.epoch = 2
    trainer.step = 0
    trainer.adaptive_logp = adaptive_logp
    trainer.use_old_for_loss = use_old_for_loss
    trainer.weight_temp = weight_temp
    trainer.crd_beta = crd_beta
    trainer.crd_loss_type = crd_loss_type
    trainer.kl_beta = kl_beta
    trainer.kl_cfg = kl_cfg
    trainer.kl_type = "v-based"
    trainer.reward_adaptive_kl = reward_adaptive_kl
    trainer.num_train_timesteps = num_train_timesteps
    trainer.time_sampling_strategy = "uniform"
    trainer.time_shift = 1.0
    trainer.timestep_range = (0.0, 1.0)
    return trainer


def _noised(
    clean: Dict[str, torch.Tensor],
    noise: Dict[str, torch.Tensor],
    sigmas: Dict[str, torch.Tensor],
) -> Tuple[NoisedState, ComponentTimes]:
    """Build a noised state exactly as the adapter hook would, without RNG."""
    times = ComponentTimes(
        timestep={name: value * 1000.0 for name, value in sigmas.items()},
        next_timestep={name: torch.zeros_like(value) for name, value in sigmas.items()},
        sigma=dict(sigmas),
        next_sigma={name: torch.zeros_like(value) for name, value in sigmas.items()},
    )
    state, target = {}, {}
    for name, clean_value in clean.items():
        sigma = to_broadcast_tensor(sigmas[name], clean_value)
        state[name] = (1 - sigma) * clean_value + sigma * noise[name]
        target[name] = noise[name] - clean_value
    return (
        NoisedState(
            state=LatentState(state),
            target_velocity=LatentState(target),
            noise=LatentState(dict(noise)),
        ),
        times,
    )


def _masked_noised(mask: torch.Tensor) -> NoisedState:
    """Noised state whose ``state`` doubles as the dynamic activity mask."""
    zeros = torch.zeros_like(mask)
    return NoisedState(
        state=LatentState({"latent": mask}),
        target_velocity=LatentState({"latent": zeros}),
        noise=LatentState({"latent": zeros}),
    )


def _samples(unique_ids: List[int], values: List[float]) -> List[BaseSample]:
    """Terminal-only rollout samples carrying a group id and an advantage."""
    return [
        BaseSample(
            timesteps=torch.tensor([1000.0, 500.0]),
            all_latents=torch.tensor([[value, value + 1.0, value + 2.0]]),
            latent_index_map=torch.tensor([-1, -1, 0]),
            prompt_embeds=torch.tensor([value]),
            extra_kwargs={"advantage": torch.tensor(value)},
            _unique_id=unique_id,
        )
        for unique_id, value in zip(unique_ids, values)
    ]


def _batch(unique_ids: List[int], values: List[float]) -> StackedSampleBatch:
    return BaseSample.stack(_samples(unique_ids, values))


def _prepped(trainer: DGPOTrainer, unique_ids: List[int], values: List[float]) -> Any:
    """Unpack a single-timestep training batch through the trainer's own helper."""
    return trainer._prep_training_batch(
        {
            "batch": _batch(unique_ids, values),
            "group_info": {
                "local_group_indices": torch.zeros(len(unique_ids), dtype=torch.int64),
                "num_groups": 1,
            },
            "timesteps": torch.tensor([[700.0, 300.0]]),
            "samples_slice": _samples(unique_ids, values),
            "inner_epoch": 0,
        }
    )


def _stub_training_loop(trainer: Any) -> None:
    """Run exactly one ``start()`` iteration without touching real training."""
    iterations = iter([True, False])
    trainer.should_continue_training = lambda: next(iterations)
    trainer.log_args = SimpleNamespace(save_freq=0, save_dir=None, run_name="run")
    trainer.eval_args = SimpleNamespace(eval_freq=0)
    trainer.sample = lambda: []
    trainer.prepare_feedback = lambda samples: None
    trainer.optimize = lambda samples: None
    trainer.sampling_context = nullcontext
    trainer._update_old_model = lambda: None
    trainer._update_sampling_model = lambda: None
    trainer.adapter.ema_step = lambda step: None
    trainer.log_data = lambda _data, step: None


def _legacy_shared_noise(
    trainer: Any,
    clean_latents: torch.Tensor,
    unique_ids: List[int],
    inner_epoch: int,
) -> torch.Tensor:
    """Reproduce the pre-migration ``DGPOTrainer._make_shared_noise``."""
    device, dtype = clean_latents.device, clean_latents.dtype
    per_sample_shape = clean_latents.shape[1:]
    group_cache: Dict[int, torch.Tensor] = {}
    noises: List[torch.Tensor] = []
    for unique_id in unique_ids:
        noise = group_cache.get(unique_id)
        if noise is None:
            generator = create_generator(
                trainer.training_args.seed,
                trainer.epoch,
                inner_epoch,
                int(unique_id),
                _SEED_TAG_SHARED_NOISE,
                device=device,
            )
            noise = randn_tensor(per_sample_shape, generator=generator, device=device, dtype=dtype)
            group_cache[unique_id] = noise
        noises.append(noise)
    return torch.stack(noises, dim=0)


# ============================ DGPO shared noise ============================


def test_dgpo_shared_noise_matches_the_legacy_generator_and_tensor() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    batch = _batch([4, 4, 9], [1.0, 2.0, 3.0])
    samples = _samples([4, 4, 9], [1.0, 2.0, 3.0])
    clean_state = adapter.get_terminal_state(batch)

    noise = trainer._shared_group_noise(clean_state, samples, inner_epoch=1)

    legacy = _legacy_shared_noise(trainer, clean_state.components["latent"], [4, 4, 9], 1)
    assert torch.equal(noise.components["latent"], legacy)


def test_dgpo_shared_noise_is_shared_within_a_group_and_differs_across_groups() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    batch = _batch([4, 4, 9], [1.0, 2.0, 3.0])
    samples = _samples([4, 4, 9], [1.0, 2.0, 3.0])
    clean_state = adapter.get_terminal_state(batch)

    noise = trainer._shared_group_noise(clean_state, samples, inner_epoch=0).components["latent"]

    assert torch.equal(noise[0], noise[1])
    assert not torch.equal(noise[0], noise[2])


def test_dgpo_shared_noise_separates_equal_ids_from_different_sources() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    samples = _samples([4, 4], [1.0, 2.0])
    samples[0].source_id = 0
    samples[1].source_id = 1
    clean_state = adapter.get_terminal_state(_batch([4, 4], [1.0, 2.0]))

    noise = trainer._shared_group_noise(clean_state, samples, inner_epoch=0).components["latent"]

    assert not torch.equal(noise[0], noise[1])


def test_dgpo_shared_noise_keeps_the_legacy_namespace_for_the_primary_component() -> None:
    """A heterogeneous adapter must not shift the single-latent seed namespace."""
    adapter = _structured_adapter()
    trainer = _dgpo_trainer(adapter)
    clean_state = LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)})
    samples = _samples([6, 6], [1.0, 2.0])

    noise = trainer._shared_group_noise(clean_state, samples, inner_epoch=2)

    expected_video = _legacy_shared_noise(trainer, torch.zeros(2, 3, 4), [6, 6], 2)
    audio_generator = create_generator(
        trainer.training_args.seed, trainer.epoch, 2, 6, _SEED_TAG_SHARED_NOISE, 1
    )
    expected_audio = randn_tensor(torch.Size([5]), generator=audio_generator, dtype=torch.float32)
    assert torch.equal(noise.components["video"], expected_video)
    assert torch.equal(noise.components["audio"][0], expected_audio)
    assert torch.equal(noise.components["audio"][0], noise.components["audio"][1])
    assert noise.component_names == ("video", "audio")


def test_dgpo_shared_noise_draws_components_in_the_declared_order() -> None:
    adapter = _structured_adapter()
    trainer = _dgpo_trainer(adapter)
    clean_state = LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)})
    samples = _samples([1, 2], [1.0, 2.0])
    original = DGPOTrainer._draw_group_component_noise
    drawn: List[Tuple[int, str]] = []

    def recording(**kwargs: Any) -> torch.Tensor:
        drawn.append((kwargs["unique_id"], kwargs["component_name"]))
        return original(trainer, **kwargs)

    trainer._draw_group_component_noise = recording
    trainer._shared_group_noise(clean_state, samples, inner_epoch=0)

    assert drawn == [(1, "video"), (1, "audio"), (2, "video"), (2, "audio")]


def test_dgpo_shared_noise_rejects_a_sample_count_mismatch() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    clean_state = LatentState({"latent": torch.zeros(3, 4)})

    with pytest.raises(ValueError, match=r"DGPOTrainer.*3 samples.*received 2"):
        trainer._shared_group_noise(clean_state, _samples([1, 2], [1.0, 2.0]), inner_epoch=0)


def test_dgpo_shared_noise_rejects_a_drawn_component_shape_mismatch() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    clean_state = LatentState({"latent": torch.zeros(2, 4)})
    trainer._draw_group_component_noise = lambda **kwargs: torch.zeros(7)

    with pytest.raises(
        ValueError,
        match=r"DGPOTrainer.*unique_id=5.*component 'latent'.*\(\(4,\).*received \(\(7,\)",
    ):
        trainer._shared_group_noise(clean_state, _samples([5, 5], [1.0, 2.0]), inner_epoch=0)


def test_dgpo_shared_noising_applies_predetermined_noise_without_drawing() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, use_shared_noise=True)
    prepped = _prepped(trainer, [4, 4], [1.0, 2.0])

    torch.manual_seed(70)
    inputs = trainer._build_noised_inputs(prepped, 0)
    after_shared = torch.randn(4)

    torch.manual_seed(70)
    assert torch.equal(after_shared, torch.randn(4))
    assert adapter.noise_calls == 0
    assert adapter.apply_calls == 1
    expected = trainer._shared_group_noise(prepped["clean_state"], prepped["samples_slice"], 0)
    assert torch.equal(inputs.noised.noise.components["latent"], expected.components["latent"])


def test_dgpo_independent_noising_delegates_to_the_adapter_draw_hook() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, use_shared_noise=False)
    prepped = _prepped(trainer, [4, 4], [1.0, 2.0])

    torch.manual_seed(71)
    inputs = trainer._build_noised_inputs(prepped, 0)

    clean = prepped["clean_state"].components["latent"]
    torch.manual_seed(71)
    legacy_noise = randn_tensor(clean.shape, device=clean.device, dtype=clean.dtype)
    assert adapter.noise_calls == 1
    assert torch.equal(inputs.noised.noise.components["latent"], legacy_noise)
    sigma = to_broadcast_tensor(flow_match_sigma(torch.tensor([700.0, 300.0])), clean)
    assert torch.equal(
        inputs.noised.state.components["latent"], (1 - sigma) * clean + sigma * legacy_noise
    )


def test_dgpo_terminal_state_reads_a_sparse_latent_index_map() -> None:
    """The legacy ``all_latents[:, -1]`` slice ignored the stored index map."""
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    prepped = _prepped(trainer, [1, 2], [1.0, 2.0])

    assert torch.equal(
        prepped["clean_state"].components["latent"], prepped["batch"]["all_latents"][:, 0]
    )
    assert torch.equal(prepped["adv"], torch.tensor([1.0, 2.0]))


# ============================ DGPO losses ============================


def test_dgpo_dsm_loss_matches_the_legacy_single_component_formula() -> None:
    torch.manual_seed(80)
    target = torch.randn(2, 3, 4)
    prediction = torch.randn(2, 3, 4)
    noised, _ = _noised(
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": target},
        {"latent": torch.zeros(2)},
    )
    trainer = _dgpo_trainer(_adapter())

    dsm = trainer._compute_dsm_loss(
        LatentState({"latent": target}), LatentState({"latent": prediction}), noised
    )

    legacy = (target - prediction).square().reshape(2, -1).mean(dim=1)
    assert torch.equal(dsm, legacy)


def test_dgpo_dsm_loss_reduces_two_components_by_active_degrees_of_freedom() -> None:
    torch.manual_seed(81)
    target = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    prediction = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    noised, _ = _noised(
        {name: torch.zeros_like(value) for name, value in target.items()},
        target,
        {"video": torch.zeros(2), "audio": torch.zeros(2)},
    )
    trainer = _dgpo_trainer(_structured_adapter())

    dsm = trainer._compute_dsm_loss(LatentState(target), LatentState(prediction), noised)

    video = (target["video"] - prediction["video"]).flatten(1).pow(2).sum(dim=1)
    audio = (target["audio"] - prediction["audio"]).flatten(1).pow(2).sum(dim=1)
    assert torch.equal(dsm, (video + audio) / 17)


def test_dgpo_dsm_loss_passes_the_noised_state_to_the_global_reducer() -> None:
    torch.manual_seed(82)
    target, prediction = torch.randn(2, 4), torch.randn(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    trainer = _dgpo_trainer(_adapter(DynamicMaskAdapterFake))

    dsm = trainer._compute_dsm_loss(
        LatentState({"latent": target}), LatentState({"latent": prediction}), _masked_noised(mask)
    )

    squared = (target - prediction) ** 2
    assert torch.equal(dsm, (squared * mask).sum(dim=1) / mask.sum(dim=1))


def test_dgpo_group_loss_matches_the_legacy_sigmoid_preference() -> None:
    torch.manual_seed(83)
    target = torch.randn(4, 3, 4)
    reference = torch.randn(4, 3, 4)
    dsm_loss = torch.rand(4).requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    group_info = {
        "local_group_indices": torch.tensor([0, 0, 1, 1]),
        "num_groups": 2,
    }
    noised, _ = _noised(
        {"latent": torch.zeros(4, 3, 4)}, {"latent": target}, {"latent": torch.zeros(4)}
    )
    trainer = _dgpo_trainer(_adapter(), dpo_beta=100.0, group_size=2)

    loss = trainer._compute_group_dgpo_loss(
        ref_v=LatentState({"latent": reference}),
        target_v=LatentState({"latent": target}),
        advantages=advantages,
        group_info=group_info,
        dsm_loss=dsm_loss,
        noised=noised,
    )

    ref_dsm = (target - reference).square().reshape(4, -1).mean(dim=1)
    per_sample = advantages * 100.0 * (dsm_loss.detach() - ref_dsm) / 2
    sums = torch.zeros(2).scatter_add_(0, group_info["local_group_indices"], per_sample)
    weights = torch.sigmoid(sums)[group_info["local_group_indices"]]
    assert torch.equal(loss, (weights * advantages * dsm_loss).mean())


def test_dgpo_builds_group_indices_from_a_packed_global_microbatch() -> None:
    trainer = _dgpo_trainer(_adapter(), group_size=2)
    trainer.accelerator.num_processes = 4
    trainer.accelerator.gather = lambda _local: torch.tensor([[-1, 7], [-1, 7], [-1, 9], [-1, 9]])

    group_info = trainer._precompute_group_info([SimpleNamespace(unique_id=9)])

    assert group_info["num_groups"] == 2
    torch.testing.assert_close(group_info["local_group_indices"], torch.tensor([1]))


def test_dgpo_equal_share_group_indices_do_not_gather() -> None:
    trainer = _dgpo_trainer(_adapter(), group_size=4)
    trainer.accelerator.num_processes = 2
    trainer.accelerator.gather = lambda _local: (_ for _ in ()).throw(
        AssertionError("equal-share group metadata must remain rank-local")
    )

    group_info = trainer._precompute_group_info(
        [SimpleNamespace(unique_id=7), SimpleNamespace(unique_id=7)]
    )

    assert group_info["num_groups"] == 1
    torch.testing.assert_close(group_info["local_group_indices"], torch.tensor([0, 0]))


def test_dgpo_overlap_reuses_cached_group_indices_without_gathering() -> None:
    trainer = _dgpo_trainer(_adapter(), group_size=2)
    trainer.accelerator.gather = lambda _local: (_ for _ in ()).throw(
        AssertionError("cached overlap group metadata must avoid an extra gather")
    )
    cached = _RewardOverlapGroupInfo(
        local_group_identities=torch.tensor([[-1, 9]]),
        local_group_indices=torch.tensor([1]),
        num_groups=2,
    )
    trainer._dgpo_reward_overlap_group_infos = [cached]

    group_info = trainer._precompute_group_info([SimpleNamespace(unique_id=9)])

    assert group_info["num_groups"] == 2
    torch.testing.assert_close(group_info["local_group_indices"], torch.tensor([1]))
    assert trainer._dgpo_reward_overlap_group_infos == []


def test_crd_overlap_attaches_ready_advantages_to_precomputed_batches() -> None:
    trainer = _crd_trainer(_adapter())
    samples = _samples([7, 7], [0.25, 0.75])
    samples[0].extra_kwargs["advantage"] = torch.tensor(-1.0)
    samples[1].extra_kwargs["advantage"] = torch.tensor(1.0)
    prepared = SimpleNamespace(batch=BaseSample.stack(samples), steps=[])
    prepared.batch.pop("advantage")
    observed: List[torch.Tensor] = []
    trainer.optimize = lambda _samples: observed.append(
        trainer._crd_overlap_precomputed_batches[0].batch["advantage"].clone()
    )

    trainer._optimize_reward_overlap_tile(
        SimpleNamespace(tile_id=0),
        samples,
        {0: [prepared]},
    )

    torch.testing.assert_close(observed[0], torch.tensor([-1.0, 1.0]))
    assert not hasattr(trainer, "_crd_overlap_precomputed_batches")


def test_dgpo_dsm_clipping_matches_the_legacy_ratio_rule() -> None:
    torch.manual_seed(84)
    target = torch.randn(2, 3, 4)
    old_velocity = torch.randn(2, 3, 4)
    dsm_loss = torch.rand(2)
    advantages = torch.tensor([1.0, -1.0])
    noised, _ = _noised(
        {"latent": torch.zeros(2, 3, 4)}, {"latent": target}, {"latent": torch.zeros(2)}
    )
    trainer = _dgpo_trainer(_adapter(), clip_dsm=True)
    loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

    should_clip, clipped = trainer._maybe_clip_dsm(
        dsm_loss=dsm_loss,
        old_v=LatentState({"latent": old_velocity}),
        target_v=LatentState({"latent": target}),
        adv=advantages,
        loss_info=loss_info,
        noised=noised,
    )

    old_dsm = (target - old_velocity).square().reshape(2, -1).mean(dim=1)
    ratio = torch.exp(-dsm_loss.detach() + old_dsm)
    expected = torch.where(advantages > 0, ratio > 1.2, ratio < 0.8)
    assert torch.equal(should_clip, expected)
    assert torch.equal(clipped, torch.where(expected, dsm_loss.detach(), dsm_loss))
    assert torch.equal(loss_info["clip_ratio"][0], expected.float().mean())


def test_dgpo_clipping_is_skipped_without_an_old_policy_velocity() -> None:
    trainer = _dgpo_trainer(_adapter())
    dsm_loss = torch.rand(2)
    noised, _ = _noised(
        {"latent": torch.zeros(2, 4)}, {"latent": torch.zeros(2, 4)}, {"latent": torch.zeros(2)}
    )

    should_clip, clipped = trainer._maybe_clip_dsm(
        dsm_loss=dsm_loss,
        old_v=None,
        target_v=LatentState({"latent": torch.zeros(2, 4)}),
        adv=torch.ones(2),
        loss_info=defaultdict(list),
        noised=noised,
    )

    assert should_clip is None
    assert clipped is dsm_loss


def test_dgpo_velocity_kl_matches_the_legacy_formula() -> None:
    torch.manual_seed(85)
    velocity, ref_velocity = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    noised, _ = _noised(
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2)},
    )
    trainer = _dgpo_trainer(_adapter())

    kl_div = trainer._velocity_kl(
        LatentState({"latent": velocity}), LatentState({"latent": ref_velocity}), noised
    )

    legacy = (velocity - ref_velocity).square().reshape(2, -1).mean(dim=1)
    assert torch.equal(kl_div, legacy)


def test_dgpo_velocity_kl_passes_the_noised_state_to_the_global_reducer() -> None:
    torch.manual_seed(86)
    velocity, ref_velocity = torch.randn(2, 4), torch.randn(2, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    trainer = _dgpo_trainer(_adapter(DynamicMaskAdapterFake))

    kl_div = trainer._velocity_kl(
        LatentState({"latent": velocity}),
        LatentState({"latent": ref_velocity}),
        _masked_noised(mask),
    )

    squared = (velocity - ref_velocity) ** 2
    assert torch.equal(kl_div, (squared * mask).sum(dim=1) / mask.sum(dim=1))


# ============================ DGPO forward routing ============================


def _dgpo_forward_inputs(
    trainer: DGPOTrainer, adapter: AdapterFake
) -> Tuple[StackedSampleBatch, ComponentTimes, NoisedState]:
    batch = _batch([1, 1], [1.0, 2.0])
    clean_state = adapter.get_terminal_state(batch)
    times = adapter.build_training_component_times(torch.tensor([700.0, 300.0]), batch=batch)
    noised = adapter.apply_forward_process_noise(
        clean_state,
        times,
        LatentState({"latent": torch.zeros_like(clean_state.components["latent"])}),
    )
    return batch, times, noised


def test_dgpo_forward_velocities_route_parameter_contexts_and_cfg() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, clip_dsm=True, use_ema_ref=False, kl_beta=0.1, kl_cfg=4.5)
    batch, times, noised = _dgpo_forward_inputs(trainer, adapter)

    velocities = trainer._forward_velocities(batch, times, noised)

    assert adapter.forward_scopes == ["ema_ref", "policy", "ref"]
    assert [call["guidance_scale"] for call in adapter.forward_calls] == [1.0, 1.0, 4.5]
    assert velocities["old_v"].components["latent"].requires_grad is False
    assert velocities["ref_dgpo_v"] is velocities["ref_v"]


def test_dgpo_forward_velocities_select_the_ema_reference() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, use_ema_ref=True, kl_beta=0.0)
    batch, times, noised = _dgpo_forward_inputs(trainer, adapter)

    velocities = trainer._forward_velocities(batch, times, noised)

    assert adapter.forward_scopes == ["ema_ref", "policy"]
    assert velocities["ref_v"] is None
    assert velocities["ref_dgpo_v"] is velocities["old_v"]


def test_dgpo_forward_velocities_skip_the_old_policy_when_unused() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, use_ema_ref=False, kl_beta=0.0, kl_cfg=0.5)

    batch, times, noised = _dgpo_forward_inputs(trainer, adapter)
    velocities = trainer._forward_velocities(batch, times, noised)

    assert adapter.forward_scopes == ["policy", "ref"]
    assert velocities["old_v"] is None
    assert [call["guidance_scale"] for call in adapter.forward_calls] == [1.0, 1.0]


def test_dgpo_forward_velocities_reject_a_malformed_velocity() -> None:
    class BroadcastAdapterFake(AdapterFake):
        def forward(self, **kwargs: Any) -> SDESchedulerOutput:
            """Return a velocity that would broadcast against the state."""
            self.forward_calls.append(kwargs)
            self.forward_scopes.append(self.active_scope)
            return SDESchedulerOutput(velocity=kwargs["latents"].mean(dim=1, keepdim=True))

    adapter = _adapter(BroadcastAdapterFake)
    trainer = _dgpo_trainer(adapter, use_ema_ref=False, kl_beta=0.0)
    batch, times, noised = _dgpo_forward_inputs(trainer, adapter)

    with pytest.raises(ValueError, match=r"policy velocity component 'latent'.*DGPOTrainer"):
        trainer._forward_velocities(batch, times, noised)


def test_dgpo_start_seeds_every_scheduler_component() -> None:
    """The legacy ``adapter.scheduler.set_seed`` reached the primary component only."""
    adapter = _structured_adapter()
    trainer = _dgpo_trainer(adapter, epoch=5)
    _stub_training_loop(trainer)

    trainer.start()

    assert adapter.scheduler_group["video"].seeds == [12]
    assert adapter.scheduler_group["audio"].seeds == [12]


# ============================ CRD ============================


def _crd_batch() -> StackedSampleBatch:
    return _batch([1, 2], [1.0, 2.0])


def test_crd_pass_two_reuses_pass_one_noise_without_drawing() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter, num_train_timesteps=3)
    batch = _crd_batch()

    torch.manual_seed(90)
    prepared = trainer._precompute_old_velocities(batch)
    after_precompute = torch.get_rng_state()

    assert adapter.noise_calls == 3
    assert len(prepared.steps) == 3
    clean_state = adapter.get_terminal_state(batch)
    for step, call in zip(prepared.steps, adapter.forward_calls):
        noised = trainer._rebuild_noised_state(clean_state, step)
        assert noised.noise is step.noise
        assert torch.equal(noised.state.components["latent"], call["latents"])
        assert step.old_velocity.components["latent"].requires_grad is False
    assert adapter.noise_calls == 3
    assert torch.equal(torch.get_rng_state(), after_precompute)


def test_crd_pass_one_uses_the_old_snapshot_or_reference_parameters() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter, use_old_for_loss=True, num_train_timesteps=2)
    trainer._precompute_old_velocities(_crd_batch())
    assert adapter.forward_scopes == [CRDTrainer._OLD_PARAMS_NAME] * 2

    reference_adapter = _adapter()
    reference_trainer = _crd_trainer(
        reference_adapter, use_old_for_loss=False, num_train_timesteps=2
    )
    reference_trainer._precompute_old_velocities(_crd_batch())
    assert reference_adapter.forward_scopes == ["ref"] * 2


def _legacy_crd_reward(
    velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    target: torch.Tensor,
    adaptive: bool,
) -> torch.Tensor:
    """Reproduce the pre-migration CRD implicit reward."""
    if adaptive:
        with torch.no_grad():
            weight_theta = (
                torch.abs(velocity.double() - target.double())
                .mean(dim=tuple(range(1, velocity.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
            weight_old = (
                torch.abs(old_velocity.double() - target.double())
                .mean(dim=tuple(range(1, old_velocity.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        reward = -(
            (velocity - target) ** 2 / weight_theta - (old_velocity - target) ** 2 / weight_old
        )
    else:
        reward = -((velocity - target) ** 2 - (old_velocity - target) ** 2)
    return reward.mean(dim=tuple(range(1, reward.ndim)))


@pytest.mark.parametrize("adaptive", [False, True])
def test_crd_implicit_reward_matches_the_legacy_single_component_formula(
    adaptive: bool,
) -> None:
    torch.manual_seed(91)
    clean = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    velocity = torch.randn(2, 3, 4)
    old_velocity = torch.randn(2, 3, 4)
    noised, _ = _noised({"latent": clean}, {"latent": noise}, {"latent": torch.tensor([0.6, 0.2])})
    trainer = _crd_trainer(_adapter(), adaptive_logp=adaptive)

    reward = trainer._implicit_reward(
        LatentState({"latent": velocity}), LatentState({"latent": old_velocity}), noised
    )

    legacy = _legacy_crd_reward(velocity, old_velocity, noise - clean, adaptive)
    assert reward.dtype is legacy.dtype
    assert torch.equal(reward, legacy)


def test_crd_adaptive_reward_keeps_the_normalization_out_of_the_gradient_graph() -> None:
    torch.manual_seed(92)
    clean = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    base_velocity = torch.randn(2, 3, 4)
    old_velocity = torch.randn(2, 3, 4)
    noised, _ = _noised({"latent": clean}, {"latent": noise}, {"latent": torch.tensor([0.6, 0.2])})
    trainer = _crd_trainer(_adapter(), adaptive_logp=True)

    velocity = base_velocity.clone().requires_grad_(True)
    trainer._implicit_reward(
        LatentState({"latent": velocity}), LatentState({"latent": old_velocity}), noised
    ).sum().backward()

    legacy_velocity = base_velocity.clone().requires_grad_(True)
    _legacy_crd_reward(legacy_velocity, old_velocity, noise - clean, True).sum().backward()
    assert torch.equal(velocity.grad, legacy_velocity.grad)


def test_crd_adaptive_reward_clamps_the_normalization_floor() -> None:
    """A perfect prediction divides by a zero deviation without the ``1e-5`` clamp."""
    clean = torch.zeros(2, 4)
    noised, _ = _noised({"latent": clean}, {"latent": clean}, {"latent": torch.tensor([0.5, 0.5])})
    trainer = _crd_trainer(_adapter(), adaptive_logp=True)

    reward = trainer._implicit_reward(
        LatentState({"latent": torch.zeros(2, 4)}),
        LatentState({"latent": torch.zeros(2, 4)}),
        noised,
    )

    assert torch.equal(reward, torch.zeros(2, dtype=torch.float64))


def test_crd_adaptive_reward_normalizes_each_component_before_the_global_reduction() -> None:
    torch.manual_seed(93)
    clean = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    noise = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    velocity = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    old_velocity = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    noised, _ = _noised(
        clean, noise, {"video": torch.tensor([0.6, 0.2]), "audio": torch.tensor([0.4, 0.9])}
    )
    trainer = _crd_trainer(_structured_adapter(), adaptive_logp=True)

    reward = trainer._implicit_reward(LatentState(velocity), LatentState(old_velocity), noised)

    total = torch.zeros(2, dtype=torch.float64)
    for name in ("video", "audio"):
        target = noise[name] - clean[name]
        with torch.no_grad():
            weight_theta = (
                torch.abs(velocity[name].double() - target.double())
                .mean(dim=tuple(range(1, target.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
            weight_old = (
                torch.abs(old_velocity[name].double() - target.double())
                .mean(dim=tuple(range(1, target.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        elements = -(
            (velocity[name] - target) ** 2 / weight_theta
            - (old_velocity[name] - target) ** 2 / weight_old
        )
        total = total + elements.flatten(1).sum(dim=1)
    assert torch.equal(reward, total / 17)


def test_crd_implicit_reward_passes_the_noised_state_to_the_global_reducer() -> None:
    torch.manual_seed(94)
    velocity, old_velocity = torch.randn(2, 4), torch.randn(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    trainer = _crd_trainer(_adapter(DynamicMaskAdapterFake), adaptive_logp=False)

    reward = trainer._implicit_reward(
        LatentState({"latent": velocity}),
        LatentState({"latent": old_velocity}),
        _masked_noised(mask),
    )

    elements = -(velocity**2 - old_velocity**2)
    assert torch.equal(reward, (elements * mask).sum(dim=1) / mask.sum(dim=1))


def test_crd_velocity_kl_matches_the_legacy_formula() -> None:
    torch.manual_seed(95)
    velocity, ref_velocity = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    noised, _ = _noised(
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2)},
    )
    trainer = _crd_trainer(_adapter())

    kl_div = trainer._velocity_kl(
        LatentState({"latent": velocity}), LatentState({"latent": ref_velocity}), noised
    )

    legacy = ((velocity - ref_velocity) ** 2).mean(dim=tuple(range(1, velocity.ndim)))
    assert torch.equal(kl_div, legacy)


@pytest.mark.parametrize("reward_adaptive_kl", [False, True])
def test_crd_kl_loss_matches_the_legacy_scaling(reward_adaptive_kl: bool) -> None:
    torch.manual_seed(96)
    kl_div = torch.rand(2)
    reward = torch.rand(2)
    trainer = _crd_trainer(_adapter(), kl_beta=0.02, reward_adaptive_kl=reward_adaptive_kl)

    kl_loss = trainer._kl_loss(kl_div, reward)

    if reward_adaptive_kl:
        min_coef = 1e-4 / max(0.02, 1e-8)
        legacy = 0.02 * torch.mean((min_coef + reward * (1 - min_coef)) * kl_div)
    else:
        legacy = 0.02 * kl_div.mean()
    assert torch.equal(kl_loss, legacy)


def _legacy_crd_loss(
    trainer: CRDTrainer,
    adv_cur: torch.Tensor,
    adv_cur_rank: torch.Tensor,
    r_theta_gathered: torch.Tensor,
    r_theta_local: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the pre-migration CRD centering loss."""
    device = adv_cur.device
    weight_temp = torch.inf if trainer.weight_temp < 0 else trainer.weight_temp

    def _centered(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adv_avg = (adv_cur * weights).sum(dim=0, keepdim=True)
        reward_avg = (r_theta_gathered * weights).sum(dim=0, keepdim=True)
        return adv_cur_rank - adv_avg, r_theta_local - reward_avg.detach()

    def _term(centered_adv: torch.Tensor, centered_reward: torch.Tensor) -> torch.Tensor:
        if trainer.crd_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(
                trainer.crd_beta * centered_reward,
                torch.sigmoid(centered_adv.detach()),
                reduction="mean",
            )
        return ((trainer.crd_beta * centered_reward - centered_adv) ** 2).mean()

    if weight_temp == torch.inf:
        return _term(*_centered(torch.softmax(adv_cur / weight_temp, dim=0)))

    def _hard(mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.ones_like(adv_cur) / adv_cur.shape[0]
        return torch.softmax(adv_cur.where(mask, torch.tensor(float("-inf"), device=device)), dim=0)

    if weight_temp == 0:
        positive = _hard(adv_cur > 0.0)
        negative = _hard(adv_cur < 0.0)
    else:
        positive = torch.softmax(adv_cur / weight_temp, dim=0)
        negative = torch.softmax(-adv_cur / weight_temp, dim=0)
    return 0.5 * _term(*_centered(positive)) + 0.5 * _term(*_centered(negative))


@pytest.mark.parametrize("weight_temp", [-1.0, 0.0, 0.5])
@pytest.mark.parametrize("crd_loss_type", ["mse", "bce"])
def test_crd_centering_loss_matches_every_weight_temp_branch(
    weight_temp: float, crd_loss_type: str
) -> None:
    torch.manual_seed(97)
    adv_cur = torch.tensor([0.9, 0.1, 0.4, 0.6])
    adv_cur_rank = adv_cur[:2]
    r_theta_gathered = torch.rand(4)
    r_theta_local = r_theta_gathered[:2]
    trainer = _crd_trainer(_adapter(), weight_temp=weight_temp, crd_loss_type=crd_loss_type)

    loss = trainer._compute_crd_loss(
        adv_cur=adv_cur,
        adv_cur_rank=adv_cur_rank,
        r_theta_gathered=r_theta_gathered,
        r_theta_local=r_theta_local,
    )

    assert torch.equal(
        loss, _legacy_crd_loss(trainer, adv_cur, adv_cur_rank, r_theta_gathered, r_theta_local)
    )


def test_crd_hard_selection_falls_back_to_uniform_weights_without_a_direction() -> None:
    trainer = _crd_trainer(_adapter(), weight_temp=0.0)
    adv_cur = torch.tensor([0.5, 0.5, 0.5, 0.5])

    loss = trainer._compute_crd_loss(
        adv_cur=adv_cur,
        adv_cur_rank=adv_cur[:2],
        r_theta_gathered=torch.rand(4),
        r_theta_local=torch.rand(2),
    )

    assert torch.isfinite(loss)


def test_crd_terminal_state_reads_a_sparse_latent_index_map() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter, num_train_timesteps=1)
    batch = _crd_batch()

    prepared = trainer._precompute_old_velocities(batch)

    clean_state = adapter.get_terminal_state(batch)
    assert torch.equal(clean_state.components["latent"], batch["all_latents"][:, 0])
    assert prepared.steps[0].noise.components["latent"].shape == (2, 3)


def test_crd_start_seeds_every_scheduler_component() -> None:
    """The legacy ``adapter.scheduler.set_seed`` reached the primary component only."""
    adapter = _structured_adapter()
    trainer = _crd_trainer(adapter)
    trainer.epoch = 4
    _stub_training_loop(trainer)

    trainer.start()

    assert adapter.scheduler_group["video"].seeds == [15]
    assert adapter.scheduler_group["audio"].seeds == [15]
