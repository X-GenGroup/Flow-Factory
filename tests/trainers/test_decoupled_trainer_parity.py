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

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerOutput
from flow_factory.trainers.forward_process import (
    forward_velocity_state,
    require_velocity_state,
    state_batch_size,
    training_forward_kwargs,
)
from flow_factory.trainers.rl.awm import AWMTrainer
from flow_factory.trainers.rl.dpo import DPOTrainer
from flow_factory.trainers.rl.nft import DiffusionNFTTrainer
from flow_factory.utils.base import to_broadcast_tensor
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

    def step(self) -> None:
        """Provide scheduler compatibility."""

    def set_seed(self, seed: int) -> None:
        """Provide scheduler compatibility."""


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter recording forward and noising calls."""

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
        """Record kwargs and return a deterministic scheduler output."""
        self.forward_kwargs = kwargs
        self.forward_calls.append(kwargs)
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

    def rollout(self, *args: Any, **kwargs: Any) -> None:
        """Record the rollout-mode switch."""
        self.modes.append("rollout")

    def train(self, mode: bool = True) -> None:
        """Record the train-mode switch."""
        self.modes.append("train" if mode else "eval")

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        """Record reference-parameter scope without touching real weights."""
        self.ref_scope_entered = True
        yield

    @contextmanager
    def use_ema_parameters(self) -> Iterator[None]:
        """Record EMA-parameter scope without touching real weights."""
        self.ema_scope_entered = True
        yield


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring the structured video/audio component contract."""

    trajectory_component_order = ("video", "audio")


class DataWardStructuredAdapterFake(StructuredAdapterFake):
    """Structured adapter using MiniMax H3's data-ward velocity convention."""

    flow_velocity_direction = "data"


class BroadcastVelocityAdapterFake(AdapterFake):
    """Adapter returning a per-sample velocity that broadcasts over the state."""

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Return a velocity reduced to one element per sample."""
        self.forward_kwargs = kwargs
        self.forward_calls.append(kwargs)
        latents = kwargs["latents"]
        return SDESchedulerOutput(velocity=latents.mean(dim=1, keepdim=True))


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
        return (values_flat * mask).sum(dim=1) / mask.sum(dim=1)


def _prepare(adapter: AdapterFake) -> AdapterFake:
    adapter.forward_calls = []
    adapter.noise_calls = 0
    adapter.modes = []
    return adapter


def _adapter(cls: type = AdapterFake) -> AdapterFake:
    adapter = object.__new__(cls)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    return _prepare(adapter)


def _structured_adapter(cls: type = StructuredAdapterFake) -> StructuredAdapterFake:
    adapter = object.__new__(cls)
    video = SchedulerFake()
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake()}, primary_name="video"
    )
    return _prepare(adapter)


def _trainer(cls: type, adapter: BaseAdapter, **training_args: Any) -> Any:
    trainer = object.__new__(cls)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(training_args)
    trainer.autocast = nullcontext
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    return trainer


def _nft_trainer(
    adapter: BaseAdapter,
    *,
    nft_beta: float = 0.5,
    num_train_timesteps: int = 2,
) -> DiffusionNFTTrainer:
    trainer = _trainer(
        DiffusionNFTTrainer,
        adapter,
        kl_beta=0.0,
        adv_clip_range=(-5.0, 5.0),
        seed=0,
        use_shared_noise=False,
    )
    trainer.epoch = 0
    trainer.nft_beta = nft_beta
    trainer.kl_type = "v-based"
    trainer.off_policy = False
    trainer.num_train_timesteps = num_train_timesteps
    trainer.time_sampling_strategy = "uniform"
    trainer.time_shift = 1.0
    trainer.timestep_range = (0.0, 1.0)
    return trainer


def _awm_trainer(
    adapter: BaseAdapter,
    *,
    weighting: str = "Uniform",
    num_train_timesteps: int = 2,
) -> AWMTrainer:
    trainer = _trainer(
        AWMTrainer, adapter, kl_beta=0.0, ema_kl_beta=0.0, seed=0, use_shared_noise=False
    )
    trainer.epoch = 0
    trainer.weighting = weighting
    trainer.ghuber_power = 0.25
    trainer.kl_beta = 0.0
    trainer.ema_kl_beta = 0.0
    trainer.off_policy = False
    trainer.num_train_timesteps = num_train_timesteps
    trainer.time_sampling_strategy = "uniform"
    trainer.time_shift = 1.0
    trainer.timestep_range = (0.0, 1.0)
    return trainer


def _dpo_trainer(adapter: BaseAdapter, *, beta: float = 2000.0) -> DPOTrainer:
    trainer = _trainer(DPOTrainer, adapter, beta=beta, seed=0)
    trainer.num_train_timesteps = 2
    return trainer


def _latent_times(primary: torch.Tensor) -> ComponentTimes:
    sigma = flow_match_sigma(primary)
    return ComponentTimes(
        timestep={"latent": primary},
        next_timestep={"latent": torch.zeros_like(primary)},
        sigma={"latent": sigma},
        next_sigma={"latent": torch.zeros_like(sigma)},
    )


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


def _legacy_batch() -> Any:
    """Legacy decoupled rollout that stored only the terminal latent."""
    return BaseSample.stack(
        [
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0]),
                all_latents=torch.tensor([[1.0, 2.0]]),
                latent_index_map=torch.tensor([-1, -1, 0]),
                height=64,
                prompt_embeds=torch.tensor([4.0]),
                sampling_sample_id=1,
            ),
            BaseSample(
                timesteps=torch.tensor([1000.0, 500.0]),
                all_latents=torch.tensor([[10.0, 20.0]]),
                latent_index_map=torch.tensor([-1, -1, 0]),
                height=64,
                prompt_embeds=torch.tensor([5.0]),
                sampling_sample_id=2,
            ),
        ]
    )


# ============================ Shared forward-process helpers ============================


def test_training_forward_kwargs_keep_batch_precedence() -> None:
    trainer = _trainer(DiffusionNFTTrainer, _adapter(), height=512, guidance_scale=3.5)

    kwargs = training_forward_kwargs(trainer, _legacy_batch())

    assert "height" not in kwargs
    assert kwargs == {"guidance_scale": 3.5}


def test_forward_velocity_state_requests_only_velocity_without_log_probs() -> None:
    adapter = _adapter()
    trainer = _trainer(DiffusionNFTTrainer, adapter, guidance_scale=3.5)
    batch = _legacy_batch()
    state = adapter.get_terminal_state(batch)
    times = _latent_times(torch.tensor([700.0, 300.0]))

    velocity = forward_velocity_state(trainer, batch, state, times, source="policy")

    assert adapter.forward_kwargs["latents"] is state.components["latent"]
    assert adapter.forward_kwargs["compute_log_prob"] is False
    assert adapter.forward_kwargs["return_kwargs"] == ("velocity",)
    assert adapter.forward_kwargs["noise_level"] == 0.0
    assert adapter.forward_kwargs["guidance_scale"] == 3.5
    assert torch.equal(adapter.forward_kwargs["t"], times.timestep["latent"])
    assert torch.equal(adapter.forward_kwargs["t_next"], torch.zeros(2))
    assert torch.equal(velocity.components["latent"], state.components["latent"] + 3)


def test_velocity_state_validation_rejects_a_missing_velocity() -> None:
    trainer = _trainer(AWMTrainer, _adapter())

    with pytest.raises(
        ValueError,
        match=r"policy velocity.*AWMTrainer.*\('latent',\).*received None.*return_fields",
    ):
        require_velocity_state(
            trainer, MultiModalStepOutput(), "policy", LatentState({"latent": torch.zeros(2, 4)})
        )


def test_velocity_state_validation_requires_the_declared_component_order() -> None:
    trainer = _trainer(AWMTrainer, _structured_adapter())
    output = MultiModalStepOutput(velocity=LatentState({"audio": torch.zeros(2, 5)}))
    expected = LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)})

    with pytest.raises(
        ValueError,
        match=r"reference velocity.*AWMTrainer.*\('video', 'audio'\).*\('audio',\)",
    ):
        require_velocity_state(trainer, output, "reference", expected)


def test_velocity_state_validation_requires_the_state_batch_size() -> None:
    trainer = _trainer(AWMTrainer, _adapter())
    output = MultiModalStepOutput(velocity=LatentState({"latent": torch.zeros(3, 4)}))

    with pytest.raises(
        ValueError,
        match=r"policy velocity component 'latent'.*\(2, 4\).*received shape \(3, 4\)",
    ):
        require_velocity_state(
            trainer, output, "policy", LatentState({"latent": torch.zeros(2, 4)})
        )


def test_velocity_state_validation_rejects_a_broadcastable_spatial_shape() -> None:
    """A (B, 1) velocity broadcasts silently against a (B, 4) state, so reject it."""
    trainer = _trainer(AWMTrainer, _adapter())
    output = MultiModalStepOutput(velocity=LatentState({"latent": torch.zeros(2, 1)}))

    with pytest.raises(
        ValueError,
        match=r"policy velocity component 'latent'.*\(2, 4\).*received shape \(2, 1\)",
    ):
        require_velocity_state(
            trainer, output, "policy", LatentState({"latent": torch.zeros(2, 4)})
        )


def test_velocity_state_validation_rejects_a_foreign_device() -> None:
    trainer = _trainer(AWMTrainer, _adapter())
    output = MultiModalStepOutput(
        velocity=LatentState({"latent": torch.zeros(2, 4, device="meta")})
    )

    with pytest.raises(ValueError, match=r"policy velocity component 'latent'.*cpu.*meta"):
        require_velocity_state(
            trainer, output, "policy", LatentState({"latent": torch.zeros(2, 4)})
        )


def test_velocity_state_validation_rejects_a_non_floating_point_velocity() -> None:
    trainer = _trainer(AWMTrainer, _adapter())
    output = MultiModalStepOutput(
        velocity=LatentState({"latent": torch.zeros(2, 4, dtype=torch.int64)})
    )

    with pytest.raises(
        ValueError, match=r"policy velocity component 'latent'.*floating point.*torch.int64"
    ):
        require_velocity_state(
            trainer, output, "policy", LatentState({"latent": torch.zeros(2, 4)})
        )


def test_velocity_state_validation_rejects_components_with_divergent_dtypes() -> None:
    """Component reductions require one shared dtype across the whole velocity."""
    trainer = _trainer(AWMTrainer, _structured_adapter())
    output = MultiModalStepOutput(
        velocity=LatentState(
            {"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5, dtype=torch.float64)}
        )
    )
    expected = LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)})

    with pytest.raises(
        ValueError,
        match=r"policy velocity component 'audio'.*'video'.*torch.float32.*torch.float64",
    ):
        require_velocity_state(trainer, output, "policy", expected)


def test_velocity_state_validation_accepts_the_autocast_compute_dtype() -> None:
    """``latent_storage_dtype`` (fp16 by default) legitimately differs from bf16 compute."""
    trainer = _trainer(AWMTrainer, _adapter())
    stored = LatentState({"latent": torch.zeros(2, 4, dtype=torch.float16)})
    computed = LatentState({"latent": torch.zeros(2, 4, dtype=torch.bfloat16)})

    velocity = require_velocity_state(
        trainer, MultiModalStepOutput(velocity=computed), "policy", stored
    )

    assert velocity is computed


def test_forward_velocity_state_validates_against_the_forwarded_state() -> None:
    adapter = _adapter(BroadcastVelocityAdapterFake)
    trainer = _trainer(DiffusionNFTTrainer, adapter)
    batch = _legacy_batch()
    state = adapter.get_terminal_state(batch)
    times = _latent_times(torch.tensor([700.0, 300.0]))

    with pytest.raises(
        ValueError,
        match=r"policy velocity component 'latent'.*\(2, 2\).*received shape \(2, 1\)",
    ):
        forward_velocity_state(trainer, batch, state, times, source="policy")


def test_state_batch_size_rejects_an_unbatched_component() -> None:
    trainer = _trainer(DiffusionNFTTrainer, _adapter())

    with pytest.raises(
        ValueError,
        match=r"component 'latent'.*leading batch dimension.*received shape \(\)",
    ):
        state_batch_size(trainer, LatentState({"latent": torch.tensor(1.0)}))


# ============================ DiffusionNFT ============================


def _legacy_nft_loss_elements(
    beta: float,
    clean: torch.Tensor,
    noised_latents: torch.Tensor,
    sigma_broadcast: torch.Tensor,
    new_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the pre-migration NFT normalized squared errors, unreduced."""
    spatial = tuple(range(1, clean.ndim))
    positive_pred = beta * new_velocity + (1 - beta) * old_velocity
    negative_pred = (1.0 + beta) * old_velocity - beta * new_velocity

    x0_pred = noised_latents - sigma_broadcast * positive_pred
    with torch.no_grad():
        weight = (
            torch.abs(x0_pred.double() - clean.double())
            .mean(dim=spatial, keepdim=True)
            .clip(min=1e-5)
        )
    neg_x0_pred = noised_latents - sigma_broadcast * negative_pred
    with torch.no_grad():
        neg_weight = (
            torch.abs(neg_x0_pred.double() - clean.double())
            .mean(dim=spatial, keepdim=True)
            .clip(min=1e-5)
        )
    return (
        (x0_pred - clean) ** 2 / weight,
        (neg_x0_pred - clean) ** 2 / neg_weight,
    )


def _legacy_nft_losses(
    beta: float,
    clean: torch.Tensor,
    noised_latents: torch.Tensor,
    sigma_broadcast: torch.Tensor,
    new_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the pre-migration NFT per-sample matching losses."""
    positive, negative = _legacy_nft_loss_elements(
        beta, clean, noised_latents, sigma_broadcast, new_velocity, old_velocity
    )
    spatial = tuple(range(1, clean.ndim))
    return positive.mean(dim=spatial), negative.mean(dim=spatial)


def test_nft_matching_losses_match_the_legacy_single_component_formula() -> None:
    torch.manual_seed(30)
    clean = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    new_velocity = torch.randn(2, 3, 4)
    old_velocity = torch.randn(2, 3, 4)
    sigma = torch.tensor([0.7, 0.3])
    noised, times = _noised({"latent": clean}, {"latent": noise}, {"latent": sigma})
    trainer = _nft_trainer(_adapter(), nft_beta=0.4)

    positive_loss, negative_loss = trainer._matching_losses(
        LatentState({"latent": clean}),
        noised,
        times,
        LatentState({"latent": new_velocity}),
        LatentState({"latent": old_velocity}),
    )

    expected_positive, expected_negative = _legacy_nft_losses(
        0.4,
        clean,
        noised.state.components["latent"],
        to_broadcast_tensor(sigma, clean),
        new_velocity,
        old_velocity,
    )
    assert torch.equal(positive_loss, expected_positive)
    assert torch.equal(negative_loss, expected_negative)


def test_nft_matching_losses_keep_the_normalization_out_of_the_gradient_graph() -> None:
    torch.manual_seed(31)
    clean = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    old_velocity = torch.randn(2, 3, 4)
    base_velocity = torch.randn(2, 3, 4)
    sigma = torch.tensor([0.7, 0.3])
    noised, times = _noised({"latent": clean}, {"latent": noise}, {"latent": sigma})
    trainer = _nft_trainer(_adapter(), nft_beta=0.4)

    migrated_velocity = base_velocity.clone().requires_grad_(True)
    positive_loss, negative_loss = trainer._matching_losses(
        LatentState({"latent": clean}),
        noised,
        times,
        LatentState({"latent": migrated_velocity}),
        LatentState({"latent": old_velocity}),
    )
    (positive_loss.sum() + negative_loss.sum()).backward()

    legacy_velocity = base_velocity.clone().requires_grad_(True)
    legacy_positive, legacy_negative = _legacy_nft_losses(
        0.4,
        clean,
        noised.state.components["latent"],
        to_broadcast_tensor(sigma, clean),
        legacy_velocity,
        old_velocity,
    )
    (legacy_positive.sum() + legacy_negative.sum()).backward()

    assert torch.equal(migrated_velocity.grad, legacy_velocity.grad)


def test_nft_matching_losses_clamp_the_normalization_floor() -> None:
    """A perfect prediction divides by a zero deviation without the ``1e-5`` clamp."""
    clean = torch.zeros(2, 4)
    noised, times = _noised(
        {"latent": clean}, {"latent": torch.zeros(2, 4)}, {"latent": torch.tensor([0.5, 0.5])}
    )
    trainer = _nft_trainer(_adapter(), nft_beta=0.5)

    positive_loss, negative_loss = trainer._matching_losses(
        LatentState({"latent": clean}),
        noised,
        times,
        LatentState({"latent": torch.zeros(2, 4)}),
        LatentState({"latent": torch.zeros(2, 4)}),
    )

    assert torch.equal(positive_loss, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(negative_loss, torch.zeros(2, dtype=torch.float64))


def test_nft_matching_losses_use_each_component_sigma_and_normalization() -> None:
    torch.manual_seed(32)
    clean = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    noise = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    new_velocity = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    old_velocity = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    sigmas = {"video": torch.tensor([0.8, 0.2]), "audio": torch.tensor([0.35, 0.95])}
    noised, times = _noised(clean, noise, sigmas)
    trainer = _nft_trainer(_structured_adapter(), nft_beta=0.4)

    positive_loss, negative_loss = trainer._matching_losses(
        LatentState(clean), noised, times, LatentState(new_velocity), LatentState(old_velocity)
    )

    positive_sum, negative_sum = torch.zeros(2, dtype=torch.float64), torch.zeros(
        2, dtype=torch.float64
    )
    for name in ("video", "audio"):
        positive_elements, negative_elements = _legacy_nft_loss_elements(
            0.4,
            clean[name],
            noised.state.components[name],
            to_broadcast_tensor(sigmas[name], clean[name]),
            new_velocity[name],
            old_velocity[name],
        )
        positive_sum = positive_sum + positive_elements.flatten(1).sum(dim=1)
        negative_sum = negative_sum + negative_elements.flatten(1).sum(dim=1)
    assert torch.equal(positive_loss, positive_sum / 17)
    assert torch.equal(negative_loss, negative_sum / 17)


def test_nft_matching_losses_recover_clean_state_for_data_ward_velocity() -> None:
    clean = {
        "video": torch.tensor([[1.0, 2.0]]),
        "audio": torch.tensor([[3.0, 4.0, 5.0]]),
    }
    noise = {
        "video": torch.tensor([[5.0, 6.0]]),
        "audio": torch.tensor([[9.0, 10.0, 11.0]]),
    }
    sigmas = {"video": torch.tensor([0.75]), "audio": torch.tensor([0.25])}
    noised, times = _noised(clean, noise, sigmas)
    data_ward_velocity = {name: clean[name] - noise[name] for name in ("video", "audio")}
    trainer = _nft_trainer(
        _structured_adapter(DataWardStructuredAdapterFake),
        nft_beta=0.5,
    )

    positive_loss, negative_loss = trainer._matching_losses(
        LatentState(clean),
        noised,
        times,
        LatentState(data_ward_velocity),
        LatentState(data_ward_velocity),
    )

    assert torch.equal(positive_loss, torch.zeros(1, dtype=torch.float64))
    assert torch.equal(negative_loss, torch.zeros(1, dtype=torch.float64))


def test_nft_reference_kl_matches_the_legacy_velocity_formula() -> None:
    torch.manual_seed(33)
    new_velocity, ref_velocity = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    noised, _ = _noised(
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2)},
    )
    trainer = _nft_trainer(_adapter())

    kl_div = trainer._velocity_kl(
        LatentState({"latent": new_velocity}), LatentState({"latent": ref_velocity}), noised
    )

    legacy = torch.mean((new_velocity - ref_velocity) ** 2, dim=tuple(range(1, new_velocity.ndim)))
    assert torch.equal(kl_div, legacy)


def test_nft_reference_kl_passes_the_noised_state_to_the_global_reducer() -> None:
    torch.manual_seed(34)
    new_velocity, ref_velocity = torch.randn(2, 4), torch.randn(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    trainer = _nft_trainer(_adapter(DynamicMaskAdapterFake))

    kl_div = trainer._velocity_kl(
        LatentState({"latent": new_velocity}),
        LatentState({"latent": ref_velocity}),
        _masked_noised(mask),
    )

    squared = (new_velocity - ref_velocity) ** 2
    assert torch.equal(kl_div, (squared * mask).sum(dim=1) / mask.sum(dim=1))


def test_nft_precompute_draws_noise_once_per_training_timestep() -> None:
    adapter = _adapter()
    trainer = _nft_trainer(adapter, num_train_timesteps=3)
    batch = _legacy_batch()
    clean_state = adapter.get_terminal_state(batch)

    torch.manual_seed(35)
    steps = trainer._precompute_sampling_policy_steps(batch, clean_state, 2)

    assert adapter.noise_calls == 3
    assert adapter.modes == ["rollout"]
    assert len(steps) == 3
    assert len(adapter.forward_calls) == 3
    for step, call in zip(steps, adapter.forward_calls):
        assert call["latents"] is step.noised.state.components["latent"]
        assert step.velocity.components["latent"].requires_grad is False
    assert not torch.equal(
        steps[0].noised.noise.components["latent"], steps[1].noised.noise.components["latent"]
    )
    torch.manual_seed(99)
    again = trainer._precompute_sampling_policy_steps(batch, clean_state, 2, batch_index=0)
    other = trainer._precompute_sampling_policy_steps(batch, clean_state, 2, batch_index=1)
    assert torch.equal(steps[0].times.timestep["latent"], again[0].times.timestep["latent"])
    assert not torch.equal(steps[0].times.timestep["latent"], other[0].times.timestep["latent"])
    drawn = torch.stack([step.times.timestep["latent"].reshape(-1)[0] for step in steps])
    assert torch.unique(drawn).numel() == drawn.numel()


# ============================ AWM ============================


def _legacy_matching_value(
    model_output: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    weighting: str,
    ghuber_power: float = 0.25,
) -> torch.Tensor:
    """Reproduce the pre-migration AWM weighted matching value in double precision."""
    model_output = model_output.double()
    target = target.double()
    log_prob = -((model_output - target) ** 2)
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    t = flow_match_sigma(timestep.view(-1))
    if weighting == "Uniform":
        pass
    elif weighting == "t":
        log_prob = log_prob * t
    elif weighting == "t**2":
        log_prob = log_prob * t**2
    elif weighting == "huber":
        log_prob = -(torch.sqrt(-log_prob + 1e-10) - 1e-5) * t
    elif weighting == "ghuber":
        eps = torch.tensor(1e-10, device=log_prob.device, dtype=log_prob.dtype)
        log_prob = (
            -(torch.pow(-log_prob + eps, ghuber_power) - torch.pow(eps, ghuber_power))
            * t
            / ghuber_power
        )
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")
    return log_prob


@pytest.mark.parametrize("weighting", ["Uniform", "t", "t**2", "huber", "ghuber"])
def test_awm_matching_log_prob_matches_every_legacy_weighting_scheme(weighting: str) -> None:
    torch.manual_seed(40)
    clean = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    velocity = torch.randn(2, 3, 4)
    timestep = torch.tensor([700.0, 250.0])
    noised, times = _noised(
        {"latent": clean}, {"latent": noise}, {"latent": flow_match_sigma(timestep)}
    )
    trainer = _awm_trainer(_adapter(), weighting=weighting)

    log_prob = trainer._matching_log_prob(LatentState({"latent": velocity}), noised, times)

    legacy = _legacy_matching_value(velocity, noise - clean, timestep, weighting).float()
    assert log_prob.dtype is torch.float32
    assert torch.equal(log_prob, legacy)


@pytest.mark.parametrize("weighting", ["Uniform", "t", "t**2", "huber", "ghuber"])
def test_awm_public_weighted_log_prob_still_matches_the_legacy_formula(weighting: str) -> None:
    torch.manual_seed(41)
    velocity, target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    timestep = torch.tensor([700.0, 250.0])

    value = AWMTrainer.compute_weighted_log_prob(velocity, target, timestep, weighting)

    legacy = _legacy_matching_value(velocity, target, timestep, weighting).float()
    assert torch.equal(value, legacy)


def test_awm_matching_log_prob_weights_each_component_by_its_own_sigma() -> None:
    torch.manual_seed(42)
    clean = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    noise = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    velocity = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    timesteps = {"video": torch.tensor([700.0, 250.0]), "audio": torch.tensor([300.0, 900.0])}
    sigmas = {name: flow_match_sigma(value) for name, value in timesteps.items()}
    noised, times = _noised(clean, noise, sigmas)
    trainer = _awm_trainer(_structured_adapter(), weighting="t")

    log_prob = trainer._matching_log_prob(LatentState(velocity), noised, times)

    video = _legacy_matching_value(
        velocity["video"], noise["video"] - clean["video"], timesteps["video"], "t"
    )
    audio = _legacy_matching_value(
        velocity["audio"], noise["audio"] - clean["audio"], timesteps["audio"], "t"
    )
    assert torch.equal(log_prob, ((video * 12 + audio * 5) / 17).float())


def test_awm_matching_log_prob_rejects_an_unknown_weighting_scheme() -> None:
    noised, times = _noised(
        {"latent": torch.zeros(2, 4)}, {"latent": torch.zeros(2, 4)}, {"latent": torch.zeros(2)}
    )
    trainer = _awm_trainer(_adapter(), weighting="quadratic")

    with pytest.raises(ValueError, match=r"awm_weighting.*Uniform.*received 'quadratic'"):
        trainer._matching_log_prob(LatentState({"latent": torch.zeros(2, 4)}), noised, times)


def test_awm_velocity_kl_matches_the_legacy_formula() -> None:
    torch.manual_seed(44)
    velocity, other_velocity = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    noised, _ = _noised(
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2, 3, 4)},
        {"latent": torch.zeros(2)},
    )
    trainer = _awm_trainer(_adapter())

    kl_div = trainer._velocity_kl(
        LatentState({"latent": velocity}), LatentState({"latent": other_velocity}), noised
    )

    legacy = ((velocity - other_velocity) ** 2).mean(dim=tuple(range(1, velocity.ndim)))
    assert torch.equal(kl_div, legacy)


def test_awm_velocity_kl_passes_the_noised_state_to_the_global_reducer() -> None:
    torch.manual_seed(45)
    velocity, other_velocity = torch.randn(2, 4), torch.randn(2, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    trainer = _awm_trainer(_adapter(DynamicMaskAdapterFake))

    kl_div = trainer._velocity_kl(
        LatentState({"latent": velocity}),
        LatentState({"latent": other_velocity}),
        _masked_noised(mask),
    )

    squared = (velocity - other_velocity) ** 2
    assert torch.equal(kl_div, (squared * mask).sum(dim=1) / mask.sum(dim=1))


def test_awm_precompute_draws_noise_once_per_training_timestep() -> None:
    adapter = _adapter()
    trainer = _awm_trainer(adapter, num_train_timesteps=3)
    batch = _legacy_batch()
    clean_state = adapter.get_terminal_state(batch)

    torch.manual_seed(46)
    steps = trainer._precompute_old_log_probs(batch, clean_state, 2)

    assert adapter.noise_calls == 3
    assert adapter.modes == ["rollout"]
    assert len(steps) == 3
    assert len(adapter.forward_calls) == 3
    for step, call in zip(steps, adapter.forward_calls):
        assert call["latents"] is step.noised.state.components["latent"]
        assert step.log_prob.shape == (2,)
        assert step.log_prob.requires_grad is False


# ============================ DPO ============================


def test_dpo_shares_one_noise_tensor_between_the_preference_arms() -> None:
    adapter = _adapter()
    chosen = LatentState({"latent": torch.arange(8, dtype=torch.float32).reshape(2, 4)})
    rejected = LatentState({"latent": -torch.arange(8, dtype=torch.float32).reshape(2, 4)})
    times = _latent_times(torch.tensor([700.0, 300.0]))

    torch.manual_seed(50)
    chosen_noised = adapter.add_forward_process_noise(chosen, times)
    rejected_noised = adapter.apply_forward_process_noise(rejected, times, chosen_noised.noise)

    noise = chosen_noised.noise.components["latent"]
    sigma = to_broadcast_tensor(times.sigma["latent"], noise)
    assert rejected_noised.noise is chosen_noised.noise
    assert adapter.noise_calls == 1
    assert torch.equal(
        rejected_noised.state.components["latent"],
        (1 - sigma) * rejected.components["latent"] + sigma * noise,
    )
    assert torch.equal(
        rejected_noised.target_velocity.components["latent"],
        noise - rejected.components["latent"],
    )


def test_dpo_arm_velocity_error_matches_the_legacy_formula() -> None:
    torch.manual_seed(51)
    clean = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    velocity = torch.randn(2, 3, 4)
    noised, _ = _noised({"latent": clean}, {"latent": noise}, {"latent": torch.tensor([0.6, 0.2])})
    trainer = _dpo_trainer(_adapter())

    error = trainer._arm_velocity_error(LatentState({"latent": velocity}), noised)

    target = noise - clean
    legacy = ((velocity.float() - target.float()) ** 2).mean(dim=tuple(range(1, velocity.ndim)))
    assert torch.equal(error, legacy)


def test_dpo_arm_velocity_error_weights_components_by_element_count() -> None:
    torch.manual_seed(52)
    clean = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    noise = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    velocity = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    noised, _ = _noised(
        clean, noise, {"video": torch.tensor([0.6, 0.2]), "audio": torch.tensor([0.4, 0.9])}
    )
    trainer = _dpo_trainer(_structured_adapter())

    error = trainer._arm_velocity_error(LatentState(velocity), noised)

    video_sum = (velocity["video"] - (noise["video"] - clean["video"])).flatten(1).pow(2).sum(dim=1)
    audio_sum = (velocity["audio"] - (noise["audio"] - clean["audio"])).flatten(1).pow(2).sum(dim=1)
    assert torch.equal(error, (video_sum + audio_sum) / 17)


def test_dpo_arm_velocity_error_passes_the_arm_state_to_the_global_reducer() -> None:
    torch.manual_seed(53)
    velocity = torch.randn(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    trainer = _dpo_trainer(_adapter(DynamicMaskAdapterFake))

    error = trainer._arm_velocity_error(LatentState({"latent": velocity}), _masked_noised(mask))

    squared = velocity.float() ** 2
    assert torch.equal(error, (squared * mask).sum(dim=1) / mask.sum(dim=1))


def test_dpo_preference_loss_matches_the_legacy_formula_and_metrics() -> None:
    torch.manual_seed(54)
    theta_w_err, theta_l_err = torch.rand(2), torch.rand(2)
    ref_w_err, ref_l_err = torch.rand(2), torch.rand(2)
    trainer = _dpo_trainer(_adapter(), beta=2000.0)

    loss, metrics = trainer._preference_loss(theta_w_err, theta_l_err, ref_w_err, ref_l_err)

    beta = 2000.0
    w_diff = theta_w_err - ref_w_err
    l_diff = theta_l_err - ref_l_err
    legacy_loss = -F.logsigmoid(-0.5 * beta * (w_diff - l_diff)).mean()
    legacy_chosen = -0.5 * beta * w_diff
    legacy_rejected = -0.5 * beta * l_diff
    assert torch.equal(loss, legacy_loss)
    assert torch.equal(metrics["implicit_reward_chosen"], legacy_chosen)
    assert torch.equal(metrics["implicit_reward_rejected"], legacy_rejected)
    assert torch.equal(
        metrics["implicit_accuracy"], (legacy_chosen > legacy_rejected).float().mean()
    )


def test_dpo_paired_terminal_states_return_the_shared_batch_size() -> None:
    trainer = _dpo_trainer(_adapter())

    batch_size = trainer._require_paired_terminal_states(
        LatentState({"latent": torch.zeros(2, 4)}), LatentState({"latent": torch.zeros(2, 4)})
    )

    assert batch_size == 2


def test_dpo_paired_terminal_states_reject_a_foreign_component_order() -> None:
    trainer = _dpo_trainer(_structured_adapter())

    with pytest.raises(
        ValueError,
        match=r"rejected terminal state.*DPOTrainer.*\('video', 'audio'\).*\('video',\)",
    ):
        trainer._require_paired_terminal_states(
            LatentState({"video": torch.zeros(2, 4), "audio": torch.zeros(2, 5)}),
            LatentState({"video": torch.zeros(2, 4)}),
        )


def test_dpo_paired_terminal_states_reject_a_shape_mismatch() -> None:
    trainer = _dpo_trainer(_adapter())

    with pytest.raises(ValueError, match=r"component 'latent'.*\(2, 4\).*received \(2, 5\)"):
        trainer._require_paired_terminal_states(
            LatentState({"latent": torch.zeros(2, 4)}), LatentState({"latent": torch.zeros(2, 5)})
        )


def test_dpo_paired_terminal_states_reject_a_dtype_mismatch() -> None:
    trainer = _dpo_trainer(_adapter())

    with pytest.raises(ValueError, match=r"component 'latent'.*torch.float32.*torch.float64"):
        trainer._require_paired_terminal_states(
            LatentState({"latent": torch.zeros(2, 4)}),
            LatentState({"latent": torch.zeros(2, 4, dtype=torch.float64)}),
        )


def test_dpo_paired_terminal_states_reject_a_mismatched_batch_size() -> None:
    trainer = _dpo_trainer(_adapter())

    with pytest.raises(ValueError, match=r"component 'latent'.*\(2, 4\).*received \(3, 4\)"):
        trainer._require_paired_terminal_states(
            LatentState({"latent": torch.zeros(2, 4)}), LatentState({"latent": torch.zeros(3, 4)})
        )


def test_dpo_paired_terminal_states_reject_an_unbatched_component() -> None:
    trainer = _dpo_trainer(_adapter())

    with pytest.raises(
        ValueError,
        match=r"component 'latent'.*leading batch dimension.*received shape \(\)",
    ):
        trainer._require_paired_terminal_states(
            LatentState({"latent": torch.tensor(1.0)}), LatentState({"latent": torch.tensor(2.0)})
        )
