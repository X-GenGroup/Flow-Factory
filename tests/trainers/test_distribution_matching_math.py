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

from typing import Any, List

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.minimax_h3 import (
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.minimax_h3 import (
    build_training_component_times as build_h3_training_component_times,
)
from flow_factory.samples import BaseSample, ComponentTimes, LatentState
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.trainers.distillation.distribution_matching import (
    add_flow_noise,
    dmd_generator_loss,
    flow_matching_loss,
    flow_velocity_target,
    tdm_conditional_renoise,
    tdm_fake_loss,
    tdm_generator_loss,
    velocity_to_x0,
)


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter for objective reduction tests."""

    trajectory_component_order = ("latent",)

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        raise NotImplementedError


def _adapter() -> AdapterFake:
    return object.__new__(AdapterFake)


def _state(tensor: torch.Tensor) -> LatentState:
    return LatentState({"latent": tensor})


def _times(sigma: torch.Tensor) -> ComponentTimes:
    return ComponentTimes(
        timestep={"latent": sigma * 1000.0},
        next_timestep={"latent": torch.zeros_like(sigma)},
        sigma={"latent": sigma},
        next_sigma={"latent": torch.zeros_like(sigma)},
    )


def test_flow_x0_round_trip() -> None:
    torch.manual_seed(0)
    x0 = _state(torch.randn(2, 4, 3, 3))
    noise = _state(torch.randn(2, 4, 3, 3))
    for value in (0.1, 0.5, 0.9):
        sigma = {"latent": torch.full((2,), value)}
        x_t = add_flow_noise(x0, noise, sigma)
        velocity = flow_velocity_target(x0, noise)
        recovered = velocity_to_x0(x_t, velocity, sigma)
        torch.testing.assert_close(recovered.components["latent"], x0.components["latent"])


def test_dmd_generator_loss_gradient_matches_normalized_direction() -> None:
    adapter = _adapter()
    x0_gen = _state(torch.randn(2, 4, 3, 3, requires_grad=True))
    x0_real = _state(torch.randn(2, 4, 3, 3))
    x0_fake = _state(torch.randn(2, 4, 3, 3))
    loss = dmd_generator_loss(adapter, x0_gen, x0_real, x0_fake)
    loss.backward()

    gen = x0_gen.components["latent"].detach()
    real = x0_real.components["latent"]
    fake = x0_fake.components["latent"]
    per_sample = (gen - real).abs().flatten(1).mean(dim=1).clamp_min(1e-6)
    direction = (fake - real) / per_sample.reshape(-1, 1, 1, 1)
    expected = direction / gen.numel()
    torch.testing.assert_close(x0_gen.components["latent"].grad, expected, atol=1e-5, rtol=1e-5)


def test_flow_matching_loss_is_velocity_mse() -> None:
    adapter = _adapter()
    v_pred = _state(torch.tensor([[[1.0, 3.0]]]))
    v_target = _state(torch.tensor([[[0.0, 1.0]]]))
    loss = flow_matching_loss(adapter, v_pred, v_target, state=v_pred)
    torch.testing.assert_close(loss, torch.tensor(2.5))


def test_tdm_fake_loss_uses_per_sample_snr_weights() -> None:
    adapter = _adapter()
    x0_fake = _state(torch.ones(2, 1, 1, 1, requires_grad=True))
    x0_target = _state(torch.zeros(2, 1, 1, 1))
    low = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.9, 0.9]),
        importance=torch.ones(2),
        snr_gamma=5.0,
    )
    high = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.1, 0.1]),
        importance=torch.ones(2),
        snr_gamma=5.0,
    )
    mixed = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.9, 0.1]),
        importance=torch.ones(2),
        snr_gamma=5.0,
    )
    assert mixed.item() != low.item()
    assert mixed.item() != high.item()
    assert low.item() < high.item()


def test_tdm_conditional_renoise_matches_official_linear_flow_formula() -> None:
    """Mixed noise, state, velocity target, and importance share one derivation."""
    torch.manual_seed(11)
    adapter = _adapter()
    clean = _state(torch.randn(2, 3, 2, 2))
    model_noise = _state(torch.randn_like(clean.components["latent"]))
    sigma_mid = torch.tensor([0.25, 0.4])
    sigma_t = torch.tensor([0.7, 0.8])

    noised, importance = tdm_conditional_renoise(
        adapter,
        clean,
        model_noise,
        mid_times=_times(sigma_mid),
        target_times=_times(sigma_t),
        importance_clip=20.0,
        generator=torch.Generator().manual_seed(11),
    )

    mixed = noised.noise.components["latent"]
    shape = (2, 1, 1, 1)
    sm = sigma_mid.reshape(shape)
    st = sigma_t.reshape(shape)
    ratio = (1 - st) / (1 - sm)
    old_noise_coeff = ratio * sm
    beta = (st.square() - old_noise_coeff.square()).sqrt()
    fresh = (st * mixed - old_noise_coeff * model_noise.components["latent"]) / beta
    x_mid = (1 - sm) * clean.components["latent"] + sm * model_noise.components["latent"]
    expected_state = ratio * x_mid + beta * fresh
    expected_target = mixed - clean.components["latent"]
    expected_importance = torch.exp(
        -0.5 * mixed.square().flatten(1).mean(1) + 0.5 * fresh.square().flatten(1).mean(1)
    ).clamp(1 / 20.0, 20.0)

    torch.testing.assert_close(noised.state.components["latent"], expected_state)
    torch.testing.assert_close(noised.target_velocity.components["latent"], expected_target)
    torch.testing.assert_close(importance, expected_importance)
    assert importance.requires_grad is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tdm_conditional_renoise_preserves_h3_latent_dtype(dtype: torch.dtype) -> None:
    """Keep float32 conditional math behind the adapter's latent-storage boundary."""
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    clean = LatentState(
        {
            "video": torch.linspace(-1, 1, 2 * 96, dtype=dtype).reshape(1, 2, 96),
            "audio": torch.linspace(-1, 1, 3 * 32, dtype=dtype).reshape(1, 3, 32),
        }
    )
    model_noise = LatentState(
        {
            name: torch.linspace(-0.5, 0.5, component.numel(), dtype=torch.float32).reshape(
                component.shape
            )
            for name, component in clean.components.items()
        }
    )
    mid_times = build_h3_training_component_times(
        torch.tensor([250.0]),
        video_shift=1.0,
        audio_shift=1.0,
    )
    target_times = build_h3_training_component_times(
        torch.tensor([750.0]),
        video_shift=1.0,
        audio_shift=1.0,
    )

    noised, importance = tdm_conditional_renoise(
        adapter,
        clean,
        model_noise,
        mid_times=mid_times,
        target_times=target_times,
        generator=torch.Generator().manual_seed(11),
    )

    for state in (noised.state, noised.target_velocity, noised.noise):
        assert all(component.dtype == dtype for component in state.components.values())
    assert importance.dtype == torch.float32


def test_tdm_conditional_renoise_keeps_importance_inputs_in_float32() -> None:
    adapter = _adapter()
    reduced_dtypes: list[tuple[torch.dtype, ...]] = []

    def capture_reduction(values: dict[str, torch.Tensor], *, state: LatentState) -> torch.Tensor:
        reduced_dtypes.append(tuple(component.dtype for component in values.values()))
        return next(iter(values.values())).flatten(1).mean(dim=1)

    adapter.reduce_latent_values = capture_reduction
    clean = _state(torch.linspace(-1, 1, 12, dtype=torch.float16).reshape(1, 3, 2, 2))
    model_noise = _state(torch.linspace(-0.5, 0.5, 12).reshape(1, 3, 2, 2))

    noised, importance = tdm_conditional_renoise(
        adapter,
        clean,
        model_noise,
        mid_times=_times(torch.tensor([0.25])),
        target_times=_times(torch.tensor([0.75])),
        generator=torch.Generator().manual_seed(11),
    )

    assert noised.noise.components["latent"].dtype == torch.float16
    assert reduced_dtypes == [(torch.float32,), (torch.float32,)]
    assert importance.dtype == torch.float32


def test_tdm_fake_loss_is_finite_with_unit_importance() -> None:
    adapter = _adapter()
    x0_fake = _state(torch.randn(2, 4, 3, 3, requires_grad=True))
    x0_target = _state(torch.randn(2, 4, 3, 3))
    loss = tdm_fake_loss(
        adapter,
        x0_fake,
        x0_target,
        sigma=torch.tensor([0.4, 0.6]),
        importance=torch.ones(2),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert x0_fake.components["latent"].grad is not None


def test_tdm_generator_huber_correction_has_real_minus_fake_sign() -> None:
    adapter = _adapter()
    x0_student = _state(torch.zeros(2, 1, 2, 2, requires_grad=True))
    x0_real = _state(torch.ones(2, 1, 2, 2))
    x0_fake = _state(torch.full((2, 1, 2, 2), 3.0))
    loss = tdm_generator_loss(
        adapter,
        x0_student,
        x0_real,
        x0_fake,
        use_huber=True,
        huber_c=1e-3,
    )
    loss.backward()
    grad = x0_student.components["latent"].grad
    assert grad is not None
    assert torch.all(grad > 0)
