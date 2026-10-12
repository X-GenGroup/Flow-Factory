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

import pytest
import torch

from flow_factory.utils.noise_schedule import (
    TIMESTEP_MAX,
    TimeSampler,
    flow_match_sigma,
    validate_flow_match_coordinates,
)


def test_flow_match_sigma_preserves_dtype_and_exact_endpoints() -> None:
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        timesteps = torch.tensor([0.0, TIMESTEP_MAX], dtype=dtype)
        sigmas = flow_match_sigma(timesteps)

        assert sigmas.dtype == dtype
        torch.testing.assert_close(
            sigmas,
            torch.tensor([0.0, 1.0], dtype=dtype),
            rtol=0,
            atol=0,
        )


def test_flow_match_sigma_keeps_representable_upper_interior_strict() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    interior = torch.nextafter(endpoint, torch.zeros_like(endpoint))

    sigma = flow_match_sigma(interior)

    assert interior.item() < TIMESTEP_MAX
    assert sigma.item() < 1.0


def test_flow_match_coordinates_accept_one_ulp_but_reject_two() -> None:
    timestep = torch.tensor([990.4219970703125], dtype=torch.float32)
    expected = flow_match_sigma(timestep)
    direction = torch.full_like(expected, float("inf"))
    one_ulp = torch.nextafter(expected, direction)
    two_ulps = torch.nextafter(one_ulp, direction)

    validate_flow_match_coordinates(timestep, one_ulp)
    with pytest.raises(ValueError, match=r"within one native ULP"):
        validate_flow_match_coordinates(timestep, two_ulps)

    binade_timestep = torch.tensor([500.0], dtype=torch.float32)
    binade_sigma = torch.tensor([0.5], dtype=torch.float32)
    lower = torch.full_like(binade_sigma, -float("inf"))
    one_ulp_down = torch.nextafter(binade_sigma, lower)
    two_ulps_down = torch.nextafter(one_ulp_down, lower)
    validate_flow_match_coordinates(binade_timestep, one_ulp_down)
    with pytest.raises(ValueError, match=r"within one native ULP"):
        validate_flow_match_coordinates(binade_timestep, two_ulps_down)


def test_flow_match_coordinates_reject_clamped_out_of_range_values() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    outside = torch.nextafter(endpoint, torch.full_like(endpoint, float("inf")))

    with pytest.raises(ValueError, match=r"timestep.*\[0, 1000\].*sigma.*\[0, 1\]"):
        validate_flow_match_coordinates(outside, torch.ones_like(outside))


@pytest.mark.parametrize(
    "timesteps",
    [
        torch.tensor([-1.0]),
        torch.tensor([TIMESTEP_MAX + 1]),
        torch.tensor([float("nan")]),
    ],
)
def test_flow_match_sigma_rejects_invalid_scheduler_coordinates(
    timesteps: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"t_scheduler.*\[0, 1000\]"):
        flow_match_sigma(timesteps)


def test_flow_match_sigma_rejects_non_tensor_input() -> None:
    with pytest.raises(TypeError, match=r"torch.Tensor t_scheduler"):
        flow_match_sigma(500.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sampler_name",
    ["independent_logit_normal_shifted", "independent_uniform"],
)
def test_independent_time_samplers_materialize_distinct_batch_draws(
    sampler_name: str,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    sampler = getattr(TimeSampler, sampler_name)

    timesteps = sampler(
        batch_size=4,
        num_timesteps=3,
        timestep_range=(0.2, 0.8),
        generator=generator,
    )

    assert timesteps.shape == (3, 4)
    assert timesteps.stride() != (1, 0)
    assert torch.all(timesteps >= 200.0)
    assert torch.all(timesteps <= 800.0)
    assert all(torch.unique(row).numel() > 1 for row in timesteps)


@pytest.mark.parametrize(
    "sampler_name",
    ["independent_logit_normal_shifted", "independent_uniform"],
)
def test_independent_time_samplers_are_generator_reproducible(sampler_name: str) -> None:
    sampler = getattr(TimeSampler, sampler_name)
    first_generator = torch.Generator(device="cpu").manual_seed(91)
    second_generator = torch.Generator(device="cpu").manual_seed(91)

    first = sampler(2, 4, 0.99, generator=first_generator)
    second = sampler(2, 4, 0.99, generator=second_generator)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("sampler_name", ["logit_normal_shifted", "uniform"])
def test_legacy_time_samplers_keep_shared_batch_coordinates(sampler_name: str) -> None:
    sampler = getattr(TimeSampler, sampler_name)
    generator = torch.Generator(device="cpu").manual_seed(321)

    timesteps = sampler(4, 3, (0.2, 0.8), generator=generator)

    assert timesteps.shape == (3, 4)
    for row in timesteps:
        torch.testing.assert_close(row, row[0].expand_as(row), rtol=0, atol=0)


@pytest.mark.parametrize("field_name", ["batch_size", "num_timesteps"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.0])
def test_independent_time_samplers_validate_materialized_shape(
    field_name: str,
    invalid: object,
) -> None:
    kwargs = {"batch_size": 2, "num_timesteps": 3, field_name: invalid}
    expected_error = TypeError if type(invalid) is not int else ValueError

    with pytest.raises(expected_error):
        TimeSampler.independent_uniform(timestep_range=0.99, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flow_match_sigma_matches_cpu_at_upper_interior_on_cuda() -> None:
    endpoint = torch.tensor([TIMESTEP_MAX], dtype=torch.float32)
    interior = torch.nextafter(endpoint, torch.zeros_like(endpoint))

    cpu_sigma = flow_match_sigma(interior)
    cuda_sigma = flow_match_sigma(interior.cuda()).cpu()

    assert cuda_sigma.item() < 1.0
    torch.testing.assert_close(cuda_sigma, cpu_sigma, rtol=0, atol=0)


_DISCRETE_TIMESTEPS = torch.linspace(1000.0, 100.0, 10)


def _discrete_draws(num_runs: int, **kwargs: object) -> set:
    """Return every scheduler timestep drawn by ``TimeSampler.discrete`` across seeded runs."""
    drawn = set()
    for seed in range(num_runs):
        draws = TimeSampler.discrete(
            batch_size=1,
            scheduler_timesteps=_DISCRETE_TIMESTEPS,
            generator=torch.Generator().manual_seed(seed),
            **kwargs,
        )
        drawn.update(draws[:, 0].tolist())
    return drawn


@pytest.mark.parametrize(
    ("include_init", "force_init", "first_index"),
    [(True, False, 0), (False, False, 1), (True, True, 0)],
)
def test_discrete_sampler_reaches_every_in_range_timestep(
    include_init: bool,
    force_init: bool,
    first_index: int,
) -> None:
    drawn = _discrete_draws(
        200, num_train_timesteps=3, include_init=include_init, force_init=force_init
    )

    assert drawn == set(_DISCRETE_TIMESTEPS[first_index:].tolist())


def test_discrete_sampler_reaches_the_last_timestep_inside_the_range() -> None:
    drawn = _discrete_draws(100, num_train_timesteps=2, timestep_range=0.5)

    assert drawn == {1000.0, 900.0, 800.0, 700.0, 600.0, 500.0}


def test_discrete_sampler_draws_each_index_once_when_counts_match() -> None:
    draws = TimeSampler.discrete(
        batch_size=2,
        num_train_timesteps=10,
        scheduler_timesteps=_DISCRETE_TIMESTEPS,
        generator=torch.Generator().manual_seed(1),
    )

    assert draws.shape == (10, 2)
    assert draws[:, 0].tolist() == _DISCRETE_TIMESTEPS.tolist()


def test_discrete_sampler_rejects_ranges_without_drawable_timesteps() -> None:
    with pytest.raises(ValueError, match="select at least one scheduler timestep"):
        TimeSampler.discrete(1, 2, _DISCRETE_TIMESTEPS, timestep_range=(0.02, 0.05))
    with pytest.raises(ValueError, match=r"non-empty index range \[1, 1\)"):
        TimeSampler.discrete(
            1, 2, _DISCRETE_TIMESTEPS, timestep_range=(0.0, 0.05), include_init=False
        )
