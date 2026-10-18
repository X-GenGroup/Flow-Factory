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

from flow_factory.hparams import Arguments


def _parse(scheduler: dict, num_inference_steps: int = 6) -> Arguments:
    return Arguments.from_dict(
        {
            "train": {
                "trainer_type": "grpo",
                "num_inference_steps": num_inference_steps,
                "unique_sample_num_per_epoch": 8,
                "group_size": 2,
                "per_device_batch_size": 1,
            },
            "scheduler": {"dynamics_type": "Flow-SDE", **scheduler},
            "rewards": [{"name": "score", "reward_model": "clip"}],
        }
    )


def test_sde_steps_within_the_rollout_are_kept() -> None:
    """Keep explicit transition indices that the rollout reaches."""
    config = _parse({"sde_steps": [0, 3, 5], "num_sde_steps": 2})

    assert config.scheduler_args.sde_steps == [0, 3, 5]
    assert config.scheduler_args.num_sde_steps == 2


def test_null_sde_steps_resolve_to_every_transition_but_the_last() -> None:
    """Resolve null sde_steps and num_sde_steps from num_inference_steps."""
    config = _parse({"sde_steps": None, "num_sde_steps": None})

    assert config.scheduler_args.sde_steps == [0, 1, 2, 3, 4]
    assert config.scheduler_args.num_sde_steps == 5


def test_sde_steps_beyond_the_rollout_fail_at_config_load() -> None:
    """Name the unreachable indices instead of failing later during replay."""
    with pytest.raises(ValueError, match=r"in \[0, 5\].*out of range: \[6, 7, 8, 9\]"):
        _parse({"sde_steps": list(range(10)), "num_sde_steps": 3})


@pytest.mark.parametrize(
    ("scheduler", "error", "message"),
    [
        ({"sde_steps": [-1, 2], "num_sde_steps": 1}, ValueError, r"out of range: \[-1\]"),
        ({"sde_steps": [1, 1, 2], "num_sde_steps": 1}, ValueError, "must not repeat"),
        ({"sde_steps": [1, 2], "num_sde_steps": 3}, ValueError, "exceeds the 2 eligible"),
        ({"sde_steps": [1, True], "num_sde_steps": 1}, TypeError, "list of int"),
        ({"sde_steps": [1, 2], "num_sde_steps": 1.0}, TypeError, "num_sde_steps must be an int"),
    ],
)
def test_malformed_sde_configuration_fails_at_config_load(
    scheduler: dict, error: type, message: str
) -> None:
    """Reject negative, repeated, over-counted, or non-integer SDE step settings."""
    with pytest.raises(error, match=message):
        _parse(scheduler)


def test_ode_dynamics_skip_sde_step_validation() -> None:
    """Leave ODE runs alone: they have no stochastic transitions to validate."""
    config = _parse({"dynamics_type": "ODE", "sde_steps": list(range(10))})

    assert config.scheduler_args.sde_steps == []
