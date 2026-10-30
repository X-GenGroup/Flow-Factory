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

from flow_factory.hparams import Arguments, ParallelismArguments


def test_unset_block_requests_no_parallelism() -> None:
    """Every dimension defaults to unset, which leaves the world purely data parallel."""
    assert ParallelismArguments().is_configured is False


@pytest.mark.parametrize("field_name", ["dp_replicate_size", "dp_shard_size", "tp_size"])
def test_any_explicit_dimension_configures_the_plan(field_name: str) -> None:
    """One explicit dimension is enough to hand Accelerate a mesh plan."""
    args = ParallelismArguments(**{field_name: 2})

    assert args.is_configured is True


@pytest.mark.parametrize("field_name", ["dp_replicate_size", "dp_shard_size", "tp_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_size_below_one_is_rejected(field_name: str, value: int) -> None:
    """A non-positive dimension cannot describe a process group."""
    with pytest.raises(ValueError, match=f"parallelism.{field_name} must be at least 1"):
        ParallelismArguments(**{field_name: value})


def test_arguments_parses_the_parallelism_block() -> None:
    """The top-level ``parallelism`` block maps onto ``parallelism_args``."""
    config = Arguments.from_dict({"parallelism": {"dp_shard_size": 4, "tp_size": 2}})

    assert config.parallelism_args.dp_shard_size == 4
    assert config.parallelism_args.tp_size == 2
    assert config.parallelism_args.dp_replicate_size is None


def test_absent_block_keeps_the_defaults() -> None:
    """A config without the block must not request any parallelism."""
    config = Arguments.from_dict({})

    assert config.parallelism_args.is_configured is False


@pytest.mark.parametrize("unknown_key", ["cp_size", "sp_size", "pp_size"])
def test_dimensions_outside_the_supported_plan_are_not_modeled(unknown_key: str) -> None:
    """Context, DeepSpeed sequence, and pipeline parallelism are not exposed.

    They are captured as extra kwargs rather than silently mapped onto dimensions the
    trainer factory would then forward to Accelerate.
    """
    args = ParallelismArguments.from_dict({unknown_key: 2})

    assert args.is_configured is False
    assert args.extra_kwargs == {unknown_key: 2}
