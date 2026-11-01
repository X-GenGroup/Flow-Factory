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
    args = ParallelismArguments()

    assert args.is_configured is False
    assert args.effective_tensor_parallel_size == 1


@pytest.mark.parametrize("field_name", ["dp_replicate_size", "dp_shard_size", "tp_size"])
def test_any_explicit_dimension_configures_the_plan(field_name: str) -> None:
    """One explicit dimension is enough to request a mesh."""
    args = ParallelismArguments(**{field_name: 2})

    assert args.is_configured is True


@pytest.mark.parametrize("field_name", ["dp_replicate_size", "dp_shard_size", "tp_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_size_below_one_is_rejected(field_name: str, value: int) -> None:
    """A non-positive dimension cannot describe a process group."""
    with pytest.raises(ValueError, match=f"parallelism.{field_name} must be at least 1"):
        ParallelismArguments(**{field_name: value})


@pytest.mark.parametrize("value", [2.5, "2", True])
def test_non_integer_size_is_rejected(value: object) -> None:
    """A float or a bool would silently become a wrong mesh dimension."""
    with pytest.raises(ValueError, match="must be an integer when set"):
        ParallelismArguments(tp_size=value)


def test_effective_tensor_parallel_size_reports_one_when_unset() -> None:
    """Callers dividing the world by this must not have to handle ``None``."""
    assert ParallelismArguments(dp_shard_size=8).effective_tensor_parallel_size == 1
    assert ParallelismArguments(tp_size=4).effective_tensor_parallel_size == 4


def test_arguments_parses_the_parallelism_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-level ``parallelism`` block maps onto ``parallelism_args``.

    The world size has to match the requested layout: ``Arguments`` validates
    that ``tp_size`` divides the launched world, because every derived batch
    quantity is already computed against that world.
    """
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = Arguments.from_dict({"parallelism": {"dp_shard_size": 4, "tp_size": 2}})

    assert config.parallelism_args.dp_shard_size == 4
    assert config.parallelism_args.tp_size == 2
    assert config.parallelism_args.dp_replicate_size is None
    # Eight ranks, two per tensor-parallel group: four replicas tile the batch.
    assert config.data_parallel_world_size() == 4


def test_tensor_parallel_size_must_divide_the_launched_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layout no launch can host is rejected at parse time, not at mesh build.

    Checked for every acquisition mode: dataset acquisition returns before batch
    geometry is aligned, so a check inside the alignment path would let an
    impossible SFT layout through and fail much later.
    """
    monkeypatch.setenv("WORLD_SIZE", "6")
    with pytest.raises(ValueError, match="must divide the launched world size 6"):
        Arguments.from_dict({"parallelism": {"tp_size": 4}})


def test_absent_block_keeps_the_defaults() -> None:
    """A config without the block must not request any parallelism."""
    config = Arguments.from_dict({})

    assert config.parallelism_args.is_configured is False


@pytest.mark.parametrize("unknown_key", ["cp_size", "sp_size", "pp_size"])
def test_dimensions_outside_the_supported_plan_are_not_modeled(unknown_key: str) -> None:
    """Context, DeepSpeed sequence, and pipeline parallelism are not exposed.

    They land in ``extra_kwargs`` rather than being silently mapped onto a
    dimension the mesh builder would then act on.
    """
    args = ParallelismArguments.from_dict({unknown_key: 2})

    assert args.is_configured is False
    assert args.extra_kwargs == {unknown_key: 2}


def test_sequence_parallelism_requires_a_tensor_parallel_group() -> None:
    """It shards the token axis *across* that group, so there must be one.

    Rejected in the arguments rather than at mesh construction, because the flag
    is meaningless on its own and the later error would name a device mesh
    rather than the thing the user wrote.
    """
    with pytest.raises(ValueError, match="requires parallelism.tp_size > 1"):
        ParallelismArguments.from_dict({"sequence_parallel": True})

    with pytest.raises(ValueError, match="requires parallelism.tp_size > 1"):
        ParallelismArguments.from_dict({"tp_size": 1, "sequence_parallel": True})


def test_sequence_parallelism_defaults_off_and_stays_out_of_the_sizes() -> None:
    """A flag, not a mesh dimension: it reuses the tensor-parallel group.

    So it must not join the product that has to equal the launched world size,
    and an absent block must not request it.
    """
    args = ParallelismArguments.from_dict({"tp_size": 2})

    assert args.sequence_parallel is False
    assert args.to_dict()["sequence_parallel"] is False
    assert ParallelismArguments.from_dict({"tp_size": 2, "sequence_parallel": True}).to_dict() == {
        "dp_replicate_size": None,
        "dp_shard_size": None,
        "tp_size": 2,
        "sequence_parallel": True,
    }


@pytest.mark.parametrize("value", ["true", 1, None])
def test_a_non_boolean_sequence_parallel_flag_is_rejected(value: object) -> None:
    """``"true"`` from a loosely written YAML must not read as enabled."""
    with pytest.raises(ValueError, match="sequence_parallel must be a boolean"):
        ParallelismArguments(tp_size=2, sequence_parallel=value)
