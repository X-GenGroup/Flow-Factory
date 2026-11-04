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

"""Data-parallel geometry read from a real Accelerate device mesh.

These run on real process groups rather than a stub mesh on purpose. The
geometry is read out of a ``DeviceMesh`` whose joint ``dp`` dimension Accelerate
creates with ``DeviceMesh._flatten``, and a flattened dimension is recorded in
the mesh's flatten mapping while staying *out* of ``mesh_dim_names``. A stub
that sets ``mesh_dim_names = ("dp", "tp")`` therefore passes while the same code
reads the wrong geometry from every real mesh, so the stub cannot be the test.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.parallelism_config import ParallelismConfig

from flow_factory.utils.dist import (
    data_parallel_geometry,
    data_parallel_process_group,
    reduce_across_replicas,
)

WORLD_SIZE = 4


def _geometry_worker(rank: int, store_path: str, result_dir: str, sizes: Dict[str, int]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        parallelism_config = ParallelismConfig(**sizes)
        mesh = parallelism_config.build_device_mesh("cpu")
        accelerator = SimpleNamespace(
            parallelism_config=parallelism_config,
            torch_device_mesh=mesh,
            num_processes=WORLD_SIZE,
            process_index=rank,
        )
        world_size, dp_rank = data_parallel_geometry(accelerator)
        result.update(
            dp_world_size=world_size,
            dp_rank=dp_rank,
            root_mesh_dim_names=list(mesh.mesh_dim_names or ()),
            dp_in_root_names="dp" in (mesh.mesh_dim_names or ()),
        )
        group = data_parallel_process_group(accelerator)
        result["has_group"] = group is not None
        if group is not None:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                repeated = [data_parallel_process_group(accelerator) for _ in range(5)]
                data_parallel_geometry(accelerator)
            result["same_group_every_call"] = all(again is group for again in repeated)
            result["warnings_on_later_calls"] = [str(w.message)[:80] for w in caught]
            result["replica_sum"] = reduce_across_replicas(
                accelerator, torch.tensor([float(rank + 1)]), "sum"
            ).tolist()
            try:
                reduce_across_replicas(accelerator, {"count": torch.ones(1), "label": 2.5}, "sum")
                result["non_tensor_leaf"] = "accepted"
            except TypeError as error:
                result["non_tensor_leaf"] = f"TypeError: {error}"
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run_four_ranks(tmp_path: Path, sizes: Dict[str, int]) -> List[Dict[str, Any]]:
    context = mp.get_context("spawn")
    store_path = str(tmp_path / "gloo-store")
    processes = [
        context.Process(target=_geometry_worker, args=(rank, store_path, str(tmp_path), sizes))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(timeout=5)
    assert not hanging, "four-rank data-parallel geometry run exceeded 120 seconds"
    assert [process.exitcode for process in processes] == [0] * WORLD_SIZE
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(WORLD_SIZE)
    ]


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


def test_without_a_parallelism_plan_the_launched_world_is_the_replica_set() -> None:
    """No plan means one replica per process, which is every current run."""
    accelerator = SimpleNamespace(num_processes=8, process_index=3)

    assert data_parallel_geometry(accelerator) == (8, 3)


def test_a_plan_reserving_no_ranks_leaves_the_launched_world_the_replica_set() -> None:
    """An all-ones plan is a real runtime state, and it must not mean one replica.

    Accelerate builds one from an accelerate config with a ``parallelism_config``
    section, or ``--use_parallelism_config``, and no sizes, and under FSDP2 it
    never validates it: ``_validate_accelerator`` returns early because
    ``multi_device`` excludes FSDP, so ``data_parallel_size`` stays 1 on all
    ranks and no mesh is built. Reading that size handed every rank the same
    samples -- silently, where ``main`` sharded by process.
    """
    accelerator = SimpleNamespace(
        parallelism_config=ParallelismConfig(
            dp_replicate_size=1, dp_shard_size=1, tp_size=1, cp_size=1
        ),
        torch_device_mesh=None,
        num_processes=8,
        process_index=3,
    )

    assert data_parallel_geometry(accelerator) == (8, 3)
    assert data_parallel_process_group(accelerator) is None


def test_tensor_parallel_ranks_share_a_data_parallel_coordinate(tmp_path: Path) -> None:
    """Two-way tensor parallelism over four ranks leaves two replicas, not four."""
    results = _run_four_ranks(tmp_path, {"dp_shard_size": 2, "tp_size": 2})

    for result in results:
        assert "error" not in result, result["error"]
        assert result["dp_world_size"] == 2, result

    # Ranks 0/1 form one tensor-parallel group and 2/3 the other, so each pair
    # must report the same replica index -- that index is what selects samples.
    by_rank = {result["rank"]: result["dp_rank"] for result in results}
    assert by_rank == {0: 0, 1: 0, 2: 1, 3: 1}


def test_the_replica_group_is_resolved_once_per_mesh(tmp_path: Path) -> None:
    """Every replica-scoped reduction asks for the group, several times a step.

    ``mesh["dp"]`` builds a fresh submesh on each call and warns that slicing a
    flattened dimension is deprecated, so the slice is kept per mesh: later
    calls hand back the same group and warn about nothing.
    """
    results = _run_four_ranks(tmp_path, {"dp_shard_size": 2, "tp_size": 2})

    for result in results:
        assert "error" not in result, result["error"]
        assert result["has_group"] is True, result
        assert result["same_group_every_call"] is True, result
        assert result["warnings_on_later_calls"] == [], result
    # Ranks hold 1..4; replicas are {0, 2} and {1, 3}, so each sums its own pair.
    assert {result["rank"]: result["replica_sum"] for result in results} == {
        0: [4.0],
        1: [6.0],
        2: [4.0],
        3: [6.0],
    }


def test_a_replica_reduction_refuses_a_leaf_it_cannot_reduce(tmp_path: Path) -> None:
    """A float beside the tensors used to come back unreduced under tensor parallelism.

    ``accelerator.reduce``, which the world branch uses, raises on such a leaf,
    so the replica branch does the same rather than passing it through.
    """
    results = _run_four_ranks(tmp_path, {"dp_shard_size": 2, "tp_size": 2})

    for result in results:
        assert "error" not in result, result["error"]
        assert result["non_tensor_leaf"].startswith("TypeError"), result
        assert "float" in result["non_tensor_leaf"], result


@pytest.mark.parametrize("reduction", ["min", "max", "none"])
def test_a_replica_reduction_accepts_only_a_sum_or_a_mean(reduction: str) -> None:
    """``accelerator.reduce`` has no ``min`` and sums instead, so neither branch offers it."""
    accelerator = SimpleNamespace(
        num_processes=1,
        process_index=0,
        reduce=lambda *_args, **_kwargs: pytest.fail("refused before any collective"),
    )

    with pytest.raises(ValueError, match="expected reduction 'sum' or 'mean'"):
        reduce_across_replicas(accelerator, torch.ones(1), reduction)


def test_the_joint_dp_dimension_is_absent_from_the_root_mesh_names(
    tmp_path: Path,
) -> None:
    """Guard the reason this is not read from ``mesh_dim_names``.

    Accelerate flattens ``dp_shard`` into a joint ``dp`` dimension, which does
    not appear in the root mesh's names. Code that gates on
    ``"dp" in mesh.mesh_dim_names`` silently falls back to the launched world
    and hands every tensor-parallel rank its own sample.
    """
    results = _run_four_ranks(tmp_path, {"dp_shard_size": 2, "tp_size": 2})

    for result in results:
        assert result["root_mesh_dim_names"] == ["dp_shard", "tp"], result
        assert result["dp_in_root_names"] is False, result


def test_a_world_fully_reserved_for_tensor_parallelism_has_one_replica(
    tmp_path: Path,
) -> None:
    """With every rank in one tensor-parallel group there is a single replica.

    The parallelism config carries no data-parallel dimension at all here, so
    the size cannot come from the mesh and has to be read from the plan.
    """
    results = _run_four_ranks(tmp_path, {"tp_size": WORLD_SIZE})

    for result in results:
        assert "error" not in result, result["error"]
        assert (result["dp_world_size"], result["dp_rank"]) == (1, 0), result
