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

"""The process RNG is keyed by data-parallel replica, not by launched rank.

Every draw a trainer still takes from the process RNG -- DMD2's and DGPO's
forward-process noise, H3's condition noise, LoRA dropout -- has to agree across
the ranks of one tensor-parallel group, which run one forward on one sample.
Keyed per rank, their all-reduces combine activations computed from different
inputs and nothing raises.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.distributed as dist
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import set_seed
from conftest import requires_gloo, run_gloo_ranks

from flow_factory.utils.dist import seed_data_parallel_replica

SEED = 1234


def _draws() -> list:
    return torch.rand(4).tolist() + [float(np.random.rand()), random.random()]


@pytest.mark.parametrize("process_index", [0, 3])
def test_without_a_plan_the_stream_is_the_device_specific_one(process_index: int) -> None:
    """Every process is its own replica, so nothing changes for current runs.

    ``set_seed(seed, device_specific=True)`` adds the process index to the seed;
    the replica coordinate is that index here.
    """
    accelerator = SimpleNamespace(num_processes=8, process_index=process_index)

    applied = seed_data_parallel_replica(accelerator, SEED)
    replica_draws = _draws()
    set_seed(SEED + process_index)
    device_specific_draws = _draws()

    assert applied == SEED + process_index
    assert replica_draws == device_specific_draws


def test_a_boolean_seed_is_refused() -> None:
    with pytest.raises(TypeError, match="expected int for seed"):
        seed_data_parallel_replica(SimpleNamespace(num_processes=1, process_index=0), True)


def _seeding_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    world_size = options["world_size"]
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    result: Dict[str, Any] = {"rank": rank}
    try:
        parallelism_config = ParallelismConfig(**options["sizes"])
        accelerator = SimpleNamespace(
            parallelism_config=parallelism_config,
            torch_device_mesh=parallelism_config.build_device_mesh("cpu"),
            num_processes=world_size,
            process_index=rank,
        )
        result["applied"] = seed_data_parallel_replica(accelerator, SEED)
        result["draws"] = _draws()
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_tensor_parallel_peers_share_one_stream_and_replicas_do_not(tmp_path: Path) -> None:
    """Ranks 0/1 and 2/3 are the two tensor-parallel groups of a 2x2 plan."""
    results = run_gloo_ranks(
        _seeding_worker,
        tmp_path,
        4,
        {"world_size": 4, "sizes": {"dp_shard_size": 2, "tp_size": 2}},
    )
    for result in results:
        assert "error" not in result, result["error"]
    by_rank = {result["rank"]: result for result in results}

    assert [by_rank[rank]["applied"] for rank in range(4)] == [SEED, SEED, SEED + 1, SEED + 1]
    assert by_rank[0]["draws"] == by_rank[1]["draws"]
    assert by_rank[2]["draws"] == by_rank[3]["draws"]
    assert by_rank[0]["draws"] != by_rank[2]["draws"]
