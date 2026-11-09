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

"""Every rank agrees to load preprocessing modules when any rank misses its cache (Gloo)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from flow_factory.data_utils.loader import _every_process_cached

        def gather(tensor: torch.Tensor) -> torch.Tensor:
            parts = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(parts, tensor)
            return torch.cat(parts)

        accelerator = SimpleNamespace(device=torch.device("cpu"), gather=gather)
        result["agreed"] = [
            _every_process_cached(accelerator, cached, True)
            for cached in (options["cached"][rank], True)
        ]
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("cached", [(True, False), (False, True), (False, False)])
def test_one_rank_missing_its_cache_makes_every_rank_load(tmp_path: Path, cached) -> None:
    results = run_gloo_ranks(_worker, tmp_path, 2, {"cached": list(cached)})
    for result in results:
        assert "error" not in result, result
        assert result["agreed"] == [False, True], result
