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

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

#: Skip every test in this package when gloo is unavailable, which is the
#: backend each spawned rank initializes.
requires_gloo = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


def run_gloo_ranks(
    worker: Callable[..., None],
    tmp_path: Path,
    world_size: int,
    options: Dict[str, Any],
    *,
    timeout: int = 300,
) -> List[Dict[str, Any]]:
    """Spawn `world_size` gloo ranks, run `worker`, and collect their JSON results.

    One copy of the harness every distributed test in this package shares, in
    place of the nine it grew: the spawn/join/collect sequence, the liveness
    assertion, and the rendezvous store path were identical in each, so a
    timeout or a result-shape change had to be made nine times.

    `worker` is called as `worker(rank, store_path, result_dir, options)` and is
    expected to write `rank-<n>.json` into `result_dir`; by convention it
    reports a caught exception under an `"error"` key so the parent asserts on
    it rather than reading a traceback out of a dead child.

    Spawn, not fork: a forked rank inherits the parent's CUDA and gloo state.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = str(tmp_path / "store")
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(rank, store, str(tmp_path), options))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=timeout)
    alive = [p for p in processes if p.is_alive()]
    for process in alive:
        process.terminate()
    assert not alive, f"{worker.__name__} exceeded {timeout} seconds on {len(alive)} rank(s)"
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]


BACKEND_CHOICES = ("ddp", "fsdp2", "zero1", "zero2")
ALGORITHM_CHOICES = ("dmd2", "tdm", "tdm-r1")
STORAGE_MODE_CHOICES = ("lora", "full")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the distributed backend acceptance selector."""
    parser.addoption(
        "--backend",
        action="store",
        choices=BACKEND_CHOICES,
        help="Distributed backend contract exercised by this test run.",
    )
    parser.addoption(
        "--algorithm",
        action="store",
        choices=ALGORITHM_CHOICES,
        help="Multi-role algorithm selected for distributed acceptance.",
    )
    parser.addoption(
        "--storage-mode",
        action="store",
        choices=STORAGE_MODE_CHOICES,
        help="Parameter storage mode selected for distributed acceptance.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Select strict distributed matrix axes from CLI or enumerate all values."""
    for fixture_name, choices in (
        ("algorithm", ALGORITHM_CHOICES),
        ("storage_mode", STORAGE_MODE_CHOICES),
    ):
        if fixture_name not in metafunc.fixturenames:
            continue
        selected = metafunc.config.getoption(
            f"--{fixture_name.replace('_', '-')}",
            default=None,
        )
        metafunc.parametrize(fixture_name, (selected,) if selected is not None else choices)


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> str:
    """Return the explicitly selected distributed backend."""
    value = request.config.getoption("--backend", default=None)
    if value is None:
        pytest.skip("distributed acceptance requires an explicit --backend")
    return value
