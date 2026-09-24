import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List

import pytest

from flow_factory.data_utils.multi_source import (
    MultiSourceTrainDataLoader,
    WeightedSourceBatchScheduler,
)

_SUBPROCESS_SCHEDULE_SCRIPT = """
import json

from flow_factory.data_utils.multi_source import WeightedSourceBatchScheduler

scheduler = WeightedSourceBatchScheduler(
    {"alpha": 8, "beta": 8, "gamma": 8},
    seed=17,
)
scheduler.set_epoch(3)
print(json.dumps(list(scheduler)))
"""


def _schedule_from_fresh_process(python_hash_seed: int) -> List[str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(python_hash_seed)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((source_root, inherited_pythonpath))
        if inherited_pythonpath
        else source_root
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCHEDULE_SCRIPT],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return json.loads(completed.stdout)


def test_multi_source_loader_exposes_batch_size_for_deepspeed() -> None:
    loader = MultiSourceTrainDataLoader(
        {},
        WeightedSourceBatchScheduler({}, seed=42),
        batch_size=3,
    )

    assert loader.batch_size == 3


def test_source_schedule_is_independent_of_process_hash_seed() -> None:
    first = _schedule_from_fresh_process(python_hash_seed=1)
    second = _schedule_from_fresh_process(python_hash_seed=2)

    assert first == second


def test_default_source_schedule_preserves_one_batch_blocks() -> None:
    counts = {"beta": 4, "alpha": 2}

    implicit = list(WeightedSourceBatchScheduler(counts, seed=29))
    explicit = list(WeightedSourceBatchScheduler(counts, seed=29, batches_per_block=1))

    assert implicit == explicit
    assert Counter(implicit) == Counter(counts)


def test_source_schedule_shuffles_only_complete_blocks() -> None:
    scheduler = WeightedSourceBatchScheduler(
        {"alpha": 6, "beta": 3, "gamma": 6},
        seed=31,
        batches_per_block=3,
    )

    schedule = list(scheduler)
    blocks = [schedule[start : start + 3] for start in range(0, len(schedule), 3)]

    assert all(len(set(block)) == 1 for block in blocks)
    assert Counter(schedule) == Counter({"alpha": 6, "beta": 3, "gamma": 6})


def test_source_schedule_rejects_incomplete_blocks() -> None:
    with pytest.raises(ValueError, match="each source batch count"):
        WeightedSourceBatchScheduler(
            {"alpha": 4, "beta": 3},
            seed=37,
            batches_per_block=3,
        )
