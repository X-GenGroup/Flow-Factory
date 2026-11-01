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

"""Every public module must import on its own, in any order.

``scheduler.sde_noise`` once took ``stable_text_id`` from
``trainers.common.group_noise``, which pulled the whole ``trainers`` package
into the scheduler's import, and ``trainers.abc`` imports the scheduler back.
Any import order that reached ``scheduler.sde_noise`` before ``trainers.abc``
therefore found it half-initialized and raised ``ImportError``. The entry point
happened to be safe because it loads ``trainers`` first, so only direct adapter
imports broke -- which is exactly the kind of breakage a single-module import
check catches and an end-to-end test does not.

Each module is imported in a *fresh* interpreter: once any module has primed
``sys.modules``, a cycle no longer reproduces.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# The adapters that import `bind_sde_sample`, plus the scheduler module itself
# and the two entry points that must keep working.
MODULES = [
    "flow_factory.scheduler.sde_noise",
    "flow_factory.trainers.common.group_noise",
    "flow_factory.models.flux.flux2",
    "flow_factory.models.flux.flux2_klein",
    "flow_factory.models.flux.flux3",
    "flow_factory.models.sensenova.sensenova",
    "flow_factory.models.qwen_image_21.qwen_image_21",
    "flow_factory.trainers",
    "flow_factory.train",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_standalone(module_name: str) -> None:
    """A fresh interpreter must be able to import this module first."""
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert (
        completed.returncode == 0
    ), f"importing {module_name} first failed:\n{completed.stderr[-2000:]}"


def test_stable_text_id_is_reachable_from_both_homes() -> None:
    """Moving it must not break callers importing it from the trainer helpers."""
    from flow_factory.trainers.common.group_noise import stable_text_id as from_trainers
    from flow_factory.utils.base import stable_text_id as from_utils

    assert from_trainers is from_utils
    assert from_utils("prompt") == from_utils("prompt")
    assert from_utils("prompt") != from_utils("other")
