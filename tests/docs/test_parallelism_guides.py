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

"""Keep the user-facing parallelism inventory aligned with the live registries."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from repo_checks.core import Repository
from repo_checks.registries import check_registries


@pytest.mark.parametrize(
    "registry,page,label",
    [
        ("model", "models.md", "model adapters"),
        ("trainer", "algorithms.md", "trainers"),
    ],
)
def test_parallelism_guide_accounts_for_every_registered_key(registry, page, label):
    view = Repository(ROOT)
    keys = check_registries(view)[registry]
    assert not view.findings
    text = (ROOT / "guidance/parallelism" / page).read_text()
    missing = {key for key in keys if f"`{key}`" not in text}
    assert not missing, f"Add an explicit support disposition for {sorted(missing)}"
    assert f"**{len(keys)} registered {label}**" in text
