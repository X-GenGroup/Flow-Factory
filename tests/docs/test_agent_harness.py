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

"""Regression tests for the repository agent harness."""

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_validator() -> ModuleType:
    """Load the harness validator without making ``scripts`` a package."""
    script_path = REPOSITORY_ROOT / "scripts/validate_agent_harness.py"
    spec = importlib.util.spec_from_file_location("validate_agent_harness", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load harness validator from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_agent_harness_is_structurally_valid() -> None:
    """Keep canonical policies, skill registration, and knowledge links aligned."""
    validator = _load_validator()

    assert validator.validate_harness(REPOSITORY_ROOT) == []


def test_validator_rejects_tool_adapter_policy_drift(tmp_path: Path) -> None:
    """Reject policy copied into a tool adapter instead of the canonical root."""
    validator = _load_validator()
    for filename in ("AGENTS.md", "CLAUDE.md"):
        shutil.copy2(REPOSITORY_ROOT / filename, tmp_path / filename)
    for directory in (".agents", ".cursor"):
        shutil.copytree(REPOSITORY_ROOT / directory, tmp_path / directory)
    (tmp_path / "CLAUDE.md").write_text(
        "@AGENTS.md\n\nDuplicated policy.\n",
        encoding="utf-8",
    )

    errors = validator.validate_harness(tmp_path)

    assert "CLAUDE.md must contain only '@AGENTS.md'" in errors
