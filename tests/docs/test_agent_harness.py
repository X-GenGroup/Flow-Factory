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
import json
import shutil
from pathlib import Path
from types import ModuleType

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    """Load a repository script without making ``scripts`` a package."""
    script_path = REPOSITORY_ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load harness validator from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_harness_fixture(destination: Path) -> None:
    """Copy the repository harness into an isolated validation fixture."""
    for filename in ("AGENTS.md", "CLAUDE.md"):
        shutil.copy2(REPOSITORY_ROOT / filename, destination / filename)
    for directory in (".agents", ".claude", ".cursor"):
        shutil.copytree(REPOSITORY_ROOT / directory, destination / directory)


def test_repository_agent_harness_is_structurally_valid() -> None:
    """Keep canonical policies, skill registration, and knowledge links aligned."""
    validator = _load_script("validate_agent_harness")

    assert validator.validate_harness(REPOSITORY_ROOT) == []


def test_validator_rejects_tool_adapter_policy_drift(tmp_path: Path) -> None:
    """Reject policy copied into a tool adapter instead of the canonical root."""
    validator = _load_script("validate_agent_harness")
    _copy_harness_fixture(tmp_path)
    (tmp_path / "CLAUDE.md").write_text(
        "@AGENTS.md\n\nDuplicated policy.\n",
        encoding="utf-8",
    )

    errors = validator.validate_harness(tmp_path)

    assert "CLAUDE.md must contain only '@AGENTS.md'" in errors


def test_validator_rejects_broken_progressive_disclosure_reference(tmp_path: Path) -> None:
    """Resolve skill references from the file that owns each progressive detail."""
    _copy_harness_fixture(tmp_path)
    reference = tmp_path / ".agents/skills/ff-debug/references/full-protocol.md"
    reference.write_text(
        reference.read_text(encoding="utf-8").replace(
            "../../../knowledge/topics/parity_testing.md",
            "../../../knowledge/topics/missing.md",
        ),
        encoding="utf-8",
    )
    validator = _load_script("validate_agent_harness")

    errors = validator.validate_harness(tmp_path)

    assert any("local Markdown reference does not exist" in error for error in errors)


def test_validator_rejects_broad_tool_permission(tmp_path: Path) -> None:
    """Keep generic interpreters outside persistent tool permission adapters."""
    _copy_harness_fixture(tmp_path)
    settings_path = tmp_path / ".claude/settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["permissions"]["allow"].append("Bash(python3:*)")
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    validator = _load_script("validate_agent_harness")

    errors = validator.validate_harness(tmp_path)

    assert any("broad permissions remain" in error for error in errors)


def test_validator_requires_docs_only_rule_and_listed_eval_paths(tmp_path: Path) -> None:
    """Reject a missing docs-only rule and eval docs-only paths outside the case paths."""
    _copy_harness_fixture(tmp_path)
    risk_path = tmp_path / ".agents/harness/risk_rules.yaml"
    risk = yaml.safe_load(risk_path.read_text(encoding="utf-8"))
    del risk["docs_only_rule"]
    risk_path.write_text(yaml.safe_dump(risk, sort_keys=False), encoding="utf-8")
    eval_path = tmp_path / ".agents/harness/evals/cases.yaml"
    evals = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    evals["cases"][0]["docs_only_paths"] = ["src/flow_factory/unlisted.py"]
    eval_path.write_text(yaml.safe_dump(evals, sort_keys=False), encoding="utf-8")
    validator = _load_script("validate_agent_harness")

    errors = validator.validate_harness(tmp_path)

    assert any("docs_only_rule must name a profile" in error for error in errors)
    assert any("docs_only_paths must be listed paths" in error for error in errors)


def test_scope_classifier_keeps_objective_only_algorithm_on_fast_path() -> None:
    """Keep an algorithm-local objective on R1 when it reuses every shared contract."""
    scope = _load_script("agent_scope")
    rules = yaml.safe_load(
        (REPOSITORY_ROOT / ".agents/harness/risk_rules.yaml").read_text(encoding="utf-8")
    )
    path = "src/flow_factory/trainers/rl/my_algorithm.py"

    result = scope.classify_changes(
        [path],
        {path: "class MyAlgorithmTrainer(BaseTrainer):\n    def optimize(self, samples): ..."},
        rules,
        intent="new-algorithm",
    )

    assert result["profile"] == "R1"
    assert "objective" in result["facets"]


def test_scope_classifier_escalates_advanced_algorithm_hook() -> None:
    """Escalate an algorithm that changes acquisition grouping or persistent state."""
    scope = _load_script("agent_scope")
    rules = yaml.safe_load(
        (REPOSITORY_ROOT / ".agents/harness/risk_rules.yaml").read_text(encoding="utf-8")
    )
    path = "src/flow_factory/trainers/distillation/my_algorithm.py"

    result = scope.classify_changes(
        [path],
        {path: "def _run_training_step(self):\n    ..."},
        rules,
        intent="new-algorithm",
    )

    assert result["profile"] == "R3"
    assert "runtime_state" in result["facets"]


def test_scope_classifier_marks_shared_execution_as_campaign() -> None:
    """Treat a BaseTrainer lifecycle change as shared framework infrastructure."""
    scope = _load_script("agent_scope")
    rules = yaml.safe_load(
        (REPOSITORY_ROOT / ".agents/harness/risk_rules.yaml").read_text(encoding="utf-8")
    )
    path = "src/flow_factory/trainers/abc.py"

    result = scope.classify_changes(
        [path],
        {path: "def start(self):\n    ..."},
        rules,
        intent="develop",
    )

    assert result["profile"] == "R4"


def test_scope_classifier_keeps_docs_only_shared_edit_on_fast_path() -> None:
    """Skip shared-lifecycle escalation when a file changed only comments and docstrings."""
    scope = _load_script("agent_scope")
    rules = yaml.safe_load(
        (REPOSITORY_ROOT / ".agents/harness/risk_rules.yaml").read_text(encoding="utf-8")
    )
    path = "src/flow_factory/trainers/abc.py"

    result = scope.classify_changes(
        [path],
        {path: "def start(self):\n    ..."},
        rules,
        intent="docs",
        docs_only_paths=[path],
    )

    assert result["profile"] == "R1"
    assert result["facets"] == ["docs_only"]
    assert result["docs_only_paths"] == [path]


def test_scope_classifier_keeps_code_changes_strict_beside_docs_only_edits() -> None:
    """Never let a docs-only file lower the profile required by another code change."""
    scope = _load_script("agent_scope")
    rules = yaml.safe_load(
        (REPOSITORY_ROOT / ".agents/harness/risk_rules.yaml").read_text(encoding="utf-8")
    )
    docs_path = "src/flow_factory/models/abc.py"
    code_path = "src/flow_factory/trainers/abc.py"

    result = scope.classify_changes(
        [docs_path, code_path],
        {docs_path: '"""Summary."""', code_path: "def start(self):\n    ..."},
        rules,
        intent="develop",
        docs_only_paths=[docs_path],
    )

    assert result["profile"] == "R4"
    assert "model_runtime" not in result["facets"]
    assert {"docs_only", "execution"}.issubset(result["facets"])


def test_fix_query_retrieves_distributed_reward_history() -> None:
    """Retrieve a prior cross-rank reward failure without loading the full archive."""
    query = _load_script("query_fix_patterns")

    results = query.search_patterns(
        REPOSITORY_ROOT / ".agents/knowledge/fixes",
        query="distributed reward gathering",
        limit=3,
    )

    assert results
    assert any("Distributed reward gathering" in result["title"] for result in results)


def test_harness_eval_cases_select_expected_profiles_and_facets() -> None:
    """Keep fast paths fast and shared lifecycle changes on strict profiles."""
    scope = _load_script("agent_scope")
    rules = yaml.safe_load(
        (REPOSITORY_ROOT / ".agents/harness/risk_rules.yaml").read_text(encoding="utf-8")
    )
    evals = yaml.safe_load(
        (REPOSITORY_ROOT / ".agents/harness/evals/cases.yaml").read_text(encoding="utf-8")
    )

    for case in evals["cases"]:
        result = scope.classify_changes(
            case["paths"],
            case["paths"],
            rules,
            intent=case["intent"],
            docs_only_paths=case.get("docs_only_paths", []),
        )
        assert result["profile"] == case["expected_profile"], case["name"]
        assert set(case["expected_facets"]).issubset(result["facets"]), case["name"]
