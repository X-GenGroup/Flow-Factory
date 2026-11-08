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

"""Check catalog semantics without importing Torch or any model runtime."""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validation_catalog", ROOT / "scripts/validation_catalog.py"
)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


@pytest.fixture
def document():
    return catalog.load_catalog()


def test_shared_tp_plan_test_selects_its_cpu_case_without_weakening_gpu_coverage():
    from test_validation_planner import (
        planner,
        test_opd_catalog_binds_transitive_fixture_imports,
    )

    path = "tests/models/test_tensor_parallel_plan.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {
        "paths": [path],
        "facets": [],
        "docs_only_paths": [],
        "logging_only_paths": [],
        "scope_complete": True,
    }
    requirements = planner.requirements_for(scope, {}, catalog, policy)
    assert "review.owner:" + path not in requirements["obligations"]
    cases, missing = planner.select_cases(catalog, requirements, [])
    assert not missing
    selected = {case["id"]: case for case in cases if case["selected"]}
    assert set(selected) == {"tp-plan-structure-contracts", "repository-consistency"}
    case = selected["tp-plan-structure-contracts"]
    assert case["command"] == ["python3", "-m", "pytest", "-q", path, "-k", "not bagel"]
    assert case["covers"] == ["parallel.tp_plan_structure"]
    assert case["qa"] == "Q0" and case["backend"] == "cpu"
    assert case["resources"]["gpus"] == 0
    test_opd_catalog_binds_transitive_fixture_imports(case["id"])
    # Compare shared-runtime selections with the old catalog: this structural
    # case must never substitute for an existing numerical/distributed proof.
    runtime_scope = {
        **scope,
        "paths": [
            "src/flow_factory/models/tensor_parallel.py",
            "src/flow_factory/trainers/role_optimization.py",
        ],
        "facets": ["distributed"],
    }
    old_catalog = {**catalog, "cases": [row for row in catalog["cases"] if row["id"] != case["id"]]}
    gpu_selections = []
    for candidate in (old_catalog, catalog):
        required = planner.requirements_for(runtime_scope, {}, candidate, policy)
        selected_cases, _ = planner.select_cases(candidate, required, [])
        gpu_selections.append(
            {row["id"] for row in selected_cases if row["selected"] and row["resources"]["gpus"]}
        )
    assert gpu_selections[0] and gpu_selections[0] == gpu_selections[1]


def test_all_qa_levels_are_executable_and_resource_geometry_is_separate(document):
    cases = {case["id"]: case for case in document["cases"]}
    assert {case["qa"] for case in cases.values()} == {"Q0", "Q1", "Q2", "Q3"}
    worker = cases["reward-worker-device"]
    assert worker["topology"]["ranks_per_node"] == 1
    assert worker["resources"]["gpus"] == 2
    assert worker["qa"] == "Q1"
    assert cases["tp-checkpoint-layout"]["topology"]["ranks_per_node"] == 4
    assert cases["tp-checkpoint-layout"]["resources"]["gpus"] == 0
    for backend in ("ddp", "zero2", "fsdp2"):
        assert f"roles.isolation.{backend}" in cases[f"roles-{backend}"]["covers"]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("id", "Bad ID", "invalid case id"),
        ("artifacts", ["../AGENTS.md"], "invalid artifact"),
        ("artifacts", ["tests/missing.py"], "missing or external"),
        ("facets", ["imaginary"], "unknown facets"),
        ("cost_seconds", True, "positive estimate"),
        ("fixture_kind", "real", "fixture_kind"),
        ("command", ["sh", "-c", "true"], "interpreter"),
    ],
)
def test_invalid_case_metadata_is_rejected(document, field, value, match):
    document["cases"][0][field] = value
    with pytest.raises(ValueError, match=match):
        catalog.validate_catalog(document)


def test_duplicate_ids_are_rejected(document):
    document["cases"].append(copy.deepcopy(document["cases"][0]))
    with pytest.raises(ValueError, match="duplicate id"):
        catalog.validate_catalog(document)


def test_missing_pytest_selector_fails_without_import(document):
    case = document["cases"][0]
    case["command"] = [
        "python3",
        "-m",
        "pytest",
        "tests/scripts/test_agent_scope.py::test_does_not_exist",
    ]
    with pytest.raises(ValueError, match="missing test selector"):
        catalog.validate_catalog(document)


def test_executed_test_must_have_an_artifact_identity(document):
    case = document["cases"][0]
    case["command"] = ["python3", "-m", "pytest", "tests/scripts/test_validation_catalog.py"]
    case["artifacts"].remove("tests/scripts/test_validation_catalog.py")
    with pytest.raises(ValueError, match="missing from artifacts"):
        catalog.validate_catalog(document)


def test_required_gpus_cannot_hide_behind_one_rank(document):
    case = next(case for case in document["cases"] if case["id"] == "roles-fsdp2")
    case["resources"]["gpus"] = 1
    with pytest.raises(ValueError, match="GPU resources"):
        catalog.validate_catalog(document)


def test_real_structure_is_not_a_pretrained_weight_claim(document):
    case = next(case for case in document["cases"] if case["id"] == "pretrained-wan-canary")
    case["fixture_kind"] = "model_structure"
    with pytest.raises(ValueError, match="Q3 must exercise pretrained"):
        catalog.validate_catalog(document)


def test_pretrained_case_requires_explicit_local_weights(document):
    case = next(case for case in document["cases"] if case["id"] == "pretrained-wan-canary")
    case["resources"]["model_paths"] = []
    with pytest.raises(ValueError, match="pretrained weights require"):
        catalog.validate_catalog(document)


def test_selected_fields_and_interpreter_binding_do_not_mutate_catalog(document):
    case = document["cases"][0]
    selected = catalog.selected_case_fields(case)
    selected["covers"].append("extra")
    assert "extra" not in case["covers"]
    assert catalog.prepare_command(case, "/explicit/python")[0] == "/explicit/python"
    assert case["command"][0] == "python3"


def test_selection_preserves_explicit_topology_scope(document):
    case = document["cases"][0]
    assert "execution_scope" not in catalog.selected_case_fields(case)
    case["execution_scope"] = "topology_only"
    assert catalog.selected_case_fields(case)["execution_scope"] == "topology_only"


def test_cli_inspection_does_not_import_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/validation_catalog.py"),
            "--case",
            "reward-worker-device",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"gpus": 2' in result.stdout


def test_explicit_run_with_unavailable_dependencies_is_blocked(document, monkeypatch):
    case = next(case for case in document["cases"] if case["id"] == "pretrained-wan-canary")

    class Probe:
        returncode = 0
        stdout = '{"missing":["torch"],"gpus":0}'

    monkeypatch.setattr(catalog.subprocess, "run", lambda *args, **kwargs: Probe())
    missing = catalog.preflight(case, sys.executable, {})
    assert any("FF_VALIDATION_WAN_TRANSFORMER" in item for item in missing)
    assert "missing package torch" in missing
    assert "needs 1 visible GPUs, found 0" in missing


def test_immutable_reader_does_not_consult_worktree_targets(document, tmp_path):
    data = {
        path: (ROOT / path).read_bytes() for case in document["cases"] for path in case["artifacts"]
    }
    data[".agents/harness/test_routes.yaml"] = (
        ROOT / ".agents/harness/test_routes.yaml"
    ).read_bytes()
    catalog.validate_catalog(document, root=tmp_path, source=data.get)
    data.pop(document["cases"][0]["artifacts"][0])
    with pytest.raises(ValueError, match="missing or external"):
        catalog.validate_catalog(document, root=tmp_path, source=data.get)
