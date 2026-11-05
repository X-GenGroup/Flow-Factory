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

"""Prove selected-plan coverage, evidence identity and minimum topology boundaries."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validation_plan", ROOT / "scripts/validate_validation_plan.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
COMMIT = "a" * 40


@pytest.fixture
def plan():
    return {
        "schema_version": 1,
        "base_commit": "b" * 40,
        "tested_commit": COMMIT,
        "risk": "R4",
        "owners": ["shared owner"],
        "facets": ["execution"],
        "artifacts": {"tests/example.py": "f" * 64},
        "obligations": [
            {"id": "contract", "description": "The affected contract", "required": True}
        ],
        "cases": [
            {
                "id": "contract-test",
                "selected": True,
                "covers": ["contract"],
                "reason": "Exercise the changed boundary",
                "fixture": "Controlled runtime",
                "command": ["python3", "-m", "pytest", "tests/example.py"],
                "qa": "Q0",
                "backend": "cpu",
                "topology": {
                    "nodes": 1,
                    "ranks_per_node": 1,
                    "dp_replicate": 1,
                    "dp_shard": 1,
                    "tp": 1,
                    "cp": 1,
                    "ep": 1,
                },
                "oracle": {
                    "kind": "assertions",
                    "description": "Explicit contract assertions",
                    "reward": "none",
                },
            }
        ],
    }


def results(plan):
    case = plan["cases"][0]
    world = case["topology"]["nodes"] * case["topology"]["ranks_per_node"]
    runtime = {
        "world_size": world,
        "ranks_completed": list(range(world)),
        "device": "cpu" if case["backend"] in {"cpu", "gloo"} else "cuda",
        "distributed_type": {
            "cpu": "NO",
            "cuda": "NO",
            "gloo": "PROCESS_GROUP",
            "nccl": "PROCESS_GROUP",
            "tp": "PROCESS_GROUP",
            "ddp": "MULTI_GPU",
            "zero2": "DEEPSPEED",
            "fsdp2": "FSDP",
        }[case["backend"]],
    }
    if case["backend"] == "zero2":
        runtime["zero_stage"] = 2
    if case["backend"] == "fsdp2":
        runtime["fsdp_version"] = 2
    if case["backend"] in {"gloo", "nccl", "tp"}:
        runtime["process_group_backend"] = "gloo" if case["backend"] == "gloo" else "nccl"
    return {
        "schema_version": 1,
        "tested_commit": plan["tested_commit"],
        "plan_sha256": validator.plan_sha256(plan),
        "artifacts": copy.deepcopy(plan["artifacts"]),
        "environment": {"python": "3.10", "hardware": "CPU", "packages": {"pytest": "9.0"}},
        "cases": [
            {
                "id": case["id"],
                "status": "pass",
                "execution_scope": case.get("execution_scope", "runtime"),
                "backend": case["backend"],
                "topology": copy.deepcopy(case["topology"]),
                "command": list(case["command"]),
                "runtime": runtime,
                "evidence": {".scratch/test.log": "e" * 64},
                "observations": {"exit_code": 0},
            }
        ],
    }


def check(plan, actual=None):
    return validator.validate_results(
        plan, results(plan) if actual is None else actual, expected_commit=COMMIT
    )


@pytest.mark.parametrize(
    "qa,backend,dimensions",
    [
        ("Q0", "cpu", {}),
        ("Q0", "gloo", {"ranks_per_node": 2, "dp_replicate": 2}),
        ("Q1", "cuda", {}),
        ("Q2", "ddp", {"ranks_per_node": 2, "dp_replicate": 2}),
        ("Q2", "ddp", {"nodes": 2, "dp_replicate": 2}),
        ("Q2", "zero2", {"ranks_per_node": 2, "dp_replicate": 2}),
        ("Q2", "fsdp2", {"ranks_per_node": 2, "dp_shard": 2}),
        ("Q2", "tp", {"ranks_per_node": 2, "tp": 2}),
        ("Q2", "fsdp2", {"ranks_per_node": 4, "dp_shard": 2, "tp": 2}),
        ("Q3", "cuda", {}),
    ],
)
def test_risk_does_not_force_qa_or_gpu_count(plan, qa, backend, dimensions):
    plan["cases"][0].update(qa=qa, backend=backend)
    plan["cases"][0]["topology"].update(dimensions)
    assert check(plan)["status"] == "pass"


@pytest.mark.parametrize(
    "backend,dimensions,match",
    [
        ("fsdp2", {"ranks_per_node": 2, "tp": 2}, "dp_shard"),
        ("ddp", {"ranks_per_node": 2, "tp": 2}, "DDP"),
        ("tp", {"ranks_per_node": 4, "tp": 2}, "placement"),
        ("fsdp2", {"ranks_per_node": 4, "dp_shard": 2, "cp": 2}, "CP/EP"),
        ("fsdp2", {"ranks_per_node": 4, "dp_shard": 2, "ep": 2}, "CP/EP"),
        ("zero2", {"ranks_per_node": 4, "dp_shard": 2, "tp": 2}, "unsupported"),
        ("zero2", {"ranks_per_node": 2, "dp_shard": 2}, "replicated data ranks"),
        ("zero2", {"ranks_per_node": 4, "dp_replicate": 2, "dp_shard": 2}, "unsupported"),
    ],
)
def test_invalid_or_unimplemented_mesh_is_rejected(plan, backend, dimensions, match):
    plan["cases"][0].update(qa="Q2", backend=backend)
    plan["cases"][0]["topology"].update(dimensions)
    with pytest.raises(ValueError, match=match):
        validator.validate_plan(plan)


@pytest.mark.parametrize("backend,qa", [("gloo", "Q0"), ("nccl", "Q2")])
def test_explicit_topology_probe_can_record_cp_groups_without_claiming_model_execution(
    plan, backend, qa
):
    case = plan["cases"][0]
    case.update(backend=backend, qa=qa, execution_scope="topology_only")
    case["topology"].update(ranks_per_node=8, dp_shard=2, tp=2, cp=2)
    assert check(plan)["status"] == "pass"
    observed = results(plan)
    observed["cases"][0].pop("execution_scope")
    with pytest.raises(ValueError, match="observed execution_scope"):
        check(plan, observed)
    observed = results(plan)
    observed["cases"][0]["runtime"]["process_group_backend"] = "wrong"
    with pytest.raises(ValueError, match="does not establish"):
        check(plan, observed)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"backend": "fsdp2"}, "raw Gloo or NCCL"),
        ({"qa": "Q3"}, "cannot qualify a real model"),
        ({"execution_scope": "model_probe"}, "bounded FSDP2 Q2"),
        ({"execution_scope": []}, "unknown execution_scope"),
        ({"oracle": {"kind": "numerical", "reward": "none"}}, "requires assertions"),
        ({"oracle": {"kind": "assertions", "reward": "varying"}}, "requires assertions"),
    ],
)
def test_topology_scope_does_not_weaken_production_capability_gates(plan, mutation, match):
    case = plan["cases"][0]
    case.update(backend="nccl", qa="Q2", execution_scope="topology_only")
    case["topology"].update(ranks_per_node=2, cp=2)
    case.update(mutation)
    with pytest.raises(ValueError, match=match):
        validator.validate_plan(plan)


def test_topology_scope_does_not_add_ep_or_unscoped_nccl(plan):
    case = plan["cases"][0]
    case.update(backend="nccl", qa="Q2")
    case["topology"].update(ranks_per_node=2, dp_replicate=2)
    with pytest.raises(ValueError, match="raw NCCL requires execution_scope"):
        validator.validate_plan(plan)
    case["execution_scope"] = "topology_only"
    case["topology"].update(dp_replicate=1, ep=2)
    with pytest.raises(ValueError, match="does not define EP ownership"):
        validator.validate_plan(plan)


def _model_probe(plan):
    case = plan["cases"][0]
    case.update(backend="fsdp2", qa="Q2", execution_scope="model_probe")
    case["topology"].update(ranks_per_node=4, dp_shard=2, cp=2)
    case["oracle"]["kind"] = "numerical"
    return case


def test_bounded_cp_model_probe_requires_explicit_numerical_evidence(plan):
    _model_probe(plan)
    observed = results(plan)
    with pytest.raises(ValueError, match="elementwise comparison"):
        check(plan, observed)
    observed["cases"][0]["observations"].update(elementwise_comparison=True, nonzero_gradients=True)
    assert check(plan, observed)["status"] == "pass"
    observed["cases"][0]["execution_scope"] = "runtime"
    with pytest.raises(ValueError, match="observed execution_scope"):
        check(plan, observed)


def test_wan_sft_cp_runtime_requires_matching_capability_observations(plan):
    case = _model_probe(plan)
    case.update(execution_scope="runtime", execution_capability="wan-sft-ulysses-bf16-v1")
    validator.validate_plan(plan)
    observed = results(plan)
    observed["cases"][0]["observations"].update(elementwise_comparison=True, nonzero_gradients=True)
    with pytest.raises(ValueError, match="observed execution_capability"):
        check(plan, observed)
    observed["cases"][0]["observations"]["execution_capability"] = case["execution_capability"]
    assert check(plan, observed)["status"] == "pass"
    case["topology"].update(tp=2, ranks_per_node=8)
    with pytest.raises(ValueError, match="without TP/replication"):
        validator.validate_plan(plan)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"backend": "ddp"}, "bounded FSDP2 Q2"),
        ({"qa": "Q3"}, "bounded FSDP2 Q2"),
        ({"execution_scope": "runtime"}, "CP/EP execution"),
        ({"oracle": {"kind": "assertions", "reward": "none"}}, "numerical oracle"),
        ({"oracle": {"kind": "numerical", "reward": "varying"}}, "without reward"),
    ],
)
def test_model_probe_does_not_advertise_production_cp(plan, mutation, match):
    _model_probe(plan).update(mutation)
    with pytest.raises(ValueError, match=match):
        validator.validate_plan(plan)


@pytest.mark.parametrize("axis", ["tp", "ep", "dp_replicate"])
def test_model_probe_rejects_unqualified_compositions(plan, axis):
    case = _model_probe(plan)
    case["topology"].update(ranks_per_node=8)
    case["topology"][axis] = 2
    with pytest.raises(ValueError, match="without TP/EP/replication"):
        validator.validate_plan(plan)


@pytest.mark.parametrize("target", ["cases", "obligations"])
def test_duplicate_plan_ids_are_rejected(plan, target):
    plan[target].append(copy.deepcopy(plan[target][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validator.validate_plan(plan)


def test_unknown_and_uncovered_obligations_are_rejected(plan):
    plan["cases"][0]["covers"] = ["typo"]
    with pytest.raises(ValueError, match="unknown obligation"):
        validator.validate_plan(plan)
    plan["cases"][0]["covers"] = ["contract"]
    plan["cases"][0]["selected"] = False
    with pytest.raises(ValueError, match="uncovered"):
        validator.validate_plan(plan)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("tested_commit", "c" * 40, "stale tested commit"),
        ("plan_sha256", "0" * 64, "stale plan"),
        ("artifacts", {"tests/example.py": "1" * 64}, "test/config"),
    ],
)
def test_stale_result_identity_is_rejected(plan, field, value, match):
    actual = results(plan)
    actual[field] = value
    with pytest.raises(ValueError, match=match):
        check(plan, actual)


def test_matching_stale_plan_and_results_do_not_evade_expected_commit(plan):
    plan["tested_commit"] = "c" * 40
    with pytest.raises(ValueError, match="plan: stale"):
        check(plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "ddp"),
        ("topology", {"nodes": 32}),
        ("command", ["true"]),
    ],
)
def test_backend_topology_and_command_are_observed_not_just_labels(plan, field, value):
    actual = results(plan)
    actual["cases"][0][field] = value
    with pytest.raises(ValueError, match=f"observed {field}"):
        check(plan, actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("distributed_type", "MULTI_GPU"),
        ("fsdp_version", 1),
        ("ranks_completed", [0]),
        ("world_size", 1),
    ],
)
def test_fsdp_label_cannot_replace_runtime_plugin_or_rank_completion(plan, field, value):
    plan["cases"][0].update(qa="Q2", backend="fsdp2")
    plan["cases"][0]["topology"].update(ranks_per_node=2, dp_shard=2)
    actual = results(plan)
    actual["cases"][0]["runtime"][field] = value
    with pytest.raises(ValueError, match="observed|rank completed"):
        check(plan, actual)


def test_missing_required_evidence_is_blocked_not_pass(plan):
    actual = results(plan)
    actual["cases"] = []
    report = check(plan, actual)
    assert report == {
        "status": "blocked",
        "cases": {"contract-test": "blocked"},
        "uncovered_obligations": ["contract"],
    }


@pytest.mark.parametrize("status", ["fail", "blocked"])
def test_failed_or_unavailable_required_case_does_not_pass(plan, status):
    actual = results(plan)
    actual["cases"] = [
        {"id": "contract-test", "status": status, "reason": "Observed failure or missing resource"}
    ]
    assert check(plan, actual)["status"] == status


def test_unselected_optional_experiment_is_not_a_gate(plan):
    plan["obligations"].append(
        {"id": "capacity", "description": "Unchanged capacity experiment", "required": False}
    )
    optional = copy.deepcopy(plan["cases"][0])
    optional.update(id="optional", covers=["capacity"], selected=False, reason="Outside this diff")
    plan["cases"].append(optional)
    actual = results(plan)
    actual["cases"].append(
        {"id": "optional", "status": "not_selected", "reason": "Outside this diff"}
    )
    assert check(plan, actual)["status"] == "pass"
    actual["cases"][0]["status"] = "not_selected"
    with pytest.raises(ValueError, match="disagrees"):
        check(plan, actual)


def test_duplicate_and_unknown_result_ids_are_rejected(plan):
    actual = results(plan)
    actual["cases"].append(copy.deepcopy(actual["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        check(plan, actual)
    actual["cases"][1]["id"] = "unknown"
    with pytest.raises(ValueError, match="unknown result"):
        check(plan, actual)


def trace(overlap=True):
    steps = [
        ("a", "submit"),
        ("b", "submit"),
        ("a", "ready"),
        ("a", "optimize_start"),
        ("a", "optimize_end"),
        ("b", "ready"),
        ("b", "optimize_start"),
        ("b", "optimize_end"),
    ]
    if not overlap:
        steps.insert(3, steps.pop(5))
    return [{"unit": unit, "event": event} for unit, event in steps]


def test_overlap_requires_event_order_not_elapsed_time(plan):
    plan["cases"][0]["oracle"].update(kind="overlap_order", reward="constant")
    actual = results(plan)
    observed = actual["cases"][0]["observations"]
    observed["overlap_seconds"] = 5.0
    with pytest.raises(ValueError, match="ordered events"):
        check(plan, actual)
    observed["events"] = trace(False)
    with pytest.raises(ValueError, match="never started"):
        check(plan, actual)
    observed["events"] = trace()
    assert check(plan, actual)["status"] == "pass"
    observed["events"].pop()
    with pytest.raises(ValueError, match="no pending work"):
        check(plan, actual)


def test_controlled_overlap_does_not_claim_measured_performance(plan):
    plan["cases"][0]["oracle"]["kind"] = "overlap_timing"
    actual = results(plan)
    actual["cases"][0]["observations"].update(
        events=trace(), artificial_delay=True, overlap_seconds=1.0
    )
    with pytest.raises(ValueError, match="controlled delays"):
        check(plan, actual)


def test_constant_or_zero_gradient_fixture_does_not_prove_numerical_update(plan):
    plan["cases"][0]["oracle"].update(kind="numerical", reward="constant")
    with pytest.raises(ValueError, match="constant rewards"):
        validator.validate_plan(plan)
    plan["cases"][0]["oracle"]["reward"] = "varying"
    actual = results(plan)
    with pytest.raises(ValueError, match="elementwise"):
        check(plan, actual)
    actual["cases"][0]["observations"].update(
        elementwise_comparison=True, nonzero_gradients=True, varying_rewards=True
    )
    assert check(plan, actual)["status"] == "pass"


def test_source_artifacts_are_checked_against_git_not_only_result_echo(plan):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    plan.update(base_commit=commit, tested_commit=commit)
    content = subprocess.check_output(
        ["git", "show", f"{commit}:requirements/quality.txt"], cwd=ROOT
    )
    plan["artifacts"] = {"requirements/quality.txt": hashlib.sha256(content).hexdigest()}
    validator.verify_source(plan, ROOT)
    plan["artifacts"]["requirements/quality.txt"] = "0" * 64
    with pytest.raises(ValueError, match="differs from tested commit"):
        validator.verify_source(plan, ROOT)


def test_cli_distinguishes_structural_success_missing_evidence_and_real_source_identity(
    plan, tmp_path
):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    content = subprocess.check_output(
        ["git", "show", f"{commit}:requirements/quality.txt"], cwd=ROOT
    )
    plan.update(base_commit=commit, tested_commit=commit)
    plan["artifacts"] = {"requirements/quality.txt": hashlib.sha256(content).hexdigest()}
    plan_path, result_path = tmp_path / "plan.json", tmp_path / "results.json"
    plan_path.write_text(json.dumps(plan))
    command = [sys.executable, str(ROOT / "scripts/validate_validation_plan.py"), str(plan_path)]
    structural = subprocess.run(command, capture_output=True, text=True)
    assert structural.returncode == 0 and '"status": "plan_valid"' in structural.stdout
    actual = results(plan)
    result_path.write_text(json.dumps(actual))
    command += ["--results", str(result_path)]
    assert subprocess.run(command, capture_output=True).returncode == 2
    command += ["--expected-commit", commit]
    assert subprocess.run(command, capture_output=True).returncode == 0
    actual["cases"] = []
    result_path.write_text(json.dumps(actual))
    blocked = subprocess.run(command, capture_output=True, text=True)
    assert blocked.returncode == 1 and '"status": "blocked"' in blocked.stdout
    plan["artifacts"]["requirements/quality.txt"] = "0" * 64
    plan_path.write_text(json.dumps(plan))
    assert subprocess.run(command, capture_output=True).returncode == 2
