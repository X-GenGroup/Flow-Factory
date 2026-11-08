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

"""Synthetic evidence controls; no test in this file claims a GPU execution."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import validation_acceptance as acceptance
import validation_evidence as transport
import validation_plan as planner
from test_validation_planner import case


def result(plan, root, names):
    document = planner.empty_results(plan)
    document["environment"] = {
        "python": "test-only",
        "hardware": "synthetic unit control",
        "packages": {"pytest": "test"},
    }
    document["cases"] = []
    log = root / ".scratch/synthetic.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("SYNTHETIC validator input, not an executed GPU claim\n")
    for row in plan["cases"]:
        if row["id"] in names:
            document["cases"].append(
                {
                    "id": row["id"],
                    "status": "pass",
                    "backend": row["backend"],
                    "topology": row["topology"],
                    "command": row["command"],
                    "runtime": {
                        "world_size": 1,
                        "ranks_completed": [0],
                        "device": row["backend"],
                        "distributed_type": "NO",
                    },
                    "observations": {"exit_code": 0},
                    "evidence": {
                        ".scratch/synthetic.log": hashlib.sha256(log.read_bytes()).hexdigest()
                    },
                }
            )
    return document


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    def write(path, content):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def commit(message):
        git("add", ".")
        git("commit", "-qm", message)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    git("config", "user.name", "Acceptance Unit Test")
    git("config", "user.email", "test@example.invalid")
    monkeypatch.setattr(
        planner,
        "POLICY_INPUTS",
        (planner.CATALOG, planner.POLICY, ".agents/harness/risk_rules.yaml"),
    )
    cpu = case("cpu", ["repository.consistency", "model.cpu"])
    old = case("broad-gpu", ["model.gpu"], backend="cuda")
    fresh = case("incremental-gpu", ["model.incremental"], backend="cuda")
    for row in (old, fresh):
        row["resources"]["gpus"] = 1
        row["qa"] = "Q1"
    for row in (cpu, old, fresh):
        row["artifacts"] = ["src/model.py", "tests/probe.py"]
    catalog = {"schema_version": 1, "cases": [cpu, old]}
    policy = {
        "schema_version": 1,
        "static_paths": ["*.md"],
        "owners": [
            {
                "paths": ["src/**", "config/**", "tests/**"],
                "requires": ["model.cpu", "model.gpu"],
                "reason": "Synthetic owner.",
            }
        ],
        "interactions": [],
    }
    write(planner.CATALOG, yaml.safe_dump(catalog))
    write(planner.POLICY, yaml.safe_dump(policy))
    write(".agents/harness/risk_rules.yaml", (ROOT / ".agents/harness/risk_rules.yaml").read_text())
    write(
        ".agents/harness/test_routes.yaml", (ROOT / ".agents/harness/test_routes.yaml").read_text()
    )
    write("src/model.py", "def run(): return 0\n")
    write("tests/probe.py", "def test_probe(): assert True\n")
    write(".gitignore", ".scratch/\n")
    base = commit("base")
    write("src/model.py", "def run(): return 1\n")
    source = commit("historical model")
    previous = planner.generate(tmp_path, base, source, [], [])
    original = result(previous, tmp_path, {"cpu", "broad-gpu"})
    baseline = tmp_path / ".scratch/baseline.json"
    baseline.write_text(json.dumps(transport.make_bundle(tmp_path, previous, original)))
    # A real nonancestor restack with a different base, not an ancestry mock.
    git("checkout", "-q", "--detach", base)
    write("parent.md", "New parent.\n")
    target_base = commit("new parent")
    write("src/model.py", "def run(): return 2\n")
    catalog["cases"].append(fresh)
    policy["owners"][0]["requires"].append("model.incremental")
    write(planner.CATALOG, yaml.safe_dump(catalog))
    write(planner.POLICY, yaml.safe_dump(policy))
    head = commit("incremental change")
    plan = planner.generate(tmp_path, target_base, head, [], [])
    current = result(plan, tmp_path, {"cpu", "incremental-gpu"})
    summary, _ = planner.report(plan, [current], head, root=tmp_path)
    manifest = {
        "schema_version": 1,
        "kind": "reviewed_incremental",
        "repository": "test/repository",
        "base_commit": target_base,
        "tested_commit": head,
        "plan_sha256": planner.evidence.plan_sha256(plan),
        "baseline": {
            "base_commit": base,
            "tested_commit": source,
            "plan_sha256": planner.evidence.plan_sha256(previous),
            "bundle_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "artifact_id": 12,
            "run_id": 34,
        },
        "review": {
            "decision": "approved_incremental",
            "scope": "model_tp_extension",
            "reviewer": "unit fixture",
            "authorization_quote": "TEST ONLY: explicit incremental scope",
            "reason": "TEST ONLY: synthetic changed model",
        },
        "delta": acceptance.delta(tmp_path, source, head),
        "required_fresh_cases": ["cpu", "incremental-gpu"],
        "historical_cases": ["broad-gpu"],
    }
    return tmp_path, plan, previous, baseline, manifest, current, summary


def decide(data):
    root, plan, _, baseline, manifest, _, summary = data
    return acceptance.decide(
        root,
        plan,
        summary,
        data[5],
        manifest,
        repository="test/repository",
        baseline_bundle=baseline,
    )


def test_nonancestor_acceptance_does_not_masquerade_as_current_pass(scenario):
    root, plan, previous, baseline, manifest, current, summary = scenario
    before = baseline.read_bytes(), copy.deepcopy(current), copy.deepcopy(summary)
    with pytest.raises(subprocess.CalledProcessError):
        planner.reuse_changes(root, previous["tested_commit"], plan["tested_commit"])
    accepted = decide(scenario)
    assert accepted["status"] == "accepted_incremental"
    assert accepted["historical_source"]["tested_commit"] == previous["tested_commit"]
    assert accepted["historical_cases_not_rerun"] == ["broad-gpu"]
    assert summary["status"] == "blocked" and summary["cases"]["broad-gpu"] == "blocked"
    assert before == (baseline.read_bytes(), current, summary)


@pytest.mark.parametrize("field", ["base_commit", "tested_commit", "plan_sha256"])
def test_rejects_stale_target_binding(scenario, field):
    scenario[4][field] = "f" * len(scenario[4][field])
    with pytest.raises(ValueError, match="mismatch"):
        decide(scenario)


@pytest.mark.parametrize("mutation", ["missing_path", "wrong_blob", "wrong_mode", "extra_path"])
def test_rejects_incomplete_or_forged_reviewed_delta(scenario, mutation):
    delta = scenario[4]["delta"]
    if mutation == "missing_path":
        delta.pop()
    elif mutation == "extra_path":
        delta.append({"path": "not-changed"})
    else:
        delta[0]["after"]["sha256" if mutation == "wrong_blob" else "mode"] = "invalid"
    with pytest.raises(ValueError, match="source delta differs"):
        decide(scenario)


@pytest.mark.parametrize(
    "mutation",
    [
        "review_rejected",
        "scope_unknown",
        "missing_quote",
        "missing_fresh",
        "missing_cpu",
        "new_as_history",
        "overlap",
        "unlisted",
        "unknown",
    ],
)
def test_review_and_complete_dispositions_are_mandatory(scenario, mutation):
    document = scenario[4]
    if mutation == "review_rejected":
        document["review"]["decision"] = "rejected"
    elif mutation == "scope_unknown":
        document["review"]["scope"] = "anything"
    elif mutation == "missing_quote":
        document["review"]["authorization_quote"] = ""
    elif mutation == "missing_fresh":
        scenario[6]["cases"]["incremental-gpu"] = "blocked"
    elif mutation == "missing_cpu":
        scenario[6]["cases"]["cpu"] = "blocked"
    elif mutation == "new_as_history":
        document["required_fresh_cases"].remove("incremental-gpu")
        document["historical_cases"].append("incremental-gpu")
    elif mutation == "overlap":
        document["historical_cases"].append("cpu")
    elif mutation == "unlisted":
        document["historical_cases"] = []
    else:
        document["historical_cases"].append("unknown")
    with pytest.raises(ValueError):
        decide(scenario)


def test_current_failure_cannot_be_overridden(scenario):
    scenario[6]["cases"]["broad-gpu"] = "fail"
    with pytest.raises(ValueError, match="Current failures"):
        decide(scenario)


@pytest.mark.parametrize("mutation", ["missing", "tamper", "failed", "incomplete", "plan", "log"])
def test_original_baseline_must_be_complete_and_hash_verified(scenario, mutation):
    baseline, manifest = scenario[3], scenario[4]
    if mutation == "missing":
        baseline.unlink()
    elif mutation == "plan":
        manifest["baseline"]["plan_sha256"] = "f" * 64
    elif mutation == "tamper":
        baseline.write_text(baseline.read_text() + " ")
    else:
        value = json.loads(baseline.read_text())
        if mutation == "failed":
            value["results"]["cases"][1] = {
                "id": "broad-gpu",
                "status": "fail",
                "reason": "unit failure",
            }
        elif mutation == "incomplete":
            value["results"]["cases"].pop()
        else:
            value["files"][".scratch/synthetic.log"] = "tampered log"
        baseline.write_text(json.dumps(value))
        manifest["baseline"]["bundle_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    with pytest.raises((ValueError, FileNotFoundError)):
        decide(scenario)


@pytest.mark.parametrize("compressed", [False, True])
def test_report_cli_returns_separate_acceptance_without_rewriting_qualification(
    scenario, monkeypatch, compressed
):
    root, plan, _, baseline, manifest, current, _ = scenario
    if compressed:
        manifest["review"]["reason"] = "Synthetic complete review. " * 4000
    wire = acceptance.pack(manifest)
    paths = {
        name: root / f".scratch/{name}.json" for name in ("plan", "results", "acceptance", "report")
    }
    for name, value in (("plan", plan), ("results", current), ("acceptance", wire)):
        paths[name].write_text(json.dumps(value))
    monkeypatch.setattr(planner, "ROOT", root)
    args = [
        "validation_plan.py",
        "report",
        "--plan",
        str(paths["plan"]),
        "--results",
        str(paths["results"]),
        "--expected-commit",
        plan["tested_commit"],
        "--output",
        str(paths["report"]),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert planner.main() == 1
    strict = json.loads(paths["report"].read_text())
    monkeypatch.setattr(
        sys,
        "argv",
        args
        + [
            "--acceptance",
            str(paths["acceptance"]),
            "--repository",
            "test/repository",
            "--baseline-bundle",
            str(baseline),
        ],
    )
    assert planner.main() == 0
    accepted = json.loads(paths["report"].read_text())
    assert accepted.pop("acceptance")["status"] == "accepted_incremental"
    assert accepted == strict


def remote_fixture(scenario, monkeypatch, mutation=None):
    _, _, previous, baseline, manifest, _, _ = scenario
    artifact = {
        "id": 12,
        "name": transport.artifact_name(previous),
        "expired": False,
        "size_in_bytes": 1000,
        "workflow_run": {"id": 34, "head_sha": previous["tested_commit"]},
    }
    run = {
        "event": "workflow_dispatch",
        "path": ".github/workflows/quality.yml",
        "conclusion": "success",
        "head_sha": previous["tested_commit"],
    }
    if mutation == "head":
        artifact["workflow_run"]["head_sha"] = "e" * 40
    elif mutation == "run":
        artifact["workflow_run"]["id"] = 35
    elif mutation == "event":
        run["event"] = "pull_request"
    elif mutation == "failed":
        run["conclusion"] = "failure"
    elif mutation == "name":
        artifact["name"] = "another-artifact"
    elif mutation == "expired":
        artifact["expired"] = True
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            "bundle.json" if mutation != "path" else "../bundle.json", baseline.read_bytes()
        )
    replies = {
        "actions/artifacts/12": json.dumps(artifact).encode(),
        "actions/runs/34": json.dumps(run).encode(),
        "actions/artifacts/12/zip": archive.getvalue(),
    }
    monkeypatch.setattr(acceptance, "_api", lambda repository, endpoint: replies[endpoint])


def test_remote_baseline_uses_original_exact_dispatch(scenario, monkeypatch):
    remote_fixture(scenario, monkeypatch)
    root, plan, _, _, manifest, _, summary = scenario
    assert (
        acceptance.decide(root, plan, summary, scenario[5], manifest, repository="test/repository")[
            "status"
        ]
        == "accepted_incremental"
    )


@pytest.mark.parametrize("mutation", ["head", "run", "event", "failed", "name", "expired", "path"])
def test_remote_baseline_rejects_untrusted_or_wrong_archive(scenario, monkeypatch, mutation):
    remote_fixture(scenario, monkeypatch, mutation)
    with pytest.raises(ValueError):
        acceptance.verify(scenario[0], scenario[1], scenario[4], repository="test/repository")


def test_ancestor_pass_cannot_satisfy_required_fresh_execution(scenario):
    scenario[5]["cases"][1]["source_commit"] = scenario[2]["tested_commit"]
    with pytest.raises(ValueError, match="cannot use ancestor"):
        decide(scenario)


def test_historical_generation_omits_only_absent_new_acceptance_consumer(scenario, monkeypatch):
    root, _, previous, _, _, _, _ = scenario
    inputs = planner.POLICY_INPUTS
    monkeypatch.setattr(planner, "POLICY_INPUTS", (*inputs, "scripts/validation_acceptance.py"))
    assert (
        planner.generate(root, previous["base_commit"], previous["tested_commit"], [], [])
        == previous
    )
    monkeypatch.setattr(planner, "POLICY_INPUTS", (*inputs, "scripts/missing_other_tool.py"))
    with pytest.raises(ValueError, match="Missing declared artifact"):
        planner.generate(root, previous["base_commit"], previous["tested_commit"], [], [])


def test_workflow_keeps_distinct_artifacts_and_requires_explicit_dispatch():
    # BaseLoader avoids interpreting GitHub's YAML `on` key as a boolean.
    workflow = yaml.load(
        (ROOT / ".github/workflows/quality.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]["workflow_dispatch"]["inputs"]) == {"bundle", "acceptance"}
    record = workflow["jobs"]["record-evidence"]
    assert record["if"] == "github.event_name == 'workflow_dispatch'"
    uploads = [
        step
        for step in record["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact")
    ]
    assert {row["with"]["path"] for row in uploads} == {
        ".scratch/recorded-evidence/bundle.json",
        ".scratch/recorded-acceptance/acceptance.json",
    }
    assert all("inputs." in row["if"] for row in uploads)
    steps = workflow["jobs"]["cpu-contracts"]["steps"]
    report = next(step for step in steps if "validation report" in step.get("run", ""))
    assert "--acceptance .scratch/validation/acceptance.json" in report["run"]
    assert "--results .scratch/validation/cpu.json" in report["run"]
    assert "if test -f .scratch/validation/acceptance.json" in report["run"]


@pytest.mark.parametrize("compressed", [False, True])
def test_acceptance_fetch_keeps_exact_target_and_dispatch_provenance(
    scenario, monkeypatch, compressed
):
    _, plan, _, _, manifest, _, _ = scenario
    if compressed:
        manifest["review"]["reason"] = "Synthetic complete review. " * 4000
    wire = acceptance.pack(manifest)
    name = acceptance.artifact_name(plan)
    artifact = {
        "id": 56,
        "name": name,
        "expired": False,
        "size_in_bytes": 1000,
        "created_at": "2026-11-08",
        "workflow_run": {"id": 78, "head_sha": plan["tested_commit"]},
    }
    run = {
        "event": "workflow_dispatch",
        "path": ".github/workflows/quality.yml",
        "conclusion": "success",
        "head_sha": plan["tested_commit"],
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("acceptance.json", json.dumps(wire))
    replies = {
        f"actions/artifacts?name={name}&per_page=100": json.dumps(
            {"artifacts": [artifact]}
        ).encode(),
        "actions/runs/78": json.dumps(run).encode(),
        "actions/artifacts/56/zip": archive.getvalue(),
    }
    monkeypatch.setattr(acceptance, "_api", lambda repository, endpoint: replies[endpoint])
    assert acceptance.fetch("test/repository", plan) == wire
    run["path"] = ".github/workflows/unrelated.yml"
    replies["actions/runs/78"] = json.dumps(run).encode()
    with pytest.raises(ValueError, match="evidence dispatch"):
        acceptance.fetch("test/repository", plan)


def test_acceptance_cli_uses_isolated_quality_python(monkeypatch):
    import quality

    calls = []
    monkeypatch.setattr(quality, "python", lambda: Path("/isolated-quality/python"))
    monkeypatch.setattr(quality, "run", lambda command: calls.append(command) or 0)
    monkeypatch.setattr(sys, "argv", ["quality.py", "acceptance", "fetch", "--plan", "plan.json"])
    assert quality.main() == 0
    assert calls == [
        [
            Path("/isolated-quality/python"),
            "scripts/validation_acceptance.py",
            "fetch",
            "--plan",
            "plan.json",
        ]
    ]
    workflow = (ROOT / ".github/workflows/quality.yml").read_text()
    assert "python3 scripts/validation_acceptance.py" not in workflow
    assert "'python3', 'scripts/validation_acceptance.py'" not in workflow


def test_quality_cpu_actually_executes_acceptance_regressions(monkeypatch):
    import quality

    calls = []
    monkeypatch.setattr(quality, "python", lambda: Path("/isolated-quality/python"))
    monkeypatch.setattr(quality, "run", lambda command: calls.append(command) or 0)
    monkeypatch.setattr(sys, "argv", ["quality.py", "cpu"])
    assert quality.main() == 0
    assert len(calls) == 1
    assert "tests/scripts/test_validation_acceptance.py" in calls[0]


@pytest.mark.parametrize("field", ["resources", "fixture_kind", "execution_capability"])
def test_expanded_historical_contract_requires_new_execution(scenario, field):
    root, plan, previous, baseline, manifest, _, _ = scenario
    path = root / planner.CATALOG
    catalog = yaml.safe_load(path.read_text())
    row = next(item for item in catalog["cases"] if item["id"] == "broad-gpu")
    if field == "resources":
        row[field]["environment"]["NEW_RUNTIME_OPTION"] = "1"
    elif field == "fixture_kind":
        row[field] = "model_structure"
    else:
        # This CUDA case accepts the label, so the historical comparison must reject it.
        row[field] = "changed-contract"
    path.write_text(yaml.safe_dump(catalog))
    subprocess.check_call(["git", "add", str(path)], cwd=root)
    subprocess.check_call(["git", "commit", "-qm", "expand historical case"], cwd=root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    changed = planner.generate(root, plan["base_commit"], head, [], [])
    manifest.update(
        tested_commit=head,
        plan_sha256=planner.evidence.plan_sha256(changed),
        delta=acceptance.delta(root, previous["tested_commit"], head),
    )
    current = result(changed, root, {"cpu", "incremental-gpu"})
    summary, _ = planner.report(changed, [current], head, root=root)
    with pytest.raises(ValueError, match="Historical case contract changed"):
        acceptance.decide(
            root,
            changed,
            summary,
            current,
            manifest,
            repository="test/repository",
            baseline_bundle=baseline,
        )


def test_external_only_baseline_keeps_missing_historical_cpu_explicit(scenario):
    # Published bundles commonly carry GPU/external proofs; CI owns quality CPU results.
    # Their absence here must not be relabeled as a historical complete-plan pass.
    path, manifest = scenario[3], scenario[4]
    bundle = json.loads(path.read_text())
    bundle["results"]["cases"] = [row for row in bundle["results"]["cases"] if row["id"] != "cpu"]
    path.write_text(json.dumps(bundle))
    manifest["baseline"]["bundle_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    accepted = decide(scenario)
    assert accepted["status"] == "accepted_incremental"
    assert accepted["historical_source"]["qualification_status"] == "blocked"
    assert accepted["historical_source"]["claimed_case_statuses"] == {"broad-gpu": "pass"}
    assert "cpu" in accepted["fresh_cases"]


@pytest.fixture
def supplemental_scenario(scenario):
    """Two real original Git sources supply disjoint synthetic historical GPU proofs."""
    root, plan, previous, primary, manifest, _, _ = scenario

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    def add_case():
        path = root / planner.CATALOG
        catalog = yaml.safe_load(path.read_text())
        added = copy.deepcopy(next(row for row in catalog["cases"] if row["id"] == "broad-gpu"))
        added.update(id="second-historical-gpu", covers=["model.second_history"])
        catalog["cases"].append(added)
        path.write_text(yaml.safe_dump(catalog))
        path = root / planner.POLICY
        policy = yaml.safe_load(path.read_text())
        policy["owners"][0]["requires"].append("model.second_history")
        path.write_text(yaml.safe_dump(policy))
        git("add", ".")
        git("commit", "-qm", "include second historical case")
        return git("rev-parse", "HEAD")

    git("checkout", "-q", "--detach", previous["tested_commit"])
    source = add_case()
    old_plan = planner.generate(root, previous["base_commit"], source, [], [])
    old_results = result(old_plan, root, {"second-historical-gpu"})
    secondary = root / ".scratch/secondary.json"
    secondary.write_text(json.dumps(transport.make_bundle(root, old_plan, old_results)))
    git("checkout", "-q", "--detach", plan["tested_commit"])
    target = add_case()
    target_plan = planner.generate(root, plan["base_commit"], target, [], [])
    manifest.update(
        tested_commit=target,
        plan_sha256=planner.evidence.plan_sha256(target_plan),
        delta=acceptance.delta(root, previous["tested_commit"], target),
    )
    manifest["historical_cases"].append("second-historical-gpu")
    manifest["supplemental_baselines"] = [
        {
            "baseline": {
                "base_commit": old_plan["base_commit"],
                "tested_commit": source,
                "plan_sha256": planner.evidence.plan_sha256(old_plan),
                "bundle_sha256": hashlib.sha256(secondary.read_bytes()).hexdigest(),
                "artifact_id": 56,
                "run_id": 78,
            },
            "delta": acceptance.delta(root, source, target),
        }
    ]
    current = result(target_plan, root, {"cpu", "incremental-gpu"})
    summary, _ = planner.report(target_plan, [current], target, root=root)
    return root, target_plan, primary, secondary, manifest, current, summary


def decide_supplemental(data):
    root, plan, primary, secondary, manifest, current, summary = data
    return acceptance.decide(
        root,
        plan,
        summary,
        current,
        manifest,
        repository="test/repository",
        baseline_bundle=primary,
        supplemental_bundles=[secondary],
    )


def test_two_baselines_cover_disjoint_cases_without_fabricating_current_passes(
    supplemental_scenario,
):
    data = supplemental_scenario
    before = [p.read_bytes() for p in data[2:4]]
    original_report = copy.deepcopy(data[6])
    approved = decide_supplemental(data)
    primary = data[4]["baseline"]["tested_commit"]
    secondary = data[4]["supplemental_baselines"][0]["baseline"]["tested_commit"]
    assert approved["historical_case_sources"] == {
        "broad-gpu": [primary],
        "second-historical-gpu": [secondary],
    }
    assert approved["historical_source"]["claimed_case_statuses"] == {"broad-gpu": "pass"}
    assert approved["supplemental_historical_sources"][0]["claimed_case_statuses"] == {
        "second-historical-gpu": "pass"
    }
    assert approved["supplemental_historical_sources"][0]["qualification_status"] == "blocked"
    assert approved["qualification_status"] == "blocked"
    assert [p.read_bytes() for p in data[2:4]] == before
    assert data[6] == original_report


def test_missing_supplemental_case_remains_rejected(supplemental_scenario):
    root, plan, primary, _, manifest, current, summary = supplemental_scenario
    manifest.pop("supplemental_baselines")
    with pytest.raises(ValueError, match="missing passing evidence"):
        acceptance.decide(
            root,
            plan,
            summary,
            current,
            manifest,
            repository="test/repository",
            baseline_bundle=primary,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "delta",
        "head",
        "bundle",
        "too_many",
        "duplicate",
        "extra_field",
        "failed_proof",
        "missing_proof",
    ],
)
def test_supplemental_sources_cannot_bypass_original_checks(supplemental_scenario, mutation):
    data = supplemental_scenario
    supplements = data[4]["supplemental_baselines"]
    item = supplements[0]
    if mutation == "delta":
        item["delta"] = []
    elif mutation == "head":
        item["baseline"]["tested_commit"] = data[1]["tested_commit"]
    elif mutation == "bundle":
        item["baseline"]["bundle_sha256"] = "f" * 64
    elif mutation == "too_many":
        supplements.extend([copy.deepcopy(item)] * 4)
    elif mutation == "duplicate":
        supplements[0] = {"baseline": data[4]["baseline"], "delta": data[4]["delta"]}
        data[3].write_bytes(data[2].read_bytes())
    elif mutation == "extra_field":
        item["assume_equivalent"] = True
    else:
        value = json.loads(data[3].read_text())
        value["results"]["cases"] = (
            [{"id": "second-historical-gpu", "status": "fail", "reason": "unit corruption"}]
            if mutation == "failed_proof"
            else []
        )
        data[3].write_text(json.dumps(value))
        item["baseline"]["bundle_sha256"] = hashlib.sha256(data[3].read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        decide_supplemental(data)


def test_offline_supplemental_file_count_is_exact(supplemental_scenario):
    root, plan, primary, _, manifest, current, summary = supplemental_scenario
    with pytest.raises(ValueError, match="bundle count"):
        acceptance.decide(
            root,
            plan,
            summary,
            current,
            manifest,
            repository="test/repository",
            baseline_bundle=primary,
            supplemental_bundles=[],
        )


def _large_review(scenario):
    manifest = copy.deepcopy(scenario[4])
    manifest["review"]["reason"] = "Synthetic full review retained without truncation. " * 4000
    return manifest


def test_compressed_review_preserves_every_field_and_verified_decision(scenario):
    root, plan, _, baseline, _, current, summary = scenario
    manifest = _large_review(scenario)
    wire = acceptance.pack(manifest)
    assert wire["schema_version"] == 2
    assert wire == acceptance.pack(manifest)
    assert wire["decoded_bytes"] > acceptance.MAX_BYTES
    path = root / ".scratch/acceptance-wire.json"
    planner.write_json(path, wire)
    assert path.stat().st_size <= acceptance.MAX_BYTES
    decoded = acceptance.load(path)
    assert decoded == manifest
    expected = acceptance.decide(
        root,
        plan,
        summary,
        current,
        manifest,
        repository="test/repository",
        baseline_bundle=baseline,
    )
    actual = acceptance.decide(
        root,
        plan,
        summary,
        current,
        decoded,
        repository="test/repository",
        baseline_bundle=baseline,
    )
    assert actual == expected
    assert acceptance.pack(scenario[4]) == scenario[4]
    decoded["delta"].pop()
    with pytest.raises(ValueError, match="source delta differs"):
        acceptance.verify(
            root,
            plan,
            acceptance.decode(acceptance.pack(decoded)),
            repository="test/repository",
            baseline_bundle=baseline,
        )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", 2.0, "envelope"),
        ("encoding", "zlib+base64", "envelope"),
        ("decoded_bytes", True, "invalid length"),
        ("decoded_bytes", 0, "invalid length"),
        ("decoded_bytes", acceptance.MAX_DECODED_BYTES + 1, "exceeds"),
        ("decoded_bytes", 7, "length mismatch"),
        ("decoded_sha256", "f" * 64, "hash mismatch"),
        ("payload", "invalid!", "base64"),
        ("payload", 7, "base64"),
        ("extra", "undeclared", "envelope"),
        ("base_commit", "f" * 40, "inner source"),
        ("tested_commit", "f" * 40, "inner source"),
        ("plan_sha256", "f" * 64, "inner source"),
        ("repository", "other/repository", "inner source"),
    ],
)
def test_compressed_review_rejects_invalid_envelope(scenario, field, value, error):
    wire = acceptance.pack(_large_review(scenario))
    wire[field] = value
    with pytest.raises(ValueError, match=error):
        acceptance.decode(wire)


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "trailing",
        "second-member",
        "bad-crc",
        "noncanonical-json",
        "nested",
        "source",
        "expansion",
    ],
)
def test_compressed_review_rejects_corrupt_payload_before_review(scenario, mutation):
    import base64
    import gzip

    manifest = _large_review(scenario)
    wire = acceptance.pack(manifest)
    compressed = base64.b64decode(wire["payload"])
    if mutation == "truncated":
        compressed = compressed[:-1]
    elif mutation == "trailing":
        compressed += b"trailing"
    elif mutation == "second-member":
        compressed += gzip.compress(b"second")
    elif mutation == "bad-crc":
        compressed = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    else:
        if mutation == "nested":
            manifest["schema_version"] = 2
        elif mutation == "source":
            manifest["tested_commit"] = "f" * 40
        raw = transport._canonical_bytes(manifest)
        if mutation == "noncanonical-json":
            raw = b" " + raw
        elif mutation == "expansion":
            raw = b"x" * (acceptance.MAX_DECODED_BYTES + 1)
        wire.update(transport.compressed_payload(raw))
        if mutation == "expansion":
            wire["decoded_bytes"] = acceptance.MAX_DECODED_BYTES
        compressed = base64.b64decode(wire["payload"])
    wire["payload"] = base64.b64encode(compressed).decode("ascii")
    with pytest.raises(ValueError):
        acceptance.decode(wire)


def test_review_wire_and_decoded_limits_are_independent(scenario):
    import random

    manifest = _large_review(scenario)
    manifest["review"]["reason"] = "x" * acceptance.MAX_DECODED_BYTES
    with pytest.raises(ValueError, match="exceeds 2 MiB"):
        acceptance.pack(manifest)
    generator = random.Random(19)
    manifest["review"]["reason"] = "".join(chr(generator.randrange(32, 127)) for _ in range(100000))
    with pytest.raises(ValueError, match="Compressed acceptance document exceeds 60 KB"):
        acceptance.pack(manifest)
    path = scenario[0] / ".scratch/oversize-review.json"
    path.write_text("x" * (acceptance.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds 60 KB"):
        acceptance.load(path)


def test_pack_cli_verifies_full_review_before_writing(scenario, monkeypatch):
    root, plan, _, baseline, _, _, _ = scenario
    manifest = _large_review(scenario)
    source = root / ".scratch/decoded-review.json"
    output = root / ".scratch/packed-review.json"
    plan_path = root / ".scratch/plan.json"
    planner.write_json(source, manifest)
    planner.write_json(plan_path, plan)
    monkeypatch.setattr(planner, "ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation_acceptance.py",
            "pack",
            "--plan",
            str(plan_path),
            "--repository",
            "test/repository",
            "--acceptance",
            str(source),
            "--baseline-bundle",
            str(baseline),
            "--output",
            str(output),
        ],
    )
    assert acceptance.main() == 0
    assert acceptance.load(output) == manifest
    output.unlink()
    manifest["delta"].pop()
    planner.write_json(source, manifest)
    with pytest.raises(SystemExit) as error:
        acceptance.main()
    assert error.value.code == 2 and not output.exists()


def test_compact_wire_writer_preserves_near_limit_schema1_arrays(scenario):
    manifest = copy.deepcopy(scenario[4])
    # Transport-only synthetic delta: source verification is deliberately separate.
    manifest["delta"] = [
        dict(path=f"synthetic/source/{index}.py", before=None, after=None) for index in range(1000)
    ]
    while len(acceptance._wire_bytes(manifest)) > acceptance.MAX_BYTES:
        manifest["delta"].pop()
    assert len(acceptance._wire_bytes(manifest)) > acceptance.MAX_BYTES - 100
    assert transport._dispatch_size(manifest) > acceptance.MAX_BYTES
    assert acceptance.pack(manifest) == manifest
    path = scenario[0] / ".scratch/fetched-wire.json"
    acceptance.write_wire(path, manifest)
    assert path.stat().st_size <= acceptance.MAX_BYTES
    assert acceptance.load(path) == manifest
    path.unlink()
    manifest["review"]["reason"] += "x" * acceptance.MAX_BYTES
    with pytest.raises(ValueError, match="exceeds 60 KB"):
        acceptance.write_wire(path, manifest)
    assert not path.exists()
