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

"""Merge provenance and corruption controls; synthetic results claim no GPU execution."""

import copy
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import validation_merge as merge
import validation_plan as planner
from test_validation_acceptance import result, scenario


@pytest.fixture
def merged(scenario, monkeypatch):
    root, _, previous, baseline, *_ = scenario

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    head = git(
        "commit-tree",
        git("rev-parse", previous["tested_commit"] + "^{tree}"),
        "-p",
        previous["base_commit"],
        "-m",
        "squash PR",
    )
    pr = {
        "number": 1,
        "merged": True,
        "merged_at": "2026-11-10",
        "merge_commit_sha": head,
        "base": {"sha": previous["base_commit"], "repo": {"full_name": "test/repository"}},
        "head": {"sha": previous["tested_commit"], "repo": {"full_name": "test/repository"}},
    }
    plan = planner.generate(root, previous["base_commit"], head, [], [])
    cpu = result(plan, root, {"cpu"})
    summary, current = planner.report(plan, [cpu], head, root=root)
    original = json.loads(baseline.read_text())
    document = original["results"]
    logs = {}
    for row in document["cases"]:
        row["evidence"] = {
            ".scratch/validation/logs/synthetic.log": next(iter(row["evidence"].values()))
        }
    logs["logs/synthetic.log"] = next(iter(original["files"].values())).encode()
    old_report, _ = planner.report(previous, [document], previous["tested_commit"], root=root)
    files = {
        "plan.json": json.dumps(previous).encode(),
        "cpu.json": json.dumps(document).encode(),
        "report.json": json.dumps(old_report).encode(),
        **logs,
    }

    def api(repo, suffix):
        assert repo == "test/repository"
        return [pr] if "/pulls?" in suffix else pr

    monkeypatch.setattr(merge, "_json", api)
    monkeypatch.setattr(
        merge, "fetch_report", lambda *args: (files, {"run_id": 123, "artifact_id": 456})
    )
    monkeypatch.setattr(planner, "FRESH_REUSE_CASES", {"cpu"})
    return root, plan, summary, current, pr, files


def decide(data):
    return merge.decide(data[0], "test/repository", *data[1:4])


def test_squash_acceptance_preserves_original_results(merged):
    before = copy.deepcopy(merged[1:4])
    receipt = decide(merged)
    assert receipt["status"] == "accepted_merged"
    assert receipt["qualification_status"] == "blocked"
    assert receipt["transitions"][0]["source_head"] == merged[4]["head"]["sha"]
    assert receipt["transitions"][0]["source_head"] != receipt["tested_commit"]
    assert merged[2]["cases"]["broad-gpu"] == "blocked"
    assert merged[1:4] == before


@pytest.mark.parametrize("side", ["base", "head"])
def test_rejects_merge_conflict_or_rebased_base(merged, side):
    merged[4][side]["sha"] = merged[4]["head" if side == "base" else "base"]["sha"]
    with pytest.raises(ValueError, match="tree differs"):
        decide(merged)


@pytest.mark.parametrize("mutation", ["unmerged", "wrong_merge", "fork"])
def test_rejects_unrelated_pr(merged, mutation):
    pr = merged[4]
    if mutation == "unmerged":
        pr["merged"] = False
    elif mutation == "wrong_merge":
        pr["merge_commit_sha"] = "f" * 40
    else:
        pr["head"]["repo"]["full_name"] = "someone/fork"
    with pytest.raises(ValueError, match="merged.*PR"):
        decide(merged)


@pytest.mark.parametrize("mutation", ["plan", "log", "report", "missing"])
def test_rejects_corrupt_source_report(merged, mutation):
    files = merged[5]
    if mutation == "plan":
        plan = json.loads(files["plan.json"])
        plan["base_commit"] = "f" * 40
        files["plan.json"] = json.dumps(plan).encode()
    elif mutation == "log":
        files["logs/synthetic.log"] = b"altered assertion"
    elif mutation == "report":
        report = json.loads(files["report.json"])
        report["status"] = "blocked"
        files["report.json"] = json.dumps(report).encode()
    else:
        del files["report.json"]
    with pytest.raises((ValueError, KeyError)):
        decide(merged)


def test_current_failure_is_not_hidden(merged):
    merged[3]["cases"][0]["status"] = "fail"
    merged[3]["cases"][0]["reason"] = "Synthetic failure control"
    merged[3]["cases"][0]["observations"]["exit_code"] = 1
    summary, _ = planner.report(merged[1], [merged[3]], merged[1]["tested_commit"], root=merged[0])
    with pytest.raises(ValueError, match="failures"):
        merge.decide(merged[0], "test/repository", merged[1], summary, merged[3])


def test_requires_fresh_cpu(merged):
    merged[3]["cases"][0]["source_commit"] = merged[4]["head"]["sha"]
    with pytest.raises(ValueError, match="cannot be inherited"):
        decide(merged)


def test_missing_transition_blocks_complete_range(merged, monkeypatch):
    monkeypatch.setattr(merge, "_json", lambda *args: [])
    with pytest.raises(ValueError, match="Missing/ambiguous merged PR"):
        decide(merged)


def test_original_blocked_report_cannot_be_promoted(merged):
    files = merged[5]
    document = json.loads(files["cpu.json"])
    document["cases"] = [row for row in document["cases"] if row["id"] == "cpu"]
    files["cpu.json"] = json.dumps(document).encode()
    source = json.loads(files["plan.json"])
    summary, _ = planner.report(source, [document], source["tested_commit"], root=merged[0])
    files["report.json"] = json.dumps(summary).encode()
    with pytest.raises(ValueError, match="not accepted"):
        decide(merged)


def archive(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as output:
        for name, content in entries:
            output.writestr(name, content)
    return buf.getvalue()


@pytest.mark.parametrize("name", ["../plan.json", "/plan.json", "a/../plan.json", "./plan.json"])
def test_rejects_unsafe_archive_names(name):
    with pytest.raises(ValueError, match="Unsafe"):
        merge.read_archive(archive([(name, "{}")]))


def test_archive_size_and_missing_members(monkeypatch):
    with pytest.raises(ValueError, match="Incomplete"):
        merge.read_archive(archive([("plan.json", "{}")]))
    monkeypatch.setattr(merge, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="transfer limit"):
        merge.read_archive(b"xx")


def test_fetch_refuses_old_green_run_after_latest_failure(monkeypatch):
    runs = [
        dict(
            id=i,
            path=".github/workflows/quality.yml",
            conclusion=status,
            status="completed",
            head_sha="a" * 40,
            event="pull_request",
        )
        for i, status in [(1, "success"), (2, "failure")]
    ]
    monkeypatch.setattr(merge, "_json", lambda *args: {"workflow_runs": runs})
    with pytest.raises(ValueError, match="not successful"):
        merge.fetch_report("test/repository", {"number": 1, "head": {"sha": "a" * 40}})


@pytest.mark.parametrize("field", ["explicit_cases", "explicit_obligations"])
def test_extra_reviewer_requirements_need_their_own_evidence(merged, field):
    merged[1]["selection"][field] = ["extra-proof"]
    with pytest.raises(ValueError, match="extra reviewer obligations"):
        decide(merged)


def test_current_log_tampering_is_rejected(merged):
    path = next(iter(merged[3]["cases"][0]["evidence"]))
    (merged[0] / path).write_text("changed after recording")
    with pytest.raises(ValueError, match="content mismatch"):
        decide(merged)


def test_rejects_duplicate_archive_members():
    with pytest.warns(UserWarning, match="Duplicate"):
        raw = archive([("plan.json", "{}"), ("plan.json", "{}")])
    with pytest.raises(ValueError, match="duplicate"):
        merge.read_archive(raw)


@pytest.mark.parametrize("mutation", ["expired", "wrong_attempt", "wrong_name", "oversize"])
def test_artifact_must_belong_to_latest_successful_attempt(monkeypatch, mutation):
    run = dict(
        id=10,
        run_attempt=2,
        path=".github/workflows/quality.yml",
        event="pull_request",
        head_sha="a" * 40,
        status="completed",
        conclusion="success",
    )
    item = dict(id=11, name="validation-report-10-2", expired=False, size_in_bytes=10)
    if mutation == "expired":
        item["expired"] = True
    elif mutation == "wrong_attempt":
        item["name"] = "validation-report-10-1"
    elif mutation == "wrong_name":
        item["name"] = "untrusted-upload"
    else:
        item["size_in_bytes"] = merge.MAX_ARCHIVE_BYTES + 1

    def api(repo, suffix):
        return {"artifacts": [item]} if "/artifacts?" in suffix else {"workflow_runs": [run]}

    monkeypatch.setattr(merge, "_json", api)
    with pytest.raises(ValueError, match="report|limit"):
        merge.fetch_report("test/repository", {"number": 1, "head": {"sha": "a" * 40}})


def test_baseline_cache_retains_target_delta_and_local_file_checks(scenario, monkeypatch):
    import validation_acceptance as acceptance

    root, plan, _, bundle, manifest, *_ = scenario
    baseline = manifest["baseline"]
    monkeypatch.setattr(
        acceptance,
        "_api",
        lambda *args: json.dumps(
            {"id": baseline["artifact_id"], "workflow_run": {"id": baseline["run_id"]}}
        ).encode(),
    )
    reads = []

    def download(*args, **kwargs):
        reads.append(True)
        return bundle.read_bytes()

    monkeypatch.setattr(acceptance, "_download", download)
    cache = {}

    def verify(delta, **kwargs):
        return acceptance._verify_baseline(
            root,
            plan,
            baseline,
            delta,
            repository="test/repository",
            baseline_cache=cache,
            **kwargs,
        )

    first = verify(manifest["delta"])
    assert verify(manifest["delta"]) == first
    assert len(reads) == 1
    with pytest.raises(ValueError, match="source delta differs"):
        verify([])
    bundle.write_text("{}")
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        verify(manifest["delta"], baseline_bundle=bundle)


@pytest.mark.parametrize("complete", [False, True])
def test_report_cli_keeps_qualification_and_merge_acceptance_separate(
    merged, monkeypatch, complete
):
    root, plan, _, current, *_ = merged
    if complete:
        current = result(plan, root, {"cpu", "broad-gpu"})

        def unexpected(*args):
            pytest.fail("A complete current pass must not require historical artifacts")

        monkeypatch.setattr(merge, "decide", unexpected)
    folder = root / ".scratch/cli"
    folder.mkdir()
    for name, value in [("plan", plan), ("cpu", current)]:
        (folder / (name + ".json")).write_text(json.dumps(value))
    monkeypatch.setattr(planner, "ROOT", root)
    monkeypatch.setattr(planner, "verify_checkout", lambda *args: None)
    output = folder / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation_plan.py",
            "report",
            "--plan",
            str(folder / "plan.json"),
            "--results",
            str(folder / "cpu.json"),
            "--expected-commit",
            plan["tested_commit"],
            "--merged-repository",
            "test/repository",
            "--output",
            str(output),
        ],
    )
    assert planner.main() == 0
    report = json.loads(output.read_text())
    if complete:
        assert report["status"] == "pass" and "merge_acceptance" not in report
    else:
        assert report["status"] == "blocked"
        assert report["merge_acceptance"]["status"] == "accepted_merged"
