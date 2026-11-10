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

"""Verify merged PR transitions without relabeling historical executions.

Each first-parent transition must have exactly the reviewed PR's base and head
source trees. A successful original PR gate supplies its authenticated report
archive; plans, results, logs and incremental reviews are revalidated locally.
This proves accepted transitions, not a new all-model execution on main.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import agent_diff
import validate_validation_plan as validator
import validation_acceptance as acceptance
import validation_evidence as transport
import validation_plan as planner

MAX_COMMITS = 32
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_CONTENT_BYTES = 32 * 1024 * 1024
require = acceptance.require


def _json(repository: str, suffix: str) -> dict | list:
    return json.loads(acceptance._api(repository, suffix))


def _ensure_commit(root: Path, commit: str) -> None:
    validator.digest(commit, 40, "PR source commit")
    try:
        agent_diff.commit(root, commit)
    except subprocess.CalledProcessError:
        agent_diff.git(root, "fetch", "--no-tags", "origin", commit)


def _tree(root: Path, commit: str) -> str:
    return agent_diff.git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip()


def verify_transition(root: Path, repository: str, parent: str, commit: str, pr: dict) -> None:
    """Require a merged same-repository PR and exact before/after tree equality."""
    require(
        pr.get("merged") is True
        and pr.get("merge_commit_sha") == commit
        and pr["base"]["repo"]["full_name"] == repository
        and pr["head"]["repo"]["full_name"] == repository,
        "Transition is not a merged same-repository PR",
    )
    for side in ("base", "head"):
        _ensure_commit(root, pr[side]["sha"])
    require(
        _tree(root, parent) == _tree(root, pr["base"]["sha"]),
        "PR base tree differs from merge parent",
    )
    require(
        _tree(root, commit) == _tree(root, pr["head"]["sha"]),
        "PR head tree differs from merged tree",
    )


def read_archive(raw: bytes) -> dict[str, bytes]:
    """Read bounded report members in memory; never extract arbitrary archive paths."""
    require(len(raw) <= MAX_ARCHIVE_BYTES, "Merge report archive exceeds transfer limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= 1024, "Invalid merge report member count")
        require(
            sum(item.file_size for item in entries) <= MAX_CONTENT_BYTES,
            "Merge report exceeds decoded limit",
        )
        files = {}
        for item in entries:
            path = PurePosixPath(item.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and path.as_posix() == item.filename
                and not item.is_dir()
                and item.filename not in files
                and (item.external_attr >> 16) & 0o170000 in (0, 0o100000),
                "Unsafe or duplicate merge report member",
            )
            files[item.filename] = archive.read(item)
        require(
            {"plan.json", "cpu.json", "report.json"} <= files.keys(),
            "Incomplete merge report archive",
        )
        return files


def fetch_report(repository: str, pr: dict) -> tuple[dict[str, bytes], dict]:
    """Use the latest PR run/attempt, never an older green run hiding a failure."""
    head = pr["head"]["sha"]
    runs = _json(repository, f"actions/runs?head_sha={head}&event=pull_request&per_page=100")[
        "workflow_runs"
    ]
    candidates = [run for run in runs if run.get("path") == ".github/workflows/quality.yml"]
    require(bool(candidates), f"No PR quality run for #{pr['number']}")
    run = max(candidates, key=lambda item: item["id"])
    require(
        run.get("event") == "pull_request"
        and run.get("head_sha") == head
        and run.get("status") == "completed"
        and run.get("conclusion") == "success",
        f"Latest PR quality run is not successful for #{pr['number']}",
    )
    name = f"validation-report-{run['id']}-{run['run_attempt']}"
    artifacts = _json(repository, f"actions/runs/{run['id']}/artifacts?per_page=100")["artifacts"]
    matches = [item for item in artifacts if item.get("name") == name and not item.get("expired")]
    require(len(matches) == 1, f"Missing/ambiguous unexpired PR report for #{pr['number']}")
    artifact = matches[0]
    require(
        artifact.get("size_in_bytes", MAX_ARCHIVE_BYTES + 1) <= MAX_ARCHIVE_BYTES,
        "Merge report artifact exceeds limit",
    )
    raw = acceptance._api(repository, f"actions/artifacts/{artifact['id']}/zip")
    return read_archive(raw), {
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "artifact_id": artifact["id"],
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify_report(
    root: Path,
    repository: str,
    pr: dict,
    files: dict[str, bytes],
    *,
    baseline_cache: dict | None = None,
) -> dict:
    """Recompute the original gate, including authenticated historical baselines."""
    plan = json.loads(files["plan.json"])
    require(
        plan["base_commit"] == pr["base"]["sha"] and plan["tested_commit"] == pr["head"]["sha"],
        "PR report base/head mismatch",
    )
    planner.verify_generated_plan(root, plan)
    validator.verify_source(plan, root)
    cpu = json.loads(files["cpu.json"])
    logs = {}
    prefix = ".scratch/validation/"
    for row in cpu["cases"]:
        for path in row.get("evidence", {}):
            require(path.startswith(prefix), "CPU report log outside validation directory")
            logs[path] = files[path[len(prefix) :]].decode()
    cpu_bundle = {
        "schema_version": 1,
        "base_commit": plan["base_commit"],
        "tested_commit": plan["tested_commit"],
        "results": cpu,
        "files": logs,
    }
    # CI CPU logs can exceed dispatch limits; verify hashes before copying only their
    # content-addressed bytes, using the same decoder as manually dispatched bundles.
    decoded = transport._canonical_bytes(cpu_bundle)
    envelope = {
        "schema_version": 2,
        "base_commit": plan["base_commit"],
        "tested_commit": plan["tested_commit"],
        **transport.compressed_payload(decoded),
    }
    documents = [transport.import_bundle(root, plan, envelope)]
    inherited = []
    if "bundle.json" in files:
        bundle = json.loads(files["bundle.json"])
        original = planner.source_plan(root, plan, bundle)
        results = transport.import_bundle(root, original, bundle)
        if original["tested_commit"] == plan["tested_commit"]:
            documents.append(results)
        else:
            inherited.append((original, results))
    summary, combined = planner.report(
        plan, documents, plan["tested_commit"], inherited=inherited, root=root
    )
    if "acceptance.json" in files:
        summary["acceptance"] = acceptance.decide(
            root,
            plan,
            summary,
            combined,
            acceptance.decode(json.loads(files["acceptance.json"])),
            repository=repository,
            baseline_cache=baseline_cache,
        )
    require(
        summary == json.loads(files["report.json"]),
        "Recomputed PR report differs from uploaded report",
    )
    require(
        summary["status"] == "pass"
        or summary.get("acceptance", {}).get("status") == "accepted_incremental",
        "Original PR evidence is not accepted",
    )
    return {
        "source_base": plan["base_commit"],
        "source_head": plan["tested_commit"],
        "source_plan_sha256": validator.plan_sha256(plan),
        "qualification_status": summary["status"],
        "acceptance": summary.get("acceptance"),
    }


def decide(root: Path, repository: str, plan: dict, summary: dict, current: dict) -> dict:
    """Accept a complete merged transition chain while preserving current test status."""
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository), "Invalid repository")
    require(
        not plan["selection"]["explicit_cases"] and not plan["selection"]["explicit_obligations"],
        "Merge acceptance does not discharge extra reviewer obligations",
    )
    planner.verify_generated_plan(root, plan)
    validator.verify_source(plan, root)
    transport.verify_files(root, current)
    actual, _ = planner.report(plan, [current], plan["tested_commit"], root=root)
    require(
        all(
            actual[key] == summary[key]
            for key in ("status", "cases", "tested_commit", "plan_sha256")
        ),
        "Current merge report/result mismatch",
    )
    require(
        not plan["selection"]["uncovered_obligations"],
        "Merge acceptance rejects uncovered selection",
    )
    require(
        "fail" not in summary["cases"].values(), "Current failures cannot be accepted after merge"
    )
    for name in planner.FRESH_REUSE_CASES:
        require(summary["cases"].get(name) == "pass", f"Merge acceptance requires fresh {name}")
        row = next(row for row in current["cases"] if row["id"] == name)
        require(
            row.get("source_commit", plan["tested_commit"]) == plan["tested_commit"],
            "Current CPU evidence cannot be inherited",
        )
    base, head = plan["base_commit"], plan["tested_commit"]
    commits = (
        agent_diff.git(root, "rev-list", "--first-parent", "--reverse", f"{base}..{head}")
        .decode()
        .splitlines()
    )
    require(0 < len(commits) <= MAX_COMMITS, "Merge chain must contain 1..32 transitions")
    parent = base
    receipts = []
    baseline_cache: dict = {}
    for commit in commits:
        require(
            agent_diff.commit(root, f"{commit}^1") == parent,
            "Comparison base is not on the first-parent merge chain",
        )
        pulls = _json(repository, f"commits/{commit}/pulls?per_page=100")
        matches = [
            pr for pr in pulls if pr.get("merged_at") and pr.get("merge_commit_sha") == commit
        ]
        require(len(matches) == 1, f"Missing/ambiguous merged PR for {commit}")
        pr = _json(repository, f"pulls/{matches[0]['number']}")
        verify_transition(root, repository, parent, commit, pr)
        files, artifact = fetch_report(repository, pr)
        source = verify_report(root, repository, pr, files, baseline_cache=baseline_cache)
        receipts.append(
            {
                "merge_commit": commit,
                "parent_commit": parent,
                "pull_request": pr["number"],
                **artifact,
                **source,
            }
        )
        parent = commit
    return {
        "status": "accepted_merged",
        "qualification_status": summary["status"],
        "base_commit": base,
        "tested_commit": head,
        "plan_sha256": validator.plan_sha256(plan),
        "repository": repository,
        "transitions": receipts,
        "claim": "Accepted merged PR transitions with identical source trees; historical executions retain their original identities and are not current-head passes.",
    }
