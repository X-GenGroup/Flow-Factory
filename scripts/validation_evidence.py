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

"""Bounded evidence bundles for manual GPU runs and the same local/CI gate."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import validate_validation_plan as validator

MAX_BUNDLE_BYTES = 60000


def verify_files(root: Path, results: dict) -> None:
    """Check log content identities, not just identity strings supplied by the recorder."""
    for case in results["cases"]:
        for relative, expected in case.get("evidence", {}).items():
            if PurePosixPath(relative).parts[:1] != (".scratch",):
                raise ValueError(f"Evidence must be stored under .scratch/: {relative}")
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or path.stat().st_size > 12 * 1024 * 1024:
                raise ValueError(f"Evidence file escapes repository or exceeds limit: {relative}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Evidence content mismatch: {relative}")


def make_bundle(root: Path, plan: dict, results: dict) -> dict:
    """Package the exact result and small assertion logs for a workflow dispatch input."""
    validator.validate_results(
        plan, results, expected_commit=plan["tested_commit"], require_coverage=False
    )
    verify_files(root, results)
    files = {}
    for case in results["cases"]:
        for relative in case.get("evidence", {}):
            files[relative] = (root / relative).read_text()
    bundle = {
        "schema_version": 1,
        "base_commit": plan["base_commit"],
        "tested_commit": plan["tested_commit"],
        "results": results,
        "files": files,
    }
    if len(json.dumps(bundle).encode()) > MAX_BUNDLE_BYTES:
        raise ValueError(
            "Evidence bundle exceeds 60 KB; retain concise assertion logs, archive full traces separately"
        )
    return bundle


def import_bundle(root: Path, plan: dict, bundle: dict) -> dict:
    """Validate first, then place immutable logs under an isolated ignored directory."""
    if len(json.dumps(bundle).encode()) > MAX_BUNDLE_BYTES:
        raise ValueError("Evidence bundle exceeds 60 KB")
    if (
        bundle.get("schema_version") != 1
        or bundle.get("base_commit") != plan["base_commit"]
        or bundle.get("tested_commit") != plan["tested_commit"]
    ):
        raise ValueError("Evidence bundle source identity mismatch")
    results = copy.deepcopy(bundle["results"])
    validator.validate_results(
        plan, results, expected_commit=plan["tested_commit"], require_coverage=False
    )
    files = bundle["files"]
    if not isinstance(files, dict):
        raise ValueError("Evidence bundle files must be a mapping")
    checked = {}
    for case in results["cases"]:
        for relative, expected in case.get("evidence", {}).items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe evidence path: {relative}")
            content = files.get(relative)
            if (
                not isinstance(content, str)
                or hashlib.sha256(content.encode()).hexdigest() != expected
            ):
                raise ValueError(f"Evidence bundle content mismatch: {relative}")
            checked[relative] = (expected, content)
    directory = root / ".scratch/validation-evidence" / plan["tested_commit"]
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence directory escapes repository")
    directory.mkdir(parents=True, exist_ok=True)
    for expected, content in checked.values():
        target = directory / (expected + ".log")
        if target.is_symlink():
            raise ValueError("Evidence target is a symlink")
        target.write_text(content)
    for case in results["cases"]:
        if "evidence" in case:
            case["evidence"] = {
                (directory / (digest + ".log")).relative_to(root).as_posix(): digest
                for digest in case["evidence"].values()
            }
    return results


def artifact_name(plan: dict) -> str:
    return f"validation-evidence-{plan['tested_commit']}-{validator.plan_sha256(plan)}"


def fetch_bundle(repository: str, plan: dict) -> dict | None:
    """Read only the newest matching artifact from the designated manual evidence workflow."""
    name = artifact_name(plan)
    data = subprocess.check_output(
        [
            "gh",
            "api",
            f"repos/{repository}/actions/artifacts?name={name}&per_page=100",
        ]
    )
    candidates = sorted(
        json.loads(data)["artifacts"], key=lambda item: item["created_at"], reverse=True
    )
    for artifact in candidates:
        run = artifact.get("workflow_run", {})
        if (
            artifact["name"] != name
            or artifact["expired"]
            or run.get("head_sha") != plan["tested_commit"]
        ):
            continue
        details = json.loads(
            subprocess.check_output(["gh", "api", f"repos/{repository}/actions/runs/{run['id']}"])
        )
        if (
            details.get("event") != "workflow_dispatch"
            or details.get("path") != ".github/workflows/quality.yml"
            or details.get("conclusion") != "success"
        ):
            continue
        if artifact.get("size_in_bytes", 0) > MAX_BUNDLE_BYTES * 2:
            raise ValueError("Evidence archive exceeds transfer limit")
        raw = subprocess.check_output(
            ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact['id']}/zip"]
        )
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != "bundle.json"
                or entries[0].file_size > MAX_BUNDLE_BYTES
            ):
                raise ValueError("Unexpected evidence archive contents")
            return json.loads(archive.read(entries[0]))
    return None
