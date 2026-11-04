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

"""Check repository consistency without importing Flow-Factory or loading models."""

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from agent_diff import git
from audit_docstrings import VENDORED_PREFIXES, audit_source
from repo_checks.core import Repository
from repo_checks.dependencies import check_dependencies
from repo_checks.documents import check_documents, check_roadmap
from repo_checks.registries import check_examples, check_registries
from validate_agent_harness import validate_harness

ROOT = Path(__file__).resolve().parents[1]
CHANGED_PATHS_ENV = "FF_QUALITY_CHANGED_PATHS"
CHECKER_INPUTS = (
    "scripts/check_repository.py",
    "scripts/repo_checks/",
    "scripts/agent_diff.py",
    "scripts/quality.py",
    ".pre-commit-config.yaml",
    ".github/workflows/quality.yml",
    "Makefile",
    "requirements/quality.txt",
)


def select_checks(paths: list[str] | None) -> set[str]:
    if paths is None or any(path.startswith(CHECKER_INPUTS) for path in paths):
        return {"registries", "documents", "dependencies", "harness", "docstrings"}
    if not paths:
        return set()
    # Every tracked path can be a link target, including deleted files and renamed directories.
    checks = {"documents"}
    if any(path.startswith(("src/flow_factory/", "examples/", "config/")) for path in paths):
        checks.add("registries")
    if any(
        path.startswith(("src/flow_factory/scheduler/", "src/flow_factory/contracts/"))
        for path in paths
    ):
        checks.add("dependencies")
    if any(
        path.startswith((".agents/", ".claude/", ".cursor/"))
        or path
        in ("AGENTS.md", "CLAUDE.md", "scripts/agent_scope.py", "scripts/validate_agent_harness.py")
        for path in paths
    ):
        checks.add("harness")
    if any(
        path.startswith("src/flow_factory/")
        and path.endswith(".py")
        or path == "scripts/audit_docstrings.py"
        for path in paths
    ):
        checks.add("docstrings")
    return checks


def check(root: Path, paths: list[str] | None = None) -> dict:
    started = time.perf_counter()
    repo = Repository(root)
    checks = select_checks(paths)
    if "registries" in checks:
        check_examples(repo, check_registries(repo))
    if "documents" in checks:
        check_roadmap(repo, check_documents(repo))
    if "dependencies" in checks:
        check_dependencies(repo)
    if "harness" in checks:
        for error in validate_harness(root):
            repo.error(".agents", 1, "harness", error)
    if "docstrings" in checks:
        full_audit = paths is None or any(
            path.startswith(CHECKER_INPUTS) or path == "scripts/audit_docstrings.py"
            for path in paths
        )
        for path in sorted(repo.paths if full_audit else paths):
            if (
                path.startswith("src/flow_factory/")
                and path.endswith(".py")
                and not path.startswith(VENDORED_PREFIXES)
                and repo.exists(path)
            ):
                for finding in audit_source(path, repo.read(path)):
                    if finding.severity == "error":
                        repo.error(
                            path, finding.line, "docstring-" + finding.check, finding.message
                        )
    return {
        "checks": sorted(checks),
        "findings": [asdict(finding) for finding in sorted(set(repo.findings))],
        "unverified": sorted(repo.unverified),
        "source_bytes": repo.bytes_read,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def hook_paths(root: Path, files: list[str] | None = None) -> list[str] | None:
    if CHANGED_PATHS_ENV in os.environ:
        paths = json.loads(os.environ[CHANGED_PATHS_ENV])
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError(f"{CHANGED_PATHS_ENV} must be a JSON list of paths")
        return paths
    paths = [
        os.fsdecode(path)
        for path in git(root, "diff", "--cached", "--name-only", "--no-renames", "-z", "--").split(
            b"\0"
        )
        if path
    ]
    # A manual pre-commit --all-files invocation can have an empty index diff.
    return sorted(set(paths + (files or []))) or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--hook", action="store_true", help="Select checks from staged/quality snapshot inputs"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "files", nargs="*", help="Paths supplied by pre-commit, plus staged deletions"
    )
    args = parser.parse_args()
    try:
        result = check(
            args.root.resolve(), hook_paths(args.root, args.files) if args.hook else None
        )
    except (ValueError, SyntaxError, OSError, subprocess.CalledProcessError) as error:
        print(f"repository consistency: input could not be checked: {error}")
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for finding in result["findings"]:
            print("{path}:{line}: [{rule}] {message}".format(**finding))
        for note in result["unverified"]:
            print(f"UNVERIFIED: {note}")
        print(
            f"Repository consistency: {len(result['findings'])} errors; checks={','.join(result['checks'])}; {result['elapsed_seconds']:.3f}s"
        )
    return bool(result["findings"])


if __name__ == "__main__":
    raise SystemExit(main())
