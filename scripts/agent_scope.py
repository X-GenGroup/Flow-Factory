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

"""Classify a repository diff into an agent-harness risk profile."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import yaml


def _run_git(root: Path, arguments: Sequence[str]) -> str:
    """Run one read-only Git command and return stdout."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def collect_changed_files(root: Path, base: str) -> List[str]:
    """Return tracked and untracked paths changed relative to ``base``."""
    tracked = {
        line.strip()
        for line in _run_git(root, ["diff", "--name-only", base, "--"]).splitlines()
        if line.strip()
    }
    status = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    for line in status.splitlines():
        if line.startswith("?? "):
            tracked.add(line[3:])
    return sorted(tracked)


def collect_diff_text(root: Path, base: str, path: str) -> str:
    """Return diff text, or full text for an untracked file."""
    status = _run_git(root, ["status", "--porcelain=v1", "--", path])
    if status.startswith("?? "):
        target = root / path
        return target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    return _run_git(root, ["diff", "--unified=0", base, "--", path])


def _matches(path: str, pattern: str) -> bool:
    """Match a repository path against a portable glob."""
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.replace("**/", ""))


def classify_changes(
    paths: Iterable[str],
    diff_by_path: Mapping[str, str],
    rules: Mapping[str, Any],
    *,
    intent: str = "inspect",
) -> Dict[str, Any]:
    """Classify paths and diff text using the declarative risk rules."""
    profiles = rules["profiles"]
    profile_name = rules.get("intents", {}).get(intent, "R0")
    profile_rank = profiles[profile_name]["rank"]
    reasons: List[Dict[str, Any]] = []
    facets: Set[str] = set()

    def apply_rule(rule: Mapping[str, Any], path: str) -> None:
        nonlocal profile_name, profile_rank
        candidate = rule["profile"]
        candidate_rank = profiles[candidate]["rank"]
        if candidate_rank > profile_rank:
            profile_name = candidate
            profile_rank = candidate_rank
        facets.update(rule.get("facets", []))
        if not rule.get("report", True):
            return
        for reason in reasons:
            if reason["profile"] == candidate and reason["reason"] == rule["reason"]:
                if path not in reason["paths"]:
                    reason["paths"].append(path)
                return
        reasons.append({"paths": [path], "profile": candidate, "reason": rule["reason"]})

    normalized_paths = sorted(set(paths))
    for path in normalized_paths:
        for rule in rules.get("path_rules", []):
            if _matches(path, rule["glob"]):
                apply_rule(rule, path)
        diff_text = diff_by_path.get(path, "")
        for rule in rules.get("content_rules", []):
            if _matches(path, rule["glob"]) and re.search(rule["pattern"], diff_text):
                apply_rule(rule, path)

    return {
        "intent": intent,
        "profile": profile_name,
        "description": profiles[profile_name]["description"],
        "paths": normalized_paths,
        "facets": sorted(facets),
        "reasons": reasons,
    }


def attach_evidence(result: Dict[str, Any], routes: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach profile and facet evidence without duplicating entries."""
    evidence: List[str] = list(routes["profiles"][result["profile"]].get("evidence", []))
    commands: List[str] = []
    tests: List[str] = []
    for facet in result["facets"]:
        route = routes.get("facets", {}).get(facet, {})
        commands.extend(route.get("commands", []))
        tests.extend(route.get("tests", []))
    result["evidence"] = list(dict.fromkeys(evidence))
    result["commands"] = list(dict.fromkeys(commands))
    result["tests"] = list(dict.fromkeys(tests))
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load one required YAML mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected YAML mapping in {path}, received {type(data).__name__}")
    return data


def _print_text(result: Mapping[str, Any]) -> None:
    """Print a compact human-readable scope report."""
    print(f"Profile: {result['profile']} — {result['description']}")
    print(f"Intent: {result['intent']}")
    print("Facets: " + (", ".join(result["facets"]) or "none"))
    print("Changed paths:")
    for path in result["paths"]:
        print(f"  - {path}")
    print("Escalation reasons:")
    for reason in result["reasons"]:
        paths = reason["paths"]
        displayed_paths = ", ".join(paths[:4])
        if len(paths) > 4:
            displayed_paths += f", +{len(paths) - 4} more"
        print(f"  - [{reason['profile']}] {displayed_paths}: {reason['reason']}")
    if result["commands"]:
        print("Commands:")
        for command in result["commands"]:
            print(f"  - {command}")
    if result["tests"]:
        print("Required tests or evidence:")
        for test in result["tests"]:
            print(f"  - {test}")


def main() -> int:
    """Classify the current repository diff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="Git base revision")
    parser.add_argument("--intent", default="develop", help="Task intent from risk_rules.yaml")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    harness_dir = root / ".agents/harness"
    rules = _load_yaml(harness_dir / "risk_rules.yaml")
    routes = _load_yaml(harness_dir / "test_routes.yaml")
    paths = collect_changed_files(root, args.base)
    diffs = {path: collect_diff_text(root, args.base, path) for path in paths}
    result = attach_evidence(
        classify_changes(paths, diffs, rules, intent=args.intent),
        routes,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
