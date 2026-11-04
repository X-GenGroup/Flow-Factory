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
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Set

import yaml


def _load_sibling_script(name: str) -> ModuleType:
    """Load a sibling script without making ``scripts`` a package."""
    script_path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load sibling script {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through ``sys.modules``.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prove_edits(pairs: Mapping[str, tuple]) -> tuple[list[str], list[str], list[str]]:
    """Apply narrow exemptions only to complete bounded before/after sources."""
    docs = _load_sibling_script("check_docs_only_diff")
    diff = _load_sibling_script("agent_diff")
    docs_only, logging_only, unverified = [], [], []
    for path, (before, after) in pairs.items():
        if not path.endswith(".py") or before is None or after is None:
            continue
        try:
            if docs.compare_sources(before, after) is None:
                docs_only.append(path)
            elif diff.literal_logging_only(before, after) and docs.directive_comments(
                before
            ) == docs.directive_comments(after):
                logging_only.append(path)
        except (ValueError, RecursionError) as error:
            unverified.append(f"{path}: {error}")
    return docs_only, logging_only, unverified


def _matches(path: str, pattern: str) -> bool:
    """Match a repository path against a portable glob."""
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.replace("**/", ""))


def classify_changes(
    paths: Iterable[str],
    diff_by_path: Mapping[str, str],
    rules: Mapping[str, Any],
    *,
    intent: str = "inspect",
    docs_only_paths: Iterable[str] = (),
    logging_only_paths: Iterable[str] = (),
) -> Dict[str, Any]:
    """Classify paths and diff text using the declarative risk rules.

    Paths in ``docs_only_paths`` changed only comments and docstrings, so they receive the
    ``docs_only_rule`` instead of path and content rules.
    """
    profiles = rules["profiles"]
    profile_name = rules.get("intents", {}).get(intent, "R0")
    profile_rank = profiles[profile_name]["rank"]
    reasons: List[Dict[str, Any]] = []
    facets: Set[str] = set()
    docs_only = set(docs_only_paths)
    logging_only = set(logging_only_paths)
    if docs_only and "docs_only_rule" not in rules:
        raise KeyError(
            f"risk rules define no docs_only_rule for docs-only paths {sorted(docs_only)}"
        )

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
        if path in docs_only:
            apply_rule(rules["docs_only_rule"], path)
            continue
        if path in logging_only:
            apply_rule(rules["logging_only_rule"], path)
            continue
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
        "docs_only_paths": sorted(docs_only.intersection(normalized_paths)),
        "logging_only_paths": sorted(logging_only.intersection(normalized_paths)),
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
    docs_only = set(result["docs_only_paths"])
    for path in result["paths"]:
        suffix = " (docs-only)" if path in docs_only else ""
        if path in result["logging_only_paths"]:
            suffix = " (literal console logging)"
        print(f"  - {path}{suffix}")
    print(f"Scope complete: {result['scope_complete']}")
    for path in result["untracked_roots"]:
        print(f"  - Untracked, not inspected: {path} (stage intended source files)")
    for entry in result["uninspected"]:
        print(f"  - Uninspected: {entry['path']}: {entry['reason']}")
    print("Escalation reasons:")
    for reason in result["reasons"]:
        paths = reason["paths"]
        displayed_paths = ", ".join(paths[:4])
        if len(paths) > 4:
            displayed_paths += f", +{len(paths) - 4} more"
        print(f"  - [{reason['profile']}] {displayed_paths}: {reason['reason']}")
    unverified = result.get("docs_only_unverified", [])
    if unverified:
        print(f"Docs-only proof unavailable for {len(unverified)} path(s); normal rules apply:")
        for entry in unverified[:4]:
            print(f"  - {entry}")
    if result["commands"]:
        print("Commands:")
        for command in result["commands"]:
            print(f"  - {command}")
    print("Evidence obligations (select QA and topology separately):")
    for evidence in result["evidence"]:
        print(f"  - {evidence}")
    if result["tests"]:
        print("Required tests or evidence:")
        for test in result["tests"]:
            print(f"  - {test}")


def main() -> int:
    """Classify the current repository diff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="Git base revision")
    parser.add_argument(
        "--head", help="Immutable end revision; otherwise inspect tracked worktree changes"
    )
    parser.add_argument("--intent", default="develop", help="Task intent from risk_rules.yaml")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    harness_dir = root / ".agents/harness"
    rules = _load_yaml(harness_dir / "risk_rules.yaml")
    routes = _load_yaml(harness_dir / "test_routes.yaml")
    snapshot = _load_sibling_script("agent_diff").collect_snapshot(root, args.base, args.head)
    docs_only, logging_only, unverified = prove_edits(
        {
            path: pair
            for path, pair in snapshot["pairs"].items()
            if path not in snapshot["mode_changes"]
        }
    )
    result = attach_evidence(
        classify_changes(
            snapshot["paths"],
            snapshot["diffs"],
            rules,
            intent=args.intent,
            docs_only_paths=docs_only,
            logging_only_paths=logging_only,
        ),
        routes,
    )
    result["docs_only_unverified"] = unverified
    result.update(
        {
            key: snapshot[key]
            for key in ("base_commit", "head_commit", "limits", "untracked_roots", "uninspected")
        }
    )
    result["scope_complete"] = snapshot["complete"]
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text(result)
    return 0 if result["scope_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
