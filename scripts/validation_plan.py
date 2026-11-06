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

"""Generate immutable validation plans and consume exact-source evidence."""

from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import agent_diff
import agent_scope
import validate_validation_plan as evidence
import validation_catalog
import validation_evidence
import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CATALOG = "config/validation/catalog.yaml"
POLICY = "config/validation/selection.yaml"
POLICY_INPUTS = (
    CATALOG,
    POLICY,
    ".agents/harness/risk_rules.yaml",
    ".agents/harness/test_routes.yaml",
    "scripts/agent_diff.py",
    "scripts/agent_scope.py",
    "scripts/validation_plan.py",
    "scripts/validation_evidence.py",
    "scripts/validation_catalog.py",
    "scripts/validate_validation_plan.py",
    "pyproject.toml",
    "requirements/quality.txt",
)


def matches(path: str, patterns: list[str]) -> bool:
    """Use the same portable path matching as risk classification."""
    return any(agent_scope._matches(path, pattern) for pattern in patterns)


def changed_symbols(before: str | None, after: str | None) -> list[str]:
    """Name changed Python definitions without executing their modules."""

    def definitions(source: str | None) -> dict[str, str]:
        found = {}
        if source is None:
            return found

        def visit(node: ast.AST, prefix: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + child.name
                    # A property getter and setter share a qualified name; both
                    # definitions belong to that owner rather than the last one.
                    found[name] = found.get(name, "") + ast.dump(child, include_attributes=False)
                    visit(child, name + ".")
                else:
                    visit(child, prefix)

        module = ast.parse(source)
        visit(module)
        # Symbol-strict owners must also see changed imports, constants and
        # module-level execution beside an otherwise mapped function edit.
        body = [
            node
            for node in module.body
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        ]
        found["<module>"] = ast.dump(
            ast.Module(body=body, type_ignores=[]), include_attributes=False
        )
        return found

    try:
        old, new = definitions(before), definitions(after)
    except (SyntaxError, ValueError, RecursionError):
        return ["<unparsed>"]
    return sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))


def package_version_only(before: str | None, after: str | None) -> bool:
    """Prove that a release-preparation edit changes only project.version."""
    if before is None or after is None:
        return False
    try:
        old, new = tomllib.loads(before), tomllib.loads(after)
        if not isinstance(old.get("project"), dict) or not isinstance(new.get("project"), dict):
            return False
        previous = old["project"].pop("version")
        current = new["project"].pop("version")
        return (
            isinstance(previous, str)
            and isinstance(current, str)
            and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", current) is not None
            and previous != current
            and old == new
        )
    except (KeyError, TypeError, tomllib.TOMLDecodeError):
        return False


def infer_requirements(scope: dict, policy: dict, symbols: dict[str, list[str]]) -> dict:
    """Retain uncovered owners and mandatory interactions before optimizing test cost."""
    reasons: dict[str, list[str]] = {}
    mandatory: dict[str, list[str]] = {}

    def add(obligation: str, reason: str) -> None:
        reasons.setdefault(obligation, [])
        if reason not in reasons[obligation]:
            reasons[obligation].append(reason)

    add(
        "repository.consistency", "Every diff preserves repository references and static contracts."
    )
    exempt = (
        set(scope.get("docs_only_paths", []))
        | set(scope.get("logging_only_paths", []))
        | set(scope.get("metadata_only_paths", []))
    )
    for path in scope["paths"]:
        if path in exempt:
            continue
        reached = False
        path_rules = [rule for rule in policy["owners"] if matches(path, rule["paths"])]
        mapped_symbols = set()
        for rule in path_rules:
            if rule.get("symbols") and not any(
                fnmatch.fnmatch(name, pattern)
                for name in symbols.get(path, [])
                for pattern in rule["symbols"]
            ):
                continue
            reached = True
            mapped_symbols.update(
                name
                for name in symbols.get(path, [])
                if not rule.get("symbols")
                or any(fnmatch.fnmatch(name, pattern) for pattern in rule["symbols"])
            )
            for obligation in rule["requires"]:
                add(obligation, f"{path}: {rule['reason']}")
        if not reached and not matches(path, policy["static_paths"]):
            add(
                f"review.owner:{path}",
                f"No catalog owner mapping covers {path}; add reviewed coverage.",
            )
        if any(rule.get("require_explicit_symbols", False) for rule in path_rules):
            for name in sorted(set(symbols.get(path, [])) - mapped_symbols):
                add(
                    f"review.symbol:{path}:{name}",
                    f"{path}: symbol {name!r} has no explicit owner route; add reviewed coverage.",
                )
    facets = set(scope["facets"])
    for rule in policy.get("interactions", []):
        if not set(rule["facets"]) <= facets:
            continue
        for obligation in rule.get("requires", []):
            add(obligation, rule["reason"])
        for name in rule.get("mandatory", []):
            mandatory.setdefault(name, []).append(rule["reason"])
    if not scope["scope_complete"]:
        add("review.incomplete_scope", "Opaque inputs prevent complete diff inspection.")
    return {"obligations": reasons, "mandatory": mandatory}


def select_cases(
    catalog: dict, requirements: dict, additions: list[str]
) -> tuple[list[dict], list[str]]:
    """Choose mandatory cases first, then a deterministic weighted covering set."""
    required = set(requirements["obligations"])
    available = {case["id"]: case for case in catalog["cases"]}
    selected: dict[str, str] = {}
    for name in [*requirements["mandatory"], *additions]:
        if name not in available:
            raise ValueError(f"Unknown mandatory/explicit catalog case: {name}")
        selected[name] = "; ".join(
            requirements["mandatory"].get(name, ["Explicit reviewer addition."])
        )
    covered = {item for name in selected for item in available[name]["covers"]}
    while required - covered:
        choices = [
            case
            for name, case in available.items()
            if name not in selected and set(case["covers"]) & (required - covered)
        ]
        if not choices:
            break
        best = min(
            choices,
            key=lambda case: (
                case["cost_seconds"] / len(set(case["covers"]) & (required - covered)),
                case["cost_seconds"],
                case["id"],
            ),
        )
        obligations = sorted(set(best["covers"]) & (required - covered))
        selected[best["id"]] = (
            "Covers " + ", ".join(obligations) + "; lowest declared cost per uncovered obligation."
        )
        covered.update(best["covers"])
    cases = []
    for original in catalog["cases"]:
        case = copy.deepcopy(original)
        case["selected"] = case["id"] in selected
        case["reason"] = selected.get(
            case["id"], "Not needed by the required coverage and mandatory interactions."
        )
        cases.append(case)
    return cases, sorted(required - covered)


def read_yaml(root: Path, commit: str, path: str) -> dict:
    """Read policy from the tested commit, never an unrelated working-tree revision."""
    blob = agent_diff.source(root, commit, path, agent_diff.MAX_SOURCE_BYTES)
    if blob is None:
        raise ValueError(f"Missing policy at {commit}: {path}")
    data = yaml.safe_load(blob)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping: {path}")
    return data


def classify_snapshot(snapshot: dict, rules: dict) -> tuple[dict, dict]:
    """Share bounded semantic classification across planning and historical replay."""
    docs, logging, unverified = agent_scope.prove_edits(
        {
            path: pair
            for path, pair in snapshot["pairs"].items()
            if path not in snapshot["mode_changes"]
        }
    )
    scope = agent_scope.classify_changes(
        snapshot["paths"],
        snapshot["diffs"],
        rules,
        intent="develop",
        docs_only_paths=docs,
        logging_only_paths=logging,
    )
    scope["scope_complete"] = snapshot["complete"] and not unverified
    pair = snapshot["pairs"].get("pyproject.toml")
    scope["metadata_only_paths"] = (
        ["pyproject.toml"]
        if pair and "pyproject.toml" not in snapshot["mode_changes"] and package_version_only(*pair)
        else []
    )
    scope["uninspected"] = snapshot["uninspected"]
    symbols = {
        path: changed_symbols(*pair)
        for path, pair in snapshot["pairs"].items()
        if path.endswith(".py")
    }
    return scope, symbols


def requirements_for(scope: dict, symbols: dict, catalog: dict, policy: dict) -> dict:
    """Use identical owner and catalog rules for executable plans and historical analysis."""
    policy = copy.deepcopy(policy)
    for case in catalog["cases"]:
        test_paths = [path for path in case["artifacts"] if path.startswith("tests/")]
        if test_paths:
            policy["owners"].append(
                {
                    "paths": test_paths,
                    "requires": case["covers"],
                    "reason": "Cataloged test artifacts require their declared coverage.",
                }
            )
    requirements = infer_requirements(scope, policy, symbols)
    for generic in ("parallel.tp_parity", "parallel.sp_parity"):
        if generic in requirements["obligations"]:
            for case in catalog["cases"]:
                for obligation in case["covers"]:
                    if obligation.startswith(generic + "."):
                        requirements["obligations"].setdefault(obligation, []).append(
                            "Shared parallel changes must cover each declared model plan."
                        )
    changed_tests = {
        path
        for path in scope["paths"]
        if path.startswith("tests/")
        and path not in set(scope["docs_only_paths"]) | set(scope["logging_only_paths"])
    }
    for case in catalog["cases"]:
        if changed_tests & set(case["artifacts"]):
            requirements["mandatory"].setdefault(case["id"], []).append(
                "Execute the changed test artifact itself."
            )
    return requirements


def generate(root: Path, base: str, head: str, additions: list[str], extra: list[str]) -> dict:
    """Bind scope, selection policy, catalog and test/config identities to one source commit."""
    snapshot = agent_diff.collect_snapshot(root, base, head)
    commit = snapshot["head_commit"]
    rules = read_yaml(root, commit, ".agents/harness/risk_rules.yaml")
    scope, symbols = classify_snapshot(snapshot, rules)
    catalog, policy = read_yaml(root, commit, CATALOG), read_yaml(root, commit, POLICY)
    blobs = {}
    remaining = agent_diff.MAX_TOTAL_BYTES

    def source(path: str) -> bytes | None:
        nonlocal remaining
        if path not in blobs:
            blob = agent_diff.source(
                root, commit, path, min(remaining, agent_diff.MAX_SOURCE_BYTES)
            )
            if blob is not None:
                remaining -= len(blob)
            blobs[path] = blob
        return blobs[path]

    validation_catalog.validate_catalog(catalog, root=root, source=source)
    requirements = requirements_for(scope, symbols, catalog, policy)
    for obligation in extra:
        requirements["obligations"].setdefault(obligation, []).append(
            "Explicit reviewer obligation."
        )
    cases, missing = select_cases(catalog, requirements, additions)
    obligations = sorted(
        set(requirements["obligations"]) | {item for case in cases for item in case["covers"]}
    )
    paths = set(POLICY_INPUTS) | {
        path for case in cases if case["selected"] for path in case["artifacts"]
    }
    identities = {}
    for path in sorted(paths):
        blob = source(path)
        if blob is None:
            raise ValueError(f"Missing declared artifact at {commit}: {path}")
        identities[path] = hashlib.sha256(blob).hexdigest()
    plan = {
        "schema_version": 1,
        "base_commit": snapshot["base_commit"],
        "tested_commit": commit,
        "risk": scope["profile"],
        "owners": scope["paths"] or ["repository"],
        "facets": scope["facets"] or ["scope"],
        "artifacts": identities,
        "obligations": [
            {
                "id": name,
                "required": name in requirements["obligations"],
                "description": "; ".join(
                    requirements["obligations"].get(name, ["Catalog coverage outside this diff."])
                ),
            }
            for name in obligations
        ],
        "cases": cases,
        "selection": {
            "schema_version": 1,
            "scope": scope,
            "changed_symbols": symbols,
            "uncovered_obligations": missing,
            "explicit_cases": sorted(set(additions)),
            "explicit_obligations": sorted(set(extra)),
            "reuse": "Only identical commit, plan hash and artifact identities are accepted.",
        },
    }
    evidence.validate_plan(plan, require_coverage=False)
    return plan


def replay(root: Path, base: str, head: str) -> dict:
    """Route an old diff using current policy without claiming current tests ran on old source."""
    snapshot = agent_diff.collect_snapshot(root, base, head)
    policy_commit = agent_diff.commit(root, "HEAD")
    rules = read_yaml(root, policy_commit, ".agents/harness/risk_rules.yaml")
    policy = read_yaml(root, policy_commit, POLICY)
    catalog = read_yaml(root, policy_commit, CATALOG)
    scope, symbols = classify_snapshot(snapshot, rules)
    return {
        "execution_eligible": False,
        "base_commit": snapshot["base_commit"],
        "head_commit": snapshot["head_commit"],
        "policy_commit": policy_commit,
        "scope": scope,
        "changed_symbols": symbols,
        "requirements": requirements_for(scope, symbols, catalog, policy),
    }


def environment(required_packages: tuple[str, ...] = ()) -> dict:
    """Record actual runner dependencies without importing the training stack."""
    packages = {}
    for name in dict.fromkeys(
        (
            *required_packages,
            "PyYAML",
            "pytest",
            "torch",
            "accelerate",
            "diffusers",
            "transformers",
            "peft",
            "deepspeed",
        )
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "hardware": platform.platform(),
        "packages": packages,
        "executable": sys.executable,
    }


def empty_results(plan: dict) -> dict:
    """Create a result envelope; absence of a selected case remains blocked."""
    return {
        "schema_version": 1,
        "tested_commit": plan["tested_commit"],
        "plan_sha256": evidence.plan_sha256(plan),
        "artifacts": plan["artifacts"],
        "environment": environment(
            tuple(
                sorted(
                    {
                        package
                        for case in plan["cases"]
                        if case["selected"]
                        for package in case.get("resources", {}).get("packages", [])
                    }
                )
            )
        ),
        "cases": [],
    }


def verify_checkout(root: Path, plan: dict) -> None:
    """Reject execution in a dirty or different source checkout."""
    evidence.verify_source(plan, root)
    if agent_diff.commit(root, "HEAD") != plan["tested_commit"]:
        raise ValueError("Execution checkout HEAD differs from the plan")
    if agent_diff.git(root, "status", "--porcelain", "--untracked-files=normal").strip():
        raise ValueError(
            "Execution requires a clean checkout; keep evidence under ignored .scratch/"
        )


def run_cpu(root: Path, plan: dict, destination: Path) -> dict:
    """Execute selected local assertion cases; missing dependencies remain blocked."""
    verify_checkout(root, plan)
    destination.mkdir(parents=True, exist_ok=True)
    results = empty_results(plan)
    for case in plan["cases"]:
        if not case["selected"]:
            continue
        resources = case["resources"]
        if case["backend"] != "cpu" or resources["gpus"] or case["oracle"]["kind"] != "assertions":
            continue
        missing = []
        for package in resources["packages"]:
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                missing.append(package)
        if missing:
            results["cases"].append(
                {
                    "id": case["id"],
                    "status": "blocked",
                    "reason": "Missing packages: " + ", ".join(missing),
                }
            )
            continue
        name = case["id"]
        if not all(char.isalnum() or char in "._-" for char in name):
            raise ValueError(f"Unsafe case ID: {name}")
        log, junit = destination / f"{name}.log", destination / f"{name}.xml"
        env = {
            **os.environ,
            **resources["environment"],
            "PYTHONPATH": str(root / "src"),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        }
        env.pop("PYTEST_ADDOPTS", None)
        expects_pytest = "pytest" in resources["packages"]
        if expects_pytest:
            env["PYTEST_ADDOPTS"] = shlex.join([f"--junitxml={junit}"])
        with log.open("wb") as output:
            process = subprocess.Popen(
                [sys.executable, *case["command"][1:]],
                cwd=root,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
        status, reason = "pass", ""
        if returncode:
            status, reason = "fail", f"Command exited {returncode}; see {log}"
        elif expects_pytest:
            try:
                suites = ET.parse(junit).getroot()
                tests = suites.findall(".//testcase")
                if not tests or any(
                    test.find("skipped") is not None
                    or test.find("failure") is not None
                    or test.find("error") is not None
                    for test in tests
                ):
                    status, reason = (
                        "blocked",
                        "Selected pytest evidence has skipped, failed, or no test cases.",
                    )
            except (OSError, ET.ParseError):
                status, reason = (
                    "blocked",
                    "Selected pytest case did not produce its assertion report.",
                )
        result = {"id": name, "status": status, "environment": results["environment"]}
        if status != "pass":
            result["reason"] = reason
        else:
            result.update(
                {
                    "backend": case["backend"],
                    "topology": case["topology"],
                    "command": case["command"],
                    "runtime": {
                        "world_size": 1,
                        "ranks_completed": [0],
                        "device": "cpu",
                        "distributed_type": "NO",
                    },
                    "observations": {"exit_code": 0},
                    "evidence": {
                        log.relative_to(root)
                        .as_posix(): hashlib.sha256(log.read_bytes())
                        .hexdigest()
                    },
                }
            )
        if status == "pass" and expects_pytest:
            result["evidence"][junit.relative_to(root).as_posix()] = hashlib.sha256(
                junit.read_bytes()
            ).hexdigest()
            result["observations"]["tests_run"] = len(tests)
        result["resolved_command"] = [sys.executable, *case["command"][1:]]
        results["cases"].append(result)
    verify_checkout(root, plan)
    return results


# These tools consume evidence; they are not inputs to native training commands.
# Changing any other executable input invalidates ancestor evidence conservatively.
REUSE_TOOL_PATHS = {
    "scripts/validation_plan.py",
    "scripts/validation_evidence.py",
    "tests/scripts/test_validation_planner.py",
}
FRESH_REUSE_CASES = {"harness-contracts", "repository-consistency"}


def reuse_changes(root: Path, source: str, head: str) -> list[str]:
    """Prove ancestry and an unchanged executable tree outside CPU evidence tooling."""
    evidence.digest(source, 40, "source commit")
    evidence.digest(head, 40, "target commit")
    agent_diff.git(root, "merge-base", "--is-ancestor", source, head)
    paths = agent_diff.changed_paths(root, source, head)
    if agent_diff.changed_modes(root, source, head):
        raise ValueError("Evidence reuse rejects additions, deletions and file-mode changes")
    for path in paths:
        documentation = path.endswith(".md") and path.startswith(("roadmap/", "guidance/"))
        if path not in REUSE_TOOL_PATHS and not documentation:
            raise ValueError(f"Evidence reuse requires unchanged execution inputs: {path}")
        for revision in (source, head):
            if agent_diff.source(root, revision, path, agent_diff.MAX_SOURCE_BYTES) is None:
                raise ValueError(f"Evidence reuse requires an existing regular file: {path}")
    return paths


def source_plan(root: Path, plan: dict, bundle: dict) -> dict:
    """Reconstruct the original plan without changing the bundle's tested identity."""
    if bundle["base_commit"] != plan["base_commit"]:
        raise ValueError("Ancestor evidence comparison base differs")
    if bundle["tested_commit"] == plan["tested_commit"]:
        return plan
    reuse_changes(root, bundle["tested_commit"], plan["tested_commit"])
    return generate(root, plan["base_commit"], bundle["tested_commit"], [], [])


def fetch_evidence(root: Path, repository: str, plan: dict) -> dict | None:
    """Prefer exact evidence, then bounded first-parent ancestors with equivalent inputs."""
    bundle = validation_evidence.fetch_bundle(repository, plan)
    if bundle is not None:
        return bundle
    commits = (
        agent_diff.git(
            root,
            "rev-list",
            "--first-parent",
            "--max-count=20",
            f"{plan['base_commit']}..{plan['tested_commit']}",
        )
        .decode()
        .splitlines()[1:]
    )
    for commit in commits:
        try:
            previous = source_plan(
                root, plan, {"base_commit": plan["base_commit"], "tested_commit": commit}
            )
        except ValueError:
            break
        bundle = validation_evidence.fetch_bundle(repository, previous)
        if bundle is not None:
            return bundle
    return None


def inherited_results(root: Path, plan: dict, previous: dict, results: dict) -> tuple[list, dict]:
    """Validate original observations and match complete case definitions before reuse."""
    if previous["base_commit"] != plan["base_commit"]:
        raise ValueError("Ancestor evidence comparison base differs")
    changes = reuse_changes(root, previous["tested_commit"], plan["tested_commit"])
    verify_generated_plan(root, previous)
    evidence.validate_results(
        previous, results, expected_commit=previous["tested_commit"], require_coverage=False
    )
    validation_evidence.verify_files(root, results)
    old_cases = {case["id"]: case for case in previous["cases"]}
    current = {case["id"]: case for case in plan["cases"] if case["selected"]}
    accepted = []
    for result in results["cases"]:
        name = result["id"]
        if name in FRESH_REUSE_CASES or name not in current:
            continue
        before, after = copy.deepcopy(old_cases[name]), copy.deepcopy(current[name])
        for case in (before, after):
            case.pop("reason", None)
        if before != after:
            raise ValueError(f"Ancestor evidence case changed: {name}")
        if set(changes) & set(after["artifacts"]):
            raise ValueError(f"Ancestor evidence test/config changed: {name}")
        for path in after["artifacts"]:
            if previous["artifacts"].get(path) != plan["artifacts"].get(path):
                raise ValueError(f"Ancestor evidence artifact changed: {path}")
        accepted.append(
            {**result, "environment": result.get("environment", results["environment"])}
        )
    return accepted, {
        "tested_commit": previous["tested_commit"],
        "plan_sha256": evidence.plan_sha256(previous),
        "accepted_for_commit": plan["tested_commit"],
        "changed_paths": changes,
        "cases": [case["id"] for case in accepted],
    }


def report(
    plan: dict,
    documents: list[dict],
    expected: str,
    *,
    inherited: list[tuple[dict, dict]] | None = None,
    root: Path = ROOT,
) -> tuple[dict, dict]:
    """Merge independently validated result envelopes without hiding conflicting observations."""
    evidence.validate_plan(plan, require_coverage=False)
    combined = empty_results(plan)
    by_id = {}
    for document in documents:
        evidence.validate_results(plan, document, expected_commit=expected, require_coverage=False)
        for result in document["cases"]:
            name = result["id"]
            normalized = {
                **result,
                "environment": result.get("environment", document["environment"]),
            }
            if name in by_id and by_id[name] != normalized:
                if by_id[name]["status"] == "blocked" and result["status"] == "pass":
                    pass
                elif by_id[name]["status"] == "pass" and result["status"] == "blocked":
                    continue
                else:
                    raise ValueError(
                        f"Conflicting results for {name}; provide one authoritative result"
                    )
            by_id[name] = normalized
    provenance = []
    if inherited:
        if any(by_id.get(name, {}).get("status") != "pass" for name in FRESH_REUSE_CASES):
            raise ValueError(
                "Ancestor evidence requires fresh harness-contracts and repository-consistency passes"
            )
        for previous, results in inherited:
            accepted, receipt = inherited_results(root, plan, previous, results)
            receipt["cases"] = []
            provenance.append(receipt)
            for result in accepted:
                name = result["id"]
                if name in by_id and by_id[name]["status"] != "blocked":
                    if by_id[name]["status"] != result["status"]:
                        raise ValueError(f"Conflicting ancestor results for {name}")
                    continue
                by_id[name] = {**result, "source_commit": previous["tested_commit"]}
                receipt["cases"].append(name)
    combined["cases"] = list(by_id.values())
    summary = evidence.validate_results(
        plan, combined, expected_commit=expected, require_coverage=False
    )
    if summary["uncovered_obligations"] and summary["status"] == "pass":
        summary["status"] = "blocked"
    summary["plan_sha256"] = evidence.plan_sha256(plan)
    summary["tested_commit"] = plan["tested_commit"]
    if provenance:
        summary["evidence_reuse"] = provenance
    return summary, combined


def verify_generated_plan(root: Path, plan: dict) -> None:
    """Reject edits that remove obligations or replace a selected test after generation."""
    selection = plan["selection"]
    canonical = generate(
        root,
        plan["base_commit"],
        plan["tested_commit"],
        selection["explicit_cases"],
        selection["explicit_obligations"],
    )
    if evidence.plan_sha256(canonical) != evidence.plan_sha256(plan):
        raise ValueError("Plan differs from canonical generation for the tested source")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("generate")
    build.add_argument("--base", default="origin/main")
    build.add_argument("--head", default="HEAD")
    build.add_argument("--case", action="append", default=[])
    build.add_argument("--require", action="append", default=[])
    build.add_argument("--output", type=Path, required=True)
    history = commands.add_parser("replay")
    history.add_argument("--base", required=True)
    history.add_argument("--head", required=True)
    history.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run-cpu")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--logs", type=Path, required=True)
    gate = commands.add_parser("report")
    gate.add_argument("--plan", type=Path, required=True)
    gate.add_argument("--results", type=Path, action="append", default=[])
    gate.add_argument("--expected-commit", required=True)
    gate.add_argument("--output", type=Path)
    gate.add_argument("--bundle", type=Path, action="append", default=[])
    pack = commands.add_parser("bundle")
    pack.add_argument("--plan", type=Path, required=True)
    pack.add_argument("--results", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    accept = commands.add_parser("import-bundle")
    accept.add_argument("--plan", type=Path, required=True)
    accept.add_argument("--bundle", type=Path, required=True)
    fetch = commands.add_parser("fetch-evidence")
    fetch.add_argument("--plan", type=Path, required=True)
    fetch.add_argument("--repository", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "replay":
            result = replay(ROOT, args.base, args.head)
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0 if result["scope"]["scope_complete"] else 2
        if args.action == "generate":
            plan = generate(ROOT, args.base, args.head, args.case, args.require)
            write_json(args.output, plan)
            print(
                json.dumps(
                    {
                        "plan": str(args.output),
                        "selected": [case["id"] for case in plan["cases"] if case["selected"]],
                        "uncovered_obligations": plan["selection"]["uncovered_obligations"],
                    }
                )
            )
            return 0
        plan = yaml.safe_load(args.plan.read_text())
        verify_checkout(ROOT, plan)
        verify_generated_plan(ROOT, plan)
        if args.action == "run-cpu":
            evidence.validate_plan(plan, require_coverage=False)
            results = run_cpu(ROOT, plan, args.logs.resolve())
            write_json(args.output, results)
            print(json.dumps({case["id"]: case["status"] for case in results["cases"]}))
            return 1 if any(case["status"] == "fail" for case in results["cases"]) else 0
        if args.action == "bundle":
            write_json(
                args.output,
                validation_evidence.make_bundle(
                    ROOT, plan, yaml.safe_load(args.results.read_text())
                ),
            )
            return 0
        if args.action == "import-bundle":
            bundle = json.loads(args.bundle.read_text())
            validation_evidence.import_bundle(ROOT, plan, bundle)
            print(validation_evidence.artifact_name(plan))
            return 0
        if args.action == "fetch-evidence":
            bundle = fetch_evidence(ROOT, args.repository, plan)
            if bundle is not None:
                original = source_plan(ROOT, plan, bundle)
                validation_evidence.import_bundle(ROOT, original, bundle)
                write_json(args.output, bundle)
            print(
                "matching evidence found" if bundle is not None else "no matching external evidence"
            )
            return 0
        documents = [yaml.safe_load(path.read_text()) for path in args.results]
        for document in documents:
            validation_evidence.verify_files(ROOT, document)
        inherited = []
        for path in args.bundle:
            bundle = json.loads(path.read_text())
            original = source_plan(ROOT, plan, bundle)
            results = validation_evidence.import_bundle(ROOT, original, bundle)
            if original["tested_commit"] == plan["tested_commit"]:
                documents.append(results)
            else:
                inherited.append((original, results))
        summary, _ = report(plan, documents, args.expected_commit, inherited=inherited)
        if args.output:
            write_json(args.output, summary)
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "pass" else 1
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"validation error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
