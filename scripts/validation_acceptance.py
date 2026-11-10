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

"""Explicit incremental acceptance, separate from canonical test qualification.

An authenticated evidence dispatch records a review, not a new GPU execution. The
normal report remains unchanged; only an exact, complete review with actual current
CPU/incremental passes can produce the separate accepted_incremental decision.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

import agent_diff
import validate_validation_plan as validator
import validation_evidence as transport
import validation_plan as planner

MAX_BYTES = 60000
MAX_DECODED_BYTES = 2 * 1024 * 1024
_IDENTITY = {"kind", "repository", "base_commit", "tested_commit", "plan_sha256"}
_ENVELOPE = _IDENTITY | {
    "schema_version",
    "encoding",
    "decoded_bytes",
    "decoded_sha256",
    "payload",
}
_FIELDS = {
    "schema_version",
    "kind",
    "repository",
    "base_commit",
    "tested_commit",
    "plan_sha256",
    "baseline",
    "review",
    "delta",
    "required_fresh_cases",
    "historical_cases",
}
_BASELINE = {
    "base_commit",
    "tested_commit",
    "plan_sha256",
    "bundle_sha256",
    "artifact_id",
    "run_id",
}
_REVIEW = {"decision", "scope", "reviewer", "authorization_quote", "reason"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict:
    """Read a bounded wire document and return the complete schema1 review."""
    require(path.stat().st_size <= MAX_BYTES, "Acceptance document exceeds 60 KB")
    return decode(json.loads(path.read_text()))


def _wire_bytes(document: dict) -> bytes:
    return transport._canonical_bytes(document) + b"\n"


def write_wire(path: Path, document: dict) -> None:
    """Write compact, validated transport bytes without expanding a fetched review."""
    decode(document)
    raw = _wire_bytes(document)
    require(len(raw) <= MAX_BYTES, "Acceptance document exceeds 60 KB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def decode(document: dict) -> dict:
    """Decode transport without weakening source, review or baseline validation."""
    require(isinstance(document, dict), "Acceptance document must be a mapping")
    require(len(_wire_bytes(document)) <= MAX_BYTES, "Acceptance document exceeds 60 KB")
    if document.get("schema_version") != 2:
        return document
    require(
        type(document["schema_version"]) is int
        and set(document) == _ENVELOPE
        and document["encoding"] == "gzip+base64",
        "Invalid compressed acceptance envelope",
    )
    inner = transport.decode_payload(
        document, max_decoded_bytes=MAX_DECODED_BYTES, label="acceptance"
    )
    require(
        type(inner.get("schema_version")) is int
        and inner["schema_version"] == 1
        and all(field in inner and inner[field] == document[field] for field in _IDENTITY),
        "Compressed acceptance inner source identity mismatch",
    )
    return inner


def pack(document: dict) -> dict:
    """Keep small schema1 reviews unchanged; compress complete larger reviews for dispatch."""
    require(
        isinstance(document, dict)
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1
        and _FIELDS <= set(document) <= _FIELDS | {"supplemental_baselines"},
        "Invalid decoded acceptance document",
    )
    decoded = transport._canonical_bytes(document)
    require(len(decoded) <= MAX_DECODED_BYTES, "Decoded acceptance document exceeds 2 MiB")
    if len(_wire_bytes(document)) <= MAX_BYTES:
        return document
    envelope = {
        "schema_version": 2,
        **{field: document[field] for field in _IDENTITY},
        **transport.compressed_payload(decoded),
    }
    require(
        len(_wire_bytes(envelope)) <= MAX_BYTES,
        "Compressed acceptance document exceeds 60 KB",
    )
    return envelope


def _ids(value: object, field: str) -> set[str]:
    require(isinstance(value, list), f"{field}: expected list")
    require(all(isinstance(item, str) and item for item in value), f"{field}: invalid ID")
    require(len(value) == len(set(value)), f"{field}: duplicate ID")
    return set(value)


def _blob(root: Path, revision: str, path: str) -> dict | None:
    entry = agent_diff.git(root, "ls-tree", "-z", revision, "--", f":(literal){path}")
    if not entry:
        return None
    mode = entry.split(b"\t", 1)[0].split()[0].decode()
    require(mode in {"100644", "100755"}, f"Unsupported reviewed file type: {path}")
    data = agent_diff.source(root, revision, path, agent_diff.MAX_SOURCE_BYTES)
    return {"mode": mode, "sha256": hashlib.sha256(data).hexdigest()}


def delta(root: Path, source: str, target: str) -> list[dict]:
    """Bind every changed tracked path, including deletions and executable-bit changes."""
    paths = agent_diff.changed_paths(root, source, target)
    require(len(paths) <= 1024, "Reviewed delta exceeds 1024 paths")
    return [
        {"path": path, "before": _blob(root, source, path), "after": _blob(root, target, path)}
        for path in paths
    ]


def _api(repository: str, suffix: str) -> bytes:
    return subprocess.check_output(["gh", "api", f"repos/{repository}/{suffix}"])


def _download(repository: str, artifact: dict, *, head: str, name: str, filename: str) -> bytes:
    require(artifact.get("name") == name and not artifact.get("expired"), "Wrong/expired artifact")
    run = artifact.get("workflow_run", {})
    require(run.get("head_sha") == head, "Artifact source head mismatch")
    run_id = run.get("id")
    require(type(run_id) is int and run_id > 0, "Invalid artifact run ID")
    details = json.loads(_api(repository, f"actions/runs/{run_id}"))
    require(
        details.get("event") == "workflow_dispatch"
        and details.get("path") == ".github/workflows/quality.yml"
        and details.get("conclusion") == "success"
        and details.get("head_sha") == head,
        "Artifact must come from a successful exact-head evidence dispatch",
    )
    limit = transport.MAX_BUNDLE_BYTES if filename == "bundle.json" else MAX_BYTES
    require(artifact.get("size_in_bytes", limit * 3) <= limit * 2, "Artifact exceeds limit")
    raw = _api(repository, f"actions/artifacts/{artifact['id']}/zip")
    require(len(raw) <= limit * 2, "Artifact transfer exceeds limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(
            len(entries) == 1 and entries[0].filename == filename and entries[0].file_size <= limit,
            "Unexpected acceptance/evidence archive contents",
        )
        return archive.read(entries[0])


def artifact_name(plan: dict) -> str:
    return f"validation-acceptance-{plan['tested_commit']}-{validator.plan_sha256(plan)}"


def fetch(repository: str, plan: dict) -> dict | None:
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository), "Invalid repository")
    name = artifact_name(plan)
    candidates = json.loads(_api(repository, f"actions/artifacts?name={name}&per_page=100"))[
        "artifacts"
    ]
    for item in sorted(candidates, key=lambda value: value["created_at"], reverse=True):
        if (
            item.get("name") != name
            or item.get("expired")
            or item.get("workflow_run", {}).get("head_sha") != plan["tested_commit"]
        ):
            continue
        document = json.loads(
            _download(
                repository, item, head=plan["tested_commit"], name=name, filename="acceptance.json"
            )
        )
        decoded = decode(document)
        require(decoded.get("repository") == repository, "Acceptance repository mismatch")
        return document
    return None


def _verify_baseline(
    root: Path,
    plan: dict,
    baseline: dict,
    reviewed_delta: list,
    *,
    repository: str,
    baseline_bundle: Path | None = None,
    baseline_cache: dict | None = None,
) -> tuple[dict, dict]:
    """Verify one original source and bundle without combining or relabeling its results."""
    require(isinstance(baseline, dict) and set(baseline) == _BASELINE, "Invalid baseline fields")
    for field in ("base_commit", "tested_commit"):
        validator.digest(baseline[field], 40, f"baseline.{field}")
    for field in ("plan_sha256", "bundle_sha256"):
        validator.digest(baseline[field], 64, f"baseline.{field}")
    for field in ("artifact_id", "run_id"):
        require(type(baseline[field]) is int and baseline[field] > 0, f"Invalid baseline {field}")
    require(
        baseline["tested_commit"] != plan["tested_commit"],
        "Baseline must retain its original source",
    )
    if baseline_bundle is None:
        # Force-pushed stacks can leave historical commits outside the fetched refs.
        # Fetch only validated commit IDs from the checkout's fixed origin.
        for field in ("base_commit", "tested_commit"):
            try:
                agent_diff.commit(root, baseline[field])
            except subprocess.CalledProcessError:
                agent_diff.git(root, "fetch", "--no-tags", "origin", baseline[field])
    require(
        reviewed_delta == delta(root, baseline["tested_commit"], plan["tested_commit"]),
        "Reviewed source delta differs",
    )
    # Only a single merge verification shares this cache. Source trees, bundle identity
    # and consumer code are immutable within that call; target deltas are still checked
    # above on every use. Local override files are never cached.
    cache_key = (str(root.resolve()), repository, transport._canonical_bytes(baseline))
    if baseline_bundle is None and baseline_cache is not None and cache_key in baseline_cache:
        return baseline_cache[cache_key]
    previous = planner.generate(root, baseline["base_commit"], baseline["tested_commit"], [], [])
    require(
        validator.plan_sha256(previous) == baseline["plan_sha256"],
        "Historical plan digest mismatch",
    )
    if baseline_bundle is not None:
        require(
            baseline_bundle.stat().st_size <= transport.MAX_BUNDLE_BYTES,
            "Baseline bundle exceeds limit",
        )
        raw = baseline_bundle.read_bytes()
    else:
        artifact = json.loads(_api(repository, f"actions/artifacts/{baseline['artifact_id']}"))
        require(
            artifact.get("id") == baseline["artifact_id"]
            and artifact.get("workflow_run", {}).get("id") == baseline["run_id"],
            "Historical artifact/run identity mismatch",
        )
        raw = _download(
            repository,
            artifact,
            head=baseline["tested_commit"],
            name=transport.artifact_name(previous),
            filename="bundle.json",
        )
    require(
        hashlib.sha256(raw).hexdigest() == baseline["bundle_sha256"],
        "Historical bundle hash mismatch",
    )
    # The normal decoder/validator owns historical runtime and numerical assertions.
    decoded = transport._decode_bundle(previous, json.loads(raw))
    results = decoded["results"]
    historical = validator.validate_results(
        previous, results, expected_commit=previous["tested_commit"], require_coverage=False
    )
    require(
        historical["status"] != "fail" and not previous["selection"]["uncovered_obligations"],
        "Historical baseline contains failures or uncovered selection",
    )
    # import_bundle verifies every assertion file before materializing its unchanged bytes.
    transport.import_bundle(root, previous, json.loads(raw))
    if baseline_bundle is None and baseline_cache is not None:
        require(len(baseline_cache) < 128, "Too many cached merge baselines")
        baseline_cache[cache_key] = (previous, historical)
    return previous, historical


def verify(
    root: Path,
    plan: dict,
    document: dict,
    *,
    repository: str,
    baseline_bundle: Path | None = None,
    supplemental_bundles: list[Path] | None = None,
    baseline_cache: dict | None = None,
) -> tuple[list[dict], dict]:
    """Validate review/source/baseline; recording this does not assert current tests passed."""
    require(
        isinstance(document, dict)
        and _FIELDS <= set(document) <= _FIELDS | {"supplemental_baselines"},
        "Invalid acceptance fields",
    )
    require(
        len(transport._canonical_bytes(document)) <= MAX_DECODED_BYTES,
        "Decoded acceptance document exceeds 2 MiB",
    )
    require(
        type(document["schema_version"]) is int and document["schema_version"] == 1,
        "Invalid acceptance schema",
    )
    require(document["kind"] == "reviewed_incremental", "Invalid acceptance kind")
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        and document["repository"] == repository,
        "Acceptance repository mismatch",
    )
    for field in ("base_commit", "tested_commit"):
        require(document[field] == plan[field], f"Acceptance {field} mismatch")
    require(
        document["plan_sha256"] == validator.plan_sha256(plan), "Acceptance plan digest mismatch"
    )
    planner.verify_generated_plan(root, plan)
    validator.verify_source(plan, root)
    require(
        not plan["selection"]["uncovered_obligations"], "Acceptance rejects uncovered selection"
    )
    review = document["review"]
    require(isinstance(review, dict) and set(review) == _REVIEW, "Invalid review fields")
    require(review["decision"] == "approved_incremental", "Review has not approved this scope")
    require(
        review["scope"] in {"restack_only", "model_tp_extension", "validation_or_documentation"},
        "Unsupported review scope",
    )
    for field in ("reviewer", "authorization_quote", "reason"):
        validator.text(review[field], f"review.{field}")
    supplements = document.get("supplemental_baselines", [])
    require(
        isinstance(supplements, list) and len(supplements) <= 4,
        "At most four supplemental baselines are allowed",
    )
    require(
        supplemental_bundles is None or len(supplemental_bundles) == len(supplements),
        "Supplemental local bundle count differs",
    )
    declarations = [{"baseline": document["baseline"], "delta": document["delta"]}, *supplements]
    sources = []
    heads = set()
    for index, item in enumerate(declarations):
        require(
            isinstance(item, dict) and set(item) == {"baseline", "delta"},
            "Invalid supplemental baseline fields",
        )
        local_bundle = (
            baseline_bundle
            if index == 0
            else (supplemental_bundles[index - 1] if supplemental_bundles is not None else None)
        )
        previous, historical = _verify_baseline(
            root,
            plan,
            item["baseline"],
            item["delta"],
            repository=repository,
            baseline_bundle=local_bundle,
            baseline_cache=baseline_cache,
        )
        require(previous["tested_commit"] not in heads, "Duplicate historical source")
        heads.add(previous["tested_commit"])
        sources.append({"baseline": item["baseline"], "plan": previous, "report": historical})
    current = {case["id"]: case for case in plan["cases"] if case["selected"]}
    fresh = _ids(document["required_fresh_cases"], "required_fresh_cases")
    history = _ids(document["historical_cases"], "historical_cases")
    require(
        fresh <= current.keys() and history <= current.keys(),
        "Unknown/nonselected disposition case",
    )
    require(not fresh & history, "Fresh and historical dispositions overlap")
    cpu = {name for name, case in current.items() if case["resources"]["gpus"] == 0}
    require(cpu <= fresh, "Every current CPU selection requires fresh evidence")
    require(fresh | history == current.keys(), "Incomplete acceptance dispositions")
    case_sources = {}
    contract_fields = (
        "backend",
        "command",
        "topology",
        "oracle",
        "covers",
        "qa",
        "execution_scope",
        "execution_capability",
        "fixture_kind",
        "resources",
    )
    for name in sorted(history):
        matching = []
        for source in sources:
            old = {case["id"]: case for case in source["plan"]["cases"] if case["selected"]}
            if (
                name in old
                and source["report"]["cases"].get(name) == "pass"
                and all(
                    old[name].get(field) == current[name].get(field) for field in contract_fields
                )
            ):
                matching.append(source["plan"]["tested_commit"])
        require(
            bool(matching), f"Historical case contract changed or missing passing evidence: {name}"
        )
        case_sources[name] = matching
    if review["scope"] == "model_tp_extension":
        require(bool(fresh - cpu), "Model TP extensions require a fresh GPU witness")
    return sources, case_sources


def decide(
    root: Path,
    plan: dict,
    summary: dict,
    current: dict,
    document: dict,
    *,
    repository: str,
    baseline_bundle: Path | None = None,
    supplemental_bundles: list[Path] | None = None,
    baseline_cache: dict | None = None,
) -> dict:
    sources, case_sources = verify(
        root,
        plan,
        document,
        repository=repository,
        baseline_bundle=baseline_bundle,
        supplemental_bundles=supplemental_bundles,
        baseline_cache=baseline_cache,
    )
    require(
        summary["tested_commit"] == plan["tested_commit"]
        and summary["plan_sha256"] == validator.plan_sha256(plan),
        "Current report identity mismatch",
    )
    require(
        "fail" not in summary["cases"].values(), "Current failures cannot be accepted incrementally"
    )
    checked = validator.validate_results(
        plan, current, expected_commit=plan["tested_commit"], require_coverage=False
    )
    require(
        checked["cases"] == summary["cases"] and checked["status"] == summary["status"],
        "Current report/result mismatch",
    )
    actual = {case["id"]: case for case in current["cases"]}
    for name in document["required_fresh_cases"]:
        require(
            actual.get(name, {}).get("source_commit", plan["tested_commit"])
            == plan["tested_commit"],
            f"Fresh case cannot use ancestor evidence: {name}",
        )
        require(summary["cases"].get(name) == "pass", f"Missing current passing evidence: {name}")
    source_receipts = [
        {
            **source["baseline"],
            "qualification_status": source["report"]["status"],
            "claimed_case_statuses": {
                name: "pass"
                for name, heads in case_sources.items()
                if source["plan"]["tested_commit"] in heads
            },
        }
        for source in sources
    ]
    return {
        "status": "accepted_incremental",
        "qualification_status": summary["status"],
        "tested_commit": plan["tested_commit"],
        "base_commit": plan["base_commit"],
        "plan_sha256": validator.plan_sha256(plan),
        "manifest_sha256": hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "review": document["review"],
        "fresh_cases": document["required_fresh_cases"],
        "historical_cases_not_rerun": [
            name for name in document["historical_cases"] if summary["cases"][name] != "pass"
        ],
        "historical_source": source_receipts[0],
        "supplemental_historical_sources": source_receipts[1:],
        "historical_case_sources": case_sources,
        "claim": "Reviewed incremental acceptance; historical cases are not current-head passes or automatic source-equivalence claims.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "fetch", "pack"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--baseline-bundle", type=Path)
    parser.add_argument("--supplemental-bundle", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        planner.verify_checkout(planner.ROOT, plan)
        planner.verify_generated_plan(planner.ROOT, plan)
        if args.action in {"record", "pack"}:
            require(args.acceptance is not None, f"{args.action} requires --acceptance")
            if args.action == "pack":
                require(args.output is not None, "pack requires --output")
                require(
                    args.acceptance.stat().st_size <= MAX_DECODED_BYTES,
                    "Decoded acceptance document exceeds 2 MiB",
                )
                document = json.loads(args.acceptance.read_text())
            else:
                document = load(args.acceptance)
            verify(
                planner.ROOT,
                plan,
                document,
                repository=args.repository,
                baseline_bundle=args.baseline_bundle,
                supplemental_bundles=args.supplemental_bundle,
            )
            if args.action == "pack":
                write_wire(args.output, pack(document))
            print(artifact_name(plan))
        else:
            require(args.output is not None, "fetch requires --output")
            document = fetch(args.repository, plan)
            if document is not None:
                write_wire(args.output, document)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"incremental acceptance error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
