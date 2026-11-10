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

import base64
import binascii
import copy
import gzip
import hashlib
import io
import json
import subprocess
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import validate_validation_plan as validator

MAX_BUNDLE_BYTES = 60000
MAX_DECODED_BUNDLE_BYTES = 2 * 1024 * 1024
_BUNDLE_FIELDS = {"schema_version", "base_commit", "tested_commit", "results", "files"}
_ENVELOPE_FIELDS = {
    "schema_version",
    "base_commit",
    "tested_commit",
    "encoding",
    "decoded_bytes",
    "decoded_sha256",
    "payload",
}


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _dispatch_size(value: dict) -> int:
    # Match validation_plan.write_json, including its final newline.
    return len(json.dumps(value, indent=2, sort_keys=True).encode()) + 1


def _decode_bundle(plan: dict, bundle: dict) -> dict:
    """Validate a bounded envelope before materializing its schema1 result document."""
    if not isinstance(bundle, dict):
        raise ValueError("Evidence bundle must be a mapping")
    if len(json.dumps(bundle).encode()) > MAX_BUNDLE_BYTES:
        raise ValueError("Evidence bundle exceeds 60 KB")
    if (
        type(bundle.get("schema_version")) is not int
        or bundle["schema_version"] not in (1, 2)
        or bundle.get("base_commit") != plan["base_commit"]
        or bundle.get("tested_commit") != plan["tested_commit"]
    ):
        raise ValueError("Evidence bundle source identity mismatch")
    if bundle["schema_version"] == 1:
        return bundle
    if set(bundle) != _ENVELOPE_FIELDS or bundle["encoding"] != "gzip+base64":
        raise ValueError("Invalid compressed evidence envelope")
    inner = decode_payload(bundle, max_decoded_bytes=MAX_DECODED_BUNDLE_BYTES)
    if (
        set(inner) != _BUNDLE_FIELDS
        or type(inner["schema_version"]) is not int
        or inner["schema_version"] != 1
        or inner["base_commit"] != bundle["base_commit"]
        or inner["tested_commit"] != bundle["tested_commit"]
    ):
        raise ValueError("Compressed evidence inner source identity mismatch")
    return inner


def compressed_payload(decoded: bytes) -> dict:
    """Encode canonical bytes deterministically; the caller owns schema and size limits."""
    return {
        "encoding": "gzip+base64",
        "decoded_bytes": len(decoded),
        "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
        "payload": base64.b64encode(gzip.compress(decoded, mtime=0)).decode("ascii"),
    }


def decode_payload(bundle: dict, *, max_decoded_bytes: int, label: str = "evidence") -> dict:
    """Decode one bounded canonical JSON payload; callers validate envelope/source fields."""
    size = bundle["decoded_bytes"]
    if type(size) is not int or not 0 < size <= max_decoded_bytes:
        raise ValueError(f"Decoded {label} bundle exceeds limit or has invalid length")
    try:
        encoded = bundle["payload"].encode("ascii")
        compressed = base64.b64decode(encoded, validate=True)
        if base64.b64encode(compressed) != encoded:
            raise ValueError("Noncanonical base64")
    except (AttributeError, UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid compressed {label} base64 payload") from exc
    decoder = zlib.decompressobj(wbits=31)
    try:
        decoded = decoder.decompress(compressed, size + 1)
    except zlib.error as exc:
        raise ValueError(f"Invalid compressed {label} gzip payload") from exc
    if len(decoded) != size:
        raise ValueError(f"Decoded {label} bundle length mismatch")
    # Do not flush or follow additional gzip members: either could expand past the bound.
    if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError(f"Incomplete or trailing compressed {label} gzip payload")
    if hashlib.sha256(decoded).hexdigest() != bundle["decoded_sha256"]:
        raise ValueError(f"Decoded {label} bundle hash mismatch")
    try:
        inner = json.loads(decoded.decode("utf-8"))
        if not isinstance(inner, dict) or _canonical_bytes(inner) != decoded:
            raise ValueError("Noncanonical inner document")
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"Invalid canonical {label} JSON payload") from exc
    return inner


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
    """Package complete evidence, compressing only when schema1 exceeds the dispatch bound.

    Schema2 retains source identity outside deterministic gzip/base64 and stores
    canonical schema1 JSON with its decoded byte count and SHA256. Both the
    60KB dispatch and 2MiB decoded limits apply; small schema1 bundles are unchanged.
    """
    validator.validate_results(
        plan, results, expected_commit=plan["tested_commit"], require_coverage=False
    )
    verify_files(root, results)
    files = {}
    for case in results["cases"]:
        for relative in case.get("evidence", {}):
            # Preserve progress-bar carriage returns: hashes bind the original UTF-8 bytes.
            files[relative] = (root / relative).read_bytes().decode("utf-8")
    bundle = {
        "schema_version": 1,
        "base_commit": plan["base_commit"],
        "tested_commit": plan["tested_commit"],
        "results": results,
        "files": files,
    }
    if _dispatch_size(bundle) <= MAX_BUNDLE_BYTES:
        return bundle
    decoded = _canonical_bytes(bundle)
    if len(decoded) > MAX_DECODED_BUNDLE_BYTES:
        raise ValueError("Decoded evidence bundle exceeds 2 MiB")
    envelope = {
        "schema_version": 2,
        "base_commit": plan["base_commit"],
        "tested_commit": plan["tested_commit"],
        **compressed_payload(decoded),
    }
    if _dispatch_size(envelope) > MAX_BUNDLE_BYTES:
        raise ValueError("Compressed evidence bundle exceeds 60 KB")
    return envelope


def import_bundle(root: Path, plan: dict, bundle: dict) -> dict:
    """Validate first, then place immutable logs under an isolated ignored directory."""
    bundle = _decode_bundle(plan, bundle)
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
