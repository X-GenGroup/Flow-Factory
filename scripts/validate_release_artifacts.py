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

"""Check built distributions against tracked runtime sources without installing them."""

import argparse
import configparser
import hashlib
import json
import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def validate(source: Path, wheel: Path, sdist: Path) -> dict:
    """Reject missing or altered runtime files, metadata drift and unsafe archive paths."""
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    tracked = (
        subprocess.check_output(["git", "-C", str(source), "ls-files", "-z", "src/flow_factory"])
        .decode()
        .split("\0")
    )
    expected = {
        name.removeprefix("src/"): (source / name).read_bytes()
        for name in tracked
        if name
        and (
            name.endswith(".py")
            or name.endswith("THIRD_PARTY_NOTICES.md")
            or name.endswith(".yaml")
        )
    }
    if not expected:
        raise ValueError("No tracked runtime sources")
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = {name: archive.read(name) for name in archive.namelist()}
    with tarfile.open(sdist) as archive:
        sdist_files = {}
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Unexpected archive link: {member.name}")
            if member.isfile():
                sdist_files[member.name] = archive.extractfile(member).read()
    for files in (wheel_files, sdist_files):
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or ".scratch" in path.parts:
                raise ValueError(f"Unsafe archive path: {name}")
    roots = {name.split("/")[0] for name in sdist_files}
    if len(roots) != 1:
        raise ValueError("sdist must have one top-level directory")
    root = roots.pop()
    for name, content in expected.items():
        if wheel_files.get(name) != content:
            raise ValueError(f"Wheel runtime mismatch: {name}")
        if sdist_files.get(f"{root}/src/{name}") != content:
            raise ValueError(f"sdist runtime mismatch: {name}")
    metadata_names = [name for name in wheel_files if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise ValueError("Wheel must have one metadata record")
    for data in (wheel_files[metadata_names[0]], sdist_files[f"{root}/PKG-INFO"]):
        metadata = BytesParser().parsebytes(data)
        for key, value in (
            ("Name", project["name"]),
            ("Version", project["version"]),
            ("Requires-Python", project["requires-python"]),
        ):
            if metadata[key] != value:
                raise ValueError(f"Distribution {key} differs: {metadata[key]!r} != {value!r}")
    entrypoints = configparser.ConfigParser()
    entrypoints.read_string(
        wheel_files[metadata_names[0].replace("METADATA", "entry_points.txt")].decode()
    )
    if dict(entrypoints["console_scripts"]) != project["scripts"]:
        raise ValueError("Console entry points differ from pyproject.toml")
    return {
        "status": "pass",
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]
            )
        ),
        "version": project["version"],
        "runtime_files": len(expected),
        "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"])
        .decode()
        .strip(),
        "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, sdist)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source, args.wheel, args.sdist)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
