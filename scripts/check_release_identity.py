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

"""Refuse release publication when the immutable tag and package identity disagree."""

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path


def check(source: Path, tag: str) -> dict:
    """Validate a stable release tag against the checked-out commit and package version."""
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError(f"Expected a stable vX.Y.Z release tag, received {tag!r}")

    def revision(ref):
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "--verify", ref], text=True
        ).strip()

    head = revision("HEAD")
    tagged = revision(f"refs/tags/{tag}^{{commit}}")
    if head != tagged:
        raise ValueError(f"Tag {tag} points to {tagged}, but checkout is {head}")
    version = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
    if version != tag[1:]:
        raise ValueError(f"Package version {version!r} differs from {tag!r}")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]
    ):
        raise ValueError("Release source contains tracked modifications")
    return {"tag": tag, "commit": head, "version": version}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.tag), indent=2))


if __name__ == "__main__":
    main()
