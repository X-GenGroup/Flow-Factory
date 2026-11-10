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

"""Publishing requires an immutable, clean source with matching package metadata."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "release_identity", ROOT / "scripts/check_release_identity.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("fault", [None, "version", "dirty", "moved_head", "invalid_tag"])
def test_release_identity(tmp_path, fault):
    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ]
        )

    git("init", "-q")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion="0.5.0"\n')
    git("add", ".")
    git("commit", "-qm", "release")
    git("tag", "v0.5.0")
    if fault in ("version", "dirty"):
        project.write_text(
            '[project]\nversion="0.4.0"\n'
            if fault == "version"
            else project.read_text() + "# changed\n"
        )
    elif fault == "moved_head":
        git("commit", "--allow-empty", "-qm", "later")
    tag = "main" if fault == "invalid_tag" else "v0.5.0"
    if fault:
        with pytest.raises(ValueError):
            module.check(tmp_path, tag)
    else:
        assert module.check(tmp_path, tag)["version"] == "0.5.0"
