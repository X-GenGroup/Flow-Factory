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

"""Release artifacts must preserve runtime contents and installation identity."""

import importlib.util
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "release_artifacts", ROOT / "scripts/validate_release_artifacts.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def artifacts(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    runtime = tmp_path / "src/flow_factory/__init__.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("VALUE = 1\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="flow-factory"\nversion="0.5.0"\nrequires-python=">=3.11"\n'
        '[project.scripts]\nff-train="flow_factory.cli:train_cli"\n'
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
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
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    metadata = b"Name: flow-factory\nVersion: 0.5.0\nRequires-Python: >=3.11\n"
    wheel_files = {
        "flow_factory/__init__.py": runtime.read_bytes(),
        "flow_factory-0.5.0.dist-info/METADATA": metadata,
        "flow_factory-0.5.0.dist-info/entry_points.txt": b"[console_scripts]\nff-train = flow_factory.cli:train_cli\n",
    }
    sdist = tmp_path / "source.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, data in {
            "source/src/flow_factory/__init__.py": runtime.read_bytes(),
            "source/PKG-INFO": metadata,
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return tmp_path, wheel_files, sdist


@pytest.mark.parametrize(
    "corruption", [None, "missing", "changed", "version", "entrypoint", "unsafe"]
)
def test_distribution_identity(artifacts, corruption):
    root, files, sdist = artifacts
    if corruption == "missing":
        del files["flow_factory/__init__.py"]
    elif corruption == "changed":
        files["flow_factory/__init__.py"] = b"VALUE = 2\n"
    elif corruption == "version":
        files["flow_factory-0.5.0.dist-info/METADATA"] = (
            b"Name: flow-factory\nVersion: 0.4.0\nRequires-Python: >=3.11\n"
        )
    elif corruption == "entrypoint":
        files["flow_factory-0.5.0.dist-info/entry_points.txt"] = (
            b"[console_scripts]\nff-train = wrong:main\n"
        )
    elif corruption == "unsafe":
        files["../outside"] = b"unexpected"
    wheel = root / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    if corruption:
        with pytest.raises(ValueError):
            module.validate(root, wheel, sdist)
    else:
        report = module.validate(root, wheel, sdist)
        assert report["status"] == "pass" and report["runtime_files"] == 1
        assert len(report["artifacts"][wheel.name]) == 64
