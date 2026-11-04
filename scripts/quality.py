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

"""Bootstrap isolated quality tools and share read-only checks with hooks and CI."""

import argparse
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 12 * 1024 * 1024
TOOLS = {
    "black": ("black", ["--workers=2"]),
    "isort": ("isort", []),
    "check-ast": ("pre_commit_hooks.check_ast", []),
    "debug-statements": ("pre_commit_hooks.debug_statement_hook", []),
    "check-yaml": ("pre_commit_hooks.check_yaml", []),
    "check-toml": ("pre_commit_hooks.check_toml", []),
    "check-json": ("pre_commit_hooks.check_json", []),
    "check-case-conflict": ("pre_commit_hooks.check_case_conflict", []),
    "check-merge-conflict": ("pre_commit_hooks.check_merge_conflict", ["--assume-in-merge"]),
    "detect-private-key": ("pre_commit_hooks.detect_private_key", []),
    "check-large-files": (
        "pre_commit_hooks.check_added_large_files",
        [f"--maxkb={MAX_FILE_BYTES // 1024}", "--enforce-all"],
    ),
    "end-of-file": ("pre_commit_hooks.end_of_file_fixer", []),
    "trailing-whitespace": (
        "pre_commit_hooks.trailing_whitespace_fixer",
        ["--markdown-linebreak-ext=md"],
    ),
}


def git(*args: str, root: Path = ROOT, no_hooks: bool = False) -> bytes:
    command = ["git", "-C", str(root)]
    if no_hooks:
        command += ["-c", "core.hooksPath=/dev/null"]
    return subprocess.check_output([*command, *args])


def paths(*args: str) -> list[str]:
    return [os.fsdecode(p) for p in git(*args).split(b"\0") if p]


def store() -> Path:
    common = os.fsdecode(git("rev-parse", "--path-format=absolute", "--git-common-dir")).strip()
    return Path(common) / "ff-quality"


def environment() -> Path:
    digest = hashlib.sha256((ROOT / "requirements/quality.txt").read_bytes()).hexdigest()[:20]
    return store() / digest


def python() -> Path:
    location = environment()
    executable = location / "env/bin/python"
    if not (location / "ready").is_file() or not executable.is_file():
        raise RuntimeError(
            "Quality environment is missing for this checkout. Run make quality-env."
        )
    return executable


def run(
    command: list[str | Path], *, root: Path = ROOT, extra_env: dict[str, str] | None = None
) -> int:
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTHONPATH", None)
    env.update(extra_env or {})
    return subprocess.call([str(part) for part in command], cwd=root, env=env)


def pre_commit(*args: str, root: Path = ROOT, extra_env: dict[str, str] | None = None) -> int:
    return run([python(), "-m", "pre_commit", *args], root=root, extra_env=extra_env)


def setup() -> int:
    """Serialize bootstrap across linked worktrees without installing Git hooks."""
    location = environment()
    store().mkdir(parents=True, exist_ok=True)
    with (store() / "bootstrap.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (location / "ready").is_file() or not (location / "env/bin/python").is_file():
            venv.EnvBuilder(with_pip=True, clear=True).create(location / "env")
            executable = location / "env/bin/python"
            result = run(
                [
                    executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    ROOT / "requirements/quality.txt",
                ]
            )
            if result:
                return result
            result = run([executable, "-m", "pip", "check"])
            if result:
                return result
            (location / "ready").write_text("Installed from requirements/quality.txt\n")
    print(python())
    return 0


def actionlint(install: bool = False) -> Path:
    """Use a checksummed upstream binary without installing a compiler toolchain."""
    pins = json.loads((ROOT / "requirements/actionlint.json").read_text())
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    key = f"{platform.system().lower()}_{arch}"
    if key not in pins["checksums"]:
        raise RuntimeError(
            f"No pinned actionlint binary for {key}; supported: {list(pins['checksums'])}"
        )
    digest = pins["checksums"][key]
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", pins["version"]
    ):
        raise RuntimeError("actionlint pins require a release version and a SHA-256 digest")
    destination = store() / "actionlint" / digest / "actionlint"
    if destination.is_file():
        return destination
    if not install:
        raise RuntimeError("Pinned actionlint is not prepared. Run make quality-warm.")
    store().mkdir(parents=True, exist_ok=True)
    with (store() / "bootstrap.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.is_file():
            version = pins["version"]
            filename = f"actionlint_{version}_{key}.tar.gz"
            url = f"https://github.com/rhysd/actionlint/releases/download/v{version}/{filename}"
            print(f"Downloading {filename}", flush=True)
            with urlopen(url, timeout=30) as response:
                archive = response.read(20 * 1024 * 1024 + 1)
            if hashlib.sha256(archive).hexdigest() != digest:
                raise RuntimeError(f"SHA-256 mismatch for {filename}; refusing to install")
            with tarfile.open(fileobj=io.BytesIO(archive)) as package:
                member = package.getmember("actionlint")
                if not member.isfile():
                    raise RuntimeError("Upstream actionlint archive must contain a regular binary")
                binary = package.extractfile(member).read()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(binary)
            temporary.chmod(0o755)
            temporary.replace(destination)
    return destination


def install_hooks(pre_push: bool) -> int:
    configured = subprocess.run(["git", "config", "--get", "core.hooksPath"], cwd=ROOT)
    if configured.returncode == 0:
        raise RuntimeError(
            "core.hooksPath is already configured; it was not changed. "
            "Integrate pre-commit with that hook manager explicitly (see guidance/quality.md)."
        )
    args = ["install", "--install-hooks", "--allow-missing-config", "--hook-type", "pre-commit"]
    if pre_push:
        args += ["--hook-type", "pre-push"]
    # Native migration mode preserves an existing hook as *.legacy; never use --overwrite.
    return pre_commit(*args)


def selected_files(base: str, head: str | None, all_files: bool) -> list[str]:
    if all_files:
        if head:
            return paths("ls-tree", "-r", "--name-only", "-z", head)
        return paths("ls-files", "-z")
    comparison = os.fsdecode(git("merge-base", base, head or "HEAD")).strip()
    end = [head] if head else []
    return paths("diff", "--name-only", "--no-renames", "-z", comparison, *end, "--")


def check_files(files: list[str], root: Path, consistency_only: bool = False) -> int:
    existing = [p for p in files if (root / p).is_file() or (root / p).is_symlink()]
    if not files:
        print("No changed files selected.")
        return 0
    # Keep deletion-only changes and the comparison-base diff visible to global checks.
    # The snapshot index may equal HEAD; it cannot reconstruct this input set on its own.
    extra_env = {"FF_QUALITY_CHANGED_PATHS": json.dumps(files)}
    if consistency_only:
        return run(
            [python(), root / "scripts/check_repository.py", "--hook"],
            root=root,
            extra_env=extra_env,
        )
    return pre_commit(
        "run",
        "--hook-stage",
        "pre-commit",
        "--show-diff-on-failure",
        "--files",
        *existing,
        root=root,
        extra_env=extra_env,
    )


def check(
    base: str,
    head: str | None,
    all_files: bool,
    fix: bool = False,
    consistency_only: bool = False,
) -> int:
    """Run exactly the configured hooks; a check uses a disposable Git checkout."""
    python()  # Fail before copying if this checkout's tools have not been bootstrapped.
    files = selected_files(base, head, all_files)
    if fix:
        if head:
            raise RuntimeError("format changes working files; --head is only valid for checks")
        return check_files(files, ROOT, consistency_only)
    overlay = [] if head else paths("diff", "--name-only", "--no-renames", "-z", "HEAD", "--")
    for name in overlay:
        source = ROOT / name
        if source.is_file() and not source.is_symlink() and source.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError(
                f"{name}: exceeds 12 MiB; refusing to copy an artifact into the check"
            )
    scratch = ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quality-", dir=scratch) as directory:
        snapshot = Path(directory) / "checkout"
        git(
            "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(snapshot), no_hooks=True
        )
        target_revision = os.fsdecode(git("rev-parse", head or "HEAD")).strip()
        git("checkout", "--quiet", "--detach", target_revision, root=snapshot, no_hooks=True)
        # Remove old paths before copying replacements, including directory/file transitions.
        for name in overlay:
            source, target = ROOT / name, snapshot / name
            if target.is_file() or target.is_symlink():
                target.unlink()
            elif target.is_dir() and (source.is_file() or source.is_symlink()):
                shutil.rmtree(target)
        for name in overlay:
            source, target = ROOT / name, snapshot / name
            if source.is_dir() and not source.is_symlink():
                continue  # A submodule is owned by its own repository.
            if source.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.readlink(source))
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        (snapshot / ".git/ff-quality").symlink_to(store(), target_is_directory=True)
        git("add", "-A", root=snapshot, no_hooks=True)
        return check_files(files, snapshot, consistency_only)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup")
    commands.add_parser("warm")
    hooks = commands.add_parser("install-hooks")
    hooks.add_argument("--pre-push", action="store_true")
    for name in ("check", "fix", "consistency"):
        command = commands.add_parser(name)
        command.add_argument("--base", default="origin/main")
        command.add_argument("--head")
        command.add_argument("--all-files", action="store_true")
    commands.add_parser("push-check")
    commands.add_parser("cpu")
    commands.add_parser("pr-title")
    tool = commands.add_parser("tool")
    tool.add_argument("name", choices=[*TOOLS, "harness", "repository", "artifacts", "actionlint"])
    tool.add_argument("files", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "setup":
        return setup()
    if args.command == "warm":
        python()
        actionlint(install=True)
        return pre_commit("install-hooks")
    if args.command == "install-hooks":
        return install_hooks(args.pre_push)
    if args.command in ("check", "fix", "consistency"):
        return check(
            args.base,
            args.head,
            args.all_files,
            fix=args.command == "fix",
            consistency_only=args.command == "consistency",
        )
    if args.command == "push-check":
        base = os.environ.get("PRE_COMMIT_FROM_REF", "")
        if not base or set(base) == {"0"}:
            base = "origin/main"
        return check(base, os.environ.get("PRE_COMMIT_TO_REF") or "HEAD", False)
    if args.command == "cpu":
        return run(
            [
                python(),
                "-m",
                "pytest",
                "-q",
                "tests/docs/test_agent_harness.py",
                "tests/docs/test_docstring_hygiene.py",
                "tests/tools/test_quality.py",
                "tests/tools/test_repository_consistency.py",
                "tests/tools/test_registry_access.py",
                "tests/scripts/test_agent_scope.py",
                "tests/scripts/test_validate_validation_plan.py",
                "tests/scripts/test_validation_catalog.py",
                "tests/scripts/test_validate_gpu_campaign.py",
            ]
        )
    if args.command == "pr-title":
        title = os.environ.get("PR_TITLE", "")
        if not re.fullmatch(
            r"\[[A-Za-z0-9_./-]+(?:,[A-Za-z0-9_./-]+)*\] "
            r"(?:feat|fix|refactor|docs|test|chore): \S.*",
            title,
        ):
            raise RuntimeError("PR title must use [modules] type: description (see AGENTS.md)")
        return 0
    if args.name == "artifacts":
        raise RuntimeError(
            "Generated checkpoints/outputs must not be committed: " + ", ".join(args.files)
        )
    if args.name == "actionlint":
        return run([actionlint(), "-shellcheck=", "-pyflakes=", *args.files])
    if args.name == "harness":
        return run([python(), ROOT / "scripts/validate_agent_harness.py"])
    if args.name == "repository":
        return run([python(), ROOT / "scripts/check_repository.py", "--hook", "--", *args.files])
    module, options = TOOLS[args.name]
    return run([python(), "-m", module, *options, "--", *args.files])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError, URLError) as error:
        print(f"quality: {error}", file=sys.stderr)
        sys.exit(1)
