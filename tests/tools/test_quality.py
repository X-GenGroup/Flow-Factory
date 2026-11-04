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

"""Exercise native hooks and read-only checks in disposable real Git repositories."""

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("quality", ROOT / "scripts/quality.py")
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)


def execute(root, env, *args, ok=True):
    result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=120)
    if ok:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def write(root, path, text):
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def call(repo, *args, ok=True):
    root, env = repo
    return execute(root, env, sys.executable, "scripts/quality.py", *args, ok=ok)


def git(repo, *args, ok=True):
    return execute(*repo, "git", *args, ok=ok)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text("")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=str(global_config),
        GIT_CONFIG_NOSYSTEM="1",
        PRE_COMMIT_HOME=str(tmp_path / "pre-commit-cache"),
        PYTHONNOUSERSITE="1",
    )
    env.pop("PYTHONPATH", None)
    result = (root, env)
    git(result, "init", "-q", "-b", "main")
    git(result, "config", "user.name", "Quality test")
    git(result, "config", "user.email", "quality@example.invalid")
    for path in (
        "scripts/quality.py",
        "requirements/quality.txt",
        "requirements/actionlint.json",
        "pyproject.toml",
    ):
        write(root, path, (ROOT / path).read_text())
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    # These Git fixtures use the real local hooks and contain no workflows. The quality CI job
    # exercises actionlint; this CPU suite needs neither its download nor a network connection.
    config["repos"] = [item for item in config["repos"] if item["repo"] == "local"]
    config["repos"][0]["hooks"] = [
        hook for hook in config["repos"][0]["hooks"] if hook["id"] != "repository-consistency"
    ]
    write(root, ".pre-commit-config.yaml", yaml.safe_dump(config, sort_keys=False))
    write(root, ".gitignore", ".scratch/\n")
    write(root, "example.py", "first = 1\n\nsecond = 2\n")
    write(root, "remove me.py", "value = 1\n")
    git(result, "add", ".")
    git(result, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Fixture baseline")
    git(result, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(result, "checkout", "-qb", "feature-without-upstream")
    # Reuse already installed pinned tools; no installer or external service in the test loop.
    (root / ".git/ff-quality").symlink_to(QUALITY.store(), target_is_directory=True)
    return result


def test_quality_environment_has_no_training_dependencies():
    result = subprocess.run(
        [
            str(QUALITY.python()),
            "-c",
            "import importlib.util; assert importlib.util.find_spec('torch') is None",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_partial_staging_commits_only_the_index(repo):
    root, _ = repo
    call(repo, "install-hooks")
    write(root, "example.py", "first = 10\n\nsecond = 2\n")
    git(repo, "add", "example.py")
    write(root, "example.py", "first = 10\n\nsecond = 20\n")
    git(repo, "commit", "-qm", "Commit staged change")
    assert git(repo, "show", "HEAD:example.py").stdout == "first = 10\n\nsecond = 2\n"
    assert (root / "example.py").read_text() == "first = 10\n\nsecond = 20\n"


def test_partial_staging_conflict_rolls_back_fixes_without_losing_work(repo):
    root, _ = repo
    call(repo, "install-hooks")
    staged = "first=10\n\nsecond = 2\n"
    write(root, "example.py", staged)
    git(repo, "add", "example.py")
    write(root, "example.py", "first=10\n\nsecond = 20\n")
    result = git(repo, "commit", "-qm", "Must inspect formatting", ok=False)
    assert result.returncode != 0
    assert git(repo, "show", ":example.py").stdout == staged
    assert (root / "example.py").read_text() == "first=10\n\nsecond = 20\n"
    assert "Rolling back fixes" in result.stdout + result.stderr


def test_check_is_read_only_and_fix_does_not_stage(repo):
    root, _ = repo
    write(root, "example.py", "first=10\n\nsecond = 2\n")
    git(repo, "add", "example.py")
    write(root, "example.py", "first=10\n\nsecond = 20\n")
    original = (root / "example.py").read_bytes()
    index = git(repo, "diff", "--cached", "--binary").stdout
    work = git(repo, "diff", "--binary").stdout
    assert git(repo, "rev-parse", "--verify", "@{upstream}", ok=False).returncode != 0
    assert call(repo, "check", ok=False).returncode != 0
    assert (root / "example.py").read_bytes() == original
    assert git(repo, "diff", "--cached", "--binary").stdout == index
    assert git(repo, "diff", "--binary").stdout == work
    assert call(repo, "fix", ok=False).returncode != 0
    assert (root / "example.py").read_text() == "first = 10\n\nsecond = 20\n"
    assert git(repo, "diff", "--cached", "--binary").stdout == index


def test_rename_delete_and_spaces(repo):
    call(repo, "install-hooks")
    git(repo, "mv", "example.py", "renamed file.py")
    git(repo, "rm", "remove me.py")
    call(repo, "check")
    git(repo, "commit", "-qm", "Rename and delete")


@pytest.fixture
def consistency_repo(repo):
    """Exercise the real checker on a small repository whose changes affect documentation."""
    root, _ = repo
    for path in (
        "scripts/check_repository.py",
        "scripts/agent_diff.py",
        "scripts/audit_docstrings.py",
        "scripts/validate_agent_harness.py",
        *[str(p.relative_to(ROOT)) for p in (ROOT / "scripts/repo_checks").glob("*.py")],
    ):
        write(root, path, (ROOT / path).read_text())
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    config["repos"][0]["hooks"] = [
        hook for hook in config["repos"][0]["hooks"] if hook["id"] == "repository-consistency"
    ]
    write(root, ".pre-commit-config.yaml", yaml.safe_dump(config, sort_keys=False))
    write(root, "README.md", "[leaf](guidance/leaf.md#ok)\n")
    write(root, "guidance/leaf.md", "# OK\n")
    git(repo, "add", ".")
    git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Consistency baseline")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    call(repo, "install-hooks")
    return repo


def test_consistency_deletion_only_rechecks_unchanged_incoming_links(consistency_repo):
    repo = consistency_repo
    git(repo, "rm", "guidance/leaf.md")
    for command in ("check", "consistency"):
        result = call(repo, command, ok=False)
        assert result.returncode != 0
        assert "missing local target" in result.stdout + result.stderr
    result = git(repo, "commit", "-qm", "Must fail on deleted target", ok=False)
    assert result.returncode != 0
    assert "missing local target" in result.stdout + result.stderr


def test_consistency_native_hook_preserves_partial_staging(consistency_repo):
    repo = consistency_repo
    root, _ = repo
    write(root, "README.md", "[staged](guidance/leaf.md#ok)\n")
    git(repo, "add", "README.md")
    write(root, "README.md", "[unstaged](guidance/leaf.md#missing)\n")
    git(repo, "commit", "-qm", "Only inspect staged link")
    assert "#ok" in git(repo, "show", "HEAD:README.md").stdout
    assert "#missing" in (root / "README.md").read_text()


def test_consistency_unstaged_repair_cannot_hide_staged_failure(consistency_repo):
    repo = consistency_repo
    root, _ = repo
    write(root, "README.md", "[staged](guidance/leaf.md#missing)\n")
    git(repo, "add", "README.md")
    write(root, "README.md", "[unstaged repair](guidance/leaf.md#ok)\n")
    result = git(repo, "commit", "-qm", "Must inspect staged failure", ok=False)
    assert result.returncode != 0
    assert "missing anchor" in result.stdout + result.stderr
    assert "#missing" in git(repo, "show", ":README.md").stdout
    assert "#ok" in (root / "README.md").read_text()


def test_consistency_committed_check_ignores_local_repair(consistency_repo):
    repo = consistency_repo
    root, _ = repo
    write(root, "guidance/leaf.md", "# Renamed\n")
    git(repo, "add", "guidance/leaf.md")
    git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Invalid committed anchor")
    write(root, "guidance/leaf.md", "# OK\n")
    for command in ("check", "consistency"):
        result = call(repo, command, "--head", "HEAD", ok=False)
        assert result.returncode != 0
        assert "missing anchor" in result.stdout + result.stderr
    assert (root / "guidance/leaf.md").read_text() == "# OK\n"


def test_consistency_linked_worktree_uses_its_own_targets(consistency_repo):
    repo = consistency_repo
    root, env = repo
    linked = root.parent / "consistency linked tree"
    git(repo, "worktree", "add", "-qb", "linked-consistency", str(linked))
    write(linked, "guidance/leaf.md", "# Renamed\n")
    git((linked, env), "add", "guidance/leaf.md")
    result = git((linked, env), "commit", "-qm", "Must inspect linked tree", ok=False)
    assert result.returncode != 0
    assert "missing anchor" in result.stdout + result.stderr
    assert (root / "guidance/leaf.md").read_text() == "# OK\n"


def test_existing_hook_and_global_settings_are_preserved(repo):
    root, env = repo
    hook = root / ".git/hooks/pre-commit"
    content = "#!/bin/sh\nprintf 'legacy\\n' >> legacy-hook-ran\n"
    hook.write_text(content)
    hook.chmod(0o755)
    global_before = Path(env["GIT_CONFIG_GLOBAL"]).read_bytes()
    call(repo, "install-hooks")
    assert (hook.parent / "pre-commit.legacy").read_text() == content
    write(root, "README.md", "Hook integration.\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-qm", "Exercise legacy hook")
    assert (root / "legacy-hook-ran").read_text() == "legacy\n"
    assert Path(env["GIT_CONFIG_GLOBAL"]).read_bytes() == global_before


def test_existing_hooks_path_is_refused_without_changing_it(repo):
    root, env = repo
    custom = root / "custom hooks"
    custom.mkdir()
    write(custom, "pre-commit", "existing manager\n")
    git(repo, "config", "--global", "core.hooksPath", str(custom))
    before = Path(env["GIT_CONFIG_GLOBAL"]).read_bytes()
    result = call(repo, "install-hooks", ok=False)
    assert result.returncode != 0 and "core.hooksPath" in result.stderr
    assert Path(env["GIT_CONFIG_GLOBAL"]).read_bytes() == before
    assert (custom / "pre-commit").read_text() == "existing manager\n"


def test_linked_worktrees_use_active_config_and_survive_installer_removal(repo):
    root, env = repo
    first, second = root.parent / "first tree", root.parent / "second tree"
    git(repo, "worktree", "add", "-qb", "first", str(first))
    git(repo, "worktree", "add", "-qb", "second", str(second))
    call((first, env), "install-hooks")
    hook = (root / ".git/hooks/pre-commit").read_text()
    assert str(first) not in hook
    git(repo, "worktree", "remove", str(first))
    config = yaml.safe_load((second / ".pre-commit-config.yaml").read_text())
    config["repos"][0]["hooks"].append(
        {
            "id": "active-worktree-policy",
            "name": "Active worktree policy",
            "language": "system",
            "entry": "python3 scripts/quality.py tool artifacts",
            "files": "^blocked.txt$",
        }
    )
    write(second, ".pre-commit-config.yaml", yaml.safe_dump(config, sort_keys=False))
    git((second, env), "add", ".pre-commit-config.yaml")
    git((second, env), "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Second tree policy")
    write(second, "blocked.txt", "Only this worktree rejects this filename.\n")
    git((second, env), "add", "blocked.txt")
    result = git((second, env), "commit", "-qm", "Check active policy", ok=False)
    assert result.returncode != 0
    assert "Active worktree policy" in result.stdout + result.stderr
    assert "Generated checkpoints/outputs" in result.stdout + result.stderr


def test_old_branch_without_config_remains_usable(repo):
    root, _ = repo
    call(repo, "install-hooks")
    git(repo, "rm", ".pre-commit-config.yaml")
    git(repo, "commit", "-qm", "Represent a branch before hook rollout")
    write(root, "legacy.txt", "Old branch work.\n")
    git(repo, "add", "legacy.txt")
    git(repo, "commit", "-qm", "Work on old branch")


def test_pre_push_checks_committed_content(repo):
    root, env = repo
    call(repo, "install-hooks", "--pre-push")
    assert (root / ".git/hooks/pre-push").is_file()
    write(root, "example.py", "def invalid(\n")
    git(repo, "add", "example.py")
    git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Invalid committed syntax")
    write(root, "example.py", "fixed_locally = True\n")
    push_env = dict(env, PRE_COMMIT_TO_REF=git(repo, "rev-parse", "HEAD").stdout.strip())
    result = call((root, push_env), "push-check", ok=False)
    assert result.returncode != 0
    assert (root / "example.py").read_text() == "fixed_locally = True\n"


def test_prompt_fixture_upstream_bytes_and_markdown_breaks_are_preserved(repo):
    root, _ = repo
    samples = {
        "README.md": "Hard break.  \nNext line.\n",
        "config/prompt.yaml": "prompt: |+\n  intentional spaces  \n\n",
        "dataset/prompts.txt": "prompt bytes  \n\n",
        "tests/fixtures/prompt.py": 'prompt="""literal  \n"""\n',
        "src/flow_factory/models/bagel/modeling/upstream.py": "value=1\n",
    }
    for path, content in samples.items():
        write(root, path, content)
    git(repo, "add", ".")
    call(repo, "fix")
    assert all((root / path).read_text() == content for path, content in samples.items())


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("broken.py", "def broken(\n"),
        ("broken.yaml", "key: [\n"),
        ("broken.toml", "key = [\n"),
        ("broken.json", '{"key":\n'),
        ("merge.txt", "<" * 7 + " HEAD\nconflict\n"),
        ("private.pem", "-----BEGIN " + "PRIVATE KEY-----\nexample\n"),
        ("weights.safetensors", "small checkpoint\n"),
        (".scratch/small.safetensors", "force-added ignored checkpoint\n"),
    ],
)
def test_accident_checks_reject_invalid_staged_files(repo, filename, content):
    write(repo[0], filename, content)
    git(repo, "add", "-f", filename)
    assert call(repo, "check", ok=False).returncode != 0


def test_large_force_added_artifact_is_rejected_before_copy(repo):
    root, _ = repo
    artifact = root / ".scratch/weights.safetensors"
    artifact.parent.mkdir()
    with artifact.open("wb") as stream:
        stream.truncate(QUALITY.MAX_FILE_BYTES + 1)
    git(repo, "add", "-f", ".scratch/weights.safetensors")
    result = call(repo, "check", ok=False)
    assert result.returncode != 0 and "refusing to copy" in result.stderr
    assert not list((root / ".scratch").glob("quality-*"))


@pytest.mark.parametrize(
    "title,valid", [("[ci,harness] chore: add hooks", True), ("update", False)]
)
def test_pr_title_convention(repo, title, valid):
    root, env = repo
    result = call((root, dict(env, PR_TITLE=title)), "pr-title", ok=False)
    assert (result.returncode == 0) == valid


def test_directory_replaced_by_file_is_actually_checked(repo):
    root, _ = repo
    write(root, "module.py/old.py", "value = 1\n")
    git(repo, "add", "module.py/old.py")
    git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Directory before replacement")
    shutil.rmtree(root / "module.py")
    write(root, "module.py", "def invalid(\n")
    git(repo, "add", "-A")
    result = call(repo, "check", ok=False)
    assert result.returncode != 0
    assert "module.py" in result.stdout + result.stderr
    assert (root / "module.py").read_text() == "def invalid(\n"


@pytest.mark.parametrize("failure", [None, "checksum", "symlink"])
def test_actionlint_install_verifies_archive_before_atomic_install(tmp_path, monkeypatch, failure):
    binary = b"verified binary fixture"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("actionlint")
        if failure == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            archive.addfile(member)
        else:
            member.size = len(binary)
            archive.addfile(member, io.BytesIO(binary))
    content = buffer.getvalue()
    digest = "0" * 64 if failure == "checksum" else hashlib.sha256(content).hexdigest()
    write(
        tmp_path,
        "requirements/actionlint.json",
        json.dumps({"version": "1.7.12", "checksums": {"linux_amd64": digest}}),
    )
    monkeypatch.setattr(QUALITY, "ROOT", tmp_path)
    monkeypatch.setattr(QUALITY, "store", lambda: tmp_path / "cache")
    monkeypatch.setattr(QUALITY.platform, "system", lambda: "Linux")
    monkeypatch.setattr(QUALITY.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(QUALITY, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    destination = tmp_path / "cache/actionlint" / digest / "actionlint"
    if failure:
        with pytest.raises(RuntimeError, match="SHA-256 mismatch|regular binary"):
            QUALITY.actionlint(install=True)
        assert not destination.exists()
    else:
        assert QUALITY.actionlint(install=True).read_bytes() == binary
        assert destination.stat().st_mode & 0o777 == 0o755
        monkeypatch.setattr(QUALITY, "urlopen", lambda *a, **kw: pytest.fail("Cached tool fetched"))
        assert QUALITY.actionlint() == destination
