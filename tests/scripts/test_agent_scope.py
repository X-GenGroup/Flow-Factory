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

"""Exercise real Git snapshots and semantic exemptions without loading training packages."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scope = load("agent_scope")
snapshot = load("agent_diff")
PREFIX = "import logging\nlogger = logging.getLogger(__name__)\ndef train():\n"
BEFORE = PREFIX + "    update()\n"


def git(root, *args):
    return (
        subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


def save(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value if isinstance(value, bytes) else value.encode())


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    save(tmp_path, "source.py", BEFORE)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def test_untracked_directory_is_reported_without_reading_or_expanding(repo, monkeypatch):
    save(repo, "checkpoints/nested/model.bin", b"\x00" * 4096)
    save(repo, "new.py", "new = True\n")
    original = snapshot.source
    reads = []

    def record(root, revision, path, limit):
        reads.append(path)
        return original(root, revision, path, limit)

    monkeypatch.setattr(snapshot, "source", record)
    result = snapshot.collect_snapshot(repo, "HEAD")
    assert reads == []
    assert set(result["untracked_roots"]) == {"checkpoints/", "new.py"}
    assert not result["complete"]


def test_staged_addition_rename_deletion_and_immutable_head(repo):
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", "source.py", "renamed file.py")
    save(repo, "added.py", "answer = 42\n")
    git(repo, "add", ".")
    staged = snapshot.collect_snapshot(repo, base)
    assert staged["complete"]
    assert staged["pairs"]["source.py"] == (BEFORE, None)
    assert staged["pairs"]["renamed file.py"] == (None, BEFORE)
    assert staged["pairs"]["added.py"] == (None, "answer = 42\n")
    git(repo, "commit", "-qm", "renamed and added")
    head = git(repo, "rev-parse", "HEAD")
    save(repo, "added.py", "unexpected()\n")
    save(repo, "untracked.bin", b"\0")
    frozen = snapshot.collect_snapshot(repo, base, head)
    assert frozen["pairs"] == staged["pairs"]
    assert frozen["untracked_roots"] == []
    assert frozen["base_commit"] == base and frozen["head_commit"] == head


def test_linked_worktree_is_resolved_without_following_dot_git_as_a_directory(
    repo, tmp_path_factory
):
    linked = tmp_path_factory.mktemp("linked-parent") / "checkout"
    git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
    save(linked, "source.py", PREFIX + '    logger.info("started")\n    update()\n')
    result = snapshot.collect_snapshot(linked, "HEAD")
    assert (linked / ".git").is_file()
    assert result["complete"] and list(result["pairs"]) == ["source.py"]


def test_binary_large_symlink_and_total_budget_remain_visible(repo):
    save(repo, "binary.bin", b"\x00binary")
    save(repo, "large.py", "a" * 512)
    (repo / "link.py").symlink_to("source.py")
    git(repo, "add", ".")
    result = snapshot.collect_snapshot(repo, "HEAD", file_limit=128)
    assert not result["complete"]
    assert {item["path"] for item in result["uninspected"]} == {"binary.bin", "large.py", "link.py"}
    assert set(result["paths"]) == {"binary.bin", "large.py", "link.py"}
    save(repo, "a.py", "x = 1\n")
    save(repo, "b.py", "y = 2\n")
    git(repo, "add", ".")
    result = snapshot.collect_snapshot(repo, "HEAD", total_limit=8)
    assert list(result["pairs"]) == ["a.py"]
    assert any(item["path"] == "b.py" for item in result["uninspected"])


@pytest.mark.parametrize(
    "body",
    [
        '    logger.info("started")\n    update()\n',
        '    update()\n    logger.warning("finished")\n',
    ],
)
def test_literal_console_logging_has_a_proven_exemption(body):
    docs, logging, unknown = scope.prove_edits({"base.py": (BEFORE, PREFIX + body)})
    assert docs == unknown == []
    assert logging == ["base.py"]


@pytest.mark.parametrize(
    "body",
    [
        '    logger.info("%s", update())\n    update()\n',
        '    logger.info(f"{update()}")\n    update()\n',
        '    self.logger.info("started")\n    update()\n',
        '    if enabled:\n        logger.info("started")\n        update()\n',
        '    logger.info("started", stack_info=True)\n    update()\n',
        '    logger.info("started")  # type: ignore\n    update()\n',
        '    logger.info("started")\n    changed_update()\n',
    ],
)
def test_side_effects_custom_receivers_control_flow_and_directives_keep_owner_rules(body):
    assert scope.prove_edits({"base.py": (BEFORE, PREFIX + body)})[1] == []


def test_custom_or_shadowed_logger_is_not_exempt():
    for prefix in [
        "logger = CustomLogger()\ndef train():\n",
        "import logging\nlogger = logging.getLogger(__name__)\ndef train(logger):\n",
        "import logging\nlogging = CustomFactory()\nlogger = logging.getLogger(__name__)\ndef train():\n",
    ]:
        assert not snapshot.literal_logging_only(
            prefix + "    update()\n", prefix + '    logger.info("hello")\n    update()\n'
        )


def test_mode_change_is_visible_even_with_only_a_comment_edit(repo):
    save(repo, "source.py", BEFORE + "# Updated explanation\n")
    (repo / "source.py").chmod(0o755)
    result = snapshot.collect_snapshot(repo, "HEAD")
    assert result["mode_changes"] == ["source.py"]


def test_mixed_docs_logging_and_geometry_keep_r3_and_no_fixed_campaign():
    rules = yaml.safe_load((ROOT / ".agents/harness/risk_rules.yaml").read_text())
    routes = yaml.safe_load((ROOT / ".agents/harness/test_routes.yaml").read_text())
    paths = [
        "src/flow_factory/models/abc.py",
        "src/flow_factory/trainers/abc.py",
        "src/flow_factory/utils/dist.py",
    ]
    result = scope.attach_evidence(
        scope.classify_changes(
            paths,
            {paths[2]: "dp_rank = rank // tp_size"},
            rules,
            intent="develop",
            docs_only_paths=[paths[0]],
            logging_only_paths=[paths[1]],
        ),
        routes,
    )
    assert result["profile"] == "R3"
    assert {"docs_only", "observability", "geometry"} <= set(result["facets"])
    assert not any("campaign" in evidence for evidence in result["evidence"])
