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

"""Bounded, read-only source snapshots and conservative console-log classification."""

import ast
import difflib
import os
import stat
import subprocess
from pathlib import Path

MAX_SOURCE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * MAX_SOURCE_BYTES


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def commit(root: Path, revision: str) -> str:
    return (
        git(root, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
        .decode()
        .strip()
    )


def changed_paths(root: Path, base: str, head: str | None = None) -> list[str]:
    """Keep both sides of renames; staged additions are tracked, untracked trees are not."""
    args = ["diff", "--name-only", "-z", "--no-renames", "--no-ext-diff", "--no-textconv", base]
    if head is not None:
        args.append(head)
    return sorted(os.fsdecode(path) for path in git(root, *args, "--").split(b"\0") if path)


def changed_modes(root: Path, base: str, head: str | None = None) -> list[str]:
    """Keep executable-bit and file-type changes outside source-only exemptions."""
    args = ["diff", "--raw", "-z", "--no-renames", "--no-ext-diff", base]
    if head is not None:
        args.append(head)
    records = git(root, *args, "--").split(b"\0")
    return [
        os.fsdecode(path)
        for header, path in zip(records[::2], records[1::2])
        if header.split()[0][1:] != header.split()[1]
    ]


def untracked_roots(root: Path) -> list[str]:
    """Report untracked directory names without enumerating their contents."""
    return [
        os.fsdecode(path)
        for path in git(
            root, "ls-files", "--others", "--exclude-standard", "--directory", "-z"
        ).split(b"\0")
        if path
    ]


def source(root: Path, revision: str | None, path: str, limit: int) -> bytes | None:
    """Read regular file bytes at a revision or in the worktree, bounded before decoding."""
    if revision is not None:
        entry = git(root, "ls-tree", "-z", revision, "--", f":(literal){path}")
        if not entry:
            return None
        metadata = entry.split(b"\t", 1)[0].split()
        if metadata[0] not in (b"100644", b"100755"):
            raise ValueError("not a regular Git file")
        oid = metadata[2].decode()
        size = int(git(root, "cat-file", "-s", oid))
        if size > limit:
            raise ValueError(f"source exceeds byte budget ({size} > {limit})")
        data = git(root, "cat-file", "blob", oid)
    else:
        target = root / path
        for parent in [target, *target.parents]:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError("symlink source is not inspected")
        try:
            info = target.stat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular worktree file")
        if info.st_size > limit:
            raise ValueError(f"source exceeds byte budget ({info.st_size} > {limit})")
        with target.open("rb") as stream:
            data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("source grew beyond byte budget")
    return data


def collect_snapshot(
    root: Path,
    base: str,
    head: str | None = None,
    *,
    file_limit: int = MAX_SOURCE_BYTES,
    total_limit: int = MAX_TOTAL_BYTES,
) -> dict:
    """Collect tracked changes; opaque and untracked inputs remain visible as incomplete scope."""
    base = commit(root, base)
    head = commit(root, head) if head is not None else None
    paths = changed_paths(root, base, head)
    pairs, diffs, issues = {}, {}, []
    remaining = total_limit

    def read_text(revision: str | None, path: str) -> str | None:
        nonlocal remaining
        data = source(root, revision, path, min(file_limit, remaining))
        if data is None:
            return None
        remaining -= len(data)
        if b"\0" in data:
            raise ValueError("binary source is not inspected")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("source is not UTF-8 text") from error

    for path in paths:
        try:
            before = read_text(base, path)
            after = read_text(head, path)
        except (OSError, ValueError) as error:
            issues.append({"path": path, "reason": str(error)})
            continue
        pairs[path] = (before, after)
        diffs[path] = "".join(
            difflib.unified_diff(
                (before or "").splitlines(keepends=True),
                (after or "").splitlines(keepends=True),
                fromfile=path,
                tofile=path,
                n=0,
            )
        )
    untracked = untracked_roots(root) if head is None else []
    return {
        "base_commit": base,
        "head_commit": head,
        "paths": paths,
        "pairs": pairs,
        "mode_changes": changed_modes(root, base, head),
        "diffs": diffs,
        "uninspected": issues,
        "untracked_roots": untracked,
        "complete": not issues and not untracked,
        "limits": {"per_source_bytes": file_limit, "total_source_bytes": total_limit},
    }


def _console_logger(tree: ast.Module) -> bool:
    """Recognize the existing module console logger, rejecting shadowed/custom logger bindings."""
    factories = set()
    factory_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name == "logging" and alias.asname is None for alias in node.names
        ):
            factories.add("logging.getLogger")
            factory_imports.update(id(alias) for alias in node.names if alias.name == "logging")
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            in (
                "utils.logger_utils",
                "logger_utils",
                "flow_factory.utils.logger_utils",
            )
            and any(alias.name == "setup_logger" and alias.asname is None for alias in node.names)
        ):
            factories.add("setup_logger")
            factory_imports.update(
                id(alias) for alias in node.names if alias.name == "setup_logger"
            )
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "logger" and isinstance(node.ctx, ast.Store)
    ]
    if len(assignments) != 1:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg in {"logger", "logging", "setup_logger"}:
            return False
        if (
            isinstance(node, ast.Name)
            and node.id in {"logging", "setup_logger"}
            and isinstance(node.ctx, ast.Store)
        ):
            return False
        if (
            isinstance(node, ast.alias)
            and (node.asname or node.name.split(".")[0]) in {"logger", "logging", "setup_logger"}
            and id(node) not in factory_imports
        ):
            return False
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)
        ) and node.name in {"logger", "logging", "setup_logger"}:
            return False
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and ast.unparse(node).startswith(("logger.", "logging.", "setup_logger."))
        ):
            return False
    return any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "logger"
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) in factories
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "__name__"
        and not node.value.keywords
        for node in tree.body
    )


class _RemoveLiteralLogs(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logger"
            and call.func.attr in {"debug", "info", "warning", "error", "critical"}
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            and not call.keywords
        ):
            return None
        return node


def literal_logging_only(before: str, after: str) -> bool:
    """Prove AST equality except literal console log statements; directives are checked by caller.

    This is a narrow routing exemption, not proof about custom handlers or logging performance.
    Calls with formatting arguments, f-strings, custom receivers, or changed control flow remain
    subject to the normal owner rules.
    """
    try:
        old, new = ast.parse(before, type_comments=True), ast.parse(after, type_comments=True)
        if not _console_logger(old) or not _console_logger(new):
            return False
        if ast.dump(old) == ast.dump(new):
            return False
        return ast.dump(_RemoveLiteralLogs().visit(old)) == ast.dump(
            _RemoveLiteralLogs().visit(new)
        )
    except (SyntaxError, ValueError, RecursionError):
        return False
