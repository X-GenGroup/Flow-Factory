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

"""Bounded tracked-file input and shared diagnostics for static checks."""

import ast
import os
from dataclasses import dataclass
from pathlib import Path

from agent_diff import MAX_SOURCE_BYTES, git, source


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


class Repository:
    """Read only tracked regular files in the current quality/hook snapshot."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.paths = frozenset(
            os.fsdecode(path) for path in git(self.root, "ls-files", "-z").split(b"\0") if path
        )
        self._text: dict[str, str] = {}
        self._trees: dict[str, ast.Module] = {}
        self.bytes_read = 0
        self.findings: list[Finding] = []
        self.unverified: set[str] = set()

    def exists(self, path: str) -> bool:
        return path in self.paths and (self.root / path).exists()

    def directory(self, path: str) -> bool:
        prefix = path.rstrip("/") + "/" if path not in ("", ".") else ""
        return any(p.startswith(prefix) and self.exists(p) for p in self.paths)

    def read(self, path: str) -> str:
        if path not in self._text:
            if path not in self.paths:
                raise ValueError(f"{path}: not a tracked file")
            budget = min(MAX_SOURCE_BYTES, 32 * MAX_SOURCE_BYTES - self.bytes_read)
            data = source(self.root, None, path, budget)
            if data is None:
                raise ValueError(f"{path}: missing tracked file")
            self.bytes_read += len(data)
            self._text[path] = data.decode("utf-8")
        return self._text[path]

    def tree(self, path: str) -> ast.Module:
        if path not in self._trees:
            self._trees[path] = ast.parse(self.read(path), filename=path)
        return self._trees[path]

    def error(self, path: str, line: int, rule: str, message: str) -> None:
        self.findings.append(Finding(path, line, rule, message))


def module_name(path: str) -> str:
    name = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def import_name(path: str, node: ast.ImportFrom) -> str:
    """Resolve relative imports without importing either package."""
    if not node.level:
        return node.module or ""
    package = module_name(path)
    if not path.endswith("/__init__.py"):
        package = package.rpartition(".")[0]
    parts = package.split(".")
    if node.level > len(parts):
        raise ValueError(f"{path}:{node.lineno}: relative import escapes package")
    return ".".join(parts[: len(parts) - node.level + 1] + ([node.module] if node.module else []))


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None
