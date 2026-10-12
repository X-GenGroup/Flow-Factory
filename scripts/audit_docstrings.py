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

"""Audit comment and docstring hygiene in Flow-Factory sources.

Error checks are high precision and are enforced by ``tests/docs/test_docstring_hygiene.py``.
Advisory checks (``--advisory``) report leads such as commented-out code, identifiers that no
longer exist, and change-history narration; verify each against the code before editing.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

VENDORED_PREFIXES: Tuple[str, ...] = (
    "src/flow_factory/models/bagel/data/",
    "src/flow_factory/models/bagel/modeling/",
    "src/flow_factory/models/sensenova/modeling/",
)
DEFAULT_TARGETS: Tuple[str, ...] = ("src/flow_factory",)
MAX_LINE_LENGTH = 100
HEADER_SCAN_LINES = 25

ERROR_CHECKS: Tuple[str, ...] = (
    "license-header",
    "path-header",
    "empty-section",
    "divider-in-function",
    "stale-arg",
    "summary-position",
    "non-google-section",
    "non-english",
    "long-line",
)
ADVISORY_CHECKS: Tuple[str, ...] = (
    "returns-without-value",
    "commented-out-code",
    "history-narration",
    "dangling-identifier",
)

LICENSE_PATTERN = re.compile(r"Copyright 20\d\d Jayce-Ping")
PATH_HEADER_PATTERN = re.compile(r"^#\s*((?:src/)?flow_factory/[\w./-]+\.py)\s*$")
DIVIDER_RUN_PATTERN = re.compile(r"([=\-*#~─━])\1{3,}")
DIVIDER_LEVELS: Dict[str, int] = {"=": 1, "#": 1, "━": 1, "-": 2, "─": 2, "~": 3, "*": 3}
SECTION_HEADER_PATTERN = re.compile(
    r"^(Args|Arguments|Parameters|Params|Keyword Args|Keyword Arguments|Other Parameters|"
    r"Attributes|Returns|Return|Yields|Yield|Raises|Note|Notes|Example|Examples|References|"
    r"See Also|Warning|Warnings|Todo|Usage):\s*$"
)
ARG_ENTRY_PATTERN = re.compile(r"^(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:")
ARG_SECTIONS = ("args", "arguments", "parameters", "params")
NON_GOOGLE_PATTERN = re.compile(
    r"^\s*(?::param\b|:returns?:|:rtype:|:type \w+:|@param\b)"
    r"|^\s*(?:Parameters|Returns|Raises|Attributes|Yields|Examples|Notes)\s*\n\s*-{3,}\s*$",
    re.MULTILINE,
)
NON_ENGLISH_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
URL_PATTERN = re.compile(r"https?://\S+")
NARRATION_PATTERN = re.compile(
    r"\b(previously|no longer|used to be|formerly|pre-refactor|pre-migration|in this PR|"
    r"historically|was (?:renamed|removed|moved)|moved (?:here )?from)\b",
    re.IGNORECASE,
)
BACKTICK_PATTERN = re.compile(r"``?([^`\n]+?)``?")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SNAKE_CASE_PATTERN = re.compile(r"(?<![\w./-])(_*[a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w(])")
CODE_COMMENT_NODES = (
    ast.Call,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Return,
    ast.If,
    ast.For,
    ast.While,
    ast.With,
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
)


@dataclass(frozen=True, order=True)
class Finding:
    """One audit finding.

    Attributes:
        path: Repository-relative file path.
        line: 1-based line number.
        check: Check identifier from ``ERROR_CHECKS`` or ``ADVISORY_CHECKS``.
        message: Actionable description.
    """

    path: str
    line: int
    check: str
    message: str

    @property
    def severity(self) -> str:
        """Return ``"error"`` for enforced checks and ``"advisory"`` otherwise."""
        return "error" if self.check in ERROR_CHECKS else "advisory"

    def format(self) -> str:
        """Return the ``path:line: [check] message`` report line."""
        return f"{self.path}:{self.line}: [{self.check}] {self.message}"


@dataclass
class _Source:
    """Parsed view of one Python file shared by every check."""

    path: str
    lines: List[str]
    tree: ast.Module
    comments: List[Tuple[int, str]]
    comment_only_lines: Set[int]
    string_lines: Set[int]
    bare_strings: List[Tuple[int, str]]


def _parse(path: str, text: str) -> _Source:
    """Parse ``text`` and index its comments and bare string statements."""
    tree = ast.parse(text, filename=path)
    lines = text.splitlines()
    comments: List[Tuple[int, str]] = []
    comment_only_lines: Set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        line_number, column = token.start
        comments.append((line_number, token.string))
        if not lines[line_number - 1][:column].strip():
            comment_only_lines.add(line_number)
    string_lines: Set[int] = set()
    bare_strings: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for statement in value:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    string_lines.update(range(statement.lineno, statement.end_lineno + 1))
                    bare_strings.append((statement.lineno, statement.value.value))
    return _Source(path, lines, tree, comments, comment_only_lines, string_lines, bare_strings)


def _indent(line: str) -> int:
    """Return the number of leading spaces of ``line``."""
    return len(line) - len(line.lstrip(" "))


def _divider(line: str) -> Optional[Tuple[int, str]]:
    """Return ``(level, title)`` when ``line`` is a section divider comment."""
    body = line.strip()[1:].lstrip()
    match = DIVIDER_RUN_PATTERN.match(body)
    if match is None:
        return None
    title = DIVIDER_RUN_PATTERN.sub("", body).strip(" =-*#~─━")
    return DIVIDER_LEVELS[match.group(1)], title


def _function_bodies(tree: ast.Module) -> List[Tuple[int, int, str]]:
    """Return ``(first body line, last line, name)`` for every function."""
    return [
        (node.body[0].lineno, node.end_lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _check_headers(source: _Source) -> List[Finding]:
    """Check the license header and the optional repository-path header comment."""
    findings: List[Finding] = []
    head = source.lines[:HEADER_SCAN_LINES]
    if not any(LICENSE_PATTERN.search(line) for line in head):
        findings.append(
            Finding(
                source.path,
                1,
                "license-header",
                "missing the 'Copyright <year> Jayce-Ping' Apache header (constraints.md #24)",
            )
        )
    for line_number, line in enumerate(head, start=1):
        match = PATH_HEADER_PATTERN.match(line.strip())
        if match is None:
            continue
        declared = match.group(1)
        actual = source.path if declared.startswith("src/") else source.path.split("src/", 1)[-1]
        if declared != actual:
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "path-header",
                    f"path header declares {declared!r} but the file is {source.path!r}",
                )
            )
        break
    return findings


def _next_boundary(source: _Source, start: int, indent: int) -> Tuple[str, int, int]:
    """Scan forward from line index ``start`` for the first member or boundary.

    Returns:
        ``(kind, level, indent)`` where ``kind`` is ``"content"``, ``"divider"``,
        ``"dedent"``, or ``"eof"``; ``level`` and ``indent`` describe a divider.
    """
    for index in range(start, len(source.lines)):
        line = source.lines[index]
        if not line.strip():
            continue
        if index + 1 in source.comment_only_lines:
            divider = _divider(line)
            if divider is not None and _indent(line) <= indent:
                return "divider", divider[0], _indent(line)
            continue
        return ("content" if _indent(line) >= indent else "dedent"), 0, 0
    return "eof", 0, 0


def _check_sections(source: _Source) -> List[Finding]:
    """Flag section dividers with no members and dividers inside function bodies."""
    findings: List[Finding] = []
    bodies = _function_bodies(source.tree)
    index = 0
    while index < len(source.lines):
        line = source.lines[index]
        if index + 1 not in source.comment_only_lines or _divider(line) is None:
            index += 1
            continue
        block_indent = _indent(line)
        titled: List[Tuple[int, int, str]] = []
        while (
            index < len(source.lines)
            and index + 1 in source.comment_only_lines
            and _indent(source.lines[index]) == block_indent
        ):
            divider = _divider(source.lines[index])
            if divider is not None:
                owners = [name for first, last, name in bodies if first <= index + 1 <= last]
                if owners:
                    findings.append(
                        Finding(
                            source.path,
                            index + 1,
                            "divider-in-function",
                            f"section divider inside {owners[-1]}(); use a plain numbered comment",
                        )
                    )
                elif divider[1]:
                    titled.append((index + 1, divider[0], divider[1]))
            index += 1
        for current, following in zip(titled, titled[1:]):
            if following[1] <= current[1]:
                findings.append(
                    Finding(
                        source.path,
                        current[0],
                        "empty-section",
                        f"section {current[2]!r} has no members before {following[2]!r}",
                    )
                )
        if not titled:
            continue
        line_number, level, title = titled[-1]
        kind, next_level, next_indent = _next_boundary(source, index, block_indent)
        umbrella = kind == "divider" and next_indent == block_indent and next_level > level
        if kind != "content" and not umbrella:
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "empty-section",
                    f"section {title!r} has no members; delete it or move members under it",
                )
            )
    return findings


def _google_sections(docstring: str) -> Dict[str, List[str]]:
    """Split a cleaned docstring into Google sections keyed by lower-case header."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in docstring.splitlines():
        if line and not line[0].isspace():
            match = SECTION_HEADER_PATTERN.match(line.strip())
            current = match.group(1).lower() if match else None
            if current is not None:
                sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _documented_args(sections: Dict[str, List[str]]) -> List[str]:
    """Return parameter names documented under the ``Args:`` section."""
    names: List[str] = []
    for key in ARG_SECTIONS:
        entries = [line for line in sections.get(key, []) if line.strip()]
        if not entries:
            continue
        entry_indent = min(_indent(line) for line in entries)
        for line in entries:
            match = ARG_ENTRY_PATTERN.match(line.strip())
            if _indent(line) == entry_indent and match is not None:
                names.append(match.group(1).lstrip("*"))
    return names


def _signature(node: ast.AST) -> Tuple[List[str], bool]:
    """Return a function's parameter names (without self/cls) and whether it takes ``**kwargs``."""
    arguments = node.args
    names = [arg.arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs]
    if arguments.vararg is not None:
        names.append(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.append(arguments.kwarg.arg)
    return [name for name in names if name not in ("self", "cls")], arguments.kwarg is not None


def _returns_value(node: ast.AST) -> bool:
    """Return whether a function returns a non-``None`` value or yields."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(child, ast.Return) and child.value is not None:
            if not (isinstance(child.value, ast.Constant) and child.value.value is None):
                return True
        stack.extend(ast.iter_child_nodes(child))
    return False


def _is_stub(node: ast.AST) -> bool:
    """Return whether a function body is only a docstring, ``pass``, ``...``, or a raise."""
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if not body:
        return True
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, (ast.Pass, ast.Raise)):
        return True
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _check_docstrings(source: _Source, advisory: bool) -> List[Finding]:
    """Check docstring layout, Google sections, and ``Args:`` against signatures."""
    findings: List[Finding] = []
    owners: List[ast.AST] = [source.tree]
    owners.extend(
        node
        for node in ast.walk(source.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    for node in owners:
        raw = ast.get_docstring(node, clean=False)
        if raw is None:
            continue
        line_number = node.body[0].lineno
        label = getattr(node, "name", "module")
        if raw.lstrip(" ").startswith("\n"):
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "summary-position",
                    f"docstring of {label} must start its summary on the first line",
                )
            )
        if NON_GOOGLE_PATTERN.search(raw):
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "non-google-section",
                    f"docstring of {label} uses reST/NumPy sections; use Google style",
                )
            )
        sections = _google_sections(ast.get_docstring(node, clean=True) or "")
        target = node
        if isinstance(node, ast.ClassDef):
            target = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__"
                ),
                None,
            )
        if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters, takes_kwargs = _signature(target)
            stale = [name for name in _documented_args(sections) if name not in parameters]
            if stale and not takes_kwargs:
                findings.append(
                    Finding(
                        source.path,
                        line_number,
                        "stale-arg",
                        f"{label} documents {stale} but its parameters are {parameters}",
                    )
                )
        if (
            advisory
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and ("returns" in sections or "return" in sections)
            and not _returns_value(node)
            and not _is_stub(node)
        ):
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "returns-without-value",
                    f"{label} documents Returns: but never returns a value",
                )
            )
    return findings


def _check_text(source: _Source) -> List[Finding]:
    """Check comments and docstrings for non-English text and over-long lines."""
    findings: List[Finding] = []
    documentation_lines = source.comment_only_lines | source.string_lines
    comment_lines = {line_number for line_number, _ in source.comments}
    for line_number in sorted(documentation_lines | comment_lines):
        line = source.lines[line_number - 1]
        if NON_ENGLISH_PATTERN.search(line):
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "non-english",
                    "comments and docstrings must be English (constraints.md #21)",
                )
            )
        if (
            line_number in documentation_lines
            and len(line) > MAX_LINE_LENGTH
            and not URL_PATTERN.search(line)
        ):
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "long-line",
                    f"comment or docstring line has {len(line)} > {MAX_LINE_LENGTH} characters",
                )
            )
    return findings


def _looks_like_code(comment: str) -> bool:
    """Return whether a comment body parses as a code statement rather than prose."""
    body = comment.lstrip("#").strip()
    if not body or PATH_HEADER_PATTERN.match(f"# {body}") or body.startswith(("!", "noqa")):
        return False
    if re.fullmatch(r"[\w\s,*+\-/.()\[\]:|=<>]*", body) and "=" not in body and "(" not in body:
        return False
    words = body.replace("_", "").split()
    if len(words) >= 4 and all(word.isalpha() for word in words):
        return False
    candidates = [body, f"{body} pass"] if body.endswith(":") else [body]
    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if not any(isinstance(node, CODE_COMMENT_NODES) for node in ast.walk(tree)):
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return True
        value = tree.body[0].value
        if isinstance(value, (ast.Tuple, ast.Subscript, ast.Name)):
            return False
        # Shape or type annotations such as ``Tensor (B, C)`` parse as a capitalized call.
        callee = ast.unparse(value.func) if isinstance(value, ast.Call) else ""
        return not callee[:1].isupper()
    return False


def _check_advisory_text(source: _Source, universe: Optional[Set[str]]) -> List[Finding]:
    """Report commented-out code, history narration, and dangling identifiers."""
    findings: List[Finding] = []
    texts: List[Tuple[int, str]] = list(source.comments)
    for line_number, value in source.bare_strings:
        texts.extend((line_number + offset, line) for offset, line in enumerate(value.splitlines()))
    for line_number, comment in source.comments:
        if line_number in source.comment_only_lines and _looks_like_code(comment):
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "commented-out-code",
                    f"comment looks like disabled code: {comment.strip()[:60]!r}",
                )
            )
    for line_number, text in texts:
        prose = URL_PATTERN.sub("", text)
        match = NARRATION_PATTERN.search(prose)
        if match is not None:
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "history-narration",
                    f"describe current behavior instead of history ({match.group(0)!r})",
                )
            )
        if universe is None:
            continue
        candidates = {
            identifier
            for inner in BACKTICK_PATTERN.findall(prose)
            for identifier in IDENTIFIER_PATTERN.findall(inner)
            if len(identifier) >= 4
            and ("_" in identifier.strip("_") or re.search(r"[a-z][A-Z]", identifier))
        }
        candidates.update(word for word in SNAKE_CASE_PATTERN.findall(prose) if len(word) >= 6)
        missing = sorted(candidates - universe)
        if missing:
            findings.append(
                Finding(
                    source.path,
                    line_number,
                    "dangling-identifier",
                    f"identifiers not found in the repository: {missing}",
                )
            )
    return findings


def build_identifier_universe(root: Path) -> Set[str]:
    """Collect identifiers from repository Python code and YAML configuration.

    Args:
        root: Repository root.

    Returns:
        Every Python name, identifier inside a string literal, and YAML word found under
        ``src``, ``tests``, ``scripts``, ``examples``, and ``config``.
    """
    universe: Set[str] = set()
    for top in ("src", "tests", "scripts"):
        for path in sorted((root / top).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.NAME:
                    universe.add(token.string)
                elif token.type == tokenize.STRING:
                    universe.update(IDENTIFIER_PATTERN.findall(token.string))
    for top in ("examples", "config"):
        for pattern in ("*.yaml", "*.yml", "*.json"):
            for path in sorted((root / top).rglob(pattern)):
                universe.update(IDENTIFIER_PATTERN.findall(path.read_text(encoding="utf-8")))
    return universe


def audit_source(
    path: str,
    text: str,
    *,
    advisory: bool = False,
    universe: Optional[Set[str]] = None,
) -> List[Finding]:
    """Audit one Python source.

    Args:
        path: Repository-relative path used in findings and header checks.
        text: Python source text.
        advisory: Whether to run advisory checks in addition to error checks.
        universe: Known repository identifiers; enables ``dangling-identifier`` when given.

    Returns:
        Sorted findings.

    Raises:
        SyntaxError: If ``text`` does not parse.
    """
    source = _parse(path, text)
    findings = _check_headers(source)
    findings.extend(_check_sections(source))
    findings.extend(_check_docstrings(source, advisory))
    findings.extend(_check_text(source))
    if advisory:
        findings.extend(_check_advisory_text(source, universe))
    return sorted(set(findings))


def iter_target_files(root: Path, targets: Iterable[str]) -> List[str]:
    """Expand target files and directories into audited repository-relative Python paths.

    Args:
        root: Repository root.
        targets: Repository-relative files or directories.

    Returns:
        Sorted paths, excluding vendored upstream code and caches.

    Raises:
        FileNotFoundError: If a target does not exist.
    """
    paths: Set[str] = set()
    for target in targets:
        location = root / target
        if location.is_file():
            candidates = [location]
        elif location.is_dir():
            candidates = sorted(location.rglob("*.py"))
        else:
            raise FileNotFoundError(f"audit target {target!r} does not exist under {root}")
        for candidate in candidates:
            relative = candidate.relative_to(root).as_posix()
            if "__pycache__" in relative or relative.startswith(VENDORED_PREFIXES):
                continue
            paths.add(relative)
    return sorted(paths)


def audit_tree(
    root: Path,
    targets: Sequence[str] = DEFAULT_TARGETS,
    *,
    advisory: bool = False,
) -> List[Finding]:
    """Audit every Python file under ``targets``.

    Args:
        root: Repository root.
        targets: Repository-relative files or directories.
        advisory: Whether to run advisory checks, including ``dangling-identifier``.

    Returns:
        Sorted findings across all audited files.
    """
    universe = build_identifier_universe(root) if advisory else None
    findings: List[Finding] = []
    for path in iter_target_files(root, targets):
        text = (root / path).read_text(encoding="utf-8")
        findings.extend(audit_source(path, text, advisory=advisory, universe=universe))
    return sorted(findings)


def main() -> int:
    """Run the audit and return a non-zero exit code when error findings exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Repository-relative files or directories (default: src/flow_factory)",
    )
    parser.add_argument("--advisory", action="store_true", help="Also run advisory checks")
    parser.add_argument("--json", action="store_true", help="Emit findings as JSON")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    findings = audit_tree(root, args.targets, advisory=args.advisory)
    errors = [finding for finding in findings if finding.severity == "error"]
    if args.json:
        payload = [dict(asdict(finding), severity=finding.severity) for finding in findings]
        print(json.dumps(payload, indent=2))
    else:
        for finding in findings:
            print(finding.format())
        print(f"{len(errors)} errors, {len(findings) - len(errors)} advisory findings")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
