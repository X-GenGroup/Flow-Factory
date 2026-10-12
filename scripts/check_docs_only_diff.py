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

"""Verify that Python changes touch only comments and docstrings.

A changed file is docs-only when its token stream, after removing comments, non-logical
newlines, and bare string statements (docstrings), matches the base revision exactly and its
tool directive comments (``# noqa``, ``# type:``, ``# isort:``, ``# fmt:``, ``# pragma``, ...)
are unchanged. Such an edit cannot change runtime behavior unless code reads ``__doc__``.
"""

from __future__ import annotations

import argparse
import ast
import io
import platform
import re
import subprocess
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DIRECTIVE_PATTERN = re.compile(
    r"^#!|#\s*(?:noqa\b|type:|isort:|fmt:|pragma\b|pylint:|mypy:|pyright:)|-\*-.*coding"
)
UNVERIFIED_PREFIX = "cannot verify: "
_SKIPPED_TOKEN_TYPES = (tokenize.COMMENT, tokenize.NL)
_BLOCK_TOKEN_TYPES = (tokenize.INDENT, tokenize.DEDENT)


@dataclass(frozen=True)
class DocsOnlyVerdict:
    """Docs-only classification of one changed path.

    Attributes:
        path: Repository-relative path.
        docs_only: Whether the change touches only comments and docstrings.
        detail: Reason for the verdict, such as the first changed code token.
        verified: ``False`` when the running interpreter cannot parse the base version, so
            the edit could not be compared at all.
    """

    path: str
    docs_only: bool
    detail: str
    verified: bool = True


def _bare_string_spans(source: str) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Map the start position of every bare string statement to its end position."""
    spans: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for node in ast.walk(ast.parse(source)):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for statement in value:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    spans[(statement.lineno, statement.col_offset)] = (
                        statement.end_lineno,
                        statement.end_col_offset,
                    )
    return spans


def code_tokens(source: str) -> List[Tuple[str, str]]:
    """Return ``(token type, text)`` pairs without comments, blank lines, and docstrings.

    Args:
        source: Python source text.

    Returns:
        The token stream that determines runtime behavior, in source order.

    Raises:
        SyntaxError: If ``source`` does not parse.
        tokenize.TokenError: If ``source`` cannot be tokenized.
    """
    spans = _bare_string_spans(source)
    tokens: List[Tuple[str, str]] = []
    skip_until: Optional[Tuple[int, int]] = None
    drop_newline = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if skip_until is not None:
            if token.start < skip_until:
                continue
            skip_until = None
            drop_newline = True
        if token.type in _SKIPPED_TOKEN_TYPES:
            continue
        if drop_newline:
            drop_newline = False
            if token.type == tokenize.NEWLINE:
                continue
        # INDENT/DEDENT tokens share the start position of the statement that follows them.
        if token.start in spans and token.type not in _BLOCK_TOKEN_TYPES:
            skip_until = spans[token.start]
            continue
        tokens.append((tokenize.tok_name[token.type], token.string))
    return tokens


def directive_comments(source: str) -> Counter:
    """Count tool directive comments, which change tool behavior despite being comments.

    Args:
        source: Python source text that tokenizes.

    Returns:
        Multiset of stripped directive comment texts.
    """
    directives: Counter = Counter()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and DIRECTIVE_PATTERN.search(token.string):
            directives[token.string.strip()] += 1
    return directives


def compare_sources(old_source: str, new_source: str) -> Optional[str]:
    """Explain why ``new_source`` is not a docs-only edit of ``old_source``.

    Args:
        old_source: File content at the base revision.
        new_source: File content in the working tree.

    Returns:
        ``None`` when code tokens and directive comments are identical, otherwise a
        description of the first difference. A working-tree source that does not tokenize is
        reported as a difference because it cannot be proven docs-only.

    Raises:
        ValueError: If ``old_source`` does not tokenize, which means the interpreter cannot
            parse the repository's syntax (Flow-Factory requires Python >= 3.10).
    """
    try:
        old_tokens = code_tokens(old_source)
    except (SyntaxError, tokenize.TokenError) as error:
        raise ValueError(
            f"base source does not tokenize under Python {platform.python_version()}: {error}; "
            "run with the project interpreter (Python >= 3.10)"
        ) from error
    try:
        new_tokens = code_tokens(new_source)
    except (SyntaxError, tokenize.TokenError) as error:
        return f"cannot tokenize: {error}"
    if old_tokens != new_tokens:
        index = next(
            (
                position
                for position, pair in enumerate(zip(old_tokens, new_tokens))
                if pair[0] != pair[1]
            ),
            min(len(old_tokens), len(new_tokens)),
        )
        before = " ".join(text for _, text in old_tokens[max(0, index - 3) : index + 4])
        after = " ".join(text for _, text in new_tokens[max(0, index - 3) : index + 4])
        return f"code token {index} changed: {before!r} -> {after!r}"
    old_directives = directive_comments(old_source)
    new_directives = directive_comments(new_source)
    if old_directives != new_directives:
        removed = sorted((old_directives - new_directives).elements())
        added = sorted((new_directives - old_directives).elements())
        return f"directive comments changed: removed {removed}, added {added}"
    return None


def _git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess:
    """Run one read-only Git command without raising on a non-zero exit status."""
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _require_commit(root: Path, base: str) -> None:
    """Fail fast when ``base`` does not name a commit."""
    completed = _git(root, ["rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"])
    if completed.returncode != 0:
        raise ValueError(f"expected --base to name a Git commit in {root}, received {base!r}")


def changed_python_files(root: Path, base: str) -> List[str]:
    """Return tracked and untracked Python paths that differ from ``base``.

    Args:
        root: Repository root.
        base: Base revision.

    Returns:
        Sorted repository-relative paths.
    """
    _require_commit(root, base)
    tracked = _git(root, ["diff", "--name-only", base, "--", "*.py"])
    if tracked.returncode != 0:
        raise RuntimeError(f"git diff against {base!r} failed: {tracked.stderr.strip()}")
    untracked = _git(root, ["ls-files", "--others", "--exclude-standard", "--", "*.py"])
    if untracked.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {untracked.stderr.strip()}")
    return sorted(set(tracked.stdout.split()) | set(untracked.stdout.split()))


def classify_paths(root: Path, base: str, paths: Sequence[str]) -> List[DocsOnlyVerdict]:
    """Classify each changed path as a docs-only Python edit or not.

    Added, deleted, renamed, and non-Python paths are never docs-only. A path whose base
    version the running interpreter cannot parse is reported with ``verified=False`` rather
    than as docs-only, so callers keep its normal escalation.

    Args:
        root: Repository root.
        base: Base revision the working tree is compared against.
        paths: Repository-relative changed paths.

    Returns:
        One verdict per path, in input order.
    """
    _require_commit(root, base)
    verdicts: List[DocsOnlyVerdict] = []
    for path in paths:
        if not path.endswith(".py"):
            verdicts.append(DocsOnlyVerdict(path, False, "not a Python file"))
            continue
        target = root / path
        if not target.is_file():
            verdicts.append(DocsOnlyVerdict(path, False, "missing from the working tree"))
            continue
        old = _git(root, ["show", f"{base}:{path}"])
        if old.returncode != 0:
            verdicts.append(DocsOnlyVerdict(path, False, f"absent from {base}"))
            continue
        try:
            mismatch = compare_sources(old.stdout, target.read_text(encoding="utf-8"))
        except ValueError as error:
            # Only an unparsable base raises; the edit cannot be proven docs-only.
            verdicts.append(DocsOnlyVerdict(path, False, f"{UNVERIFIED_PREFIX}{error}", False))
            continue
        verdicts.append(DocsOnlyVerdict(path, mismatch is None, mismatch or "docs-only"))
    return verdicts


def main() -> int:
    """Check changed Python files and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="Git base revision")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Repository-relative paths to check (default: every changed Python file)",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = args.paths or changed_python_files(root, args.base)
    verdicts = classify_paths(root, args.base, paths)
    failures = [verdict for verdict in verdicts if not verdict.docs_only]
    for verdict in failures:
        print(f"NOT DOCS-ONLY: {verdict.path}: {verdict.detail}")
    print(f"checked {len(verdicts)} files against {args.base}; not docs-only: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
