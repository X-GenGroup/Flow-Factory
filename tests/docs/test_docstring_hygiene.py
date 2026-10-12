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

"""Docstring hygiene enforcement and regressions for the documentation audit tools."""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HEADER = "# Copyright 2026 Jayce-Ping\n"


def _load_script(name: str) -> ModuleType:
    """Load a repository script without making ``scripts`` a package."""
    script_path = REPOSITORY_ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through ``sys.modules``.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script("audit_docstrings")
DOCS_ONLY = _load_script("check_docs_only_diff")


def _checks(
    source: str,
    path: str = "src/flow_factory/example.py",
    *,
    header: str = HEADER,
    advisory: bool = False,
) -> List[str]:
    """Return the check identifiers reported for one dedented source snippet."""
    text = header + textwrap.dedent(source)
    return [finding.check for finding in AUDIT.audit_source(path, text, advisory=advisory)]


def test_repository_sources_pass_docstring_hygiene() -> None:
    """Keep every non-vendored source free of enforced documentation errors."""
    findings = AUDIT.audit_tree(REPOSITORY_ROOT)
    errors = [finding.format() for finding in findings if finding.severity == "error"]

    assert errors == [], "\n".join(errors)


def test_divider_followed_by_same_level_divider_is_empty_section() -> None:
    """Flag a leftover section header whose members moved elsewhere."""
    source = '''
    class Trainer:
        # ======== Advantage Computation ========
        # ======== Optimization Loop ========
        def optimize(self):
            """Optimize."""
    '''

    assert _checks(source) == ["empty-section"]


def test_divider_at_end_of_class_is_empty_section() -> None:
    """Flag a trailing section header that no longer owns any method."""
    source = '''
    class Trainer:
        def optimize(self):
            """Optimize."""

        # ======== Advantage Computation ========


    def helper():
        """Help."""
    '''

    assert _checks(source) == ["empty-section"]


def test_umbrella_divider_with_sub_sections_is_allowed() -> None:
    """Accept a top-level divider that groups deeper-level sub-sections."""
    source = '''
    class Adapter:
        # ======== Encoding & Decoding ========

        # -------- Text Encoding --------
        def encode_prompt(self):
            """Encode prompts."""
    '''

    assert _checks(source) == []


def test_divider_inside_function_body_is_flagged() -> None:
    """Require plain numbered comments for steps inside a function body."""
    source = '''
    def inference():
        """Run inference."""
        # ======== 1. Encode ========
        return 1
    '''

    assert _checks(source) == ["divider-in-function"]


def test_documented_argument_missing_from_signature_is_flagged() -> None:
    """Flag ``Args:`` entries that refer to removed or renamed parameters."""
    source = '''
    def forward(latents, guidance_scale):
        """Run one step.

        Args:
            latents: Noisy latents.
            guidance: Removed parameter.
        """
    '''

    assert _checks(source) == ["stale-arg"]


def test_keyword_argument_documentation_is_allowed_with_var_keyword() -> None:
    """Accept documented keys consumed through ``**kwargs``."""
    source = '''
    def load(path, **kwargs):
        """Load a model.

        Args:
            path: Checkpoint path.
            revision: Hub revision forwarded through kwargs.
        """
    '''

    assert _checks(source) == []


def test_class_args_are_checked_against_explicit_init() -> None:
    """Compare a class docstring's ``Args:`` with its explicit ``__init__``."""
    source = '''
    class Runtime:
        """Own components.

        Args:
            pipeline: Backend pipeline.
            aliases: Removed parameter.
        """

        def __init__(self, pipeline):
            self.pipeline = pipeline
    '''

    assert _checks(source) == ["stale-arg"]


def test_docstring_layout_violations_are_flagged() -> None:
    """Require Google layout: summary on the first line and no reST sections."""
    source = '''
    def first():
        """
        Summary on the second line.
        """


    def second(value):
        """Use reST fields.

        :param value: A value.
        """
    '''

    assert _checks(source) == ["summary-position", "non-google-section"]


def test_non_english_and_long_documentation_lines_are_flagged() -> None:
    """Keep comments English and at most 100 characters, exempting URLs."""
    long_comment = "# " + "x" * 120
    url_comment = "# https://example.com/" + "y" * 120
    source = f"""
    # 中文注释
    {long_comment}
    {url_comment}
    VALUE = 1
    """

    assert _checks(source) == ["non-english", "long-line"]


def test_headers_are_checked() -> None:
    """Require the license header and an accurate repository-path header."""
    source = """
    # src/flow_factory/other.py
    VALUE = 1
    """

    assert _checks(source, header="") == ["license-header", "path-header"]


def test_advisory_checks_run_only_on_request() -> None:
    """Keep heuristic findings out of the enforced error set."""
    source = '''
    def validate(value):
        """Validate a value.

        Returns:
            Nothing useful.
        """
        # result = compute(value)
        if value is None:
            raise ValueError("value is required")
    '''

    assert _checks(source) == []
    assert _checks(source, advisory=True) == ["returns-without-value", "commented-out-code"]


def test_docstring_and_comment_edits_are_docs_only() -> None:
    """Accept edits that change only comments, docstrings, and their layout."""
    old = '"""Old module."""\n\n\ndef f(x):\n    """Old summary."""\n    return x + 1  # old\n'
    new = (
        '"""New module.\n\nMore detail.\n"""\n\n\ndef f(x):\n'
        "    # explanatory comment\n    return x + 1\n"
    )

    assert DOCS_ONLY.compare_sources(old, new) is None


def test_bare_string_converted_to_comment_after_dedent_is_docs_only() -> None:
    """Keep the DEDENT before a class-level bare string in the compared stream."""
    old = 'class A:\n    def f(self):\n        return 1\n\n    """Section note."""\n\n    def g(self):\n        return 2\n'
    new = old.replace('"""Section note."""', "# Section note.")

    assert DOCS_ONLY.compare_sources(old, new) is None


def test_code_and_directive_changes_are_not_docs_only() -> None:
    """Reject token, formatting, and tool-directive changes."""
    old = "import os  # noqa: F401\n\n\ndef f(x):\n    return x + 1\n"

    assert "code token" in DOCS_ONLY.compare_sources(old, old.replace("x + 1", "x + 2"))
    assert "code token" in DOCS_ONLY.compare_sources(
        old, old.replace("return x + 1", "return (\n        x + 1\n    )")
    )
    assert "directive" in DOCS_ONLY.compare_sources(old, old.replace("  # noqa: F401", ""))
    assert "cannot tokenize" in DOCS_ONLY.compare_sources(old, old + "def broken(:\n")


def test_unparsable_base_source_fails_fast() -> None:
    """Report an interpreter that cannot parse the base instead of guessing a verdict."""
    with pytest.raises(ValueError, match="base source does not tokenize"):
        DOCS_ONLY.compare_sources("def broken(:\n", "VALUE = 1\n")


def test_regular_comment_mentioning_type_is_not_a_directive() -> None:
    """Treat prose such as 'Return type:' as an ordinary comment."""
    old = "def f():\n    # Return type: tensor\n    return 1\n"

    assert DOCS_ONLY.compare_sources(old, old.replace("tensor", "tensor of shape (B,)")) is None


def test_classify_paths_uses_the_base_revision(tmp_path: Path) -> None:
    """Classify modified, added, and non-Python paths against a Git base."""

    def git(*arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (tmp_path / "docs.py").write_text('def f():\n    """Old."""\n    return 1\n')
    (tmp_path / "code.py").write_text("VALUE = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    (tmp_path / "docs.py").write_text('def f():\n    """New summary."""\n    return 1\n')
    (tmp_path / "code.py").write_text("VALUE = 2\n")
    (tmp_path / "added.py").write_text("NEW = 1\n")

    verdicts = DOCS_ONLY.classify_paths(
        tmp_path, "HEAD", ["docs.py", "code.py", "added.py", "notes.md"]
    )

    assert [verdict.docs_only for verdict in verdicts] == [True, False, False, False]


def test_classify_paths_marks_unparsable_base_as_unverified(tmp_path: Path) -> None:
    """Keep normal escalation when the interpreter cannot parse the base revision."""

    def git(*arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (tmp_path / "future.py").write_text("def broken(:\n    pass\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    (tmp_path / "future.py").write_text("# comment\ndef broken(:\n    pass\n")

    (verdict,) = DOCS_ONLY.classify_paths(tmp_path, "HEAD", ["future.py"])

    assert not verdict.docs_only
    assert not verdict.verified
    assert verdict.detail.startswith(DOCS_ONLY.UNVERIFIED_PREFIX)
