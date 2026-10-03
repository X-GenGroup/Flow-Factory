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

"""Validate the structural contracts of the repository agent harness."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import yaml

FRONTMATTER_PATTERN = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
RELATIVE_MARKDOWN_PATTERN = re.compile(r"`(?P<path>\.\.?/[^`#]+\.md)(?:#[^`]*)?`")
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _read(path: Path) -> str:
    """Read one UTF-8 harness file."""
    return path.read_text(encoding="utf-8")


def _parse_frontmatter(path: Path, errors: List[str]) -> Dict[str, object]:
    """Return parsed YAML frontmatter and append actionable validation errors."""
    text = _read(path)
    match = FRONTMATTER_PATTERN.match(text)
    if match is None:
        errors.append(f"{path}: missing YAML frontmatter")
        return {}
    try:
        data = yaml.safe_load(match.group("body"))
    except yaml.YAMLError as error:
        errors.append(f"{path}: invalid YAML frontmatter: {error}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"{path}: frontmatter must be a mapping")
        return {}
    return data


def _validate_skills(root: Path, errors: List[str]) -> None:
    """Validate skill discovery, registration, routing, and local references."""
    agents_text = _read(root / "AGENTS.md")
    skills_readme = _read(root / ".agents/skills/README.md")
    seen_names = set()

    for skill_path in sorted((root / ".agents/skills").glob("*/SKILL.md")):
        metadata = _parse_frontmatter(skill_path, errors)
        name = metadata.get("name")
        description = metadata.get("description")
        expected_name = skill_path.parent.name
        if name != expected_name:
            errors.append(
                f"{skill_path}: frontmatter name {name!r} must match directory {expected_name!r}"
            )
        if not isinstance(name, str) or SKILL_NAME_PATTERN.fullmatch(name) is None:
            errors.append(f"{skill_path}: invalid skill name {name!r}")
        elif name in seen_names:
            errors.append(f"{skill_path}: duplicate skill name {name!r}")
        else:
            seen_names.add(name)
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{skill_path}: description must be a non-empty string")

        text = _read(skill_path)
        if text.count("## Context Routing") != 1:
            errors.append(f"{skill_path}: expected exactly one '## Context Routing' section")
        if f"/{expected_name}" not in agents_text:
            errors.append(f"{skill_path}: skill is not registered in AGENTS.md")
        if f"`{expected_name}`" not in skills_readme:
            errors.append(f"{skill_path}: skill is not registered in .agents/skills/README.md")

        for reference in RELATIVE_MARKDOWN_PATTERN.finditer(text):
            target = (skill_path.parent / reference.group("path")).resolve()
            if not target.is_file():
                errors.append(
                    f"{skill_path}: relative Markdown reference does not exist: "
                    f"{reference.group('path')}"
                )


def _validate_knowledge(root: Path, errors: List[str]) -> None:
    """Validate knowledge leaf structure and known cross-layer drift points."""
    topics_dir = root / ".agents/knowledge/topics"
    for topic_path in sorted(topics_dir.glob("*.md")):
        if "## Cross-refs" not in _read(topic_path):
            errors.append(f"{topic_path}: missing '## Cross-refs' section")

    constraints = _read(root / ".agents/knowledge/constraints.md")
    if "## Code Quality (21–27)" not in constraints:
        errors.append("constraints.md: missing Code Quality (21–27) section")
    if "## Agent Workflow (28–30)" not in constraints:
        errors.append("constraints.md: missing Agent Workflow (28–30) section")

    architecture = _read(root / ".agents/knowledge/architecture.md")
    if re.search(r"^#### .+\n- \*\*Date\*\*:", architecture, re.MULTILINE):
        errors.append("architecture.md: historical fix records must live in a knowledge leaf")

    maintenance = _read(root / ".agents/knowledge/docs_maintenance.md")
    if "one `## Context Routing` section" not in maintenance:
        errors.append("docs_maintenance.md: skill routing rule is not canonical")


def validate_harness(root: Path) -> List[str]:
    """Return every harness validation error under ``root``."""
    errors: List[str] = []
    required_paths = (
        root / "AGENTS.md",
        root / "CLAUDE.md",
        root / ".agents/knowledge/README.md",
        root / ".agents/knowledge/constraints.md",
        root / ".agents/knowledge/architecture.md",
        root / ".agents/knowledge/docs_maintenance.md",
        root / ".agents/skills/README.md",
    )
    for path in required_paths:
        if not path.is_file():
            errors.append(f"missing required harness file: {path}")
    if errors:
        return errors

    if _read(root / "CLAUDE.md") != "@AGENTS.md\n":
        errors.append("CLAUDE.md must contain only '@AGENTS.md'")

    cursor_contract = _read(root / ".cursor/rules/base-class-contract.mdc")
    if "TDMR1Trainer → TDMTrainer" not in cursor_contract:
        errors.append("base-class-contract.mdc: missing sanctioned TDM-R1 inheritance")

    _validate_skills(root, errors)
    _validate_knowledge(root, errors)
    return errors


def main() -> int:
    """Run harness validation and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to validate.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors = validate_harness(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Agent harness validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
