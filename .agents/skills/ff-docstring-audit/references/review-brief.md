# Docstring Audit Reviewer Brief

Give this brief, a file list, and the matching `scripts/audit_docstrings.py --advisory` findings to
every reviewer. Reviewers edit only their own files and never run state-changing Git commands or
whole-file formatters (some files carry pre-existing formatter drift).

## Hard Boundaries

1. Change only comments and docstrings (bare string statements). Never change a code token: no
   renames, no reflowed code, no edits to string values such as error or log messages.
2. If a docstring is the only statement in a body, edit it but never delete it.
3. Keep directive comments (`# noqa`, `# type:`, `# isort:`, `# fmt:`, `# pragma`), license headers,
   and vendored directories unchanged.
4. Write English, keep comment and docstring lines within 100 characters (URLs exempt), and match
   the surrounding indentation.
5. Verify every statement against current code: the body, callers, base-class contracts, and
   helpers. Leave unverifiable text unchanged and report it.

## What to Fix, in Priority Order

**A. Stale content**
- Dividers without members: delete them. An umbrella `# ====` divider followed by `# ----`
  sub-dividers is an intentional hierarchy.
- Divider titles that no longer match the members beneath: rename or insert a divider in the
  file's style. Dividers inside function bodies become plain numbered comments.
- `Args:` entries for removed or renamed parameters; missing entries a caller needs.
- `Returns:` on functions that return nothing (unless a test-enforced contract requires it) and
  wrong return, shape, default, dtype, or raise descriptions.
- Text that names removed functions, attributes, config fields, files, or stages, and outdated
  examples or doctests.
- Commented-out code. Shape annotations such as `# (B, C, H, W)` are documentation.
- Change-history narration ("previously", "pre-refactor", "in this PR", "historically"):
  restate the current behavior or constraint. In this repository "legacy" is also a domain term
  for the single-latent trajectory path and migrated config locations.
- Completed or obsolete TODO notes.

**B. Layout (Google style)**
- Summary on the first line after the opening quotes, one line, ending with a period; a blank
  line before further content.
- Sections `Args:`, `Returns:`, `Yields:`, `Raises:`, `Note:`, `Example:`, `References:`,
  `Attributes:`; convert `Usage:` and reST/NumPy forms. One blank line between sections, bodies
  indented four spaces, no blank line before the closing quotes.

**C. Concision**: remove text that restates code line by line. Keep contracts, shapes, units,
invariants, failure modes, distributed or precision constraints, and non-obvious reasons.
Do not rewrite accurate, well-formatted docstrings for taste.

**D. Language**: translate non-English comments and docstrings.

Do not add docstrings to undocumented functions unless a removal leaves a reader without needed
information.

## Second-Pass Verification

1. Read `git diff origin/main -- <files>` hunk by hunk and verify each added statement, fixing
   overstated, too narrow, or wrong text and restoring deleted constraints that still hold.
2. Explicitly check the risky claims reported by the first pass.
3. Search unchanged comments for claim-bearing words (`no-op`, `only`, `always`, `never`,
   `same as`, `fallback`, `by default`, `called by`, `used by`) and verify them.

## Evidence and Report

Run `python3 scripts/check_docs_only_diff.py --base origin/main <files>` (must report zero
non-docs files) and `black --check` on changed files. Report one line per changed file with the
categories fixed, an unsure list with `file:line` and reason, any code defects noticed, and the
checker output.

## Cross-refs

- UP: [`ff-docstring-audit`](../SKILL.md), `../../../knowledge/constraints.md` #21, #24, #27
