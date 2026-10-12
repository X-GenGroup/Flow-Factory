---
name: ff-docstring-audit
description: Audit and clean stale docstrings and comments in Flow-Factory without changing code, using mechanical hygiene checks, verification against current code, partitioned two-pass review, and a token-level docs-only proof.
---

# Docstring and Comment Audit

Use after refactors that move, rename, or remove behavior, for periodic documentation audits, and
to review any change that claims to touch only comments or docstrings.

## Context Routing

1. Read `../../knowledge/constraints.md` #21, #24, and #27 (English, license header, docstring
   style and currency) and `.cursor/rules/no-section-divider-comments.mdc`.
2. Run `python3 ../../../scripts/agent_scope.py --base origin/main --intent docs`. Python files
   proven docs-only receive the `docs_only` facet instead of path escalation.
3. Read [references/review-brief.md](references/review-brief.md) before editing or delegating.

## Workflow

1. Start from the latest `origin/main` in an isolated branch or worktree.
2. Run `python3 ../../../scripts/audit_docstrings.py --advisory`. Error findings are high
   precision and must reach zero; advisory findings (commented-out code, dangling identifiers,
   history narration, `Returns:` without a value) are leads to verify, never automatic edits.
3. Review every file in scope against the code it describes: the body, callers, overridden base
   methods, and referenced helpers. Fix stale content first, then layout, then concision. Never
   invent behavior; list unverifiable statements instead of rewriting them.
4. For a large scope, partition non-vendored files into groups of about 10k lines with disjoint
   ownership. Give each reviewer the brief, its file list, and its audit findings.
5. Run an independent second pass over the diff. Verify every changed statement, especially
   claims with "always", "only", "no-op", defaults, shapes, or callers, and search unchanged
   claim-bearing comments in the same files.
6. Prove the result: `python3 ../../../scripts/check_docs_only_diff.py --base origin/main` passes,
   `python3 -m pytest -q tests/docs/test_docstring_hygiene.py` passes, formatter results match
   the base for files with pre-existing drift, and every test that reads docstrings passes
   (`rg -n "getdoc|__doc__" tests`). Then run `/ff-review`.

## Boundaries

- Comments and docstrings only. Error messages, log text, `Field(description=...)`, and other
  string values are code; change them in a separate code commit.
- Keep directive comments, license headers, and vendored upstream code unchanged.
- Respect docstring contracts enforced by tests, such as the explicit `Returns:` sections of
  MiniMax H3 public APIs.
- Report code defects found during the audit separately instead of fixing them in a docs-only
  change.
