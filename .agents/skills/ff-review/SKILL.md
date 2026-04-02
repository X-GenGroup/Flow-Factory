---
name: ff-review
description: Pre-commit code review with constraint checking
---

# Code Review Workflow

## Process Overview

```
1. Capture changes → git diff
2. Load constraints → .agents/knowledge/constraints.md
3. Review against constraints and architecture
4. Route by verdict:
   ✓ Safe → Proceed with commit
   ⚠ Needs-attention → Fix issues, then commit
   ✗ Risky → Halt and report
```

## Step 1: Capture Changes

```bash
git diff HEAD          # All changes
git status             # Modified files
```

## Step 2: Load Context

- Read `.agents/knowledge/constraints.md` — All 24 hard constraints
- Reference `.agents/knowledge/architecture.md` — Module dependencies
- Identify which modules are affected by the changes

## Step 3: Review Checklist

### Constraint Compliance
- [ ] No constraint violations found (check all 24 constraints)
- [ ] Registry entries updated if classes moved/renamed (#1–4)
- [ ] Pipeline order preserved (#6)
- [ ] Coupled/decoupled paradigm respected (#7)
- [ ] Base class interfaces not broken (#11–13)
- [ ] Config fields synchronized with YAML examples (#15–17)

### Cross-Module Consistency
- [ ] Changes to `abc.py` base classes reflected in ALL subclasses
- [ ] Changes to `hparams/` reflected in ALL example configs
- [ ] Registry keys match actual import paths
- [ ] Sample dataclass `_shared_fields` consistent

### Implementation Quality
- [ ] No hardcoded devices (use `self.device` or `accelerator.device`)
- [ ] `@torch.no_grad()` on reward model `__call__`
- [ ] Proper synchronization barriers for distributed code
- [ ] No ZeRO-3 usage
- [ ] Type annotations on public methods

### Code Style
- [ ] Black formatting (`line-length=100`)
- [ ] isort compliance (`profile="black"`)
- [ ] English comments and docstrings
- [ ] Apache 2.0 license header on new files
- [ ] No unnecessary wildcard imports (except `hparams`)

### Documentation
- [ ] `guidance/` docs updated if behavior changed
- [ ] New features documented in relevant guidance file
- [ ] PR title follows format: `[{modules}] {type}: {description}`

## Step 4: Route by Verdict

### ✓ Safe
No issues found. Proceed with commit.

### ⚠ Needs-Attention
Issues found but fixable:
1. List each issue with file and line
2. Fix identified problems
3. Re-stage and re-review

### ✗ Risky
Potential breaking changes:
1. Halt commit
2. Report findings with severity
3. Await explicit user approval

## Common Issues Found in Review

1. **Registry path stale** — Class moved but registry not updated
2. **Config field renamed** — YAML examples still use old name
3. **Base class change not propagated** — Subclass override now has wrong signature
4. **Missing `wait_for_everyone()`** — Distributed deadlock risk
5. **Reward shape mismatch** — Pointwise returning wrong batch dim
6. **License header missing** — New files without Apache 2.0 header
