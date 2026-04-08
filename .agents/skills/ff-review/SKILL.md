---
name: ff-review
description: "Mandatory pre-commit code review gate. Checks constraint violations, cross-module consistency, and implementation quality. Trigger proactively when changes span multiple files or touch shared infrastructure. Trigger: 'review', 'check before commit'."
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

- Read `.agents/knowledge/constraints.md` — All hard constraints
- Reference `.agents/knowledge/architecture.md` — Module dependencies
- Identify which modules are affected by the changes

## Step 3: Review Checklist

### Constraint Compliance
- [ ] No constraint violations found
- [ ] Registry entries updated if classes moved/renamed (#1–4)
- [ ] Pipeline order preserved (#6)
- [ ] Coupled/decoupled paradigm respected (#7)
- [ ] Base class interfaces not broken (#11–13)
- [ ] Config fields synchronized with YAML examples (#15–17)

### Cross-Module Consistency
- [ ] Changes to `abc.py` base classes reflected in ALL subclasses (GRPO, GRPOGuard, DPO, NFT, AWM)
- [ ] Changes to `hparams/` reflected in ALL example configs
- [ ] Changes to `AdvantageProcessor` compatible with all trainers
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
- [ ] New config fields added to ALL example configs with defaults and `# Options:` comments
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

## After Commit

- Run `black --check src/ && isort --check src/` to confirm formatting compliance.
- Verify PR title follows `[{modules}] {type}: {description}` format.
- If this was a bug fix, follow `topics/fix_patterns.md` archival process.

## Pre-Review Reading

Before reviewing, always read Tier 1: `constraints.md`, `architecture.md`, `philosophy.md`.

Additionally, read based on diff scope:

| Diff touches... | Also read |
|----------------|-----------|
| `models/` | `topics/adapter_conventions.md`, `topics/parity_testing.md` |
| `trainers/` | `topics/train_inference_consistency.md` |
| `scheduler/` | `topics/train_inference_consistency.md`, `topics/dtype_precision.md` |
| New adapter | `topics/adapter_conventions.md`, `topics/parity_testing.md` |
| dtype/precision | `topics/dtype_precision.md` |

## Common Issues Found in Review

1. **Registry path stale** — Class moved but registry not updated
2. **Config field renamed** — YAML examples still use old name
3. **New config field not in examples** — Users won't discover it; add with default value and `# Options:` comment
4. **Base class change not propagated** — Subclass override now has wrong signature
5. **Missing `wait_for_everyone()`** — Distributed deadlock risk
6. **Reward shape mismatch** — Pointwise returning wrong batch dim
7. **License header missing** — New files without Apache 2.0 header
