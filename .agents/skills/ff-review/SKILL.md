---
name: ff-review
description: "Mandatory pre-commit code review gate. Launches a review subagent that checks constraint violations, cross-module consistency, and implementation quality. Also trigger proactively when changes span multiple files, touch shared infrastructure (BaseTrainer, BaseAdapter, BaseRewardModel), or you are unsure if a fix is safe. Trigger: 'review', 'check before commit'."
---

# Code Review Workflow

## Process Overview

```
1. Capture changes → git diff
2. Load constraints → .agents/knowledge/constraints.md
3. Launch review subagent (isolated, no author reasoning)
4. Act on verdict:
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

- Read `.agents/knowledge/constraints.md` — All 25+ hard constraints
- Reference `.agents/knowledge/architecture.md` — Module dependencies
- Identify which modules are affected by the changes

## Step 3: Launch Review Subagent

**Launch a separate review subagent** using the Task tool. The subagent receives ONLY the diff and constraints — NOT your reasoning — to avoid confirmation bias.

Use the Task tool with this prompt:

```
You are a code reviewer for Flow-Factory, a unified online RL fine-tuning framework for diffusion/flow-matching models. Your job is to find problems in the following diff. You are NOT validating the author's intent — you are looking for bugs, risks, and constraint violations.

## Diff
<paste full git diff here>

## Known Constraints
<paste constraints.md content here>

## Review Checklist

For each changed file, check:

### Implementation Quality
- Hidden risks or edge cases not handled?
- Simpler alternative that achieves the same result?
- Boundary conditions (tensor shapes, distributed rank handling, gradient accumulation steps)?
- Does the fix depend on downstream code to "clean up"?

### Cross-Module Consistency
- If a BaseTrainer method changed, do all subclasses (GRPO, GRPOGuard, DPO, NFT, AWM) need matching changes?
- If a BaseAdapter method changed, are all model adapters updated?
- If a BaseRewardModel method changed, are all reward models updated?
- If hparams/ changed, are ALL example configs in examples/ updated?

### Constraint Violations
- Does this violate any entry in the known-constraints list?
- Does this repeat a pattern that previously caused bugs?

### Flow-Factory-Specific Checks
- PR title format: `[{modules}] {type}: {description}`?
- All comments, docstrings, and documentation in English? (Core Operating Principle #11)
- Black formatting (line-length=100)?
- isort compliance (profile="black")?
- Apache 2.0 license header on new files?

## Output

### Verdict: safe / needs-attention / risky

### Findings (for needs-attention or risky)
For each issue:
- **File**: path:line
- **Concern**: what could go wrong
- **Suggestion**: what to do instead
```

## Step 4: Route by Verdict

### Safe
No issues found. Proceed with commit.

### Needs-Attention
Issues found but fixable:
1. List each issue with file and line
2. Fix identified problems
3. Re-stage and re-review

### Risky
Potential breaking changes:
1. Halt commit
2. Report findings with severity
3. Await explicit user approval

## After Commit

- Run `black --check src/ && isort --check src/` to confirm formatting compliance.
- Verify PR title follows `[{modules}] {type}: {description}` format.

## Common Issues Found in Review

1. **Registry path stale** — Class moved but registry not updated
2. **Config field renamed** — YAML examples still use old name
3. **New config field not in examples** — Users won't discover it; add with default value and `# Options:` comment
4. **Base class change not propagated** — Subclass override now has wrong signature
5. **Missing `wait_for_everyone()`** — Distributed deadlock risk
6. **Reward shape mismatch** — Pointwise returning wrong batch dim
7. **License header missing** — New files without Apache 2.0 header
