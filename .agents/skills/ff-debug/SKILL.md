---
name: ff-debug
description: Debug Flow-Factory crashes, hangs, OOMs, numerical failures, distributed errors, and wrong results using phase routing, historical fixes, and falsifiable experiments.
---

# Debug Workflow

Classify the failing lifecycle phase before loading broad context or changing code.

## Context Routing

1. Resolve the trainer, adapter, execution contract, backend, optimizer, dtype, and checkpoint
   mode from the failing configuration.
2. Query prior incidents before the first experiment:
   `python3 ../../../scripts/query_fix_patterns.py --query "<symptom or owner>" --paths <paths>`.
3. Use `../../knowledge/README.md` to load only the matching topic and constraint category.
4. Read [references/full-protocol.md](references/full-protocol.md) when the cause is uncertain,
   numerical, distributed, asynchronous, stateful, or survives one focused attempt.

## Failure Boundary

Locate the first causal failure in one phase: preflight, native load, preprocessing/cache, bundle
prepare, routed forward, acquisition, feedback, optimize/backward, save, resume, or cleanup. Read
all rank traces for distributed failures and distinguish the first cause from peer fallout.

## Quick Path

Use only for a deterministic local failure whose owning boundary is clear:

1. Reproduce with the smallest representative test or config.
2. State one falsifiable cause and the observation that would disprove it.
3. Add a regression for the same failure boundary.
4. Fix the authoritative owner and run the affected contract tests.
5. Classify the final diff with `agent_scope.py --intent debug` and run `/ff-review`.

## Escalation

- Numerical/parity failures require an oracle or real tensor endpoint comparison.
- Hangs require collective-order and synchronized-error evidence from every rank.
- OOMs require graph/buffer lifetime analysis before shrinking semantic workload.
- Overlap failures require stable row identity, tile readiness, and work-unit traces.
- Resume failures require uninterrupted versus save/resume equivalence.
- Media failures require real processor/decoder representation checks, not shape-only stubs.

After three rejected hypotheses, record the evidence under `.scratch/` and request review.

## Knowledge Capture

Every fix needs a regression and clear root cause in its commit. Promote it to
`../../knowledge/fixes/` only when it creates a reusable failure signature, ownership lesson, or
durable constraint; follow `../../knowledge/topics/fix_patterns.md`.
