---
name: ff-review
description: Review Flow-Factory diffs by independently classifying risk, checking ownership and compatibility, and verifying that evidence matches the final change.
---

# Code Review

Review the complete intended range, not only the latest working-tree diff.

## Context Routing

1. Capture staged, unstaged, untracked, base-to-HEAD, and rebase scope as applicable.
2. Run `python3 ../../../scripts/agent_scope.py --base <base> --intent review` independently of
   the author's declared profile.
3. Load only the topic leaves selected by the observed facets.
4. For R3/R4 boundary checks, read
   [references/contract-review.md](references/contract-review.md).

## Review Order

1. Compare declared and observed profiles. Any hidden state, schema, role, collective, resume,
   backend, or shared-lifecycle change raises the profile before evidence is assessed.
2. Verify the authoritative owner and all reached registries, typed contracts, callers,
   subclasses, examples, docs, cache identities, and checkpoint identities.
3. Check correctness first: ordering, shape, dtype, device, group identity, numerical invariants,
   fail-fast timing, cleanup, and failure atomicity.
4. Match tests and runtime artifacts to `../../harness/test_routes.yaml`. A passing broad suite does
   not replace a missing focused invariant; an R1 change does not need unrelated broad tests.
5. Re-run scope classification on the final diff. After a rebase or factual conflict, search clean
   downstream assertions that encode the same fact.

## Evidence by Profile

- **R1**: ensure no escalation trigger is hidden; require focused tests and config/registry parsing.
- **R2**: require the affected integration path and closest working analogue.
- **R3**: require state/resume and every affected distributed/backend composition.
- **R4**: require a complete result bundle for the exact final commit and manifest digest; skipped,
  stale, observe-only-for-required, or mislabeled backend cells block a safe verdict.

## Verdict

Report:

```text
Declared profile:
Observed profile:
Escalation reasons:
Changed owners:
Required evidence:
Available evidence:
Missing evidence:
Verdict: Safe | Needs attention | Risky
```

**Safe** may proceed only within the task mode in `AGENTS.md`. **Needs attention** lists concrete
file/line findings and requires re-review. **Risky** stops before commit or delivery.
