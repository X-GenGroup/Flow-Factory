---
name: ff-debug
description: Bug fixing and debugging workflow with structured protocol
---

# Debug Workflow

## Two Pathways

### Quick Path (obvious root cause)

Use when: Error message clearly points to the issue (typo, missing import, wrong type).

1. Reproduce the error
2. Check `.agents/knowledge/constraints.md` for relevant constraints
3. Write targeted fix
4. Verify with test
5. Commit

### Full Protocol (complex issues)

Use when:
- Distributed training bugs (deadlocks, rank mismatches)
- Numerical issues (NaN, loss divergence, wrong gradients)
- Silent failures (training runs but produces garbage)
- Multiple failed fix attempts

## Full Protocol — Five Phases

### Phase 1: Root Cause Investigation

1. **Read complete error messages** — Full stack traces matter, don't skim
2. **Consult constraints** — Check `.agents/knowledge/constraints.md`
3. **Reproduce consistently** — Isolate the exact trigger condition
4. **Trace execution path** — Follow through the 6-stage pipeline

#### Distributed-Specific Checklist
- Does the error appear on all ranks or just one?
- Is `accelerator.wait_for_everyone()` missing before the failure point?
- Are frozen components synchronized across ranks? (Constraint #19)
- Is ZeRO-3 being used? (Constraint #10 — unsupported)

### Phase 2: Pattern Analysis

1. **Find working examples** — Compare with a similar model/algorithm that works
2. **Diff analysis** — What's different between working and broken paths?
3. **Isolate variables** — Change one thing at a time

### Phase 3: Hypothesis Testing

1. **One hypothesis per iteration** — Formulate a single falsifiable hypothesis
2. **Minimal test case** — Reproduce with smallest possible config
3. **Low confidence (<80%)?** — Add debug logging before applying fix

### Phase 4: Fix Implementation

1. **Write failing test first** (if possible)
2. **Implement targeted fix** — Only fix the bug, don't refactor
3. **Check cross-algorithm impact** — Does this fix break GRPO? NFT? AWM?
4. **Check cross-model impact** — Test with at least two model adapters

### Phase 5: Knowledge Capture

After fix is verified:
- Update `constraints.md` if a new constraint was discovered
- Add regression test if applicable
- Document the root cause in the commit message

## Three-Strike Rule

If the same approach fails three times:
1. **HALT** all fix attempts
2. Document what was tried and why it failed
3. Escalate for architectural review

## Common Issue Categories

### Training Loop Issues
- [ ] Stage ordering violated? (Constraint #6)
- [ ] Coupled/decoupled paradigm mismatch? (Constraint #7)
- [ ] Component not on correct device? (Constraint #8)
- [ ] Dataloader incorrectly prepared via accelerator? (Constraint #9)

### Model Adapter Issues
- [ ] `load_pipeline()` returning wrong type? (Constraint #5)
- [ ] `target_module_map` mapping incorrect components?
- [ ] `_shared_fields` causing data corruption? (Constraint #14)
- [ ] Preprocessing modules not offloaded after Stage 1?

### Reward Issues
- [ ] Pointwise/Groupwise confusion? (Constraint #13)
- [ ] Wrong reward shape returned?
- [ ] `required_fields` not set correctly?
- [ ] Device mismatch between reward model and generated samples?

### Configuration Issues
- [ ] YAML key doesn't match Pydantic field name? (Constraint #17)
- [ ] Algorithm-specific args using wrong subclass? (Constraint #16)
- [ ] Registry key doesn't match? (Constraint #1)

### Distributed Issues
- [ ] Missing synchronization barrier? (Constraint #18)
- [ ] FSDP frozen components uninitialized on Rank > 0? (Constraint #19)
- [ ] Mixed precision casting order incorrect? (Constraint #20)
- [ ] Using ZeRO-3? (Constraint #10 — not supported)
