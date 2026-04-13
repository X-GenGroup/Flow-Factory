---
name: ff-debug
description: "Bug fixing and debugging for ANY error, crash, loss divergence, gradient explosion, distributed hang, NaN, or unexpected behavior. Covers quick fixes and full protocol with 5-phase investigation. Trigger: 'fix bug', 'fix error', 'broken', 'crash', 'doesn't work', 'fails with', 'loss NaN', 'training hangs', 'OOM'."
---

# Debug Workflow

## Related Topics (read for numerical / consistency issues)

- NaN, loss divergence, wrong gradients -> `topics/train_inference_consistency.md`
- Dtype mismatch, overflow, precision -> `topics/dtype_precision.md`

## Two Pathways

### Quick Path (obvious root cause)

Use when: Error message clearly points to the issue (typo, missing import, wrong type).

1. Reproduce the error
2. Check `.agents/knowledge/constraints.md` for relevant constraints
3. Write targeted fix
4. Verify with test
5. Run `/ff-review`, commit

If not resolved in 15 min -> switch to Full Protocol.

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
5. **Check recent changes** — `git log --oneline -10` — what changed recently?

#### Distributed-Specific Checklist
- Does the error appear on all ranks or just one?
- Is `accelerator.wait_for_everyone()` missing before the failure point?
- Are frozen components synchronized across ranks? (Constraint #19)
- Is ZeRO-3 being used? (Constraint #10 — unsupported)

### Phase 2: Pattern Analysis

1. **Find working examples** — Compare with a similar model/algorithm that works
2. **Diff analysis** — What's different between working and broken paths? Compare **completely** — diff line by line, not skim. Include config YAML and environment vars.
3. **Isolate variables** — Change one thing at a time
4. **Check dependencies** — Different diffusers version? Different PyTorch version?

### Phase 3: Hypothesis Testing

1. **One hypothesis per iteration** — Formulate a single falsifiable hypothesis
2. **Minimal test case** — Reproduce with smallest possible config
3. **Low confidence (<80%)?** — Add debug logging before applying fix

**Red flags — STOP and restart from Phase 1:**
- "Let me just try changing X and see what happens"
- "Quick fix for now, clean up later"
- "It probably works, let me move on"

**Verification gate** — before acting on a conclusion, check:
- Does the evidence actually support this cause, or just correlate?
- Could a different root cause produce the same symptoms?
- What observation would disprove this hypothesis? Have you looked for it?

### Phase 4: Fix Implementation

1. **Write failing test first** (if possible)
2. **Implement targeted fix** — Only fix the bug, don't refactor
3. **Check cross-algorithm impact** — Does this fix break GRPO? NFT? AWM?
4. **Check cross-model impact** — Test with at least two model adapters
5. Before committing: run `/ff-review` skill.

### Phase 5: Knowledge Capture

After fix is verified:
- Update `constraints.md` if a new constraint was discovered
- Add regression test if applicable
- Document the root cause in the commit message
- Follow fix archival process in `topics/fix_patterns.md`

## Three-Strike Rule

If the same approach fails three times:
1. **HALT** all fix attempts
2. Question whether the underlying approach/architecture is wrong
3. Step back and re-examine: are you solving the right problem?
4. Report to user with analysis before continuing

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
- [ ] Mixed precision casting order incorrect? (Constraint #20) — see also `topics/dtype_precision.md` for precision diagnosis checklist
- [ ] Using ZeRO-3? (Constraint #10 — not supported)
