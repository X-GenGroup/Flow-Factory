---
name: ff-new-reward
description: Add a Flow-Factory reward with validation depth selected from pointwise, groupwise, remote, reconstruction, distributed, and overlap semantics.
---

# Reward Integration

Keep model scoring separate from applicability, reconstruction, aggregation, and optimizer work
units owned by the framework.

## Context Routing

Read `../../../guidance/rewards.md`, `src/flow_factory/rewards/abc.py`, and the current loader and
registry. Load `../../knowledge/topics/samplers.md` only for groupwise, distributed, or overlap
behavior. Use [references/integration.md](references/integration.md) for media conversion,
dispatch, and configuration details.

Run `python3 ../../../scripts/agent_scope.py --base origin/main --intent new-reward`.

## Profile Selection

- **R1 fast**: stateless local pointwise scoring with existing public sample fields and media
  representations.
- **R2 focused**: groupwise scoring, new required-field composition, or a remote pointwise client
  using the existing executor contract.
- **R3 strict**: new asynchronous lifecycle, retry/timeout semantics, partial-gather reconstruction,
  cross-rank grouping, or persistent client state.
- **R4 campaign**: changes to `BaseRewardModel`, `RewardProcessor`, canonical group identity,
  advantage communication, tile planning, or reward/optimization overlap.

## Required Invariants

Return exactly one finite score per provided input in the same order. Pointwise calls accept full
and tail chunks; groupwise calls receive one complete canonical `(source_id, unique_id)` group.
Declare only consumed `required_fields`; do not replace sample reconstruction or collator contracts.
Keep per-dataset applicability outside model scoring and fail before heavyweight loading when a
required capability is absent.

R1 verifies direct full/tail calls, output order/shape/finiteness, applicability, registry, and
loader context. Higher profiles add reconstruction, executor cleanup, local/global grouping,
failure propagation, and affected overlap trainers. Run `/ff-review` before commit.
