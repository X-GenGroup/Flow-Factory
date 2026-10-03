---
name: ff-new-algorithm
description: Add or extend a Flow-Factory training algorithm and choose fast, focused, strict, or campaign validation from its execution and ownership semantics.
---

# Algorithm Integration

An algorithm name does not determine its validation cost. Classify the semantic delta before
implementation and again from the final diff.

## Context Routing

Read `../../../guidance/algorithms.md`, `../../../guidance/workflow.md`, and
`../../knowledge/topics/train_inference_consistency.md`. Read dataset, sampler, component-variant,
trajectory, or checkpoint topics only when those facets are present. Use
[references/integration.md](references/integration.md) for detailed hooks and ownership rules.

Run `python3 ../../../scripts/agent_scope.py --base origin/main --intent new-algorithm`.

## Algorithm Fingerprint

Declare acquisition, feedback, paradigm, dynamics, supervision schema, trajectory representation,
live-role topology, reference/snapshot state, group geometry, overlap capability, collectives,
backend assumptions, and exact-resume identity.

Use the existing immutable contracts when they fit:

- generation + runtime reward;
- generation + no feedback;
- dataset + no feedback.

A new composition is an R4 execution-driver design, not a trainer-local condition.

## Validation Strategy

**R1 objective-only** requires all of the following: existing execution contract and acquisition
driver; direct `BaseTrainer` implementation; existing sample/trajectory/feedback primitives; one
live role; no custom collective, overlap hook, runtime child, resume identity, backend branch, or
shared helper change. Verify registry parity, argument validation, a numerical oracle and limiting
cases, finite gradients, two lightweight acquisition cycles, and one representative adapter path.

**R2 focused** covers sanctioned strict inheritance, algorithm-local sampling scope, pairwise or
groupwise composition through existing helpers, and existing snapshot mechanisms. Add base/derived
behavior, working-analogue, and affected integration evidence.

**R3 strict** is required for `_run_training_step`, sampling provenance, new runtime children,
resume identity, several live roles, algorithm-owned collectives, new typed schemas, trajectory
semantics, backend policies, or algorithm-specific overlap. Verify failure atomicity, exact resume,
affected adapters, and affected distributed backends.

**R4 campaign** is required only when the implementation changes shared execution, acquisition,
feedback/advantage, sampler, overlap, loading, optimizer, or checkpoint infrastructure.

## Implementation Boundary

Implement the objective in the hook selected by acquisition mode. Do not override `start()`, infer
execution mode from batch fields, reimplement advantage communication, or push algorithm names
into model infrastructure. Keep trainer and argument registries and class-level contracts aligned.
Run `/ff-review` before commit.
