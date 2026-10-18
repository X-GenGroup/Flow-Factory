---
name: ff-new-accelerator
description: Add or change a Flow-Factory acceleration plugin while preserving stage ownership, train-inference consistency, lifecycle cleanup, and backend compatibility.
---

# Acceleration Plugin Integration

## Context Routing

Read `../../../guidance/acceleration.md`, `../../knowledge/topics/train_inference_consistency.md`,
and `../../knowledge/topics/parity_testing.md`. Add component-runtime, dtype, checkpointing, or
backend topics only when touched. Run
`python3 ../../../scripts/agent_scope.py --base origin/main --intent new-accelerator`.

## Contract

Subclass `BaseAccelerator`, declare `stage` (`both` or `rollout`) and `safety` (`lossless` or
`lossy`), and register the class. A `both` accelerator mutates the shared prepared module through
`setup()`; a `rollout` accelerator uses a bounded `rollout_context()` and restores state on every
exit, including errors.

Lossy rollout acceleration is valid only for decoupled or distillation trainers. Lossy shared
acceleration must quantify cross-stage residuals and coupled-ratio behavior. Do not bypass prepared
component routes or let model-specific vocabulary enter the plugin layer.

## Profiles and Evidence

- **R2**: lossless shared plugin using an existing adapter API. Verify registry, slot validation,
  ordering, idempotent setup, and representative adapters.
- **R3**: lossy behavior, rollout context, compile/cache/checkpoint/offload, backend-specific state,
  or new adapter capability. Verify cleanup, unsupported-paradigm rejection, numerical parity or
  measured divergence, and affected backends.
- **R4**: change to the acceleration lifecycle, validator semantics, prepared routing, or shared
  execution path. Run the framework campaign when constraint #30 applies.

Document the measured safety claim and supported adapters/policies. Run `/ff-review` before commit.
