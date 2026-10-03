---
name: ff-new-model
description: Add a Flow-Factory model adapter with validation depth selected from runtime, modality, trajectory, offline, loading, and backend capabilities.
---

# Model Adapter Integration

Start from the closest adapter by contract, not model-family name alone.

## Context Routing

Read `../../../guidance/new_model.md`, `../../knowledge/topics/adapter_conventions.md`, and
`../../knowledge/topics/parity_testing.md`. Add component-runtime, structured-trajectory, dataset,
or distributed detail only when the adapter claims those capabilities. Use
[references/integration.md](references/integration.md) for the complete contract checklist.

Run `python3 ../../../scripts/agent_scope.py --base origin/main --intent new-model`.

## Profile Selection

- **R1 fast**: family-local classic eager pipeline, existing modality and sample type, one latent
  component, existing dtype policy, online-only, no new runtime or backend capability.
- **R2 focused**: new preprocessing, checkpoint specialization, or an existing output-codec
  pattern without shared contract changes.
- **R3 strict**: modular/pseudo runtime, structured multimodal trajectory, ordered references,
  pack-sensitive batches, offline geometry specialization, prefix/cache behavior, FSDP capability,
  or adapter-owned checkpoint/offload policy.
- **R4 campaign**: change to `BaseAdapter`, component runtime, loading coordinator, shared media/I/O
  contract, trajectory bridge, or distributed preparation infrastructure.

## Required Boundaries

Implement the four abstract methods, preserve train-inference parity, declare component ownership,
register the adapter, and add only the examples and docs for modes actually supported. Offline
support requires a lossless `PipelineIOContract`, declaration-only output codec and condition
preparer, geometry validation, and complete forward overrides. Unsupported offline modes must fail
before weights load.

R1 uses focused import/registry, fake-pipeline, and adapter-specific tests. R3 adds real parity,
legacy/structured claims, model-only checkpoint, exact resume when state changes, and affected
backends. Never claim GPU or quality evidence that was not executed. Run `/ff-review` before commit.
