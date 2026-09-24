---
name: ff-debug
description: "Debug Flow-Factory crashes, hangs, OOMs, numerical failures, finite-loader errors, component routing, distributed preparation, optimizer roles, and checkpoint/resume mismatches. Use for bug fixing or unexpected training behavior."
---

# Debug Workflow

## Load the Relevant Contracts

Always read Tier 1: `../../knowledge/constraints.md`, `../../knowledge/architecture.md`, and
`../../knowledge/philosophy.md`. Then route by symptom:

| Symptom or area | Also read |
|---|---|
| Wrong gradients, ratio drift, bad output | `../../knowledge/topics/train_inference_consistency.md`, `../../knowledge/topics/parity_testing.md` |
| Dtype mismatch, overflow, flat loss or KL | `../../knowledge/topics/dtype_precision.md`, `../../knowledge/topics/autocast_param_swap.md` |
| Missing component, wrong device, lazy load, wrap/OOM | `../../knowledge/topics/component_runtime.md` |
| Multi-component rollout or replay | `../../knowledge/topics/structured_trajectory.md` |
| Variant, role cadence, optimizer group, Muon, role checkpoint | `../../knowledge/topics/component_variants.md` |
| Reward hang, wrong group, overlap ordering or latency | `../../knowledge/topics/samplers.md`, `../../../guidance/rewards.md` |
| Finite dataset, target encoding, SFT/offline DPO | `../../../guidance/workflow.md`, `../../../guidance/datasets.md` |

## Classify the Execution Path First

Resolve the trainer and its algorithm-specific `TrainingArguments`; their immutable
`ExecutionContract` values must be equal.

| Composition | Expected path |
|---|---|
| `generation + runtime_reward` | `sample()` -> feedback/reward -> advantage -> `optimize()` |
| `generation + none` | `sample()` -> `optimize()`; no training reward/advantage stage |
| `dataset + none` | Exhaust one official finite `DistributedSampler` loader through `optimize_batch()` |

Do not infer execution mode from batch fields. Track `optimizer_step` independently from
`rollout_iteration` or `data_epoch`; a dataset epoch advances only after clean loader exhaustion.

## Quick Path

Use when the failure is deterministic and the stack trace identifies one local contract breach.

1. Reproduce it with the smallest representative test or config.
2. Trace the owning boundary and relevant constraint.
3. Add a regression that fails for the same reason.
4. Apply the narrow fix and run affected contract tests.
5. Run `/ff-review` before committing.

If the cause is uncertain, distributed, numerical, or survives one focused attempt, use the full
protocol.

## Full Protocol

### 1. Establish the Failure Boundary

- Read every rank's complete traceback and first causal error.
- Record the resolved trainer, adapter, execution contract, model I/O contract, backend, optimizer
  types, finetune type, dtype policy, and checkpoint mode.
- Compare against a working path one variable at a time, including YAML and backend config.
- Identify when failure occurs: preflight, native component load, preprocessing, bundle prepare,
  proxy-routed forward, acquisition, optimizer step, save, or resume.
- Check recent changes with a focused file/commit diff; do not assume temporal correlation is cause.

### 2. Check Ownership Invariants

#### Dataset acquisition

- Uses PyTorch's official `DistributedSampler`, even at one rank, and calls
  `set_epoch(data_epoch)`.
- Uses explicit positive `gradient_accumulation_steps`; rank-local batch count closes every
  accumulation window without an implicit flush.
- Caches prompt/input conditions only. Target, chosen, and rejected media is decoded and encoded on
  demand.
- Calls `prepare_condition_state()` once per batch. Offline DPO reuses that object, schedule, noise,
  and one reference scope across both arms.
- Applies the adapter's complete `offline_training_forward_overrides`, not rollout CFG semantics.

#### Component runtime and loading

- Resolve membership with `has_component`, `get_component`, or `_require_component`, never
  `hasattr(adapter, name)`.
- Keep canonical components, prepared/replacement overrides, declared specs, materialized modules,
  optional `None`, and pseudo aliases distinct.
- `materialize_components(None)` means already-materialized modules, not all lazy declarations.
- Trace logical names to physical roots through `ModelLoadCoordinator`; adapters/trainers must not
  reproduce FSDP loading state or broadcast weights directly.
- All target and frozen-but-shardable routes enter one `ModelBundle` with one optimizer prepare
  root. Adapter forwards after prepare route through `RoutedComponentProxy`.
- For FSDP OOM, inspect bundle-exposed `_no_split_modules`/`_repeated_blocks`, wrap classes, adapter
  memory capabilities, checkpoint replay, unshard stream, and backward prefetch before shrinking
  the workload.

#### Variants, optimizers, and distributed plans

- A temporal reference/EMA/snapshot is not a live component variant.
- Variants are declared before prepare; role parameter and optimizer-group ownership is disjoint and
  exhaustive.
- Do not assume one group per role: Muon uses matrices plus an optional AdamW remainder inside one
  `CompositeOptimizer` root.
- Reject ZeRO-3. Reject Muon before model loading when its PyTorch API is absent or the backend is
  DeepSpeed/FSDP1. Multi-role DeepSpeed requires ZeRO-1/2. Multi-role FSDP2 requires
  `use_orig_params=True`; after prepare, registry and optimizer references must point to the
  DTensor-backed parameters owned by the prepared model root.
- Activation checkpointing has one owner. FSDP2 full policy is normalized to backend ownership;
  selective model checkpointing is rejected. Inspect adapter-owned in-forward checkpointing only
  when the adapter explicitly opts in.

#### Checkpoint and exact resume

- Model-only export scope and resumable state scope are different. Resumable multi-role saves include
  training-only roles, role counters, optimizer ownership, and variant snapshots.
- Validate variant/runtime metadata before Accelerate mutates prepared state.
- Exact identity covers changed objective, model/backend/optimizer semantics, realized data order,
  and replayed evaluation configuration; cadence and resume location remain operational controls.
- Preserve all-rank phase symmetry and atomic publication ordering.

#### Numerical and reward paths

- Wrap each policy/reference/EMA forward in its own autocast region when weights can change in place.
- Consume trajectories only through adapter bridge methods and authoritative component order.
- A partial cross-rank gather must union the concrete sample class's
  `reconstruction_required_fields` before reconstruction. This is independent of reward
  `required_fields` and collation `_shared_fields`.
- Pointwise rewards return one finite value per actual input chunk, which may be smaller than
  `batch_size`; requests may cross optimizer work-unit boundaries and must route results by stable
  acquisition row. Groupwise rewards preserve complete canonical `(source_id, unique_id)` order.
- For overlap hangs, verify every rank constructs the same work units and enters readiness
  collectives in the same order. Distinguish remote-reward wait from readiness coordination using
  the `timing/reward_overlap/` metrics before changing poll cadence.
- Per-dataset reward applicability is framework-owned; model NaN/Inf is an error, not a routing
  sentinel.

### 3. Test One Falsifiable Hypothesis

State the proposed cause, the observation that would disprove it, and the smallest experiment that
separates it from alternatives. Add instrumentation when confidence is below 80%. Avoid speculative
fallbacks or several behavioral changes in one experiment.

### 4. Fix and Verify in Scope

- Write the regression first when practical.
- Fix the authoritative owner rather than patching downstream consumers.
- Verify the narrow unit contract, then affected compositions:
  - execution kernel: GRPO, a reward-free generation trainer, and SFT/offline DPO as applicable;
  - adapter/trajectory: legacy single-component and structured multimodal paths;
  - runtime/loading: the affected classic/modular/pseudo runtime and supported backends;
  - variants/optimizer: single-role AdamW plus multi-role/Muon cases when touched;
  - checkpoint: model-only round trip and exact-state resume when touched.
- Test at least two adapters only when the changed abstraction is shared across adapters.

### 5. Capture the Fix

Follow `../../knowledge/topics/fix_patterns.md`: record symptom, root cause, fix, lesson, related
constraint, test evidence, and commit. Update Tier 1 only when the fix establishes a durable
cross-module invariant.

## Three-Strike Rule

After three failed approaches to the same cause, stop patching, document evidence and rejected
hypotheses, reassess the ownership model, and request review before continuing.
