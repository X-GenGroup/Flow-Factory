---
name: ff-review
description: "Review Flow-Factory changes before commit or merge for contract violations, cross-module drift, distributed/checkpoint safety, docs consistency, implementation quality, and test evidence."
---

# Code Review Workflow

## 1. Capture the Exact Scope

For a commit review, inspect staged and unstaged scopes separately:

```bash
git status --short
git diff --check
git diff --cached --stat
git diff --cached
git diff
```

For a PR-wide review, record the base and commit range and inspect that range in addition to local
changes. Do not assume `git diff HEAD` represents every intended change.

Read Tier 1 and derive affected trainers, adapters, rewards, accelerators, and argument classes from
their registries; avoid hard-coded component lists.

## 2. Load Scope-Specific References

| Diff touches | Also read |
|---|---|
| Execution contracts, trainer loop, offline data | `../../../guidance/workflow.md`, `../../../guidance/algorithms.md`, `../../../guidance/datasets.md` |
| Grouped sampler, advantage layout, reward overlap | `../../knowledge/topics/samplers.md`, `../../../guidance/rewards.md` |
| Adapter semantics or model parity | `../../knowledge/topics/adapter_conventions.md`, `../../knowledge/topics/parity_testing.md` |
| Component runtime, loading, bundle | `../../knowledge/topics/component_runtime.md` |
| Trajectory, sample, scheduler group | `../../knowledge/topics/structured_trajectory.md`, `../../knowledge/topics/train_inference_consistency.md` |
| Variant, role optimizer, Muon | `../../knowledge/topics/component_variants.md` |
| Dtype, autocast, parameter swaps | `../../knowledge/topics/dtype_precision.md`, `../../knowledge/topics/autocast_param_swap.md` |
| Gradient checkpoint/FSDP memory | `../../../guidance/new_model.md` checkpointing contract |
| Reward processing | `../../../guidance/rewards.md` |
| Acceleration | `../../../guidance/acceleration.md` |
| Framework-wide dataflow, execution, loading, distributed, or optimizer infrastructure | `../../../guidance/gpu_validation.md` |
| `.agents/` | `../../knowledge/docs_maintenance.md`, agent-doc maintenance rule |

## 3. Review Contract Boundaries

### Registries and configuration

- Static registry keys, lazy import paths, direct-path fallback, and argument registry agree.
- Keys follow canonical naming; moved/renamed classes update every registry/export.
- Trainer and algorithm-specific arguments declare the same immutable `ExecutionContract`; users
  cannot configure its axes.
- Added/renamed/removed fields update all affected examples and consumers. Example paths follow the
  project convention.

### Execution and data acquisition

- `BaseTrainer.start()` and acquisition drivers remain authoritative. The selected hook is
  `optimize(samples)` for generation or `optimize_batch(batch)` for dataset acquisition.
- `generation + runtime_reward`, `generation + none`, and `dataset + none` execute only their
  declared stages.
- Dataset training uses an official finite `DistributedSampler`, explicit accumulation, full clean
  traversal, unit source weights, and no Accelerator preparation of the training loader.
- Only input conditions enter preprocessing cache; output supervision is encoded on demand.
- `optimizer_step`, `rollout_iteration`, and `data_epoch` advance at their own boundaries.
- Exact runtime identity includes changed objective/data/backend/optimizer and replayed evaluation
  semantics.

### Adapter, pipeline I/O, and trajectories

- Base adapter keeps four abstract methods; adapters and model-specific samples retain flat/task-
  level inheritance contracts.
- Offline support declares a valid effective `PipelineIOContract`, declaration-only condition
  preparer/output codec, exact geometry validation, one prepared condition per request, complete
  offline forward overrides, and an explicit blocker when unsupported.
- Public boundary-owning wrappers are not overridden; protected hooks carry specialization.
- `forward()`/`inference()` preserve inputs, precision, scheduler state, and parity.
- Structured trajectories own component order, maps, callbacks, masks, noise, and reductions;
  trainers consume only bridge APIs. Legacy single-component behavior remains unchanged.
- Concrete sample fields required after partial gather are inherited through
  `reconstruction_required_fields`; reward `required_fields` and collator `_shared_fields` remain
  separate contracts.

### Component runtime, loading, and prepared ownership

- Canonical, override, declared, materialized, optional, and alias paths remain distinct.
- Membership uses runtime APIs, not adapter attributes; omitted lazy materialization does not load
  all specs.
- Logical-to-physical loading goes through `ModelLoadCoordinator`; auxiliary/reward roots remain
  replicas and target-only backend state does not leak.
- Every target/frozen-shardable/variant route enters one `ModelBundle` prepared with one optimizer;
  canonical forwards route through `RoutedComponentProxy` afterward.
- Stable names and `_no_split_modules`/`_repeated_blocks` metadata survive the bundle boundary.
- Save/load iterates trainable ownership symmetrically and skips frozen-only checkpoint artifacts.

### Variants, optimizers, distributed plans, and checkpoints

- Algorithm vocabulary and role cadence stay in trainers. Temporal ref/EMA/old state is not modeled
  as a live trainable variant.
- Variants are declared before prepare; parameter and optimizer ownership is disjoint/exhaustive.
- No code assumes one group per role. Muon matrices and AdamW fallback groups share one
  `CompositeOptimizer` root.
- ZeRO-3 is rejected. Muon availability and DeepSpeed/FSDP1 incompatibility fail before model load;
  multi-role DeepSpeed is ZeRO-1/2. Multi-role FSDP2 requires `use_orig_params=True`; registry and
  optimizer references must point to the replacement DTensor-backed parameters owned by the
  prepared model root.
- Activation checkpointing has one owner. FSDP2 full policy moves to backend ownership, selective
  model policy is rejected, and adapter-owned in-forward boundaries require explicit capability.
- Resumable checkpoints include all training roles/runtime children and validate metadata before
  Accelerate state mutation. Model-only export scope remains intentional.
- Distributed checkpoint phases and publication are all-rank symmetric and synchronized.

### Rewards, acceleration, and numerical quality

- Pointwise calls accept tail/source-gated chunks and return one finite value per actual input;
  groupwise paths preserve complete group order.
- Canonical group identity is sampler-assigned exact `(source_id, unique_id)` int64 data across
  reward, advantage, pairing, noise, and objective paths; planned training identity is never
  recomputed from content or packed into a floating-point reward payload.
- Sampler placement, objective capability, and reducer scope agree. Reward request batches stay
  independent of optimizer work units, every readiness collective is rank-symmetric, and
  pack-composition-dependent replay preserves original rollout microbatches. Multi-source overlap
  mixes sources only between group-complete sampler windows.
- Per-dataset applicability/weights, async tail flush, and train/eval model deduplication remain
  correct.
- Reward-free contracts do not create incidental training reward work.
- Acceleration entries preserve declared safety/stage and list order; lossy rollout-only plugins do
  not run on coupled trainers.
- No hardcoded device bypasses adapter/reward/backend ownership.
- Each forward has an appropriate autocast boundary when optimizer steps or param swaps occur.
- Required rank barriers are present without introducing asymmetric collectives or filesystem work.

### Code and documentation quality

- Public functions/methods are typed and have English Google-style docstrings; new source files
  carry the Apache header.
- Imports follow project style and sanctioned local-import exceptions only.
- Errors fail fast with concrete user-facing config values; no silent fallback weakens a contract.
- README, guidance, examples, code comments/docstrings, and `.agents/` docs describe current owners,
  supported modes, and paths without claiming unexecuted quality.
- Published text/log snippets contain no credentials, tokens, personal absolute paths, hostnames, or
  machine-specific details.

## 4. Verify Proportionally

Run focused tests tied to each changed contract before broad tests. Useful suites include:

- execution: `tests/contracts/test_execution_contract.py`,
  `tests/trainers/test_execution_kernel.py`;
- offline: `tests/hparams/test_offline_training_args.py`, offline data/trainer tests;
- runtime/I/O: component runtime, pipeline contract, output-state lifecycle tests;
- trajectory: `tests/models/trajectory/` and bridge/reduction tests;
- variants/optimizer: component variant, role optimization, Muon, and multirole tests;
- distributed/checkpoint: distributed-plan, checkpoint layout/runtime identity/resume tests;
- rewards: loader-context and processor/reconstruction tests.

Run Black/isort on changed Python files as the commit gate, then run the documented full-tree checks.
If the repository has pre-existing full-tree failures, prove the baseline and distinguish them from
new regressions; do not hide them or expand scope silently. Validate Markdown links and example
paths for docs changes.

GPU/distributed evidence should cover only affected compositions/backends, but any claimed support
must have a representative run. Multi-role/Muon or loading/checkpoint changes normally require DDP,
ZeRO-2, and FSDP2 coverage plus intended early-rejection cases.

If the PR reaches any trigger in `constraints.md` #30, representative coverage is insufficient:
verify that the exact head commit has a complete result bundle for every job in
`config/gpu_validation/framework_upgrade.yaml`, and run `scripts/validate_gpu_campaign.py` against
that bundle. A skipped or infrastructure-blocked cell, stale commit SHA, stale manifest digest, or
backend name without matching runtime plugin observation blocks a safe verdict.

## 5. Verdict

- **Safe**: contracts, tests, docs, and evidence agree; proceed only with the user's authorized
  commit/push scope.
- **Needs attention**: list each issue with file/line and fix/re-review before commit.
- **Risky**: halt when behavior, compatibility, data, or distributed correctness remains uncertain
  and request explicit direction.

After an authorized commit, verify the final diff and formatting. Bug fixes also follow
`../../knowledge/topics/fix_patterns.md`.

## Frequent Review Findings

- Trainer/argument contract drift or wrong acquisition hook.
- Offline path invoking rollout/reward or caching supervision state.
- Adapter capability checked only after weights load.
- Runtime membership via `hasattr`, alias double movement, or component prepared outside the bundle.
- Missing frozen-member checkpoint symmetry or lost repeated-block wrap metadata.
- Frozen reference represented as a variant or one-group-per-role assumption.
- Training role/runtime child omitted from resume metadata.
- Muon accepted on an unsupported backend or duplicate activation-checkpoint owners.
- Public docs naming a pre-refactor function, owner, path, or unverified support status.
