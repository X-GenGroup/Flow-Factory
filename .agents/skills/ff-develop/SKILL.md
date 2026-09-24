---
name: ff-develop
description: "Develop or refactor Flow-Factory features with cross-module impact analysis across execution contracts, typed data/model I/O, component loading, prepared bundles, rewards, variants, optimizers, distributed backends, and checkpoints."
---

# Feature Development Workflow

## Read by Change Area

Always read Tier 1. Add only the topic docs relevant to the change:

| Change area | Read |
|---|---|
| Adapter semantics or parity | `../../knowledge/topics/adapter_conventions.md`, `../../knowledge/topics/parity_testing.md` |
| Trainer forward/replay | `../../knowledge/topics/train_inference_consistency.md`, `../../knowledge/topics/autocast_param_swap.md` |
| Dtype or mixed precision | `../../knowledge/topics/dtype_precision.md` |
| Component discovery, lifecycle, load, prepare | `../../knowledge/topics/component_runtime.md` |
| Multi-component rollout/replay | `../../knowledge/topics/structured_trajectory.md` |
| Variants, roles, optimizers, role checkpoints | `../../knowledge/topics/component_variants.md` |
| Grouped sampler, reward/advantage layout, reward overlap | `../../knowledge/topics/samplers.md`, `../../../guidance/rewards.md` |
| Dataset acquisition/offline objective | `../../../guidance/workflow.md`, `../../../guidance/datasets.md` |
| Acceleration plugin | `../../../guidance/acceleration.md` |

## Plan Around Ownership

Before editing, state:

- the authoritative owner and public contract being changed;
- affected execution compositions (`generation + runtime_reward`, `generation + none`,
  `dataset + none`);
- registries, arguments, examples, docs, and checkpoint compatibility surfaces;
- affected component runtime types and distributed backends;
- unit, integration, and GPU evidence required.

Prefer one coherent invariant per commit. A contract change may need code, tests, docs, examples, and
knowledge updates atomically; do not split it merely to minimize file count.

## Impact Analysis

### 1. Execution Kernel and Trainers

- Derive current trainers from both trainer and TrainingArguments registries; do not maintain a
  hard-coded subclass list.
- Trainer and algorithm-specific arguments declare the same immutable `ExecutionContract`.
- `BaseTrainer.start()` owns acquisition dispatch, periodic boundaries, progress, and cycle hooks.
  Generation implements `optimize(samples)`; dataset acquisition implements
  `optimize_batch(batch)`.
- Keep `optimizer_step`, `rollout_iteration`, and `data_epoch` independent. Exact runtime identity
  must include any changed objective, data, backend, optimizer, or replayed evaluation semantics.
- New trainers inherit directly from `BaseTrainer`. Existing sanctioned strict extensions are
  GRPO-Guard/DPPO from GRPO and TDM-R1 from TDM; another trainer-to-trainer extension requires an
  explicit architectural justification.

### 2. Data Acquisition and Schemas

- Generation uses framework grouped loaders and adapter inference. Dataset acquisition uses an
  official finite `DistributedSampler`, calls `set_epoch(data_epoch)`, and does not prepare its
  training loader through Accelerator.
- Dataset source weights stay `1`; gradient accumulation is explicit and every rank-local epoch
  closes cleanly without an implicit partial-window flush.
- Strict V2 supervision records and collators remain model-neutral. Cache prompt/input conditions,
  never target/chosen/rejected pixels or latents.
- Evaluation remains generation-based, including for SFT and offline DPO.

### 3. Adapter and Model I/O Contracts

- Preserve the four abstract adapter methods and opt-in modality encoders. New sample classes use
  the task-level hierarchy, not another model-specific class.
- Assess the immutable `PipelineIOContract`, checkpoint-realized specialization, condition-state
  preparer, output-state codec, exact geometry validation, offline forward overrides, and objective
  reduction whenever offline capability changes.
- Public boundary-owning adapter wrappers validate shared semantics; extend their protected hooks
  instead of overriding them.
- Structured multimodal trajectories use adapter-owned component order and bridge APIs; trainer code
  never branches on legacy vs structured storage.
- Fields required by concrete sample reconstruction belong in inherited
  `reconstruction_required_fields`; do not conflate this with reward `required_fields` or collator
  `_shared_fields`.

### 4. Component Runtime and Loading

- Preserve canonical, override, declared, materialized, optional, and alias boundaries across
  `ClassicPipelineRuntime`, `ModularPipelineRuntime`, and `PseudoPipelineRuntime`.
- Resolve component membership through the runtime, not adapter attributes. Lazy materialization
  names required components explicitly.
- Public adapter lifecycle hooks remain the trainer-facing seam. `ModelLoadCoordinator` compiles
  logical declarations into exactly-once physical-root ownership; trainer/model code does not
  duplicate backend loading policy.
- Loading dtype is applied during native materialization; frozen/trainable dtype policy also reaches
  components materialized later.

### 5. Prepared Model Ownership

- All target components, frozen-but-shardable siblings, and live variant routes form one
  `ModelBundle`; prepare it with one optimizer root.
- After prepare, canonical adapter forwards route through `RoutedComponentProxy`. Preserve stable
  member names and `_no_split_modules`/`_repeated_blocks` metadata needed by FSDP policy discovery.
- Do not manually move, wrap, or offload a prepared route or individual trainable variant.
- Checkpoint save/load iterates trainable component ownership symmetrically; frozen bundle members
  are not given nonexistent per-component artifacts.

### 6. Variants, Roles, Optimizers, and Resume

- Spatial live trainable copies use `ComponentVariantRegistry`; temporal references, EMA, and old
  policies use named/ref snapshots.
- Algorithms own role names and update cadence. Variants are declared before prepare; role
  parameter and optimizer-group ownership is disjoint and exhaustive.
- The top-level `optimizers:` list has one entry per trainable role. Do not assume one group per
  role: Muon splits matrices from an optional AdamW remainder and still participates in one
  `CompositeOptimizer` root.
- Muon is supported only when `torch.optim.Muon` exists and the backend is DDP or FSDP2. Multi-role
  DeepSpeed requires ZeRO-1/2. Multi-role FSDP2 requires `use_orig_params=True`; the registry is
  rebound after prepare and optimizer references must point to the prepared root's DTensor-backed
  parameters.
- Resumable checkpoints include training-only roles, role counters, optimizer layout, and variant
  snapshots; validate metadata before prepared state mutation.

### 7. Rewards, Advantages, and Acceleration

- Pointwise/groupwise routing, per-dataset applicability, async execution, and train/eval model
  deduplication are shared reward contracts.
- Keep canonical `(source_id, unique_id)` group identity exact across reward, advantage, and
  objective paths. Sampler layout owns where a group closes; reward request batching and optimizer
  work-unit geometry remain independent.
- Reward/optimization overlap requires a trainer capability, compatible sampler placement, a
  group-local reducer, globally symmetric readiness selection, and preservation of original
  rollout microbatches for pack-composition-dependent adapters. Multi-source mixing may switch
  sources only at a group-complete sampler boundary.
- `feedback=none` bypasses training reward/advantage structurally; do not emulate it with incidental
  no-op overrides. Evaluation rewards remain independently configurable.
- Reward-based algorithms delegate advantage communication to `AdvantageProcessor`.
- Acceleration entries preserve ordered application and declared safety/stage. Lossy rollout-only
  acceleration is incompatible with coupled trainers.

### 8. Distributed Precision and Checkpointing

- Validate unsupported backend and optimizer plans before model weights load. ZeRO-3 remains
  unsupported.
- Activation checkpointing has one owner. FSDP2 normalizes full model checkpointing to backend
  ownership and rejects selective model policies; adapter-owned in-forward boundaries require an
  explicit capability.
- Wrap each forward in its own autocast region when optimizer steps or parameter swaps can occur.
- Preserve synchronization at preprocessing, evaluation, checkpoint, and publication boundaries.

## Implementation and Verification

1. Establish a passing baseline for the affected tests.
2. Change the authoritative contract and update all discovered callers/subclasses.
3. Add focused invariant tests before broad integration tests.
4. Verify only the affected matrix, choosing representatives from:
   - coupled generated feedback: GRPO, plus GRPO-Guard/DPPO if loss behavior changed;
   - decoupled generated feedback: online DPO or NFT/AWM/DGPO/CRD;
   - generated no-feedback: DiffusionOPD or DMD2/TDM;
   - dataset no-feedback: SFT and offline DPO;
   - multi-role: DMD2/TDM/TDM-R1;
   - legacy single-component and structured multimodal adapters.
5. Cover DDP, ZeRO-2, and FSDP2 only where the changed abstraction reaches those backends. Add
   Muon positive/negative coverage when optimizer selection is touched.
6. Run `/ff-review` before commit.

## Documentation and Examples

- API/workflow changes update the matching `guidance/` document.
- User-facing config fields update every affected example with defaults/options; example paths
  follow `examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml`.
- Architecture changes update the appropriate knowledge layer. New durable invariants go in
  `constraints.md`; detailed discoveries and fix history stay in topic docs.
- Bug fixes follow `../../knowledge/topics/fix_patterns.md`.

## Pre-Commit Gate

- All affected registries, arguments, callers, tests, examples, and docs agree.
- Changed Python files pass Black/isort; new source files carry the license header.
- Public methods are typed and use English Google-style docstrings.
- No silent fallback weakens a typed contract or ownership boundary.
