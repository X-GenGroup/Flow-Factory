# Hard Constraints

Quick index: **#1-5** Registry | **#6-10** Training Pipeline | **#11-14** Base Classes | **#15-17** Config | **#18-20** Distributed | **#21-27** Code Quality | **#28-30** Agent Workflow

These constraints MUST NOT be violated. Consult this file before making any code changes.

---

## Registry & Loading (1–5)

### 1. Registry Path Accuracy
The four registries (`_TRAINER_REGISTRY`, `_MODEL_ADAPTER_REGISTRY`, `_REWARD_MODEL_REGISTRY`, `_ACCELERATOR_REGISTRY`) map string identifiers to **fully qualified Python class paths** for lazy import. If you move, rename, or restructure a class, the corresponding registry entry MUST be updated, or `ImportError` will occur at runtime.

### 2. Registry Identifier Convention
Registry keys are **case-insensitive** (lowered at lookup). Preserve the canonical registered
spelling: most model adapter keys use lowercase with hyphens (for example `flux1-kontext`), while
the public Wan/LTX2 keys retain underscores (for example `wan2_t2v` and `ltx2_t2av`). Trainer and
reward keys use lowercase (for example `grpo-guard` and `pickscore`). New entries must choose one
canonical lowercase spelling and use it consistently across registries, arguments, and examples.

### 3. Dynamic Import Fallback
All four registries support a **direct Python path** fallback (e.g., `my_package.models.CustomAdapter`). If an identifier is not found in the registry, it is treated as a fully qualified import path. Do not break this two-mode resolution logic.

### 4. Decorator Registration
`@register_trainer` and `@register_reward_model` decorators exist for convenience but the canonical entries are the static dicts. If you use the decorator, ensure the static dict is also updated if the class should be discoverable by default.

### 5. Adapter Component Runtime Contract
Existing `BaseAdapter` subclasses keep implementing `load_pipeline()` and use the default
`ClassicPipelineRuntime`. Adapters backed by lazy modular pipelines or explicit pseudo-pipeline
containers override the concrete `build_component_runtime()` hook instead; they must retain
`adapter.pipeline` as the backend compatibility alias and ensure the canonical scheduler is
materialized before scheduler construction. Trainer stage lifecycle must call the adapter's public
component methods, preserving model-specific override points. Runtime-wide device and dtype
enumeration includes only materialized canonical `torch.nn.Module` entries; declared lazy specs,
optional `None` entries, and pseudo-pipeline aliases are not implicitly loaded or moved.
`materialize_components(None)` means already-materialized modules, never all declared specs.
Text-encoder/transformer role groups include non-`None` declarations/specs only.
The canonical scheduler remains `adapter.scheduler`; `adapter.scheduler_group` owns ordered mode
and seed dispatch, and its immutable names must equal `trajectory_component_order`.

---

## Training Pipeline (6–10)

### 6. Execution Contract and Stage Order
Every trainer and its training arguments declare the same immutable acquisition/feedback
`ExecutionContract`. Generation + runtime reward preserves Data Preprocessing → K-Repeat →
Trajectory Generation → Reward → Advantage → Policy Optimization. Dataset + no feedback exhausts
the finite loader through `optimize_batch()` and must not call rollout/reward stages. Do not infer
either axis from batch fields or make it user-configurable independently of `trainer_type`.

### 7. Coupled vs Decoupled Paradigm
- **Coupled** (GRPO, GRPO-Guard, DPPO): Training timesteps are coupled with SDE-based sampling. Requires log-probability computation. Must use SDE dynamics (`Flow-SDE`, `Dance-SDE`, `CPS`).
- **Decoupled** (SFT, offline DPO, online DPO, NFT, AWM, DGPO, CRD, TDM-R1): Training timesteps are decoupled from sampling. They may use ODE subject to algorithm-specific rules; TDM-R1 requires ODE.
- **Distillation** (`diffusion-opd`, DMD2, TDM): Generated acquisition with no runtime reward/advantage stage. DiffusionOPD supports ODE or SDE; DMD2 and TDM require ODE.

Mixing paradigms (e.g., using `ODE` dynamics with `GRPO`) will produce incorrect gradients silently.

### 8. Component Offloading Lifecycle
Text and condition encoders are loaded for preprocessing, then may be offloaded before the
training loop. Online inference reloads its declared components. Dataset acquisition separately
loads adapter-declared output codec components for on-the-fly target/chosen/rejected encoding;
these output latent states must never enter the input-condition cache.

### 9. Accelerator `prepare()` Scope
All target components (trainable **and** frozen-but-shardable) are bundled into a single `ModelBundle` (`models/model_bundle.py`) and prepared with the **optimizer** as one root via `accelerator.prepare()` — DeepSpeed (one engine) and FSDP2 (one root) cannot prepare multiple models separately. After prepare, each component is exposed as a `RoutedComponentProxy` that routes forwards through the bundle root; the optimizer/EMA/reference params still target only the `requires_grad` subset (frozen members are sharded for memory but never trained). Generation dataloaders use the framework's grouped samplers; dataset acquisition uses PyTorch's official `DistributedSampler`, calls `set_epoch(data_epoch)`, and requires one complete traversal per offline epoch. Neither train dataloader path is prepared via Accelerator. Breaking this causes duplicate data or incorrect gradient accumulation.

### 9a. Sampler Geometric Constraints
Generation sampler geometry is owned by `SamplerLayoutContract` and realized by one immutable
`SamplingPlan`; alignment must not branch on sampler class names. Let U=unique groups, K=group
size, W=world size, and B=per-device batch size. Every layout closes complete rank batches.
`rank_local` additionally requires U to be a multiple of `W*B/gcd(B,K)`; `global_batch` requires
`K <= W*B` and `(W*B) % K == 0`; `global_tile` closes in `K/gcd(W*B,K)` batches; and
`subgroup_tile` with R ranks per subgroup requires `W % R == 0`, closes in
`K/gcd(R*B,K)` batches, and aligns U to `(W/R)*(R*B/gcd(R*B,K))`. `global_random` and
`contiguous_shard` have no bounded group-complete window. The optimizer-step alignment is combined
with the selected layout step by LCM. See `topics/samplers.md` for the full formulas and aliases.

Generated group identity is assigned by the plan before placement, never inferred from prompt or
condition content. Reserved DataLoader columns carry exact group/member/sample IDs; trainers strip
them before adapter inference and attach them one-to-one to returned samples. The legacy content
hash is only a fallback for samples created outside a planned training loader.

Dataset acquisition does not use grouped geometry: every source weight is `1`,
`gradient_accumulation_steps` is an explicit positive integer, and each rank's finite batch count
must be divisible by it. Do not add batches merely to close or implicitly flush a partial
accumulation window. PyTorch's official `DistributedSampler` remains authoritative for cross-rank
tail handling: with `drop_last=False` it may repeat tail indices to equalize rank lengths, and one
offline epoch means one complete traversal of that resulting finite loader.

### 9b. Checkpoint Save/Load Symmetry Under the Bundle
Checkpoints are written and read for **trainable members only** — components whose `target_module_map[name]` is non-empty (`adapter.trainable_component_names`). Frozen-but-shardable bundle members (e.g. Wan2.2's `transformer_2`, kept in `target_components` only to be FSDP-sharded for memory; see #9) map to `None` and are skipped by both `save_checkpoint` and `_load_lora`/`_load_full_model`. Loaders MUST iterate `trainable_component_names`, not `target_components`, or resume logs a spurious error for a per-component subdir that was never written. `resume_type='state'` restores via `accelerator.load_state` into the prepared bundle root and is therefore keyed to bundle membership — resuming into a different `target_components` / bundle composition will mismatch.

### 9c. Reward/Optimization Overlap
Reward/optimization overlap is a trainer capability below `ExecutionContract`, never another
acquisition or feedback mode. Every rank selects the same globally ready tile before entering
optimizer collectives. GRPO, GRPO-Guard, DPPO, NFT, AWM, CRD, online DPO, DGPO, and TDM-R1
declare the capability. Per-sample group-relative objectives accept rank-local, global-batch,
global-tile, or subgroup-tile work units; online DPO overlap requires rank-local groups; DGPO requires complete
groups in each global microbatch; TDM-R1 accepts rank-local or global-batch groups. Cross-rank
overlap supports async pointwise rewards only. All modes require async-only CPU reward clients,
built-in `sum` or `gdpo` group-local advantages (`global_std=false`), one unshuffled inner epoch,
and no acquisition-wide advantage standardization. Reward-model request batches follow each
model's own `batch_size` and may cross optimizer work-unit boundaries; stable acquisition row
identity, not shared batching, connects the two stages. Pack-composition-dependent adapters must
validate work-unit boundaries against the original rollout microbatches recorded in the
`AcquisitionManifest`. In multi-source overlap, the source scheduler may switch datasets only at
the selected sampler layout's smallest group-complete window.
For `subgroup_tile`, group identity gathers and advantage-statistic reductions use its contiguous
R-rank process group, while readiness, logging statistics, DDP gradients, and optimizer steps
remain globally symmetric.
Unsupported algorithms and geometries fail before model loading; they never silently fall back or
change their objective.

**Common violation:** Do not infer normalization scope from `advantage_aggregation`. GDPO with
`global_std=false` closes within each complete group; only the explicit global-standardization
setting creates an acquisition-wide dependency.

### 10. DeepSpeed ZeRO-3 Is Unsupported
Supported distributed plans are DDP, FSDP, and DeepSpeed ZeRO-1/2. Reward model sharding under ZeRO-3 is broken even with DeepSpeed's own `zero.GatheredParameters` context manager, and parameter sharding also breaks frozen-component synchronization. `validate_supported_distributed_plan` is defined in `trainers/multirole/backend.py`; `trainers/loader.py` calls it before model construction, while `BaseTrainer.__init__` repeats the check defensively. `config/deepspeed/` ships no ZeRO-3 profile. Multi-role training narrows this further: `_validate_multirole_backend` requires ZeRO-1/2 and, under FSDP2, `use_orig_params=True`. Muon narrows the plan independently to DDP/FSDP2 and requires a build exposing `torch.optim.Muon`; `validate_optimizer_backend_plan` rejects DeepSpeed, FSDP1, and an unavailable Muon API before weights load.

---

## Base Class Interfaces (11–14)

### 11. BaseTrainer Execution Contract
`BaseTrainer.__init__` expects `(accelerator, config, adapter)`. Generation trainers override
`optimize(samples)`; dataset trainers override `optimize_batch(batch)`, and construction validates
the hook selected by the execution contract. `start()`, acquisition drivers,
`prepare_feedback()`, `compute_advantages()` and `evaluate()` are concrete base behavior — prefer
the hooks over restating the loop. `_initialization()` owns dataloaders, optimizer, distributed
preparation, rewards, and advantage processing.

**Acquisition hook order**: generation calls `sample()` → optional `prepare_feedback()` →
`optimize()`; dataset acquisition iterates the official finite loader and calls optional feedback →
`optimize_batch()`. Online `DPOTrainer` forms pairs at `optimize()` entry. Offline DPO consumes
dataset pairs directly.

**Trainer hierarchy**: New trainers MUST inherit directly from `BaseTrainer`. The sanctioned
existing behavioral extensions are `GRPOGuardTrainer → GRPOTrainer`, `DPPOTrainer → GRPOTrainer`,
and `TDMR1Trainer → TDMTrainer`; TDM-R1 reuses TDM's deterministic trajectory and multi-role
runtime while restoring runtime reward feedback. Trainer-to-trainer inheritance creates fragile
coupling; when in doubt, inherit from `BaseTrainer` and extract shared logic into helper methods.
Every runtime-reward trainer delegates advantage computation to `AdvantageProcessor`. Trainers
with feedback `none` (`diffusion-opd`, DMD2, and TDM) bypass reward/advantage structurally; their
no-op `prepare_feedback()` overrides are compatibility shims, not the execution mechanism.

### 12. BaseAdapter Abstract Methods
Subclasses of `BaseAdapter` MUST implement these **4 abstract methods**:
- `load_pipeline()` → returns the adapter backend pipeline/container (the default runtime expects
  a DiffusionPipeline-compatible eager object)
- `decode_latents()` → latents → pixels
- `inference()` → full multi-step denoising (corresponds to pipeline `__call__`)
- `forward()` → single-step denoising for training loss computation

**Optional encoder overrides (no-op default)**: All four per-modality encoders are non-abstract on `BaseAdapter`. Their default body is `pass` (returns `None`). Override only the modalities your model actually consumes — text/image/video-only adapters do **not** need stub `pass` overrides for unused modalities.
- `encode_prompt()` → text → embeddings
- `encode_image()` → image → latents
- `encode_video()` → video frames → latents
- `encode_audio()` → audio waveforms → embeddings/features

Note: `preprocess_func()` is a **concrete method** on `BaseAdapter` that dispatches to all four encoders (`prompt`, `images`, `videos`, `audios`) and skips integration when the called encoder returns `None`. It does NOT need to be overridden unless the model requires cross-modal preprocessing (e.g. prompt rewriting from images).

Breaking the signature of any of the four abstract methods (or changing the encoder return contract from "dict-or-`None`") breaks the entire training pipeline.

Offline-capable adapters additionally declare a `PipelineIOContract`, a declaration-only
`OutputStateCodec`, logical encoding components, and exact geometry validation. Condition and
output paths may share role-neutral transforms, but the official posterior `sample`/`argmax`
policy stays explicit at their semantic boundaries. Unsupported adapters declare an actionable
`output_state_codec_unavailable_reason` and fail before heavyweight loading.

**Adapter hierarchy**: All model adapters MUST inherit directly from `BaseAdapter` — never from another adapter. Shared logic between adapters for the same model family should use private helper functions, code duplication, or mixins — not adapter-to-adapter inheritance. Adapter subclassing creates fragile coupling where changes to a parent adapter silently break child adapters, and makes the 4-abstract-method contract harder to verify (the 4 per-modality encoders have no-op defaults, so a fresh subclass of `BaseAdapter` is always valid; chained inheritance hides which encoder a model actually overrides).

### 13. BaseRewardModel Paradigm Split
- `PointwiseRewardModel.__call__` receives an applicable sub-batch of at most configured
  `batch_size` and returns one reward per received sample.
- `GroupwiseRewardModel.__call__` receives one complete applicable canonical
  `(source_id, unique_id)` group and returns one reward per group member in input order.

The `RewardProcessor` dispatches differently based on the model type. Do not change the calling convention.

### 14. Sample Dataclass Hierarchy
`BaseSample` → `T2ISample`, `ImageConditionSample`, `T2VSample`, `T2AVSample`, etc. The `_shared_fields` class variable determines which fields are NOT stacked across a batch. Incorrect `_shared_fields` causes silent data corruption during collation.

`reconstruction_required_fields` separately names fields required to instantiate a concrete sample
after a partial distributed gather, even when the downstream consumer did not request them. See
[`topics/sample_lifecycle.md`](topics/sample_lifecycle.md#partial-gather-reconstruction).

**Two-layer hierarchy**: Task-level samples (`T2ISample`, `I2VSample`, `I2AVSample`, ...) are defined in `samples/samples.py` and inherit from `BaseSample` or its condition mixins (`ImageConditionSample`, `VideoConditionSample`). Model-specific samples (`LTX2Sample`, `LTX2I2AVSample`, ...) MUST inherit from the appropriate task-level sample — never from another model-specific sample across files. This mirrors the flat adapter hierarchy: `LTX2I2AVSample(I2AVSample)`, NOT `LTX2I2AVSample(LTX2Sample)`.

Legacy trajectory fields remain authoritative when `BaseSample.trajectory is None`. Structured
trajectory collation requires identical ordered component keys and shared state/log-prob index maps;
component mapping iteration never defines scheduler RNG order.

---

## Configuration System (15–17)

### 15. Pydantic Hparams Synchronization
All config dataclasses live in `hparams/`. The top-level `Arguments` aggregates `DataArguments`, `ModelArguments`, `TrainingArguments`, `RewardArguments`, `LogArguments`, etc. Field changes MUST be reflected in:
1. The dataclass definition
2. ALL YAML configs under `examples/` (renames/removals: search-replace; new user-facing fields: add with defaults and `# Options:` comments)
3. Any code that accesses `config.<field_name>`

### 16. Algorithm-Specific Training Args
`TrainingArguments` has algorithm-specific subclasses (`SFTTrainingArguments`,
`OfflineDPOTrainingArguments`, `GRPOTrainingArguments`, `DPPOTrainingArguments`,
`DPOTrainingArguments`, `DGPOTrainingArguments`, `NFTTrainingArguments`, `AWMTrainingArguments`,
`CRDTrainingArguments`, `DiffusionOPDTrainingArguments`, and the multi-role distillation classes).
The correct subclass is resolved by `get_training_args_class()`; adding an algorithm requires a
corresponding subclass and registry entry.

### 17. YAML Config Structure
Config keys must exactly match Pydantic field names. Typos fail silently with default values. See `examples/` for canonical config templates; structure defined in `hparams/args.py`.

---

## Distributed Training (18–20)

### 18. All-Rank Synchronization Points
`accelerator.wait_for_everyone()` must be called at critical synchronization points (after preprocessing, before/after evaluation, checkpoint saving). Missing barriers cause deadlocks or race conditions.

### 18a. Exact Resume Must Cover Replayed Evaluation
Exact-state identity MUST lock evaluation cadence, sampling configuration, ordered realized eval
loaders, per-dataset overrides, and eval rewards. Online checkpoints are saved before evaluation
and replay that evaluation after resume, so treating eval as an operational control permits global
device RNG drift. Exact-state save MUST fail before adapter or filesystem mutation on a device
whose RNG Accelerate cannot serialize; MPS users must use model-only checkpoints.

### 19. FSDP CPU Efficient Loading
Distributed loading is owned by `ModelLoadCoordinator` and its `BackendLoadRuntime`. TARGET roots may use rank-zero/meta FSDP2 loading only when the adapter declares that capability. AUXILIARY and REWARD resources are materialized as full per-rank replicas; FSDP auxiliary roots receive a cached sampled-fingerprint check, while reward loading is isolated from target-only loading state. Trainer code must not manipulate FSDP loading environment variables or broadcast component weights directly.

### 20. Mixed Precision Consistency
The adapter resolves `component_load_dtypes` at native load/materialization time, then applies frozen/trainable storage policy in `_mix_precision()`. Components materialized later must receive both policies through the component runtime and `on_load_components()`; laziness must not bypass either policy. Autocast context is configured in `BaseTrainer.__init__`. Do not manually cast tensors unless you understand the precision boundary. Details: `topics/dtype_precision.md`.

### 20a. Autocast Weight Cache Must Not Span a Forward
`torch.autocast`'s weight cache (keyed by tensor `data_ptr`) serves **stale** casts after any in-place weight change — `optimizer.step()` or a `use_ref/ema/named_parameters` swap (`param.data.copy_`). So wrap **each** forward (and its KL) in its own `with self.autocast():`; never one autocast around the optimize loop. Active for fp32 trainable weights (`trainable_parameters_dtype: fp32`), dormant for the bf16 default, LoRA `disable_adapter()` safe. Details + DDP/DeepSpeed caveat: `topics/autocast_param_swap.md`.

---

## Code Quality (21–27)

### 21. Formatting Standards
- **Black** with `line-length=100`, targeting Python 3.10–3.12
- **isort** with `profile="black"`, `line_length=100`
- Comments and docstrings in **English**

### 22. Import Style
- Use relative imports within `flow_factory` package (e.g., `from ..hparams import *`)
- Use absolute imports for external packages
- Follow existing wildcard import patterns for `hparams`
- **Top-level imports only**: All `import` / `from ... import ...` statements MUST live at the top of the module, never inside function bodies, methods, `__init__`, or conditional branches. Sanctioned exceptions: (a) optional dependencies wrapped in `try/except ImportError` (e.g., `deepspeed`, `xformers`); (b) backend-gated imports where the target symbol is only resolvable under a specific runtime backend already selected by a preceding feature check (e.g., DeepSpeed/FSDP submodules guarded by `is_deepspeed()` / `is_fsdp2()` in `models/abc.py`); (c) genuine unresolvable circular imports documented inline. Lazy imports added merely for "import speed" or "to keep the module light" are NOT acceptable — every hard dependency already runs through Python's import machinery on a typical import path. Inline imports hide the dependency surface from readers, `isort`, and static-analysis tools, and re-execute on every call in hot loops.

### 23. Type Annotations
All public methods must have type annotations. Use `typing` module types (`List`, `Dict`, `Optional`, `Tuple`, `Union`) for Python 3.10 compatibility.

### 24. License Header
All source files must include the Apache 2.0 license header with `Copyright 2026 Jayce-Ping`.

### 25. Logger Message Style
Logger messages referencing config parameters MUST use user-facing field names (not shorthand like `M`, `K`, `W`), show concrete values in parentheses (e.g., `unique_sample_num_per_epoch(32)`), and structure multi-constraint messages with numbered lines.

### 26. Fail-Fast Error Handling
Raise exceptions with detailed debug information over silent auto-fallback. Do not add defensive fallback code that silently recovers from invalid inputs. Auto-fallback is only acceptable when documented as intentional design. Details: `.cursor/rules/no-defensive-except.mdc`.

### 27. Docstring Style
All public functions and methods must have Google-style docstrings in English: imperative one-liner summary, `Args:`, `Returns:`, optional `Note:`. Private helpers (`_func`) may use a one-liner docstring if the behavior is obvious.

### 28. Agent Scratch Files
When an agent (sub-agent, background agent, or any automated tool) needs to write temporary files — investigation reports, analysis documents, checklists, diagrams, or any intermediate artifact that is NOT part of the final deliverable — it MUST write them under the `.scratch/` directory at the repository root. **Never** write temporary files to the project root or any tracked directory (`src/`, `guidance/`, `.agents/`, `.docs/`, `examples/`). `.scratch/` is git-ignored, so files there will not pollute the working tree or accidentally get staged.

### 29. Examples Directory Convention
Example configs follow the path convention `examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml`. Model directory names use underscores matching the config `model_type` field (e.g., `sd3_5`, `flux1_kontext`). The baseline config for a model is `default.yaml`. When adding, renaming, or removing examples, update all path references in `README.md`, `guidance/*.md`, and `examples/README.md`.

### 30. Framework-Upgrade GPU Merge Gate
Any broad change to the execution kernel, training dataflow, distributed backend, model
loading/preparation, sampler or batch geometry, reward/advantage pipeline, or optimizer/checkpoint
infrastructure MUST pass the exact-commit framework-upgrade GPU campaign before merge. The
machine-readable source of truth is `config/gpu_validation/framework_upgrade.yaml`. Its core jobs
must complete one full acquisition/optimization cycle on DDP, DeepSpeed ZeRO-2, and FSDP2; focused
supplemental jobs may reuse that backend coverage while isolating a constrained pairwise capability
matrix. Reward-pipeline changes MUST select task-appropriate rewards and cover every affected
overlap-capable trainer, compatible sampler placement, ready/ordered policy, single/multi-source
deployment, reducer ordering, and packed-batch boundary declared by the manifest. Enabled overlap
jobs must use real async pointwise services through CPU clients and prove optimization progressed
while reward work remained pending. Synchronous or in-process rewards are explicit non-overlap
boundary cases, never fabricated async evidence.
The reward-deployment/layout matrix is problem-specific constrained pairwise coverage, not a blind
Cartesian product. Every enabled overlap cell must resolve to at least the manifest-declared number
of independently schedulable optimizer work units; use per-run cycle overrides when an algorithm's
minimal synchronous smoke cycle would otherwise collapse to one tile, without changing trainer
semantics or inflating explicit synchronous boundary jobs.
Skipped, capacity-blocked, or infrastructure-blocked jobs do not count as passes, and a launcher
label is not backend evidence: the runtime distributed type and plugin version/stage must match the
manifest. Validate the manifest and attached result bundle with
`scripts/validate_gpu_campaign.py`. See `guidance/gpu_validation.md` for trigger scope, workload
geometry, artifact requirements, and the distinction between this merge gate and performance
benchmarks.
