# Flow-Factory Architecture Overview

## Module Dependency Graph

```
                         ┌──────────┐
                         │ cli.py   │
                         │ train.py │
                         └────┬─────┘
                              │
                    ┌─────────▼─────────┐
                    │     Arguments     │  (hparams/)
                    │  Top-level config │
                    └──┬────┬────┬──────┘
                       │    │    │
          ┌────────────┘    │    └────────────┐
          ▼                 ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  BaseTrainer  │  │ BaseAdapter  │  │BaseRewardModel│
   │  (trainers/)  │  │  (models/)   │  │  (rewards/)  │
   └──┬───┬───┬───┘  └──┬───┬───┬──┘  └──┬───┬───┬───┘
      │   │   │         │   │   │         │   │   │
      ▼   ▼   ▼         ▼   ▼   ▼         ▼   ▼   ▼
    GRPO SFT DPO     Flux SD3 Wan     PickScore CLIP OCR
```

### Key Dependency Rules

| Module | Depends On | Depended By |
|--------|-----------|-------------|
| `hparams/` | (standalone) | Everything |
| `models/abc.py` | `hparams`, `samples`, `ema`, `scheduler`, `utils` | All model adapters, `trainers/abc.py` |
| `trainers/abc.py` | `hparams`, `models/abc.py`, `rewards/`, `advantage/`, `data_utils/`, `logger/` | All trainer subclasses |
| `advantage/` | `hparams`, `rewards/`, `samples/` | `trainers/abc.py` |
| `rewards/abc.py` | `hparams` | All reward models, `trainers/abc.py` |
| `data_utils/` | `hparams` | `trainers/abc.py` |
| `scheduler/` | (standalone) | `models/abc.py` |
| `samples/` | `utils/` | `models/`, `rewards/`, `advantage/`, `trainers/` |
| `ema/` | `utils/` | `models/abc.py` |
| `logger/` | `hparams` | `trainers/abc.py` |
| `utils/` | (standalone) | Most modules |

---

## Execution Pipelines

> Authoritative reference: `guidance/workflow.md`

`ExecutionContract` separates acquisition (`generation` or `dataset`) from feedback
(`runtime_reward` or `none`). `PipelineIOContract` independently owns model input/output media,
rates, geometry, and batching.

| Composition | Driver | Optimization entry | Cycle counter |
|---|---|---|---|
| Generation + runtime reward | K-repeat → inference → reward → advantage | `optimize(samples)` | `rollout_iteration` |
| Generation + no feedback | Generation/distillation path | `optimize(samples)` | `rollout_iteration` |
| Dataset + no feedback | Finite official `DistributedSampler` traversal | `optimize_batch(batch)` | `data_epoch` |

`optimizer_step` advances independently. A dataset epoch advances only after clean loader
exhaustion; offline output media is decoded and encoded on the fly, while only prompt/input
conditions enter the preprocessing cache.

Generated runtime-feedback trainers may separately declare a
`RewardOptimizationOverlapContract`. `BaseTrainer` then seals the async `RewardBuffer`, builds
optimizer work units from the algorithm's rank-local, global-batch, or global-tile geometry,
intersects readiness across ranks, and invokes internal prepare/tile/finalize lifecycle hooks
without changing the public `sample`/`prepare_feedback`/`optimize` API. `SamplerLayoutContract`
owns where complete groups become available; `SamplerSelectionContract` declares what an
algorithm can consume; `FeedbackReducerContract` declares whether reward-to-advantage reduction
needs acquisition-wide statistics. Reward-model request batching remains independent of optimizer
work-unit boundaries. An `AcquisitionManifest` preserves canonical `(source_id, unique_id)`
identity and original rollout microbatch boundaries for pack-composition-dependent adapters such
as Bagel. Rank-local objectives close groups and GAS per work unit; TDM-R1 may close one role
accumulation window across all work units. This capability does not change `ExecutionContract`;
dataset and no-feedback compositions remain structurally bypassed.
Generated cycles also publish distributed critical-path timings from `BaseTrainer`: generic stage
durations use the top-level `timing/` namespace, streamed-reward diagnostics use
`timing/reward_overlap/`, and algorithm loss/reward metrics remain under `train/`.

Exact runtime identity is built from realized prepared state. It locks optimizer/model/backend
semantics, the checkpoint-realized pipeline I/O contract, ordered training data, and the complete
replayed evaluation path (cadence, arguments,
per-dataset overrides, rewards, and ordered prepared loaders). Logging, checkpoint cadence, run
budget, and resume location remain operational. Exact-state save fails before mutation on MPS
because Accelerate does not persist the device RNG needed for exact continuation.

---

## Registry System

All four registries map string keys → lazy import paths. Resolution: registry lookup → fallback to direct Python path → dynamic import. See `trainers/registry.py`, `models/registry.py`, `rewards/registry.py`, `acceleration/registry.py` for implementation.

### Registered Components

**Trainers** (`trainers/registry.py`):

| Key | Class | Paradigm | Base Class |
|-----|-------|----------|------------|
| `sft` | `SFTTrainer` | Decoupled, dataset | `BaseTrainer` |
| `offline-dpo` | `OfflineDPOTrainer` | Decoupled, dataset | `BaseTrainer` |
| `grpo` | `GRPOTrainer` | Coupled | `BaseTrainer` |
| `grpo-guard` | `GRPOGuardTrainer` | Coupled | `GRPOTrainer` |
| `dppo` | `DPPOTrainer` | Coupled | `GRPOTrainer` |
| `dpo` | `DPOTrainer` | Decoupled | `BaseTrainer` |
| `dgpo` | `DGPOTrainer` | Decoupled | `BaseTrainer` |
| `nft` | `DiffusionNFTTrainer` | Decoupled | `BaseTrainer` |
| `awm` | `AWMTrainer` | Decoupled | `BaseTrainer` |
| `crd` | `CRDTrainer` | Decoupled | `BaseTrainer` |
| `diffusion-opd` | `DiffusionOPDTrainer` | Distillation, generation + no feedback | `BaseTrainer` |
| `dmd2` | `DMD2Trainer` | Distillation, generation + no feedback, ODE | `BaseTrainer` |
| `tdm` | `TDMTrainer` | Distillation, generation + no feedback, ODE | `BaseTrainer` |
| `tdm-r1` | `TDMR1Trainer` | Decoupled, generation + runtime reward, ODE | `TDMTrainer` |

**Flat hierarchy**: New trainers inherit from `BaseTrainer` directly. The sanctioned existing
extensions are `GRPOGuardTrainer → GRPOTrainer`, `DPPOTrainer → GRPOTrainer`, and
`TDMR1Trainer → TDMTrainer`; see constraint #11.

**Model Adapters** (`models/registry.py`):
| Key | Class | Task |
|-----|-------|------|
| `sd3-5` | `SD3_5Adapter` | Text-to-Image |
| `flux1` | `Flux1Adapter` | Text-to-Image |
| `flux1-kontext` | `Flux1KontextAdapter` | Image-to-Image |
| `flux2` | `Flux2Adapter` | Text-to-Image & Image(s)-to-Image |
| `flux2-klein` | `Flux2KleinAdapter` | Text-to-Image & Image(s)-to-Image |
| `qwen-image` | `QwenImageAdapter` | Text-to-Image |
| `qwen-image-edit-plus` | `QwenImageEditPlusAdapter` | Image(s)-to-Image |
| `qwen-image-2.1` | `QwenImage21Adapter` | Text-to-Image & Image(s)-to-Image |
| `z-image` | `ZImageAdapter` | Text-to-Image |
| `wan2_t2v` | `Wan2_T2V_Adapter` | Text-to-Video |
| `wan2_i2v` | `Wan2_I2V_Adapter` | Image-to-Video |
| `ltx2_t2av` | `LTX2_T2AV_Adapter` | Text-to-Audio-Video |
| `ltx2_i2av` | `LTX2_I2AV_Adapter` | Image-to-Audio-Video |
| `minimax-h3-t2va` | `MiniMaxH3T2VAAdapter` | Text-to-Video-Audio |
| `minimax-h3-fl2va` | `MiniMaxH3FL2VAAdapter` | Sparse First/Last-Frame-to-Video-Audio |
| `minimax-h3-ref2va` | `MiniMaxH3Ref2VAAdapter` | Ordered-Reference-to-Video-Audio |
| `bagel` | `BagelAdapter` | Text-to-Image & Image(s)-to-Image (T2I & I2I both batched via NaViT packing; subset-round packing handles variable I2I reference-image count, no per-sample fallback — see `topics/adapter_conventions.md`) |
| `sensenova` | `SenseNovaAdapter` | Text-to-Image & Image(s)-to-Image (SenseNova-U1 1.0/1.5; ordered variable-count references remain grouped in `images` and preserve within-type order; independent samples use B=1 prefixes rather than Bagel-style NaViT packing) |

**Reward Models** (`rewards/registry.py`):
| Key | Class | Type |
|-----|-------|------|
| `pickscore` | `PickScoreRewardModel` | Pointwise |
| `pickscore_rank` | `PickScoreRankRewardModel` | Groupwise |
| `clip` | `CLIPRewardModel` | Pointwise |
| `clap` | `CLAPRewardModel` | Pointwise |
| `imagebind` | `ImageBindRewardModel` | Pointwise |
| `ocr` | `OCRRewardModel` | Pointwise |
| `vllm_evaluate` | `VLMEvaluateRewardModel` | Pointwise |
| `rational_rewards_t2i` | `RationalRewardsT2IRewardModel` | Pointwise |
| `rational_rewards_edit` | `RationalRewardsEditRewardModel` | Pointwise |
| `geneval` | `GenEvalRewardModel` | Pointwise |
| `geneval2_soft_tifa` | `GenEval2SoftTIFARewardModel` | Pointwise |
| `hpsv2` | `HPSv2RewardModel` | Pointwise |
| `qwen_image_bench` | `QwenImageBenchRewardModel` | Pointwise |

**Accelerators** (`acceleration/registry.py`):
| Key | Class | Safety | Stage | Notes |
|-----|-------|--------|-------|-------|
| `attention_backend` | `AttentionBackendAccelerator` | lossless | both | Sets the diffusers attention backend on every transformer (requires a `backend` param). Listed as a `shared` entry (before `torch_compile`); this is the single code path for backend selection (the old `BaseAdapter._set_attention_backend` and the `model.attn_backend` knob were both removed — a config still setting `model.attn_backend` fails fast). Bagel forces flash_attention_2 at load and does not use it. |
| `torch_compile` | `CompileAccelerator` | lossy | both | `torch.compile` of the shared transformer (`auto` default: regional when `_repeated_blocks` is available, otherwise full; explicit regional/full overrides); applied in-place after `post_init` so checkpoint keys / param identity stay stable. Marked `lossy` because it is applied symmetrically but is **not bit-exact across rollout vs training** (Inductor's grad/no-grad graph split → intermittent ~1e-5 on-policy residual, within `clip_range`); still allowed on coupled algos, validator warns. |
| `diffusers_cache` | `DiffusersCacheAccelerator` | lossy | rollout | Diffusers `CacheMixin` feature caching (first_block/faster/pyramid/taylorseer/magcache), gated before mutation by `supports_diffusers_cache` plus the optional `supported_diffusers_cache_policies` whitelist. MiniMax H3 T2VA/FL2VA/Ref2VA internally bridge diffusers 0.40.0 for `first_block` only and reject combination with Flow-Factory `torch_compile`. |

Configured via the `acceleration:` block (`hparams/acceleration_args.py`): two ordered lists of `{name, params}` entries — `shared` (persistent `stage='both'` accelerators applied to rollout and training) and `rollout` (Stage-3 only). **List order is application order**: `shared` entries run their `setup()` in order (so `attention_backend` must precede `torch_compile`), and `rollout` entries nest their `rollout_context()` in order. The `acceleration/validator.py` enforces that a **lossy `rollout`** accelerator runs only on `decoupled`/`distillation` trainers (each trainer declares a `paradigm`), preserving train-inference consistency (constraint #7). A **lossy `stage='both'` accelerator in the `shared` slot** (e.g. `torch_compile`, applied symmetrically but not bit-exact across stages) is allowed on any paradigm but the validator **warns** on coupled trainers that the on-policy ratio will be ≈1, not exactly 1. Off by default.

---

## Extension Points

- **New model adapter**: `guidance/new_model.md`, skill `/ff-new-model`, conventions `topics/adapter_conventions.md`
- **New reward model**: `guidance/rewards.md`, skill `/ff-new-reward`
- **New algorithm**: `guidance/algorithms.md`, skill `/ff-new-algorithm`. `BaseTrainer` owns the cycle loop, acquisition drivers, feedback/advantages, optimizer step, and velocity KL. Generation trainers implement `optimize(samples)`; dataset trainers implement `optimize_batch(batch)`. An algorithm that trains several model copies declares them in `_declare_model_variants()` (`topics/component_variants.md`).
- **New accelerator**: subclass `acceleration/abc.py::BaseAccelerator` (declare `safety`/`stage`), register in `acceleration/registry.py`

---

## Key Design Patterns

### Timestep & Sigma Convention

Timesteps are `[0, 1000]` (scheduler scale); sigmas are `[0, 1]` (flow-matching noise level). Details: `topics/timestep_sigma.md`.

### Adapter Pattern (Models)
Each model adapter wraps a diffusers pipeline into the `BaseAdapter` interface:
- `preprocess_func()` — prompt/input-condition preprocessing and cache projection
- `pipeline_io_contract` — model-neutral input/output modality and geometry declaration
- `effective_pipeline_io_contract` — checkpoint-realized specialization of the class contract
- `prepare_condition_state()` — one validated input-owned runtime realization reused across
  candidate encoding and model forwards
- `encode_output_state()` — validated on-the-fly offline target encoding through an optional codec
- `inference()` — full denoising loop (Stage 3)
- `forward()` — single-step denoising (Stage 6)

**Per-modality encoders** (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`) are no-op by default on `BaseAdapter` — override only the modalities your model consumes. `preprocess_func` dispatches to all four and skips any that return `None`, so text/image/video-only adapters need no stub overrides for unused modalities.

Offline condition preparers and codecs declare logical required components without materializing
them. Condition/output encoders share role-neutral transforms where possible, while callers retain
explicit official posterior `sample` versus `argmax` semantics. Candidate-specific output context
cannot overwrite cached or prepared input fields. A separate flow-matching objective reducer lets
multi-modal SFT/DPO specialize loss aggregation without changing online trajectory reductions.

Input contracts may declare semantic media slots and aggregate cross-type cardinality rules. In
strict V2 data, an explicit input-only `slot` reserves its argument; unslotted media fills remaining
slots in declaration order. Outputs reject slots. This keeps algorithm data model-neutral while the
adapter owns bindings such as first/last frame and ordered heterogeneous references.

**Flat hierarchy**: All adapters inherit directly from `BaseAdapter` — never from another adapter (see constraint #12). Shared logic within a model family uses helper functions, code duplication, or mixins — not adapter subclassing.

Details: `topics/adapter_conventions.md`

### Sample Dataclass Hierarchy
Two-layer structure (constraint #14): task-level samples (`T2ISample`, `I2VSample`, `I2AVSample`, ...) live in `samples/samples.py` and inherit from `BaseSample` or condition mixins. Model-specific samples (`LTX2Sample`, `LTX2I2AVSample`, ...) inherit from the matching task-level sample — never from another model-specific sample.

`BaseSample.trajectory` is the opt-in structured path for independently shaped/timed latent
components; legacy trajectory fields remain unchanged and authoritative when it is `None`.
LTX2 T2AV/I2AV rollouts are the first adapters to opt in: they publish an authoritative
`StructuredTrajectory` and leave every legacy trajectory field `None`
(see `topics/adapter_conventions.md` gotcha #12).

### Component Management
`BaseAdapter` delegates component discovery, canonical access, runtime overrides, lazy
materialization, and stage-device lifecycle to `models/runtime/`. Runtime overrides include
prepared/proxied modules and LoRA/checkpoint replacements; all are excluded from manual device
management. Declared lazy specs are separate from materialized module enumeration, so stage-wide
operations and `materialize_components(None)` never materialize tokenizers, schedulers, processors,
or configs implicitly. Explicit names are required to materialize a lazy spec. Role groups retain
non-`None` modular specs but exclude absent classic optional components. The default
`ClassicPipelineRuntime` preserves eager DiffusionPipeline behavior;
`ModularPipelineRuntime` materializes selected lazy component specs; and
`PseudoPipelineRuntime` manages explicit containers and non-enumerated aliases such as Bagel's
`transformer -> bagel.language_model`, as well as direct canonical components such as SenseNova's
`transformer -> SenseNovaDenoiser`. `adapter.pipeline` remains the backend compatibility alias,
while `ModelBundle` and `RoutedComponentProxy` remain the sole distributed preparation runtime.
`loading/` compiles logical component names into physical ownership roots before trainer-stage
loads. `ModelLoadCoordinator` excludes prepared target routes from auxiliary movement while
allowing frozen siblings in the same composite root to become resident. `BackendLoadRuntime`
owns FSDP target-buffer setup, replicated-load scopes, and cached sampled fingerprints. Trainer
methods only request component loading and distributed preparation.
`SchedulerGroup` separately provides immutable component names and ordered scheduler mode/seed
dispatch; its primary scheduler remains available through `adapter.scheduler`.

Component membership resolves through the runtime, never through `hasattr(adapter, name)`:
`has_component` asks whether a name is declared, and `_require_component` fetches a module a
lifecycle loop cannot proceed without. Details: `topics/component_runtime.md`.

### Component Variants
`BaseAdapter` is infrastructure: it supplies mechanisms and holds no algorithm vocabulary. Two
mechanisms cover parameter ownership. Named parameter snapshots (`add_named_parameters` /
`use_named_parameters`) are temporal, one set of weights installed at a time, and cover references,
EMAs and old snapshots. `ComponentVariantRegistry` (`models/variants.py`) is spatial: several
trainable copies live at once, each with its own optimizer group, storage (`lora` or `full`) and
`component_routes`. Variant names are caller-chosen and the base variant is positional, so the
model layer never learns what a "generator" is. `RoutedComponentProxy` resolves a canonical
component name through the active variant, so adapter code is unchanged.

A variant is always a live trainable copy. A frozen reference is the same weights at another point
in time, so it belongs to the temporal mechanism: `use_ref_parameters()` for the pre-finetune
weights, or a named snapshot. Only the trainable copy carries gradients and optimizer state, which
is what forces it into the prepared bundle in the first place.

Roles are the trainer's vocabulary. `RoleOptimizationCoordinator`
(`trainers/role_optimization.py`) is a utility a trainer composes to drive disjoint role updates
through one physical optimizer, and `BaseTrainer._validate_multirole_backend` rejects the
distributed layouts multi-role cannot support. Algorithms may duplicate their own small role
helpers rather than share an abstraction that would push their vocabulary down a layer.
Details: `topics/component_variants.md`.

#### Component runtime enumeration boundaries
- **Date**: 2026-08-10
- **Symptom**: Lazy stage-wide operations could materialize non-module specs, Bagel's nested
  transformer could be moved twice, and trainers bypassed adapter lifecycle overrides.
- **Root Cause**: The first runtime abstraction conflated declared specs, materialized modules,
  aliases, and prepared/replacement overrides under one component-name path.
- **Fix**: Split declared and materialized discovery, added non-enumerated pseudo aliases and
  generic device-excluded overrides, and restored trainer routing through adapter lifecycle APIs.
- **Lesson**: Discovery for explicit lookup and enumeration for lifecycle operations require
  separate contracts; aliases and overrides must remain addressable without becoming lifecycle
  roots.
- **Related Constraint**: #5.

#### Optional role discovery and lazy default materialization
- **Date**: 2026-08-10
- **Symptom**: A declared classic `transformer_2=None` entered the transformer role group and
  adapter freezing called `requires_grad_` on `None`; separately, `materialize_components(None)`
  eagerly loaded every modular spec.
- **Root Cause**: Role discovery filtered names rather than non-`None` values, and the default
  materialization request expanded declared names instead of materialized modules.
- **Fix**: Role discovery now excludes `None` values while retaining non-`None` modular specs;
  default materialization uses already-materialized module names, and normal canonical lookup
  returns direct materialized attributes before consulting the expensive declared component map.
- **Lesson**: Optional declarations are valid for explicit compatibility lookup but cannot imply
  role membership, and an omitted lazy-materialization selection must never mean "load all."
- **Related Constraint**: #5.

#### Structured trajectory bridge ownership boundaries
- **Date**: 2026-08-10
- **Symptom**: Batch-level state arguments could be forwarded twice, partial active-count
  overrides were rejected, and a plain mapping with structured trajectory data raised an
  incidental attribute error.
- **Root Cause**: The legacy bridge did not separate bridge-owned forward arguments from
  batch conditioning, and treated optional component metadata as a complete mapping.
- **Fix**: The bridge now strips state-owned batch keys, accepts ordered partial active-count
  overrides while rejecting unknown components, and validates the structured batch type before
  accessing batch metadata.
- **Lesson**: Bridge-owned values must have one authoritative source; optional component
  metadata should be consumed in authoritative component order without requiring every key.
- **Related Constraint**: #5, #26.

### Reward Processing
`RewardProcessor` dispatches by model type:
- **Pointwise**: applicable sub-batches of at most `batch_size`
- **Groupwise**: group by canonical `(source_id, unique_id)` identity (local or distributed path)
- **Multi-reward**: weighted aggregation
- **Async**: optional non-blocking computation

### Advantage Computation
`AdvantageProcessor` (`advantage/advantage_processor.py`): communication-aware, auto-selects gather vs local path. Strategies: `"sum"` (GRPO) and `"gdpo"`. Runtime-reward trainers delegate to `self.advantage_processor.compute_advantages()`. Feedback-`none` trainers (`diffusion-opd`, DMD2, and TDM) bypass reward and advantage stages structurally.

### Configuration Hierarchy
```
Arguments (top-level)
├── ModelArguments        # model_type, model_path, finetune_type, LoRA config
├── TrainingArguments     # Algorithm-specific (SFT/offline-DPO/GRPO/DPO/etc.)
├── SchedulerArguments    # dynamics_type, timestep_range, num_inference_steps
├── DataArguments         # dataset, preprocessing, resolution, sampler_type
├── MultiRewardArguments  # reward_model configs (list of RewardArguments)
├── MultiOptimizerArguments  # YAML `optimizers:`, one entry per variant (AdamW or Muon)
├── AccelerationArguments # YAML `acceleration:`, compile / attention backend / caching
├── LogArguments          # logger type, verbose, project name
└── EvaluationArguments   # evaluation settings
```
