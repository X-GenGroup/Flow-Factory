---
name: ff-new-model
description: "Add a Flow-Factory model adapter, including component runtime, model I/O, online trajectory, offline output-state, distributed loading, checkpointing, registry, examples, and parity verification."
---

# New Model Adapter Integration

## Context Routing

Read `../../../guidance/new_model.md`, Tier 1,
`../../knowledge/topics/adapter_conventions.md`, `../../knowledge/topics/component_runtime.md`, and
`../../knowledge/topics/parity_testing.md`. Read
`../../knowledge/topics/structured_trajectory.md` for multi-component models and
`../../../guidance/datasets.md` when adding offline support.

## 1. Design the Adapter Contracts

Decide before implementation:

- task inputs and outputs: image, video, audio-video, first/last frame, or ordered heterogeneous
  references;
- eager Diffusers, lazy modular, or explicit pseudo component runtime;
- canonical logical components, physical ownership roots, optional components, and aliases;
- legacy single-component or structured multi-component trajectory;
- online-only or lossless SFT/offline-DPO output-state support;
- class-level I/O superset and any checkpoint-realized specialization;
- supported finetune types, batch semantics, dtype policy, and FSDP2 capabilities.

Study a reference by contract rather than name alone: Flux/SD3 for classic image pipelines, Wan for
conditioned video, LTX2 for structured AV, MiniMax H3 for modular ordered AV, and Bagel/SenseNova for
pseudo multi-reference runtimes.

All adapters inherit directly from `BaseAdapter`. Model-specific samples inherit the matching
task-level sample (`T2ISample`, `I2VSample`, `T2AVSample`, `I2AVSample`, etc.), never another
model-specific sample.

## 2. Build the Component Runtime

All adapters implement `load_pipeline()`. The default `build_component_runtime()` wraps it in
`ClassicPipelineRuntime`. Lazy modular or explicit-container adapters override
`build_component_runtime()` and retain `adapter.pipeline` as the compatibility alias.

Declare component behavior precisely:

- canonical lookup owns component identity;
- overrides hold prepared proxies or checkpoint/LoRA replacements;
- declared specs are available for explicit lookup and role discovery;
- lifecycle enumeration contains materialized canonical `torch.nn.Module` values only;
- optional `None`, lazy-only specs, pseudo aliases, and prepared overrides are not implicit
  lifecycle roots;
- `materialize_components(None)` means already materialized modules. Name lazy requirements
  explicitly.

Use `has_component`, `get_component`, and `_require_component`; never gate component behavior with
`hasattr(adapter, name)`. Declare `preprocessing_modules` and `inference_modules`. Condition/output
encoding components are declared by their preparer/codec rather than loaded inside an encode call.

Do not call Accelerator prepare, FSDP wrapping, rank broadcasts, or manual target movement in an
adapter. `ModelLoadCoordinator` maps logical names to physical roots, and trainer initialization
prepares one `ModelBundle` plus one optimizer. After prepare, adapter access routes through
`RoutedComponentProxy`.

## 3. Implement the Core Adapter Surface

`BaseAdapter` keeps four abstract methods:

| Method | Contract |
|---|---|
| `load_pipeline()` | Return the native pipeline/container used by the runtime. |
| `decode_latents()` | Decode generated latent state to media. |
| `inference()` | Run the full denoising/generation loop. |
| `forward()` | Run one model step; this is the train-inference parity boundary. |

`encode_prompt`, `encode_image`, `encode_video`, and `encode_audio` are opt-in no-op encoders.
`preprocess_func()` dispatches them and should be overridden only for cross-modal preprocessing.
Preserve exact batch nesting: the outer list indexes samples and inner lists hold each sample's
media items; empty samples contribute `[]`, never `None` or a bare singleton.

Configure training through `default_target_modules`; YAML `target_components`/`target_modules`
builds `target_module_map`. Do not override the realized map.

`forward()` and `inference()` must agree on all generation-affecting inputs, scheduler state,
precision, and component order. Use `cast_latents()` symmetrically. Keep algorithm-specific logic
out of the adapter.

## 4. Add Structured State When Components Differ

For independently shaped or scheduled latent components:

- declare immutable `trajectory_component_order`;
- build a `SchedulerGroup` with exactly the same names and a canonical primary scheduler;
- emit `StructuredTrajectory` only, leaving legacy trajectory fields `None`;
- retain per-component schedules, state/log-prob index maps, callbacks, and active masks;
- extend protected state/bridge/reduction hooks so trainers consume terminal state, replay steps,
  forward-process noise, and reductions without inspecting storage format;
- derive scheduler/RNG/reduction order from the declared tuple, never mapping iteration.

Single-component adapters may retain legacy storage. Do not generalize their reduction order unless
the change preserves parity.

If a new concrete sample field is required by `__post_init__`, identity normalization, or
constructor invariants after a partial distributed gather, union it into the inherited
`reconstruction_required_fields`. That transport contract is separate from collator
`_shared_fields` and reward-model `required_fields`.

## 5. Declare Offline Output-State Support Explicitly

An adapter that claims SFT/offline-DPO support must provide every boundary below:

1. An immutable `PipelineIOContract` describing model-neutral ordered input/output media,
   semantic slots/cardinality, rates, geometry owner, and batch capability.
2. `_resolve_pipeline_io_contract()` only when checkpoint metadata narrows a class-level superset.
3. A declaration-only `OutputStateCodec` from `build_output_state_codec()`. It declares logical
   required components but cannot materialize, load, move, replace, or recast them.
4. `_validate_encoded_output_geometry()` comparing codec output against adapter-owned facts.
5. A declaration-only `ConditionStatePreparer` only when cached conditions are not the exact
   forward/output-codec condition.
6. One `PreparedConditionState` reused across every candidate and policy/reference forward.
7. A complete immutable `offline_training_forward_overrides` mapping. These values define finite-
   data model conditioning and are independent of rollout CFG settings.
8. An offline flow-matching objective reducer only when modality aggregation differs from online
   trajectory reduction.

Input-condition caches never contain target/chosen/rejected pixels or latent states. Shared numeric
transforms may serve condition and output paths, but posterior `sample` versus `argmax` policy stays
explicit at the semantic boundary. Candidate output context cannot overwrite input-owned fields.

If the adapter cannot represent output state losslessly, set a non-empty actionable
`output_state_codec_unavailable_reason`. Dataset acquisition then fails before model weights load;
online construction remains valid.

Do not override public boundary-owning wrappers such as `prepare_condition_state`,
`encode_output_state`, `forward_state`, or the shared reducers. Extend the protected hooks named by
their errors/docs.

## 6. Preserve Distributed and Checkpoint Ownership

- Declare every target component and frozen-but-shardable sibling needed in the prepared bundle.
- Preserve `_no_split_modules` or `_repeated_blocks` metadata so FSDP wrap discovery survives the
  bundle boundary.
- Opt into `supports_fsdp2_cpu_efficient_loading` only when selective rank-zero/meta target
  construction is correct. Treat additional wrap classes, default-stream unshard, backward-prefetch
  opt-out, and in-forward checkpointing as explicit model capabilities.
- Do not enable activation checkpointing inside `load_pipeline()`. The early backend plan selects
  one owner: FSDP2 moves a full model policy to backend ownership and rejects selective model
  policies; adapter-owned in-forward block boundaries require explicit opt-in.
- Apply component load dtype during native load, then frozen/trainable storage policy. Ensure lazy
  materialization receives both policies.
- Save/load trainable components symmetrically. Frozen-but-shardable bundle members do not own
  checkpoint artifacts. Test through prepared proxies as well as unprepared model-only load.

## 7. Register and Add Examples

Add a lowercase canonical key and lazy class path to `_MODEL_ADAPTER_REGISTRY`; preserve direct
Python-path fallback. Follow
`examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml` and update path references.

Add a generation example. For declared offline support, add SFT/offline-DPO examples or tested
fixtures whose strict V2 media matches the effective checkpoint contract. Document supported modes,
batch limits, geometry, rates, and intentionally unsupported paths.

## 8. Verification

- Runtime construction and lifecycle tests for the chosen classic/modular/pseudo runtime.
- Native pipeline config, components, one-step forward, final latent, and visual parity.
- Rollout/training `forward()` parity and initial on-policy ratio for coupled support.
- Legacy or structured trajectory bridge, active-mask, noise, and reducer tests.
- Selective-field distributed gather reconstructs every concrete sample with its inherited
  `reconstruction_required_fields` intact.
- DDP, ZeRO-2, and FSDP2 initialization/step where supported; verify bundle/proxy routing and FSDP
  wrap/checkpoint policy.
- LoRA/full and model-only checkpoint round trips only for finetune types claimed.
- One complete SFT epoch and offline-DPO epoch for every declared offline I/O mode, including exact
  geometry and prepared-condition reuse.
- An online-only adapter's offline selection fails before heavyweight loading.
- Registry and example parsing tests. Run `/ff-review` before commit.

## Common Failures

- Membership through adapter attributes, eager materialization of all lazy specs, or alias double
  movement.
- Preparing components outside the bundle or dropping repeated-block wrap metadata.
- Scheduler order from a mapping or multimodal state written into legacy fields.
- Adapter-to-adapter or model-sample-to-model-sample inheritance.
- Codec/preparer construction with materialization or device/dtype side effects.
- Using the class I/O superset instead of the checkpoint-effective contract.
- Redrawing an input condition per preference candidate or leaking sampling CFG into offline loss.
- Overriding public contract wrappers instead of protected hooks.
- Enabling both model and backend activation checkpointing.
