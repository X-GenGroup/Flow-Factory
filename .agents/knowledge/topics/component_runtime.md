# Component Runtime

**Read when**: Changing component discovery, loading, lifecycle, scheduler groups, or distributed
preparation.

## Runtime matrix

| Backend | Runtime | Loading contract |
|---|---|---|
| Eager diffusers pipeline | `ClassicPipelineRuntime` | `load_pipeline()` returns materialized components |
| Lazy modular pipeline | `ModularPipelineRuntime` | specs are declared first; explicit names materialize modules |
| Explicit non-diffusers container | `PseudoPipelineRuntime` | canonical roots and non-enumerated aliases are declared explicitly |

All adapters implement `load_pipeline()`. Non-classic adapters also override
`build_component_runtime()`. `adapter.pipeline` remains the backend compatibility alias.

## Lookup and enumeration

- **Canonical lookup** returns the backend-owned component or declaration.
- **Override lookup** returns a prepared proxy or LoRA/checkpoint replacement without changing
  canonical ownership.
- **Declared names/specs** are available for explicit lookup and role discovery.
- **Materialized names** include canonical `torch.nn.Module` instances only.
- Device/dtype lifecycle enumeration excludes declared-only specs, optional `None` entries,
  pseudo aliases, and prepared/replacement overrides.
- `materialize_components(None)` means already-materialized modules. Lazy specs require explicit
  names.

Implementation: `src/flow_factory/models/runtime/` and
`src/flow_factory/models/abc.py::BaseAdapter`.

## Canonical identity

The runtime is the only authority on which components exist. Never test membership with
`hasattr(adapter, name)`. That only agrees with the runtime for backends whose components happen
to be adapter properties; a lazy or modular runtime declares components that are never attributes,
and those silently drop out of every `hasattr`-gated loop. A declared training target that never
reaches the optimizer produces an empty parameter list rather than an error, which looks like a
training bug rather than a configuration one. Aliasing such a component onto `self.transformer`
does not fix it: that registers an extra override named `transformer` while the real component
stays invisible to the gated loops.

Use the three-way distinction:

| Question | Call |
|---|---|
| Does the runtime declare this name? | `adapter.has_component(name)` |
| Give me the module, tolerating a declared-optional absence | `adapter.get_component(name)` |
| Give me the module, this loop cannot proceed without it | `adapter._require_component(name)` |

`get_component` returns `None` for a component the backend declares but leaves unset — a Wan 2.1
checkpoint has no `transformer_2`, an image-only pipeline has no `audio_vae`. Dereferencing that
yields a bare `AttributeError` naming neither the component nor the loop, so lifecycle loops that
need a real module go through `_require_component`, which names the component and the declared set.

Overrides must name a declared component: one installed under an unknown name is unreachable,
because every reader resolves through the declared set. `_load_full_model` writes through
`set_component()` so a replacement stays visible to the runtime instead of shadowing it with a bare
instance attribute.

## Scheduler and distributed boundaries

- `adapter.scheduler` is the canonical primary scheduler.
- `SchedulerGroup.names` is immutable and equals `trajectory_component_order`.
- `SchedulerGroup` owns ordered mode and seed dispatch; mapping iteration never defines RNG order.
- `ModelBundle` plus `RoutedComponentProxy` is the sole distributed preparation runtime.
- Trainer stages call public adapter lifecycle methods so model-specific overrides remain active.
- `ModelLoadCoordinator` compiles logical names and physical roots into one immutable load plan;
  its backend runtime owns rank-zero/meta target loading and replicated auxiliary/reward loading.
- A target-owned composite root may still contain frozen auxiliary siblings. Pseudo runtimes move
  only that remainder and exclude every prepared target route.

## Activation-checkpoint ownership

`configure_checkpointing_backend_plan()` resolves one owner before model loading; direct trainer
construction repeats the same plan defensively after adapter realization.

| FSDP plan | Activation-checkpoint owner |
|---|---|
| FSDP1 with train-level model checkpointing | Model-side owner; a simultaneously enabled backend owner is disabled |
| FSDP1 + TDM-R1 | Both owners are disabled because reference/snapshot forwards invalidate FSDP1 recomputation |
| FSDP2 with no train-level model policy | The configured backend policy is retained; otherwise checkpointing stays off |
| FSDP2 with a full train-level policy | Accelerate/FSDP2 backend wrappers own recomputation; the train-level model policy is disabled |
| FSDP2 with a selective train-level policy | Rejected because backend wrappers cannot preserve selective model boundaries |
| FSDP2 with an adapter-owned in-forward capability | The adapter installs block-local boundaries after the FSDP input cast; the duplicate plugin owner is disabled during preparation |

MiniMax H3 Ref2VA currently owns the adapter-specific exception. Its prepared FSDP units also
receive a replay-time public `unshard()` hook for the affected PyTorch lifecycle, where a second
checkpoint graph can replay after another graph resharded the same unit. These flags are
model-specific memory policy and must not be copied to another adapter without evidence.

## Fix records

### Repeated-block metadata must survive the distributed bundle boundary

- **Date**: 2026-08-30
- **Symptom**: Two-rank LTX2 FSDP2 runs exhausted a 95 GiB GPU while
  `fully_shard()` initialized the prepared model root, before the first training step.
- **Root Cause**: LTX2 declares its 48 transformer units through Diffusers'
  `_repeated_blocks`, but `ModelBundle` surfaced only `_no_split_modules` to Accelerate;
  the empty auto-wrap policy therefore sharded the complete 19B transformer as one unit.
- **Fix**: `ModelBundle._no_split_modules` now falls back to `_repeated_blocks` for a
  member that has no legacy no-split declaration, with a regression that resolves the
  repeated block class through Accelerate's transformer-based FSDP policy.
- **Lesson**: A distributed wrapper becomes the metadata boundary seen by backend
  policy discovery. It must preserve both legacy and current model block declarations,
  or a correct component graph can silently collapse into one memory-prohibitive shard.
- **Related Constraint**: #9

## Failure modes

- Expanding omitted materialization to all declarations loads tokenizers, configs, and weights
  unexpectedly.
- Enumerating aliases or overrides as lifecycle roots moves or wraps the same parameters twice.
- Including optional `None` declarations in role groups fails during freeze/device operations.
- Preparing components outside `ModelBundle` breaks FSDP/DeepSpeed root ownership.
- Building scheduler order from mappings breaks deterministic multi-component RNG dispatch.

## Review checklist

- [ ] Canonical, override, declared, and materialized paths remain distinct.
- [ ] Membership resolves through the runtime, never `hasattr` on the adapter.
- [ ] A loop that dereferences a component uses `_require_component`.
- [ ] Lazy loading names every required component explicitly.
- [ ] Lifecycle enumeration contains canonical modules only.
- [ ] `adapter.pipeline`, `adapter.scheduler`, and public lifecycle hooks remain compatible.
- [ ] Scheduler names equal `trajectory_component_order`.
- [ ] Distributed preparation still routes through `ModelBundle`/`RoutedComponentProxy`.
- [ ] Exactly one activation-checkpoint owner remains active for the selected FSDP plan.

## Cross-refs

- UP: [`constraints.md` #5](../constraints.md#5-adapter-component-runtime-contract), [`architecture.md` Component Management](../architecture.md#component-management)
- PEER: [Structured Trajectory](structured_trajectory.md), [Component Variants](component_variants.md), [Adapter Conventions](adapter_conventions.md), [MiniMax H3](minimax_h3.md)

## FSDP2 replicated initialization

`ModelPreparation` asks `ModelLoadCoordinator.synchronize_initial_state()` to establish rank-zero
weights after all variants exist and before TP/FSDP sharding. FSDP2 without CPU-efficient loading
does not synchronize random LoRA weights itself. The backend broadcasts parameters and buffers
without replacing objects or changing RNG streams; metadata disagreement fails on all ranks
before mutation. Efficient loading retains Accelerate's broadcast.

CP preparation additionally consumes the native attention owner before Diffusers hooks and
attaches a prepared-root contract. Wan TP/SP+CP moves the native CP split before the torch SP
scatter and CP gather after the SP gather, retaining handles in that contract. Leaving the
native forward wrappers inside the torch hooks permutes tokens relative to rotary positions.
See [context parallelism](../../../docs/reference/parallelism/context-parallel.md).
