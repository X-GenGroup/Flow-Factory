# MiniMax H3

**Read when**: Changing H3 adapters, workflows, schedules, references, dependency pins, examples, or
verification claims.

## Dependency and workflows

Required dependency: `diffusers>=0.40.0`.

| Registry key | Workflow input | Trainable component |
|---|---|---|
| `minimax-h3-t2va` | prompt | `transformer` |
| `minimax-h3-fl2va` | prompt plus first frame, optionally last frame | `transformer` |
| `minimax-h3-ref2va` | prompt plus ordered heterogeneous references | `transformer_ref` |

The feature probe must validate public symbols, workflow maps, block call surfaces,
row-timestep construction, reference constructors, and modular APIs. A no-weight
`from_config`/component-spec/workflow build proves API compatibility only.

## Adapter structure

The three adapters differ only in workflow identity, canonical transformer name, and component
lists. Everything workflow-invariant lives on the `_MiniMaxH3WorkflowAdapter` mixin, so each
method has exactly one implementation (`test_workflow_adapters_share_one_implementation_per_method`
enforces this). Each adapter is still a direct `BaseAdapter` subclass and the mixin is not one, so
the registry's single-base contract holds; tests assert `BaseAdapter in cls.__bases__` plus no
second `BaseAdapter` subclass in the MRO, rather than an exact `__bases__` tuple.

`_forward_state` replays a stored transition through the public `forward()` instead of a private
path, which keeps rollout and replay on one entry point. Replay receives every collated batch
field, so `build_h3_replay_forward_kwargs()` selects the conditioning arguments `forward()` accepts
and lets that boundary stay strict.

## Hard execution contract

- Batch size is B=1 for preprocessing, rollout, and evaluation.
- H3 has no CFG: guidance is `1.0`, and negative prompts are empty or absent.
- Rollout is structured-only with separate video and audio states.
- Authoritative scheduler/component order is `("video", "audio")`.
- Video uses shift 12; audio uses shift 3.
- Model output is data-ward velocity. Adapters declare `flow_velocity_direction="data"` so both
  scheduler conversion and trainer `x0` projection use the correct sign.
- `num_inference_steps=N` means N transitions and N + 1 state coordinates.

## Diffusers cache contract

All three workflows opt into rollout-only diffusers caching for `first_block` only. Diffusers
0.40.0 does not register `MiniMaxH3TransformerBlock` for FirstBlockCache, so the adapter installs
the required `(hidden=0, encoder=None)` metadata before cache enablement. Registration targets each
actual unwrapped main-block class, including the runtime subclass produced by FSDP2, is idempotent,
and rejects incompatible upstream metadata instead of overwriting it. The token-refiner stack is
not registered because diffusers FirstBlockCache scans only the top-level `transformer_blocks`.

`infer_h3_workflow()` resets stateful cache hooks once at the start of each independent B=1
generation. Every step opens the stable `minimax_h3_<workflow>` context on the prepared component;
the same proxy routes the actual call through the prepared bundle, including `transformer_ref` for
Ref2VA. An exceptional forward resets the state before re-raising. This lifecycle has a real tiny-H3
CPU cache-hit/reset test on diffusers 0.40.0. Treat it as lifecycle evidence; deployment speed,
memory, reward, and quality remain workload-dependent.

Do not combine H3 FirstBlockCache with Flow-Factory `torch_compile`. Compile's forced-grad rollout
path would let the diffusers 0.40.0 cache retain autograd graphs across denoising steps, so the H3
shim fails before registry mutation. Because caching is lossy and rollout-only, only decoupled or
distillation trainers may use it; coupled GRPO/GRPO-Guard/DPPO remain validator-rejected.

## Input contracts

- T2VA accepts prompt-only workflow input.
- FL2VA accepts semantic `first_frame`, `last_frame`, or both image slots. Unslotted legacy
  generation input retains first-then-last positional shorthand; strict V2 can express last-only.
- Ref2VA preserves and hashes 1-12 ordered image/video/audio references and requires at least one
  image or video.
- Ref2VA declares `supports_ordered_references=True`; all H3 adapters explicitly declare hidden
  geometry cache fields and a preprocessing cache version.
- Reference paths are dataset-relative. Positive finite `fps` and `sample_rate` overrides follow
  `samples/references.py`.
- PyAV >=17.0.0 decodes video/audio references, including embedded or separate soundtracks.

## Offline audiovisual output contract

All three H3 workflows support SFT and offline DPO with one exact ordered output pair: video first,
then audio. Both `fps` and `sample_rate` are required in V2 supervision.
Targets are decoded on demand; neither pixels, waveforms, nor VAE latents enter the condition
cache. The pipeline's single-sample capability also forces condition-cache preprocessing to B=1,
independently of the global preprocessing batch-size setting.

The codec cross-validates cached layout and geometry against the current training config. It
resamples video onto the configured fixed 24-fps grid and canvas, truncates audio on its declared
source clock before a single conversion to the audio-VAE rate, and aligns stereo audio to the exact
latent duration, takes and normalizes both posterior modes, then packs structured rows in
`("video", "audio")` order. The deterministic video mode follows the H3 SFT reference data flow;
the released Diffusers pipeline does not define an offline target encoder. The codec does
not duplicate input-owned fields in output forward context. Replay nests the flat cached layout
and derives empty T2VA condition prefixes from the current state, preserving storage dtype and
device. Exact velocity-only offline forwards return before either component scheduler steps, so
SFT and offline DPO do not sample unused transitions or perturb scheduler RNG cadence. Every
encoded row count must match the cached layout before transformer execution.

FL2VA and Ref2VA use a separately owned runtime condition-prefix preparer. It realizes the official
condition noise once per batch, then exposes immutable model-forward and output-binding views. Both
offline-DPO arms and their policy/reference forwards consume the same prepared prefix object; target
encoding never draws condition noise independently. Offline flow matching sums the separate video
and audio means, while online likelihood and distillation retain their existing globally
element-weighted reducer.

## Fix records

### H3 offline targets preserve the reference posterior and modality objective

- **Date**: 2026-08-29
- **Symptom**: H3 clean video targets changed on every encode and audiovisual flow loss weighted
  modalities by tensor cardinality, heavily downweighting audio.
- **Root Cause**: The first offline codec inferred stochastic video-posterior sampling where
  Diffusers has no target recipe, and inherited the online global reducer for a two-term SFT loss.
- **Fix**: Video and audio target codecs now take deterministic posterior modes, matching the H3
  SFT reference data flow, and the offline flow hook returns `video_mean + audio_mean` without
  changing the online reducer. Unequal-cardinality regression tests lock both decisions.
- **Lesson**: When an inference pipeline omits training semantics, use the nominated training
  reference for posterior selection and keep objective-specific modality weighting separate from
  trajectory likelihood aggregation.
- **Related Constraint**: #7, #8

### Offline targets preserve configured geometry and logical source clocks

- **Date**: 2026-08-28
- **Symptom**: An internally valid stale H3 condition cache could select another output canvas or
  frame count, while an audio `sample_rate` override caused decoder resampling followed by a
  second codec resample.
- **Root Cause**: The output codec trusted cached geometry without comparing the current training
  config, and the generic audio decoder treated source-rate metadata as a target decode rate.
- **Fix**: The codec now cross-validates cached H/W and the officially aligned frame count against
  current training arguments. Audio decoding preserves file samples; the codec truncates on the
  declared source clock before exactly one model-rate conversion.
- **Lesson**: Cached geometry must be checked against its configured authority, and media rate
  overrides describe logical source clocks rather than preprocessing requests.
- **Related Constraint**: #8, #26

### Velocity-only H3 forwards bypass both schedulers

- **Date**: 2026-08-28
- **Symptom**: SFT and offline DPO requested only velocity but still sampled unused video/audio
  scheduler transitions, wasting memory and changing RNG cadence.
- **Root Cause**: The adapter boundary did not route the exact velocity-only request to the
  existing scheduler-free `forward_h3_state` path.
- **Fix**: The adapter detects a non-log-probability `("velocity",)` request with no replay next
  state, returns `MultiModalStepOutput(velocity=...)`, and never enters either scheduler.
- **Lesson**: Decoupled velocity objectives must stop at model prediction; requesting fewer return
  fields is not sufficient if the adapter still executes transition side effects.
- **Related Constraint**: #7

### Single-sample capability governs condition preprocessing

- **Date**: 2026-08-28
- **Symptom**: A multi-row H3 offline manifest could reach its B=1 preprocessor with the global
  condition-cache batch size, even though the training loader correctly rejected B>1.
- **Root Cause**: The offline cache builder forced row-wise preprocessing only for ordered
  references and did not apply the adapter's general batching capability.
- **Fix**: The cache builder now derives its effective preprocessing batch size from both ordered
  reference binding and `BatchCapability.SINGLE_SAMPLE`.
- **Lesson**: One adapter-owned batching contract must govern preprocessing and model execution;
  otherwise framework stages can disagree before training starts.
- **Related Constraint**: #8, #12

## Verification boundary

All workflows have pinned API/schema/no-weight verification and local offline codec/forward
coverage. FirstBlockCache additionally has a real diffusers-0.40.0 tiny-model CPU cache-hit/reset
lifecycle test, including FSDP2-style runtime block subclasses and Ref2VA prepared-route ownership.
T2VA `debug.yaml` additionally completed real-weight LoRA CPS rollout, video/audio decode, CLAP
reward, replay, backward, checkpoint, and resume on one GPU and with FSDP2 on 16 GPUs across two
nodes, using the 61 GB checkpoint (61.74 GiB transformer); its 64x96 canvas is a correctness and
memory-fit geometry. The native-resolution `quality_720p_fsdp2.yaml` path completed FSDP2
initialization, checkpoint, decode, and CLAP evaluation. The
[PR #220 matrix](gpu_validation.md#pr-220-result) then completed all 36 H3
real-weight smoke cells: T2VA, FL2VA, and Ref2VA across DDP/DeepSpeed ZeRO-2/FSDP2 and
GRPO/SFT/offline DPO/TDM. The FL2VA first-plus-last SFT/offline-DPO variant gate also passed. Do
not claim long-run reward improvement, convergence, quality parity, or numerical parity.

## Upgrade checklist

- [ ] Update the project pin and H3 dependency constant together.
- [ ] Run `require_minimax_h3_support()`.
- [ ] Rerun the real public-symbol and no-weight component-spec/workflow probes.
- [ ] Run H3 scheduler/runtime/registry/reference tests in the pinned environment.
- [ ] Parse all H3 examples through `Arguments.load_from_yaml`.
- [ ] Run the T2VA output-codec and common SFT/offline-DPO structured-state tests.
- [ ] Rerun the affected H3 cells in the documented real-weight matrix before changing support or
      memory claims.

## Cross-refs

- UP: [`constraints.md` #5](../constraints.md#5-adapter-component-runtime-contract), [`constraints.md` #14](../constraints.md#14-sample-dataclass-hierarchy), [`architecture.md` registered components](../architecture.md#registered-components)
- PEER: [Component Runtime](component_runtime.md), [Structured Trajectory](structured_trajectory.md), [Parity Testing](parity_testing.md), [GPU Validation](gpu_validation.md), [Acceleration](../../../docs/acceleration.md)
