# Execution Fix Patterns

**Read when**: Debugging or changing execution behavior.

### Custom Transformers models must implement their declared checkpointing seam
- **Date**: 2026-08-30
- **Symptom**: Every Bagel trainer failed during initialization when full gradient checkpointing
  called `Qwen2ForCausalLM.gradient_checkpointing_enable()` and Transformers reported that the
  architecture was incompatible.
- **Root Cause**: Bagel's custom Qwen2-NaViT model inherited
  `supports_gradient_checkpointing = True` but did not expose a `gradient_checkpointing` state or
  invoke the installed checkpoint function in its active decoder loop.
- **Fix**: The custom Qwen2 model now owns the standard checkpointing flag and routes each pure
  decoder layer through Transformers' installed non-reentrant checkpoint function while training.
  Cache-updating and TaylorSeer paths remain direct to avoid replaying mutations. A backward
  regression proves the layer is recomputed rather than merely accepting the API call.
- **Lesson**: A custom `PreTrainedModel` must pair its capability declaration with both the state
  seam expected by Transformers and an execution-time checkpoint boundary in the forward path.
- **Related Constraint**: #7

### Discrete time sampling must keep the last in-range step
- **Date**: 2026-10-12
- **Symptom**: With `time_sampling_strategy: discrete*` (the AWM, NFT, DGPO, and CRD default), the
  last scheduler step inside `timestep_range` was never trained, and some other steps were drawn
  in every batch.
- **Root Cause**: `TimeSampler._stratified_sample` draws from the half-open range
  `[start, stop)`. The original sampler passed the exclusive `int(N * fraction)`; #113 replaced it
  with the inclusive `valid_indices.max()` without adjusting the half-open consumer.
- **Fix**: `discrete()` passes `max_i + 1` as the stop index, the helper documents its half-open
  contract, and empty draw ranges now raise instead of producing clamped out-of-range indices.
- **Lesson**: When a refactor changes a bound from exclusive to inclusive, audit every consumer of
  that bound; a half-open sampler fed an inclusive maximum fails silently by narrowing coverage.
- **Related Constraint**: N/A
- **Evidence**: `tests/test_noise_schedule.py` discrete coverage tests fail on the previous
  implementation; 4000-draw simulations of the AWM, NFT, DGPO, and CRD SD3.5 defaults drew the
  last in-range step zero times. The official DiffusionNFT trainer samples uniformly from every
  rollout step.

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
