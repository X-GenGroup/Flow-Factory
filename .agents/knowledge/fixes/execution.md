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

### DMD2 fake TTUR steps need fresh rollouts
- **Date**: 2026-11-06
- **Symptom**: DMD2 drew one rollout set per generator step and ran all `update_frequency`
  fake steps on it before the generator step, so the fake score saw `update_frequency` times
  fewer distinct generated samples than the DMD2 reference and was repeatedly fit on the exact
  samples the generator step then queried it at.
- **Root Cause**: The shared distillation step (`run_distillation_training_step`) samples one
  set per cycle and hands it to `optimize()`, so every TTUR repeat reused that set. The DMD2
  reference loop instead generates new samples on every iteration.
- **Fix**: `DMD2Trainer._run_training_step` draws one fresh GAS-sized set per fake step and
  runs the generator step on the last set, after the fake score has fit it. Fake stays first:
  the generator gradient is only exact when the fake score tracks the current generator, so
  the reference loop's generator-before-fake order is deliberately not followed.
  `rollout_sets_per_cycle()` multiplies the exact-resume cursor, and the runtime identity
  records the schedule so shared-rollout checkpoints fail fast instead of resuming at the
  wrong batch.
- **Lesson**: Rollout reuse across TTUR repeats is an algorithm choice, not a framework
  default. When a trainer consumes several rollout sets per cycle, the exact-resume cursor
  must count every set, and the schedule must be part of the resume identity.
- **Related Constraint**: #18
- **Evidence**: `tests/trainers/test_dmd2.py` covers fresh-set ownership per fake step,
  fake-before-generator order, the resume identity, and the cursor multiplier;
  `tests/distributed/test_multirole_backend.py` runs real DMD2 cycles on two-GPU DDP (LoRA and
  full) and FSDP2 (full).

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
