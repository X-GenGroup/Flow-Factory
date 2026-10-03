# Rewards Fix Patterns

**Read when**: Debugging or changing rewards behavior.

### Distributed reward gathering must preserve reconstruction invariants
- **Date**: 2026-08-30
- **Symptom**: Two-rank H3 Ref2VA GRPO failed before reward execution because `MiniMaxH3Ref2VASample` was reconstructed with `reference_manifest=None`.
- **Root Cause**: The distributed group-reward path gathered only reward-consumed fields, although `gather_samples` reconstructs the concrete sample class and that class can require additional state.
- **Fix**: `BaseSample` now declares an empty `reconstruction_required_fields` contract,
  `OrderedReferenceConditionSample` adds `reference_manifest`, and `gather_samples` automatically
  unions that contract at the concrete reconstruction boundary. Reward callers therefore transport
  constructor state without forwarding it to the reward call or duplicating gather policy.
- **Lesson**: Communication payload requirements and reward-call requirements are distinct contracts; partial gathers must preserve constructor invariants even for fields that downstream computation does not consume.
- **Related Constraint**: N/A

### Exact resume must lock future execution and preserve acquisition boundaries
- **Date**: 2026-08-28
- **Symptom**: An exact checkpoint could pass preflight after objective, seed, scheduler, ordered training-data, or replayed evaluation semantics changed; a remote-rank runtime-child failure could let peers return from resume; an online resume retried its immutable source `checkpoint-N`; offline save-before-eval captured RNG that did not match the next uninterrupted epoch; and MPS could publish an exact checkpoint that its loader would always reject.
- **Root Cause**: Resume identity stopped at physical model/optimizer/backend structure and treated RNG-consuming evaluation as operational, runtime-child commit was outside the synchronized distributed phase, one generic checkpoint/evaluation order ignored the different replay boundaries of generated versus finite-dataset acquisition, and save preflight did not reject a device RNG unsupported by Accelerate.
- **Fix**: Runtime identity now includes trainer-extensible execution and rank-free data-contract digests derived from resolved objective/forward settings, realized training loader provenance/order/geometry, and the ordered realized evaluation path (cadence, arguments, dataset overrides, rewards, and prepared loaders). Runtime-child commit and attachment use a synchronized all-rank error phase. The first online boundary skips a save only when its resolved real path equals the exact-resume source, while still evaluating; offline boundaries evaluate before saving so exact checkpoints capture post-evaluation RNG, model-only saves retain the same observable order without claiming RNG restoration, and MPS exact save fails before adapter or filesystem mutation.
- **Lesson**: Exact resume compatibility covers every computation and ordered data stream that can affect future state, including evaluation replay, not only the state container. Every rank-local resume mutation needs a synchronized failure boundary, checkpoint placement must match whether acquisition resumes before a rollout or after a completed data epoch, and save must not publish a state the matching load path cannot restore.
- **Related Constraint**: #18

### Runtime identity excludes transport-only global source IDs
- **Date**: 2026-08-28
- **Symptom**: Inserting or reordering an eval-only dataset renumbered training sources and caused exact-resume rejection even though the ordered training data and reward mathematics were unchanged.
- **Root Cause**: Data identity hashed the full global name-to-ID registry and offline numeric `source_id`, which are transport metadata assigned across both train and eval entries.
- **Fix**: `trainers/common/runtime_identity.py` now locks ordered realized training source names and loader schemas while excluding global numeric IDs. Structural and real-`Arguments` regressions verify eval-only changes preserve both execution and data digests, while training-source order/count changes remain incompatible.
- **Lesson**: Compatibility identities should include semantic names and order, not remappable integer handles whose only purpose is runtime transport.
- **Related Constraint**: #18

### No-feedback algorithms must keep monitoring rewards eval-only
- **Date**: 2026-08-31
- **Symptom**: Every shipped DiffusionOPD example failed argument loading even though its reward
  comments described monitoring rather than a training signal.
- **Root Cause**: The examples duplicated evaluation rewards under top-level `rewards`, but the
  algorithm declares a no-feedback execution contract that rejects every training reward.
- **Fix**: The DiffusionOPD examples now keep monitoring models only under `eval_rewards`; the
  algorithm guide states that evaluation scores never enter the distillation loss, and the shared
  distillation contract tests cover DiffusionOPD alongside DMD2 and TDM.
- **Lesson**: Monitoring intent does not change configuration semantics. A no-feedback algorithm
  must express quality metrics through the evaluation-only reward surface.
- **Related Constraint**: #7

### Distributed group identities must never share a floating-point payload with rewards
- **Date**: 2026-09-24
- **Symptom**: Cross-rank reward normalization could merge or misroute groups whose prompt hash
  exceeded the exact integer range of float32, and equal hashes from different datasets could be
  treated as the same comparison group.
- **Root Cause**: Reward values, `unique_id`, and `source_id` were packed into one float32 gather;
  grouping also treated `unique_id` alone as globally authoritative.
- **Fix**: Canonicalize group identity as the int64 pair `(source_id, unique_id)`, transport it in a
  separate integer collective, and reuse one acquisition-level dense mapping throughout streamed
  work units. Reward grouping, advantages, DPO pairing, DGPO noise, and TDM-R1 group loss all use
  the same key.
- **Lesson**: Communication coalescing cannot erase type semantics. Pack fields only when their
  exact representation and identity scope match; dataset provenance is part of a comparison-group
  key, not logging metadata.
- **Related Constraint**: #9

### Reward request batches and optimizer work units have independent ownership
- **Date**: 2026-09-24
- **Symptom**: Tying pointwise reward submissions to optimizer tile boundaries fragmented remote
  batches and reduced server utilization, even though a returned row can be routed to its stable
  sample index independently of when that sample becomes optimizable.
- **Root Cause**: The reward buffer was configured with `samples_per_tile` and flushed a short
  request whenever it reached an optimizer boundary.
- **Fix**: Let each reward model own its batch size and executor lane. Requests may cross optimizer
  work-unit boundaries; their futures populate stable acquisition rows, and every work unit waits
  only for the rows it contains. Optimizer geometry remains responsible for complete groups and
  gradient-accumulation closure.
- **Lesson**: Do not make one pipeline stage's batching policy an invariant of another stage.
  Connect stages through stable row identity and readiness dependencies instead.
- **Related Constraint**: #9

### Distributed readiness polling must back off as one synchronized schedule
- **Date**: 2026-09-24
- **Symptom**: Slow remote rewards caused tens of thousands of readiness all-reduces in one
  acquisition, making coordination itself a measurable part of the critical path.
- **Root Cause**: Every rank polled at one fixed short interval even after repeated globally empty
  results, so remote-service latency was converted into high-frequency collective traffic.
- **Fix**: Treat the configured interval as the initial delay, exponentially back off after each
  unsuccessful global poll to a bounded cap, and reset the delay whenever a work unit advances.
  Because every rank consumes the same readiness result, the backoff schedule stays symmetric.
- **Lesson**: A distributed poll is communication, not a free local status check. Adapt its cadence
  from globally observed progress while preserving identical collective order on every rank.
- **Related Constraint**: #9c, #18

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
