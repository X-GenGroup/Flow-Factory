# CRD context parallelism

CRD declares the existing full-output context-parallel training contract. Its
representative native case is Qwen Image full training with FSDP2/BF16 on DP2 × CP2,
TP1, ordered reward overlap and `global_batch` sampling. Other CRD/model endpoints
require their own selected evidence; this case does not qualify Qwen TP+CP or
in-flight/topology-changing checkpoint recovery.

The fixture uses two training times, two acquisition cycles, positive-temperature
MSE centering, adaptive implicit rewards and nonzero reward-adaptive reference KL.
Rollout guidance and reference guidance differ. `_crd_old` and `_crd_sampling`
follow distinct nontrivial schedules. The existing overlap runner exercises
pending rewards, optimizer/adapter EMA checks, drained evaluation and saving,
then exact loading and ordered continuation in a fresh process.

The leaf observer proves raw-velocity implicit rewards and analytic local velocity
derivatives, including actual intermediate casts, detached centers and detached
adaptive normalizers. It retains every pass-one source/sample/time, conditioning,
noise, noised state and old prediction by value, verifies acquisition-wide pass one
precedes optimizer work, and checks exactly-once consumption and both snapshot
recurrences. Frozen reference and inactive Qwen parameters remain unchanged.

CPU controls cover MSE/BCE, uniform/soft/hard weighting, empty pools, adaptive/plain
implicit rewards, ordinary/reward-adaptive KL, masked unequal component sizes,
`use_old_for_loss=False`, multiple ordinary inner epochs and failure cleanup.
Four real Gloo ranks use deliberately unequal CP peers to prove logical-DP gather
cardinality/order, homogeneous packing and heterogeneous dtype preservation.
Ordinary DP retains the Accelerator gather path and DP1 never communicates.

Hard weighting keeps the current `[0, 1]` normalized-advantage input: its negative
pool falls back to uniform weights. The native witness uses positive temperature;
it does not claim native hard-branch qualification. All acquisition batches and
targets remain device resident between passes.

Selected cases are `crd-objective-contracts`, `crd-logical-gather-contracts` and
`qwen-crd-overlap-native` in [the catalog](../../../../config/validation/catalog.yaml).
An execution result qualifies only the immutable source and evidence identities
accepted by [the selected-plan validator](../../../development/validation/validation.md).
