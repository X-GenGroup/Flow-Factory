# Online DPO context parallelism

Online `dpo` declares the same full-output gradient convention as offline DPO. It keeps its
existing generation + runtime-reward acquisition: complete source-aware groups become one
best/worst pair each, then both policy arms and the frozen reference use identical forward-process
noise and training times. The objective remains the shared `dpo_objective`; CP normalization
belongs to the common backward owner.

Reward/optimization overlap is **rank-local only**. Use `sampler_type: rank_local`,
`global_std: false`, one inner epoch and no sample shuffling. A complete two-sample group supplies
one optimizer example, not two: with batch size one and two training times, one pair per tile
requires GAS 2. Existing samplewise overlap recipes still require GAS 4 for two samples and two
times. This distinction is independently asserted by the native validation fixture.

Outside overlap, cross-rank pair gathering uses the sampler's logical-DP communication scope.
Pairs are assigned round-robin; shorter shards repeat local pairs to keep update counts equal.
Metrics count the original, unpadded pairs. CP/TP peers do not add logical examples. Equal group
IDs from different sources stay separate, ties retain first-occurrence ordering, singleton groups
are skipped and no valid pairs fail before optimization.

The representative GPU recipe is native tiny Flux1 LoRA with FSDP2/BF16, DP2 × CP2 on four local
GPUs, ODE acquisition, two training times and ordered asynchronous feedback. It checks actual
pair identities, independent preference derivatives, CP/GAS scaling, frozen-reference queries,
pending rewards during optimizer work, Adam updates, EMA evaluation, drained save, exact loaded
state and fresh-process continuation. The raw records and final source-bound report are required
for qualification; a passing CPU test alone does not claim native support.

Validation entry points:

- `online-dpo-cp-contracts`: objective/gradient, pair/source and tile-accounting corruption controls.
- `online-dpo-logical-pairs`: real four-rank CPU/Gloo DP2 × CP2 sample gathering and pair statistics.
- `flux-online-dpo-overlap-native`: the single focused four-GPU lifecycle witness.

The existing admitted model layouts and backend restrictions remain authoritative. This leaf does
not add Flux TP+CP, cross-rank online-DPO overlap, a new checkpoint boundary, or a new training loop.
