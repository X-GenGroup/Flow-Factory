# TDM-R1 context parallelism

TDM-R1 declares its own full-output CP training capability. Its parent TDM declaration does not
implicitly admit subclasses. Shared preparation, native model attention communication, gathered
outputs, CP loss-gradient normalization, role optimizers and checkpoint state remain authoritative.
This extends the algorithm to already admitted model layouts; it does not admit another model.

Use the [CP configuration requirements](../context-parallel.md) and the selected model's guide.
The composed native witness is full-weight Wan T2V with FSDP2/BF16, DP2 × TP2/SP × CP2 on eight
local GPUs. Other backends retain their existing CP1 support; this declaration does not enable CP
for DDP, ZeRO-2 or an unqualified model layout.

## Objective and role ownership

One acquisition stores every deterministic trajectory boundary. With K denoising steps and GAS
boundary losses per role update, the loader contributes GAS/K rollout microbatches. K must divide
GAS. The native recipe uses K=2, GAS=4 and two rollout microbatches per logical data replica.

The update sequence is `fake × R → surrogate → generator`. All roles share one prepared model
bundle and one optimizer root. The frozen pretrained reference is a parameter snapshot, and the
slow surrogate is a named variant snapshot. Each role's inactive parameters and optimizer moments
must remain unchanged during another role's update. Reference scoring explicitly selects the
base generator route before installing its frozen parameters, including inside live and slow
surrogate scopes. A parameter snapshot alone does not select a component route.

The surrogate sees the same perturbed stored boundary, target velocity and scheduler coordinates
as the frozen reference and slow surrogate. Its preference keeps the legacy detached sigmoid group
weight. Trust and runaway clipping suppress only the preference derivative; the reference anchor
continues to constrain the original live DSM values. Time weighting retains the current formula
`(boundary_index + 1) / K`; stored boundary indices are 1..K, so K=2 weights are 1 and 1.5.

The generator replays one live transition, retaining strict same-policy replay checks. Its objective
is the TDM distribution anchor plus `tdm_weight * guidance_reward + (1 - tdm_weight) * surrogate_reward`.
The guidance direction queries the frozen reference explicitly at scale 1, independent of student
rollout guidance, and uses the actual reference-query scale in its denominator. Explicit batch
conditioning retains precedence for the reference query. The scale-1 query uses the adapter's
canonical guidance mapping, preserving model-specific argument names. Missing or inconsistent
nontrivial mapped scales fail instead of guessing. No auxiliary generator preference scoring is
added, and DMD2's accepted feedback-error exception does not apply.

Rank-local groups need no group collective. Equal-share groups check shape and identity extrema
within logical DP before summing indexed group statistics. Both extrema are compared so every
logical replica rejects a mismatch, including the replica holding the maximum. Packed groups and
overlap retain the existing coordinator/cached mapping. TP/CP peers do not count as extra samples.

## Reward overlap and continuation

Ordered overlap performs the complete reward-independent fake phases first. Each ready tile then
contributes its K surrogate backward terms to one acquisition-wide accumulation window. The last
boundary term performs the single surrogate optimizer step. Finalization closes the phase, updates
the slow copy and runs the generator; it does not create a second surrogate step.

The slow copy uses `min(max_decay, min_decay + 0.001 * public_step)` before the generator advances
that public step. Evaluation and exact state saves happen only after the acquisition drains.
Fresh-process fixed-topology continuation restores roles, Adam moments, slow snapshot, EMA, query
provenance identity and RNG. Asynchronous thread timing is not a checkpoint identity.

A late reward failure can follow partial surrogate accumulation. The stream abort closes role and
variant scopes and prevents the unfinished surrogate, generator and slow-snapshot updates. Earlier
completed fake updates are not rolled back. Diagnostic partial gradients remain; the existing phase
entry guard rejects their reuse. The outer cycle remains incomplete and exact checkpoint save is
rejected. This is a terminal failure contract, not an arbitrary in-flight resume promise.

## Focused evidence

- `tdm-r1-objective-reference`: actual masked multi-component surrogate and three-direction generator
  objectives against independent scalar/derivative references, clipped and unclipped branches,
  rank-local/equal-share/packed logical operands, public FP32 Huber cancellation, CP/GAS controls,
  nonunit and mapped/batch guidance, and real role/stream late-future cleanup.
- `tdm-r1-logical-identity`: real four-rank Gloo ordinary DP and DP2/CP2 with deliberately different
  model peers, value/shape mismatch rejection on every logical replica, caller immutability and DP1
  no-communication controls.
- `wan-tdm-r1-cp-lifecycle`: two actual native acquisitions and a fresh-process resumed acquisition,
  strict trajectory replay, real role and snapshot queries, independent raw objective derivatives,
  pending reward work during fake and early surrogate work, clipping/Adam/EMA, evaluation and exact
  loaded state followed by continued updates.

The owning PR records final source-bound results. These selected witnesses establish algorithm
ownership on supported layouts; they do not claim all model/backend combinations, pretrained-model
convergence, hidden reward latency, or an in-flight/topology-changing resume contract.
