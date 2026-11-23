# Differentiable feedback and component execution

Design checkpoint for [P18–P22](../milestones/v0.5x-v0.6.md).
ART merged in [#377](https://github.com/X-GenGroup/Flow-Factory/pull/377) on the existing
reward-regression path. Differentiable rewards and ReFL merged in
[#381](https://github.com/X-GenGroup/Flow-Factory/pull/381) and
[#382](https://github.com/X-GenGroup/Flow-Factory/pull/382). Flexible component placement and
DRaFT-K establish the baseline. [P22 DRTune](../milestones/v0.5x-v0.6.md#p22)
completes the v0.6 algorithm scope before [NCCL EP](expert-parallelism.md) implementation.

## Separate three decisions

| Decision | Examples | Authoritative owner |
|---|---|---|
| Which parameters may change? | Policy, fake, frozen scorer, EMA snapshot | Role/variant and optimizer contracts |
| Which derivatives are required in this phase? | Parameter gradients, input gradients through frozen weights, no gradients | Algorithm execution contract |
| Where do weights and work live? | Replicated, FSDP2 on an existing group, CPU offload | Component runtime and loading/preparation |

A frozen reward or VAE can still require input gradients. `eval()` controls layer behavior;
`requires_grad_(False)` freezes parameters; `no_grad()` disables the required input path too.
A replicated frozen component is not necessarily a DDP wrapper. Do not encode these distinctions
as a single `backend` string or introduce separate trainers for each distribution mode.

## Execution and feedback

The ordinary reward buffer detaches scores for asynchronous paths such as ART and GRPO.
ReFL uses the merged differentiable reward capability and a synchronous optimization-stage
feedback path. DRaFT can reuse that path to score its decoded terminal state and backpropagate
through the sampler.

A scorer's differentiable path accepts tensors and exposes its direction, reduction and precision.
Keep resize/crop/normalization in Torch; PIL, NumPy, remote calls and accidental detach are not
admitted. Reward parameters stay out of the optimizer. Compare differentiable preprocessing with
the scoring reference and quantify intentional interpolation differences instead of silently
claiming identical scores. Frame selection and temporal aggregation belong to the reward contract.

Start with direct end-to-end autograd. A staged VJP may later reduce memory: score the decoded
output, obtain its derivative, then replay decoder/policy under the same weights, randomness and
precision. This needs an explicit lifetime and identity contract, not an asynchronous buffer hack.

## Reuse the merged replay boundary

The [shared sampling contract](../../guidance/training/sampling.md) from #377 already owns
`scheduler.replay_steps`, transition retention and trajectory index mapping. Configuration
admission runs after scheduler/sampler resolution even with explicit gradient accumulation.
Reuse these owners instead of introducing algorithm-specific copies of replay selection.

Retention selects which rollout states or transitions are saved. The trainer owns the derivative
boundary. DRaFT-K acquires initial noise, the native schedule and conditioning, then computes
the suffix entry state using the same current policy that differentiates the suffix. Each
transition consumes a newly computed state. DRTune can reuse this acquisition path and owns
its training mask and early stop. Terminal-state re-noising for NFT/AWM/DGPO/CRD continues to
use the existing `TimeSampler` controls.

## Distribution ownership

Extend the existing component runtime, shared-root registry, load coordinator and preparation
policy. User configuration should choose component placement, group and precision; the algorithm
supplies phase derivative requirements. Any concrete field names are decided in P20a after P19's
working replicated path, rather than accepted as a second generic framework now.

Replicated remains the auxiliary default. FSDP2 is explicit for admitted components; ZeRO-2/DDP
remain policy backend choices. Frozen sharded weights still need gathering for input backward.
Calls such as VAE `decode` must enter a prepared forward boundary, including recomputation.
An arbitrary method call on a wrapper is not proof that FSDP hooks ran.

Groups reference existing logical topology; a reward shard group does not automatically multiply
the world size or inherit the policy's TP/CP layout. Specify who owns each logical reward, how a
policy's full output is exposed and how gradients return without duplicate normalization. Model
peers must agree on the schedule. Sharded rewards cannot independently launch collectives from
arrival-ordered worker threads. Keep current asynchronous overlap for replicated ordinary rewards;
new overlap paths need separate admission.

Preprocessing uses the same placement machinery with no gradients. Cache hits, misses, empty ranks
and partial batches require a coordinated plan so every member enters collectives in the same
order. Cache identity includes transformation and precision semantics, not transient rank numbers.
Avoid duplicate preparation for shared VAE/text encoder roots across acquisition and optimization.

## Algorithm boundaries

ART needs complete endpoint/reward groups but no reward gradient. Restricting the sampler to
rank-local groups avoids new cross-rank trajectory gathers. Existing source/group/member IDs,
packed replay order and shared optimizer geometry remain authoritative. The official/default
recipe uses FP32 trajectory storage, one inner epoch and two updates; these are recipe settings,
not algorithm admission restrictions. Ordinary training may configure updates, inner epochs and
shared reward reduction. Overlap retains the common one-inner-epoch and group-local-statistics
requirements, preserving complete-group readiness and frozen rollout anchors.

ReFL replays one selected late transition and decodes its projected clean estimate. The landed
[ReFL guide](../../guidance/training/refl.md) describes this flow-matching adaptation.

[DRaFT-K](https://arxiv.org/html/2309.17400#S4) evaluates the final generated state. Run the first
`T - K` transitions without gradients, then differentiate through the last `K` contiguous
transitions. The state within that suffix retains gradients through both the denoiser input and
the solver update. With `K = T`, the same loop represents full-chain DRaFT. Start with Euler ODE
as the flow-matching adaptation and use the model's normal conditioning, guidance, state masks,
and per-step precision. The paper's image experiments used DDIM.

DRaFT-K uses the typed state forward, native Euler update and differentiable decoder. It
requests a fresh ODE update at every transition, so the terminal reward reaches the
earliest differentiated policy call. Without activation checkpointing, autograd can retain
a separate set of intermediates for each of the `K` denoiser calls even though they share
weights. Checkpointing trades recomputation for lower activation residency. It leaves the
gradient estimator intact. The SD3.5 recipe enables it through the existing model/backend
policy, which has a single owner for checkpointing each block.

DRaFT-LV is a follow-up estimator. It draws additional noises from the forward process at the
last-step noise level of a generated endpoint and combines the resulting reward gradients.
Its noise, reward reduction and gradient weighting need their own reference checks.

### DRTune

DRTune trains denoiser predictions at spaced positions with a random offset. The sampler's direct
state term stays live when the denoiser receives a detached input.
[Wu et al. (ECCV 2024), §3.2 and Algorithm 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10922.pdf)

For the Euler ODE adaptation, let `delta_sigma = sigma_next - sigma`.

```text
model_input = stop_gradient(z)
velocity = denoiser(model_input, sigma, condition)
z_next = z + delta_sigma * velocity
```

Selected calls retain the parameter derivative of `velocity`. Other calls still advance the same
forward trajectory and can compute their velocity under `no_grad`. The scheduler update then
runs with the live `z`. In the FP32 Euler reference, the inter-state derivative through the direct
term is the identity, so a reward gradient can reach an early selected call through many
unselected steps. DRTune removes the denoiser-input Jacobian by design. DRaFT-K retains that
Jacobian on the contiguous suffix. Both methods score the same native endpoint at fixed inputs
and weights in the full-endpoint variant. Activation checkpointing is independent of this
choice. The selected DRTune calls can still retain parameter-backward activations.

| Algorithm | Scored state | Parameter-gradient path |
| --- | --- | --- |
| ReFL | Clean estimate from a saved late state | One replayed prediction |
| DRaFT-K | Native final sample | Last K contiguous predictions with full state derivatives |
| DRTune | Native final sample in P22a, early clean estimate in P22b | K spaced predictions with a live solver state and detached prediction input |

**Reuse.** Keep the existing `generation + differentiable_reward` contract, source weights,
frozen input-gradient decoding and reward, accumulation, logging, reducer composition and
checkpoint ownership. Use DRaFT-K's SD3.5 initial-noise, conditioning and schedule acquisition.
Use ReFL's `project_velocity_to_clean_state` boundary for the early-stop estimate. Keep native
CFG, timestep/sigma, state casting and solver math in the adapter and scheduler.

**New ownership.** Give DRTune a direct `BaseTrainer` implementation and typed arguments.
The trainer chooses which calls keep parameter gradients and where to stop. Extract only the
needed model-local prediction seam so the trainer can supply a detached denoiser input while
the native scheduler consumes the live state. The current SD3.5 `forward_state` couples those
two inputs through one state. Detaching the entire input to that call would also cut the
solver gradient. Keep DRaFT-K's suffix helper and ReFL's rollout/replay loop separate. Start
with direct autograd and defer a staged VJP until equivalent outputs and gradients and a
measurable memory benefit are established.

**Selection and projection.** Define the mask over the `N` visited prediction positions,
with `1 <= K <= N`. Use stride `floor(N / K)` and a seeded zero-based offset in
`[0, N - 1 - (K - 1) * stride]`. This yields exactly K distinct positions for every legal N
and K, including divisible N/K and K=N. Algorithm 1 prints a start interval that is empty when
T is divisible by K and can select steps outside an early-stopped trace. For the Euler adaptation,
draw the legal stop first, then select within the executed positions. Count the terminal projection
prediction among them if selected. An unselected stop prediction is detached while the projection
still differentiates its direct state term. Bind the selection and stop rules to resume identity
and keep call order consistent across prepared ranks.

## State, failures and validation

Reuse existing checkpoint/role ownership. Include reward revision, policy/precision and execution
identity where they affect continuation. Frozen components have no optimizer state. Resume first
at a drained acquisition boundary; no arbitrary in-flight graph serialization is promised.
Validate capability, group and precision conflicts before loading or collectives. Abort and cleanup
must release temporary graphs and restore component placement without publishing a partial update.

Use CPU derivative references and two-GPU component comparisons first. Combine policy TP/SP/CP,
video or multi-role behavior only where a changed boundary reaches them; selected compositions
fit within eight local GPUs. Preserve historical evidence rather than relabeling it as validation
of new differentiable code. Measure memory/throughput separately from correctness.
