# Differentiable feedback and component execution

Design checkpoint for [P18–P21](../milestones/v0.5x-v0.6.md), reviewed 2026-11-11.
ART extends the existing reward-regression path in v0.5.x. ReFL/DRTune and flexible component
placement follow for v0.6, before [NCCL EP](expert-parallelism.md). This is a design, not an
implemented differentiable reward/backend API.

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

The current reward buffer detaches outputs and permits asynchronous scoring. That is appropriate
for ART/GRPO, but it cannot transport the autograd graph needed by ReFL/DRTune. Add an explicit
differentiable feedback capability and optimization-stage boundary, then audit all resolved
execution contracts. Preserve ordinary reward evaluation and evaluation-only scorers.

A scorer's differentiable path accepts tensors and exposes its direction, reduction and precision.
Keep resize/crop/normalization in Torch; PIL, NumPy, remote calls and accidental detach are not
admitted. Reward parameters stay out of the optimizer. Compare differentiable preprocessing with
the scoring reference and quantify intentional interpolation differences instead of silently
claiming identical scores. Frame selection and temporal aggregation belong to the reward contract.

Start with direct end-to-end autograd. A staged VJP may later reduce memory: score the decoded
output, obtain its derivative, then replay decoder/policy under the same weights, randomness and
precision. This needs an explicit lifetime and identity contract, not an asynchronous buffer hack.

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

ReFL first establishes a no-grad sampling prefix followed by a selected differentiable prediction,
clean estimate, decode and reward loss. Record how the flow-matching adaptation differs from the
original diffusion implementation, including its optional pretraining regularizer.

DRTune owns a more specific derivative program. Stopping the denoiser input gradient does not
remove the direct state branch of the scheduler update. Selected steps, early stopping and solver
history must be explicit and tested against an independent unrolled reference. Begin with Euler;
solver-agnostic admission would hide an unproved contract.

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
