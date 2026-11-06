# Context parallelism

Context parallelism (CP) splits one video's token sequence across ranks. The first experimental
training path uses native Diffusers Ulysses with Wan T2V and FSDP2. Configure it explicitly:

```yaml
parallelism:
  cp_size: 2
mixed_precision: bf16
```

On a four-rank launch, the omitted `dp_shard_size` resolves to two logical data replicas.
Use the [shared configuration rules](parallelism.md#configuration-and-runtime) to infer from
the launched world, or set an explicit shard degree to reject a mismatched launch.
CP peers receive the same record,
conditions, noise and timestep; each logical replica receives a different sample. Omitting
`cp_size` preserves CP=1. See the [SFT example](../examples/sft/lora/wan2_t2v/wan21_cp.yaml).

## Experimental capability

The model layout capabilities retain the identifiers `wan-sft-ulysses-bf16-v1`
and `wan-sft-tp-ulysses-bf16-fp32-sdpa-v1`. A separate, explicit declaration on each
concrete trainer admits SFT, GRPO and DMD2; subclasses do not inherit admission.
These paths share the following constraints:

- `wan2_t2v`, one `WanTransformer3DModel`, dense fixed-size video, scalar sample timesteps,
  equal input/output channels, text conditioning without added image KV projections.
- SFT, GRPO or DMD2, full or LoRA tuning, one `transformer` target, one example per device,
  BF16 mixed precision and AdamW. Backend activation checkpointing may be enabled.
  SFT/DMD2 use explicit gradient accumulation; GRPO may derive it from logical sampling geometry.
- FSDP2 on CUDA/NCCL, replicated-DP=1 and sharded-DP at least two. CP-only uses TP=1;
  the composed capability uses TP>=2 with Wan's training-aware TP plan and optional SP.
  Attention heads must divide by TP × CP. Video tokens must divide by CP, and additionally
  by TP when SP is enabled; latent dimensions must divide by patches.
- One shared `attention_backend` accelerator with `params: {backend: native}`. No other shared
  or rollout accelerators; reward/optimization overlap remains gated. Evaluation uses the shared
  generation path with logical-DP ownership.
- Model-only weight export or fixed-topology state continuation at a drained acquisition or
  finite-data-epoch boundary. EMA, frozen references and DMD2 named variant snapshots retain
  their existing owners and checkpoint formats.
- PyTorch 2.10.x, Accelerate 1.15.x and Diffusers 0.41.x release lines. Development/prerelease
  versions are rejected. The tested tuple uses PyTorch 2.10.0+cu129, Accelerate 1.15.0,
  Diffusers 0.41.0 and PEFT 0.21.2.

Wan2.2's expanded per-token timestep layout and two-transformer schedules are not included.
Neither are I2V, masks, prefix caches, uneven tokens, native Ring, other algorithms,
in-flight recovery, or topology-changing exact continuation. A public capability is a bounded experimental path, not a guarantee
for every checkpoint in the model family.

## Logical reward subgroups

`data.sampler_type: subgroup_tile` and `data.sampler_subgroup_size: R` partition logical DP
replicas, independently of TP/CP degrees. `R` must divide the logical DP size. For DP4 × CP2,
`R=2` maps logical subgroups `{0,1}` and `{2,3}` to physical groups `{0,2}`, `{4,6}` at CP coordinate
zero, and `{1,3}`, `{5,7}` at CP coordinate one. TP adds another fixed peer coordinate.

Every physical rank creates these groups in the same order. Group-relative feedback uses only
its subgroup; optimizer synchronization retains the backend's original scope. `R=1` performs no
group collective, and `R=DP` reuses the existing data group. No additional sampler or model-peer
process group is introduced. This geometry support does not yet enable CP reward/optimization
overlap; that composition is qualified separately.

The required subgroup evidence combines the following cases, each on the reviewed source. CUDA
cases run sequentially and need at most eight devices; reducing DP alone would lose the
multiple-subgroup proof.

| Evidence | Topology and scope |
| --- | --- |
| Native full/LoRA GRPO with two R2 subgroups | TP1 × CP2 × DP4, compared with CP1 at fixed DP4 |
| Native full/LoRA GRPO with all three axes | TP2 × CP2 × DP2, compared with CP1 at fixed TP2/DP2; R2 equals the full DP group |
| Actual NCCL subgroup isolation | TP2 × CP1 × DP4 and TP1 × CP2 × DP4; separate R2 groups, R1 bypass and R4 reuse |
| Complete logical membership and reward ownership | Gloo TP2 × CP2 × DP4, plus each independent model-peer axis and replicated/sharded DP |

Both native cases retain exact same-policy velocity/transition-mean/log-prob replay, independent
Adam/clipping and elementwise update checks, varying feedback, EMA and fresh-process continuation.
The NCCL probes check CUDA sample transport, peer-specific payloads, group creation order and
isolated reductions, while full-DP reductions still span every logical replica. They make no model
update claim. The 16-process Gloo case checks the full three-axis DP4 membership and single-owner
groupwise rewards; it makes no NCCL claim.

This decomposition covers three-axis native training and multiple nontrivial subgroup isolation
through separate required cases. It does not qualify native TP2 × CP2 × DP4 training on 16 GPUs,
its capacity or throughput, or cross-node execution. Those remain additional, unqualified layouts
until their own native evidence is selected and run.

## Evaluation

SFT, GRPO and DMD2 can evaluate with CP and TP+CP using the existing `data.datasets[*].eval`
entries and `eval.eval_freq`. Offline training still evaluates by generation. Dataset overrides
for resolution, guidance and inference steps apply to preprocessing and generation; `eval.seed`
uses the training seed only when unset (zero is a valid explicit seed).

The [GRPO TP+CP example](../examples/grpo/lora/wan21/t2v_tp_cp.yaml) keeps evaluation disabled
and requires no evaluation dataset by default. To opt in, uncomment its `evaluation` source
and sampling settings, create `dataset/evaluation/test.jsonl`, and set `eval.eval_freq: 1`.
Evaluation reuses the configured reward models unless `eval_rewards` is supplied separately.

Accelerate shards sequential evaluation batches across logical DP replicas. TP/CP peers see
identical inputs, including padded tail batches needed for symmetric FSDP collectives. The trainer
removes padded outputs before reward submission or media logging, so each real row is scored once
across its model peers. A replica with no real rows contributes zero count. Configured reward
columns remain present on empty async buffers, allowing every DP rank to join the same summary.

Only five FP64 values per reward and logical replica are gathered: count, mean, centered squared
sum, minimum and maximum. Fixed-rank Welford merging produces global population statistics without
collecting reward vectors or generated media. `reward_<name>_count` records the real dataset size;
empty datasets have undefined mean/std/min/max. Existing main-rank media logging remains local.
An entirely empty prompt source can use an empty `test.txt`. Its preprocessing cache preserves
the zero-row input schema and never calls an encoder; no output columns are invented. This differs
from an empty rank partition of a nonempty dataset, which uses the processed schema inferred
from the complete source before sharding.

The shared evaluation scope installs EMA and restores live parameters even on exceptions. Each
dataset reward buffer is cleared on entry and exit. Unsupported custom sharding (random, split or
uneven batches) is rejected before that dataset's generation. Active evaluation now records
`sample_ownership: logical_dp_unpadded_v1` in exact-state identity: older checkpoints with replayed
evaluation are rejected because padding and seed changes alter execution. Disabled or absent
evaluation keeps its previous identity. Arbitrary in-flight recovery remains outside this scope.

## TP/SP composition

The [eight-rank example](../examples/sft/lora/wan2_t2v/wan21_tp_cp.yaml) declares
`tp_size: 2`, `cp_size: 2` and `sequence_parallel: true`; `dp_shard_size` is inferred as two
from its eight-process launch. Those ranks represent two logical data replicas. TP peers own different heads/weights;
CP peers own different token intervals. Disabling SP preserves the TP+CP composition while
keeping the residual token stream replicated across TP peers.

Both CP-only and TP+CP require `model.compute_precision_policy: cross_attention_fp32_v1`.
This [model-declared operation policy](compute_precision.md) keeps cross-attention SDPA in
FP32 with BF16 boundaries. It is applied equally to CP-off and CP-on comparisons, including
CP-only. Precision and its operator cost are explicit. Native BF16 CP-only SFT LoRA failed a
retained four-update fixture at step two (3.22098% update error against a 3.125% cap). The parent
and evaluation candidate reproduced identical tensors. On those exact inputs, this existing
policy passed every original bound (maximum update error 1.66779%); no tolerance changed.
GRPO and DMD2 already used this policy in their CP qualification. The adapter declares the
requirement once, and shared preflight checks it before loading any model. Older CP exact
states have a different capability identity and must start a new run from model-only weights.

The Diffusers mesh retains an independent TP dimension and must reproduce the existing FF
TP and CP peer lists exactly. At the first Wan block, CP splits tokens before SP scatters them
across TP peers. At `proj_out`, SP gathers before CP restores the full canonical sequence.
Diffusers wraps `forward`, whereas TP/SP uses PyTorch module hooks; naïvely stacking them
reverses those boundaries and misaligns tokens with rotary positions. The preparation owner
moves only the existing native split/gather callbacks to the outer module-hook boundaries,
retains their handles in the prepared contract and rejects removed/replaced hooks. Native
Diffusers attention, rotary slicing and gather autograd remain authoritative.

The backward scaling remains CP, not TP × CP: FF's TP styles already reduce their partial
activation/parameter gradients. The strict FP32 scale control and complete SFT comparisons
verify this with SP off/on and backend recomputation.

## Preparation and reduction ownership

`ModelPreparation` runs after all model variants exist. FSDP2 replicated loading synchronizes
initial parameters and buffers before sharding, including randomly initialized LoRA weights.
The existing attention accelerator selects native attention exactly once on every distinct
transformer route, before Diffusers installs its Ulysses hooks. Full DMD2 has separate generator
and fake roots; LoRA variants share one physical root and install its hooks once. The optional
SP boundary bridge is installed after native hooks. All hooks precede optimizer construction and FSDP2
preparation. Accelerate builds the mesh and owns FSDP2; its CP preparation bridge only accepts
one FF bundle carrying the installed-hook contract. It cannot independently replace the hooks.

The adapter returns the full denoiser output in canonical order. Diffusers' output gather sends
only each rank's local output slice through backward. FSDP2 averages over DP × CP, so admitted trainers scale
the backward loss by the CP degree to recover the mean over logical samples. Reported loss
remains unscaled. SFT, GRPO and DMD2 explicitly declare this full-output objective contract;
token-local, masked, parameter-only penalties and other algorithms must derive their reduction
contract independently. DMD2 applies the factor at the role-phase backward boundary, preserving
its fake/generator cadence and the normalization of each objective.

## Online feedback and continuation

TP × CP peers describe one logical sample. Reward processing splits pointwise requests over
these peers, keeps groupwise requests indivisible, and combines their results through the existing
TP and CP groups. Non-repeatable and asynchronous rewards are scored once per logical request;
different data replicas keep their distinct samples. Logical group statistics and sampling RNG
do not depend on the number of physical model ranks.

Set `log.save_model_only: false` to save optimizer, scheduler, progress, RNG and auxiliary state.
Resume with `model.resume_type: state` and the checkpoint directory, retaining the original
physical topology, precision policy and trainer capability. Checkpoint preflight checks those
identities and rank layouts before restoring state. Exported weights can initialize a new run
on another supported layout; this does not continue its old optimizer/RNG trajectory.

Trainer checkpoints begin after the first real optimizer update for every backend and both
checkpoint kinds. Periodic and direct saves at step zero are skipped before serialization or
filesystem mutation; standalone adapter weight exports remain separate. See
[parameter state](parameter_state.md) for the shared policy. Online saves occur after reward
work and all role updates are drained; arbitrary in-flight continuation is excluded.

## GRPO rollout/replay consistency

Same-policy replay must preserve velocity, transition mean and log probability. This is a
separate contract from comparing two BF16 training trajectories with different CP degrees.
Rounding in local parameter gradients, accumulation and FSDP reduction can change subsequent
weights across those trajectories without breaking rollout/replay within either trajectory.

The supported Ulysses forward exchanges sequence/head partitions using all-to-all and restores
the output with rank-ordered all-gather. It introduces no cross-CP floating-point summation in
the forward path. TP normalization and row projections do reduce values, using the same
prepared modules and peer groups in rollout and replay. The native path keeps its communication
algorithm selection; no global deterministic mode or forced NCCL Ring/Simple policy is needed
for this capability. Native Ring attention and expert parallelism remain outside this scope.
Any future expert-combination path must preserve the same combination order in rollout and
replay before coupled-policy algorithms can opt in.

The online probe stores actual rollout velocity and transition-mean callbacks and compares them,
along with log probabilities, exactly during on-policy replay. It repeats these checks after
optimizer updates and fixed-topology checkpoint restore. Numerical exceptions in cross-layout
gradient qualification never relax these within-run assertions.

## Online numerical evidence

The online probe exercises actual GRPO feedback and DMD2 fake/generator updates, activation
recomputation, EMA, a frozen reference, named snapshots and fixed-topology resume. GRPO's
rollout/replay velocity, transition mean and log probability must agree exactly within each run.
The native Ulysses exchanges described above follow the
[Diffusers parallelism design](https://huggingface.co/docs/diffusers/training/distributed_inference)
and [rank-ordered NCCL all-gather semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).

**Known DMD2 limitation: BF16 feedback sensitivity.** In the tested LoRA configurations,
fake/reference score cancellation amplifies small BF16 differences through subsequent training
feedback. Generator update deltas differ by roughly 8–13% across the retained CP trajectories,
exceeding the original acceptance limit. Record this as numerical instability of the tested
DMD2 training configuration; it is not evidence of a rollout/replay mismatch, nor a finding that
all DMD2 configurations diverge. The local derivative and optimizer checks below do not establish
equivalent trajectories, convergence or model quality. This update-limit exception applies only
to the demonstrated DMD2 generator cases.

BF16 DMD2 and guided GRPO can amplify local gradient rounding through subsequent score or
guidance feedback. Only these demonstrated cases use a separate trajectory qualification;
Full-weight SFT, unguided GRPO and fixed-topology continuation retain their original checks.
SFT LoRA has the separate bounded-gradient qualification below, without the DMD2 feedback allowance. Two labeled
common-state controls install the other trajectory's parameters after each observed update:
CP2 follows CP1's actual states, and CP1 follows CP2's actual states. Before the next
derivative comparison, initial/live parameter values, samples, conditioning, replay inputs and
objective forward operands must agree exactly. Each common-state gradient still passes the
original elementwise and global gradient limits, including zero/half/double negative controls.
This control is derivative evidence; its overwritten updates do not qualify optimizer behavior.

The two ordinary, independently evolving runs separately verify clipping, every Adam moment
and actual parameter delta against FP64 arithmetic. Cross-trajectory gradient/moment norms are
reported as diagnostics for these exceptional cases. Parameter caps and propagated elementwise
Adam bounds remain enforced. Guided GRPO and DMD2 fake-role updates retain the original cross-CP
update cap. DMD2 generator feedback may also exceed that cap: the test reports this failure
explicitly and independently predicts the update from CP1 native gradients at the actual CP2
parameter states. It reconstructs clipping and the complete moment history in FP64, using no
shadow optimizer states or recorded deltas. Actual-to-predicted updates must pass the original
`4/128` global limit and input-derived elementwise bounds. The difference between the two
independent predictions must explain the actual cross-trajectory update difference within those
bounds. Both same-state gradient gates are prerequisites; an unexplained local gradient error
cannot become a feedback allowance. This qualifies local CP derivatives and optimizer behavior
on both actual histories, not closeness or convergence of the two closed-loop trajectories.

DMD2 additionally checks objective and root-output
derivatives against an independent FP64 reference, including detached-target construction,
normalizer clamping, CP/GAS scaling and localized corruption controls. No observed derivative
residual sets its arithmetic budget. Loaded state, fixed-topology continuation and snapshot
recurrences remain separate checks. The SFT numerical qualification below is independent of
this DMD2-generator feedback allowance.

<a id="numerical-evidence"></a>
## SFT numerical evidence

[`test_context_parallel_probe.py`](../tests/distributed/test_context_parallel_probe.py) uses real
Wan blocks and strict FP32 elementwise comparisons (`rtol=3e-4`, `atol=2e-6`) over three updates,
for full and nonzero-initialized LoRA weights, with and without backend recomputation. Its
unscaled-loss negative control detects the erroneous
half-sized gradients.

[`test_context_parallel_training.py`](../tests/distributed/test_context_parallel_training.py)
runs `load_trainer` and `SFTTrainer.start` with a tiny native Wan pipeline, real decoded videos,
text encoding, VAE encoding, scheduler/noise construction, GAS=2, two sampled timesteps and four
optimizer steps. It compares FSDP2 DP2 against DP2 × CP2 for full and LoRA weights, each with and
without recomputation. Initial parameters, sample identities, stochastic inputs and CP-peer
observations must match exactly; every trainable gradient must exist and gradients must be finite
and nonzero. Frozen weights remain unchanged and every exported tensor must equal the final prepared model state in its declared storage dtype
(BF16 full weights; FP32 LoRA adapters).

BF16 reduction order differs between the two meshes. The test checks every output, gradient,
parameter, update delta and Adam moment. Output/gradient/parameter comparisons include explicit
absolute floors for cancellation near zero; update bounds propagate the accepted gradient error
through the optimizer. Gradient comparisons use `rtol=0.02`, `atol=1e-3`; a separate global relative L2 bound of
`2/128` prevents the floor from accepting missing or mis-scaled gradients. Actual update deltas
retain a `4/128` relative L2 bound for full-weight SFT. SFT LoRA records that same cap and its
pass/fail result as a diagnostic: small accepted near-zero gradient differences can cross zero
and produce disproportionate Adam deltas. This opt-in applies only to cross-CP SFT LoRA
comparisons; default callers, GRPO and fixed-topology continuation keep the enforced cap.
Every SFT LoRA gradient, parameter, loss, moment, independently reconstructed clipping/Adam
history and propagated per-element update bound must still pass. Missing/zero/half/double
gradients, nonfinite values and clipping/moment/parameter/update corruption remain failures.
Negative oracle tests reject zero, half and double gradients even below the absolute floor.
The strict FP32 component proof remains independent of these BF16 bounds.

The TP+CP matrix repeats the strict FP32 probe against an unsharded reference, with SP off/on,
full/LoRA and recomputation off/on. Complete BF16 SFT compares DP2 × CP2 × TP2 against
DP2 × TP2: TP/SP stay fixed while CP changes. Both use the same four-head fixture, records,
stochastic inputs and two logical replicas. Every TP × CP peer must retain identical full
observations. The CP-only matrix and its two-head fixtures remain separate regressions.

Changing TP and CP simultaneously is a different BF16 comparison. In the retained LoRA
investigation, TP2 alone and TP2+CP2 both exceeded the CP-only update cap against TP1
(roughly 3.5–4.7% update relative L2). Four small cross-attention q/k LoRA-B gradients dominated
the difference. FP64 reconstruction verified their matrix products, reductions and Adam updates;
common-input attention replay localized the remaining sensitivity to native BF16 backward.
Joining the same heads, inputs and upstream gradients reproduced native BF16 outputs and derivatives
exactly in that investigation. Three additional retained fixtures also failed the unchanged
update bound at fixed TP, motivating the explicit FP32 cross-attention policy. The complete
LoRA test observes actual FP32 SDPA operands and derivatives in CP-off and CP-on runs, with
common-input replay, zero/half/double derivative controls and FP64 diagnostics.
Native FP32 efficient attention can accumulate dQ partials in different orders. Its local
reference therefore checks every derivative against an explicit FP64 softmax VJP with a
forward-only error budget. The budget includes the qualified Torch 2.10 Sm80+ CUTLASS FastF32
three-product arithmetic, partial reductions, underflow and CPU FP64/SLEEF evaluation; it is a
conservative numerical model, not a machine-certified interval proof for arbitrary kernels.
Actual forward outputs and replayed LSE must be reproducible exactly. Captured derivatives and
joined-head native derivatives are checked independently, and local corruption controls use the
same budgets. The gradient and propagated Adam/update bounds above remain enforced, including for SFT LoRA.
The layered evidence qualifies CP composition at fixed TP;
it does not claim that changing TP degree preserves a BF16 trajectory within the CP-only cap.
The original failing cross-TP measurements remain part of the investigation evidence.

Adam moments are checked with an independent FP64 arithmetic reference. The test records actual
pre-clip and post-clip gradients and optimizer hyperparameters, verifies global-norm clipping,
then reconstructs every first moment, second moment and parameter update. FP32 roundoff budgets
come from operation counts (`gamma(k) = k*u/(1-k*u)`, `u = eps32/2`). Cross-CP moment bounds
propagate the already bounded gradient and clipping errors through the Adam recurrence, retaining
the global relative L2 limit. A fixed absolute moment floor would incorrectly ignore accumulated
history when signed gradients cancel. CPU controls cover that cancellation, warm-state corruption,
incorrect clipping and an omitted parameter update.

Each recorded FP32 delta must exactly equal the observed after-minus-before parameters. Its
cross-CP elementwise bound comes from actual, independently accepted post-clip gradient
differences propagated through both moment recurrences. For bias-corrected moments `M`, `V`,
let `s=sqrt(V)` and `d=s+eps`. The direction difference is bounded by
`B_M/d_a + abs(M_r)*B_V/((s_a+s_r)*d_a*d_r)`; the square-root term is zero when both variances
are zero. Taking the smaller bound after swapping the two runs preserves validity. The test
adds weight-decay drift and each run's FP32 update/subtraction roundoff. All denominators come
from independently verified moment states, and no bound is inferred from observed deltas.

A fixed absolute delta floor does not follow from the accepted raw-gradient floor: near-zero
sign changes can produce a larger Adam update difference even when the complete gradient and
update norms pass. The earlier retained CP-only regression passes this case under the original gradient
thresholds, optimizer epsilon and `4/128` actual-update cap. Local delta-only corruption and
consistent parameter-plus-delta corruption must fail even when their global error is below that
cap. These checks explain accepted low-precision input differences; they do not establish
high-precision gradient accuracy or convergence.

The later retained TP2/CP2 SFT LoRA fixture (checkpointing on, SP off) exceeds that global
update cap at step one: 3.37103% against 3.125%, despite a 0.43781% global gradient difference.
A cross-attention output-projection LoRA-B gradient changes from +3.05176e-5 to -1.14441e-4;
independent FP64 Adam reconstruction explains the amplified delta. Both the original cap
failure and a controlled FP32 cross-attention linear intervention that still fails are retained.
The SFT-LoRA-only diagnostic treatment was explicitly approved after this evidence. It does
not establish equal training trajectories, high-precision gradient accuracy or model quality.

This is Q2 evidence on small random-weight native components. It does not certify pretrained
Wan output quality, convergence, peak-memory scaling, or every CP degree. Preserve the exact
source, dependency tuple and physical placement in the selected validation results.
