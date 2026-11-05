# Context parallelism

Context parallelism (CP) splits one video's token sequence across ranks. The first experimental
training path uses native Diffusers Ulysses with Wan T2V and FSDP2. Configure it explicitly:

```yaml
parallelism:
  cp_size: 2
  dp_shard_size: 2
mixed_precision: bf16
```

Four physical ranks represent two logical data replicas. CP peers receive the same record,
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
  or rollout accelerators, evaluation or reward/optimization overlap. Set `eval.eval_freq: 0`.
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

## TP/SP composition

The [eight-rank example](../examples/sft/lora/wan2_t2v/wan21_tp_cp.yaml) declares
`tp_size: 2`, `cp_size: 2`, `dp_shard_size: 2` and `sequence_parallel: true`. Eight
physical ranks still represent two logical data replicas. TP peers own different heads/weights;
CP peers own different token intervals. Disabling SP preserves the TP+CP composition while
keeping the residual token stream replicated across TP peers.

TP+CP additionally requires `model.compute_precision_policy: cross_attention_fp32_v1`.
This [model-declared operation policy](compute_precision.md) keeps cross-attention SDPA in
FP32 with BF16 boundaries. It is applied equally to CP-off and CP-on comparisons; CP-only
remains a separate native-precision regression. Precision and its operator cost are explicit.

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
SFT, unguided GRPO and fixed-topology continuation retain their original checks. Two labeled
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
recurrences remain separate checks. The SFT numerical limits below are unchanged.

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
have a `4/128` relative L2 bound because Adam amplifies small near-zero gradient differences.
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
same budgets. The end-to-end gradient and Adam/update bounds above remain unchanged.
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
update norms pass. The retained CP-only regression covers this case without changing gradient
thresholds, optimizer epsilon or the `4/128` actual-update cap. Local delta-only corruption and
consistent parameter-plus-delta corruption must fail even when their global error is below that
cap. These checks explain accepted low-precision input differences; they do not establish
high-precision gradient accuracy or convergence.

This is Q2 evidence on small random-weight native components. It does not certify pretrained
Wan output quality, convergence, peak-memory scaling, or every CP degree. Preserve the exact
source, dependency tuple and physical placement in the selected validation results.
