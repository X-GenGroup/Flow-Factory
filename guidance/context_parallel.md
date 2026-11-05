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

The CP-only capability is `wan-sft-ulysses-bf16-v1`. TP composition uses
`wan-sft-tp-ulysses-bf16-v1`. Both share these constraints:

- `wan2_t2v`, one `WanTransformer3DModel`, dense fixed-size video, scalar sample timesteps,
  equal input/output channels, text conditioning without added image KV projections.
- SFT, full or LoRA tuning, one `transformer` target, one example per device, BF16 mixed precision,
  AdamW, and explicit gradient accumulation. Backend activation checkpointing may be enabled.
- FSDP2 on CUDA/NCCL, replicated-DP=1 and sharded-DP at least two. CP-only uses TP=1;
  the composed capability uses TP>=2 with Wan's training-aware TP plan and optional SP.
  Attention heads must divide by TP × CP. Video tokens must divide by CP, and additionally
  by TP when SP is enabled; latent dimensions must divide by patches.
- One shared `attention_backend` accelerator with `params: {backend: native}`. No other shared
  or rollout accelerators, evaluation or EMA. Set `eval.eval_freq: 0` and `train.ema_decay: 0`.
- Model-only weight export (`log.save_model_only: true`). Exact state resume is rejected.
- PyTorch 2.10.x, Accelerate 1.15.x and Diffusers 0.41.x release lines. Development/prerelease
  versions are rejected. The tested tuple uses PyTorch 2.10.0+cu129, Accelerate 1.15.0,
  Diffusers 0.41.0 and PEFT 0.21.2.

Wan2.2's expanded per-token timestep layout and two-transformer schedules are not included.
Neither are I2V, masks, prefix caches, uneven tokens, native Ring, online
algorithms, reference policies, or state continuation. Later roadmap steps qualify those
compositions separately. A public capability is a bounded experimental path, not a guarantee
for every checkpoint in the model family.

## TP/SP composition

The [eight-rank example](../examples/sft/lora/wan2_t2v/wan21_tp_cp.yaml) declares
`tp_size: 2`, `cp_size: 2`, `dp_shard_size: 2` and `sequence_parallel: true`. Eight
physical ranks still represent two logical data replicas. TP peers own different heads/weights;
CP peers own different token intervals. Disabling SP preserves the TP+CP composition while
keeping the residual token stream replicated across TP peers.

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
The existing attention accelerator selects native attention exactly once, before Diffusers
installs its Ulysses hooks. The optional SP boundary bridge is installed after native hooks. All hooks precede optimizer construction and FSDP2
preparation. Accelerate builds the mesh and owns FSDP2; its CP preparation bridge only accepts
one FF bundle carrying the installed-hook contract. It cannot independently replace the hooks.

The adapter returns the full denoiser output in canonical order. Diffusers' output gather sends
only each rank's local output slice through backward. FSDP2 averages over DP × CP, so SFT scales
the backward loss by the CP degree to recover the mean over logical samples. Reported loss
remains unscaled. This rule belongs to SFT's full-output mean objective; token-local, masked or
other algorithms must derive their reduction contract independently.

## Numerical evidence

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
parameter, update delta and Adam moment, with explicit absolute floors for cancellation near
zero. Gradient comparisons use `rtol=0.02`, `atol=1e-3`; a separate global relative L2 bound of
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
Joining the same heads, inputs and upstream gradients reproduced native outputs and derivatives
exactly. The complete LoRA test retains this local check and zero/half/double derivative controls,
plus FP64 diagnostic residuals. The layered evidence qualifies CP composition at fixed TP;
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

This is Q2 evidence on small random-weight native components. It does not certify pretrained
Wan output quality, convergence, peak-memory scaling, or every CP degree. Preserve the exact
source, dependency tuple and physical placement in the selected validation results.
