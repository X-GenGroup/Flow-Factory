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

The capability identifier is `wan-sft-ulysses-bf16-v1`. Its supported configuration is:

- `wan2_t2v`, one `WanTransformer3DModel`, dense fixed-size video, scalar sample timesteps,
  equal input/output channels, text conditioning without added image KV projections.
- SFT, full or LoRA tuning, one `transformer` target, one example per device, BF16 mixed precision,
  AdamW, and explicit gradient accumulation. Backend activation checkpointing may be enabled.
- FSDP2 on CUDA/NCCL, TP=1, replicated-DP=1 and sharded-DP at least two. Attention heads and
  video tokens must divide evenly by the Ulysses degree; latent dimensions must divide by patches.
- One shared `attention_backend` accelerator with `params: {backend: native}`. No other shared
  or rollout accelerators, evaluation or EMA. Set `eval.eval_freq: 0` and `train.ema_decay: 0`.
- Model-only weight export (`log.save_model_only: true`). Exact state resume is rejected.
- PyTorch 2.10.x, Accelerate 1.15.x and Diffusers 0.41.x release lines. Development/prerelease
  versions are rejected. The tested tuple uses PyTorch 2.10.0+cu129, Accelerate 1.15.0,
  Diffusers 0.41.0 and PEFT 0.21.2.

Wan2.2's expanded per-token timestep layout and two-transformer schedules are not included.
Neither are I2V, masks, prefix caches, uneven tokens, native Ring, TP/SP composition, online
algorithms, reference policies, or state continuation. Later roadmap steps qualify those
compositions separately. A public capability is a bounded experimental path, not a guarantee
for every checkpoint in the model family.

## Preparation and reduction ownership

`ModelPreparation` runs after all model variants exist. FSDP2 replicated loading synchronizes
initial parameters and buffers before sharding, including randomly initialized LoRA weights.
The existing attention accelerator selects native attention exactly once, before Diffusers
installs its Ulysses hooks. Those hooks are attached before optimizer construction and FSDP2
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
