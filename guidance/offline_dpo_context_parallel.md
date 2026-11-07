# Offline DPO with context parallelism

Offline DPO uses the same full-output backward, FSDP2 preparation and drained checkpoint owners
as SFT. Its concrete CP declaration also requires both preference arms and the frozen reference
to preserve the same conditions, noise and timestep coordinates. No training rollout, reward
service or overlap stage is added to the finite dataset path.

Use the [offline preference schema](datasets.md) and existing `trainer_type: offline-dpo` settings.
For a supported Wan T2V checkpoint, the parallel delta is:

```yaml
mixed_precision: bf16
parallelism:
  dp_shard_size: 2
  tp_size: 2
  sequence_parallel: true
  cp_size: 2
model:
  model_type: wan2_t2v
  target_components: [transformer]
  compute_precision_policy: cross_attention_fp32_v1
train:
  trainer_type: offline-dpo
  per_device_batch_size: 1
  gradient_accumulation_steps: 2
  num_train_timesteps: 2
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

Merge this delta into a complete model/data/optimizer configuration and launch eight local ranks
with FSDP2. The [existing CP constraints](context_parallel.md) still apply. For Flux1 use four
ranks, `tp_size: 1`, `sequence_parallel: false`, `model_type: flux1`, and omit the Wan-specific
precision policy. [Flux TP+CP](flux_context_parallel.md#tensor-parallel-composition) has separate
model-composition qualification; this offline-DPO case covers TP1 only.

The chosen/rejected pair keeps one record identity on all model peers. The condition is prepared
once; target encoding draws are distinct, while the subsequent forward-process noise and time
are shared between arms. The reference scope is detached and restores the live policy before
backward. The pair objective stays unscaled in metrics; the shared backward boundary applies the
CP factor once, and Accelerate applies GAS normalization. Each finite epoch closes complete GAS
windows and checkpoints at its drained boundary.

## Evidence composition

| Selector | Purpose |
| --- | --- |
| `offline-dpo-cp-contracts` | Actual objective CPU values and analytic upstream derivatives; exact noise/time, reference isolation, CP/GAS and corruption controls |
| `flux-offline-dpo-cp` | Full native Flux packed-image preference training, CP1/CP2 at DP2, reference snapshots, gradients/Adam, EMA and fresh-process continuation |
| `wan-offline-dpo-cp` | Native Wan LoRA video preference training with TP2+SP, CP1/CP2 at DP2; the same pair/update/state guarantees |

Both native cases use real tiny Diffusers pipelines, codecs and finite datasets, two epochs,
two Monte Carlo time terms per microbatch and GAS2. They compare every trainable gradient,
clipped gradient, Adam state and update through the existing numerical bounds. Restored weights,
optimizer moments, runtime children and RNG must match the saved state exactly; later updates
use the declared BF16 bounds. The SFT LoRA cross-layout exception does not apply to Offline DPO.
These are bounded functionality proofs, not pretrained convergence or throughput measurements.

The existing shared SFT dataset and concrete coupled-model evidence remains useful for unchanged
preparation, topology and state ownership. It does not replace the preference calculus and native
pair witnesses above. Reward overlap is structurally inapplicable to this offline algorithm.
