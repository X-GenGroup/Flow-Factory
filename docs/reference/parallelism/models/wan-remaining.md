# Remaining Wan context-parallel layouts

The `wan2_t2v` and `wan2_i2v` adapters admit the native Wan layouts below with
FSDP2, BF16, the native attention backend, and
`model.compute_precision_policy: cross_attention_fp32_v1`.
The existing scalar [Wan T2V](../context-parallel.md) and
[first-frame Wan2.1 I2V](wan-i2v.md) paths remain available.

| Adapter | Native layout | Conditioning and denoisers |
| --- | --- | --- |
| `wan2_t2v` | Wan2.2 T2V-A14B | 40 heads, equal 16 input/output channels, two scalar-time denoisers |
| `wan2_t2v` | Wan2.2 TI2V-5B text-to-video | 24 heads, equal 48 input/output channels, one denoiser with per-token time |
| `wan2_i2v` | Wan2.2 I2V-A14B | 40 heads, 36 input / 16 output channels, VAE first/optional-last frame, no CLIP, two scalar-time denoisers |
| `wan2_i2v` | Wan2.2 TI2V-5B image-to-video | 24 heads, equal 48 input/output channels, first-frame VAE blend and per-token mask/time |
| `wan2_i2v` | Wan2.1 FLF2V | 40 heads, 36 input / 16 output channels, ordered first/last VAE conditions and 257 CLIP tokens per frame with 514 learned positions |

Declare both canonical names with `target_components: [transformer, transformer_2]`
to retain both denoisers in the prepared component graph. Both consume identical
latent and conditioning geometry; `boundary_ratio` chooses the active denoiser and
its corresponding guidance scale. To train only the first denoiser while preparing
the frozen sibling, retain both names and scope `target_modules: [transformer.*]`.
This produces a `None` module selection for `transformer_2`. Omitting that name from
`target_components` leaves the auxiliary denoiser replicated and unprepared.
Expanded time uses one transformer and native `(1, 2, 2)` patches. Its optional
last-frame input remains rejected because the native first-frame blend does not consume it.
FLF2V requires exactly two ordered input images. Its native image embedder folds
`[2,257,image_dim]` into `[1,514,inner_dim]`; all image/text keys and values stay
replicated over CP, while the generated video tokens are split.

```yaml
mixed_precision: bf16
parallelism:
  dp_shard_size: 2
  tp_size: 2
  cp_size: 2
  sequence_parallel: true
model:
  model_type: wan2_i2v
  target_components: [transformer, transformer_2] # A14B dual schedule
  compute_precision_policy: cross_attention_fp32_v1
train:
  per_device_batch_size: 1
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

Heads must be divisible by CP × TP. Generated tokens must be divisible by CP × TP
when SP is enabled, or by CP otherwise. Each latent dimension must be divisible by
its patch dimension. Native Diffusers splits per-token timesteps before embedding;
Wan's existing sequence alignment then splits their modulation over SP. The existing
family composition preserves CP-before-SP input order and SP-before-CP output gathering.
No model-specific gradient normalization is added: the shared full-output loss owner
continues to compensate for native CP gather's local-slice backward.

## Evidence and limits

`test_wan_remaining_context_parallel_contracts.py` compares all five layouts in full
and LoRA modes against independently constructed native operands, including CFG,
both schedule branches, every parameter gradient, strict metadata and mask rejection,
and real native CLIP/UMT5/Wan-VAE save/reload. The same realized image conditions are
compared between rollout preparation and offline condition preparation. Swapping first
and last images changes both their VAE condition and position-sensitive CLIP conditioning.

`test_cuda_wan_remaining_tp_sp_cp_reference` selects all five layouts in full/LoRA
modes at DP2 × TP2/SP × CP2. It compares independent unsharded FP32 outputs, every
trainable gradient and three Adam updates/moments, with an omitted-CP-factor negative
control. Native hooks record actual CP-local timestep values, three-token SP modulation,
six-token attention, and both dual denoiser roots. Twelve generated tokens put the
first-frame mask boundary at token 4, inside both the six-token CP shard and three-token
SP shard. Fixtures preserve published head counts, latent channels, patch geometry and
conditioning lengths; layers and internal widths are reduced. The 5B conditioning fixture
retains its residual patchified Wan VAE and 16× spatial compression, using 64px inputs to
produce the same 4×4 latent grid; scalar layouts use 32px inputs and 8× compression.

Three native GRPO cases exercise dual I2V/full, expanded I2V/full and FLF2V/LoRA with
actual ordered images, native preprocessing, guided acquisition, finite video rewards,
ordered reward/optimization overlap, parameter/Adam/clipping/EMA observations,
evaluation and fresh-process exact checkpoint loading/work identity. They use the
[ordered continuation contract](../context-parallel.md#ordered-continuation-evidence):
the first same-state update retains the original BF16 bounds; every later actual
objective/derivative, clipping/Adam, EMA and state check remains required. Later
cross-run numerical statistics are diagnostic. Published qualification records retain
their original tested source and comparison semantics.

These are structural/numerical qualification cases, not pretrained output-quality or
large-model memory benchmarks. Consult the exact selected-plan result for executed
CPU/GPU evidence; an admitted metadata declaration alone is not a passing GPU result.

## Shared SP scope

Each active denoiser uses the family's continuous video residual/norm/modulation SP region through
its final output projection. Per-token timestep rows follow the corresponding video slices;
per-sample conditioning retains its separate broadcast-gradient ownership. Text and optional
image-conditioning token streams remain whole over TP, even where their projections use TP.
Scheduled transformer roots retain their existing component order and state ownership. The
[Wan I2V boundary](wan-i2v.md#sp-activation-boundary) and model-specific native controls describe
the corresponding conditioning layouts; these declarations do not supply a memory/speed figure.
