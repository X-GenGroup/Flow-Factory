# Experimental Wan2.1 I2V context parallelism

`wan2_i2v` declares a distinct `wan_image_conditioned_video` layout for ordinary Wan2.1
first-frame image-to-video conditioning. Native Diffusers owns generated-video token splitting,
rotary slices, Ulysses self-attention and the final gather. The 512 text tokens and 257 CLIP image
tokens remain replicated across CP peers. Text and image cross-attention both use
`cross_attention_fp32_v1`; the image keys and values use the adapter's existing TP projections.

The bounded layout uses one `WanTransformer3DModel`, scalar sample timesteps, concatenated VAE
condition channels and four mask channels (`in_channels = 2 * out_channels + 4`). Wan2.2 expanded
timesteps, dual-transformer schedules and FLF2V positional image embeddings have separate
[variant contracts and evidence](wan-remaining.md). For this first-frame layout,
a second image, malformed condition geometry,
non-512 text context, non-257 CLIP context and custom attention kwargs fail before collectives.

## Configuration

Use the [shared CP runtime constraints](../context-parallel.md#experimental-capability): FSDP2,
BF16, native attention, AdamW, one sample per device, one transformer target, and sharded DP at
least two. This configuration delta selects an eight-device DP2 × TP2/SP × CP2 composition:

```yaml
mixed_precision: bf16
parallelism:
  dp_shard_size: 2
  tp_size: 2
  cp_size: 2
  sequence_parallel: true
model:
  model_type: wan2_i2v
  target_components: [transformer]
  compute_precision_policy: cross_attention_fp32_v1
train:
  per_device_batch_size: 1
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

The attention head count must be divisible by CP × TP. Generated video tokens must be divisible
by CP × TP when SP is enabled, or by CP otherwise; each latent dimension must be divisible by its patch size. The adapter
owns the same native split-before-SP / gather-after-SP boundary ordering as Wan T2V. Image/text
conditioning is not padded or split to satisfy token divisibility.

## Bounded evidence

`test_wan_i2v_context_parallel_contracts.py` checks metadata and tensor rejection before native
collectives, ordinary CP=1 compatibility, native conditional/CFG output and every trainable
gradient, and sensitivity to both CLIP image embeddings and VAE condition channels. Its saved
fixture uses native CLIP, UMT5, Wan VAE and Wan transformer components, actual 512/257 context
lengths, conditioning PNGs and strict finite-video demonstration records.

`test_cuda_wan_i2v_tp_sp_cp_reference` compares full and LoRA DP2 × TP2/SP × CP2 preparation with
an independent unsharded FP32 model using identical inputs and logical samples. It covers native
activation checkpointing, every gradient, three Adam updates, warm optimizer state, frozen
parameters and the deliberately incorrect unscaled-CP-loss control.

`test_cuda_wan_i2v_tdm_full_tp_cp_lifecycle` selects native BF16 full-weight I2V TDM on the same
eight-device topology, guided student/reference queries, K=2/GAS=4, two fake updates per generator
update, two acquisition cycles, generation evaluation, EMA and drained fresh-process resume.
It reuses the existing independent TDM objective/optimizer oracle and additionally records exact
image embeddings and VAE condition channels across continuation.

These model-structure cases do not establish pretrained output quality, Wan2.2 or FLF2V support,
all algorithm/model intersections, reward overlap, topology-changing resume or in-flight recovery.
Algorithm admission remains the concrete trainer capability; the I2V TDM intersection is the
selected lifecycle witness. The unchanged finite-data codec and condition preparer retain their
separate CPU geometry/reuse tests; no new finite-data GPU claim is made here.

## Pretrained structural correspondence

The official [Wan2.1 I2V 480P transformer config](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/blob/main/transformer/config.json)
declares 36 input channels, 16 output channels, 40 heads of width 128, image width 1280,
added-KV width 5120, text width 4096 and patches `(1, 2, 2)`. Its
[CLIP config](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/blob/main/image_encoder/config.json)
has image size 224 and patch size 14, yielding 256 image patches plus the class token.
The native processor splits image/text context using the final 512 text tokens.

The fixture preserves those token counts, channel relation, patch layout, separate image-KV
projections, across-head normalization and native component implementations while reducing
channel width and layer count. The independent gradient controls target image-KV reduction and
sequence ordering; the lifecycle case targets condition reuse, guided replay and prepared roles.
No Wan2.1 I2V pretrained checkpoint was executed for this slice. Large-geometry numerical drift,
memory capacity and pretrained perceptual acceptance remain P13 real-weight acceptance work.

## SP activation boundary

The video residual remains token-sharded across all denoising blocks, final normalization and
output projection; gathering follows the projection and precedes native output reconstruction.
Text and CLIP conditioning remain whole over TP as well as under the CP contract above. Their
K/V weights can be tensor-sharded without token-sharding their conditioning sequence. Patch and
conditioning embeddings are produced before SP. This distinguishes activation coverage from
parameter placement and does not add a new performance claim.
