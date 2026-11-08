# Experimental Qwen Image Edit Plus context parallelism

`qwen-image-edit-plus` declares its own `qwen_edit_ordered_reference_image_text` layout.
It supports the native Edit-2509 transformer (`zero_cond_t=False`) and Edit-2511 transformer
(`zero_cond_t=True`). Both retain ordered image references through the native Qwen2.5-VL
processor and VAE, target-first rotary positions, and the final generated-token crop.

Native Diffusers CP splits the concatenated target/reference image sequence, text, masks and
rotary positions. Edit-2511 also splits its per-token time-modulation index. The Qwen family
composition helper orders these splits before TP sequence parallelism and restores both image
shards before the adapter crops the generated target. Text SP begins after native rotary
construction; the first-block CP text split runs before the SP scatter. Both residual streams
remain sharded across blocks.
An individual target/reference boundary need not align with a CP or SP shard.

## Configuration

The [shared CP runtime constraints](../context-parallel.md#experimental-capability) apply: FSDP2,
BF16, native attention, AdamW, one sample per device, one transformer target and sharded DP of
at least two. An eight-device DP2 × TP2/SP × CP2 setup uses:

```yaml
mixed_precision: bf16
parallelism:
  dp_shard_size: 2
  tp_size: 2
  cp_size: 2
  sequence_parallel: true
model:
  model_type: qwen-image-edit-plus
  target_components: [transformer]
train:
  per_device_batch_size: 1
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

Use an ordered `images` list in each input record. The existing adapter derives output aspect
ratio from the last reference. Attention heads must be divisible by CP × TP. Total target plus
reference image tokens must be divisible by CP × TP when SP is enabled, or by CP otherwise. Text embeddings receive
zero-valued, masked padding to a CP multiple, extended to CP × TP with SP. Prompt trimming preserves the last active token
for masks with holes at both CP1 and CP>1; ordinary prefix masks retain their previous behavior.
The adapter validates both CFG branches before starting the native transformer.
Cached evaluation reference sizes are normalized from integer tensors to Python geometry before
constructing rotary positions. Floating, boolean, malformed or nonpositive sizes are rejected.
Single-sample forward preserves the native scheduler tensors and existing public result type, so
its mean and velocity retain their log-probability graph. Multi-sample aggregation is unchanged.

The admitted transformer has native joint attention, ordinary three-axis rotary positions,
patch size two, matching packed input/output channels and boolean `zero_cond_t`. Guidance
embedders, additional timestep conditions, layer-3D rotary layouts and custom attention
processors or kwargs are separate layouts and fail before weights load or collectives.
Missing references, inconsistent ordered shapes and incompatible token counts also fail early.

## Focused evidence

`test_qwen_edit_context_parallel_contracts.py` compares independent native conditional/CFG
passes and every trainable gradient for both variants in full and LoRA modes. It checks
reference ablation and order, target cropping, time-modulation indices, malformed input
rejection, and native saved-pipeline reload. Its genuine Qwen2.5-VL processor/vision/text and
Qwen VAE fixture checks image-token expansion, the native 64-token prompt-template crop,
VL pixel influence and independently packed reference latents.

`test_cuda_qwen_edit_tp_sp_cp_reference` covers both variants in full and LoRA modes on the
eight-device mesh with activation checkpointing. Identical inputs feed an independent
unsharded FP32 model. The oracle compares target-only outputs, every gradient, three Adam
updates and optimizer moments, and requires a deliberately unscaled CP loss to disagree.
Target 8 plus ordered reference spans 6 and 10 form 24 tokens: CP chunks of 12 and SP chunks
of 6 cross target/reference boundaries. Edit-2511 checks the exact local modulation slices.

`test_cuda_qwen_edit_guard_tp_sp_cp_ordered_full` exercises Edit-2511 full-weight Guard with
actual two-reference dataset records, CFG, native VL/VAE preprocessing, ordered reward overlap,
odd-tail evaluation and drained fresh-process continuation. It reuses the independent Guard
objective, gradient, optimizer and replay controls and checks retained reference latents and
positions. The separate full/LoRA reference cases provide variant and parameter-mode coverage.

These bounded model-structure cases do not establish pretrained visual quality or every
algorithm/model intersection. The selected lifecycle uses Edit-2511; Edit-2509 is covered by
the native full/LoRA numerical controls. Topology-changing resume and in-flight recovery are
outside this layout qualification.

## Pretrained correspondence

The official [Edit-2509 transformer configuration](https://huggingface.co/Qwen/Qwen-Image-Edit-2509/blob/main/transformer/config.json)
and [Edit-2511 transformer configuration](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/blob/main/transformer/config.json)
use 24 heads of width 128, packed input width 64, output width 16, text width 3584, patch size
two and rotary axes `(16, 56, 56)`. Edit-2511 enables `zero_cond_t`; Edit-2509 uses the native
false default. The tiny fixture reduces widths and depth while preserving native attention,
ordered position construction, both modulation paths and real VL/VAE components. Pretrained
output acceptance, large-geometry numerical drift and memory capacity remain P13 work.

## Shared text-SP extension

Edit-Plus delegates to the [ordinary Qwen TP/SP plan](qwen.md#text-stream-sp-evidence), including
masked odd-text padding, text residual/norm/modulation/FFN sharding and wrapper-owned final
projection gathering. Edit-2511 retains its per-token modulation-index cuts and ordered reference
layout. The shared text-stream CPU/two-GPU cases and `qwen-edit-text-sp-cp-reference` select this
expanded region; earlier image-only-SP receipts keep their original scope. Custom final projection
and modulation LoRA are checked by the expanded-target component controls.
