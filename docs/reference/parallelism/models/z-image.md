# Z-Image TP/SP and context parallelism

Z-Image's ordinary text-to-image layout supports native Ulysses CP with its TP and SP plans for both refiners and the main stack. The model-local processor retains native FP32 attention and rotary math, the image/text refiners, learned padding parameters and the unified main transformer stack.

```yaml
model:
  model_type: z-image
  target_components: [transformer]
parallelism:
  dp_shard_size: 2
  tp_size: 2
  sequence_parallel: true
  cp_size: 2
mixed_precision: bf16
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

Use the existing FSDP2 launcher and ordinary algorithm/data/model settings. The eight-rank shape is DP2 x TP2/SP x CP2. Z-Image retains its FP32 transformer storage policy. The [shared admission and checkpoint rules](../context-parallel.md) still apply, including explicit algorithm capabilities and drained fixed-topology continuation.

## Native layout ownership

TP shards all three stacks. Noise and caption refiners independently scatter once at entry and
gather at exit, retaining token-local residuals/norms inside each refiner. Native masks and rotary
tables stay complete for gathered attention. The outputs return in full before
`_build_unified_sequence` performs native assembly. CP partitions the unified main-stack input
and rotary positions; main-stack SP then partitions each CP-local residual. The final image
projection computes on SP tokens and gathers SP then CP before unpatchification.

Refiner padding follows the native multiple-of-32 rules; each resulting length must divide TP.
Affine norms sum token-partial parameter derivatives. Each modulated block reduces its own
conditioning derivative, so the shared timestep producer receives complete contributions once.
Learned padding parameters and their gradient paths remain native.

The actual 30 attention heads become 15 per TP2 rank. Independent dummy Q/K/V heads pad only attention communication operands to 16, and are discarded before the native projection. No parameter, head declaration or checkpoint layout changes. TP4/8 remain incompatible with the model's 30 heads; the padding does not relax TP divisibility.

Native learned image/text padding to multiples of 32 remains active. Ragged batches retain the native full key-validity mask and positions. This path admits the existing single-frame T2I layout, patch size 2 and temporal patch size 1. Omni inputs, SigLIP conditioning and ControlNet residuals are rejected before collectives. Generation, CFG, reversed timesteps, output sign and scheduler behavior stay with the existing adapter.

## Evidence

- CPU contracts preserve ordinary processor outputs and every gradient, actual Qwen3 text and sixteen-channel VAE save/reload/decode, unequal prompt lengths and early metadata/input failures.
- Four-rank Gloo development controls use full/LoRA, eager/checkpointed execution and native zero-initialized learned padding. FP64 reference controls retain three Adam updates/moments and ragged 19/41 text positions. This isolates near-zero RMSNorm cancellation observed in the FP32 development diagnostic without loosening its comparison threshold.
- `z_image-tp-sp-cp-reference` uses native 30 x 128 attention geometry on eight CUDA ranks with FSDP2, TP2/SP and CP2. Full/LoRA FP32 controls compare every gradient and three Adam updates/moments against independent logical-DP means, with the missing CP loss factor as a negative control. Learned pad parameters use a declared nonzero, pretrained-like initialization; the separate FP64 control retains the zero initialization edge case.
- `z_image-guard-ordered-lora` uses real text encoding and decoded-image rewards in Guard, CFG, ordered reward overlap, on-policy replay, evaluation, EMA and fresh-process drained continuation. It requires exact checkpoint/work restoration, the first same-state update under the original bounds, and each later actual run's objective/derivative, optimizer, EMA and state checks under the [ordered continuation contract](../context-parallel.md#ordered-continuation-evidence). Later cross-run numerical statistics are diagnostic.

The fixtures bound native depth and encoder widths. Exact-source validation records establish the executed configurations, not pretrained convergence or throughput.

## Refiner SP selection

`z_image-refiner-sp-cpu`, `z_image-refiner-sp-cuda` and `z_image-refiner-sp-cp-reference` select
multiple-block native refiner coverage, full/expanded-LoRA checkpoint paths, all derivatives,
three updates/moments and observed local refiner rows. The CPU TP2CP2 control is a four-process
Gloo test; its launcher declaration is not a one-rank collective claim. Existing main-stack
results remain historical evidence for their source. A passing current refiner gate establishes
numerical/layout coverage, not a measured peak-memory or throughput improvement.
