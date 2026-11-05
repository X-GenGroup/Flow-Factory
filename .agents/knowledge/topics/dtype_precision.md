# Dtype & Precision

**Read when**: Touching dtype/precision, mixed precision config, or debugging NaN/overflow.

---

## Model-declared operation precision

`model.compute_precision_policy` is an explicit adapter-owned, versioned operation policy,
independent of load/storage dtype. Rules select module paths and native SDPA FP32 execution
inside disabled autocast, restoring the query dtype at the output. Install after loading/LoRA
and before variant cloning; preserve the inspectable forward signature required by TP regions.
Full-variant copies must bind original methods to their own parameters. Exact-resume identity
locks declarations and resolved paths; native/null keeps the earlier identity. Only Wan T2V
cross-attention is opted in. Details and costs: [compute precision](../../../guidance/compute_precision.md).

## Precision Boundaries

| Component | Runtime dtype | Why |
|-----------|--------------|-----|
| Pretrained component load | `component_load_dtypes` over adapter defaults | Lets the native loader enforce model rules such as Diffusers FP32 islands |
| Frozen params/buffers (frozen transformer base, VAE, text encoders) | `frozen_parameters_dtype`; default `None` performs no post-load mutation | Released components may use different dtypes; use an explicit override only when the model permits it |
| Transformer (trainable) | `trainable_parameters_dtype` (fp32/bf16) | Gradient precision |
| Scheduler math | `float32` always | `1/sigma` amplification (see below) |
| Latent storage (trajectory) | `latent_storage_dtype` (configurable) | Memory vs. precision tradeoff |
| Forward-process adapter boundary | Exact dtype/device of each clean latent component | Structured adapters reject noise with a different storage representation |
| Advantage computation | `float64` (numpy) | Normalization stability |

Boundaries are set in `BaseAdapter._mix_precision()` (`models/abc.py`) and `BaseTrainer.__init__` (autocast context). Autocast weight-cache invariant + in-place ref/EMA/named swaps: `topics/autocast_param_swap.md` (#20a).

`component_load_dtypes` and `frozen_parameters_dtype` accept either one dtype or a selector mapping. Resolution order is
concrete component, component group (`transformers` / `text_encoders`), then `default`. A null
load result delegates to the native loader, while a null frozen result performs no post-load
mutation. `transformer` names one concrete component; `transformers` selects every transformer
declared by the pipeline. To disable an adapter's load manifest, set
`component_load_dtypes: {default: null}`. Under FSDP2 mixed precision, trained components keep
uniform fp32 original parameters while the resolved frozen policy still applies to fully frozen
components. Late variant creation and role-checkpoint alignment preserve the same FP32 original
dtype before TP constructs DTensor metadata; a later parameter-data cast is insufficient.

## `cast_latents()` Contract

`BaseAdapter.cast_latents()` (`models/abc.py`) casts latents to `latent_storage_dtype` for trajectory storage.

- **float16 overflow protection**: clamps values exceeding 65504.0 with a warning.
- **Identity when no target**: returns latents unchanged if `latent_storage_dtype` is unset and no default provided.
- **Must be applied identically** in both rollout and training paths — inconsistency breaks train-inference consistency (-> `train_inference_consistency.md` item #3).

```python
def cast_latents(self, latents, default_dtype=None):
    target = self.latent_storage_dtype or default_dtype
    if target is None or latents.dtype == target:
        return latents
    if target == torch.float16:
        abs_max = latents.abs().max().item()
        if abs_max > 65504.0:
            latents = latents.clamp(-65504.0, 65504.0)
    return latents.to(target)
```

## 1/sigma Error Amplification

Scheduler math uses `1/sigma` to scale velocity predictions. Near the end of the denoising schedule, sigma approaches zero and errors are amplified:

```
Example: sigma = 0.01, epsilon_error = 1.5e-4
Amplified error: epsilon_error / sigma = 1.5e-4 / 0.01 = 1.5e-2

Over 30 steps with accumulated error: total_drift ≈ 6.0
```

This is why scheduler math is always `float32` and why the dtype round-trip guard exists in schedulers:

```python
next_latents = next_latents.to(_input_dtype).float()
```

The round-trip ensures that the precision of stored latents matches what training will see — without it, float32 scheduler output stored as bf16 loses precision, and the training replay produces different `log_prob`.

## Diagnosis Checklist

| Symptom | Check |
|---------|-------|
| NaN in loss after few steps | `latent_storage_dtype=float16` with large latent values? Check `cast_latents()` clamp warnings. |
| `ratio` drifts from 1.0 at epoch start | Compare `forward()` output dtype between rollout and training. Verify `cast_latents()` is called in both paths. |
| Gradients explode near end of schedule | Scheduler using lower-than-float32 precision? Check `_input_dtype` round-trip in scheduler. |
| Reward NaN but generation looks normal | Advantage normalization overflow — verify `float64` in `advantage_processor.py`. |
| Forward noising rejects float32 noise for fp16/bf16 latents | Keep stochastic/likelihood math in float32, then cast the adapter-bound noise component with `.to(clean_component)` before calling `apply_forward_process_noise()`. |

## Fix records

### TDM conditional noise crossed the latent boundary in float32
- **Date**: 2026-08-27
- **Symptom**: MiniMax H3 rejected TDM fake-stage noise because the clean video/audio latents were float16 while conditionally mixed noise was float32.
- **Root Cause**: TDM correctly promoted conditional re-noising and likelihood math to float32 but passed that compute representation directly into the adapter's strict storage boundary.
- **Fix**: TDM now retains float32 mixed/fresh noise for importance weighting and creates a separate adapter-bound noise state cast component-wise to each clean latent's actual dtype and device.
- **Lesson**: Compute precision and boundary representation are separate contracts. Restore representation at the producer boundary instead of weakening adapter validation or casting from global configuration.
- **Related Constraint**: N/A

### H3 cached layout coordinates lost their semantic dtype
- **Date**: 2026-08-30
- **Symptom**: MiniMax H3 offline SFT and DPO rejected cached `position_ids` as float32 even though the preprocessing cache schema declared float64 coordinates.
- **Root Cause**: Hugging Face Dataset's torch formatter defaults floating Arrow columns to float32, and H3 layout normalization preserved that runtime downcast.
- **Fix**: H3 layout normalization now restores matrix coordinates to float64 before output encoding, with a regression covering a float32 cached tensor.
- **Lesson**: An Arrow schema dtype does not guarantee the dtype returned by a runtime formatter. Restore model-semantic precision at the adapter normalization boundary before strict validation.
- **Related Constraint**: #20

## Cross-refs

- UP: [`constraints.md` #18](../constraints.md#18-all-rank-synchronization-points), [`constraints.md` #20](../constraints.md#20-mixed-precision-consistency)
- PEER: [Autocast and Parameter Swaps](autocast_param_swap.md), [Train/Inference Consistency](train_inference_consistency.md), [Timestep and Sigma](timestep_sigma.md)
