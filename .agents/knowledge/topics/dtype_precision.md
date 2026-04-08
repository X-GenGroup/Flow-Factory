# Dtype & Precision

**Read when**: Touching dtype/precision, mixed precision config, or debugging NaN/overflow.

---

## Precision Boundaries

| Component | Runtime dtype | Why |
|-----------|--------------|-----|
| Transformer (frozen) | `inference_dtype` (bf16/fp16) | Memory savings for frozen params |
| Transformer (trainable) | `master_weight_dtype` (fp32/bf16) | Gradient precision |
| Scheduler math | `float32` always | `1/sigma` amplification (see below) |
| Latent storage (trajectory) | `latent_storage_dtype` (configurable) | Memory vs. precision tradeoff |
| Advantage computation | `float64` (numpy) | Normalization stability |

Boundaries are set in `BaseAdapter._mix_precision()` (`models/abc.py` L818) and `BaseTrainer.__init__` (autocast context).

## `cast_latents()` Contract

`BaseAdapter.cast_latents()` (`models/abc.py` L165) casts latents to `latent_storage_dtype` for trajectory storage.

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

Scheduler math uses `1/sigma` to scale noise predictions. Near the end of the denoising schedule, sigma approaches zero and errors are amplified:

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

## Cross-refs

- `constraints.md` #18 (all-rank synchronization — precision errors may manifest differently per rank)
- `constraints.md` #20 (mixed precision consistency)
- `train_inference_consistency.md` (log_prob mismatch from precision)
- `topics/timestep_sigma.md` (scheduler math always float32)
