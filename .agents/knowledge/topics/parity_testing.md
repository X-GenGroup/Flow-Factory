# Parity Testing

**Read when**: Adding a new adapter, upgrading diffusers, or debugging generation quality.

---

## 4-Layer Test Strategy

| Layer | What | How | Pass criteria |
|-------|------|-----|---------------|
| 1. Config diff | Compare pipeline config against adapter config | Diff `pipeline.config` keys vs adapter `__init__` args | All generation-affecting keys accounted for |
| 2. Component | Test individual components (text encoder, VAE, scheduler) | Feed identical input, compare output tensors | `max_abs_diff < 1e-5` (fp32) or `< 1e-3` (bf16) |
| 3. E2E visual | Full generation with same seed | Run pipeline and adapter with identical seed, prompt, params | Visual match + `max_abs_diff` on final latents < threshold |
| 4. Stage isolation | Test each of the 6 pipeline stages independently | Freeze upstream stages, compare one stage at a time | Per-stage `max_abs_diff < 1e-6` (fp32) |

## Stage Isolation Order (Layer 4)

Test stages in dependency order. When a stage fails, fix it before testing downstream:

1. **Prompt encoding** — `encode_prompt()` vs `pipeline._encode_prompt()`
2. **Latent preparation** — `prepare_latents()` vs pipeline equivalent
3. **Scheduler setup** — `set_timesteps()` output comparison
4. **Single denoise step** — `forward()` with same input vs pipeline's inner loop body
5. **Full denoising loop** — `inference()` vs `pipeline.__call__()` (seed-matched)
6. **VAE decode** — `decode_latents()` vs `pipeline.vae.decode()`

## Flow-Factory Specific Pitfalls (append-only)

1. **`cast_latents()` not applied**: Pipeline doesn't cast; adapter does. If `latent_storage_dtype` is set, parity requires applying `cast_latents()` to pipeline output too before comparison.
2. **Scheduler state not reset**: `scheduler.set_timesteps()` must be called before each comparison run. Leftover `step_index` from a previous run causes different sigma lookups.
3. **`guidance_scale` embedding**: Some models (FLUX, SD3.5) embed `guidance_scale` as a conditioning input, not just a classifier-free guidance weight. Verify the adapter passes it to `forward()`.
4. **Tokenizer padding mismatch**: Pipeline may use `max_length` padding while adapter uses `longest`. Compare `attention_mask` shape and values.
5. **VAE scaling factor**: Pipeline applies `vae.config.scaling_factor` during encode/decode. Adapter must apply the same factor at the same point.
6. **`do_classifier_free_guidance` batching**: Pipeline concatenates conditional + unconditional embeddings along batch dim. Adapter must replicate this exactly if the model expects it.
7. **Timestep offset**: Some schedulers use `timestep_spacing="trailing"` — adapter must match the pipeline's scheduler config exactly.
8. **Dual-modality scheduling**: Models with video+audio (e.g., LTX2-T2AV) need separate scheduler instances. Sharing one scheduler corrupts `step_index` for the second modality.

## `compare_tensors()` Utility

```python
def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5):
    """Compare two tensors and report differences."""
    if a.shape != b.shape:
        raise ValueError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    match = max_diff < atol
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}: max={max_diff:.2e}, mean={mean_diff:.2e}, atol={atol:.0e}")
    if not match:
        worst_idx = diff.argmax().item()
        print(f"  worst at flat index {worst_idx}: a={a.flatten()[worst_idx]:.6f}, b={b.flatten()[worst_idx]:.6f}")
    return match
```

## Cross-refs

- `adapter_conventions.md` (inference/forward identity, upstream alignment rules)
- `dtype_precision.md` (noise dtype for comparison tolerance)
