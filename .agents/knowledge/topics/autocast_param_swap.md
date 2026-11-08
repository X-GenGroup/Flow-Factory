# Autocast Weight Cache & Parameter Swaps

**Read when**: editing a trainer `optimize()` loop / autocast scope, the param-swap contexts (`use_ref_parameters` / `use_ema_parameters` / `use_named_parameters`), or any KL / ref / EMA / teacher forward.

---

## Invariant: autocast must not span a forward (#20a)

`torch.autocast`'s weight cache (default `cache_enabled=True`) caches each fp32→bf16 weight cast keyed by tensor `data_ptr`, assuming weights are constant in the region. An **in-place** weight change is invisible to it and serves a **stale** cast. Two triggers:

- **Ref/EMA/named swap** — `copy_ema_to` does `param.data.copy_` (same `data_ptr`); a ref forward after the policy forward in one region reuses the policy cast → KL ≈ 0.
- **`optimizer.step()`** — updates weights in place; if the region spans steps, later forwards reuse the pre-step cast → training frozen (loss flat).

**Bites only** for fp32 trainable weights (`trainable_parameters_dtype: fp32`); dormant for the bf16 default (nothing cached); LoRA's `disable_adapter()` ref path is safe (no `.data.copy_`).

**Rule**: wrap **each** forward (and its KL math) in its own `with self.autocast():`; never one autocast around the optimize loop. Per-forward (not `cache_enabled=False`) keeps the legit intra-forward reuse of two-pass-CFG adapters (e.g. `qwen_image_edit_plus.py` calls the transformer twice per forward). Precompute / sampling regions keep their outer autocast (weights constant).

```python
with self.autocast():
    output = self.adapter.forward(**forward_inputs)
if self.enable_kl_loss:
    with self.autocast():
        with torch.no_grad(), self.adapter.use_ref_parameters():
            ref_output = self.adapter.forward(**ref_forward_inputs)
        kl_div = ...; loss = loss + kl_loss
```

## FSDP2 retained and prefetched views

A separate FSDP2 cache can retain full parameter values after forward or prefetch them before a
snapshot copy. Correct authoritative shards do not prove that the next forward reads them.
The preparation owner binds the registry's parameter-write callback: settle an already pending
gather through public `unshard()`, then `reshard()` before installation and restoration. Inspect
backend fields only in that owner. Snapshot writers share the callback and preserve parameter
identities; unwrapped forwards must not bypass the prepared root. Scopes between forwards may
retain a pending policy graph; writes inside an executing FSDP forward/backward fail before copy.
See [parameter state](../../../docs/development/parameter-state.md) and the
[fix record](../fixes/distributed.md#snapshot-writes-must-invalidate-retained-fsdp2-parameter-views).

## DDP/DeepSpeed caveat (separate mechanism)

`.data.copy_` may not reach the params a wrapped forward reads: vanilla DDP shares params (swap reflected); DeepSpeed may hold a working copy. If a ref/EMA forward returns the policy output even with per-forward autocast, run the no-grad swap forward on the unwrapped component (`get_component_unwrapped` / `set_component`). ZeRO-3 unsupported (#10).

## In-place swap, not separate modules

Ref/EMA/named snapshots share **one** model and swap in place via `EMAModuleWrapper.copy_ema_to`; do not replace with a separate frozen module. Full-FT is often partial (`_freeze_components` unfreezes only `target_modules`), so the swap duplicates only the trainable subset T and shares frozen F in place — memory-optimal; a `deepcopy` module would hold T+F (wastes F). `named_parameters` also stores N snapshots cheaply (flat CPU tensors, one GPU swap at a time). Cost: `copy_ema_to` does ~3×T copies per KL step (incl. a D2H temp clone) — if it bottlenecks, optimize the swap, not the architecture.

## Cross-refs

- UP: [`constraints.md` #20a](../constraints.md#20a-autocast-weight-cache-must-not-span-a-forward), [`constraints.md` #20](../constraints.md#20-mixed-precision-consistency), [`constraints.md` #10](../constraints.md#10-deepspeed-zero-3-is-unsupported)
- PEER: [Dtype and Precision](dtype_precision.md), [Train/Inference Consistency](train_inference_consistency.md)
- CODE: [`BaseAdapter` parameter-swap contexts](../../../src/flow_factory/models/abc.py), [`EMAModuleWrapper`](../../../src/flow_factory/ema/ema.py)
