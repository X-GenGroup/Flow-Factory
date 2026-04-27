# Sample Lifecycle

**Read when**: Touching the `sample()` / `prepare_feedback()` / `optimize()` data flow, debugging `sample()` or `optimize()` OOM, or adding a new high-resolution / video example config.

---

## Default Lifecycle

```
sample()
  for batch in dataloader:
    sample_batch = adapter.inference(...)            # GPU
    self._maybe_offload_samples_to_cpu(sample_batch) # D2H if enabled, else no-op
    samples.extend(sample_batch)
    reward_buffer.add_samples(sample_batch)          # async tasks see deterministic state
prepare_feedback(samples)
  reward_buffer.finalize(...)                        # reward path may H2D per model.device
  compute_advantages(...)
optimize(samples)
  for inner_epoch ...:
    for batch_idx ...:
      batch_samples = [sample.to(device) for sample in slice]   # H2D iff samples on CPU
      batch = BaseSample.stack(batch_samples)
      ...
```

The producer (`_maybe_offload_samples_to_cpu`, `BaseTrainer` in `trainers/abc.py`) and the consumer (`sample.to(device)` per micro-batch in each trainer's `optimize()`) are gated by a single switch.

## Switch: `offload_samples_to_cpu`

Defined on `TrainingArguments` (`hparams/training_args.py`). Default `False`: legacy GPU-resident behaviour. Setting `True` activates the full producer + consumer pipeline.

| Setting | Sample location after `sample()` | `optimize()` per-batch reload | Reward path |
|---------|----------------------------------|-------------------------------|-------------|
| `False` (default) | GPU | `sample.to(device)` is a no-op | `move_tensors_to_device` is a no-op (same device) |
| `True` | CPU | `sample.to(device)` performs H2D for the current micro-batch only | `move_tensors_to_device` performs H2D into `model.device` inside the reward path |

## Adoption Matrix (Example YAMLs)

Three tiers, intentionally deviating from the strict ALL-YAML rule in `.cursor/rules/examples-yaml-sync.mdc`:

| Tier | Models | YAML field | Rationale |
|------|--------|------------|-----------|
| T1 | Wan video (T2V / I2V / V2V, 13 YAMLs) | explicit `true` | per-sample tensors are GB-scale; `sample()`/`optimize()` OOMs without offload |
| T2 | Flux2 + Qwen-Image-Edit-Plus (13 YAMLs) | explicit `false` + multi-line pros/cons comment | moderate VRAM pressure; user-decision point with documentation co-located |
| T3 | FLUX1 / SD3 / Qwen-Image / Z-Image / DPO / AWM-non-Flux2 / template (23 YAMLs) | not added; relies on code default `False` | low pressure; zero-migration cost |

When in doubt, set `True` for any of: latent shape > FLUX1 1024², `num_batches_per_epoch` > 16, full finetune of a model > 7B params.

## Reward Path Device Responsibility

When the offload is enabled, sample tensors arrive at `RewardProcessor` already on CPU. Each of the three reward computation sites does:

```python
batch_input = self._convert_media_format(batch_input, model)
batch_input = move_tensors_to_device(batch_input, model.device)  # H2D into reward model device
output = model(**batch_input)
```

`move_tensors_to_device` (in `utils/base.py`) is a recursive `pytree`-style helper: it walks `list` / `tuple` / `dict` containers and copies `torch.Tensor` leaves to the target device. Non-tensor leaves (PIL, str, int, `np.ndarray`) pass through unchanged. The local `batch_input` dict is reconstructed; sample objects are NOT mutated, so the producer-side CPU residency invariant holds throughout `prepare_feedback()`.

The distributed groupwise path (`_compute_groupwise_distributed`) needs no change: `gather_samples(...)` already accepts `device=self.accelerator.device` and handles any input device.

## Async Reward Race-Free Argument

`reward_buffer.add_samples()` records a CUDA `sync_event` and dispatches workers that read `sample.image` / `sample.video` / etc. Calling `_maybe_offload_samples_to_cpu` BEFORE `add_samples` ensures the recorded event captures "D2H complete + data ready on CPU"; workers wait on the event and then deterministically see CPU-resident samples. The reverse order would race the worker thread's `getattr` against the main thread's in-place `setattr` that `BaseSample.to('cpu')` performs.

## NFT / AWM Precompute Interleave

`optimize()` in `trainers/nft.py` and `trainers/awm.py` follows a per-batch interleave layout (matches the official DiffusionNFT / AWM implementations):

```
for each micro-batch:
  1. lazy reload to GPU + stack
  2. precompute under sampling policy:
       adapter.rollout()
       with sampling_context():
         compute (_all_timesteps, _all_random_noise,
                  _old_v_pred_list / _old_log_probs) for THIS batch only
  3. train under current policy:
       adapter.train()
       with self.autocast():
         per-timestep forward / backward / optimizer step
```

Compared with the previous "eager precompute over ALL batches, then train all batches" design, this caps the precompute footprint to a single batch (`_old_v_pred_list` was 5+ GB on FLUX1 1024² 32-batch in the eager design; tens of GB on Wan).

Train-inference consistency invariant is preserved: `ema_step()` runs once per outer epoch in `start()`, so all batches within a single `optimize()` call see the same EMA snapshot regardless of interleave timing. RNG consumption order changes (per-batch vs upfront), so noise sequences are not bit-identical to the eager design — regression tests must use statistical metrics rather than numeric diffs.

## `extra_kwargs` Device Asymmetry (Caveat)

`BaseSample.to(device, depth=1)` does NOT recurse into `extra_kwargs` (it is a `Dict[str, Any]` field). Actual device residency in the framework data flow:

| Field | Device | Source |
|-------|--------|--------|
| `extra_kwargs['rewards']` | CPU | `reward_processor.py` builds with `device='cpu'` |
| `extra_kwargs['advantage']` | GPU (`accelerator.device`) | `advantage_processor.py` `_to_local` calls `.to(self.accelerator.device)` |

`BaseSample.stack()` flattens `extra_kwargs` into the top-level batch dict via `to_dict()` (see `samples.py`), so `batch['advantage']` is directly accessible at GPU.

Effect of the offload pipeline: `sample.to('cpu')` and `sample.to(device)` both leave `extra_kwargs` untouched. `advantage` remains on GPU regardless of the switch (a few bytes per sample, irrelevant); `rewards` remains on CPU. No change to current behaviour.

If a future custom adapter stores large GPU tensors in `extra_kwargs`, either handle them adapter-side or refactor `BaseSample.to` to delegate to `move_tensors_to_device(value, device, max_depth=1)` in an independent PR (note: that refactor will start moving `extra_kwargs['advantage']` together with the sample, which is benign for the current data flow but is a contract change).

## Cross-refs

- `constraints.md` #11 (BaseTrainer hook order: `sample()` → `prepare_feedback()` → `optimize()`)
- `constraints.md` #14 (BaseSample dataclass hierarchy and `_shared_fields`)
- `constraints.md` #15 + `.cursor/rules/examples-yaml-sync.mdc` (the three-tier strategy is an intentional deviation)
- `topics/train_inference_consistency.md` item #4 (EMA swap without restore — preserved by per-batch interleave)
- `topics/dtype_precision.md` (device-move never changes dtype; orthogonal to autocast)
