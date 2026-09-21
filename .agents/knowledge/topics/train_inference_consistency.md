# Train-Inference Consistency

**Read when**: Touching `trainers/*.optimize()`, `adapter.forward()`, `adapter.inference()`, or `scheduler.step()`.

---

## The Atomic Unit: `adapter.forward()`

`forward()` is the single function that must produce identical output given identical input across rollout and training. During rollout, `inference()` calls `forward()` per denoising step; during training, `optimize()` replays each step by calling `forward()` with the same arguments.

Train-inference consistency means: **same `forward()` inputs -> same `forward()` outputs**.

ALL arguments that affect `forward()` output must be preserved between rollout and training:

- **Latent state**: `latents`, `next_latents`, `t`, `t_next` (stored in trajectory)
- **Model conditioning**: `prompt_embeds`, `connector_prompt_embeds`, `guidance_scale`, `stg_scale`, etc. (stored on Sample dataclass, replayed from there)
- **Scheduler config**: `noise_level`, `dynamics_type` (must not change between phases)
- **Precision**: `cast_latents()` applied identically in both paths (-> `dtype_precision.md`)

## The Invariant

Before any gradient update: `ratio = exp(new_log_prob - old_log_prob) == 1.0` (up to float precision).

```
Rollout:  inference() -> forward(args) -> scheduler.step(next_latents=None)    -> sample & store
Training: optimize()  -> forward(same args) -> scheduler.step(next_latents=STORED) -> recompute log_prob
```

If rollout and training `forward()` diverge, `ratio` deviates from 1.0 at epoch start and the policy gradient is wrong.

## What Breaks It

1. **Different forward arguments**: `guidance_scale`, `noise_level`, or prompt embeddings differ between rollout and training.
2. **Different `noise_level`**: rollout uses one value, training uses another.
3. **Inconsistent `cast_latents()`**: rollout stores bf16 latents, training reloads as float32.
4. **Model weight change**: EMA swap without restore between `inference()` and first `forward()` call.
5. **Scheduler state mismatch**: `step_index` not matching (e.g., dual-scheduler models).
6. **`num_inference_steps` changed**: invalidates sigma schedule, all trajectory timesteps are wrong.
7. **Batch/pack composition mismatch (pack-dependent adapters)**: For adapters whose batched `forward()` is *pack-composition-dependent* (e.g. Bagel NaViT sequence packing, where a sample's linear-projection matmuls run over the concatenated `sum_seqlen` of the whole pack), bf16 rounding depends on *which* samples share the pack. If a training micro-batch packs a different sample set than the corresponding rollout pack, the on-policy `forward()` is no longer bit-identical -> `ratio != 1` (~1e-4) even though every stored argument matches. Per-sample (B=1) adapters are immune. The trigger is the optimize-time sample shuffle reordering `samples` before chunking into micro-batches.
8. **Stochastic conditioning encoder recomputed per forward**: When `forward()` *rebuilds* its conditioning from raw inputs each call (rather than replaying a stored embedding) and that encoder is non-deterministic, rollout and training diverge. Bagel I2I is the canonical case: the condition-image VAE (`DiagonalGaussian`, default `sample=True` -> `mean + std*randn`) is encoded **once** at rollout but **re-encoded every training `forward()`** (`_forward_rebuild` -> `_build_gen_context`), so each draws different noise -> different KV context -> on-policy `ratio != 1` (~2e-4). T2I is immune (text-only context, deterministic). Fix: make the condition encode deterministic (`vae.reg.sample = False`, posterior mean) or cache the rollout encoding and replay it. (Only affects `vae.encode` of conditions; generation uses init noise and `vae.decode` is unaffected.) **Bagel applies this fix**: `pipeline.vae.reg.sample = False` in `BagelAdapter.load_pipeline()`.
9. **Preference arms replayed with the wrong batch**: DPO shares forward-process noise across chosen/rejected states, but each arm still owns its conditioning batch. Both policy and reference forwards for the rejected state must receive `rejected_batch`; pairing it with `chosen_batch` evaluates the rejected trajectory under another conditioning context.
10. **Condition realization redrawn per candidate**: Geometry-bound VAE conditions and stochastic
    condition prefixes belong to the input, not to a demonstration/preference candidate. Offline
    training must call `prepare_condition_state()` once per batch and reuse the resulting tensor
    leaves for every target arm and policy/reference forward. MiniMax H3 FL2VA/Ref2VA is the
    stochastic reference: its official condition augmentation must be drawn once before target
    noise. Wan I2V and LTX2 I2AV use deterministic posterior mode but follow the same ownership
    boundary so mask/layout binding cannot drift between candidates.
11. **Model-specific velocity direction assumed by a trainer**: Standard flow adapters predict noise-ward velocity (`noise - clean`), while MiniMax H3 predicts data-ward velocity (`clean - noise`). Any `x0` target must use `adapter.project_velocity_to_clean_state()` rather than spelling `xt - sigma * velocity` inside a trainer.

## Pack-composition-dependent adapters: `shuffle_samples`

`samples` reaches `optimize()` in rollout-pack order (`generate_samples` appends each `inference` pack in order; `compute_advantages(store_to_samples=True)` attaches advantages in place without reordering), so `samples[i*B:(i+1)*B]` is exactly rollout pack `i`. Set `train.shuffle_samples: false` so the optimize loop keeps that order: each training micro-batch then packs the *same* samples as its rollout pack -> deterministic bf16 -> on-policy `ratio == 1`.

- Requires matched sampling and training `per_device_batch_size` (so contiguous chunks reproduce the rollout packs). These are the *same* field: rollout uses `training_args.per_device_batch_size` for the train DataLoader (`data_utils/loader.py`) and `optimize()` chunks by the same value, so they cannot drift unless a separate sampling batch size is introduced.
- Default is `True` (per-inner-epoch shuffle); only pack-dependent adapters (Bagel) need `false`. The off-policy decorrelation cost is minor because the rollout order is already sampler-randomized. `BagelAdapter` warns at init if `shuffle_samples` is left `True`.
- Wired via `BaseTrainer._order_samples_for_optimize(samples, inner_epoch)` (used by grpo/dppo/nft/awm/opd; dpo shuffles chosen/rejected pairs separately).

## Where in Code

- Rollout: `adapter.inference()` -> `forward()` -> `scheduler.step()` -> `sample.log_probs[i]`
- Training: `Trainer.optimize()` -> `adapter.forward()` -> `output.log_prob`
- Ratio: `trainers/rl/grpo.py` (`GRPOTrainer.optimize`): `ratio = torch.exp(output.log_prob - old_log_prob)`
- PPO clip: `max(-adv * ratio, -adv * clamp(ratio, 1-eps, 1+eps))`
- Dtype round-trip guard: `scheduler/*.py` — `next_latents = next_latents.to(_input_dtype).float()` ensures stored trajectory matches training replay (grep the expression in `scheduler/flow_match_euler_discrete.py` and `scheduler/unipc_multistep.py`; it appears once per dynamics branch)
- `cast_latents()`: `BaseAdapter.cast_latents()` (`models/abc.py`) — applied identically in `inference()` before/after each `forward()` call

## Fix Records

### Stabilize exact on-policy log-prob reductions
- **Date**: 2026-09-21
- **Symptom**: Qwen-Image 2.1 rollout and replay produced bit-identical velocities, next latents, and elementwise transition statistics, but `ratio_mean` could still differ from `1` by one ULP.
- **Root Cause**: CUDA FP32 parallel reduction can choose a different summation tree for otherwise identical tensors at different memory addresses.
- **Fix**: `scheduler/flow_match_euler_discrete.py` now accumulates per-sample transition log-prob means in FP64 and casts the result back to the input dtype; the regression test checks both exact values and preserved gradients.
- **Lesson**: Exact coupled-policy parity must include the final scheduler reduction, not only transformer and trajectory tensors; use a stable accumulator for scalar statistics that become PPO old/new log-probs.
- **Related Constraint**: Constraint #7

## Cross-refs

- UP: [`constraints.md` #7](../constraints.md#7-coupled-vs-decoupled-paradigm), [Architecture Execution Pipelines](../architecture.md#execution-pipelines)
- PEER: [Dtype and Precision](dtype_precision.md), [Adapter Conventions](adapter_conventions.md)
