# LTX2 Integration — Commit Plan

## Progress

```
        Step 1 ✅ d8b99e9
       ┌──┼──┐
       ▼  ▼  ▼
    Step2 Step3  ✅ fe63e2a, 27b5967
       │  │
       ▼  ▼
    Step4 Step5  ✅ 4242d6d, 7f11e88
       │  │
       └──┼──┘
          ▼
        Step 6   ✅ b6529a8, 93df1d9, b26fd56, 7b8bb4a
          │
          ▼
        Step 7   ✅ 6cf5523, 880ed32, 193f8db
          │
          ▼
      Step 8–9   ← NEXT
```

## Completed Steps

| Step | Commit | Files | Summary |
|------|--------|-------|---------|
| 1 | `d8b99e9` | `utils/audio.py` (new), `utils/base.py` | Audio utility module: types, validation, load/save, convert, standardize, hash |
| 2 | `fe63e2a` | `samples/samples.py`, `samples/__init__.py` | `audio` field on `BaseSample`, `T2AVSample` class |
| 3 | `27b5967` | `hparams/data_args.py`, `data_utils/dataset.py` | `audio_dir` config, audio loading in `_preprocess_batch` |
| 4 | `4242d6d` | `models/abc.py` | `encode_audio()` default, `audio_vae` property, `preprocess_func` audio routing, `_freeze_vae` audio |
| 5 | `7f11e88` | `rewards/abc.py`, `rewards/reward_processor.py` | `audio` param on reward interfaces, `MEDIA_FIELDS`, `standardize_audio_batch` in convert |
| 6a | `b6529a8` | `models/ltx2/__init__.py`, `models/ltx2/ltx2_t2av.py` (new) | LTX2Sample + adapter scaffold + load_pipeline + audio scheduler |
| 6b | `93df1d9` | `models/ltx2/ltx2_t2av.py` | encode_prompt (Gemma3 + connectors + additive mask) + decode_latents (video/audio) |
| 6c | `b26fd56` | `models/ltx2/ltx2_t2av.py` | forward() with CFG + dual scheduler step (video SDE + audio ODE) |
| 6d | `7b8bb4a` | `models/ltx2/ltx2_t2av.py`, `models/registry.py` | inference() full loop + registry entry `ltx2_t2av` |
| 7a | `6cf5523` | `models/ltx2/ltx2_t2av.py` | Type safety: `FlowMatchEulerDiscreteSDESchedulerOutput`, remove unused imports, `self.scheduler` type |
| 7b | `880ed32` | `models/ltx2/ltx2_t2av.py` | Promote `num_frames`, `frame_rate`, `video_seq_len` to explicit LTX2Sample fields |
| 7c | `193f8db` | `models/ltx2/ltx2_t2av.py` | Unified latent interface: forward() accepts cat(video, audio), splits/steps/cats internally, returns single output |
| 7d | `25f0fcc` | `models/ltx2/ltx2_t2av.py` | x0-space multi-guidance (CFG+STG+Modality Isolation), sigma=timestep, prompt enhancement, independent audio guidance, sigma-based x0 conversion (no step_idx) |

## Key Design Decisions (implemented)

- **inference() = Pipeline.__call__()**: Full generation loop with trajectory collection for RL. forward() is called per-step.
- **forward() = single denoising step**: Transformer forward + x0-space multi-guidance (CFG + STG + Modality Isolation) + dual scheduler.step. Called by both inference() loop and trainer optimize(). Returns single `FlowMatchEulerDiscreteSDESchedulerOutput`.
- **Unified latent interface**: forward() accepts `latents = cat([video, audio], dim=1)`, splits by `video_seq_len`, runs dual schedulers, cats `next_latents` back. Trainers see standard single-modality interface.
- **x0-space guidance**: velocity -> x0 -> apply all deltas (CFG + STG + modality) -> x0 -> velocity. Matches official pipeline.
- **sigma from timestep**: `sigma = t / 1000`, no step_idx dependency. convert_velocity_to_x0/convert_x0_to_velocity use sigma tensor directly.
- **Channel dim constraint**: video packed C (128) == audio packed C (8*16=128). Documented in forward() docstring.
- **Video SDE + Audio ODE**: Dual schedulers (separate step_index). Audio uses deterministic ODE.
- **LTX2Sample fields**: `num_frames`, `frame_rate`, `video_seq_len` as explicit fields; `duration_s` in extra_kwargs.

## Remaining Steps

### Step 8a — Inference alignment refinements

**Status**: Not started. Gaps between inference() and official __call__:

| Gap | Fix |
|-----|-----|
| **Input validation** | Add `_check_inputs()`: height/width divisible by vae_spatial, STG blocks required when stg_scale > 0 |
| **num_frames rounding** | Round `(num_frames - 1) % vae_temporal != 0` to nearest valid value |
| **Coord pre-duplication** | Move CFG coord duplication from forward() to inference() (before loop), pass pre-duplicated coords |
| **Distilled sigmas** | Check `pipeline.transformer.config` for distilled sigma schedule, use if available |
| **Embed concatenation** | In inference(), pass already-concatenated [neg, pos] connector embeds to forward() for CFG efficiency (forward() currently re-cats every step) |

**Scope**: Only `ltx2_t2av.py` changes. No trainer changes.

### Step 8b — Example YAML configs

**Status**: Not started. No `*ltx2*` files exist in `examples/`.

Follow existing convention: `examples/{algorithm}/{finetune_type}/{model_name}.yaml`

| Path | Algorithm | Finetune |
|------|-----------|----------|
| `examples/grpo/lora/ltx2_t2av.yaml` | GRPO | LoRA |
| `examples/grpo/full/ltx2_t2av.yaml` | GRPO | Full |
| `examples/nft/lora/ltx2_t2av.yaml` | NFT | LoRA |
| `examples/awm/lora/ltx2_t2av.yaml` | AWM | LoRA |

Key settings per config:
- `model_name_or_path: Lightricks/LTX-2`, `model_type: ltx2_t2av`
- LoRA: rank=16, alpha=16, target_modules from `default_target_modules` (28 Linear layers)
- Scheduler: base_shift=0.95, max_shift=2.05, dynamics_type=Flow-SDE
- Eval: guidance_scale=4.0, num_inference_steps=40, height=768, width=512, num_frames=121, frame_rate=24
- Rewards: placeholder for video quality + audio quality + AV sync
- Reference: `grpo/lora/wan22_t2v.yaml` for structure

### Step 9 — Audio-video test dataset

**Status**: Not started.

**Recommended**: VGGSound-50k (HuggingFace `Gray1y/VGGSound-50k`)
- License: CC-BY 4.0
- Size: ~49K clips, <5GB
- Content: 10s video clips with class labels + synchronized audio
- Integration: extract audio to wav, convert labels to prompts, structure as video_dir + audio_dir + metadata.jsonl

**Alternative**: VALOR-32K (higher-quality human captions, but harder to access)

## Deferred features (future PRs)

| Feature | Diffusers Source | Notes |
|---------|-----------------|-------|
| **I2V/I2AV conditioning** | `pipeline_ltx2_image2video.py` | Image encoded as first-frame video latent + conditioning_mask. Compatible with unified latent design. |
| **Latent upsampling** | `pipeline_ltx2_latent_upsample.py` | `adain_filter_latent`, `tone_map_latents` utilities. |
| **Distilled sigma schedule** | `utils.py` -> `DISTILLED_SIGMA_VALUES` | 8-step distilled schedule for faster inference. |
| **Audio SDE optimization** | Already supported by architecture | Switch `audio_scheduler.dynamics_type` from ODE to SDE. |

## Housekeeping

- **STEP6_SUBPLAN.md**: Deleted (obsolete).
- **COMMIT_PLAN.md**: This file is the single source of truth for the LTX2 integration.

## Housekeeping

- **STEP6_SUBPLAN.md**: Obsolete — Step 6 is fully implemented. Plan diverged from actual implementation (e.g., unified latents vs separate audio trajectory). Can be deleted.
