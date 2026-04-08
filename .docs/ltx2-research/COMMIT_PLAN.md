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

## Key Design Decisions (implemented)

- **Unified latent interface**: forward() accepts `latents = cat([video, audio], dim=1)` and returns a single `FlowMatchEulerDiscreteSDESchedulerOutput`. Internally splits by `video_seq_len`, runs dual schedulers, then cats `next_latents` back. Trainers see a standard single-modality interface with zero framework changes.
- **Channel dim constraint**: Concatenation works because video packed C (128 = 128×1×1×1) == audio packed C (128 = 8×16). Documented in forward() docstring.
- **Video SDE + Audio ODE**: Only video gets stochastic sampling and log_prob for RL; audio uses deterministic ODE. Both schedulers kept separate (matching official pipeline).
- **Separate audio scheduler**: Avoids `step_index` collision with video scheduler.
- **Cache connector outputs**: Connectors are frozen; cache their output (not raw Gemma3 hidden states).
- **CFG in velocity-space**: Matches diffusers 0.38.0.dev0 behavior.
- **LTX2Sample explicit fields**: `num_frames`, `frame_rate`, `video_seq_len` are explicit dataclass fields.

## Remaining Steps

### Step 8 — Example YAML configs

**Status**: Not started. No `*ltx2*` files exist in `examples/`.

Follow existing convention: `examples/{algorithm}/{finetune_type}/{model_name}.yaml`

| Path | Algorithm | Finetune |
|------|-----------|----------|
| `examples/grpo/lora/ltx2_t2av.yaml` | GRPO | LoRA |
| `examples/grpo/full/ltx2_t2av.yaml` | GRPO | Full |
| `examples/nft/lora/ltx2_t2av.yaml` | NFT | LoRA |
| `examples/awm/lora/ltx2_t2av.yaml` | AWM | LoRA |

Each config should specify:
- `model_type: ltx2_t2av`, model path, scheduler params (base_shift=0.95, max_shift=2.05)
- LoRA: target_modules matching `default_target_modules` (28 Linear layers), rank, alpha
- Audio-specific settings (audio_dir, vocoder, audio_vae)
- Video resolution/frames defaults aligned with LTX2 (e.g., 768×512, 121 frames, 24fps)
- Reward model placeholders (video quality, audio quality, AV sync)

Reference existing video model configs (e.g., `grpo/lora/wan22_t2v.yaml`) for structure.

### Step 9 — Audio-video dataset for testing

**Status**: Not started.

**Criteria**:
- Paired video + audio with text captions
- Permissive license (CC-BY, Apache, MIT)
- Manageable size for testing (< 10GB subset)

**Candidates to evaluate**:
- AudioCaps / AudioSet (audio-centric, may need video pairing)
- VGGSound (video + audio with labels)
- VALOR-32K (video-audio-language)
- Panda-70M subset (video + text, may lack audio)

## Deferred features (future PRs)

Features not in current adapter but available in the installed diffusers 0.38.0.dev0:

| Feature | Diffusers Source | Notes |
|---------|-----------------|-------|
| **x0-space guidance** | `pipeline_ltx2.py` L1252+ | Current adapter uses velocity-space CFG. Official pipeline converts to x0-space first (`convert_velocity_to_x0`), applies all guidance deltas (CFG + STG + modality) in x0-space, then converts back (`convert_x0_to_velocity`). Enables cleaner multi-guidance composition. |
| **STG (Spatio-Temporal Guidance)** | `pipeline_ltx2.py` L1298-1335 | Extra transformer forward with `spatio_temporal_guidance_blocks` to perturb specific blocks. Separate `stg_scale` / `audio_stg_scale`. Requires x0-space guidance as prerequisite. |
| **Modality Isolation Guidance** | `pipeline_ltx2.py` L1337-1377 | Extra transformer forward with `isolate_modalities=True` (disables A2V/V2A cross-attn). Separate `modality_scale` / `audio_modality_scale`. Requires x0-space guidance as prerequisite. |
| **Prompt Enhancement** | `pipeline_ltx2.py` L424-467 | Uses Gemma3 `text_encoder.generate()` with system prompt to rewrite user prompts. Needs `processor` (tokenizer + image processor). |
| **I2V/I2AV conditioning** | `pipeline_ltx2_image2video.py` | Image encoded as first-frame video latent + `conditioning_mask`. Channel dim unchanged (128). Compatible with unified latent design. |
| **Latent upsampling** | `pipeline_ltx2_latent_upsample.py` | `adain_filter_latent`, `tone_map_latents` utilities for spatial/temporal upsampling. |
| **Distilled sigma schedule** | `utils.py` -> `DISTILLED_SIGMA_VALUES` | 8-step distilled schedule for faster inference. |
| **Audio SDE optimization** | Already supported by architecture | Switch `audio_scheduler.dynamics_type` from ODE to SDE. Unified latent design means log_prob will naturally cover audio. |

### Note: x0-space vs velocity-space guidance

Our current forward() uses **velocity-space CFG** (simpler, matches earlier diffusers versions):
```
noise_pred = uncond + scale * (cond - uncond)  # in velocity space
```

The latest official pipeline uses **x0-space guidance** (more general):
```
x0_cond = convert_velocity_to_x0(latents, cond_pred, step_idx)
x0_uncond = convert_velocity_to_x0(latents, uncond_pred, step_idx)
x0_guided = x0_cond + cfg_delta + stg_delta + modality_delta  # all deltas in x0 space
velocity = convert_x0_to_velocity(latents, x0_guided, step_idx)
```

Upgrading to x0-space is a prerequisite for STG and Modality Isolation Guidance.

## Housekeeping

- **STEP6_SUBPLAN.md**: Obsolete — Step 6 is fully implemented. Plan diverged from actual implementation (e.g., unified latents vs separate audio trajectory). Can be deleted.
