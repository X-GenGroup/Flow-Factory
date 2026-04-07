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
        Step 6   ← NEXT (see STEP6_SUBPLAN.md)
```

## Completed Steps

| Step | Commit | Files | Summary |
|------|--------|-------|---------|
| 1 | `d8b99e9` | `utils/audio.py` (new), `utils/base.py` | Audio utility module: types, validation, load/save, convert, standardize, hash |
| 2 | `fe63e2a` | `samples/samples.py`, `samples/__init__.py` | `audio` field on `BaseSample`, `T2AVSample` class |
| 3 | `27b5967` | `hparams/data_args.py`, `data_utils/dataset.py` | `audio_dir` config, audio loading in `_preprocess_batch` |
| 4 | `4242d6d` | `models/abc.py` | `encode_audio()` default, `audio_vae` property, `preprocess_func` audio routing, `_freeze_vae` audio |
| 5 | `7f11e88` | `rewards/abc.py`, `rewards/reward_processor.py` | `audio` param on reward interfaces, `MEDIA_FIELDS`, `standardize_audio_batch` in convert |

All framework-level audio support is complete. Steps 1–5 are fully backward compatible — no existing model/reward/dataset behavior is changed.

## Remaining: Step 6 — LTX2 Adapter

Detailed sub-plan in **STEP6_SUBPLAN.md** (v4, verified against diffusers 0.38.0.dev0).

4 sub-commits:

| Sub | Scope | Key Files |
|-----|-------|-----------|
| 6a | Scaffold: `LTX2Sample` + adapter skeleton + `load_pipeline` + `_create_audio_scheduler` | `models/ltx2/__init__.py`, `models/ltx2/ltx2_t2av.py` (new) |
| 6b | `encode_prompt()` (pipeline.encode_prompt → connectors with additive mask) + `decode_latents()` (video: unpack→denorm→VAE; audio: denorm→unpack→VAE→vocoder) | `models/ltx2/ltx2_t2av.py` |
| 6c | `forward()`: CFG in velocity-space + dual scheduler step (video SDE + audio ODE) | `models/ltx2/ltx2_t2av.py` |
| 6d | `inference()`: full loop + trajectory collection + registry entry | `models/ltx2/ltx2_t2av.py`, `models/registry.py` |

### Key design decisions

- **Video SDE + Audio ODE**: Only video gets stochastic sampling and log_prob for RL optimization; audio uses deterministic ODE
- **Separate audio scheduler**: Avoids `step_index` collision with video scheduler
- **Cache connector outputs**: Connectors are frozen, so cache their output (not raw Gemma hidden states)
- **CFG in velocity-space**: Matches installed diffusers 0.38.0.dev0 behavior
- **Audio latents detached during training**: Gradients flow only through video pathway

### Deferred features (future PRs)

- STG / Modality Isolation Guidance (requires transformer param upgrade)
- Prompt enhancement via Gemma3 generate (requires `enhance_prompt` method)
- x0-space guidance (requires `convert_velocity_to_x0` helpers)
- I2V conditioning variant
- Latent upsampling / distilled sigmas
