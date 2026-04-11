# LTX2 Integration — Commit Plan

## Status: Core Complete — GPU Validation In Progress

All adapter code, infrastructure, and example config are implemented and verified
against the official diffusers `LTX2Pipeline`. PR (#118) is open with review fixes
applied.

## Completed (Steps 1–8)

| Area | Key commits | What was built |
|------|-------------|----------------|
| Audio infra | `d8b99e9` | `utils/audio.py`: types, load/save, convert, standardize, hash |
| Samples | `fe63e2a`, `b3cedef` | `audio`+`audio_sample_rate` on `BaseSample`; `T2AVSample`; `_hash_id_fields` refactor; `negative_prompt` in unique ID |
| Data pipeline | `27b5967` | `audio_dir` config, audio loading in `_preprocess_batch` |
| BaseAdapter | `4242d6d`, `a8ebb45` | `audio_vae` property, `encode_audio()` interface, `_resolve_component_names` auto-discovers `nn.Module` components |
| Rewards | `7f11e88` | `audio` param on reward interfaces, `standardize_audio_batch` in processor |
| LTX2 adapter | `b6529a8`–`7b8bb4a`, `6cf5523`–`25f0fcc` | Full `LTX2_T2AV_Adapter`: scaffold, encode_prompt (Gemma3 inline), forward (x0-space multi-guidance), inference loop, decode_latents, registry |
| Logger | `757c7f6` | `T2AVSample` handler, audio-video muxing via PyAV |
| Cross-adapter | `7d2c819` | Unified CFG via `guidance_scale` (removed `do_classifier_free_guidance` bool) across all adapters |
| Example config | `cd45c99` | `examples/grpo/lora/ltx2_t2av.yaml` |
| Bug fixes | `7b8ef98`, `a902429` | `calculate_shift` arg alignment, `audio_sample_rate` using vocoder output rate, PR review fixes (`_check_inputs`, mux guard, ndim validation) |

### Key design decisions

- **Unified latent interface**: `forward()` accepts `cat([video, audio], dim=1)`, splits by `video_seq_len`, runs dual schedulers (video SDE + audio ODE), cats back. Trainers see standard single-modality interface.
- **x0-space guidance**: velocity → x0 → apply CFG + STG + modality isolation deltas → x0 → velocity. Matches official pipeline.
- **sigma from timestep**: `sigma = t / 1000`, no step_idx dependency.
- **Channel dim**: video packed C (128) == audio packed C (8×16 = 128).

## GPU Validation Status

### Verified

- `guidance_scale = 1.0` (no CFG): on-policy step ratio === 1 ✅
- `guidance_scale > 1.0` (CFG enabled): on-policy step ratio === 1 ✅

### TODO — remaining guidance combos

- [ ] `stg_scale > 0`: STG forward with perturbed blocks
- [ ] `modality_scale != 1.0`: modality isolation forward with `isolate_modalities=True`
- [ ] `stg_scale > 0` + `modality_scale != 1.0`: combined STG + modality
- [ ] `guidance_scale > 1.0` + `stg_scale > 0` + `modality_scale != 1.0`: all three

## Next Steps

### Step 9 — Prompt Enhancement (DONE)

Integrated official `LTX2Pipeline.enhance_prompt` into the adapter's `encode_prompt`.

- `LTX2_DEFAULT_SYSTEM_PROMPT` constant with official Lightricks prompt
- `_enhance_prompt_batch()` method with `isolated_rng` context manager for RNG isolation
- `encode_prompt()` accepts `system_prompt` and `prompt_enhancement_seed` params
- YAML: `system_prompt: "default"` / `null` / custom string; `prompt_enhancement_seed: 10`
- Deterministic: same prompt + seed + weights = same result, compatible with dataset cache
- RNG isolation: `isolated_rng(seed)` saves/restores CPU + CUDA RNG state around `torch.manual_seed(seed)`, preventing seed leakage into downstream noise sampling

### Step 10 — Additional example configs (Priority: LOW)

Currently only `examples/grpo/lora/ltx2_t2av.yaml` exists. Other models have
full/nft/awm variants. These can be added after validation:

| Path | Notes |
|------|-------|
| `examples/grpo/full/ltx2_t2av.yaml` | Full finetune — needs FSDP for 3.6B transformer |
| `examples/nft/lora/ltx2_t2av.yaml` | DiffusionNFT — same adapter, different trainer |
| `examples/awm/lora/ltx2_t2av.yaml` | AWM — advantage-weighted model |

These are trivial derivations from the GRPO LoRA config (change `trainer_type` and
algorithm-specific hyperparams). Not blocking.

### Step 11 — Audio reward models (DONE)

Implemented two audio reward models and replaced PickScore in the LTX2 config:

| Reward | File | What it scores |
|--------|------|----------------|
| CLAP (audio-text) | `rewards/clap.py` | Cosine similarity between audio and text embeddings via `transformers.ClapModel` (48 kHz, mono). Zero new deps. |
| ImageBind (AV alignment) | `rewards/imagebind_reward.py` | Cosine similarity between audio mel-spectrogram and video embeddings. Supports `audio_video`, `text_audio`, `text_video`, and `all` modes. CC-BY-NC-SA 4.0. |

- Registry: `clap`, `imagebind` entries added to `_REWARD_MODEL_REGISTRY`
- YAML: `ltx2_t2av.yaml` rewards section updated from PickScore to CLAP + ImageBind
- Audio preprocessing: CLAP resamples to 48 kHz mono via `torchaudio.functional.resample`; ImageBind resamples to 16 kHz mono, splits into 2 s clips, computes mel-spectrogram via `torchaudio.compliance.kaldi.fbank`
- Video preprocessing (ImageBind): temporal subsample 5 clips x 2 frames, resize short side 224, CLIP normalize, 3 spatial crops = 15 views per sample
- `audio_sample_rate` flows from `BaseSample` to reward models via the existing `filter_kwargs` mechanism

### Step 12 — I2AV support (DONE)

Added `LTX2_I2AV_Adapter` for image-conditioned audio-video generation.

- `I2AVSample(ImageConditionSample)` task-level sample in `samples/samples.py`
- `LTX2I2AVSample(I2AVSample)` model-specific sample with `conditioning_mask` and connector fields
- `LTX2_I2AV_Adapter(BaseAdapter)` — flat hierarchy, code-duplicated from T2AV
- `load_pipeline()` loads `LTX2ImageToVideoPipeline` (provides `prepare_latents(image=...)`)
- `encode_image()` preprocesses at generation `(height, width)` — `condition_image_size` in signature for API compat but unused (LTX2 I2V: condition image = first video frame)
- `forward()`: CFG-doubles `conditioning_mask` internally, builds per-token `video_timestep = t * (1 - mask)`, scheduler.step on frames `1:` only, preserves frame 0
- `inference()`: dual-path image input (raw PIL or preprocessed tensor), passes `conditioning_mask` as single-batch to every `forward()` call
- `_enhance_prompt_batch()`: multimodal Gemma3 enhancement with image when available
- Registry: `ltx2_i2av` in `models/registry.py`
- Example: `examples/grpo/lora/ltx2_i2av.yaml`

See `.docs/ltx2-research/I2AV_PLAN.md` for full design.
