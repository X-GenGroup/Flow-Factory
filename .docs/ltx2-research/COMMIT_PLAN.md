# LTX2 Integration — Step-by-Step Commit Plan

## Dependency Graph

```
Step 1: utils/audio.py (standalone)
   │
   ├──► Step 2: samples/samples.py (depends on audio types from Step 1)
   │       │
   │       ├──► Step 4: models/abc.py (depends on audio field in Sample)
   │       │       │
   │       │       └──► Step 6: models/ltx2/ (depends on all framework changes)
   │       │
   │       └──► Step 5: rewards/ (depends on audio field in Sample)
   │
   └──► Step 3: data_utils/ + hparams/ (depends on audio loading from Step 1)
               │
               └──► Step 6: models/ltx2/ (depends on audio_dir in data pipeline)
```

---

## Step 1 — Audio utility module

**New file**: `src/flow_factory/utils/audio.py`
**Edit**: `src/flow_factory/utils/base.py`

**Scope**: Pure utility, zero dependencies on framework internals.

**Design references**:
- Module structure mirrors `utils/image.py` and `utils/video.py` (type hierarchy docstring, type aliases, validation, conversion, standardization, hashing sections)
- Tensor convention `(C, T)` follows torchaudio / audiocraft / diffusers standard (channel-first, time-last)
- Value range `[-1.0, 1.0]` float32 follows diffusers AudioPipelineOutput and LTX2 vocoder convention
- `load_audio` / `save_audio` use a 3-tier fallback chain: torchaudio → soundfile → stdlib `wave` module (always available)
- `convert_audio(waveform, from_rate, to_rate, to_channels)` follows audiocraft's convention for channel conversion (downmix = mean, upmix = repeat)
- `hash_audio` quantizes to int16 (matches WAV precision) before hashing, following `hash_tensor` pattern
- `standardize_audio_batch` has `output_type: Literal['np', 'pt']` (no 'pil' — audio has no PIL equivalent)
- Resampling fallback chain: torchaudio.functional.resample → scipy.signal.resample_poly → torch.nn.functional.interpolate

| Action | File | Detail |
|--------|------|--------|
| Create | `utils/audio.py` | `AudioSingle`, `AudioBatch` type aliases; `is_audio()`, `is_audio_batch()` validators; `load_audio()`, `save_audio()` with 3-tier backend fallback; `audio_to_tensor()`, `audio_to_numpy()`, `convert_audio()` converters; `standardize_audio_batch()` standardizer; `hash_audio()`, `hash_audio_list()` hashers |
| Edit | `utils/base.py` | Add `from .audio import *` (mirrors existing `from .image import *` / `from .video import *`) |

**Commit message**: `[utils] feat: add audio utility module with type aliases, standardization, and loading`

**Verifiable**: Import `from flow_factory.utils.base import AudioSingle, load_audio` succeeds. Full test suite (7 groups: validation, conversion, convert_audio, standardize, hashing, load/save round-trip, error handling) passes.

---

## Step 2 — Audio field on BaseSample

**Edit**: `src/flow_factory/samples/samples.py`
**Edit**: `src/flow_factory/samples/__init__.py`

**Scope**: Add `audio` field to `BaseSample`; add `T2AVSample` class.

| Action | File | Detail |
|--------|------|--------|
| Edit | `samples/samples.py` | Add `audio: Optional[torch.Tensor] = None` field on `BaseSample` with `__post_init__` standardization. Add `T2AVSample(BaseSample)` as a semantic alias (like `T2VSample`). |
| Edit | `samples/__init__.py` | Export `T2AVSample` |

**Commit message**: `[samples] feat: add audio field to BaseSample and T2AVSample class`

**Verifiable**: `BaseSample(audio=torch.randn(1, 16000))` constructs without error; `T2AVSample` importable.

---

## Step 3 — Data pipeline audio support

**Edit**: `src/flow_factory/hparams/data_args.py`
**Edit**: `src/flow_factory/data_utils/dataset.py`

**Scope**: Enable loading audio files from disk and passing them through the preprocessing pipeline.

| Action | File | Detail |
|--------|------|--------|
| Edit | `hparams/data_args.py` | Add `audio_dir: Optional[str] = None` field |
| Edit | `data_utils/dataset.py` | 1) Add `AudioEncodeCallable` protocol. 2) Update `PreprocessCallable` to include `audios` param. 3) Add `audio_dir` to `__init__`, `_load_raw_dataset` default path logic, `_preprocess_batch` fn_kwargs. 4) Implement audio loading block in `_preprocess_batch` (mirrors video block). 5) Add `'audios'` to `PREPROCESS_KEYS`. |

**Commit message**: `[data] feat: support audio loading and preprocessing in dataset pipeline`

**Verifiable**: A JSONL with `"audio": "test.wav"` and a corresponding wav file can be preprocessed without error (even if `encode_audio` returns None — the raw tensor is cached).

---

## Step 4 — BaseAdapter audio encoding interface

**Edit**: `src/flow_factory/models/abc.py`

**Scope**: Add audio-related methods and properties to the base adapter class.

| Action | File | Detail |
|--------|------|--------|
| Edit | `models/abc.py` | 1) Add `encode_audio()` default method (returns None). 2) Add `audio_vae` property (auto-detect from pipeline). 3) Update `preprocess_func()` to route `audios` → `encode_audio()`. 4) Update `_freeze_components()` to freeze `audio_vae` if present. |

**Commit message**: `[models] feat: add encode_audio interface and audio_vae support to BaseAdapter`

**Verifiable**: All existing adapters still work (encode_audio returns None, audio_vae returns None). `preprocess_func(prompt=["test"], audios=[torch.randn(1,16000)])` runs without error on any existing adapter.

---

## Step 5 — Reward model audio interface

**Edit**: `src/flow_factory/rewards/abc.py`
**Edit**: `src/flow_factory/rewards/reward_processor.py`

**Scope**: Allow reward models to receive audio data.

| Action | File | Detail |
|--------|------|--------|
| Edit | `rewards/abc.py` | Add `audio: Optional[List[torch.Tensor]] = None` parameter to `PointwiseRewardModel.__call__` and `GroupwiseRewardModel.__call__` |
| Edit | `rewards/reward_processor.py` | Add `'audio'` to `MEDIA_FIELDS`. Add audio handling in `_convert_media_format()` (passthrough as tensor or convert to numpy). |

**Commit message**: `[rewards] feat: add audio parameter to reward model interfaces`

**Verifiable**: Existing reward models ignore `audio` via `filter_kwargs`. RewardProcessor can handle samples with `.audio` set.

---

## Step 6 — LTX2 adapter implementation

**New files**: `src/flow_factory/models/ltx2/__init__.py`, `src/flow_factory/models/ltx2/ltx2_t2av.py`
**Edit**: `src/flow_factory/models/registry.py`

**Scope**: Complete LTX2 model adapter — the only step that touches model-specific logic.

| Action | File | Detail |
|--------|------|--------|
| Create | `models/ltx2/__init__.py` | Package init |
| Create | `models/ltx2/ltx2_t2av.py` | `LTX2Sample(T2AVSample)` dataclass, `LTX2_T2AV_Adapter(BaseAdapter)` with: `load_pipeline`, `load_scheduler` (+ ODE audio_scheduler), `default_target_modules`, `preprocessing_modules`, `inference_modules`, `encode_prompt` (Gemma3 + Connectors), `encode_audio`, `decode_latents` (video + audio), `inference` (joint denoising loop, video SDE + audio ODE), `forward` (single step, video log_prob only) |
| Edit | `models/registry.py` | Add `'ltx2_t2av': 'flow_factory.models.ltx2.ltx2_t2av.LTX2_T2AV_Adapter'` |

**Commit message**: `[models] feat: add LTX2 text-to-audio-video adapter with video-only SDE optimization`

**Verifiable**: `get_model_adapter_class('ltx2_t2av')` resolves. Full integration test with actual model weights.

---

## Summary

| Step | Commit | Files Modified | Files Created | Depends On |
|------|--------|---------------|---------------|------------|
| 1 | `[utils] audio module` | 1 | 1 | — |
| 2 | `[samples] audio field` | 2 | 0 | Step 1 |
| 3 | `[data] audio pipeline` | 2 | 0 | Step 1 |
| 4 | `[models] encode_audio` | 1 | 0 | Step 1, 2 |
| 5 | `[rewards] audio param` | 2 | 0 | Step 2 |
| 6 | `[models] LTX2 adapter` | 1 | 2 | Step 1–5 |

Steps 2, 3 can be done in parallel (independent). Steps 4, 5 can be done in parallel after their dependencies are met. Step 6 requires all previous steps.

```
        Step 1
       ┌──┼──┐
       ▼  ▼  ▼
    Step2 Step3
       │  │
       ▼  ▼
    Step4 Step5
       │  │
       └──┼──┘
          ▼
        Step 6
```
