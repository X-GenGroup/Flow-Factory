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

## Step 1 — Audio utility module ✅ `d8b99e9`

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

---

## Step 2 — Audio field on BaseSample ✅ `fe63e2a`

**Edit**: `src/flow_factory/samples/samples.py`
**Edit**: `src/flow_factory/samples/__init__.py`

**Scope**: Add `audio` field to `BaseSample`; add `T2AVSample` class.

| Action | File | Detail |
|--------|------|--------|
| Edit | `samples/samples.py` | Add `audio: Optional[torch.Tensor] = None` field on `BaseSample` with `__post_init__` standardization via `audio_to_tensor`. Add `T2AVSample(BaseSample)`. |
| Edit | `samples/__init__.py` | Export `T2AVSample` |

---

## Step 3 — Data pipeline audio support ✅ `27b5967`

**Edit**: `src/flow_factory/hparams/data_args.py`
**Edit**: `src/flow_factory/data_utils/dataset.py`

**Scope**: Enable loading audio files from disk and passing them through the preprocessing pipeline.

| Action | File | Detail |
|--------|------|--------|
| Edit | `hparams/data_args.py` | Add `audio_dir: Optional[str] = None` field |
| Edit | `data_utils/dataset.py` | Add `audio_dir` to `__init__` / `_load_raw_dataset` / `fn_kwargs`. Audio loading block in `_preprocess_batch` (mirrors video pattern). `'audios'` added to `PREPROCESS_KEYS`. |

---

## Step 4 — BaseAdapter audio encoding interface ✅ `4242d6d`

**Edit**: `src/flow_factory/models/abc.py`

**Scope**: Add audio-related methods and properties to the base adapter class.

| Action | File | Detail |
|--------|------|--------|
| Edit | `models/abc.py` | 1) `encode_audio()` default method (returns None, non-abstract). 2) `audio_vae` property with getter/setter. 3) `preprocess_func()` routes `audios` → `encode_audio()`. 4) `_freeze_vae()` also freezes `audio_vae`. |

---

## Step 5 — Reward model audio interface ← **NEXT**

**Edit**: `src/flow_factory/rewards/abc.py`
**Edit**: `src/flow_factory/rewards/reward_processor.py`

**Scope**: Allow reward models to receive audio data as a **separate field** alongside video.

**Design Decision — `separate` vs `muxed` audio+video**:

| Approach | Description | Verdict |
|----------|-------------|---------|
| A. **Separate fields** | `video=List[frames]`, `audio=List[Tensor(C,T)]` passed independently | **Chosen** |
| B. Muxed mp4 | Merge audio+video into temp mp4, pass file paths | Rejected — IO overhead, PyAV dependency, breaks "field=modality" design |
| C. User-configurable | Per-model `audio_video_format` flag | Rejected — over-engineered |

Rationale for **separate fields**:
1. Consistent with existing field-per-modality design
2. Zero overhead for non-audio rewards (`filter_kwargs` auto-strips `audio`)
3. Composable: pure-visual, pure-audio, and cross-modal rewards all work
4. No IO: no temp files, no muxing/demuxing
5. Extensible: future muxed-video reward models can mux internally

### Detailed changes

#### 5a. `rewards/abc.py`

Add `audio` parameter to **both** abstract `__call__` signatures (PointwiseRewardModel and GroupwiseRewardModel). Insert after `video`, before `condition_images`:

```python
def __call__(
    self,
    prompt: List[str],
    image: Optional[List[Image.Image]] = None,
    video: Optional[List[List[Image.Image]]] = None,
    audio: Optional[List[torch.Tensor]] = None,  # NEW: List of (C, T) waveforms
    condition_images: Optional[List[List[Image.Image]]] = None,
    condition_videos: Optional[List[List[List[Image.Image]]]] = None,
    **kwargs,
) -> RewardModelOutput:
```

Add `torch` import at top of file (currently only uses torch indirectly via type hints).

Update docstrings for both classes:
```
audio: Optional list of audio waveforms. Each element is a torch.Tensor
    of shape (C, T), float32 in [-1, 1]. If `use_tensor_inputs` is True,
    same format. If False, each element is an np.ndarray (C, T).
```

Note: `BaseRewardModel.__call__` has `*args, **kwargs` so needs no change.

#### 5b. `rewards/reward_processor.py`

Two changes:

1. **`MEDIA_FIELDS`**: Add `'audio'` to the set:
   ```python
   MEDIA_FIELDS = {'image', 'video', 'audio', 'condition_images', 'condition_videos'}
   ```

2. **`_convert_media_format`**: Add `elif k == 'audio':` branch. Audio has no PIL
   representation, so:
   - `use_tensor_inputs=True` (output_type='pt') → passthrough
   - `use_tensor_inputs=False` (output_type='pil') → convert each tensor to numpy

   ```python
   elif k == 'audio':
       # Audio is always tensor-based; convert to numpy for non-tensor models
       if output_type == 'pt':
           result[k] = v
       else:
           result[k] = [
               a.cpu().numpy() if isinstance(a, torch.Tensor) else a
               for a in v
           ]
   ```

   Note: the variable `output_type` is already computed at top of the method ('pt' or 'pil').
   For audio, 'pil' means "non-tensor" → numpy is the appropriate fallback.

### Backward compatibility

- **Existing reward models** (PickScore, CLIP, OCR, VLM) don't have `audio` in their
  concrete `__call__` signatures. `filter_kwargs(model.__call__, ...)` will strip it.
  No changes needed in any existing reward model.
- **`required_fields`**: existing models list `("prompt", "image", "video")`. The `audio`
  field on `BaseSample` won't be gathered unless a reward model adds `"audio"` to its
  `required_fields`. Zero overhead for groupwise distributed gather.

**Commit message**: `[rewards] feat: add audio parameter to reward model interfaces`

**Verification**:
```python
# Existing rewards still work with audio-bearing samples
sample = T2AVSample(prompt="test", video=torch.randn(16,3,64,64), audio=torch.randn(1,16000))
# PickScore.__call__ doesn't have `audio` param → filter_kwargs removes it → works
# New audio reward model can declare `audio` in __call__ → receives it
```

---

## Step 6 — LTX2 adapter implementation

**New files**: `src/flow_factory/models/ltx2/__init__.py`, `src/flow_factory/models/ltx2/ltx2_t2av.py`
**Edit**: `src/flow_factory/models/registry.py`

**Scope**: Complete LTX2 model adapter — the only step that touches model-specific logic.

This is the largest step. Should be further broken down when we reach it, but the key components are:

### 6a. `LTX2Sample` dataclass

```python
@dataclass
class LTX2Sample(T2AVSample):
    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        'height', 'width', 'num_frames', 'duration_s',
        'latent_index_map', 'log_prob_index_map',
        'audio_latent_index_map',
    })
    # Audio latent trajectory (for training reconstruction, NOT for RL)
    audio_all_latents: Optional[torch.Tensor] = None
    audio_latent_index_map: Optional[torch.Tensor] = None
    # LTX2-specific text embeddings (separate video/audio streams)
    audio_prompt_embeds: Optional[torch.Tensor] = None
    connector_prompt_embeds: Optional[torch.Tensor] = None
    connector_audio_prompt_embeds: Optional[torch.Tensor] = None
    prompt_attention_mask: Optional[torch.Tensor] = None
```

### 6b. `LTX2_T2AV_Adapter` — key methods

| Method | Key Logic |
|--------|-----------|
| `load_pipeline()` | `LTX2Pipeline.from_pretrained(...)` |
| `load_scheduler()` | Call super for video SDE scheduler; create separate ODE-only `audio_scheduler` instance |
| `default_target_modules` | LTX2 transformer attention + FFN module names |
| `preprocessing_modules` | `['text_encoders', 'connectors']` (no VAE needed for T2AV preprocess) |
| `inference_modules` | `['transformer', 'vae', 'audio_vae', 'connectors', 'vocoder']` |
| `encode_prompt()` | Gemma3 all-layer hidden states → LTX2TextConnectors → `{prompt_embeds, audio_prompt_embeds, ...}` |
| `inference()` | Joint denoising loop: video SDE + audio ODE. Velocity ↔ x0 conversion for guidance. Collect video trajectory + log_probs; audio trajectory for decoding only. Decode both streams. Return `List[LTX2Sample]`. |
| `forward()` | Single step: transformer(video, audio) → video_pred, audio_pred. Video: `scheduler.step(SDE)` with log_prob. Audio: `audio_scheduler.step(ODE)` without log_prob. Return `SDESchedulerOutput` with video log_prob; attach `audio_next_latents` for trajectory tracking. |
| `decode_latents()` | Unpack video → VAE decode. Unpack audio → audio_vae decode → vocoder. |

### 6c. Training data flow (how `optimize()` uses LTX2Sample)

The GRPO trainer reads from batch dict. For LTX2:
- `batch['all_latents']` → video trajectory (RL-optimized)
- `batch['audio_all_latents']` → audio trajectory (**detached**, for transformer context only)
- `batch['log_probs']` → video log_probs (for PPO ratio)
- `batch['connector_prompt_embeds']` / `batch['connector_audio_prompt_embeds']` → text embeddings

In `forward()`, audio latents are detached:
```python
audio_latents = batch['audio_all_latents'][:, audio_map[t_idx]].detach()
```

### 6d. Registry

```python
'ltx2_t2av': 'flow_factory.models.ltx2.ltx2_t2av.LTX2_T2AV_Adapter',
```

**Commit message**: `[models] feat: add LTX2 text-to-audio-video adapter with video-only SDE optimization`

**Verification**: `get_model_adapter_class('ltx2_t2av')` resolves. Full integration test with actual model weights.

---

## Summary

| Step | Commit | Files | Status |
|------|--------|-------|--------|
| 1 | `[utils] audio module` | `utils/audio.py` (new), `utils/base.py` | ✅ `d8b99e9` |
| 2 | `[samples] audio field` | `samples/samples.py`, `samples/__init__.py` | ✅ `fe63e2a` |
| 3 | `[data] audio pipeline` | `hparams/data_args.py`, `data_utils/dataset.py` | ✅ `27b5967` |
| 4 | `[models] encode_audio` | `models/abc.py` | ✅ `4242d6d` |
| 5 | `[rewards] audio param` | `rewards/abc.py`, `rewards/reward_processor.py` | **← NEXT** |
| 6 | `[models] LTX2 adapter` | `models/ltx2/` (new), `models/registry.py` | Pending |

```
        Step 1 ✅
       ┌──┼──┐
       ▼  ▼  ▼
    Step2 Step3  ✅
       │  │
       ▼  ▼
    Step4 Step5  ← HERE
       │  │
       └──┼──┘
          ▼
        Step 6
```
