# LTX2 Integration Plan (v3 — Data Flow Deep-Dive)

## Core Design Decision

**Only optimize video modality via SDE + log_prob; audio uses pure ODE (no log_prob, no RL optimization).**

---

## Part A: Complete Data Flow Analysis

### Current Data Flow (No Audio)

```
JSONL file                     GeneralDataset._preprocess_batch()
─────────                      ────────────────────────────────────
{"prompt": "...",         ──►  1. prompt = batch["prompt"]
 "images": "a.png",       ──►  2. images = [PIL.open(path)]     ──► adapter.preprocess_func(prompt, images, videos)
 "videos": "b.mp4"}       ──►  3. videos = [load_frames(path)]       │
                                                                      ▼
                                                                 encode_prompt() → {prompt_ids, prompt_embeds}
                                                                 encode_image()  → {condition_images, image_latents, ...}
                                                                 encode_video()  → {condition_video_latents, ...}
                                                                      │
                                                                      ▼
                                                              HF Dataset cache (Arrow)
                                                                      │
                                                                      ▼
                                                              DataLoader batch
                                                                      │
                                                                      ▼
                                                              adapter.inference(**batch)
```

### Touch Points That Need Audio Support

Tracing every point where audio data must flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. JSONL Schema  (user data)                                        │
│    {"prompt": "...", "audio_path": "cat.wav"}                       │
│    OR {"prompt": "...", "video_path": "v.mp4"}  ← extract audio     │
│                                                                     │
│ 2. DataArguments  (hparams/data_args.py)                            │
│    + audio_dir: Optional[str]                                       │
│                                                                     │
│ 3. GeneralDataset._preprocess_batch()  (data_utils/dataset.py)      │
│    + Load audio from disk → raw waveform tensor                     │
│    + Pass to preprocess_func(... audios=...)                        │
│    + Store raw audio tensor in cache for reward model consumption   │
│                                                                     │
│ 4. GeneralDataset._preprocess_batch() PREPROCESS_KEYS               │
│    + Add 'audios' to PREPROCESS_KEYS tuple                          │
│                                                                     │
│ 5. BaseAdapter.preprocess_func()  (models/abc.py)                   │
│    + Route audios → encode_audio()                                  │
│                                                                     │
│ 6. BaseAdapter.encode_audio()  (models/abc.py)                      │
│    + Default: return None                                           │
│    + LTX2: audio_vae.encode() → audio latents                      │
│                                                                     │
│ 7. BaseSample  (samples/samples.py)                                 │
│    + audio field for decoded waveform (reward model consumption)    │
│                                                                     │
│ 8. RewardProcessor.MEDIA_FIELDS  (rewards/reward_processor.py)      │
│    + Add 'audio' to MEDIA_FIELDS set                                │
│    + Add audio format conversion in _convert_media_format()         │
│                                                                     │
│ 9. PointwiseRewardModel / GroupwiseRewardModel  (rewards/abc.py)    │
│    + Add optional audio parameter                                   │
│                                                                     │
│10. LTX2 Adapter  (models/ltx2/ltx2_t2av.py)                        │
│    + inference() stores decoded audio on sample.audio               │
│    + forward() uses stored audio_latents for transformer input      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part B: Detailed Changes Per File

### B1. `hparams/data_args.py` — Add `audio_dir`

```python
@dataclass
class DataArguments(ArgABC):
    ...
    audio_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to folder containing audio files. Defaults to 'audios' subfolder in dataset_dir."},
    )
```

**Why**: Follows the exact same pattern as `image_dir` and `video_dir`. The dataset needs to know where audio files live on disk.

---

### B2. `data_utils/dataset.py` — Audio Loading & Preprocessing

#### B2a. New Protocol

```python
class AudioEncodeCallable(Protocol):
    """Protocol for audio encoding functions."""
    def __call__(self, audios: Union[torch.Tensor, List[torch.Tensor]], **kwargs: Any) -> Dict[str, Any]:
        ...
```

#### B2b. `PreprocessCallable` Protocol Update

```python
class PreprocessCallable(Protocol):
    def __call__(
        self,
        prompt: Optional[...],
        images: Optional[...],
        videos: Optional[...],
        audios: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,  # NEW
        **kwargs: Any
    ) -> Dict[str, Any]:
        ...
```

#### B2c. `GeneralDataset.__init__()` — Accept `audio_dir`

```python
def __init__(self, ..., audio_dir: Optional[str] = None, **kwargs):
    ...
    self.audio_dir = audio_dir
```

#### B2d. `_load_raw_dataset()` — Set Default `audio_dir`

```python
if os.path.exists(jsonl_path):
    ...
    self.audio_dir = os.path.join(self.data_root, "audios") if self.audio_dir is None else self.audio_dir
```

#### B2e. `_preprocess_batch()` — The Critical Change

This is the core data loading function. Currently it handles `prompt`, `images`, `videos`. We add `audios` as a 4th modality following the exact same pattern:

```python
def _preprocess_batch(self, batch, image_dir, video_dir, audio_dir):  # +audio_dir
    PREPROCESS_KEYS = ('prompt', 'negative_prompt', 'images', 'videos', 'audios')  # +audios

    # ... existing prompt, image, video handling ...

    # 4. Prepare audio inputs (NEW — follows video pattern exactly)
    if 'audio' in batch:
        batch['audios'] = batch.pop('audio')  # Rename for consistency

    audio_args = {'audios': None}
    if audio_dir is not None and "audios" in batch:
        audio_paths_list = batch["audios"]
        batch['audios'] = []
        audio_args['audios'] = []
        for audio_paths in audio_paths_list:
            if not audio_paths:
                audio_args['audios'].append(None)
                batch['audios'].append(None)
            else:
                if isinstance(audio_paths, str):
                    audio_paths = [audio_paths]
                # Load audio(s) as waveform tensors
                audios = [
                    load_audio(_resolve_path(audio_dir, p))
                    for p in audio_paths
                ]
                audio_args['audios'].append(audios[0] if len(audios) == 1 else audios)
                batch['audios'].append(audios[0] if len(audios) == 1 else audios)

    # 5. Call preprocess function (now includes audios)
    input_args = {**prompt_args, **image_args, **video_args, **audio_args, **self._preprocess_kwargs}
    ...
```

Key design decisions:
- Audio column in JSONL: `"audio"` (singular) or `"audios"` (plural), auto-normalized to `"audios"`
- Each audio is loaded as a **raw waveform tensor** `(channels, samples)` via a new `load_audio()` utility
- Unlike images (always PIL), audio is kept as tensors from the start — there's no "PIL equivalent" for audio
- Raw waveform is stored in batch cache for two purposes:
  1. Passed to `adapter.preprocess_func()` → `encode_audio()` → audio latents (for conditioning/reconstruction)
  2. Kept in sample for reward model consumption (audio quality reward models may need raw waveform)

#### B2f. `_preprocess_batch()` — Pass `audio_dir` from `.map()`

```python
processed_dataset = raw_dataset.map(
    self._preprocess_batch,
    ...
    fn_kwargs={
        "image_dir": self.image_dir,
        "video_dir": self.video_dir,
        "audio_dir": self.audio_dir,    # NEW
    },
    ...
)
```

#### B2g. New Utility: `load_audio()`

```python
def load_audio(audio_path: str, sr: int = 16000) -> torch.Tensor:
    """
    Load audio file as waveform tensor.

    Args:
        audio_path: Path to audio file (.wav, .mp3, .flac, etc.)
        sr: Target sample rate for resampling

    Returns:
        torch.Tensor of shape (channels, num_samples), float32, range [-1, 1]
    """
    try:
        import torchaudio
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        return waveform
    except ImportError:
        raise ImportError(
            "torchaudio is required for audio loading. "
            "Install it with: pip install torchaudio"
        )
```

**Why torchaudio**: It's the PyTorch-native audio library, handles all formats, and integrates seamlessly with the tensor pipeline. It's an optional dependency — only needed when audio_dir is configured.

---

### B3. `utils/audio.py` — New Audio Utility Module

Following the pattern of `utils/image.py` and `utils/video.py`:

```python
# Type aliases
AudioSingle = Union[torch.Tensor, np.ndarray]          # (channels, samples) or (samples,)
AudioBatch = Union[torch.Tensor, List[torch.Tensor]]    # (N, channels, samples) or List[(channels, samples)]

def standardize_audio_batch(
    audios: Union[AudioSingle, AudioBatch],
    output_type: Literal['pt', 'np'] = 'pt',
) -> AudioBatch:
    """Standardize audio to consistent format."""
    ...

def is_audio(audio: Any) -> bool: ...
def is_audio_batch(audios: Any) -> bool: ...
```

**Why a separate module**: Matches the existing pattern (`utils/image.py`, `utils/video.py`). Keeps audio-specific logic isolated.

---

### B4. `samples/samples.py` — Add `audio` Field

```python
@dataclass
class BaseSample:
    ...
    # Generated media
    image: Optional[ImageSingle] = None
    video: Optional[VideoSingle] = None
    audio: Optional[torch.Tensor] = None  # NEW: (channels, num_samples) waveform

    def __post_init__(self):
        ...
        # Audio: ensure tensor format
        if self.audio is not None and not isinstance(self.audio, torch.Tensor):
            self.audio = torch.as_tensor(self.audio)
```

**No new T2AVSample base class needed** — the `audio` field is on `BaseSample`, and the LTX2-specific fields (audio latent trajectory) go in the model-specific `LTX2Sample` subclass.

---

### B5. `models/abc.py` — BaseAdapter Changes

#### B5a. `encode_audio()` default method

```python
def encode_audio(
    self,
    audios: Union[torch.Tensor, List[torch.Tensor]],
    **kwargs,
) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
    """
    Encode audio inputs into latent representations.
    Default: return None (no audio processing).
    Override for models that accept audio input.
    """
    return None
```

**Not abstract** — it has a default `None` return so existing adapters don't break.

#### B5b. `preprocess_func()` — Add audio routing

```python
def preprocess_func(
    self,
    prompt=None, images=None, videos=None,
    audios=None,  # NEW
    **kwargs,
):
    results = {}
    for input, encoder_method in [
        (prompt, self.encode_prompt),
        (images, self.encode_image),
        (videos, self.encode_video),
        (audios, self.encode_audio),   # NEW
    ]:
        if input is not None:
            res = encoder_method(input, **(filter_kwargs(encoder_method, **kwargs)))
            if res is not None and isinstance(res, dict):
                results.update(res)
    return results
```

#### B5c. `audio_vae` property

```python
@property
def audio_vae(self) -> Optional[torch.nn.Module]:
    """Get audio VAE if available in pipeline."""
    if hasattr(self.pipeline, 'audio_vae'):
        return self.get_component('audio_vae')
    return None
```

#### B5d. `_freeze_components()` — Freeze audio_vae

```python
def _freeze_components(self):
    ...
    # Freeze audio VAE if present
    if self.audio_vae is not None:
        self.audio_vae.requires_grad_(False)
        self.audio_vae.eval()
```

---

### B6. `rewards/reward_processor.py` — Audio Support

```python
class RewardProcessor:
    MEDIA_FIELDS = {'image', 'video', 'audio', 'condition_images', 'condition_videos'}  # +audio

    def _convert_media_format(self, batch_input, model):
        ...
        for k, v in batch_input.items():
            ...
            elif k == 'audio':
                # Audio is always tensor-based, no PIL conversion needed
                # Just pass through as-is (or convert to numpy if model wants it)
                if getattr(model, 'use_tensor_inputs', False):
                    result[k] = v  # List[Tensor]
                else:
                    # For non-tensor reward models, convert to numpy
                    result[k] = [a.numpy() if isinstance(a, torch.Tensor) else a for a in v]
```

---

### B7. `rewards/abc.py` — Add `audio` Parameter

```python
class PointwiseRewardModel(BaseRewardModel):
    @abstractmethod
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        audio: Optional[List[torch.Tensor]] = None,  # NEW
        condition_images: Optional[...] = None,
        condition_videos: Optional[...] = None,
        **kwargs,
    ) -> RewardModelOutput:
        ...

# Same for GroupwiseRewardModel
```

**Backward compatible**: `audio=None` default, existing reward models just ignore it via `**kwargs` / `filter_kwargs`.

---

### B8. `data_utils/loader.py` — Pass `audio_dir`

No explicit changes needed — `audio_dir` flows through automatically:
1. `DataArguments.audio_dir` → `data_args`
2. `filter_kwargs(GeneralDataset.__init__, **data_args)` picks up `audio_dir` since it's a parameter of `__init__`
3. `GeneralDataset.__init__` receives and stores it
4. `.map(fn_kwargs={"audio_dir": self.audio_dir})` passes it to `_preprocess_batch`

This works because `loader.py` already does:
```python
base_kwargs.update(filter_kwargs(GeneralDataset.__init__, **data_args))
```

---

## Part C: LTX2-Specific Data Flow

### C1. Dataset Format

```json
{"prompt": "A pianist performing Chopin under rain"}
```

For LTX2 text-to-audio-video, most training data is **prompt-only** (the model generates both audio and video). No `audio_path` needed in typical T2AV training.

If conditioning audio is provided:
```json
{"prompt": "...", "audio": "background_music.wav"}
```

### C2. LTX2 Preprocessing

```python
class LTX2_T2AV_Adapter(BaseAdapter):
    @property
    def preprocessing_modules(self):
        return ['text_encoders', 'connectors']
        # Note: vae/audio_vae NOT needed at preprocess time
        # because T2AV generates from text only — no image/video/audio to encode

    def encode_prompt(self, prompt, **kwargs):
        # 1. Gemma3 encoding → all 49 layer hidden states
        # 2. LTX2TextConnectors → video_prompt_embeds, audio_prompt_embeds
        return {
            'prompt_ids': token_ids,
            'prompt_embeds': video_prompt_embeds,       # for video cross-attention
            'audio_prompt_embeds': audio_prompt_embeds,  # for audio cross-attention
            'prompt_attention_mask': mask,
        }

    def encode_audio(self, audios, **kwargs):
        # Only needed if conditioning on input audio
        # For pure T2AV, this is not called
        return None
```

### C3. LTX2 Inference → Sample Construction

```python
def inference(self, prompt_embeds, audio_prompt_embeds, ...):
    ...
    # After denoising loop:
    video = self.decode_video(video_latents)    # VAE decode
    audio = self.decode_audio(audio_latents)    # audio_vae decode → vocoder

    return [LTX2Sample(
        prompt=prompts[j],
        prompt_embeds=prompt_embeds[j],
        audio_prompt_embeds=audio_prompt_embeds[j],

        # Video output (for reward model + display)
        video=video[j],                           # (T, C, H, W) tensor

        # Audio output (for reward model + display)
        audio=audio[j],                           # (channels, num_samples) waveform

        # Video trajectory (for RL training)
        all_latents=video_trajectory[j],          # (stored_steps, seq_len, C)
        timesteps=timesteps,
        log_probs=video_log_probs[j],             # (num_sde_steps,)
        latent_index_map=latent_index_map,
        log_prob_index_map=log_prob_index_map,

        # Audio trajectory (for training reconstruction, NOT RL)
        audio_all_latents=audio_trajectory[j],    # (stored_steps, audio_seq, C)
        audio_latent_index_map=audio_latent_index_map,
    )]
```

### C4. Training: optimize() Data Flow

The GRPO trainer's `optimize()` accesses stored data via `BaseSample.stack()` → batch dict.

For LTX2, the batch dict will contain:
```python
batch = {
    'all_latents': (B, stored_steps, video_seq, C),      # video trajectory
    'log_probs': (B, num_sde_steps),                       # video log_probs
    'timesteps': (num_steps+1,),                           # shared
    'latent_index_map': (num_steps+1,),                    # shared

    'audio_all_latents': (B, stored_steps, audio_seq, C), # audio trajectory
    'audio_latent_index_map': (num_steps+1,),              # shared

    'prompt_embeds': (B, seq, D),
    'audio_prompt_embeds': (B, seq, D),
    ...
}
```

In the training loop:
```python
for timestep_index in scheduler.train_timesteps:
    # Video latents (gradient flows)
    video_latents = batch['all_latents'][:, latent_map[timestep_index]]
    video_next    = batch['all_latents'][:, latent_map[timestep_index + 1]]

    # Audio latents (DETACHED — no gradient)
    audio_latents = batch['audio_all_latents'][:, audio_map[timestep_index]].detach()

    output = adapter.forward(
        t=t, t_next=t_next,
        latents=video_latents,
        next_latents=video_next,
        audio_latents=audio_latents,         # detached, for transformer context
        prompt_embeds=batch['prompt_embeds'],
        audio_prompt_embeds=batch['audio_prompt_embeds'],
        compute_log_prob=True,
    )

    # PPO loss on video log_prob only
    ratio = exp(output.log_prob - old_log_prob)
    loss = clipped_ppo(ratio, advantage)
```

**Key**: `audio_latents.detach()` ensures no gradient flows through the audio pathway during training. The audio latents are only there because the transformer's cross-modal attention requires both streams as input.

---

## Part D: Scheduler Strategy

The existing `FlowMatchEulerDiscreteSDEScheduler.step()` already supports `dynamics_type='ODE'`.

For LTX2, we need **two scheduler instances** (because `step()` mutates internal `step_index`):
- `self.scheduler` → SDE for video (loaded via `load_scheduler()` as usual)
- `self.audio_scheduler` → ODE for audio (separate instance, always deterministic)

```python
def load_scheduler(self):
    # Video: standard SDE scheduler
    video_scheduler = super().load_scheduler()

    # Audio: ODE-only copy
    self.audio_scheduler = FlowMatchEulerDiscreteSDEScheduler(
        dynamics_type='ODE',
        noise_level=0.0,
        **{k: v for k, v in self.pipeline.scheduler.config.items()
           if k != '_class_name'},
    )
    return video_scheduler
```

---

## Part E: Summary of All Changes

| Layer | File | Change | Breaking? |
|-------|------|--------|-----------|
| Config | `hparams/data_args.py` | Add `audio_dir` field | No |
| Utils | `utils/audio.py` | **New file**: AudioSingle/AudioBatch types, standardize_audio_batch | No |
| Dataset | `data_utils/dataset.py` | Add `load_audio()`, handle `audios` in `_preprocess_batch`, add `audio_dir` param | No |
| Samples | `samples/samples.py` | Add `audio` field to BaseSample | No (None default) |
| Adapter | `models/abc.py` | Add `encode_audio()` default, `audio_vae` property, update `preprocess_func` | No |
| Rewards | `rewards/abc.py` | Add `audio` param to interfaces | No (Optional) |
| Rewards | `rewards/reward_processor.py` | Add `'audio'` to MEDIA_FIELDS, handle in `_convert_media_format` | No |
| Model | `models/ltx2/` | **New**: LTX2_T2AV_Adapter + LTX2Sample | Self-contained |
| Registry | `models/registry.py` | Add `ltx2_t2av` entry | One line |

**Total framework changes**: 7 existing files modified (all backward compatible) + 2 new files.
