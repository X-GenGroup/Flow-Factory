# Step 6 — LTX2 Adapter Implementation Sub-Plan (v4)

## Installed Environment: diffusers 0.38.0.dev0

| Feature | Installed (0.38.0.dev0) | GitHub main (latest) |
|---------|------------------------|---------------------|
| Base T2AV pipeline | ✅ | ✅ |
| CFG | ✅ (velocity-space) | ✅ (x0-space) |
| STG | ❌ (no transformer param) | ✅ |
| Modality Isolation | ❌ (no transformer param) | ✅ |
| Prompt Enhancement | ❌ (`enhance_prompt` absent) | ✅ |
| Separate audio guidance scales | ❌ | ✅ |
| `DEFAULT_NEGATIVE_PROMPT` | ❌ (not in utils) | ✅ |

**Strategy**: Implement against **installed 0.38.0.dev0**. STG/modality isolation/prompt enhancement/x0-space guidance are deferred until diffusers upgrades — they require transformer-level param changes (`isolate_modalities`, `spatio_temporal_guidance_blocks`) that don't exist in the installed version.

---

## What's importable (verified on 0.38.0.dev0)

```python
# Pipeline class
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline, rescale_noise_cfg

# Constants
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

# Connectors class (for type hints)
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

# Transformer class (for type hints / config inspection)
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

# Pipeline instance methods (via self.pipeline.*):
#   .encode_prompt(prompt, negative_prompt, do_classifier_free_guidance, ...) → (embeds, mask, neg_embeds, neg_mask)
#   ._get_gemma_prompt_embeds(prompt, num_videos_per_prompt, max_sequence_length, scale_factor, device, dtype) → (embeds, mask)
#   ._pack_text_embeds(hidden_states, seq_lengths, device, padding_side, scale_factor) → packed_embeds
#   .connectors(prompt_embeds, additive_mask, additive_mask=True) → (video_embeds, audio_embeds, mask)
#   .prepare_latents(batch_size, num_channels_latents, height, width, num_frames, noise_scale, dtype, device, generator, latents) → packed
#   .prepare_audio_latents(batch_size, num_channels_latents, audio_latent_length, num_mel_bins, noise_scale, dtype, device, generator, latents) → packed
#   ._unpack_latents(latents, F, H, W, patch_size, patch_size_t) → (B, C, F, H, W)
#   ._unpack_audio_latents(latents, T, num_mel_bins) → (B, C, T, mel)
#   ._denormalize_latents(latents, mean, std, scaling_factor) → latents
#   ._denormalize_audio_latents(latents, mean, std) → latents
#   .video_processor.postprocess_video(video, output_type) → processed
#   .transformer.rope.prepare_video_coords(B, F, H, W, device, fps) → coords
#   .transformer.audio_rope.prepare_audio_coords(B, T, device) → coords
#   .transformer.cache_context(name) → context manager

# Pipeline instance attributes (set in __init__):
#   .vae_spatial_compression_ratio (default 32)
#   .vae_temporal_compression_ratio (default 8)
#   .audio_vae_mel_compression_ratio (default 4)
#   .audio_vae_temporal_compression_ratio (default 4)
#   .audio_sampling_rate (default 16000)
#   .audio_hop_length (default 160)
#   .transformer_spatial_patch_size (default 1)
#   .transformer_temporal_patch_size (default 1)
```

### Key API details (verified)

**Connectors call** — uses additive attention mask, NOT binary:
```python
additive_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
video_embeds, audio_embeds, mask = self.pipeline.connectors(prompt_embeds, additive_mask, additive_mask=True)
```

**Transformer forward** — installed version has NO `isolate_modalities` / `spatio_temporal_guidance_blocks`:
```python
video_pred, audio_pred = self.transformer(
    hidden_states=..., audio_hidden_states=...,
    encoder_hidden_states=..., audio_encoder_hidden_states=...,
    timestep=...,
    encoder_attention_mask=..., audio_encoder_attention_mask=...,
    num_frames=..., height=..., width=..., fps=...,
    audio_num_frames=...,
    video_coords=..., audio_coords=...,
    attention_kwargs=...,
    return_dict=False,
)
```

**Decoding order** (critical — different for video vs audio):
```python
# Video: unpack → denormalize → VAE decode
latents = pipeline._unpack_latents(latents, F, H, W, patch_size, patch_size_t)
latents = pipeline._denormalize_latents(latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
video = vae.decode(latents, timestep, return_dict=False)[0]

# Audio: denormalize → unpack → VAE decode → vocoder
audio_latents = pipeline._denormalize_audio_latents(audio_latents, audio_vae.latents_mean, audio_vae.latents_std)
audio_latents = pipeline._unpack_audio_latents(audio_latents, T, mel_bins)
mel = audio_vae.decode(audio_latents, return_dict=False)[0]
audio = vocoder(mel)
```

**CFG** — velocity-space in installed version:
```python
# Chunk: [uncond, cond] order (NOT [cond, uncond])
noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

---
## Sub-step 6a — Scaffold

**New files**: `models/ltx2/__init__.py`, `models/ltx2/ltx2_t2av.py`

### File structure

```python
# models/ltx2/ltx2_t2av.py

# --- Imports ---
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline, rescale_noise_cfg
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

# --- Local helpers (not in diffusers 0.37.1) ---
DEFAULT_NEGATIVE_PROMPT = "blurry, out of focus, ..."  # Hardcoded

def convert_velocity_to_x0(sample, velocity, sigma):
    return sample - velocity * sigma

def convert_x0_to_velocity(sample, x0, sigma):
    return (sample - x0) / sigma

# --- Sample dataclass ---
@dataclass
class LTX2Sample(T2AVSample): ...

# --- Adapter ---
class LTX2_T2AV_Adapter(BaseAdapter): ...
```

### LTX2Sample

```python
@dataclass
class LTX2Sample(T2AVSample):
    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        'height', 'width', 'num_frames', 'frame_rate', 'duration_s',
        'latent_index_map', 'log_prob_index_map',
        'audio_latent_index_map',
    })

    # Audio latent trajectory (for training reconstruction only, NOT RL-optimized)
    audio_all_latents: Optional[torch.Tensor] = None
    audio_latent_index_map: Optional[torch.Tensor] = None

    # Connector outputs (separate video/audio text embeddings)
    connector_prompt_embeds: Optional[torch.Tensor] = None
    connector_audio_prompt_embeds: Optional[torch.Tensor] = None
    connector_attention_mask: Optional[torch.Tensor] = None

    # Negative prompt connector outputs (for CFG during training forward)
    negative_connector_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_attention_mask: Optional[torch.Tensor] = None
```

### Adapter skeleton

```python
class LTX2_T2AV_Adapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: LTX2Pipeline
        self.audio_scheduler: FlowMatchEulerDiscreteSDEScheduler = self._create_audio_scheduler()

    # --- Pipeline loading ---

    def load_pipeline(self) -> LTX2Pipeline:
        return LTX2Pipeline.from_pretrained(
            self.model_args.model_name_or_path,
        )

    def _create_audio_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """Separate ODE-only scheduler for audio."""
        base_config = {
            k: v for k, v in dict(self.pipeline.scheduler.config).items()
            if k not in ('_class_name', '_diffusers_version')
        }
        return FlowMatchEulerDiscreteSDEScheduler(
            dynamics_type='ODE', noise_level=0.0, **base_config,
        )

    # --- Properties ---

    @property
    def default_target_modules(self) -> List[str]:
        # Will be verified against actual transformer layer names in 6a
        return [...]

    @property
    def preprocessing_modules(self) -> List[str]:
        return ['text_encoders', 'connectors']

    @property
    def inference_modules(self) -> List[str]:
        return ['transformer', 'vae', 'audio_vae', 'connectors', 'vocoder']

    # --- Stubs (replaced in 6b-6d) ---
    def encode_prompt(self, prompt, **kwargs): raise NotImplementedError
    def encode_image(self, images, **kwargs): return None
    def encode_video(self, videos, **kwargs): return None
    def decode_latents(self, latents, **kwargs): raise NotImplementedError
    def forward(self, *args, **kwargs): raise NotImplementedError
    def inference(self, *args, **kwargs): raise NotImplementedError
```

### `default_target_modules` verification

At implementation time, inspect `self.pipeline.transformer.named_modules()` to get exact layer names. Expected pattern based on `LTX2VideoTransformerBlock`:

```python
# Per-block structure (48 blocks, index as transformer_blocks.{i}.{layer}):
# Video self-attn:    transformer_blocks.{i}.attn1.{to_q,to_k,to_v,to_out.0}
# Audio self-attn:    transformer_blocks.{i}.audio_attn1.{to_q,to_k,to_v,to_out.0}
# Video cross-attn:   transformer_blocks.{i}.attn2.{to_q,to_k,to_v,to_out.0}
# Audio cross-attn:   transformer_blocks.{i}.audio_attn2.{to_q,to_k,to_v,to_out.0}
# A→V cross-attn:    transformer_blocks.{i}.attn_a2v.{to_q,to_k,to_v,to_out.0}
# V→A cross-attn:    transformer_blocks.{i}.attn_v2a.{to_q,to_k,to_v,to_out.0}
# Video FFN:          transformer_blocks.{i}.ff.net.{0.proj,2}
# Audio FFN:          transformer_blocks.{i}.audio_ff.net.{0.proj,2}
```

**Commit**: `[models/ltx2] feat: scaffold LTX2 adapter with sample dataclass and pipeline loading`
**Verify**: `from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter, LTX2Sample` succeeds.

---

## Sub-step 6b — Encoding & Decoding

### `encode_prompt()`

Use `self.pipeline.encode_prompt()` directly — it handles Gemma3 encoding + `_pack_text_embeds` internally. Then pass through connectors with additive mask.

```python
def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    do_classifier_free_guidance: bool = True,
    max_sequence_length: int = 1024,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    device = device or self.pipeline.text_encoder.device

    # 1. Use pipeline's encode_prompt (handles Gemma3 + _pack_text_embeds)
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
        self.pipeline.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            max_sequence_length=max_sequence_length, device=device,
        )
    )

    # Tokenize for prompt_ids (needed for unique_id hashing)
    prompt_ids = self.pipeline.tokenizer(
        prompt, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    # 2. Concat [negative, positive] for connectors if CFG
    if do_classifier_free_guidance:
        combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        combined_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    else:
        combined_embeds = prompt_embeds
        combined_mask = prompt_attention_mask

    # 3. Connectors — CRITICAL: use additive mask, not binary
    additive_mask = (1 - combined_mask.to(combined_embeds.dtype)) * -1000000.0
    connector_out, connector_audio_out, connector_mask = self.pipeline.connectors(
        combined_embeds, additive_mask, additive_mask=True,
    )

    # 4. Split neg/pos if CFG
    if do_classifier_free_guidance:
        neg_conn, pos_conn = connector_out.chunk(2)
        neg_audio_conn, pos_audio_conn = connector_audio_out.chunk(2)
        neg_conn_mask, pos_conn_mask = connector_mask.chunk(2)
    else:
        pos_conn, pos_audio_conn, pos_conn_mask = connector_out, connector_audio_out, connector_mask
        neg_conn = neg_audio_conn = neg_conn_mask = None

    return {
        'prompt_ids': prompt_ids,
        'connector_prompt_embeds': pos_conn,
        'connector_audio_prompt_embeds': pos_audio_conn,
        'connector_attention_mask': pos_conn_mask,
        'negative_connector_prompt_embeds': neg_conn,
        'negative_connector_audio_prompt_embeds': neg_audio_conn,
        'negative_connector_attention_mask': neg_conn_mask,
    }
```

### `decode_latents()`

```python
def decode_latents(
    self,
    video_latents: torch.Tensor,
    audio_latents: Optional[torch.Tensor] = None,
    height: int = 512, width: int = 768,
    num_frames: int = 121, frame_rate: float = 24.0,
    decode_timestep: float = 0.0,
    decode_noise_scale: Optional[float] = None,
    output_type: str = 'pt',
    **kwargs,
):
    vae = self.pipeline.vae
    patch_size = self.pipeline.transformer_spatial_patch_size     # 1
    patch_size_t = self.pipeline.transformer_temporal_patch_size  # 1
    vae_spatial = self.pipeline.vae_spatial_compression_ratio     # 32
    vae_temporal = self.pipeline.vae_temporal_compression_ratio   # 8

    # --- Video ---
    latent_h = height // vae_spatial
    latent_w = width // vae_spatial
    latent_f = (num_frames - 1) // vae_temporal + 1

    # Unpack: (B, seq, C) → (B, C, F, H, W)
    vid = self.pipeline._unpack_latents(video_latents, latent_f, latent_h, latent_w, patch_size, patch_size_t)
    vid = vid.to(vae.dtype)

    # Optional decode noise injection
    if decode_noise_scale is not None and vae.config.timestep_conditioning:
        noise = torch.randn_like(vid)
        vid = (1 - decode_noise_scale) * vid + decode_noise_scale * noise

    # Denormalize: latents * std / scaling_factor + mean
    vid = self.pipeline._denormalize_latents(vid, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
    video = vae.decode(vid, timestep=decode_timestep, return_dict=False)[0]
    video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)

    # --- Audio ---
    audio = None
    if audio_latents is not None:
        audio_vae = self.pipeline.audio_vae
        sr = self.pipeline.audio_sampling_rate                    # 16000
        hop = self.pipeline.audio_hop_length                        # 160
        mel_compression = self.pipeline.audio_vae_mel_compression_ratio  # 4
        temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio  # 4
        latent_mel_bins = 64 // mel_compression  # 16

        duration_s = (num_frames - 1) / frame_rate
        audio_num_frames = round(duration_s * sr / hop / temporal_compression)

        # Denormalize FIRST (before unpack — order matters!)
        aud = self.pipeline._denormalize_audio_latents(audio_latents, audio_vae.latents_mean, audio_vae.latents_std)
        # Unpack: (B, seq, C) → (B, C, T, mel_bins)
        aud = self.pipeline._unpack_audio_latents(aud, audio_num_frames, latent_mel_bins)
        aud = aud.to(audio_vae.dtype)
        # Audio VAE decode → mel spectrogram
        mel = audio_vae.decode(aud, return_dict=False)[0]
        # Vocoder → waveform
        audio = self.pipeline.vocoder(mel)  # (B, C_audio, T_audio)

    return video, audio
```

**Commit**: `[models/ltx2] feat: implement encode_prompt and decode_latents for LTX2`

---

## Sub-step 6c — Forward (single denoising step)

```python
def forward(
    self,
    t: torch.Tensor,
    t_next: Optional[torch.Tensor] = None,
    latents: torch.Tensor,
    next_latents: Optional[torch.Tensor] = None,
    audio_latents: Optional[torch.Tensor] = None,
    # Text embeddings (from connectors, stored on Sample)
    connector_prompt_embeds: torch.Tensor = None,
    connector_audio_prompt_embeds: torch.Tensor = None,
    connector_attention_mask: torch.Tensor = None,
    negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
    negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
    negative_connector_attention_mask: Optional[torch.Tensor] = None,
    # Guidance
    guidance_scale: float = 4.0,
    guidance_rescale: float = 0.0,
    # Generation shape (for RoPE)
    height: int = 512, width: int = 768,
    num_frames: int = 121, frame_rate: float = 24.0,
    audio_num_frames: Optional[int] = None,
    # Positional coords (cached from inference)
    video_coords: Optional[torch.Tensor] = None,
    audio_coords: Optional[torch.Tensor] = None,
    # Control
    noise_level: Optional[float] = None,
    compute_log_prob: bool = True,
    return_kwargs: List[str] = ['next_latents', 'log_prob', 'noise_pred'],
    **kwargs,
) -> SDESchedulerOutput:
    batch_size = latents.shape[0]
    device = latents.device
    do_cfg = guidance_scale > 1.0 and negative_connector_prompt_embeds is not None

    # --- Compute latent dims for RoPE (if coords not cached) ---
    vae_spatial = self.pipeline.vae_spatial_compression_ratio
    vae_temporal = self.pipeline.vae_temporal_compression_ratio
    latent_h = height // vae_spatial
    latent_w = width // vae_spatial
    latent_f = (num_frames - 1) // vae_temporal + 1
    if audio_num_frames is None:
        duration_s = (num_frames - 1) / frame_rate
        sr = self.pipeline.audio_sampling_rate
        hop = self.pipeline.audio_hop_length
        tc = self.pipeline.audio_vae_temporal_compression_ratio
        audio_num_frames = round(duration_s * sr / hop / tc)

    # --- Prepare RoPE coords if not cached ---
    if video_coords is None:
        video_coords = self.pipeline.transformer.rope.prepare_video_coords(
            batch_size, latent_f, latent_h, latent_w, device, fps=frame_rate)
    if audio_coords is None:
        audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
            batch_size, audio_num_frames, device)

    # --- 1. Prepare CFG inputs ---
    if do_cfg:
        lat_in = torch.cat([latents, latents])
        aud_in = torch.cat([audio_latents, audio_latents])
        text_in = torch.cat([negative_connector_prompt_embeds, connector_prompt_embeds])
        audio_text_in = torch.cat([negative_connector_audio_prompt_embeds, connector_audio_prompt_embeds])
        mask_in = torch.cat([negative_connector_attention_mask, connector_attention_mask])
        vid_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
        aud_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
        ts = t.expand(batch_size * 2)
    else:
        lat_in = latents
        aud_in = audio_latents
        text_in = connector_prompt_embeds
        audio_text_in = connector_audio_prompt_embeds
        mask_in = connector_attention_mask
        vid_coords = video_coords
        aud_coords = audio_coords
        ts = t.expand(batch_size)

    # --- 2. Transformer forward ---
    lat_in = lat_in.to(connector_prompt_embeds.dtype)
    aud_in = aud_in.to(connector_prompt_embeds.dtype)

    video_pred, audio_pred = self.transformer(
        hidden_states=lat_in,
        audio_hidden_states=aud_in,
        encoder_hidden_states=text_in,
        audio_encoder_hidden_states=audio_text_in,
        timestep=ts,
        encoder_attention_mask=mask_in,
        audio_encoder_attention_mask=mask_in,
        num_frames=latent_f,
        height=latent_h,
        width=latent_w,
        fps=frame_rate,
        audio_num_frames=audio_num_frames,
        video_coords=vid_coords,
        audio_coords=aud_coords,
        attention_kwargs=None,
        return_dict=False,
    )
    video_pred = video_pred.float()
    audio_pred = audio_pred.float()

    # --- 3. CFG ---
    if do_cfg:
        v_uncond, v_cond = video_pred.chunk(2)
        video_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
        a_uncond, a_cond = audio_pred.chunk(2)
        audio_pred = a_uncond + guidance_scale * (a_cond - a_uncond)

        if guidance_rescale > 0:
            video_pred = rescale_noise_cfg(video_pred, v_cond, guidance_rescale=guidance_rescale)
            audio_pred = rescale_noise_cfg(audio_pred, a_cond, guidance_rescale=guidance_rescale)

    # --- 4. Video: SDE scheduler step (with log_prob) ---
    video_output = self.scheduler.step(
        noise_pred=video_pred, timestep=t, latents=latents,
        timestep_next=t_next, next_latents=next_latents,
        compute_log_prob=compute_log_prob,
        return_dict=True, return_kwargs=return_kwargs,
        noise_level=noise_level,
    )

    # --- 5. Audio: ODE scheduler step (no log_prob) ---
    audio_output = self.audio_scheduler.step(
        noise_pred=audio_pred, timestep=t, latents=audio_latents,
        timestep_next=t_next, compute_log_prob=False,
        return_dict=True, return_kwargs=['next_latents'],
        dynamics_type='ODE',
    )

    # Attach audio result
    video_output.audio_next_latents = audio_output.next_latents
    return video_output
```

Note: This follows the **installed 0.37.1** CFG pattern (velocity-space). When diffusers upgrades to include x0-space guidance, STG, and modality isolation, these can be added as additional forward passes using the same structure described in v2 plan.

**Commit**: `[models/ltx2] feat: implement forward() with CFG and dual scheduler steps`

---

## Sub-step 6d — Inference loop + Registry

### `inference()` — complete skeleton

```python
@torch.no_grad()
def inference(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: int = 512, width: int = 768,
    num_frames: int = 121, frame_rate: float = 24.0,
    num_inference_steps: int = 40,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 4.0,
    guidance_rescale: float = 0.0,
    noise_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    # Pre-encoded
    prompt_ids: Optional[torch.Tensor] = None,
    connector_prompt_embeds: Optional[torch.Tensor] = None,
    connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
    connector_attention_mask: Optional[torch.Tensor] = None,
    negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
    negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
    negative_connector_attention_mask: Optional[torch.Tensor] = None,
    # Decode
    decode_timestep: float = 0.0,
    decode_noise_scale: Optional[float] = None,
    max_sequence_length: int = 1024,
    # RL-specific
    compute_log_prob: bool = True,
    trajectory_indices: TrajectoryIndicesType = 'all',
    extra_call_back_kwargs: List[str] = [],
    **kwargs,
) -> List[LTX2Sample]:

    device = self.device

    # ========== 1. Encode prompts ==========
    if connector_prompt_embeds is None:
        encoded = self.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            do_classifier_free_guidance=(guidance_scale > 1.0),
            max_sequence_length=max_sequence_length, device=device,
        )
        prompt_ids = encoded['prompt_ids']
        connector_prompt_embeds = encoded['connector_prompt_embeds']
        connector_audio_prompt_embeds = encoded['connector_audio_prompt_embeds']
        connector_attention_mask = encoded['connector_attention_mask']
        negative_connector_prompt_embeds = encoded.get('negative_connector_prompt_embeds')
        negative_connector_audio_prompt_embeds = encoded.get('negative_connector_audio_prompt_embeds')
        negative_connector_attention_mask = encoded.get('negative_connector_attention_mask')
    else:
        connector_prompt_embeds = connector_prompt_embeds.to(device)
        connector_audio_prompt_embeds = connector_audio_prompt_embeds.to(device)
        connector_attention_mask = connector_attention_mask.to(device)
        if negative_connector_prompt_embeds is not None:
            negative_connector_prompt_embeds = negative_connector_prompt_embeds.to(device)
            negative_connector_audio_prompt_embeds = negative_connector_audio_prompt_embeds.to(device)
            negative_connector_attention_mask = negative_connector_attention_mask.to(device)

    batch_size = connector_prompt_embeds.shape[0]

    # ========== 2. Compute dimensions ==========
    vae_spatial = self.pipeline.vae_spatial_compression_ratio    # 32
    vae_temporal = self.pipeline.vae_temporal_compression_ratio  # 8
    latent_h = height // vae_spatial
    latent_w = width // vae_spatial
    latent_f = (num_frames - 1) // vae_temporal + 1

    duration_s = (num_frames - 1) / frame_rate
    sr = self.pipeline.audio_sampling_rate
    hop = self.pipeline.audio_hop_length
    audio_temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio
    audio_mel_compression = self.pipeline.audio_vae_mel_compression_ratio
    audio_num_frames = round(duration_s * sr / hop / audio_temporal_compression)
    latent_mel_bins = 64 // audio_mel_compression  # 16

    # ========== 3. Prepare latents ==========
    video_latents = self.pipeline.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=self.transformer_config.in_channels,  # 128
        height=height, width=width, num_frames=num_frames,
        noise_scale=noise_scale, dtype=torch.float32,
        device=device, generator=generator,
    )
    audio_latents = self.pipeline.prepare_audio_latents(
        batch_size=batch_size,
        num_channels_latents=self.transformer_config.audio_in_channels,  # 8
        audio_latent_length=audio_num_frames,
        num_mel_bins=64,  # raw mel bins (before compression)
        noise_scale=noise_scale, dtype=torch.float32,
        device=device, generator=generator,
    )

    # ========== 4. Set timesteps ==========
    from ..scheduler import set_scheduler_timesteps
    video_seq_len = video_latents.shape[1]
    timesteps = set_scheduler_timesteps(
        self.scheduler, num_inference_steps, seq_len=video_seq_len, device=device,
        sigmas=sigmas,
    )
    # Audio scheduler uses same timesteps
    set_scheduler_timesteps(
        self.audio_scheduler, num_inference_steps, seq_len=video_seq_len, device=device,
        sigmas=sigmas,
    )

    # ========== 5. Prepare positional coords ==========
    video_coords = self.pipeline.transformer.rope.prepare_video_coords(
        batch_size, latent_f, latent_h, latent_w, device, fps=frame_rate,
    )
    audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
        batch_size, audio_num_frames, device,
    )

    # ========== 6. Setup trajectory collectors ==========
    video_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
    audio_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
    video_latents = self.cast_latents(video_latents)
    video_collector.collect(video_latents, step_idx=0)
    audio_collector.collect(audio_latents, step_idx=0)
    if compute_log_prob:
        log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
    callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

    # ========== 7. Denoising loop ==========
    for i, t in enumerate(timesteps):
        noise_level = self.scheduler.get_noise_level_for_timestep(t)
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
        return_kw = list(set(['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs))
        current_compute_lp = compute_log_prob and noise_level > 0

        output = self.forward(
            t=t, t_next=t_next,
            latents=video_latents, audio_latents=audio_latents,
            connector_prompt_embeds=connector_prompt_embeds,
            connector_audio_prompt_embeds=connector_audio_prompt_embeds,
            connector_attention_mask=connector_attention_mask,
            negative_connector_prompt_embeds=negative_connector_prompt_embeds,
            negative_connector_audio_prompt_embeds=negative_connector_audio_prompt_embeds,
            negative_connector_attention_mask=negative_connector_attention_mask,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords, audio_coords=audio_coords,
            noise_level=noise_level,
            compute_log_prob=current_compute_lp,
            return_kwargs=return_kw,
        )

        video_latents = self.cast_latents(output.next_latents)
        audio_latents = output.audio_next_latents
        video_collector.collect(video_latents, i + 1)
        audio_collector.collect(audio_latents, i + 1)
        if current_compute_lp:
            log_prob_collector.collect(output.log_prob, i)
        callback_collector.collect_step(i, output, extra_call_back_kwargs, capturable={'noise_level': noise_level})

    # ========== 8. Decode ==========
    video, audio_waveform = self.decode_latents(
        video_latents, audio_latents,
        height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
        decode_timestep=decode_timestep, decode_noise_scale=decode_noise_scale,
        output_type='pt',
    )

    # ========== 9. Construct samples ==========
    all_vid_lats = video_collector.get_result()
    vid_lat_map = video_collector.get_index_map()
    all_aud_lats = audio_collector.get_result()
    aud_lat_map = audio_collector.get_index_map()
    all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
    lp_map = log_prob_collector.get_index_map() if compute_log_prob else None
    cb_res = callback_collector.get_result()
    cb_map = callback_collector.get_index_map()

    prompt_list = prompt if isinstance(prompt, list) else [prompt] * batch_size

    samples = [
        LTX2Sample(
            timesteps=timesteps,
            all_latents=torch.stack([l[b] for l in all_vid_lats], dim=0) if all_vid_lats else None,
            log_probs=torch.stack([l[b] for l in all_log_probs], dim=0) if all_log_probs else None,
            latent_index_map=vid_lat_map,
            log_prob_index_map=lp_map,
            audio_all_latents=torch.stack([l[b] for l in all_aud_lats], dim=0) if all_aud_lats else None,
            audio_latent_index_map=aud_lat_map,
            video=video[b],
            audio=audio_waveform[b] if audio_waveform is not None else None,
            height=height, width=width,
            prompt=prompt_list[b],
            prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
            connector_prompt_embeds=connector_prompt_embeds[b],
            connector_audio_prompt_embeds=connector_audio_prompt_embeds[b],
            connector_attention_mask=connector_attention_mask[b],
            negative_connector_prompt_embeds=negative_connector_prompt_embeds[b] if negative_connector_prompt_embeds is not None else None,
            negative_connector_audio_prompt_embeds=negative_connector_audio_prompt_embeds[b] if negative_connector_audio_prompt_embeds is not None else None,
            negative_connector_attention_mask=negative_connector_attention_mask[b] if negative_connector_attention_mask is not None else None,
            extra_kwargs={
                **{k: v[b] for k, v in cb_res.items()},
                'callback_index_map': cb_map,
                'num_frames': num_frames,
                'frame_rate': frame_rate,
                'duration_s': duration_s,
            },
        )
        for b in range(batch_size)
    ]

    self.pipeline.maybe_free_model_hooks()
    return samples
```

### Registry

```python
# models/registry.py — add one line:
'ltx2_t2av': 'flow_factory.models.ltx2.ltx2_t2av.LTX2_T2AV_Adapter',
```

**Commit**: `[models/ltx2] feat: implement inference loop and register ltx2_t2av adapter`

---

## Summary

| Sub-step | What | Key Imports from diffusers |
|----------|------|--------------------------|
| **6a** | Scaffold + load_pipeline + audio_scheduler | `LTX2Pipeline`, `DISTILLED_SIGMA_VALUES`, `rescale_noise_cfg` |
| **6b** | encode_prompt + decode_latents | `pipeline.encode_prompt`, `pipeline.connectors(additive_mask=True)`, `pipeline._denormalize_*`, `pipeline._unpack_*` |
| **6c** | forward() with CFG + dual scheduler | `pipeline.transformer.rope.prepare_*_coords`, `transformer.cache_context` |
| **6d** | inference() loop + registry | `pipeline.prepare_latents`, `pipeline.prepare_audio_latents`, `set_scheduler_timesteps` |

### API corrections vs earlier plan versions

| Item | Previous assumption | Actual (verified) |
|------|-------------------|-------------------|
| diffusers version | 0.37.1 | **0.38.0.dev0** |
| Connectors call | `connectors(embeds, mask, padding_side="left")` | **`connectors(embeds, additive_mask, additive_mask=True)`** where `additive_mask = (1 - binary_mask) * -1e6` |
| Transformer forward | Has `sigma`, `audio_timestep`, `isolate_modalities`, `spatio_temporal_guidance_blocks` | **Only** `timestep` (no `sigma`, no `audio_timestep`, no STG/modality params) |
| CFG formula | `uncond + (gs-1) * (cond - uncond)` (x0-space) | **`uncond + gs * (cond - uncond)`** (velocity-space) |
| CFG chunk order | `[cond, uncond]` | **`[uncond, cond]`** (concat as `[neg, pos]`, chunk gives `[uncond, text]`) |
| Compression ratios | Class properties | **Instance attributes** set in `__init__` (`.vae_spatial_compression_ratio`, etc.) |
| Audio params source | `self.transformer_config.audio_*` | **`self.pipeline.audio_sampling_rate`**, **`self.pipeline.audio_hop_length`** |
| `enhance_prompt` | Available | **Not available** in installed version |
| `DEFAULT_NEGATIVE_PROMPT` | Importable from utils | **Not available** — define locally |

### Forward-compatibility notes

When diffusers upgrades to include STG/modality isolation/prompt enhancement:
- **STG**: Add extra transformer forward with `spatio_temporal_guidance_blocks=...` kwarg
- **Modality isolation**: Add extra forward with `isolate_modalities=True` kwarg
- **Prompt enhancement**: Call `self.pipeline.enhance_prompt(...)` before encoding
- **x0-space guidance**: Replace velocity-space CFG with velocity↔x0 conversion + x0-space deltas
- **Separate audio guidance**: Add `audio_guidance_scale` etc. parameters

All of these are additive changes to `forward()` and `inference()` — no architectural changes needed.
