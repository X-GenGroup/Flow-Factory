# Flow-Factory Quick Reference Card

## 1️⃣ Sample Basics

```python
from flow_factory.samples import T2VSample, BaseSample

# Create a sample
sample = T2VSample(
    video=tensor_video,           # Tensor(T, C, H, W)
    prompt="a dog jumping",
    prompt_ids=token_ids,          # Tensor
    prompt_embeds=embeddings,      # Tensor
    all_latents=trajectory,        # Tensor(steps, ...)
    timesteps=noise_schedule,      # Tensor(steps+1,)
    height=480,
    width=832,
)

# Access fields (dict-like)
video = sample['video']
prompt = sample.prompt

# Stack multiple samples
batched = T2VSample.stack([sample1, sample2, sample3])
# Returns: Dict with stacked fields

# Move to device
sample.to('cuda')

# Get unique ID for grouping
group_id = sample.unique_id  # SHA256 hash of prompt
```

## 2️⃣ Video Output Format

| Format | Use Case | Output |
|--------|----------|--------|
| **Tensor** | DL models | `Tensor(T, C=3, H=480, W=832)` |
| **PIL** | Display/saving | `List[PIL.Image]` × T |
| **NumPy** | Scientific computing | `np.ndarray(T, C, H, W)` |

From WAN2:
```python
videos = adapter.decode_latents(latents, output_type='pt')
# Shape: (B, T, C=3, H, W) or (T, C=3, H, W)
```

## 3️⃣ Reward Model Pattern

### Quick Implementation
```python
from flow_factory.rewards.abc import PointwiseRewardModel

class MyReward(PointwiseRewardModel):
    use_tensor_inputs: bool = True
    required_fields = ('video', 'prompt')
    
    def __call__(self, prompt, video=None, **kwargs):
        scores = [self.model(v) for v in video]
        return RewardModelOutput(
            rewards=torch.tensor(scores),
            extra_info={'method': 'custom'}
        )
```

### Input Formats
```python
# Input to reward model:
reward_model(
    prompt=['a dog', 'a cat'],              # List[str] - always
    video=[tensor1, tensor2],               # If use_tensor_inputs=True
    # OR
    video=[[frame1, frame2, ...], [...]],   # If use_tensor_inputs=False (PIL)
)

# For each video element:
# - If use_tensor_inputs=True: Tensor(T, C, H, W)
# - If use_tensor_inputs=False: List[PIL.Image]
```

## 4️⃣ Sample Type Reference

| Type | Input | Output | Fields |
|------|-------|--------|--------|
| **T2ISample** | prompt | image | `image` |
| **T2VSample** | prompt | video | `video` |
| **I2ISample** | image + prompt | image | `image` + `condition_images` |
| **I2VSample** | image + prompt | video | `video` + `condition_images` |
| **V2VSample** | video + prompt | video | `video` + `condition_videos` |

## 5️⃣ Video Adapter Checklist

```python
class MyAdapter(BaseAdapter):
    def inference(self, prompt, height, width, num_frames, **kwargs):
        # 1. Encode prompt
        prompt_embeds = self.encode_prompt(prompt)
        
        # 2. Initialize latents (B, C_latent, T, H_latent, W_latent)
        latents = torch.randn(...)
        
        # 3. Diffusion loop
        all_latents = []
        for t in timesteps:
            latents = self.denoise_step(latents, t, prompt_embeds)
            all_latents.append(latents.clone())
        
        # 4. Decode to video (T, C=3, H, W)
        videos = self.vae.decode(latents)
        
        # 5. Return samples
        return [
            T2VSample(
                video=videos[i],
                prompt=prompt[i],
                prompt_embeds=prompt_embeds[i],
                all_latents=...,
                timesteps=timesteps,
                height=height,
                width=width,
            )
            for i in range(batch_size)
        ]
```

## 6️⃣ Data Flow (Copy-Paste Ready)

```python
# 1. Generate
adapter = Wan2_T2V_Adapter(config, accelerator)
samples = adapter.inference(
    prompt=["a dog jumping", "a cat sleeping"],
    height=480, width=832, num_frames=81
)
# Returns: List[WanT2VSample] with video field (81, 3, 480, 832)

# 2. Batch (optional)
batched = T2VSample.stack(samples)
# Returns: Dict with stacked/collected fields

# 3. Evaluate
reward_model = MyVideoReward(config, accelerator)
output = reward_model(
    prompt=[s.prompt for s in samples],
    video=[s.video for s in samples],
)
# Returns: RewardModelOutput(rewards=(2,), extra_info={...})

# 4. Train
loss = compute_loss(samples, output.rewards)
```

## 7️⃣ Field Shapes Reference

```
Single Sample (per-element):
├─ image: Tensor(3, 512, 512)
├─ video: Tensor(81, 3, 480, 832)
├─ prompt_ids: Tensor(512)
├─ prompt_embeds: Tensor(512, 768)
├─ all_latents: Tensor(50, 8, 81, 60, 104)
└─ timesteps: Tensor(51)

Batch (after stack):
├─ video: Tensor(B, 81, 3, 480, 832)
├─ prompt_ids: Tensor(B, 512)
├─ prompts: List[str] (not stacked)
├─ height: int (shared - first only)
└─ rewards: Tensor(B)
```

## 8️⃣ Video Constraints (WAN2)

```python
# Frame validation
vae_scale_factor_temporal = 8
valid_frames = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, ...]
# Formula: (num_frames - 1) % 8 == 0

# Resolution validation
h_multiple = 16  # vae_scale * patch_size
w_multiple = 16
valid_sizes = [480×832, 512×512, 576×1024, ...]
```

## 9️⃣ PyTree Registration (DDP/FSDP)

Samples are automatically registered as PyTorch pytrees:
```python
# Flatten for serialization
values, context = sample._flatten()

# Unflatten for deserialization
sample = T2VSample._unflatten(values, context)

# Automatic in DDP/FSDP - no manual work needed!
```

## 🔟 Extra Fields (extra_kwargs)

```python
# Store arbitrary metadata
sample = T2VSample(
    video=video,
    prompt=prompt,
    extra_kwargs={
        'generation_time': 2.3,
        'seed': 42,
        'model_version': 'v1.2',
        'custom_data': {...},
    }
)

# Access extra fields
time = sample['generation_time']  # Automatic via __getattr__
```

---

## 📚 File Map

| Need | File | Key Class/Function |
|------|------|-------------------|
| **Sample Types** | `samples/samples.py` | `BaseSample`, `T2VSample`, `T2ISample` |
| **Reward Base** | `rewards/abc.py` | `BaseRewardModel`, `PointwiseRewardModel` |
| **Video Example** | `models/wan/wan2_t2v.py` | `Wan2_T2V_Adapter`, `decode_latents()` |

---

**Key Insight:** Flow-Factory provides a unified data structure (Sample) that works across:
- Multiple modalities (image, video, conditions)
- Multiple adapters (SDXL, WAN2, etc.)
- Multiple reward types (pointwise, groupwise)

The `video` field is always `Tensor(T, C, H, W)` → this consistency enables modality-agnostic reward models!
