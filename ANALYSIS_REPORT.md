# 📊 Flow-Factory Complete Analysis Report

**Date:** April 7, 2026  
**Repository:** `/Users/bowenping/code/Flow-Factory`  
**Scope:** Sample module, Reward model base classes, Video adapter implementation

---

## 📋 TABLE OF CONTENTS

1. [Sample Dataclass Fields](#1-sample-dataclass-fields)
2. [Reward Model Interface](#2-reward-model-interface)
3. [Video Adapter Implementation](#3-video-adapter-implementation)
4. [Integration Guide](#4-integration-guide)

---

# 1. SAMPLE DATACLASS FIELDS

## File Location
`/src/flow_factory/samples/samples.py` (lines 65-456)

## 1.1 BaseSample Overview

**Purpose:** Universal output container for generative models  
**Scope:** Holds media, trajectory, prompts, and metadata  
**Design Pattern:** Python dataclass with PyTree registration for DDP/FSDP

### Complete Field Reference

```python
@dataclass
class BaseSample:
    # Denoising trajectory fields (for reward-guided training)
    timesteps: Optional[torch.Tensor] = None                    # (T+1,)
    all_latents: Optional[torch.Tensor] = None                 # (num_steps, Seq_len, C)
    latent_index_map: Optional[torch.Tensor] = None            # (T+1,) LongTensor
    log_probs: Optional[torch.Tensor] = None                   # (num_steps,)
    log_prob_index_map: Optional[torch.Tensor] = None          # (T+1,) LongTensor
    
    # Output media (canonicalized in __post_init__)
    image: Optional[ImageSingle] = None                         # → Tensor(C, H, W)
    video: Optional[VideoSingle] = None                         # → Tensor(T, C, H, W)
    height: Optional[int] = None
    width: Optional[int] = None
    
    # Prompt information
    prompt: Optional[str] = None
    prompt_ids: Optional[torch.Tensor] = None
    prompt_embeds: Optional[torch.Tensor] = None
    
    # Negative prompt (for classifier-free guidance)
    negative_prompt: Optional[str] = None
    negative_prompt_ids: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    
    # Extensibility & grouping
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    _unique_id: Optional[int] = field(default=None, repr=False, compare=False)
```

### Class Variables

```python
_id_fields: ClassVar[frozenset[str]] = frozenset({'prompt', 'prompt_ids'})
# ^ Used to compute unique_id for grouping samples

_shared_fields: ClassVar[frozenset[str]] = frozenset({
    'height', 'width', 'latent_index_map', 'log_prob_index_map'
})
# ^ Fields that are identical across batch (taken once during stacking)
```

## 1.2 Sample Type Hierarchy

### Complete Inheritance Tree

```
BaseSample (25 fields)
├── T2ISample (Text-to-Image)
│   └── Inherits all 25 fields from BaseSample
│
├── T2VSample (Text-to-Video)
│   └── Inherits all 25 fields from BaseSample
│   └── Used by: Wan2_T2V_Adapter
│
├── ImageConditionSample (adds condition_images field)
│   ├── _id_fields: +{'condition_images'}
│   ├── condition_images: Optional[ImageBatch] → List[Tensor(C, H, W)]
│   ├── I2ISample (Image-to-Image)
│   └── I2VSample (Image-to-Video)
│
└── VideoConditionSample (adds condition_videos field)
    ├── _id_fields: +{'condition_videos'}
    ├── condition_videos: Optional[VideoBatch] → List[Tensor(T, C, H, W)]
    └── V2VSample (Video-to-Video)
```

### Condition Field Details

#### ImageBatch (for condition_images)
```python
# Input types (automatically canonicalized):
ImageBatch = Union[
    List[PIL.Image],                  # List of PIL images
    torch.Tensor,                     # Tensor(B, C, H, W) or (C, H, W)
    np.ndarray,                       # Array(B, C, H, W) or (C, H, W)
    List[torch.Tensor | np.ndarray]  # Mixed list
]

# After canonicalization in __post_init__:
# Always → List[torch.Tensor] of shape (C, H, W) each
```

#### VideoBatch (for condition_videos)
```python
# Input types (automatically canonicalized):
VideoBatch = Union[
    List[List[PIL.Image]],                    # List of frame sequences
    torch.Tensor,                             # Tensor(B, T, C, H, W) or (T, C, H, W)
    np.ndarray,                               # Array(B, T, C, H, W) or (T, C, H, W)
    List[torch.Tensor | List[PIL.Image]]     # Mixed list
]

# After canonicalization in __post_init__:
# Always → List[torch.Tensor] of shape (T, C, H, W) each
```

## 1.3 Key Methods

### __post_init__() - Canonicalization
```python
def __post_init__(self):
    """Convert all media to standard tensor formats."""
    if self.image is not None:
        self.image = standardize_image_batch(self.image, 'pt')[0]
        # Output: Tensor(C, H, W)
    
    if self.video is not None:
        self.video = standardize_video_batch(self.video, 'pt')[0]
        # Output: Tensor(T, C, H, W)
```

### compute_unique_id() - Sample Grouping
```python
def compute_unique_id(self) -> int:
    """Hash-based grouping for GroupwiseRewardModel."""
    hasher = hashlib.sha256()
    
    if self.prompt_ids is not None:
        hasher.update(self.prompt_ids.cpu().numpy().tobytes())
    elif self.prompt is not None:
        hasher.update(self.prompt.encode('utf-8'))
    
    return int.from_bytes(hasher.digest()[:8], byteorder='big', signed=True)
    # Returns: 64-bit signed integer
```

### stack() - Batch Processing
```python
@classmethod
def stack(cls, samples: List[BaseSample]) -> Dict[str, Union[torch.Tensor, Dict, List, Any]]:
    """Stack list of samples following smart rules:
    
    1. Shared fields → return first element only
    2. Stackable tensors → torch.stack if shapes match
    3. Dicts → recursively stack values
    4. Other fields → return as list
    """
    sample_dicts = [s.to_dict() for s in samples]
    return {
        key: cls._stack_values(key, [d[key] for d in sample_dicts])
        for key in sample_dicts[0].keys()
    }
```

---

# 2. REWARD MODEL INTERFACE

## File Location
`/src/flow_factory/rewards/abc.py` (lines 36-179)

## 2.1 BaseRewardModel (Abstract Base Class)

```python
class BaseRewardModel(ABC):
    """Abstract base for all reward models."""
    
    # Configuration
    required_fields: Tuple[str, ...] = ()  # Which Sample fields are needed
    use_tensor_inputs: bool = False         # True: torch.Tensor, False: PIL.Image
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        self.accelerator = accelerator
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.model: Optional[nn.Module] = None
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> RewardModelOutput:
        """Compute rewards. Signature varies by subclass."""
        pass
    
    def to(self, device: torch.device) -> BaseRewardModel:
        """Move model to device."""
        if self.model is not None:
            self.model.to(device)
        self.device = device
        return self
```

## 2.2 PointwiseRewardModel

**Purpose:** Compute independent per-sample rewards  
**Use Cases:** Quality ratings, preference models, single-sample scoring

### Class Definition

```python
class PointwiseRewardModel(BaseRewardModel):
    required_fields: Tuple[str, ...] = ('image', 'prompt')
    use_tensor_inputs: bool = False
    
    @abstractmethod
    def __call__(
        self,
        prompt: List[str],                                        # (batch_size,)
        image: Optional[List[Image.Image]] = None,               # (B,) or List[Tensor(C,H,W)]
        video: Optional[List[List[Image.Image]]] = None,         # (B,) or List[Tensor(T,C,H,W)]
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute per-sample rewards."""
        pass
```

### Input Format Specification

#### Prompt
```python
prompt: List[str]  # Length = batch_size
# Example: ["a dog jumping", "a cat sleeping"]
```

#### Image
```python
# If use_tensor_inputs = False:
image: List[PIL.Image]  # Length = batch_size

# If use_tensor_inputs = True:
image: List[torch.Tensor]  # Length = batch_size
# Each element: Tensor(C=3, H, W)
```

#### Video
```python
# If use_tensor_inputs = False:
video: List[List[PIL.Image]]  # List[frames_per_video]

# If use_tensor_inputs = True:
video: List[torch.Tensor]  # Length = batch_size
# Each element: Tensor(T, C=3, H, W)
# Example shapes:
#   - (81, 3, 480, 832) for WAN2 output
#   - (16, 3, 512, 512) for other models
```

#### Condition Images
```python
# If use_tensor_inputs = False:
condition_images: List[List[PIL.Image]]  # (B, num_conditions)

# If use_tensor_inputs = True:
# Case 1: All conditions same size
condition_images: List[torch.Tensor]  # (B, num_conditions, C, H, W)

# Case 2: Conditions with varying sizes
condition_images: List[List[torch.Tensor]]  # (B, num_conditions, (C, H, W))
```

#### Condition Videos
```python
# If use_tensor_inputs = False:
condition_videos: List[List[List[PIL.Image]]]  # (B, num_conditions, frames)

# If use_tensor_inputs = True:
# Case 1: All conditions same size
condition_videos: List[torch.Tensor]  # (B, num_conditions, T, C, H, W)

# Case 2: Conditions with varying sizes
condition_videos: List[List[torch.Tensor]]  # (B, num_conditions, (T, C, H, W))
```

### Output Format

```python
RewardModelOutput(
    rewards=torch.Tensor,  # Shape: (batch_size,)
    extra_info={           # Optional metadata
        'component_scores': {...},
        'debug_info': {...}
    }
)
```

## 2.3 GroupwiseRewardModel

**Purpose:** Compute group-aware rewards (pairwise, ranking, contrastive)  
**Key Difference:** All samples with same `unique_id` are processed together  
**Use Cases:** Preference learning, ranking losses, DPO, IPO

### Class Definition

```python
class GroupwiseRewardModel(BaseRewardModel):
    required_fields: Tuple[str, ...] = ('image', 'prompt')
    use_tensor_inputs: bool = False
    
    @abstractmethod
    def __call__(
        self,
        prompt: List[str],                                        # (group_size,)
        image: Optional[List[Image.Image]] = None,               # All from same group
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute group-aware rewards."""
        pass
```

### Key Semantics

1. **Input Grouping:** All samples share same `unique_id` (same prompt/condition)
2. **Computation:** Model receives all samples in the group simultaneously
3. **Output:** Rewards shape `(group_size,)` **in same order as input**

### Example Usage

```python
# Group 1: "a dog" - 3 samples
# Group 2: "a cat" - 2 samples

reward_model.config.batch_size = 5  # Ignored; processes by groups

# Call 1: samples from Group 1 (dog)
rewards_1 = model(
    prompt=["a dog", "a dog", "a dog"],
    video=[vid1, vid2, vid3],
    ...
)  # Returns: RewardModelOutput(rewards=(3,))

# Call 2: samples from Group 2 (cat)
rewards_2 = model(
    prompt=["a cat", "a cat"],
    video=[vid4, vid5],
    ...
)  # Returns: RewardModelOutput(rewards=(2,))
```

## 2.4 RewardModelOutput

```python
@dataclass
class RewardModelOutput(BaseOutput):
    rewards: Union[torch.Tensor, np.ndarray, List[float]]
    extra_info: Optional[Dict[str, Any]] = None
```

---

# 3. VIDEO ADAPTER IMPLEMENTATION

## File Location
`/src/flow_factory/models/wan/wan2_t2v.py` (lines 48-557)

## 3.1 WanT2VSample

```python
@dataclass
class WanT2VSample(T2VSample):
    """Sample output from WAN2 text-to-video model."""
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Inherits all 25 fields from T2VSample
```

## 3.2 Inference Pipeline

### Overview

The `Wan2_T2V_Adapter.inference()` method executes a complete text-to-video generation:

```
Input: prompt, negative_prompt, num_frames, height, width, ...
   ↓
1. Encode Prompt (T5 text encoder)
   ├─ Tokenize and embed
   ├─ Return: prompt_ids, prompt_embeds
   └─ Also prepare negative_prompt embeddings (if CFG)
   ↓
2. Initialize Latent Variables
   ├─ Shape: (batch_size, channels_latent, num_frames, height_latent, width_latent)
   ├─ Denoted: (B, C, T, H, W)
   ├─ Typical: (B, 8, 81, 60, 104) for 480×832 @ 81 frames
   └─ Use generator for deterministic sampling
   ↓
3. Diffusion Loop (typically 50 steps)
   ├─ For each timestep t:
   │  ├─ Predict noise via transformer
   │  ├─ Apply classifier-free guidance (combine cond + uncond)
   │  ├─ Step scheduler (SDE step)
   │  ├─ Collect trajectory (all_latents[t], log_probs[t])
   │  └─ Update latents for next step
   ├─ Collect at indices specified by trajectory_indices
   └─ Optional: compute log probabilities for each step
   ↓
4. VAE Decode Latents
   ├─ Input: latents_final (B, C_latent, T, H_latent, W_latent)
   ├─ Rescale: latents / latent_std + latent_mean (VAE calibration)
   ├─ Decode: vae.decode(latents) → (B, T, C=3, H, W)
   └─ Post-process: video_processor.postprocess_video(output_type='pt')
   ↓
5. Construct WanT2VSample Objects
   └─ One sample per batch element
   └─ Wrap with trajectory + metadata

Output: List[WanT2VSample] (length = batch_size)
```

### Detailed Signature

```python
@torch.no_grad()
def inference(
    self,
    # Text inputs
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    
    # Video dimensions
    height: int = 480,                      # Must be multiple of (vae_scale * patch_size)
    width: int = 832,                       # Must be multiple of (vae_scale * patch_size)
    num_frames: int = 81,                   # Must satisfy (n-1) % vae_scale_temporal == 0
    
    # Diffusion control
    num_inference_steps: int = 50,          # Typically 20-100
    guidance_scale: float = 5.0,            # CFG scale for main transformer
    guidance_scale_2: Optional[float] = None,  # CFG scale for secondary transformer
    generator: Optional[torch.Generator] = None,
    
    # Pre-encoded prompts (skip encoding if provided)
    prompt_ids: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    
    # Trajectory collection
    compute_log_prob: bool = False,         # Compute per-step log probabilities
    trajectory_indices: str = 'all',        # Which steps to collect
    
    # Advanced
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    extra_call_back_kwargs: List[str] = [],
) -> List[WanT2VSample]:
```

## 3.3 Video Decoding

### decode_latents() Method

```python
def decode_latents(
    self,
    latents: torch.Tensor,  # (B, C_latent, T, H_latent, W_latent)
    output_type: Literal['pt', 'pil', 'np'] = 'pil'
) -> torch.Tensor:
    """Decode latent representation to video frames.
    
    Args:
        latents: Latent tensor from diffusion process
        output_type: 
            'pt' → torch.Tensor(B, T, C, H, W)
            'pil' → List[PIL.Image] frames
            'np' → np.ndarray
    
    Returns:
        Decoded video in specified format
    """
    # Step 1: Rescale latents using VAE statistics
    latents = latents.float()
    latents_mean = torch.tensor(
        self.pipeline.vae.config.latents_mean
    ).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    
    latents_std = 1.0 / torch.tensor(
        self.pipeline.vae.config.latents_std
    ).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    
    latents = latents / latents_std + latents_mean
    
    # Step 2: VAE decode
    video = self.pipeline.vae.decode(latents, return_dict=False)[0]
    
    # Step 3: Post-process
    video = self.pipeline.video_processor.postprocess_video(
        video, 
        output_type=output_type
    )
    
    return video  # Tensor(T, C, H, W) or List[PIL.Image]
```

### Video Output Formats

#### PyTorch Tensor Format ('pt')
```
Shape: (T, C, H, W) per sample
- T: number of frames (typically 81)
- C: color channels (3 for RGB)
- H: height (480 for WAN2 preset)
- W: width (832 for WAN2 preset)
- Values: float32, range [0, 1] or [0, 255] depending on post-processor

Example:
    video.shape = torch.Size([81, 3, 480, 832])
    video.dtype = torch.float32
    video.min() ≈ 0.0, video.max() ≈ 1.0
```

#### PIL Format ('pil')
```
Format: List[PIL.Image] - one per frame
- Length: T (typically 81)
- Each image: PIL.Image.Image in RGB mode
- Size: (W, H) = (832, 480)

Example:
    len(video_frames) = 81
    type(video_frames[0]) = PIL.Image.Image
    video_frames[0].size = (832, 480)
    video_frames[0].mode = 'RGB'
```

## 3.4 Sample Construction

### Code from inference() lines 408-435

```python
samples = [
    WanT2VSample(
        # 1. Denoising trajectory
        timesteps=timesteps,                                # (51,) for 50 steps
        all_latents=torch.stack([
            lat[b] for lat in all_latents
        ], dim=0),                                          # (num_collected_steps, C_latent, T, H, W)
        log_probs=torch.stack([
            lp[b] for lp in all_log_probs
        ], dim=0) if all_log_probs is not None else None,  # (num_collected_steps,)
        latent_index_map=latent_index_map,                  # (51,)
        log_prob_index_map=log_prob_index_map,              # (51,)
        
        # 2. Generated media
        video=decoded_videos[b],                            # Tensor(81, 3, 480, 832)
        height=height,                                      # 480
        width=width,                                        # 832
        
        # 3. Prompt information
        prompt=prompt[b] if isinstance(prompt, list) else prompt,
        prompt_ids=prompt_ids[b],                           # (max_seq_len,)
        prompt_embeds=prompt_embeds[b],                     # (max_seq_len, 768) for T5-XXL
        
        # 4. Negative prompt (CFG)
        negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
        negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
        negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
        
        # 5. Extra metadata
        extra_kwargs={
            **{k: v[b] for k, v in extra_call_back_res.items()},
            'callback_index_map': callback_index_map,
        },
    )
    for b in range(batch_size)
]
```

## 3.5 Dimension Constraints

### Frame Count
```python
# WAN2 has temporal VAE scaling factor
vae_scale_factor_temporal = 8  # From pipeline config

# Valid frame counts must satisfy:
(num_frames - 1) % vae_scale_factor_temporal == 0

# Valid options: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, ...

# Adjustment logic:
if (num_frames - 1) % vae_scale_factor_temporal != 0:
    num_frames = (num_frames // vae_scale_factor_temporal) * vae_scale_factor_temporal + 1
```

### Height & Width
```python
# Must be multiples of: vae_scale_factor_spatial * patch_size

vae_scale_factor_spatial = 8    # From VAE config
patch_size = (1, 2, 2)          # From transformer config

h_multiple_of = vae_scale_factor_spatial * patch_size[1] = 16
w_multiple_of = vae_scale_factor_spatial * patch_size[2] = 16

# Valid resolutions: 480×832, 512×512, 576×1024, etc.

# Adjustment logic:
calc_height = height // h_multiple_of * h_multiple_of
calc_width = width // w_multiple_of * w_multiple_of
if height != calc_height or width != calc_width:
    logger.warning(f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})")
    height, width = calc_height, calc_width
```

---

# 4. INTEGRATION GUIDE

## 4.1 Sample Data Flow Example: Text-to-Video

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ADAPTER GENERATES                                         │
└─────────────────────────────────────────────────────────────┘

adapter = Wan2_T2V_Adapter(config, accelerator)
samples = adapter.inference(
    prompt=["a dog jumping", "a cat sleeping"],
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
)

# Result: List[WanT2VSample] of length 2
# samples[0].video.shape = (81, 3, 480, 832)
# samples[0].prompt = "a dog jumping"
# samples[0].all_latents.shape = (num_collected_steps, 8, 81, 60, 104)


┌─────────────────────────────────────────────────────────────┐
│ 2. BATCH SAMPLE OPERATION (Optional)                         │
└─────────────────────────────────────────────────────────────┘

batched = T2VSample.stack(samples)

# Result: Dict with stacked fields
# batched['video'] = Tensor(2, 81, 3, 480, 832)  # Stacked
# batched['prompt'] = ["a dog jumping", "a cat sleeping"]  # As list
# batched['height'] = 480  # Shared (first only)
# batched['width'] = 832   # Shared (first only)


┌─────────────────────────────────────────────────────────────┐
│ 3. REWARD MODEL EVALUATION                                   │
└─────────────────────────────────────────────────────────────┘

reward_model = VideoQualityReward(config, accelerator)
# Assume: reward_model.use_tensor_inputs = True

# Extract video + prompt from samples
videos = [s.video for s in samples]  # List of Tensor(81, 3, 480, 832)
prompts = [s.prompt for s in samples]

# Call reward model
output = reward_model(
    prompt=prompts,
    video=videos,
)

# Result: RewardModelOutput
# output.rewards = Tensor([7.2, 6.8])  # Quality scores for each video
# output.extra_info = {'method': 'CLIP-based quality'}


┌─────────────────────────────────────────────────────────────┐
│ 4. USE REWARDS FOR TRAINING                                  │
└─────────────────────────────────────────────────────────────┘

advantage = compute_advantage(output.rewards, baselines)
loss = compute_loss(samples, output.rewards, advantage)
loss.backward()
optimizer.step()
```

## 4.2 New Video Adapter Template

```python
from flow_factory.samples import T2VSample
from flow_factory.models.abc import BaseAdapter

class MyVideoAdapter(BaseAdapter):
    def inference(
        self,
        prompt: List[str],
        height: int = 512,
        width: int = 512,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        **kwargs
    ) -> List[T2VSample]:
        """Generate videos and return samples."""
        
        # 1. Prepare inputs
        batch_size = len(prompt)
        device = self.device
        
        # 2. Encode prompts
        prompt_embeds = self.encode_prompt(prompt)  # (B, seq_len, hidden_dim)
        
        # 3. Initialize latents
        latents = torch.randn(
            batch_size, 8, num_frames,
            height // 8, width // 8,
            device=device, dtype=self.dtype
        )
        
        # 4. Diffusion loop
        all_latents = []
        timesteps = self.scheduler.timesteps  # e.g., 50 values
        
        for t in timesteps:
            # Predict noise
            noise_pred = self.model(
                x=latents,
                t=t,
                cond=prompt_embeds
            )
            
            # Scheduler step
            latents = self.scheduler.step(
                noise_pred, t, latents
            )
            
            # Collect trajectory
            all_latents.append(latents.clone())
        
        # 5. Decode to video
        videos = self.vae.decode(latents)  # (B, T, C, H, W)
        
        # 6. Construct samples
        samples = [
            T2VSample(
                video=videos[i],                    # Tensor(T, C, H, W)
                height=height,
                width=width,
                prompt=prompt[i],
                prompt_embeds=prompt_embeds[i],
                all_latents=torch.stack(
                    [lat[i] for lat in all_latents], dim=0
                ),
                timesteps=timesteps,
            )
            for i in range(batch_size)
        ]
        
        return samples
```

## 4.3 New Video Reward Model Template

```python
from flow_factory.rewards.abc import PointwiseRewardModel, RewardModelOutput

class MyVideoReward(PointwiseRewardModel):
    """Custom video quality reward model."""
    
    use_tensor_inputs: bool = True  # Prefer torch.Tensor
    required_fields: Tuple = ('video', 'prompt')
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        
        # Load your model
        self.model = load_video_quality_model(
            model_name=config.model_name_or_path
        ).to(self.device).to(self.dtype)
    
    def __call__(
        self,
        prompt: List[str],                                  # (B,)
        video: Optional[List[torch.Tensor]] = None,        # (B,) Tensor(T, C, H, W)
        **kwargs
    ) -> RewardModelOutput:
        """Compute video quality rewards."""
        
        if video is None:
            raise ValueError("video required for video reward model")
        
        batch_size = len(video)
        scores = []
        
        # Process each video
        for vid in video:  # vid: Tensor(T, C, H, W)
            # Sample frames for efficiency
            num_frames_sample = min(vid.shape[0], 8)  # Use at most 8 frames
            frame_indices = torch.linspace(
                0, vid.shape[0] - 1,
                num_frames_sample,
                dtype=torch.long
            )
            frames = vid[frame_indices]  # (num_frames_sample, C, H, W)
            
            # Compute score
            with torch.no_grad():
                score = self.model(frames)  # scalar or (1,)
            
            scores.append(score.item() if isinstance(score, torch.Tensor) else score)
        
        return RewardModelOutput(
            rewards=torch.tensor(scores, device=self.device),
            extra_info={'method': 'my_video_quality_model'}
        )
```

---

## 📌 Quick Reference: Key Shapes

| Component | Shape | Example |
|-----------|-------|---------|
| **Input** |
| prompt | List[str] | ["a dog jumping", "a cat sleeping"] |
| |
| **Sample Fields** |
| image | Tensor(C, H, W) | (3, 512, 512) |
| video | Tensor(T, C, H, W) | (81, 3, 480, 832) |
| prompt_ids | Tensor(seq_len,) | (512,) |
| prompt_embeds | Tensor(seq_len, hidden) | (512, 768) |
| all_latents | Tensor(steps, C_l, T_l, H_l, W_l) | (50, 8, 81, 60, 104) |
| timesteps | Tensor(steps+1,) | (51,) |
| |
| **Batch Operation** |
| stacked_video | Tensor(B, T, C, H, W) | (2, 81, 3, 480, 832) |
| prompts_list | List[str] | ["dog", "cat"] |
| |
| **Reward Output** |
| rewards | Tensor(B,) | (2,) with scores [7.2, 6.8] |

---

## 📚 File Reference

| Component | File | Lines |
|-----------|------|-------|
| BaseSample | samples.py | 65-353 |
| T2VSample | samples.py | 438-441 |
| ImageConditionSample | samples.py | 356-392 |
| VideoConditionSample | samples.py | 394-431 |
| BaseRewardModel | abc.py | 43-76 |
| PointwiseRewardModel | abc.py | 78-118 |
| GroupwiseRewardModel | abc.py | 121-172 |
| WanT2VSample | wan2_t2v.py | 48-51 |
| Wan2_T2V_Adapter.inference | wan2_t2v.py | 252-439 |
| Wan2_T2V_Adapter.decode_latents | wan2_t2v.py | 233-248 |
| Wan2_T2V_Adapter.forward | wan2_t2v.py | 442-556 |

---

**End of Report**
