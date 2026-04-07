# Flow-Factory Modality Data Flow Analysis

**Date:** April 7, 2026  
**Focus:** Detailed data flow for multi-modal support and extensibility patterns

---

## 1. HIGH-LEVEL PIPELINE FLOW

```
User Input (YAML Config)
    ↓
Arguments.load_from_yaml()
    ↓
[Model Adapter] ← [Data Pipeline] ← [Dataset]
    ↓                   ↓
Generate Samples   Preprocess
    ↓
[Reward Models] (Multi-modal evaluation)
    ↓
[Advantage Processor] (Score → Advantage)
    ↓
[Trainer] (Gradient update)
    ↓
[Logger] (Metrics)
    ↓
[EMA Model] (Policy update)
```

---

## 2. TEXT MODALITY DATA FLOW

### Data Input Format
```yaml
# JSONL entry
{
  "prompt": "a cat sitting on a table",
  "negative_prompt": "blurry, low quality"
}
```

### Processing Pipeline
```
1. Dataset Load (dataset.py)
   ├─ Read JSONL line
   ├─ Extract "prompt" → batch["prompt"]
   └─ Extract "negative_prompt" → batch["negative_prompt"]

2. Preprocessing (dataset.py, lines 290-296)
   ├─ Text tokenization (via preprocess_func)
   ├─ Store as prompt_ids: torch.Tensor
   └─ Store embeddings if needed (prompt_embeds)

3. Model Processing (model adapters)
   ├─ Text encoder forward pass
   ├─ Generate from embeddings
   └─ Output: Image or Video tensors

4. Sample Creation (samples.py)
   ├─ prompt: str
   ├─ prompt_ids: torch.Tensor
   └─ prompt_embeds: Optional[torch.Tensor]

5. Reward Computation
   ├─ Pass prompt to reward models
   ├─ CLIP: Image-text alignment
   ├─ OCR: Text recognition matching
   └─ Aggregate multi-reward scores
```

### Code Locations
- **Loading:** `data_utils/dataset.py` lines 265-296
- **Sample:** `samples/samples.py` lines 91-97 (prompt fields)
- **Rewards:** `rewards/clip.py`, `rewards/ocr.py`

---

## 3. IMAGE MODALITY DATA FLOW

### Data Input Format
```yaml
# JSONL entry with images
{
  "prompt": "edit this image",
  "images": ["image1.jpg", "image2.jpg"]
}

# Directory structure
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── data.jsonl
```

### Processing Pipeline
```
1. Dataset Load (dataset.py)
   ├─ Read JSONL: "images" field
   ├─ Image filenames: ["image1.jpg", "image2.jpg", ...]
   └─ Construct full paths: [dataset_dir/images/image1.jpg, ...]

2. Image Loading (dataset.py, lines 299-321)
   ├─ Load via PIL.Image.open()
   ├─ Convert to torch.Tensor
   ├─ Shape: (3, H, W) with values in [0, 255]
   └─ Return List[torch.Tensor]

3. Standardization (utils/image.py)
   ├─ normalize_to_uint8()
   │  ├─ Detect range: [-1, 1] → multiply by 127.5 + 128
   │  ├─ Detect range: [0, 1] → multiply by 255
   │  └─ Already [0, 255] → pass through
   ├─ standardize_image_batch()
   │  ├─ Input: PIL.Image | torch.Tensor | np.ndarray
   │  ├─ Output: torch.Tensor(C, H, W) or List[torch.Tensor]
   │  └─ Format: 'pt' for pytorch, 'np' for numpy, 'pil' for PIL
   └─ Convert to VAE latents (in model)

4. Preprocessing with Caching (dataset.py, lines 264-348)
   ├─ Compute fingerprint: hash_pil_image_list(images)
   ├─ Check cache directory
   ├─ If cached: load preprocessed latents
   ├─ If not: forward through VAE
   │  └─ VAE encode: PIL → latent tensors
   ├─ Store cache with fingerprint
   └─ Return preprocessed batch

5. Sample Creation (samples.py)
   ├─ image: Optional[ImageSingle]
   │  └─ Auto-standardized to (C, H, W) tensor
   ├─ condition_images: Optional[List[torch.Tensor]]
   │  └─ Each shape (C, H, W)
   └─ __post_init__: auto-standardize (lines 134-140)

6. Reward Computation
   ├─ Pass image to reward models
   ├─ CLIP: Image-text alignment
   ├─ PickScore: Aesthetic + alignment
   ├─ VLM: Vision-language evaluation
   └─ Aggregate scores

7. Model Forward
   ├─ Input: condition_images (latent form)
   ├─ Diffusion steps with guidance
   └─ Output: Generated image tensor
```

### Code Locations
- **Loading:** `data_utils/dataset.py` lines 299-321
- **Standardization:** `utils/image.py` (~150 lines)
- **Caching:** `data_utils/dataset.py` lines 264-348
- **Sample Types:** `samples/samples.py` lines 356-446
- **Rewards:** `rewards/clip.py`, `rewards/pick_score.py`, `rewards/vllm_evaluate.py`

---

## 4. VIDEO MODALITY DATA FLOW

### Data Input Format
```yaml
# JSONL entry with videos
{
  "prompt": "generate a video of",
  "videos": ["video1.mp4", "video2.mp4"]
}

# Directory structure
dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── data.jsonl
```

### Processing Pipeline
```
1. Dataset Load (dataset.py)
   ├─ Read JSONL: "videos" field
   ├─ Video filenames: ["video1.mp4", "video2.mp4", ...]
   └─ Construct full paths

2. Video Frame Loading (dataset.py, lines 323-348)
   ├─ Call load_video_frames() [lines 518-539]
   ├─ Using imageio.v3.imopen()
   ├─ Optional FPS resampling:
   │  ├─ Source FPS: detect from file
   │  ├─ Target FPS: config.video_fps
   │  ├─ Example: 30 fps → 8 fps
   │  └─ Downsample: keep every (30/8) frames
   ├─ Return List[PIL.Image] (frames)
   └─ One entry per video file

3. Video Standardization (utils/video.py, lines 1-200)
   ├─ normalize_video_to_uint8()
   │  ├─ Auto-detect range: [-1,1], [0,1], or [0,255]
   │  └─ Normalize to [0, 255]
   ├─ standardize_video_batch()
   │  ├─ Input: List[PIL.Image] | torch.Tensor | np.ndarray
   │  ├─ Input shape options:
   │  │  ├─ Tensor(T, H, W, C) → (T, C, H, W)
   │  │  ├─ Tensor(B, T, C, H, W) → batch form
   │  │  └─ List[PIL.Image] → (T, C, H, W)
   │  ├─ Output shape: (T, C, H, W) or batch (B, T, C, H, W)
   │  └─ Format options: 'pt', 'np', 'pil'
   └─ Result: torch.Tensor(T, C, H, W)

4. Preprocessing with Caching (dataset.py, lines 323-348)
   ├─ Compute fingerprint: hash_pil_image_list(frames)
   ├─ Check cache directory
   ├─ If cached: load preprocessed video latents
   ├─ If not: forward through VAE
   │  └─ Process each frame or use temporal VAE
   ├─ Store cache with fingerprint
   └─ Return preprocessed batch (T, latent_dim)

5. Sample Creation (samples.py)
   ├─ video: Optional[VideoSingle]
   │  └─ Auto-standardized to (T, C, H, W) tensor
   ├─ condition_videos: Optional[List[torch.Tensor]]
   │  └─ Each shape (T, C, H, W)
   └─ __post_init__: auto-standardize (lines 142-144)

6. Reward Computation
   ├─ Pass video frames to reward models
   ├─ Sample evenly-spaced frames (e.g., 4 frames from T frames)
   ├─ CLIP: Frame-text alignment (average across frames)
   ├─ VLM: Multi-frame video understanding
   └─ Aggregate scores

7. Model Forward
   ├─ Input: condition_videos (latent form)
   ├─ Diffusion steps with temporal attention
   ├─ Guidance applied across frames
   └─ Output: Generated video tensor (T, C, H, W)
```

### Code Locations
- **Loading:** `data_utils/dataset.py` lines 323-348
- **Video Frame Loading:** `data_utils/dataset.py` lines 518-539
- **Standardization:** `utils/video.py` (~831 lines)
- **Caching:** `data_utils/dataset.py` lines 264-348
- **Sample Types:** `samples/samples.py` lines 394-456
- **Rewards:** `rewards/clip.py` (with frame sampling), `rewards/vllm_evaluate.py`

### Video Loading Function
```python
# From data_utils/dataset.py lines 518-539
def load_video_frames(video_path: str, fps: Optional[int] = None) -> List[Image.Image]:
    """
    Load video frames with optional FPS resampling.
    
    Args:
        video_path: Path to video file
        fps: Target FPS for resampling (None = keep original)
    
    Returns:
        List[PIL.Image]: Frames extracted from video
    """
    # Implementation uses imageio.v3.imopen()
    # Supports FPS resampling via frame index calculation
```

---

## 5. CONDITIONING MODALITY FLOWS

### Image Conditioning (I2I, I2V Tasks)
```
Input Structure:
{
  "prompt": "edit this image",
  "images": ["image_to_edit.jpg"]  # Condition image
}

Processing:
Condition Image Load → Standardize → VAE Encode → Latent Conditioning
    ↓
Model Input: prompt_embeds + condition_latents
    ↓
Diffusion steps with conditioning
    ↓
Output: Modified/generated image
```

### Video Conditioning (V2V Tasks)
```
Input Structure:
{
  "prompt": "extend this video",
  "videos": ["video_to_condition.mp4"]  # Condition video
}

Processing:
Condition Video Load → Frame Extraction → Standardize → VAE Encode → Latent Conditioning
    ↓
Model Input: prompt_embeds + condition_latents
    ↓
Temporal diffusion with frame conditioning
    ↓
Output: Generated video with temporal consistency
```

### Multi-Reference Conditioning
```
Example: I2V with multiple reference images
{
  "prompt": "animate these characters",
  "images": ["char1.jpg", "char2.jpg", "char3.jpg"]
}

Processing:
- Load all condition images
- Encode each to latent space
- Concatenate along batch/sequence dimension
- Pass to model for multi-reference guidance
```

### Code Locations
- **Sample Classes:** `samples/samples.py` lines 356-446
  - `ImageConditionSample`: Image conditioning support
  - `VideoConditionSample`: Video conditioning support
  - `I2ISample`: Image-to-Image with condition_images
  - `I2VSample`: Image-to-Video with condition_images
  - `V2VSample`: Video-to-Video with condition_videos

---

## 6. SAMPLE TYPES & MODALITY ROUTING

### Sample Type Hierarchy
```
BaseSample (Base class with common fields)
├── Image/Video fields: auto-standardized in __post_init__
├── Prompt fields: text and embeddings
├── Trajectory fields: timesteps, latents, log_probs
└── extra_kwargs: flexible extensibility

↓

Specialized Types:
├── ImageConditionSample
│   ├─ Adds: condition_images: List[torch.Tensor]
│   ├─ _id_fields: include 'condition_images'
│   └─ Subclasses: I2ISample, I2VSample
│
├── VideoConditionSample
│   ├─ Adds: condition_videos: List[torch.Tensor]
│   ├─ _id_fields: include 'condition_videos'
│   └─ Subclasses: V2VSample
│
└── Direct Subclasses:
    ├─ T2ISample (Text → Image)
    ├─ T2VSample (Text → Video)
    └─ I2ISample, I2VSample, V2VSample (inherited conditioning)
```

### Sample Type Determination
```python
# In model adapters (models/*/adapter.py)
def _get_sample_class(task_type: str):
    """Route to correct sample type based on task"""
    mapping = {
        'text-to-image': T2ISample,
        'image-to-image': I2ISample,
        'text-to-video': T2VSample,
        'image-to-video': I2VSample,
        'video-to-video': V2VSample,
    }
    return mapping[task_type]
```

### Code Locations
- **Sample Classes:** `samples/samples.py` lines 65-456
- **Sample Stacking:** `samples/samples.py` lines 325-353
  - `BaseSample.stack()`: Batch multiple samples
  - Handles shared fields (height, width)
  - Stacks tensors with matching shapes
  - Recursively processes nested dicts

---

## 7. REWARD MODEL MODALITY HANDLING

### Reward Model Interface
```python
class PointwiseRewardModel(BaseRewardModel):
    def __call__(
        self,
        prompt: List[str],                              # Text modality
        image: Optional[List[Image.Image]] = None,      # Image modality
        video: Optional[List[List[Image.Image]]] = None,  # Video modality
        condition_images: Optional[...] = None,         # Condition images
        condition_videos: Optional[...] = None,         # Condition videos
        **kwargs
    ) -> RewardModelOutput:
        """Compute per-sample rewards"""
```

### Modality-Specific Implementations

#### CLIP Rewards (Multi-modal alignment)
```
Input Processing:
├─ Text: prompt → tokenize → text encoder
├─ Image: image → image encoder (single frame)
├─ Video: video frames → sample K evenly-spaced frames → image encoder
└─ Average frame embeddings for video

Reward Computation:
├─ Image: cosine_similarity(text_emb, image_emb)
├─ Video: mean(cosine_similarity(text_emb, frame_embs))
└─ Output: score ∈ [0, 1] or higher scale

Location: rewards/clip.py
```

#### PickScore (Aesthetic + Alignment)
```
Input Processing:
├─ Image: image → PickScore model
├─ Prompt: used for caption alignment
└─ Combines aesthetic and semantic scores

Reward Computation:
├─ Per-image scoring
├─ Not applied to videos directly
└─ Output: aesthetic score

Location: rewards/pick_score.py
```

#### OCR Rewards (Text Recognition)
```
Input Processing:
├─ Image: image → PaddleOCR → extracted text
├─ Video: frames → extract text from each frame
├─ Prompt: expected text to find

Reward Computation:
├─ Levenshtein distance between recognized and expected
├─ Distance → reward score (0-1 or higher)
└─ Combines frame scores for videos

Location: rewards/ocr.py
```

#### VLM Evaluation (Vision-Language Model)
```
Input Processing:
├─ Image: image → VLM encoder
├─ Video: frames → VLM (multi-frame understanding)
├─ Prompt: question/instruction to VLM
└─ Optional condition images/videos for context

Reward Computation:
├─ VLM generates response to prompt
├─ Response → scoring function (custom)
└─ Output: scalar reward

Location: rewards/vllm_evaluate.py
```

### Reward Processing Pipeline
```python
# rewards/reward_processor.py
class RewardProcessor:
    def __call__(
        self,
        samples: List[BaseSample],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        1. Extract required fields from samples based on reward_model.required_fields
        2. Convert tensors ↔ PIL format as needed (use_tensor_inputs flag)
        3. Call each reward model
        4. Stack/aggregate results
        5. Return: {reward_name: scores_tensor}
        """
```

### Code Locations
- **Base Classes:** `rewards/abc.py` lines 1-150
- **CLIP:** `rewards/clip.py`
- **PickScore:** `rewards/pick_score.py`
- **OCR:** `rewards/ocr.py`
- **VLM:** `rewards/vllm_evaluate.py`
- **Processor:** `rewards/reward_processor.py`

---

## 8. MODEL ADAPTER MODALITY HANDLING

### Adapter Architecture
```python
class BaseAdapter(ABC):
    """Base class for all model adapters"""
    
    # Component management
    inference_modules: List[str]       # Modules loaded during inference
    preprocessing_modules: List[str]   # Modules for preprocessing (VAE, text encoder)
    target_module_map: Dict[str, ...]  # LoRA target modules
    
    # Modality support
    preprocess_func: Callable           # Custom preprocessing function
    
    @abstractmethod
    def forward(
        self,
        prompt_embeds: torch.Tensor,
        image: Optional[torch.Tensor] = None,    # For I2I, I2V
        video: Optional[torch.Tensor] = None,    # For V2V
        negative_prompt_embeds: Optional[...] = None,
        height: int = 512,
        width: int = 512,
        num_frames: Optional[int] = None,        # For video
        **kwargs
    ) -> ImageSample | VideoSample:
        pass
```

### Multi-Modal Task Handling
```
Text-to-Image (T2I):
Input: prompt_embeds(B, seq_len, 768)
Output: image tensor (B, C, H, W)

Image-to-Image (I2I):
Input: prompt_embeds + condition_image_latents
Output: modified image (B, C, H, W)

Text-to-Video (T2V):
Input: prompt_embeds
Output: video tensor (B, T, C, H, W)

Image-to-Video (I2V):
Input: prompt_embeds + condition_image_latents
Output: video (B, T, C, H, W)

Video-to-Video (V2V):
Input: prompt_embeds + condition_video_latents
Output: video (B, T, C, H, W)
```

### Component Loading Strategy
```python
# From trainers/abc.py lines 209-233
def _load_inference_components(self, trainable_module_names):
    """
    Load non-trainable components needed at inference time.
    
    Modules already on device (from accelerator.prepare):
    - Trainable modules (e.g., LoRA layers)
    
    Modules to load here:
    - Inference modules (transformer, VAE decoder, scheduler)
    - Preprocessing modules (if preprocess disabled)
    
    Dynamic resolution with component_name mapping:
    - Group names (e.g., 'unet') → concrete names ('unet_block1', 'unet_block2')
    - Deduplicate and exclude already-prepared modules
    - Load to accelerator device
    """
```

### Code Locations
- **Base Adapter:** `models/abc.py` (~250 lines)
- **Model Implementations:**
  - `models/flux/flux1.py` - FLUX.1 adapter
  - `models/flux/flux2.py` - FLUX.2 adapter
  - `models/wan/wan2_t2v.py` - Wan T2V adapter
  - `models/wan/wan2_i2v.py` - Wan I2V adapter
  - `models/wan/wan2_v2v.py` - Wan V2V adapter
  - `models/qwen_image/qwen_image.py` - Qwen image adapter
  - `models/stable_diffusion/sd3_5.py` - SD3.5 adapter
  - `models/z_image/z_image.py` - Z-Image adapter

---

## 9. TENSOR FORMAT CONVENTIONS

### Image Format
```
Single Image: (C, H, W)
- C: 3 for RGB, 1 for grayscale
- H: Height (e.g., 512, 1024)
- W: Width (e.g., 512, 1024)
- Values: uint8 [0, 255], float16/32 [-1, 1] or [0, 1]

Batch Images: (B, C, H, W)
- B: Batch size
- Rest same as single

Multi-Image (Conditioning): List[(C, H, W)]
- For multi-reference conditioning
- Each image separately encoded
```

### Video Format
```
Single Video: (T, C, H, W)
- T: Temporal frames (e.g., 8, 25, 120)
- C: 3 for RGB
- H: Height (e.g., 384, 720)
- W: Width (e.g., 720, 1280)
- Values: uint8 [0, 255], float16/32 [-1, 1] or [0, 1]

Batch Videos: (B, T, C, H, W)
- B: Batch size
- Rest same as single

Multi-Video (Conditioning): List[(T, C, H, W)]
- For multi-reference video conditioning
```

### Tensor Value Ranges
```
Detection in standardize_*_batch() functions:

Range [-1, 1]:
- Source: Model outputs (many diffusion models)
- Convert to [0, 1]: (tensor + 1) / 2
- Convert to [0, 255]: ((tensor + 1) / 2 * 255).uint8

Range [0, 1]:
- Source: Post-sigmoid outputs
- Convert to [0, 255]: (tensor * 255).uint8

Range [0, 255]:
- Source: PIL images, uint8 numpy arrays
- Pass through

numpy array format: (H, W, C) or (T, H, W, C)
- Convert to PyTorch: (C, H, W) or (T, C, H, W)
```

### Code Locations
- **Image Utils:** `utils/image.py` (~150 lines)
- **Video Utils:** `utils/video.py` (~831 lines)
- **Base Utils:** `utils/base.py` (type definitions)

---

## 10. BATCH COLLATION WITH MIXED SHAPES

### Challenge: Variable-Size Images/Videos
```
Dataset Example:
- Sample 1: image (3, 512, 512)
- Sample 2: image (3, 1024, 1024)  ← Different size!
- Sample 3: image (3, 512, 512)

Naive stacking fails: shapes incompatible

Solution in data_utils/loader.py:
```

### Collation Strategy
```python
def custom_collate_fn(batch):
    """
    Handles mixed image/video sizes in batch.
    
    Strategy:
    1. Group samples by output size (height, width)
    2. Stack samples of same size
    3. Return mixed-size batch or pad to max
    4. Store metadata for later reshaping
    """
```

### Sample Stacking (BaseSample.stack())
```python
# samples/samples.py lines 325-353
@classmethod
def stack(cls, samples: List[BaseSample]) -> Dict:
    """
    Stack multiple samples into batch structure.
    
    Processing:
    1. Shared fields (height, width) → return first only
    2. Tensor fields → stack if shapes match, else list
    3. Dict fields → recursively stack
    4. Other fields → return as list
    """
    # Example:
    stacked = BaseSample.stack([
        T2ISample(prompt="cat", image=tensor1),
        T2ISample(prompt="dog", image=tensor2),
    ])
    # Result:
    # {
    #   'prompt': ['cat', 'dog'],
    #   'image': stack([tensor1, tensor2]) if same shape
    #   'height': 512,  # shared (first only)
    #   'width': 512    # shared (first only)
    # }
```

### Code Locations
- **Collation:** `data_utils/loader.py`
- **Sample Stacking:** `samples/samples.py` lines 325-353
- **Shared Fields:** `samples/samples.py` lines 73-76

---

## 11. END-TO-END EXAMPLE: TEXT-TO-VIDEO TRAINING

### Data Preparation Phase
```
Input: examples/nft/full/z_image.yaml
   ↓
Arguments.load_from_yaml()
   ↓
Create DataArguments:
  - dataset_dir: "dataset/video_dataset"
  - image_dir: None (T2V has no image conditioning)
  - video_dir: None (output only, no condition videos)
  - enable_preprocess: True
  - preprocessing_batch_size: 32
```

### Dataset Initialization
```
GeneralDataset.__init__()
   ↓
Load JSONL file:
{
  "prompt": "a cat walking through a garden",
  "negative_prompt": "blurry, distorted"
}
   ↓
Store file list for lazy loading
```

### Preprocessing (First Epoch)
```
for batch in dataloader:
   ↓
dataset._preprocess_batch()
   ↓
1. Text Processing:
   - prompt → tokenize → (seq_len,)
   - negative_prompt → tokenize → (seq_len,)
   - Encoder: embed → (seq_len, 768)
   
2. VAE Preprocessing:
   - text_encoder.encode(prompts)
   - Store: prompt_embeds tensor
   
3. Caching:
   - Compute fingerprint from prompts
   - Save to cache_dir/fingerprint.pkl
   - Store embeddings for future epochs
   ↓
Return preprocessed batch
```

### Sample Generation
```
model_adapter.forward(prompt_embeds)
   ↓
1. Model Inference:
   - Initialize noise: (B, T, C, H, W)
   - Denoise steps: t from 999 to 0
   - Each step: transformer(latent, t, condition=prompt_embeds)
   - Apply guidance: pred = pred + w * (pred_cond - pred_uncond)
   
2. Collect Trajectory:
   - timesteps: [999, 980, 961, ..., 0]
   - all_latents: (num_steps, latent_dim)
   - log_probs: (num_steps,)
   
3. Decode to Video:
   - VAE.decode(final_latent)
   - Output: (B, T, C, H, W) video tensor
   ↓
Return: T2VSample with video, trajectory
```

### Reward Computation
```
Multi-reward aggregation:
   ↓
1. CLIP Reward (Image-Text Alignment):
   - Sample 4 frames from video: [frame_0, frame_25%, frame_50%, frame_100%]
   - For each frame: CLIP_score = cosine(text_emb, image_emb)
   - Average: clip_reward = mean(scores)
   - Result: scalar ∈ [0, ~1]
   
2. VLM Reward (Video Understanding):
   - VLM processes video frames
   - Prompt: "Does this video match the description?"
   - VLM output → score
   - Result: scalar score
   
3. Aggregate:
   - final_reward = w1 * clip_reward + w2 * vlm_reward
   - w1, w2 from config
   ↓
Return: reward_dict = {'clip': score, 'vlm': score}
```

### Advantage Computation
```
AdvantageProcessor:
   ↓
K-repeat sampling (group_size=4):
- 4 samples from same prompt
- K-repeat groups: [[sample1, sample2, sample3, sample4], ...]
   ↓
Groupwise advantage:
   ↓
For each group:
- Base score: min/mean of group rewards
- Advantage per sample: sample_reward - base_score
- Normalize: advantage / std(advantages)
   ↓
Return: advantage tensor per sample
```

### Gradient Update
```
Trainer.optimize():
   ↓
For each sample in batch:
- Get advantage
- Get trajectory (timesteps, latents, log_probs)
- Compute loss:
  ↓
  1. Predict noise: pred = unet(latent, t, prompt_embeds)
  2. Target noise: target = noise
  3. MSE: loss = (pred - target)^2
  4. Weight by advantage: loss = loss * advantage
  ↓
5. Aggregate: batch_loss = mean(loss)
6. Backward + update
7. Update EMA model
   ↓
Return: loss metrics
```

### Logging & Checkpointing
```
log_data():
   ↓
Metrics to log:
- training_loss: scalar
- avg_reward: scalar
- avg_advantage: scalar
- learning_rate: scalar
   ↓
Logger.log_data():
- W&B: upload to dashboard
- TensorBoard: write events
- Console: print [Step XXXX | Epoch YYY] metrics
   ↓
Every N steps:
save_checkpoint():
- Save adapter state (LoRA weights if used)
- Save optimizer state
- Save EMA model weights
```

---

## 12. AUDIO INTEGRATION PATHWAY

### Proposed Audio Modality Support

Following existing patterns from video/image implementations:

#### 1. Audio Utilities Module
```python
# src/flow_factory/utils/audio.py (NEW)

from typing import List, Union
import torch
import numpy as np
from scipy import signal
import librosa

# Type definitions
AudioSingle = Union[np.ndarray, torch.Tensor]
AudioBatch = Union[List[np.ndarray], List[torch.Tensor], torch.Tensor]
MultiAudioBatch = List[AudioBatch]

class Audio:
    """Audio utilities parallel to video.py"""
    
    @staticmethod
    def load_audio(
        audio_path: str,
        sr: int = 16000,
        mono: bool = True
    ) -> torch.Tensor:
        """Load audio file to tensor (1, time) or (channels, time)"""
        waveform, _ = librosa.load(audio_path, sr=sr, mono=mono)
        return torch.from_numpy(waveform)
    
    @staticmethod
    def normalize_audio_to_uint8(audio: AudioSingle) -> AudioSingle:
        """Auto-detect range and normalize to [0, 255]"""
        # Similar to normalize_video_to_uint8()
        pass
    
    @staticmethod
    def standardize_audio_batch(
        audio: AudioBatch,
        output_type: str = 'pt'
    ) -> torch.Tensor | List[torch.Tensor]:
        """Convert audio to standardized format"""
        # Parallel to standardize_video_batch()
        pass
    
    @staticmethod
    def audio_to_spectrogram(
        audio: torch.Tensor,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 64
    ) -> torch.Tensor:
        """Convert audio waveform to mel-spectrogram"""
        # For vision models that work with spectrograms
        pass
    
    @staticmethod
    def spectrogram_to_image(
        spectrogram: torch.Tensor,
        cmap: str = 'viridis'
    ) -> torch.Tensor:
        """Convert spectrogram to RGB image for T2I models"""
        pass
```

#### 2. Sample Types for Audio
```python
# In src/flow_factory/samples/samples.py (EXTEND)

@dataclass
class AudioConditionSample(BaseSample):
    """Sample for tasks with audio conditioning."""
    _id_fields: ClassVar[frozenset[str]] = BaseSample._id_fields | frozenset({'condition_audio'})
    
    condition_audio: Optional[torch.Tensor] = None  # (channels, time)

@dataclass
class T2ASample(BaseSample):
    """Text-to-Audio sample"""
    audio: Optional[torch.Tensor] = None  # (channels, time)

@dataclass
class A2ISample(AudioConditionSample):
    """Audio-to-Image (audio as conditioning)"""
    image: Optional[torch.Tensor] = None

@dataclass
class A2VSample(AudioConditionSample):
    """Audio-to-Video (audio as conditioning)"""
    video: Optional[torch.Tensor] = None
```

#### 3. Audio Reward Models
```python
# src/flow_factory/rewards/audio_similarity.py (NEW)

class AudioSimilarityReward(PointwiseRewardModel):
    """Audio similarity / alignment reward"""
    required_fields = ('prompt', 'audio')
    use_tensor_inputs = True
    
    def __call__(
        self,
        prompt: List[str],
        audio: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> RewardModelOutput:
        """Score audio-text alignment"""
        # Use audio embedding model (e.g., CLAP)
        # Compute cosine similarity with text embeddings

# src/flow_factory/rewards/music_alignment.py (NEW)

class MusicAlignmentReward(PointwiseRewardModel):
    """Music genre/mood/tempo alignment with text"""
    required_fields = ('prompt', 'audio')
    use_tensor_inputs = True
    
    def __call__(
        self,
        prompt: List[str],
        audio: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> RewardModelOutput:
        """Score music properties match prompt"""
        # Extract: genre, mood, tempo, instruments from audio
        # Match with prompt description
```

#### 4. Audio Data Pipeline
```python
# In src/flow_factory/data_utils/dataset.py (EXTEND)

# Add to DataArguments:
audio_dir: Optional[str] = None
audio_sr: int = 16000  # Sample rate

# Add to _preprocess_batch():
if 'audio' in batch:
    # Load audio files
    audio_paths = batch['audio']
    audio_list = []
    for path in audio_paths:
        audio = load_audio(
            os.path.join(self.config.audio_dir, path),
            sr=self.config.audio_sr
        )
        audio_list.append(audio)
    batch['audio'] = audio_list
```

#### 5. Audio Model Adapters
```python
# src/flow_factory/models/audio_to_image/a2i.py (NEW)

class Audio2ImageAdapter(BaseAdapter):
    """Audio-to-Image generation"""
    
    def forward(
        self,
        prompt_embeds: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ImageSample:
        """Generate image from audio prompt"""
        # Convert audio → spectrogram → image representation
        # Pass to FLUX or similar image model
        # Output: (B, C, H, W) image

# src/flow_factory/models/music_to_video/m2v.py (NEW)

class Music2VideoAdapter(BaseAdapter):
    """Music-to-Video generation with rhythm sync"""
    
    def forward(
        self,
        prompt_embeds: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        **kwargs
    ) -> VideoSample:
        """Generate video from music with temporal sync"""
        # Extract audio features: beat, tempo, energy, etc.
        # Use as guidance for video generation
        # Output: (B, T, C, H, W) video
```

#### 6. Configuration Changes
```yaml
# examples/audio/t2a.yaml (NEW)
launcher: accelerate
data_args:
  dataset_dir: dataset/audio_dataset
  audio_dir: auto  # Discover audio/ subdirectory
  preprocessing_batch_size: 32
  enable_preprocess: true

model_args:
  model_name: "t2a-musicgen"
  torch_dtype: bfloat16

training_args:
  trainer_type: nft
  per_device_batch_size: 4
  group_size: 4

rewards:
  - name: audio_similarity
    type: audio_similarity
    weight: 1.0
```

#### 7. Integration Points Summary
```
Data Layer:
- Load audio files (librosa/soundfile)
- Preprocessing caching with fingerprints
- Audio-specific preprocessing functions

Sample Layer:
- AudioConditionSample, A2ISample, A2VSample, T2ASample
- Auto-standardization in __post_init__
- unique_id including audio hash

Model Layer:
- Audio feature extraction
- Audio-to-spectrogram-to-image conversion
- Cross-modal conditioning

Reward Layer:
- AudioSimilarityReward
- MusicAlignmentReward
- Multi-reward aggregation

Trainer Layer:
- No changes needed (works with any sample type)
```

### Implementation Checklist
- [ ] Create `utils/audio.py` with audio I/O and standardization
- [ ] Add audio sample types to `samples/samples.py`
- [ ] Implement audio reward models in `rewards/`
- [ ] Extend `data_utils/dataset.py` for audio loading
- [ ] Create audio model adapters (T2A, A2I, A2V, M2V)
- [ ] Add audio configuration dataclass to `hparams/`
- [ ] Create example configs in `examples/audio/`
- [ ] Add audio tests to `tests/` directory
- [ ] Document audio modality in README

---

## 13. DESIGN PRINCIPLES & EXTENSIBILITY

### Key Patterns for Adding New Modalities

#### Pattern 1: Type Standardization
```python
# For each modality, create utils/MODALITY.py:
- Define: SingleType, BatchType, MultiBatchType
- Implement: normalize_to_uint8()/normalize_to_standard()
- Implement: standardize_batch() (input → standardized format)
- Support: conversions between PIL, NumPy, PyTorch
```

#### Pattern 2: Sample Class Hierarchy
```python
# In samples/samples.py:
- Extend BaseSample with modality-specific fields
- Add to _id_fields if used for grouping
- Add to _shared_fields if batch-wide
- Create subclasses for specific tasks (T2X, X2Y, etc.)
```

#### Pattern 3: Data Loading
```python
# In data_utils/dataset.py:
- Add modality_dir to DataArguments
- Extend _preprocess_batch() with modality handling
- Create load_MODALITY_frames() function
- Add caching with fingerprints
```

#### Pattern 4: Reward Models
```python
# In rewards/:
- Create RewardModel subclasses
- Define required_fields tuple
- Set use_tensor_inputs flag
- Implement __call__ signature matching interface
```

#### Pattern 5: Model Adapters
```python
# In models/MODALITY/:
- Extend BaseAdapter
- Implement preprocessing_modules, inference_modules
- Override forward() with modality-specific logic
- Handle sample type determination
```

### Extension Points Illustrated
```
Flow-Factory is designed as a modular system:

Dataset → Model → Reward Model → Trainer
  ↓         ↓         ↓            ↓
Load    Generate   Evaluate     Update
 ↓         ↓         ↓            ↓
Auto-   Trajectory Scores →   Advantages
detect  Collect    Aggregate    Loss
```

Each layer can be extended independently while maintaining compatibility.

---

## CONCLUSION

Flow-Factory's architecture demonstrates a mature, extensible design for multi-modal generative model training. The systematic handling of images, videos, and text through standardized utilities, type systems, and modular components creates a clear pathway for audio integration and beyond.

Key strengths:
1. **Modular Design:** Each component is independent and replaceable
2. **Type System:** Clear conventions for data formats and conversions
3. **Scalability:** Distributed training built in from the start
4. **Extensibility:** Plugin architecture for rewards, models, trainers
5. **Quality:** Comprehensive preprocessing, caching, and validation

