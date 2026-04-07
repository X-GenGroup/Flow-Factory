# Flow-Factory Codebase Exploration Report

**Date:** April 7, 2026  
**Total Lines of Code:** ~27,091 lines of Python  
**Project Type:** Distributed Flow Matching / Diffusion Model Training Framework

---

## 1. TOP-LEVEL DIRECTORY STRUCTURE

```
Flow-Factory/
├── src/flow_factory/           # Main package (27,091 lines)
├── dataset/                     # Sample datasets (8 different datasets)
├── config/                      # Distributed training configs (accelerate, deepspeed)
├── examples/                    # Training examples (awm, dpo, grpo, nft, template)
├── guidance/                    # Guidance implementations
├── inference/                   # Inference code
├── reward_server/               # External reward server
├── multinode_examples/          # Multi-node training examples
├── README.md, LICENSE, etc.
```

---

## 2. MAIN SOURCE PACKAGE STRUCTURE (`src/flow_factory/`)

### **Core Modules:**

```
src/flow_factory/
├── __init__.py
├── cli.py                       # Command-line interface
├── train.py                     # Main training entry point (63 lines)

├── models/                      # Model adapters (image/video generation)
│   ├── abc.py                   # BaseAdapter abstract class
│   ├── loader.py                # Model loader registry
│   ├── registry.py              # Model registration
│   ├── flux/                    # Flux models (1 & 2)
│   │   ├── flux1.py             # FLUX.1 adapter
│   │   ├── flux1_kontext.py     # FLUX.1 with context
│   │   ├── flux2.py             # FLUX.2 adapter
│   │   └── flux2_klein.py       # FLUX.2 Klein adapter
│   ├── wan/                     # Wan models (video generation)
│   │   ├── wan2_t2v.py          # Text-to-Video
│   │   ├── wan2_i2v.py          # Image-to-Video
│   │   └── wan2_v2v.py          # Video-to-Video
│   ├── qwen_image/              # Qwen image models
│   │   ├── qwen_image.py        # Base Qwen image adapter
│   │   └── qwen_image_edit_plus.py
│   ├── stable_diffusion/        # SD models
│   │   └── sd3_5.py             # SD3.5 adapter
│   └── z_image/                 # Z-Image models
│       └── z_image.py           # Z-Image adapter

├── trainers/                    # Training algorithms
│   ├── abc.py                   # BaseTrainer abstract class
│   ├── loader.py                # Trainer loader
│   ├── registry.py              # Trainer registration
│   ├── dpo.py                   # Diffusion-DPO trainer
│   ├── grpo.py                  # GRPO trainer
│   ├── nft.py                   # NFT (Negative Feedback Training) trainer
│   └── awm.py                   # AWM (Adaptive Weighting Model) trainer

├── data_utils/                  # Data pipeline
│   ├── dataset.py               # GeneralDataset class (multi-modal)
│   ├── loader.py                # DataLoader creation
│   ├── sampler.py               # Custom samplers
│   └── sampler_loader.py        # Sampler loader

├── samples/                     # Sample/trajectory data structures
│   ├── samples.py               # BaseSample + specialized sample types
│   │   - BaseSample             # Base class for all samples
│   │   - ImageConditionSample   # Image conditioning
│   │   - VideoConditionSample   # Video conditioning
│   │   - T2ISample              # Text-to-Image
│   │   - T2VSample              # Text-to-Video
│   │   - I2ISample              # Image-to-Image
│   │   - I2VSample              # Image-to-Video
│   │   - V2VSample              # Video-to-Video

├── rewards/                     # Reward models
│   ├── abc.py                   # Base reward classes
│   │   - BaseRewardModel
│   │   - PointwiseRewardModel
│   │   - GroupwiseRewardModel
│   ├── loader.py                # Reward model loader
│   ├── registry.py              # Reward registration
│   ├── reward_processor.py      # Processes rewards
│   ├── clip.py                  # CLIP-based rewards
│   ├── pick_score.py            # PickScore reward
│   ├── ocr.py                   # OCR reward (PaddleOCR)
│   ├── vllm_evaluate.py         # VLM evaluation rewards
│   ├── my_reward.py             # Custom reward template
│   └── my_reward_remote.py      # Remote reward server

├── scheduler/                   # Noise schedulers
│   ├── abc.py                   # Scheduler base class
│   ├── loader.py                # Scheduler loader
│   ├── registry.py              # Scheduler registration
│   ├── flow_match_euler_discrete.py
│   └── unipc_multistep.py

├── ema/                         # Exponential Moving Average
│   ├── ema.py                   # EMA implementation
│   ├── ema_utils.py
│   └── __init__.py

├── advantage/                   # Advantage estimation
│   ├── advantage_processor.py
│   └── __init__.py

├── hparams/                     # Hyperparameter dataclasses
│   ├── args.py                  # Main Arguments class
│   ├── abc.py                   # Base argument class
│   ├── data_args.py             # DataArguments
│   ├── model_args.py            # ModelArguments
│   ├── training_args.py         # TrainingArguments
│   ├── reward_args.py           # RewardArguments
│   ├── scheduler_args.py        # SchedulerArguments
│   └── log_args.py              # LogArguments

├── logger/                      # Logging backends
│   ├── abc.py                   # Base logger class
│   ├── loader.py                # Logger loader
│   ├── registry.py              # Logger registration
│   ├── wandb.py                 # W&B integration
│   ├── swanlab.py               # SwanLab integration
│   ├── tensorboard.py           # TensorBoard integration
│   └── formatting.py

├── utils/                       # Utility functions (27,091 total lines)
│   ├── base.py                  # Core utilities
│   │   - filter_kwargs()
│   │   - split_kwargs()
│   │   - create_generator()
│   │   - Hash utilities
│   │   - Grid latent utilities
│   ├── image.py                 # Image type conversions
│   │   - ImageSingle, ImageBatch, MultiImageBatch types
│   │   - normalize_to_uint8()
│   │   - standardize_image_batch()
│   │   - pil_image_to_tensor(), tensor_to_pil_image()
│   ├── video.py                 # Video type conversions (831 lines)
│   │   - VideoSingle, VideoBatch, MultiVideoBatch types
│   │   - normalize_video_to_uint8()
│   │   - standardize_video_batch()
│   │   - tensor_to_video_frames(), video_frames_to_tensor()
│   │   - load_video_frames() with fps support
│   ├── checkpoint.py            # Checkpoint saving/loading
│   ├── dist.py                  # Distributed utilities
│   ├── logger_utils.py          # Logging setup
│   ├── memory_tracker.py        # Memory tracking
│   ├── imports.py               # Import utilities
│   ├── noise_schedule.py        # Timestep sampling
│   ├── reward_utils.py          # Reward utilities
│   └── trajectory_collector.py  # Trajectory collection

└── inference/                   # Inference pipelines
```

---

## 3. MODALITY HANDLING ARCHITECTURE

### **Key Finding: Multi-Modal Support is Built-in**

Flow-Factory supports **3 primary modalities:**

1. **Images (Image Generation)**
   - Models: FLUX.1, FLUX.2, Z-Image, Qwen-Image, SD3.5
   - Input: Text prompts → Output: PIL Images or Tensors
   - Tensor format: `(C, H, W)` or batch `(B, C, H, W)`

2. **Videos (Video Generation)**
   - Models: Wan2 (T2V, I2V, V2V)
   - Input: Text/Image prompts → Output: Video frames (PIL or Tensors)
   - Tensor format: `(T, C, H, W)` or batch `(B, T, C, H, W)`
   - Built-in video loading: `load_video_frames()` from video files

3. **Conditioning Modalities**
   - Condition images: Multi-ref conditioning images
   - Condition videos: Multi-ref conditioning videos
   - Text conditions: Prompts and negative prompts

### **Modality-Related Code Search Results:**

**Found "modality/modal" keyword in:**
- `src/flow_factory/data_utils/dataset.py` (1 occurrence)
  - `PreprocessCallable` protocol handles multi-modal inputs
  - Supports prompts, images, videos simultaneously

**Video files (27 matches):**
- Core video utilities: `utils/video.py` (831 lines)
- Models: All video generation models
- Data pipeline: Dataset and loader
- Rewards: Video evaluation rewards
- Samples: Video sample types

**Audio files (1 match):**
- Only in OCR reward comments: torchaudio installation note

---

## 4. DATA PIPELINE / DATASET HANDLING

### **GeneralDataset Class** (`data_utils/dataset.py`)

**Features:**
- **Multi-modal Support:**
  - Loads from JSONL (structured) or TXT (prompts only)
  - Supports images, videos, and text prompts simultaneously
  - Directory-based: `images/`, `videos/` subdirectories

- **Preprocessing Pipeline:**
  - Batched preprocessing with caching
  - Distributed preprocessing across GPUs
  - Fingerprint-based cache management
  - Custom preprocess functions (e.g., for embeddings)

- **Video Loading:**
  ```python
  load_video_frames(video_path, fps=None)
  # Returns List[PIL.Image] with optional FPS resampling
  ```

- **Sharding for Distributed Training:**
  - Global or local parallelism modes
  - K-repeat distributed sampling
  - Group contiguous sampler

- **Data Format:**
  ```python
  # JSONL example:
  {
    "prompt": "a cat sitting on a table",
    "negative_prompt": "blurry, low quality",
    "images": ["image1.jpg", "image2.jpg"],  # Optional
    "videos": ["video1.mp4"]                 # Optional
  }
  ```

### **Sample Classes** (`samples/samples.py`)

**Base Sample Structure:**
```python
@dataclass
class BaseSample:
    # Denoising trajectory
    timesteps: Optional[torch.Tensor]      # (T+1,)
    all_latents: Optional[torch.Tensor]    # (num_steps, Seq_len, C)
    log_probs: Optional[torch.Tensor]      # (num_steps,)
    
    # Generated media (auto-standardized to tensor)
    image: Optional[ImageSingle]           # → (C, H, W) tensor
    video: Optional[VideoSingle]           # → (T, C, H, W) tensor
    
    # Prompt info
    prompt: Optional[str]
    prompt_ids: Optional[torch.Tensor]
    prompt_embeds: Optional[torch.Tensor]
    
    # Negative prompt info
    negative_prompt: Optional[str]
    negative_prompt_ids: Optional[torch.Tensor]
    negative_prompt_embeds: Optional[torch.Tensor]
```

**Specialized Samples:**
- `T2ISample` - Text-to-Image
- `T2VSample` - Text-to-Video
- `I2ISample` - Image-to-Image
- `I2VSample` - Image-to-Video
- `V2VSample` - Video-to-Video
- `ImageConditionSample` - With image conditioning
- `VideoConditionSample` - With video conditioning

---

## 5. REWARD MODEL INFRASTRUCTURE

### **Architecture:** (`rewards/abc.py`)

**Base Classes:**
```python
class BaseRewardModel(ABC):
    required_fields: Tuple[str, ...]  # Fields needed from Sample
    use_tensor_inputs: bool           # Tensors vs PIL Images
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> RewardModelOutput:
        """Compute rewards"""
        pass

class PointwiseRewardModel(BaseRewardModel):
    """Per-sample rewards (independent)"""
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs
    ) -> RewardModelOutput:
        pass

class GroupwiseRewardModel(BaseRewardModel):
    """Group-based rewards (pairwise/ranking)"""
    # Same signature but processes entire groups
```

**Implemented Reward Models:**
1. **CLIP Rewards** (`clip.py`)
   - Image-text alignment
   - Supports both PointwiseRewardModel and GroupwiseRewardModel

2. **PickScore** (`pick_score.py`)
   - Aesthetic and caption alignment
   - Pointwise rewards

3. **OCR Rewards** (`ocr.py`)
   - PaddleOCR-based text recognition
   - Text-image matching
   - Uses Levenshtein distance

4. **VLM Evaluation** (`vllm_evaluate.py`)
   - Vision-Language Model evaluation
   - Flexible scoring

5. **Custom Reward Template** (`my_reward.py`, `my_reward_remote.py`)
   - For user implementations

**Reward Processing:**
- `RewardProcessor`: Aggregates multiple rewards
- `RewardBuffer`: Buffers and groups rewards
- Multi-reward orchestration

---

## 6. CONFIGURATION SYSTEM

### **Hierarchical Configuration** (`hparams/`)

**Main Configuration Class:**
```python
@dataclass
class Arguments:
    launcher: Literal['accelerate']
    config_file: Optional[str]              # Distributed config
    num_processes: int
    mixed_precision: Optional[Literal['no', 'fp16', 'bf16']]
    
    # Nested argument groups
    data_args: DataArguments
    model_args: ModelArguments
    scheduler_args: SchedulerArguments
    training_args: TrainingArguments
    eval_args: EvaluationArguments
    log_args: LogArguments
    reward_args: MultiRewardArguments
    eval_reward_args: Optional[MultiRewardArguments]
```

**YAML Loading:**
```python
config = Arguments.load_from_yaml("config.yaml")
```

**Data Configuration:**
```python
@dataclass
class DataArguments:
    dataset_dir: str                           # Dataset folder
    image_dir: Optional[str]                   # Image subfolder
    video_dir: Optional[str]                   # Video subfolder
    preprocessing_batch_size: int              # Cache batch size
    enable_preprocess: bool                    # Preprocessing enabled
    force_reprocess: bool                      # Force cache rebuild
    max_dataset_size: Optional[int]            # Dataset limit
    preprocess_parallelism: Literal["global", "local"]
    sampler_type: Literal["auto", "distributed_k_repeat", "group_contiguous"]
```

**Training Configuration:**
```python
@dataclass
class TrainingArguments:
    trainer_type: str                          # dpo, grpo, nft, awm
    num_train_timesteps: Optional[int]         # Timestep range
    time_sampling_strategy: str                # Timestep distribution
    per_device_batch_size: int
    group_size: int                            # K in K-repeat sampling
    unique_sample_num_per_epoch: int           # Total unique samples
    gradient_accumulation_steps: Union[int, str]  # auto or number
    learning_rate: float
    ema_decay: float
    ema_update_interval: int
    # ... many more parameters
```

### **Example Configuration File:**
- Location: `examples/nft/full/z_image.yaml`
- Organized YAML with clear sections:
  - Environment (launcher, distributed config)
  - Data (dataset, preprocessing)
  - Model (architecture, checkpoint)
  - Training (optimization, sampling)
  - Rewards (multiple reward models)
  - Evaluation

---

## 7. TRAINING LOOP & TRAINER CODE

### **BaseTrainer** (`trainers/abc.py`)

**Key Methods:**
```python
class BaseTrainer(ABC):
    def __init__(self, accelerator, config, adapter)
    def should_continue_training(self) -> bool
    def log_data(self, data: Dict[str, Any], step: int)
    def _init_reward_model(self)
    @abstractmethod
    def start(self)
```

**Trainer Implementations:**

1. **DPO Trainer** (`trainers/dpo.py`)
   - Diffusion-DPO implementation
   - K-repeat sampling → score → form pairs
   - Velocity MSE loss: `L = -log σ(-β/2 * Δerr)`
   - Loss: `|noise_pred - (noise - x_0)|²`

2. **GRPO Trainer** (`trainers/grpo.py`)
   - Group Relative Policy Optimization
   - Advantage computation within groups
   - Supports groupwise rewards

3. **NFT Trainer** (`trainers/nft.py`)
   - Negative Feedback Training
   - GDPO advantage aggregation
   - EMA policy for off-policy sampling

4. **AWM Trainer** (`trainers/awm.py`)
   - Adaptive Weighting Model
   - Dynamic weight adjustment

**Common Training Flow:**
```
1. Sample generation (with K repeats per prompt)
2. Reward computation
3. Advantage estimation
4. Pair/group formation
5. Gradient update
6. EMA model update (optional)
7. Logging and checkpoint saving
```

---

## 8. AUDIO-RELATED CODE

### **Current Status: NO NATIVE AUDIO SUPPORT**

**Search Results:**
- Only 1 mention of "audio": In `rewards/ocr.py` comments
- Reference: `torchaudio==2.8.0` installation note for PyTorch

**Implications:**
- Audio modality NOT implemented
- No audio loading utilities
- No audio-to-image/video models
- No audio reward models
- Perfect opportunity for extension!

---

## 9. KEY ARCHITECTURE PATTERNS

### **Plugin Architecture:**
- **Registry-based:** Models, trainers, rewards, loggers, schedulers
- **Loader pattern:** Each component has a `load_*` function
- **Factory pattern:** Dynamic component creation from config

### **Type System:**
- Extensive use of Python type hints
- Custom type aliases for clarity:
  - `VideoSingle`, `VideoBatch`, `MultiVideoBatch`
  - `ImageSingle`, `ImageBatch`, `MultiImageBatch`
- Protocol classes for callbacks

### **Distributed Training:**
- **Accelerate Framework:** Primary distributed backend
- **DeepSpeed Support:** Zero-1, Zero-2, Zero-3
- **FSDP Support:** Full shard and grad+op shard
- **Multi-node:** Explicit multi-node examples
- **Rank-aware:** Logging, progress bars, checkpoint saving

### **Modular Design:**
- Clear separation of concerns
- Easy to extend with new models, rewards, trainers
- Custom preprocessing functions
- Pluggable loss functions

---

## 10. SUPPORTED MODELS

### **Image Generation:**
- FLUX.1 (dev)
- FLUX.1 with Kontext
- FLUX.2 (full)
- FLUX.2 Klein
- Stable Diffusion 3.5
- Qwen-Image
- Qwen-Image-EditPlus
- Z-Image
- Z-Image-Turbo

### **Video Generation:**
- Wan2 Text-to-Video (T2V)
- Wan2 Image-to-Video (I2V)
- Wan2 Video-to-Video (V2V)

### **Video Models Details:**
- Resolutions: 384×720, configurable fps
- Frame counts: Configurable (1-N frames)
- Inference steps: 8-28 steps
- Guidance scales: Model-specific

---

## 11. KEY FINDINGS FOR AUDIO INTEGRATION

### **Potential Integration Points:**

1. **Sample Types:** Add `AudioSingle`, `A2ISample`, `A2VSample`, etc.
2. **Data Loading:** Add audio file loading utilities (librosa, soundfile)
3. **Reward Models:** Audio similarity, music-text alignment
4. **Model Adapters:** Audio-to-Image/Video model adapters
5. **Utils:** Audio type conversions, normalization, augmentation

### **Natural Extensions:**

```
# Audio utilities (parallel to video.py)
utils/audio.py
├── AudioSingle, AudioBatch types
├── normalize_audio()
├── standardize_audio_batch()
├── load_audio_frames() with resampling
└── Audio-to-spectrogram conversions

# Audio models
models/audio_to_image/
models/music_to_video/
models/audio_to_video/

# Audio rewards
rewards/audio_similarity.py
rewards/music_alignment.py
```

---

## 12. CODEBASE STATISTICS

| Component | Files | Lines |
|-----------|-------|-------|
| Models | 14 | ~3,500 |
| Trainers | 6 | ~5,000 |
| Rewards | 8 | ~2,000 |
| Data Utils | 4 | ~2,500 |
| Utils | 10 | ~4,500 |
| HParams | 8 | ~800 |
| Logger | 6 | ~800 |
| Scheduler | 5 | ~500 |
| Samples | 2 | ~800 |
| **TOTAL** | **63** | **~27,091** |

---

## 13. SUMMARY: ARCHITECTURE OVERVIEW

```
Flow-Factory: Multi-Modal Flow Matching Training Framework
│
├─ Models Layer (Adapters)
│  ├─ Image Models: FLUX, Z-Image, SD3.5, Qwen
│  └─ Video Models: Wan2 (T2V, I2V, V2V)
│
├─ Training Layer
│  ├─ Trainers: DPO, GRPO, NFT, AWM
│  ├─ Samples: Multi-modal sample types
│  └─ Advantage Processing: Reward aggregation
│
├─ Data Layer
│  ├─ Multi-modal dataset loading
│  ├─ Distributed preprocessing with caching
│  ├─ Video/image loading utilities
│  └─ K-repeat sampling strategies
│
├─ Reward Layer
│  ├─ Pointwise & groupwise rewards
│  ├─ CLIP, PickScore, OCR implementations
│  ├─ Multi-reward orchestration
│  └─ Scalable reward processing
│
├─ Infrastructure
│  ├─ Distributed training (Accelerate, DeepSpeed)
│  ├─ Configuration management (YAML)
│  ├─ Logging backends (W&B, SwanLab, TensorBoard)
│  └─ Checkpointing & EMA models
│
└─ Utilities
   ├─ Image type conversions & standardization
   ├─ Video type conversions & standardization
   ├─ Hashing & memory tracking
   └─ Distributed utilities
```

**KEY INSIGHT:** The codebase is architecture is extremely well-designed for multi-modal support with clear extension points. Audio integration would follow the same patterns as video.

