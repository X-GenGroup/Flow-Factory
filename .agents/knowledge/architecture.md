# Flow-Factory Architecture Overview

## Module Dependency Graph

```
                         ┌──────────┐
                         │ cli.py   │
                         │ train.py │
                         └────┬─────┘
                              │
                    ┌─────────▼─────────┐
                    │     Arguments     │  (hparams/)
                    │  Top-level config │
                    └──┬────┬────┬──────┘
                       │    │    │
          ┌────────────┘    │    └────────────┐
          ▼                 ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  BaseTrainer  │  │ BaseAdapter  │  │BaseRewardModel│
   │  (trainers/)  │  │  (models/)   │  │  (rewards/)  │
   └──┬───┬───┬───┘  └──┬───┬───┬──┘  └──┬───┬───┬───┘
      │   │   │         │   │   │         │   │   │
      ▼   ▼   ▼         ▼   ▼   ▼         ▼   ▼   ▼
    GRPO NFT AWM     Flux SD3 Wan     PickScore CLIP OCR
```

### Key Dependency Rules

| Module | Depends On | Depended By |
|--------|-----------|-------------|
| `hparams/` | (standalone) | Everything |
| `models/abc.py` | `hparams`, `samples`, `ema`, `scheduler`, `utils` | All model adapters, `trainers/abc.py` |
| `trainers/abc.py` | `hparams`, `models/abc.py`, `rewards/`, `data_utils/`, `logger/` | All trainer subclasses |
| `rewards/abc.py` | `hparams` | All reward models, `trainers/abc.py` |
| `data_utils/` | `hparams` | `trainers/abc.py` |
| `scheduler/` | (standalone) | `models/abc.py` |
| `samples/` | (standalone) | `models/`, `rewards/` |

---

## Six-Stage Training Pipeline

> Authoritative reference: `guidance/workflow.md`

```
Stage 1: Data Preprocessing (offline, cached)
  │  GeneralDataset + adapter.preprocess_func()
  │  Text/image/video → encoded tensors (prompt_embeds, image_latents, ...)
  │  Result cached with hash fingerprint
  ▼
Stage 2: K-Repeat Sampling
  │  DistributedKRepeatSampler duplicates each prompt K times
  │  K = training_args.group_size
  ▼
Stage 3: Trajectory Generation
  │  adapter.inference() — full multi-step SDE/ODE denoising
  │  Produces: generated images/videos + trajectory data (noises, log-probs)
  ▼
Stage 4: Reward Computation
  │  RewardProcessor dispatches to Pointwise or Groupwise models
  │  Multi-reward aggregation with configurable weights
  ▼
Stage 5: Advantage Computation
  │  Group-wise normalization of rewards
  │  Algorithm-specific advantage estimation
  ▼
Stage 6: Policy Optimization
  │  adapter.forward() — single-step denoising for loss computation
  │  Policy gradient (GRPO) or weighted matching (NFT/AWM)
  │  Gradient update via accelerator
  ▼
  (Repeat Stages 2–6 for next epoch)
```

---

## Registry System

All three registries follow the same pattern:

```python
# Static dict mapping string → lazy import path
_REGISTRY: Dict[str, str] = {
    'key': 'flow_factory.module.ClassName',
}

# Resolution: registry lookup → fallback to direct Python path → dynamic import
def get_class(identifier: str) -> Type:
    class_path = _REGISTRY.get(identifier.lower(), identifier)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

### Registered Components

**Trainers** (`trainers/registry.py`):
| Key | Class | Paradigm |
|-----|-------|----------|
| `grpo` | `GRPOTrainer` | Coupled |
| `grpo-guard` | `GRPOGuardTrainer` | Coupled |
| `nft` | `DiffusionNFTTrainer` | Decoupled |
| `awm` | `AWMTrainer` | Decoupled |

**Model Adapters** (`models/registry.py`):

> **Terminology**: *Image-to-Image* = single condition image (e.g., FLUX.1-Kontext). *Image(s)-to-Image* = supports multi-image conditioning (e.g., FLUX.2, Qwen-Image-Edit).

| Key | Class | Task |
|-----|-------|------|
| `sd3-5` | `SD3_5Adapter` | Text-to-Image |
| `flux1` | `Flux1Adapter` | Text-to-Image |
| `flux1-kontext` | `Flux1KontextAdapter` | Image-to-Image |
| `flux2` | `Flux2Adapter` | Text-to-Image & Image(s)-to-Image |
| `flux2-klein` | `Flux2KleinAdapter` | Text-to-Image & Image(s)-to-Image |
| `qwen-image` | `QwenImageAdapter` | Text-to-Image |
| `qwen-image-edit-plus` | `QwenImageEditPlusAdapter` | Image(s)-to-Image |
| `z-image` | `ZImageAdapter` | Text-to-Image |
| `wan2_t2v` | `Wan2_T2V_Adapter` | Text-to-Video |
| `wan2_i2v` | `Wan2_I2V_Adapter` | Image-to-Video |
| `wan2_v2v` | `Wan2_V2V_Adapter` | Video-to-Video |

**Reward Models** (`rewards/registry.py`):
| Key | Class | Type |
|-----|-------|------|
| `pickscore` | `PickScoreRewardModel` | Pointwise |
| `pickscore_rank` | `PickScoreRankRewardModel` | Groupwise |
| `clip` | `CLIPRewardModel` | Pointwise |
| `ocr` | `OCRRewardModel` | Pointwise |
| `vllm_evaluate` | `VLMEvaluateRewardModel` | Pointwise |

---

## Extension Points

### Adding a New Model Adapter
1. Create `src/flow_factory/models/<family>/<model>.py`
2. Define a Sample dataclass extending `BaseSample` (or `T2ISample`, `T2VSample`, etc.)
3. Implement `BaseAdapter` subclass with 7 abstract methods: `load_pipeline()`, `encode_prompt()`, `encode_image()`, `encode_video()`, `decode_latents()`, `inference()`, `forward()`
4. Add entry to `_MODEL_ADAPTER_REGISTRY` in `models/registry.py`
5. Reference: `guidance/new_model.md`

### Adding a New Reward Model
1. Create `src/flow_factory/rewards/<reward>.py`
2. Extend `PointwiseRewardModel` or `GroupwiseRewardModel`
3. Implement `__call__()` returning `RewardModelOutput`
4. Add entry to `_REWARD_MODEL_REGISTRY` in `rewards/registry.py`
5. Reference: `guidance/rewards.md`, template: `rewards/my_reward.py`

### Adding a New Algorithm
1. Create `src/flow_factory/trainers/<algorithm>.py`
2. Extend `BaseTrainer`, implement `start()` method
3. Add algorithm-specific `TrainingArguments` subclass in `hparams/training_args.py`
4. Update `get_training_args_class()` in `hparams/training_args.py`
5. Add entry to `_TRAINER_REGISTRY` in `trainers/registry.py`
6. Reference: `guidance/algorithms.md`

---

## Key Design Patterns

### Adapter Pattern (Models)
Each model adapter wraps a diffusers pipeline into the `BaseAdapter` interface. The adapter decomposes the pipeline's monolithic `__call__` into:
- `preprocess_func()` — offline encoding (Stage 1)
- `inference()` — full denoising loop (Stage 3)
- `forward()` — single-step denoising (Stage 6)

### Component Management
`BaseAdapter` automatically discovers pipeline components (text encoders, VAEs, transformers) and manages their lifecycle:
- **Freezing**: Non-trainable components are frozen in `__init__`
- **LoRA**: Applied to `target_components` via `apply_lora()`
- **Offloading**: `on_load_components()` / `off_load_components()` for VRAM management
- **Mode switching**: `train()`, `eval()`, `rollout()` modes

### Reward Processing
`RewardProcessor` handles the dispatch:
- **Pointwise**: Batches samples by `batch_size`, calls reward model per batch
- **Groupwise**: Groups samples by `unique_id`, calls reward model per group
- **Multi-reward**: Aggregates scores from multiple reward models with configurable weights
- **Async**: Optional non-blocking reward computation

### Configuration Hierarchy
```
Arguments (top-level)
├── ModelArguments      # model_type, model_path, finetune_type, LoRA config
├── TrainingArguments   # Algorithm-specific (GRPO/NFT/AWM subclass)
├── DataArguments       # dataset, preprocessing, resolution
├── RewardArguments     # reward_model, batch_size, dtype
├── LogArguments        # logger type, verbose, project name
└── EvalArguments       # evaluation settings
```
