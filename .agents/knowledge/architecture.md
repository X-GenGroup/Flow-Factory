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
| `trainers/abc.py` | `hparams`, `models/abc.py`, `rewards/`, `advantage/`, `data_utils/`, `logger/` | All trainer subclasses |
| `advantage/` | `hparams`, `rewards/`, `samples/` | `trainers/abc.py` |
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
  │  Two sampler strategies (see `.agents/knowledge/topics/samplers.md`):
  │  - GroupContiguousSampler (preferred, auto-selected): keeps K copies on same rank
  │  - DistributedKRepeatSampler (fallback): shuffles K copies across ranks
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
  │  AdvantageProcessor (advantage/advantage_processor.py)
  │  Communication-aware: auto-selects gather vs local path
  │  Strategies: weighted-sum (GRPO) or GDPO
  ▼
Stage 6: Policy Optimization
  │  adapter.forward() — single-step denoising for loss computation
  │  Policy gradient (GRPO) or weighted matching (NFT/AWM) or DPO preference loss
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
| Key | Class | Paradigm | Base Class |
|-----|-------|----------|------------|
| `grpo` | `GRPOTrainer` | Coupled | `BaseTrainer` |
| `grpo-guard` | `GRPOGuardTrainer` | Coupled | `GRPOTrainer` |
| `dpo` | `DPOTrainer` | Decoupled | `BaseTrainer` |
| `nft` | `DiffusionNFTTrainer` | Decoupled | `BaseTrainer` |
| `awm` | `AWMTrainer` | Decoupled | `BaseTrainer` |

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

### Timestep & Sigma Convention

Throughout the codebase, two related but distinct scales are used for time:

| Name | Variable | Scale | Meaning |
|------|----------|-------|---------|
| **Timestep** | `t`, `timestep` | `[0, 1000]` | Scheduler-scale time. All public interfaces (`TimeSampler` outputs, `adapter.forward(t=...)`, `scheduler.step(timestep=...)`) use this scale. |
| **Sigma** | `σ`, `sigma` | `[0, 1]` | Flow-matching noise level. Used for latent interpolation `x_t = (1-σ) x_0 + σ ε` and loss weighting. Obtained via `flow_match_sigma(t) = t / 1000`. |

**Rules**:
- `TimeSampler` always returns `t` in `[0, 1000]`. Trainers pass it directly to `adapter.forward(t=...)` without scaling.
- When interpolating latents or computing noise-level-dependent weights, convert explicitly: `sigma = flow_match_sigma(t)`.
- Each model adapter internally converts `t` to whatever its underlying transformer expects (e.g., Flux divides by 1000, SD3.5 passes as-is). This conversion is encapsulated inside the adapter's `forward()` method.
- `timestep_range=(frac_lo, frac_hi)` is a fraction along the denoising axis from 1000 (noisy) toward 0 (clean), mapped via `t = 1000 * (1 - frac)`. So `(0, 0.99)` corresponds to `t ∈ [10, 1000]`.

### Adapter Pattern (Models)
Each model adapter wraps a diffusers pipeline into the `BaseAdapter` interface. The adapter decomposes the pipeline's monolithic `__call__` into:
- `preprocess_func()` — offline encoding (Stage 1)
- `inference()` — full denoising loop (Stage 3)
- `forward()` — single-step denoising (Stage 6)

**Batch boundary convention**: All inputs to `preprocess_func()`, `encode_image()`, `encode_video()`, `inference()`, and `forward()` carry a batch dimension — tensors have shape `(B, ...)`. `condition_images` at the method level is **model-dependent**: `Tensor(B, C, H, W)` when each sample has one uniform-shape condition image (e.g. Flux1-Kontext), or `List[List[Tensor(C,H,W)]]` of length `B` when samples can have multiple or variable-shape condition images (e.g. Flux2, Qwen-Image-Edit). In both cases, `condition_images[b]` yields the per-sample value and `sample.condition_images` stored on `ImageConditionSample` is always `List[Tensor(C,H,W)]` (no batch dimension). Fields stored on `BaseSample` instances are always **per-sample** (no batch dimension).

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
  - **Local path** (`group_on_same_rank=True`): all K copies on same rank, no cross-rank communication
  - **Distributed path** (`group_on_same_rank=False`): gather → stride-partition → compute → all_reduce → scatter
- **Multi-reward**: Aggregates scores from multiple reward models with configurable weights
- **Async**: Optional non-blocking reward computation

### Advantage Computation
`AdvantageProcessor` (`advantage/advantage_processor.py`) is a standalone, communication-aware component:
- Instantiated in `BaseTrainer._init_reward_model()` and shared by all trainer subclasses
- **Communication optimization**: When `group_on_same_rank=True`, skips `accelerator.gather()` for rewards/ids and uses `all_reduce(count, sum, sum_sq)` for `global_std` computation (3 scalars instead of full tensor gather)
- **Single-gather optimization**: When `group_on_same_rank=False`, packs all rewards + unique_ids into one tensor for a single `accelerator.gather()` call
- **Strategies**: `"sum"` (weighted-sum GRPO) and `"gdpo"` (GDPO-style per-reward normalization)
- All trainers (GRPO, GRPOGuard, NFT, AWM, DPO) delegate to `self.advantage_processor.compute_advantages()`; `DPOTrainer` additionally uses the computed advantages to drive `_form_pairs_from_advantages()` for chosen/rejected pair selection

### Configuration Hierarchy
```
Arguments (top-level)
├── ModelArguments      # model_type, model_path, finetune_type, LoRA config
├── TrainingArguments   # Algorithm-specific (GRPO/NFT/AWM subclass)
├── DataArguments       # dataset, preprocessing, resolution, sampler_type
├── RewardArguments     # reward_model, batch_size, dtype
├── LogArguments        # logger type, verbose, project name
└── EvaluationArguments  # evaluation settings
```

---

## Testing

### Test Commands by Change Area

| Change in | How to verify |
|-----------|--------------|
| `trainers/` | Run training for >= 2 epochs with GRPO (coupled) and NFT or AWM (decoupled) |
| `models/` | Verify with at least 2 model adapters (e.g., Flux + SD3.5 for T2I, Wan for T2V) |
| `rewards/` | Verify with both pointwise (PickScore) and groupwise (PickScore_rank) reward models |
| `hparams/` | Check ALL `examples/` YAML configs parse correctly; run `pytest` |
| `data_utils/` | Verify sampler constraints with different M/K/W/B combinations; test both sampler types |
| `scheduler/` | Verify with SDE dynamics (Flow-SDE) and ODE dynamics |
| `samples/` | Verify `_shared_fields` and collation across affected sample types |
| `advantage/` | Test advantage computation with both `group_contiguous` and `distributed_k_repeat` samplers |
| `ema/` | Verify EMA step/save/load cycle |
| Full regression | `pytest` |
