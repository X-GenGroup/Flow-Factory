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
  │  Two sampler strategies (see `topics/samplers.md`):
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

**Trainer methods vs stages** (each epoch, after Stage 1):

| Method | Stages |
|--------|--------|
| `sample()` | 2–3 (K-repeat batches + `adapter.inference` trajectories) |
| `prepare_feedback()` | 4–5: reward buffer finalize, `AdvantageProcessor` |
| `optimize()` | 6: `adapter.forward` and optimizer step (DPO: form chosen/rejected pairs at entry, then loss) |

---

## Registry System

All three registries map string keys → lazy import paths. Resolution: registry lookup → fallback to direct Python path → dynamic import. See `trainers/registry.py`, `models/registry.py`, `rewards/registry.py` for implementation.

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

- **New model adapter**: `guidance/new_model.md`, skill `/ff-new-model`, conventions `topics/adapter_conventions.md`
- **New reward model**: `guidance/rewards.md`, skill `/ff-new-reward`
- **New algorithm**: `guidance/algorithms.md`, skill `/ff-new-algorithm`

---

## Key Design Patterns

### Timestep & Sigma Convention

Timesteps are `[0, 1000]` (scheduler scale); sigmas are `[0, 1]` (flow-matching noise level). Details: `topics/timestep_sigma.md`.

### Adapter Pattern (Models)
Each model adapter wraps a diffusers pipeline into the `BaseAdapter` interface:
- `preprocess_func()` — offline encoding (Stage 1)
- `inference()` — full denoising loop (Stage 3)
- `forward()` — single-step denoising (Stage 6)

Details: `topics/adapter_conventions.md`

### Component Management
`BaseAdapter` discovers pipeline components and manages lifecycle: freezing, LoRA, offloading, mode switching (`train`/`eval`/`rollout`).

### Reward Processing
`RewardProcessor` dispatches by model type:
- **Pointwise**: batch by `batch_size`
- **Groupwise**: group by `unique_id` (local or distributed path)
- **Multi-reward**: weighted aggregation
- **Async**: optional non-blocking computation

### Advantage Computation
`AdvantageProcessor` (`advantage/advantage_processor.py`): communication-aware, auto-selects gather vs local path. Strategies: `"sum"` (GRPO) and `"gdpo"`. All trainers delegate to `self.advantage_processor.compute_advantages()`.

### Configuration Hierarchy
```
Arguments (top-level)
├── ModelArguments        # model_type, model_path, finetune_type, LoRA config
├── TrainingArguments     # Algorithm-specific (GRPO/DPO/NFT/AWM subclass)
├── SchedulerArguments    # dynamics_type, timestep_range, num_inference_steps
├── DataArguments         # dataset, preprocessing, resolution, sampler_type
├── MultiRewardArguments  # reward_model configs (list of RewardArguments)
├── LogArguments          # logger type, verbose, project name
└── EvaluationArguments   # evaluation settings
```
