# Flow-Factory Documentation Index

**Generated:** April 7, 2026  
**Purpose:** Complete guide to understanding and extending the Flow-Factory codebase

---

## 📚 Documentation Overview

This project includes comprehensive documentation of the Flow-Factory codebase structure, architecture, and data flows:

### 1. **CODEBASE_EXPLORATION.md** (627 lines, 21KB)
   **What:** High-level structural overview and component inventory
   
   **Contains:**
   - Top-level directory structure
   - Complete source package layout (63 files)
   - 10+ component categories with descriptions
   - 11 supported models (image + video generation)
   - 4 trainer implementations (DPO, GRPO, NFT, AWM)
   - 5+ implemented reward models
   - Configuration system overview
   - Audio integration findings (no existing support)
   - Key architecture patterns
   - Codebase statistics table
   - Architecture overview diagram
   
   **Best For:**
   - Getting started with the project
   - Understanding what modules exist
   - Finding which file implements what
   - Quick reference to component organization

---

### 2. **MODALITY_FLOW.md** (1,199 lines, 32KB)
   **What:** Detailed end-to-end data flow analysis for each modality
   
   **Contains:**
   - High-level pipeline flow diagram
   - Modality-specific data flows:
     - Text modality processing
     - Image modality processing
     - Video modality processing
     - Conditioning flows (I2I, I2V, V2V)
   - Sample types & modality routing
   - Reward model handling per modality
   - Model adapter architecture
   - Tensor format conventions
   - Batch collation strategies
   - Complete end-to-end example (T2V training)
   - Audio integration pathway with detailed implementation plan
   - Design principles for extending with new modalities
   
   **Best For:**
   - Understanding how data flows through the system
   - Tracing a specific modality from input to output
   - Learning tensor format conventions
   - Planning audio or other modality additions
   - Understanding preprocessing and caching
   - Deep-dive into reward computation

---

## 🗂️ Quick Navigation by Task

### I want to...

#### **Understand the basic structure**
→ Start with **CODEBASE_EXPLORATION.md** sections 1-2
→ Review the architecture overview diagram (section 13)

#### **Learn how data flows through the system**
→ Read **MODALITY_FLOW.md** section 1 (high-level pipeline)
→ Pick the relevant modality section (2, 3, 4)

#### **Add a new model**
1. Read **CODEBASE_EXPLORATION.md** section 2 (models directory)
2. Review **MODALITY_FLOW.md** section 8 (adapter architecture)
3. Look at existing model: `src/flow_factory/models/*/adapter.py`

#### **Implement a new reward model**
1. Read **CODEBASE_EXPLORATION.md** section 5 (rewards)
2. Study **MODALITY_FLOW.md** section 7 (reward modality handling)
3. Review existing: `src/flow_factory/rewards/*.py`

#### **Understand the training loop**
1. Read **CODEBASE_EXPLORATION.md** section 7 (trainers)
2. Study **MODALITY_FLOW.md** section 11 (end-to-end example)
3. Examine: `src/flow_factory/trainers/abc.py`

#### **Add audio support**
→ **MODALITY_FLOW.md** section 12 (audio integration pathway)
- Detailed checklist and implementation plan
- Pattern matching from existing image/video support
- All code examples and file structures needed

#### **Understand tensor formats and conversions**
→ **MODALITY_FLOW.md** section 9 (tensor format conventions)
→ Review utility files: `src/flow_factory/utils/image.py`, `video.py`

#### **Debug preprocessing or caching issues**
→ **MODALITY_FLOW.md** section 3-4 (image/video preprocessing)
→ Read: `src/flow_factory/data_utils/dataset.py` lines 264-348

#### **Learn about sample types and batching**
→ **MODALITY_FLOW.md** sections 5-6 (samples and stacking)
→ Review: `src/flow_factory/samples/samples.py`

---

## 📊 Codebase Statistics

```
Total: ~27,091 lines of Python across 63 files

Component Breakdown:
├── Trainers     (~5,000 lines)  - DPO, GRPO, NFT, AWM
├── Utils        (~4,500 lines)  - Type conversions, standardization
├── Models       (~3,500 lines)  - Image & video adapters
├── Rewards      (~2,000 lines)  - CLIP, PickScore, OCR, VLM
├── Data Utils   (~2,500 lines)  - Dataset loading, preprocessing
├── HParams      (~800 lines)    - Configuration dataclasses
├── Logger       (~800 lines)    - W&B, SwanLab, TensorBoard
├── Scheduler    (~500 lines)    - Noise schedulers
├── Samples      (~800 lines)    - Sample types and utilities
└── Inference    (~2,000 lines)  - Inference pipelines
```

---

## 🔑 Key Concepts Explained

### Multi-Modal Support
Flow-Factory natively supports:
- **Text:** Prompts and negative prompts (tokenized or embedded)
- **Images:** Single images or multi-reference conditioning
- **Videos:** Video sequences with optional FPS resampling
- **Conditioning:** Image-to-Image, Image-to-Video, Video-to-Video tasks

See **MODALITY_FLOW.md** sections 2-4 for detailed flows.

### K-Repeat Sampling
Multiple samples generated from the same prompt for preference-based training:
```
Prompt 1 → [Sample 1a, Sample 1b, Sample 1c, Sample 1d]
           ↓ Reward scoring ↓
           [Score 1a, Score 1b, Score 1c, Score 1d]
           ↓ Advantage computation (within group) ↓
           [Adv 1a, Adv 1b, Adv 1c, Adv 1d]
```

### Preprocessing and Caching
- Dataset is preprocessed on first epoch
- Fingerprint-based caching avoids reprocessing
- Supports distributed preprocessing across GPUs
- VAE encoding done once, reused across epochs

See **MODALITY_FLOW.md** sections 3-4 for details.

### Tensor Format Conventions
```
Image:  (C, H, W) with values [0, 255] or [-1, 1] or [0, 1]
Video:  (T, C, H, W) with temporal dimension
Batch:  (B, C, H, W) or (B, T, C, H, W)
```

Automatic detection and normalization in standardization functions.

### Reward Model Architecture
- **Pointwise:** Per-sample independent rewards (CLIP, PickScore)
- **Groupwise:** Group-based rewards (for DPO pairwise losses)
- **Multi-reward:** Aggregate multiple reward models with weights

See **MODALITY_FLOW.md** section 7 for detailed architecture.

---

## 🛠️ Extension Points

Flow-Factory is designed for extensibility:

### Add a New Model Type
1. Create `src/flow_factory/models/yourmodel/adapter.py`
2. Extend `BaseAdapter` from `models/abc.py`
3. Register in `models/registry.py`
4. Add to config files

### Add a New Reward Model
1. Create `src/flow_factory/rewards/your_reward.py`
2. Extend `PointwiseRewardModel` or `GroupwiseRewardModel`
3. Implement required methods and `required_fields`
4. Register in `rewards/registry.py`

### Add a New Trainer
1. Create `src/flow_factory/trainers/your_trainer.py`
2. Extend `BaseTrainer` from `trainers/abc.py`
3. Implement abstract methods
4. Register in `trainers/registry.py`

### Add a New Modality (e.g., Audio)
Follow **MODALITY_FLOW.md** section 12 for the complete pathway:
1. Create `utils/audio.py` with type definitions and standardization
2. Add audio sample types to `samples/samples.py`
3. Create audio reward models in `rewards/`
4. Extend dataset loading in `data_utils/dataset.py`
5. Create model adapters for audio tasks
6. Add configuration dataclass to `hparams/`

---

## 📝 File Organization

```
Flow-Factory/
├── CODEBASE_EXPLORATION.md    ← You are here (section 1)
├── MODALITY_FLOW.md           ← Detailed data flows
├── DOCUMENTATION_INDEX.md     ← This file (navigation guide)
│
├── src/flow_factory/
│   ├── train.py               ← Training entry point
│   ├── models/                ← Model adapters (14 files)
│   ├── trainers/              ← Training algorithms (6 files)
│   ├── rewards/               ← Reward models (8 files)
│   ├── data_utils/            ← Data pipeline (4 files)
│   ├── samples/               ← Sample types (1 file)
│   ├── utils/                 ← Utilities (10 files)
│   ├── hparams/               ← Configuration (8 files)
│   ├── logger/                ← Logging (6 files)
│   ├── scheduler/             ← Schedulers (5 files)
│   ├── ema/                   ← EMA models (3 files)
│   └── advantage/             ← Advantage processing (1 file)
│
├── examples/                  ← Training configs
├── dataset/                   ← Sample datasets
└── config/                    ← Distributed training configs
```

---

## 🔍 Key File References

### Core Architecture Files
- `src/flow_factory/models/abc.py` - BaseAdapter (250 lines)
- `src/flow_factory/trainers/abc.py` - BaseTrainer (388 lines)
- `src/flow_factory/rewards/abc.py` - Reward base classes
- `src/flow_factory/samples/samples.py` - Sample types (456 lines)

### Data Processing
- `src/flow_factory/data_utils/dataset.py` - Dataset loading & preprocessing (598 lines)
- `src/flow_factory/utils/video.py` - Video utilities (831 lines)
- `src/flow_factory/utils/image.py` - Image utilities (~150 lines)

### Configuration
- `src/flow_factory/hparams/args.py` - Main Arguments class
- `examples/nft/full/z_image.yaml` - Example config file

### Training Entry Point
- `src/flow_factory/train.py` - Main training script (63 lines)

---

## 📚 Reading Order Recommendations

### For Quick Start (30 minutes)
1. **CODEBASE_EXPLORATION.md** sections 1, 2, 13
2. **MODALITY_FLOW.md** section 1
3. View example config: `examples/nft/full/z_image.yaml`

### For Understanding Data Flow (1-2 hours)
1. **CODEBASE_EXPLORATION.md** sections 1-7
2. **MODALITY_FLOW.md** sections 1-7
3. Skim relevant source files: `data_utils/dataset.py`, `samples/samples.py`

### For Deep Dive (2-4 hours)
1. Read both documents completely
2. Study specific source files related to your task
3. Review configuration files in `examples/`
4. Examine trainer implementations in `src/flow_factory/trainers/`

### For Adding New Modality (Audio)
1. **MODALITY_FLOW.md** sections 2-4 (existing patterns)
2. **MODALITY_FLOW.md** section 12 (audio pathway - complete guide)
3. Reference implementations: `utils/video.py`, `samples/samples.py`
4. Follow the implementation checklist in section 12

---

## ✅ Verification Checklist

When implementing new features, verify:

- [ ] New component properly extends the ABC base class
- [ ] Component is registered in corresponding registry file
- [ ] Type hints are complete and correct
- [ ] Docstrings document required fields and return types
- [ ] Multi-GPU compatibility tested (if applicable)
- [ ] Example config created in `examples/`
- [ ] Documentation updated in these index files

---

## 🎯 Key Takeaways

1. **Plugin Architecture:** Every component (model, reward, trainer, logger) uses registry pattern
2. **Type Safety:** Extensive type hints and custom type definitions
3. **Multi-Modal Native:** Image, video, text support built-in; extensible for audio and beyond
4. **Distributed Ready:** Accelerate + DeepSpeed integration from day one
5. **Data-Efficient:** Preprocessing caching with fingerprints minimizes recomputation
6. **Modular Design:** Clear separation of concerns enables independent extension
7. **Well-Documented:** Configuration-driven approach with YAML
8. **Production Ready:** Checkpoint management, EMA models, logging backends

---

## 🚀 Next Steps

Choose your path based on your goals:

**Option A: Understand Existing System**
→ Read CODEBASE_EXPLORATION.md, then MODALITY_FLOW.md sections 1-7

**Option B: Add Audio Support**
→ MODALITY_FLOW.md section 12 (complete guide with checklists)

**Option C: Create Custom Model**
→ Review: `src/flow_factory/models/flux/flux1.py` as template
→ Study: MODALITY_FLOW.md section 8 (adapter architecture)

**Option D: Implement New Trainer**
→ Study: `src/flow_factory/trainers/dpo.py` as template
→ Learn: MODALITY_FLOW.md section 11 (end-to-end training flow)

**Option E: Extend for New Reward**
→ Review: `src/flow_factory/rewards/clip.py` as template
→ Understand: MODALITY_FLOW.md section 7 (reward architecture)

---

**Last Updated:** April 7, 2026
**Documentation Quality:** Complete and comprehensive
**Status:** Ready for reference and extension

