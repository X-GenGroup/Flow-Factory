# Flow-Factory Development Guide

## Project Overview

Flow-Factory is a unified **online RL fine-tuning framework** for diffusion/flow-matching models. It provides a modular architecture where trainers, model adapters, and reward models are independently extensible via a registry-based plugin system.

- **Algorithms**: GRPO, GRPO-Guard, DiffusionNFT, AWM
- **Models**: FLUX.1/2, SD3.5, Wan2.1/2.2, Qwen-Image, Z-Image
- **Rewards**: PickScore, CLIP, OCR, VLM-Evaluate, and custom rewards
- **Python**: >=3.10 | **PyTorch**: >=2.6.0 | **License**: Apache-2.0

## Key Technical Requirements

- Distributed training via **Accelerate** (primary) and **DeepSpeed** (ZeRO-1/2/3)
- Model adapters wrap **diffusers** pipelines into a unified `BaseAdapter` interface
- Training follows a **6-stage pipeline**: Data Preprocessing → K-Repeat Sampling → Trajectory Generation → Reward Computation → Advantage Computation → Policy Optimization
- Configuration via **Pydantic-based dataclasses** (`hparams/`) and YAML config files

## Core Operating Principles

1. **Read constraints first** — Before making changes, consult `.agents/knowledge/constraints.md` and `.agents/knowledge/architecture.md`
2. **Cross-component awareness** — Changes to shared code (`abc.py` in trainers/models/rewards) affect ALL subclasses
3. **Consult existing docs** — The `guidance/` directory contains authoritative documentation for workflow, algorithms, rewards, and new model integration
4. **Structured planning** — For complex multi-file tasks, plan before implementing
5. **Verify across algorithms** — Test changes under GRPO, NFT, and AWM when touching shared trainer code
6. **Escalation when stuck** — After three failed approaches, document findings and request review

## Critical Restrictions

- **Never break base class interfaces** — `BaseTrainer`, `BaseAdapter`, `BaseRewardModel` abstract method signatures are contracts; changes require updating ALL subclasses
- **Never mix reward paradigms** — Pointwise and Groupwise reward models have different input/output contracts; don't interchange them
- **Never modify registry entries without updating imports** — Registry maps (`_TRAINER_REGISTRY`, `_MODEL_ADAPTER_REGISTRY`, `_REWARD_MODEL_REGISTRY`) use lazy import paths that must match actual module locations
- **DeepSpeed ZeRO-3 is not supported** — Reward model sharding bugs persist; do NOT use ZeRO-3 (see `trainers/abc.py` comment)
- **Config key changes silently break YAML** — Renaming or removing Pydantic fields in `hparams/` requires updating ALL example configs under `examples/`. Adding new user-facing fields also requires adding them to ALL example configs with default values and `# Options:` comments so users can discover them.

## Development Commands

```bash
# Installation
pip install -e "."              # Core only
pip install -e ".[all]"         # With DeepSpeed + quantization
pip install -e ".[deepspeed]"   # DeepSpeed only

# Training
ff-train <config.yaml>          # Main entry point
flow-factory-train <config.yaml> # Alternative

# Code Quality
black --check src/              # Format check
isort --check src/              # Import sort check
pytest                          # Run tests
```

## Project Structure

```
src/flow_factory/
├── trainers/          # RL algorithms (GRPO, NFT, AWM) — extend BaseTrainer
├── models/            # Model adapters (FLUX, SD3.5, Wan, ...) — extend BaseAdapter
├── rewards/           # Reward models (PickScore, CLIP, ...) — extend BaseRewardModel
├── data_utils/        # Dataset loading, preprocessing, sampling
├── hparams/           # Pydantic config dataclasses (Arguments, *Args)
├── logger/            # Experiment tracking (Wandb, SwanLab, ...)
├── scheduler/         # SDE/ODE scheduler (Flow-SDE, Dance-SDE, CPS, ODE)
├── ema/               # EMA parameter management
├── samples/           # Sample dataclasses (BaseSample, T2ISample, ...)
├── utils/             # Shared utilities
├── cli.py             # CLI entry point
└── train.py           # Main training orchestration
```

## Documentation Reference

| Document | Purpose |
|----------|---------|
| `guidance/workflow.md` | 6-stage training pipeline with code examples |
| `guidance/algorithms.md` | GRPO, DiffusionNFT, AWM deep dive |
| `guidance/rewards.md` | Reward system design, custom model creation |
| `guidance/new_model.md` | Step-by-step model adapter integration |

## Available Skills

| Skill | Purpose | Use When |
|-------|---------|----------|
| `/ff-develop` | Feature development with impact analysis | Implementing new functionality or refactoring |
| `/ff-debug` | Bug fixing with structured protocol | Debugging errors, crashes, unexpected behavior |
| `/ff-review` | Pre-commit code review | Before committing changes |
| `/ff-new-model` | Model adapter integration | Adding support for a new diffusion model |
| `/ff-new-reward` | Reward model integration | Adding a new reward function |
| `/ff-new-algorithm` | RL algorithm integration | Adding a new training algorithm |

## Commit & PR Conventions

- **Commit messages**: Concise, descriptive, in English
- **PR title format**: `[{modules}] {type}: {description}` (e.g., `[trainer,reward] feat: add multi-reward weighting`)
- **Valid types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Run code quality checks before committing
