# Flow-Factory Development Guide

## Project Overview

Flow-Factory is a unified **online RL fine-tuning framework** for diffusion/flow-matching models. It provides a modular architecture where trainers, model adapters, and reward models are independently extensible via a registry-based plugin system.

- **Algorithms**: GRPO, GRPO-Guard, DPO, DiffusionNFT, AWM
- **Models**: FLUX.1/2, SD3.5, Wan2.1/2.2, Qwen-Image, Z-Image
- **Rewards**: PickScore, CLIP, OCR, VLM-Evaluate, and custom rewards
- **Python**: >=3.10 | **PyTorch**: >=2.6.0 | **License**: Apache-2.0

**Language**: Match user's language.

## Context Loading

On session start, read **Tier 1** (see `.agents/knowledge/README.md`):
- `.agents/knowledge/philosophy.md` — design principles, coding style index
- `.agents/knowledge/constraints.md` — hard rules, indexed by category
- `.agents/knowledge/architecture.md` — module graph, pipeline stages, registries

**Tier 2**: Topic docs triggered by change area. See `knowledge/README.md` for triggers.

## Key Technical Requirements

- Distributed training via **Accelerate** (primary) and **DeepSpeed** (ZeRO-1/2)
- Model adapters wrap **diffusers** pipelines into a unified `BaseAdapter` interface
- Training follows a **6-stage pipeline**: Data Preprocessing → K-Repeat Sampling → Trajectory Generation → Reward Computation → Advantage Computation → Policy Optimization
- Configuration via **Pydantic-based dataclasses** (`hparams/`) and YAML config files

## Core Operating Principles

1. **Constraints first** — Read `constraints.md` + `architecture.md` before changes; search codebase before attempting fixes.
2. **Cross-component awareness** — Changes to `abc.py` affect ALL subclasses; verify across algorithms (GRPO + NFT/AWM).
3. **Plan before implement** — Multi-file tasks -> TodoWrite. Plan must state which skills apply.
4. **Challenge first, execute second** — Spot logic flaws or simpler alternatives? Raise before executing.
5. **Escalation** — After three failed approaches, document findings and request review.
6. **Fix capture** — After every bug fix, generate summary per `topics/fix_patterns.md` template.
7. **English-only docs** — All code comments, docstrings, commit messages, and agent docs must be English.

Hard rules: see `constraints.md`.

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
├── trainers/          # RL algorithms (GRPO, DPO, NFT, AWM) — extend BaseTrainer
├── models/            # Model adapters (FLUX, SD3.5, Wan, ...) — extend BaseAdapter
├── rewards/           # Reward models (PickScore, CLIP, ...) — extend BaseRewardModel
├── advantage/         # Advantage computation (AdvantageProcessor, communication-aware)
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

### Quick Decision Guide

- **"Add support for model X"** -> `/ff-new-model`
- **"Add a new reward function"** -> `/ff-new-reward`
- **"Add a new training algorithm"** -> `/ff-new-algorithm`
- **"Fix this error" / "training hangs" / "wrong results"** -> `/ff-debug`
- **"Add a new capability" / "refactor" / "clean up"** -> `/ff-develop`
- **"Review before committing"** -> `/ff-review`

## Commit & PR Conventions

- **Commit messages**: Concise, descriptive, in English
- **PR title format**: `[{modules}] {type}: {description}` (e.g., `[trainer,reward] feat: add multi-reward weighting`)
- **Valid types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Run code quality checks before committing

## Commit Flow

1. Complete and verify the change.
2. Update related documentation: `guidance/`, `examples/`, `.agents/knowledge/` — if the change introduces, modifies, or removes any API, config field, or workflow.
3. Run `/ff-review` skill.
4. **safe** -> commit. **risky** -> report to user, wait for approval.
5. Each fix -> immediate commit. Do not batch unrelated changes.
6. Run `black --check src/ && isort --check src/` before every commit.
7. **Skill gap check**: If the task didn't match any existing skill, briefly assess after completion: Was this a one-off, or a repeatable pattern? If repeatable, suggest creating a new skill to the user.
