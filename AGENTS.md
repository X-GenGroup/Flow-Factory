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
- `.agents/knowledge/constraints.md` — hard constraints to check before any code change
- `.agents/knowledge/architecture.md` — module map, dependency graph, extension points

**Tier 2 (as needed):** `.agents/knowledge/dependencies.md` when changing installs or versions; `.agents/knowledge/topics/samplers.md` when touching `data_utils/sampler*`, `DataArguments.sampler_type`, or batch-geometry alignment in `hparams/args.py`.

## Key Technical Requirements

- Distributed training via **Accelerate** (primary) and **DeepSpeed** (ZeRO-1/2)
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
7. **Challenge First, Execute Second** — Spot logic flaws or simpler alternatives? Raise concerns before executing.
8. **Search Before You Act** — On unexpected behavior, search codebase + check constraints + review `git log` before attempting fixes.
9. **Planning Discipline** — Complex tasks (multi-file, >30 min) -> use TodoWrite. Plan must state which skills will be used.
10. **Fix Experience Capture** — After completing any bug fix or error resolution, generate a fix summary using the template in `.agents/knowledge/topics/fix_patterns.md`, then use `AskUserQuestion` to ask the user whether to archive it. Refer to the archival location decision table in that file to propose a destination. Never skip this step, even for quick fixes.
11. **English-Only Documentation** — All code comments, docstrings, commit messages, and agent documentation (files under `.agents/`, `guidance/`, `AGENTS.md`) MUST be written in English. User-facing chat responses should still match the user's language per the "Language" directive above, but all persisted text in the repository must be English.

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

Skills follow the [Agent Skills](https://agentskills.io) open standard. Each skill is a folder in `.agents/skills/<name>/` containing a `SKILL.md` with YAML frontmatter. Skills are auto-discovered by compatible agents (Cursor, Claude Code, Codex, etc.) and can also be invoked manually with `/skill-name` in chat.

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
