# Flow-Factory Development Guide

## Project Overview

Flow-Factory is a unified **online and offline fine-tuning framework** for diffusion/flow-matching models. It provides a modular architecture where trainers, model adapters, data acquisition, and reward models are independently extensible through typed contracts and registries.

- **Algorithms**: SFT, offline DPO, online DPO, GRPO, GRPO-Guard, DPPO, DGPO, DiffusionNFT, AWM, CRD, DiffusionOPD, DMD2, TDM, TDM-R1
- **Models**: FLUX.1 (+Kontext), FLUX.2 (+Klein), SD3.5, Qwen-Image (+Edit-Plus), Z-Image, Wan2 (T2V/I2V), LTX2 (T2AV/I2AV), MiniMax H3 (T2VA/FL2VA/Ref2VA), Bagel, SenseNova-U1 (1.0/1.5; T2I + ordered multi-reference I2I)
- **Rewards**: PickScore (+Rank), CLIP, CLAP, ImageBind, OCR, GenEval/GenEval2, HPSv2, VLM-Evaluate, rational-rewards, and custom rewards
- **Python**: >=3.10 | **PyTorch**: >=2.10.0 | **License**: Apache-2.0

**Language**: Match user's language.

## Context Loading

On session start, read **Tier 1** (see `.agents/knowledge/README.md`):
- `.agents/knowledge/philosophy.md` — design principles, coding style index
- `.agents/knowledge/constraints.md` — hard rules, indexed by category
- `.agents/knowledge/architecture.md` — module graph, pipeline stages, registries

**Tier 2**: Topic docs triggered by change area. See `.agents/knowledge/README.md` for triggers.

## Core Operating Principles

1. **Constraints first** — Read `constraints.md` + `architecture.md` before changes; search codebase before attempting fixes.
2. **Cross-component awareness** — Changes to base classes or typed contracts affect every registry-resolved implementation; verify the affected coupled-reward, decoupled-reward, no-feedback, and dataset-acquisition paths.
3. **Plan before implement** — Multi-file tasks require an explicit task plan using the agent's supported planning mechanism. The plan must state which skills apply.
4. **Challenge first, execute second** — Spot logic flaws or simpler alternatives? Raise before executing.
5. **Escalation** — After three failed approaches, document findings and request review.
6. **Fix capture** — After every bug fix, generate summary per `.agents/knowledge/topics/fix_patterns.md` template.
7. **English-only docs** — All code comments, docstrings, commit messages, and agent docs must be English.
8. **Scratch files only** — All temporary/intermediate files (analysis reports, investigation notes, checklists) MUST go under `.scratch/` (git-ignored). Never pollute the project root or tracked directories.

Hard rules: see `constraints.md`.

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

See `.agents/knowledge/architecture.md` "Module Dependency Graph" for full details.

## Documentation Reference

| Document | Purpose |
|----------|---------|
| `guidance/workflow.md` | Unified generation/dataset acquisition contracts plus the online 6-stage pipeline |
| `guidance/algorithms.md` | All algorithms (SFT, offline DPO, GRPO, GRPO-Guard, DPPO, online DPO, DGPO, DiffusionNFT, AWM, CRD, DiffusionOPD, DMD2, TDM, TDM-R1) deep dive |
| `guidance/rewards.md` | Reward system design, custom model creation |
| `guidance/new_model.md` | Step-by-step model adapter integration |
| `guidance/acceleration.md` | Acceleration plugin layer (compile, attention backend, feature caching) |
| `guidance/gpu_validation.md` | Mandatory exact-commit GPU gate for broad framework upgrades |

## Available Skills

Skills follow the [Agent Skills](https://agentskills.io) open standard. Each skill is a folder in `.agents/skills/<name>/` containing a `SKILL.md` with YAML frontmatter. Skills are auto-discovered by compatible agents (Cursor, Claude Code, Codex, etc.) and can also be invoked manually with `/skill-name` in chat.

| Skill | Purpose | Use When |
|-------|---------|----------|
| `/ff-develop` | Feature development with impact analysis | Implementing new functionality or refactoring |
| `/ff-debug` | Bug fixing with structured protocol | Debugging errors, crashes, unexpected behavior |
| `/ff-review` | Pre-commit code review | Before committing changes |
| `/ff-new-model` | Model adapter integration | Adding support for a new diffusion model |
| `/ff-new-reward` | Reward model integration | Adding a new reward function |
| `/ff-new-algorithm` | Online/offline algorithm integration | Adding a new training algorithm |

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
5. For changes covered by `constraints.md` #30, run the manifest-defined GPU gate against the final
   commit and attach validated evidence before merge.
6. Each fix -> immediate commit. Do not batch unrelated changes.
7. Run `black --check src/ && isort --check src/` before every commit.
8. **Skill gap check**: If the task didn't match any existing skill, briefly assess after completion: Was this a one-off, or a repeatable pattern? If repeatable, suggest creating a new skill to the user.
