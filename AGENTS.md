# Flow-Factory Development Guide

Flow-Factory is a typed, registry-driven framework for online and offline fine-tuning of diffusion
and flow-matching models. Python >=3.10 and PyTorch >=2.10 are supported. Match the user's language;
write code, comments, commit messages, and agent documentation in English.
Supported families include SenseNova-U1 (1.0/1.5; T2I + ordered multi-reference I2I); adapter
details remain in `guidance/new_model.md`.

## Start Here

Read this file and `.agents/knowledge/README.md` at session start. Do not preload
`philosophy.md`, `constraints.md`, or `architecture.md`; use the router to load only the leaves
required by the task and changed paths.

For repository changes, select the relevant skill, inspect the implementation before proposing a
solution, and run `python3 scripts/agent_scope.py --base origin/main` before implementation and
again against the final diff. Risk profiles and evidence requirements live under `.agents/harness/`.

## Universal Invariants

1. Typed contracts and registries are authoritative. A base class, execution contract, schema, or
   shared runtime change requires inspection of every registry-resolved implementation it reaches.
2. Training and inference must agree on generation-affecting inputs, scheduler state, precision,
   component order, and replay semantics.
3. Keep ownership explicit across acquisition, feedback, optimization, loading, distributed
   preparation, checkpointing, and resume. Fail fast with concrete values instead of silently
   weakening a contract.
4. Multi-file tasks require an explicit plan naming the applicable skills. Raise a simpler or safer
   design before implementation when one is evident.
5. After three failed approaches to the same cause, record the evidence in `.scratch/`, reassess
   the ownership model, and request review.
6. Put temporary reports, checklists, traces, and generated investigation artifacts under
   `.scratch/`, which is git-ignored.

Use `.agents/knowledge/constraints.md` as the stable constraint index. Read the linked detail only
when the router or risk profile selects it.

## Task Modes and Mutation Scope

- **Inspect**: analysis, explanation, planning, or review. Read and run non-mutating diagnostics;
  do not edit or commit.
- **Change**: the user asks to implement, fix, refactor, or update the repository. Edit and test in
  scope; after `/ff-review`, a **safe** verdict may be committed without a second confirmation.
- **Deliver**: the user explicitly asks to push, open a PR, publish, release, or run a remote GPU
  campaign. Perform only the requested delivery actions after the change is reviewable.

A **risky** review verdict always stops before commit. Push, merge, release, and remote GPU work
require Deliver scope; Change scope alone does not imply them.

## Development Commands

```bash
pip install -e "."             # Core
pip install -e ".[all]"        # DeepSpeed + quantization
ff-train <config.yaml>          # Training
black --check src/              # Format
isort --check-only src/         # Imports
python3 -m pytest               # Tests
```

## Available Skills

Skills follow the [Agent Skills](https://agentskills.io) standard. Each `SKILL.md` is a short
router; load its referenced detail only when the selected risk facets require it.

| Skill | Purpose | Use When |
|-------|---------|----------|
| `/ff-develop` | Feature development with impact analysis | Implementing new functionality or refactoring |
| `/ff-debug` | Bug fixing with structured protocol | Debugging errors, crashes, unexpected behavior |
| `/ff-review` | Pre-commit code review | Before committing changes |
| `/ff-new-model` | Model adapter integration | Adding support for a new diffusion model |
| `/ff-new-reward` | Reward model integration | Adding a new reward function |
| `/ff-new-algorithm` | Online/offline algorithm integration | Adding a new training algorithm |
| `/ff-new-accelerator` | Acceleration plugin integration | Adding compile, cache, attention, or other acceleration behavior |
| `/ff-harness-maintenance` | Agent harness maintenance | Editing agent docs, skills, rules, routes, validators, or evals |

### Quick Decision Guide

- **"Add support for model X"** -> `/ff-new-model`
- **"Add a new reward function"** -> `/ff-new-reward`
- **"Add a new training algorithm"** -> `/ff-new-algorithm`
- **"Add an acceleration plugin"** -> `/ff-new-accelerator`
- **"Update the agent harness"** -> `/ff-harness-maintenance`
- **"Fix this error" / "training hangs" / "wrong results"** -> `/ff-debug`
- **"Add a new capability" / "refactor" / "clean up"** -> `/ff-develop`
- **"Review before committing"** -> `/ff-review`

## Commit & PR Conventions

- **Commit messages**: Concise, descriptive, in English
- **PR title format**: `[{modules}] {type}: {description}` (e.g., `[trainer,reward] feat: add multi-reward weighting`)
- **Valid types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Run code quality checks before committing

## Commit and Delivery

1. Complete implementation, documentation, examples, and the focused evidence selected by the
   final risk profile.
2. Run `/ff-review`. For R4 changes, validate the exact final commit with
   `config/gpu_validation/framework_upgrade.yaml` and `scripts/validate_gpu_campaign.py` before
   merge.
3. In Change or Deliver mode, **safe** -> commit. **risky** -> report unresolved evidence and wait
   for direction.
4. Keep commits coherent. Commit independent fixes separately; do not split one contract change
   merely to reduce file count.
5. Before each commit, run Black/isort on changed Python files and report pre-existing full-tree
   failures separately.
