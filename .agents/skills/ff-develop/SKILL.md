---
name: ff-develop
description: "Feature development with cross-module impact analysis. Covers trainer hierarchy, model adapters, reward pipeline, config system, sample dataclasses, and distributed training paths. Trigger: 'add feature', 'implement', 'refactor', 'reorganize', 'new capability'."
---

# Feature Development Workflow

## Impact Analysis Checklist

Before implementing features or refactoring, analyze impacts across these areas:

### 1. Trainer Hierarchy
- Changes to `BaseTrainer` affect `GRPOTrainer`, `GRPOGuardTrainer`, `DPOTrainer`, `DiffusionNFTTrainer`, `AWMTrainer`
- New or changed abstract methods on `BaseTrainer` (e.g. `prepare_feedback`) must be implemented on every concrete trainer
- Changes to `AdvantageProcessor` affect all trainers that delegate advantage computation
- Check: Does your change alter the `_initialization()`, `_init_reward_model()`, or `_init_dataloader()` flow?

### 2. Model Adapter Hierarchy
- Changes to `BaseAdapter` affect ALL model adapters (Flux, SD3.5, Wan, Qwen-Image, Z-Image)
- Check: Does your change modify component management, LoRA logic, or mode switching?

### 3. Reward Pipeline
- Changes to `BaseRewardModel` or `RewardProcessor` affect all reward models
- Check: Does your change alter the Pointwise/Groupwise dispatch or `RewardModelOutput` format?

### 4. Configuration System
- Changes to `hparams/` dataclasses affect YAML parsing
- Check: Did you rename, remove, or **add** fields? ALL configs in `examples/` must be updated:
  - **Renames/removals**: Search-and-replace across all YAML files (old keys fail silently with defaults)
  - **New user-facing fields**: Add to ALL example configs with the default value and an `# Options: ...` comment
  - **New algorithm-specific fields**: Add to that algorithm's configs only

### 5. Sample Dataclasses
- Changes to `BaseSample` or its subclasses affect data flow through all 6 stages
- Check: Did you change `_shared_fields` or add new fields?

### 6. Distributed Training Paths
- Changes may behave differently under Accelerate vs DeepSpeed
- Check: Does your change involve `accelerator.prepare()`, gradient accumulation, or model sharding?

## Refactoring Safety Rules

1. **Establish baseline** — Run tests before making changes
2. **One at a time** — ONE structural change → update ALL callers → verify → commit
3. **Never combine** — Don't combine multiple refactoring steps in one commit

## Workflow Steps

1. **Understand scope**
   - Read relevant `abc.py` base classes
   - Identify all affected subclasses and callers
   - Read related `guidance/` docs

2. **Plan changes**
   - List all files that need modification
   - Document expected behavior changes
   - Identify test scenarios

3. **Implement methodically**
   - Make ONE change at a time
   - Update ALL callers/subclasses
   - Run tests after each change

4. **Cross-algorithm verification**
   - Test with GRPO (coupled paradigm)
   - Test with NFT or AWM (decoupled paradigm)
   - Verify with at least two different model adapters

## Documentation

Before committing, check if the change requires documentation updates:

- **New/changed API** -> update relevant `guidance/` doc
- **New/changed config fields** -> update ALL example configs in `examples/`
- **Architecture change** -> update `.agents/knowledge/architecture.md`
- **New constraint discovered** -> add to `.agents/knowledge/constraints.md`
- **Bug fix experience?** -> follow `.agents/knowledge/topics/fix_patterns.md` archival process (generate fix summary, ask user, write to appropriate location)

## Common Traps

- `preprocess_func()` registration: forgetting to list new encoding components in `preprocessing_modules` causes OOM
- Config field renames **silently break** existing YAML configs — grep all YAML files first
- `_shared_fields` in sample dataclasses: incorrect fields cause silent data corruption during collation
- LoRA `target_module_map`: mapping wrong components means no training effect
- Mixing coupled/decoupled paradigms: using ODE with GRPO produces incorrect gradients silently
- `BaseAdapter` has 7 abstract methods — missing any one breaks the entire pipeline
- Renaming config keys in `hparams/` without updating ALL `examples/` YAML causes silent default-fallback

## When to Delegate

- **Adding a new model** → `/ff-new-model`
- **Adding a new reward** → `/ff-new-reward`
- **Adding a new algorithm** → `/ff-new-algorithm`
- **Debugging a bug** → `/ff-debug`
- **Pre-commit review** → `/ff-review`

## Pre-Commit Checks

- [ ] Impact analysis completed for all 6 areas
- [ ] All callers/subclasses updated
- [ ] Tests pass
- [ ] Code formatted with Black and isort
- [ ] YAML configs in `examples/` updated: new fields added, renamed fields updated, removed fields cleaned up
- [ ] License header present on new files
