---
name: ff-develop
description: Feature development workflow with cross-module impact analysis
---

# Feature Development Workflow

## Impact Analysis Checklist

Before implementing features or refactoring, analyze impacts across these areas:

### 1. Trainer Hierarchy
- Changes to `BaseTrainer` affect `GRPOTrainer`, `GRPOGuardTrainer`, `DiffusionNFTTrainer`, `AWMTrainer`
- Check: Does your change alter the `_initialization()`, `_init_reward_model()`, or `_init_dataloader()` flow?

### 2. Model Adapter Hierarchy
- Changes to `BaseAdapter` affect ALL model adapters (Flux, SD3.5, Wan, Qwen-Image, Z-Image)
- Check: Does your change modify component management, LoRA logic, or mode switching?

### 3. Reward Pipeline
- Changes to `BaseRewardModel` or `RewardProcessor` affect all reward models
- Check: Does your change alter the Pointwise/Groupwise dispatch or `RewardModelOutput` format?

### 4. Configuration System
- Changes to `hparams/` dataclasses affect YAML parsing
- Check: Did you rename/remove fields? Update ALL configs in `examples/`

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
- [ ] YAML configs in `examples/` updated if needed
- [ ] License header present on new files
