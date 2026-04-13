---
name: ff-develop
description: "Feature development with cross-module impact analysis. Covers trainer hierarchy, model adapters, reward pipeline, config system, sample dataclasses, and distributed training paths. Trigger: 'add feature', 'implement', 'refactor', 'reorganize', 'new capability'."
---

# Feature Development Workflow

## Related Topics (read if your change touches these areas)

- Adapter changes -> `topics/adapter_conventions.md`
- Trainer/scheduler changes -> `topics/train_inference_consistency.md`
- Precision changes -> `topics/dtype_precision.md`

## Impact Analysis Checklist

Before implementing features or refactoring, analyze impacts across these areas:

### 1. Trainer Hierarchy (`constraints.md` #11)
- Changes to `BaseTrainer` affect all 5 concrete trainers; changed abstract methods must be implemented on every one
- Changes to `AdvantageProcessor` affect all trainers (`architecture.md` "Advantage Computation")
- Check: Does your change alter `_initialization()`, `_init_reward_model()`, or `_init_dataloader()`?

### 2. Model Adapter Hierarchy (`constraints.md` #12)
- Changes to `BaseAdapter` affect ALL model adapters
- Check: Does your change modify component management, LoRA logic, or mode switching?

### 3. Reward Pipeline (`constraints.md` #13)
- Changes to `BaseRewardModel` or `RewardProcessor` affect all reward models
- Check: Does your change alter the Pointwise/Groupwise dispatch?

### 4. Configuration System (`constraints.md` #15–17)
- Check: Did you rename, remove, or **add** fields? ALL configs in `examples/` must be updated

### 5. Sample Dataclasses (`constraints.md` #14)
- Changes to `BaseSample` or its subclasses affect data flow through all 6 stages
- Check: Did you change `_shared_fields` or add new fields?

### 6. Distributed Training Paths (`constraints.md` #9, #18–20)
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
- **Bug fix experience?** -> follow `.agents/knowledge/topics/fix_patterns.md` archival process

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
