# Flow-Factory Agent Skills Framework

## Overview

Reusable workflow definitions for AI coding agents. Skills follow the Agent Skills open standard with YAML frontmatter for auto-discovery.

## Skill Structure

```
.agents/skills/<skill-name>/
├── SKILL.md          # Required: YAML frontmatter + workflow documentation
├── scripts/          # Optional: Helper scripts
└── references/       # Optional: Reference materials
```

## SKILL.md Frontmatter

```yaml
---
name: ff-develop          # Must match folder name (lowercase, hyphens only)
description: Feature development with impact analysis
---
```

## Available Skills

| Skill | Purpose | Use When |
|-------|---------|----------|
| `ff-develop` | Feature development with impact analysis | Implementing new features or refactoring |
| `ff-debug` | Bug fixing with structured protocol | Debugging errors or unexpected behavior |
| `ff-review` | Pre-commit code review | Before committing changes |
| `ff-new-model` | Model adapter integration | Adding a new diffusion model |
| `ff-new-reward` | Reward model integration | Adding a new reward function |
| `ff-new-algorithm` | RL algorithm integration | Adding a new training algorithm |
| `ff-new-accelerator` | Acceleration plugin integration | Adding or changing acceleration behavior |
| `ff-harness-maintenance` | Agent harness maintenance | Editing agent docs, skills, rules, routes, validators, or evals |

## Invocation

Users invoke skills via `/skill-name` syntax (e.g., `/ff-develop`). Agents auto-discover skills via the `name` and `description` fields in SKILL.md frontmatter.

## Knowledge Base

See `.agents/knowledge/README.md` for task- and path-triggered context routing.

## Adding New Skills

1. Create folder: `.agents/skills/<new-skill>/`
2. Create `SKILL.md` with required YAML frontmatter
3. Name must use lowercase letters and hyphens only
4. Update this README
5. Register in `AGENTS.md` skills table
6. Add or update harness evals when selection or escalation behavior changes

## Compliance

Only include reusable workflows here. Temporary scripts belong in `scripts/` or `tests/`.
