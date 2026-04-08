# Design Philosophy

Flow-Factory is a simple, extensible RL fine-tuning framework for diffusion/flow-matching models.
Models, algorithms, and rewards are decoupled via registries and base-class contracts.
Both rollout and training use FSDP as backend. The single most important invariant is
**train-inference consistency**.

## Principles

| # | Principle | One-liner | Detail |
|---|-----------|-----------|--------|
| 1 | Train-inference consistency | `forward()` is the atomic unit: same inputs -> same outputs across rollout and training | `topics/train_inference_consistency.md` |
| 2 | Decoupled extensibility | Models, trainers, rewards are independently pluggable via registries; adding a model never changes trainer code | `architecture.md` "Registry System" |
| 3 | Fail-fast, no speculative code | Raise with context on invalid state; never silently correct or swallow | `constraints.md` #26 |
| 4 | Top-down readability | `forward()` and `inference()` read linearly without tracing through utilities | `constraints.md` #27 |
| 5 | Structural vs behavioral separation | First commit matches reference numbers; second commit cleans up style | `topics/adapter_conventions.md` |

## Coding Style Index

| Concern | Source |
|---------|--------|
| Black/isort | `constraints.md` #21 |
| Import style | `constraints.md` #22 |
| Type annotations | `constraints.md` #23 |
| License header | `constraints.md` #24 |
| Logger messages | `constraints.md` #25 |
| Error handling | `constraints.md` #26, `.cursor/rules/no-defensive-except.mdc` |
| Docstrings | `constraints.md` #27 |
| Variable naming | `.cursor/rules/variable-naming.mdc` |
| No section dividers | `.cursor/rules/no-section-divider-comments.mdc` |
| Knowledge docs structure | `docs_maintenance.md`, `.cursor/rules/agents-docs-maintenance.mdc` |
