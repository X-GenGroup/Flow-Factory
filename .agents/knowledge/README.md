# Agent Knowledge Base

Documents in this directory support AI coding agents. They are grouped by how often you should read them.

## Tier 1 — Read at session start

| File | Purpose |
|------|---------|
| [`constraints.md`](constraints.md) | Hard constraints; check before any code change |
| [`architecture.md`](architecture.md) | Module map, dependency graph, extension points |

## Tier 2 — Read when relevant

| File | Purpose |
|------|---------|
| [`dependencies.md`](dependencies.md) | `pyproject.toml`, extras, version and DeepSpeed notes |
| [`topics/samplers.md`](topics/samplers.md) | Stage 2 samplers: geometry, auto-adjustment, `sampler_type`, interaction with advantages/rewards |

Use **topics/** for deep dives on a single subsystem; new topic docs (e.g. samples/collation) can be added alongside `samplers.md` without diluting Tier 1.
