# Agent Knowledge Base

## Tier 1 — Read at session start

| File | Purpose |
|------|---------|
| `philosophy.md` | Design philosophy, coding style index |
| `constraints.md` | 27 hard rules (#1-27), indexed by category |
| `architecture.md` | Module graph, 6-stage pipeline, registries |

## Tier 2 — Read when triggered

| File | Read when... |
|------|-------------|
| `topics/train_inference_consistency.md` | Touching `trainers/*.optimize`, `adapter.forward`/`inference`, `scheduler.step` |
| `topics/dtype_precision.md` | Touching dtype/precision, debugging NaN/overflow |
| `topics/adapter_conventions.md` | Adding or modifying a model adapter |
| `topics/parity_testing.md` | Adding adapter, upgrading diffusers, debugging output quality |
| `topics/samplers.md` | Editing `data_utils/sampler*`, hparams sampler/batch fields |
| `topics/fix_patterns.md` | After completing a bug fix |
| `dependencies.md` | Changing `pyproject.toml`, deps, install commands |
| `docs_maintenance.md` | Adding or editing `.agents/` documentation |
