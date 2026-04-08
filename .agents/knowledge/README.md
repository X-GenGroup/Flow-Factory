# Agent Knowledge Base

| Trigger | Read |
|---------|------|
| Session start | `philosophy.md`, `constraints.md`, `architecture.md` |
| Touching `trainers/*.optimize`, `adapter.forward`/`inference`, `scheduler.step` | `topics/train_inference_consistency.md` |
| Touching dtype/precision, debugging NaN/overflow | `topics/dtype_precision.md` |
| Adding or modifying a model adapter | `topics/adapter_conventions.md` |
| Adding adapter, upgrading diffusers, debugging output quality | `topics/parity_testing.md` |
| Touching `TimeSampler`, `adapter.forward(t=...)`, `timestep_range`, `flow_match_sigma` | `topics/timestep_sigma.md` |
| Editing `data_utils/sampler*`, hparams sampler/batch fields | `topics/samplers.md` |
| After completing a bug fix | `topics/fix_patterns.md` |
| Changing `pyproject.toml`, deps, install commands | `dependencies.md` |
| Adding or editing `.agents/` documentation | `docs_maintenance.md` |
