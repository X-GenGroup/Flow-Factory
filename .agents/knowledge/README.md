# Agent Knowledge Router

| Task intent | Skill or context |
|---|---|
| Feature or refactor | `../skills/ff-develop/SKILL.md` |
| Bug, crash, hang, OOM, or numerical failure | `../skills/ff-debug/SKILL.md`; query historical fixes before the first experiment |
| Pre-commit or PR review | `../skills/ff-review/SKILL.md` |
| New model adapter | `../skills/ff-new-model/SKILL.md` |
| New reward | `../skills/ff-new-reward/SKILL.md` |
| New or extended algorithm | `../skills/ff-new-algorithm/SKILL.md` |
| Agent docs, rules, skills, or harness scripts | `docs_maintenance.md`, `../harness/README.md` |

| Changed area or symptom | Read |
|---|---|
| Design principles or framework architecture decision | `philosophy.md`, `architecture.md` |
| Hard constraint lookup | `constraints.md`; follow only the category needed by the task |
| Trainer objective, replay, adapter forward/inference, scheduler step | `topics/train_inference_consistency.md` |
| Dtype, mixed precision, NaN, overflow | `topics/dtype_precision.md`, `topics/autocast_param_swap.md` |
| Model adapter, model I/O, output codec, output geometry | `topics/adapter_conventions.md`, `topics/parity_testing.md` |
| Component discovery, lifecycle, loading, FSDP preparation | `topics/component_runtime.md` |
| Multi-component trajectory, replay bridge, component order | `topics/structured_trajectory.md` |
| Multi-role variants, optimizer ownership, Muon | `topics/component_variants.md`, `dependencies.md` |
| Sampler geometry, reward groups, overlap, multi-source | `topics/samplers.md` |
| Sample reconstruction, media ownership, rollout lifecycle | `topics/sample_lifecycle.md` |
| MiniMax H3 workflow or memory policy | `topics/minimax_h3.md` |
| Timestep, sigma, or TDM time sampling | `topics/timestep_sigma.md` |
| Dependency or installation change | `dependencies.md` |
| Prior bug with a similar symptom or owner | `python3 scripts/query_fix_patterns.py --query "<symptom or owner>"` |
