# Agent Knowledge Router

| Task | Skill or context |
|---|---|
| Feature or refactor | `../skills/ff-develop/SKILL.md` |
| Crash, hang, OOM or numerical failure | `../skills/ff-debug/SKILL.md`; query fixes first |
| Commit/PR review | `../skills/ff-review/SKILL.md` |
| Roadmap planning | [Roadmap](../../roadmap/README.md) |
| New model adapter | `../skills/ff-new-model/SKILL.md` |
| New reward | `../skills/ff-new-reward/SKILL.md` |
| Algorithm extension | `../skills/ff-new-algorithm/SKILL.md` |
| Accelerator extension | `../skills/ff-new-accelerator/SKILL.md` |
| Agent docs/rules/skills/harness | `../skills/ff-harness-maintenance/SKILL.md`, `docs_maintenance.md` |
| Docstrings/comments or docs-only diff | `../skills/ff-docstring-audit/SKILL.md` |

| Area or symptom | Read |
|---|---|
| Design or architecture | `philosophy.md`, `architecture.md` |
| Constraints | `constraints.md`; needed category only |
| Objective, replay, forward/inference, scheduler | `topics/train_inference_consistency.md` |
| Dtype, precision, NaN, overflow | `topics/dtype_precision.md`, `topics/autocast_param_swap.md` |
| Adapter, I/O, codec, output geometry | `topics/adapter_conventions.md`, `topics/parity_testing.md` |
| Component lifecycle, loading, FSDP/TP | `topics/component_runtime.md`, `topics/tensor_parallel.md` |
| TP/SP/CP support or composition | `topics/parallelism_support.md` |
| Component trajectory, replay bridge, order | `topics/structured_trajectory.md` |
| Role variants, optimizer ownership, Muon | `topics/component_variants.md`, `dependencies.md` |
| Samplers, reward groups, overlap, multi-source | `topics/samplers.md` |
| Reconstruction, media ownership, rollout lifecycle | `topics/sample_lifecycle.md` |
| MiniMax H3 workflow/memory | `topics/minimax_h3.md` |
| Selected evidence/GPU qualification | `topics/gpu_validation.md` |
| Timestep, sigma, TDM sampling | `topics/timestep_sigma.md` |
| Dependencies/installation | `dependencies.md` |
| Bug patterns | `python3 scripts/query_fix_patterns.py --query "<symptom or owner>"` |
