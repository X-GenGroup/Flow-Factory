# Agent Harness Risk Routing

The harness selects context and evidence from semantic risk, not task labels or file count.

| Profile | Meaning | Default evidence |
|---|---|---|
| `R0` | Read-only inspection | No mutation |
| `R1` | Leaf extension or objective-only change | Focused unit and contract tests |
| `R2` | Component-local composition change | Affected integration paths |
| `R3` | State, schema, role, collective, resume, or backend change | Strict local matrix and affected distributed backends |
| `R4` | Shared execution, dataflow, loading, overlap, optimizer, or checkpoint infrastructure | Exact-commit framework campaign |

Declare an initial profile from intent, then run `scripts/agent_scope.py` against the final diff.
The observed profile may only raise the required evidence. A fast profile narrows the proof surface;
it does not relax numerical, type, ordering, or fail-fast correctness.

`risk_rules.yaml` maps paths and diff symbols to profiles and facets. `test_routes.yaml` maps those
facets to evidence. Both files are repository policy and are validated by
`scripts/validate_agent_harness.py`.

## Cross-refs

- UP: [`AGENTS.md`](../../AGENTS.md), [Knowledge Router](../knowledge/README.md)
- REVIEW: [`ff-review`](../skills/ff-review/SKILL.md)
