# Agent Harness Risk Routing

The harness selects context and evidence from semantic risk, not task labels or file count.

| Profile | Meaning | Default evidence |
|---|---|---|
| `R0` | Read-only inspection | No mutation |
| `R1` | Leaf extension or objective-only change | Focused unit and contract tests |
| `R2` | Component-local composition change | Affected integration paths |
| `R3` | State, schema, role, collective, resume, or backend change | Selected state/backend boundary evidence |
| `R4` | Shared execution, dataflow, loading, overlap, optimizer, or checkpoint infrastructure | All affected obligations, selected exact-commit evidence |

Declare an initial profile from intent, then run `scripts/agent_scope.py` against the final diff.
The observed profile sets a conservative floor for review; QA and resources are selected separately. A fast profile narrows the proof surface;
it does not relax numerical, type, ordering, or fail-fast correctness.

`risk_rules.yaml` maps paths and diff symbols to profiles and facets. `test_routes.yaml` maps those
facets to evidence. Both files are repository policy and are validated by
`scripts/validate_agent_harness.py`. A Python file whose token stream, excluding comments and
docstrings, matches the base (`scripts/check_docs_only_diff.py`) receives `docs_only_rule`
instead of path and content rules. Existing YAML files receive the same rule only when
`agent_diff.yaml_comments_only` proves identical scanner tokens and raw spelling after strict
parsing. Duplicate keys, merge keys, unsupported tags and tool-directive comments retain owner
rules; new/deleted files and mode changes are never exempt. Loaded-value equality is not a proof.
Every other changed file keeps its own escalation. A file
whose base version the interpreter cannot parse is reported as unverified and keeps its rules.

`evals/cases.yaml` locks representative fast, focused, strict, and shared-contract selections. Add a
case whenever a path or semantic escalation boundary changes. Logging exemptions use complete
source pairs and accept only unchanged AST outside literal console-log statements; no keyword-only
exemption is allowed. `agent_scope.py --head <commit>` inspects an immutable snapshot. Worktree
mode requires intended new files to be staged and reports untracked roots and opaque inputs.

R3/R4 use [selected plans](../../docs/development/validation/validation.md) with coverage and exact-source result
validation. The [automatic planner](../../docs/development/validation/automatic-validation.md) selects from the catalog using
explicit owner rules; manual plans remain available for focused investigations. R4 does not imply a fixed
QA level, GPU count, reward service, or full model/backend product. The old campaign reader remains
available only for explicitly selected historical qualification.

Canonical plans keep raw R4 classification separate from structural coverage routing.
Same-binding import rewrites, AST-identical export bridges, complete symbol partitions and
bounded test-helper dependencies are checked by `scripts/validation_changes.py`. Unknown or
mixed edits retain conservative coverage; direct and prepared numerical witnesses remain
mandatory. See the automatic planner's structural-routing section and its corruption regressions.

## Cross-refs

- UP: [`AGENTS.md`](../../AGENTS.md), [Knowledge Router](../knowledge/README.md)
- REVIEW: [`ff-review`](../skills/ff-review/SKILL.md)
