# Fix Pattern Router

Search the compact index and domain cards before debugging:

```bash
python3 scripts/query_fix_patterns.py --query "distributed reward hang"
python3 scripts/query_fix_patterns.py --paths src/flow_factory/trainers/abc.py
```

Load only the returned domain file and matching entries. Promote a fix into this knowledge base
when it establishes a reusable failure signature, ownership lesson, or durable constraint. Local
one-off fixes still require a regression and a clear commit message, but not a permanent card.

Each promoted record contains symptom, root cause, fix, lesson, related constraints, and concrete
evidence when available. `index.yaml` is the machine-readable inventory.

## Cross-refs

- UP: [Knowledge Router](../README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
