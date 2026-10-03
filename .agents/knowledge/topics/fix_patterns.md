# Fix Pattern Capture

**Read when**: Capturing a reusable lesson after a bug fix.

Query existing history before adding a record:

```bash
python3 scripts/query_fix_patterns.py --query "<symptom, owner, or invariant>"
```

Promote a fix to `.agents/knowledge/fixes/<domain>.md` when it establishes a reusable failure
signature, ownership lesson, or durable constraint. A local one-off fix needs a regression and a
clear commit message, but does not need a permanent knowledge card.

## Fix Entry Template

```markdown
### [Short Title]
- **Date**: YYYY-MM-DD
- **Symptom**: User-visible failure
- **Root Cause**: Authoritative cause
- **Fix**: Changed owner and behavior
- **Lesson**: Reusable decision rule
- **Related Constraint**: #N or N/A
- **Evidence**: Regression, trace, or campaign artifact
- **Commit**: Commit or PR reference
```

Update `fixes/index.yaml` for every promoted card. Put a new hard rule in the appropriate
constraint category and link the card instead of duplicating its full narrative.

## Cross-refs

- UP: [Fix Pattern Router](../fixes/README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
