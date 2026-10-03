# Knowledge Documentation Maintenance

**Read when**: Adding or editing `.agents/` documentation.

---

## Architecture

The knowledge system uses a 3-layer design with bidirectional cross-references:

```
Root:    AGENTS.md                          — universal invariants and task modes
Routing: README.md                          — startup task/path router
Indexes: constraints.md, architecture.md    — stable authority loaded on demand
Leaves:  topics/*.md, fixes/*.md            — self-contained detail loaded by trigger
Harness: harness/*.yaml                     — machine-readable risk and evidence routes
Skills:  skills/*/SKILL.md                  — short workflows with on-demand references
```

## Node Roles

**Non-leaf** (root + indexes): Keep universal operating rules and current architecture summaries
concise, then route specialized detail to leaves. `AGENTS.md` may retain universal commands and the
commit workflow; `constraints.md` may include the minimum rationale needed to make a hard rule
enforceable; `architecture.md` may retain the current module graph, registries, extension points,
and short design summaries. Model-specific recipes, deep implementation walkthroughs, new
chronological fix records, and specialized checklists belong in leaves. Do not add Tier-1 detail
when a leaf pointer is sufficient.

**Leaf** (`topics/*.md`, `fixes/*.md`): Self-contained, concise, essential knowledge. Include code
refs, checklists, or reusable failure patterns. Do not restate parent content.

**Routing** (`README.md`): Tables only. Map task intent and changed area to the smallest useful
skill, index, or leaf.

## Cross-Reference Rules

1. Every leaf links **UP** to its constraint/architecture source via a `## Cross-refs` section at the bottom.
2. Every skill links **DOWN** to relevant topics via one `## Context Routing` section.
3. Reference constraint numbers (e.g., `constraints.md #7`) instead of re-explaining the rule.
4. No duplication across layers — if detail exists in a leaf, the parent points to it rather than restating it.

## Maintenance Checklist

When modifying the knowledge system, verify these steps:

| Change | Required updates |
|--------|-----------------|
| New topic doc | Add row to `README.md` routing table with trigger condition |
| New topic doc | Add it to the relevant skill's `## Context Routing` table or list |
| New constraint | Update quick index range in `constraints.md` header + section header (e.g., extend `#21-27` Code Quality or add a new category such as `#28-29` Agent Workflow) |
| Append-only list | `Numbered Gotchas`, `FF-Specific Pitfalls` — only append, never reorder or remove |
| Any doc change | All text in English (`constraints.md` #21) |
| Moved detail | Replace inline content with pointer to the leaf that now holds it |
