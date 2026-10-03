# Knowledge Documentation Maintenance

**Read when**: Adding or editing `.agents/` documentation.

---

## Architecture

The knowledge system uses a 3-layer design with bidirectional cross-references:

```
Root:    AGENTS.md                          — project identity and universal operating guide
Tier 1:  philosophy.md, constraints.md,     — always-read current authority + concise indexes
         architecture.md
Routing: README.md                          — trigger-based table from Tier 1 to Tier 2
Leaves:  topics/*.md                        — self-contained detail (read when triggered)
Skills:  skills/*/SKILL.md                  — workflows with downward refs to topics
```

## Node Roles

**Non-leaf** (root + Tier 1): Keep universal operating rules and current architecture summaries
concise, then route specialized detail to leaves. `AGENTS.md` may retain universal commands and the
commit workflow; `constraints.md` may include the minimum rationale needed to make a hard rule
enforceable; `architecture.md` may retain the current module graph, registries, extension points,
and short design summaries. Model-specific recipes, deep implementation walkthroughs, new
chronological fix records, and specialized checklists belong in leaves. Do not add Tier-1 detail
when a leaf pointer is sufficient.

**Leaf** (`topics/*.md`): Self-contained, concise, essential knowledge. Include code refs, checklists, numbered gotchas. No filler prose, no introductory fluff, no restating what parent docs already say. Format examples: `adapter_conventions.md`, `train_inference_consistency.md`.

**Routing** (`README.md`): Trigger-based table only. Each row maps a change area to the topic doc an agent should read. No prose.

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
