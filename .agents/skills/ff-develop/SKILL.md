---
name: ff-develop
description: Develop or refactor Flow-Factory features by classifying semantic risk, changing the authoritative owner, and selecting proportionate evidence.
---

# Feature Development

Use the repository risk profile to plan and verify a feature. File count is not a risk measure;
contract, state, ownership, distributed, and lifecycle changes are.

## Context Routing

1. Read `../../knowledge/README.md` and run
   `python3 ../../../scripts/agent_scope.py --base origin/main --intent develop`.
2. Load only the topic leaves selected by the reported facets.
3. For an R3/R4 contract impact analysis, read
   [references/contract-impact.md](references/contract-impact.md).
4. Read `../../knowledge/philosophy.md` and `../../knowledge/architecture.md` only for an
   architecture decision or shared framework change.

## Workflow

1. Identify the authoritative owner, public contract, current callers, and closest working
   analogue. Search registries and code rather than relying on static component lists.
2. Declare the initial profile, risk facets, affected execution compositions, compatibility
   surface, and acceptance criteria. Multi-file work needs an explicit plan.
3. Prefer the smallest owner-level change. Update callers, registries, arguments, examples, docs,
   and migrations only where the contract reaches them.
4. Add focused invariant tests before integration evidence. Preserve fail-fast behavior and test
   negative capability/configuration cases at the earliest lifecycle boundary.
5. Run `agent_scope.py` again against the final diff. The observed profile may raise the required
   evidence; it cannot lower it.
6. Update durable knowledge only when the change establishes a reusable invariant. Run
   `/ff-review` before commit.

## Profile Guidance

- **R1**: leaf extension with existing contracts and no new persistent state. Use focused unit,
  registry, and parser tests; do not run a broad GPU matrix.
- **R2**: component-local composition or strict extension. Add affected integration paths and a
  working-analogue comparison.
- **R3**: state, schema, role, collective, resume, backend, or advanced adapter-runtime change.
  Verify the affected strict matrix and distributed backends.
- **R4**: shared execution, dataflow, overlap, loading, optimizer, or checkpoint infrastructure.
  Plan the complete exact-commit framework campaign required by constraint #30.

Do not redesign a shared registry or runtime merely to reduce a leaf extension by one small static
entry. Treat that redesign as a separate profile-bearing change.
