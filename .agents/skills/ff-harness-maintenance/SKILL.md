---
name: ff-harness-maintenance
description: Maintain Flow-Factory agent docs, skills, rules, risk routes, historical knowledge, validators, and harness evaluations without increasing startup context.
---

# Agent Harness Maintenance

## Context Routing

Read `../../knowledge/docs_maintenance.md`, `../../harness/README.md`, and
`../../knowledge/README.md`. Use `skill-creator` guidance when creating or substantially revising a
skill. Run `python3 ../../../scripts/agent_scope.py --base origin/main --intent harness`.

## Workflow

1. Keep `AGENTS.md` and the knowledge router within the validator's startup budget. Universal
   invariants belong at the root; conditional detail belongs in leaves or skill references.
2. Keep canonical policy in repository-neutral docs. Claude/Cursor adapters contain only imports,
   path triggers, or tool-specific mechanics.
3. Update risk rules and evidence routes together. Add a deterministic eval case for every new
   escalation or fast-path decision.
4. Keep `SKILL.md` as a discriminating router. Move conditional procedures to `references/` and
   reusable deterministic logic to `scripts/`.
5. Query existing fix knowledge before adding a card. Promote only reusable lessons and update the
   machine index.
6. Run `python3 ../../../scripts/validate_agent_harness.py`, the harness tests, and every changed
   helper script. Re-run scope classification on the final diff and use `/ff-review`.

## Review Criteria

- No startup Tier 1 preload or duplicated canonical rule.
- Every skill is registered, validates, and has one context-routing section.
- Every local reference, fix index target, risk profile, facet route, and eval case resolves.
- Fast-path cases do not select broad validation; shared-lifecycle cases cannot remain fast.
- Tool permissions do not grant generic interpreters, installers, network commands, or delivery
  actions without task-specific authorization.
