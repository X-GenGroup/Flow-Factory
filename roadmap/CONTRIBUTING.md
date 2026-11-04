# Contributing to the roadmap

Start with the [roadmap index](README.md), then read only the milestone and design documents
relevant to your work. Follow [AGENTS.md](../AGENTS.md) for repository workflows and task scope.
Planning a release or GPU experiment does not itself authorize performing that delivery action.

## Find and coordinate work

1. Choose a P-number and inspect its dependencies, current code, and linked issues/PRs. Planned
   work may already have changed shape since the last roadmap update.
2. For substantial work, use an issue or draft PR to record the intended scope and link the
   P-number. Search existing coordination links first. Small, independent corrections can arrive
   directly as PRs.
3. Update the item's coordination field to that issue/PR when work begins. Keep contributor
   ownership and active handoffs in that conversation so two agents do not claim the same work.
4. Keep implementation and evidence within the selected contract. If a discovery changes a
   dependency, compatibility promise, or tag boundary, update the roadmap with the implementation.
5. Link the merged implementation and durable evidence before marking the item done. Update the
   release record when a tag is actually published.

Useful starting contributions include lightweight tooling for [P01a](milestones/v1.1.md#p01a),
consistency checks for [P01c](milestones/v1.1.md#p01c), cataloging existing tests for P02, replaying
historical diffs for P04, reproducing a specific P03 layout, and reviewing CP input contracts for
P12. A contribution does not need to implement an entire milestone or access a large GPU cluster.

## Status and ownership

The milestone item is the canonical source for its status, dependencies, and coordination link.
The linked issue/PR owns the active contributor list and detailed progress. Do not maintain a
second status board in agent-private files.

| Status | Meaning |
|---|---|
| `planned` | Proposed work; implementation is not complete. Check dependencies before starting. |
| `in_progress` | An issue/PR records active work and its owner. |
| `blocked` | A specific unmet dependency or missing resource is recorded with the next action. |
| `done` | Implementation merged and acceptance evidence linked. This does not imply a release tag. |
| `deferred` | Removed from the current milestone with a reason and destination recorded. |

Preserve P-numbers and explicit anchors when moving an item. If an item needs several independently
reviewable PRs, use child IDs such as `P07a` and keep P07 as their acceptance boundary. Do not split
one inseparable contract merely to reduce the apparent size of a diff. Never reuse a retired ID.

## Work-item format

New items should follow the existing milestone format:

```text
ID and outcome
Status: planned
Depends on: linked work-item IDs, or none
Coordination: unclaimed, or issue/PR link

Outcome: the user-visible or architectural result.
Scope: affected owners and compatibility changes; reference existing code.
Acceptance: observable behavior and required evidence.
Release boundary: the tag unlocked, if any.
Open decisions: concrete questions that still need experiments or design review.
```

Before a cross-agent handoff, record the branch/worktree, base and current commit, scope already
completed, remaining acceptance items, commands and results, and the next action. Raw logs can live
in git-ignored `.scratch/` during development. Evidence required to review or close an item must be
available through the PR or another durable artifact, not only a path on one contributor's host.

## Review and release records

Use the repository's English PR-title convention, for example
`[docs] docs: add the distributed training roadmap`. Include the work-item IDs, resulting behavior,
compatibility impact, validation results, and unresolved decisions in the PR description.

Implementation PRs use the **current executable harness policy**. The proposed design in
[validation.md](design/validation.md) is implemented by P01-P04; it is not an implicit exception to
existing policy. Once those items land, record the chosen tests, why they cover the diff, and any
required evidence still blocked by unavailable hardware.

[Developer automation](design/automation.md) is planned in P01a-P01c. Once implemented, hooks and
CI will call the same validators; hook installation will be an explicit development setup step.
The current roadmap PR does not yet provide those commands or install Git hooks.

For each published checkpoint, record the tag, immutable target commit, release link, compatible
dependency versions, capability restrictions, and validation artifacts in its milestone's release
record. Use annotated tags and keep existing published tags immutable. Align package metadata with
the release version; package prerelease spelling may use PEP 440, for example `2.0.0a1` for Git tag
`v2.0.0-alpha.1`. Patch tags are reserved for actual fixes rather than future feature buckets.

New 1.x work preserves existing public configuration and extension entry points. Any necessary
incompatible change belongs in v2 with an explicit migration. Distinguish weight import, restarting
training, and exact-state continuation in compatibility claims.

## Maintaining these documents

Keep the index short, outcomes in milestone files, and shared rationale in `design/`. Update the
smallest authoritative section and link it elsewhere. Plans describe desired behavior; current
API instructions belong under `guidance/`. Claims about dependencies or an open PR need a date and
a pinned source or reproducible experiment. Resolve or supersede an old observation rather than
copying it into later milestones as a permanent blocker.

For a documentation change, check relative links and P-number dependencies, run the harness checks
if its router changes, and review the resulting diff. A prose-only roadmap update does not need
training experiments.
