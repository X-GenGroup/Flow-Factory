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

Useful starting contributions include risk-policy migration for [P01b](milestones/v0.3.md#p01b),
consistency checks for [P01c](milestones/v0.3.md#p01c), cataloging existing tests for P02, replaying
historical diffs for P04, reproducing a specific P03 layout, and reviewing CP input contracts for
P12. A contribution does not need to implement an entire milestone or access a large GPU cluster.
Start from current main, which contains #270, #274 and the P01a automation from #276. P03's
implementation is merged; its follow-up owns qualification and fixes. Read [the baseline](milestones/v0.2.md)
before carrying an old TP finding or campaign result into a new task. Preserve P00-P16 and the
P01 child IDs; P17 is the new final v1.0 qualification item.

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

[Developer automation](design/automation.md) spans P01a-P01c. P01a has landed: hooks and CPU CI
call the same validators, and hook installation remains an explicit development setup step.
Use the [quality workflow](../guidance/quality.md) for commands; P01b/P01c own the remaining policy
migration and repository consistency work.

For each published checkpoint, record the tag, immutable target commit, release link, compatible
dependency versions, capability restrictions, and validation artifacts in its milestone's release
record. Use annotated tags and keep existing published tags immutable. A tag link is sufficient
for a source-only checkpoint; do not invent a GitHub Release or package artifact that was not
published. The historical v0.2.0 source checkpoint explicitly retains package metadata `0.1.0`.
For future release-preparation commits, align package metadata with the release version; package
prerelease spelling may use PEP 440, for example `0.5.0a1` for Git tag `v0.5.0-alpha.1`. Patch tags
are reserved for actual fixes rather than future feature buckets.

During 0.x, preserve existing public behavior where practical and document/test every necessary
migration; a pre-1.0 version is not permission for silent breakage. P17 freezes the published 1.0
compatibility guarantees. Subsequent incompatible public changes require a major-version decision.
Distinguish weight import, restarting training, and exact-state continuation in compatibility claims.

## Maintaining these documents

Keep the index short, outcomes in milestone files, and shared rationale in `design/`. Update the
smallest authoritative section and link it elsewhere. Plans describe desired behavior; current
API instructions belong under `guidance/`. Claims about dependencies or an open PR need a date and
a pinned source or reproducible experiment. Resolve or supersede an old observation rather than
copying it into later milestones as a permanent blocker.

For a documentation change, check relative links and P-number dependencies, run the harness checks
if its router changes, and review the resulting diff. A prose-only roadmap update does not need
training experiments.
