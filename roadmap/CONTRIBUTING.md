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

The next development entry is [P19a differentiable feedback](milestones/v0.5x-v0.6.md#p19a),
followed by ReFL, component execution policies and DRTune. P15a and P18 ART are merged;
[P18 experiment/publication follow-through](milestones/v0.5x-v0.6.md#p18) is separate.
[P15b EP model admission](milestones/v0.6-v0.7.md#p15b) remains deferred until P21 qualification.
P01–P14 are complete and v0.5.0 is published. Start from latest main; historical P11/P12 handoffs
retain evidence, not an instruction to rerun or reopen their completed stacks. A concrete new
model layout still needs its own admission and evidence.

A contribution need not implement an entire milestone or require a large GPU cluster. Preserve
P00–P21 and existing child IDs. P19–P21 split the next algorithm/component work into coherent
PRs; P15/P16 retain the deferred EP work; the milestone pages own their status and dependency links.

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

An open PR marked ready remains `in_progress` until merged. Distinguish implemented candidates,
CPU/Gloo checks, development GPU probes and complete exact-source qualification in each checkpoint.
A selected-case count is not a physical GPU count. Record confirmed collective/correctness defects
separately from accepted floating-point or asynchronous scheduling variation.
Retained-tensor diagnostics do not convert a failed selected case into a pass. Review fixture
acceptance changes before a new run, preserve the original failure, and qualify the final source.
Keep partial execution counts in the handoff; the milestone records the remaining acceptance boundary.
Keep actual-parent and cumulative results separate, with their source and plan identities; quality
cases belong within the reported CPU subtotal. A reviewed but unexecuted capture is not evidence.
Deferring an execution round does not remove its algorithm/model obligations from the milestone.

For the bounded A–E strategy, deterministic CPU tests exercise each real objective's branches and
upstream derivatives; native cross-model cases exercise shared preparation, backward and lifecycle
owners. Record that composition and every unexecuted endpoint explicitly. A leaf objective change
does not require an unrelated backend matrix when those owners are unchanged. Changes to a shared
owner still select all affected consumers, with real transitive test files bound to the evidence.

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
For an uncommitted development probe, retain the patch and untracked-file hashes; do not present
that probe as the immutable candidate's selected evidence. Keep current actions in the handoff
and link historical numerical records instead of repeating them across the index and milestone.

## Review and release records

Use the repository's English PR-title convention, for example
`[docs] docs: add the distributed training roadmap`. Include the work-item IDs, resulting behavior,
compatibility impact, validation results, and unresolved decisions in the PR description.

Implementation PRs use the **current executable harness policy**, P04 automatic selection and
P01b's manual procedure where a reviewed custom plan is needed. [The design](design/validation.md)
records the rationale for the merged fixtures and selection mechanism.
Record the chosen tests, why they cover the diff, and any required evidence still blocked by
unavailable hardware.

[Developer automation](design/automation.md) spans P01a-P01c. P01a has landed: hooks and CPU CI
call the same validators, and hook installation remains an explicit development setup step.
Use the [quality workflow](../docs/development/validation/quality.md) for commands and [validation plans](../docs/development/validation/validation.md)
for merged P01b policy; [repository consistency](../docs/development/validation/repository-consistency.md) describes
the merged P01c implementation.

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
