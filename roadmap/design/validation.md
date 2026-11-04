# Proposed validation design

This document specifies the outcome of [P01-P04](../milestones/v0.3.md), not the current executable
merge policy. Its purpose is to provide strong evidence for a diff at a proportionate cost as
models, backends, TP/SP/CP, and eventually EP increase the configuration space.

[Developer automation](automation.md) assigns local hooks, PR CI and heavier jobs to appropriate
stages. P01a establishes their entry points; P01b owns the semantic policy migration; P01c adds
repository consistency checks. Cheap static success does not replace the semantic evidence below.

## Separate impact, proof, and resources

Keep four independent decisions: semantic impact, affected contracts, evidence needed, and the
smallest valid execution topology. A high-impact change can have a small decisive test; a local
formula can require a strong numerical oracle.

| Proposed impact level | Meaning |
|---|---|
| R0 | Read-only inspection |
| R1 | Local change within an unchanged contract |
| R2 | Component-local composition or implementation |
| R3 | State ownership, schema, collective, resume or backend semantics |
| R4 | Shared execution or contracts spanning multiple owners |

R4 requires coverage of every affected obligation, not a fixed cluster size or full Cartesian
product. File paths and keywords help find owners but cannot decide impact alone. For example,
logging in BaseTrainer and arithmetic in distributed geometry need different treatment even when
path-only rules suggest the opposite.

Partition the proof surface by owner: algorithm mathematics; adapter I/O and replay; backend
preparation/update/state; parallel layout and sample geometry; acquisition/feedback/tile lifecycle;
and checkpoint/resume. Inspect every registry implementation reached by a shared contract. Run
cheap contract checks broadly and select GPU representatives by actual behavioral differences.

## Evidence levels

These levels are selectable evidence types, not a mandatory ladder for every PR.

| Level | Evidence | Typical use |
|---|---|---|
| Q0 | CPU math/contract tests and Gloo process-group tests | Formula, routing, identity, collective membership, state transitions |
| Q1 | One GPU with a real model class at a small valid configuration | CUDA precision, autograd, LoRA, recomputation and local model behavior |
| Q2 | Actual distributed backend on the smallest valid mesh | Preparation, sharded updates, parallel composition and distributed state |
| Q3 | Selected real pretrained weights/configuration | Loading, weight layout, dtype/quantization and production adapter compatibility |

Capacity, throughput, and large deployment qualification are separate experiments with their own
questions. Real reward models are selected when their adapter or measured performance is relevant;
they are not a dependency of every training correctness test.

Small model fixtures must preserve the structure being tested, such as dual streams, head norms,
conditioning broadcasts, trainable adapters, or multiple roles. An unrelated tiny Linear cannot
prove a production model plan. Keep meaningful nonzero gradients and multiple updates where
optimizer momentum or state isolation matters.

## Selecting a plan

P04 should implement this explainable pipeline with explicit owner mappings, changed symbols,
registry inspection, and agent review. A whole-program static analyzer is not a prerequisite.

1. Inspect the diff and identify changed owners and contracts; ignore unrelated binary/artifact
   trees when collecting scope.
2. Enumerate proof obligations and known interactions affected by the change.
3. Match those obligations to a test catalog containing coverage, fixture, backend, mesh,
   placement, reward mode, oracle and expected cost.
4. Select the smallest valid covering set, satisfying mandatory interactions before optimizing
   cost. Constrained pairwise coverage can supplement known higher-order cases.
5. Report every selected test's reason and any uncovered required obligation. Uncertain scope
   needs investigation or an explicit addition, not an empty successful plan.
6. Bind results to the tested source, test/config revisions, dependency versions and topology.

Known interactions deserve explicit cases, including FSDP2 + TP/SP + LoRA, activation recomputation
with asynchronous collectives, and role-isolated updates with an already-populated optimizer state.
A changed model-specific plan needs that model structure, even if another model covers the same
broad family.

| Diff | Expected selection |
|---|---|
| Backend-independent objective formula | Q0 oracle, optionally one CUDA reference; no automatic backend sweep |
| One model's conditioning or parallel plan | Its structure/configuration and affected mode; Q3 when weights/loading matter |
| Shared optimizer preparation | Affected backend semantics on representative models/roles |
| DP/TP/CP geometry | Identity, group membership, loss scaling and the interacting mesh dimensions |
| Reward/tile lifecycle | Controlled multi-tile scheduling and cleanup plus necessary GPU integration |
| CUDA reward worker device selection (#274) | Fresh-process two-GPU context/stream assertion; no automatic model or backend matrix |
| Checkpoint state | Save/load and next-update equivalence for affected state types |
| Harness policy/planner | CPU routing tests and historical-diff replay |

## Topology selection

Two local GPUs are the default distributed budget. Two machines with one GPU each test a basic
cross-machine path when that path changes. Choose a larger shape when the actual contract requires
it, rather than shrinking a test until it no longer exercises the claim.

| Obligation | Typical minimum or constraint |
|---|---|
| One nontrivial distributed group | 2 ranks |
| TP primitive | TP=2, 2 ranks |
| Merged TP implementation's FSDP2 training path | DP-shard=2, TP=2, 4 ranks |
| Independent TP and CP axes | TP=2, CP=2, at least 4 ranks; backend constraints still apply |
| Direct DP + TP + CP interaction | D=2, T=2, C=2, 8 ranks |
| Multiple subgroups or hierarchical placement | Enough ranks to make each claimed subgroup nontrivial; often 4 |
| Intra-node and inter-node hierarchy together | Often 2 nodes x 2 GPUs |

Use logical data replicas, not physical world size, to compute dataset and batch expectations.
Compare parallel/unsharded references on the same logical samples and update geometry. Changing
both parallel layout and effective batch would confound the result.

## Oracles and asynchronous rewards

Check trainable/gradient membership, elementwise gradients, parameter updates, inactive parameter
and optimizer-state isolation, and checkpoint continuation as appropriate. Finite loss, checksums,
or a zero exit status alone do not establish numerical equivalence. Set justified tolerances from
the same model/precision/reference rather than using a loose global gradient-norm threshold.

Use deterministic rewards with variation within groups for training fixtures. Constant group
rewards can eliminate the learning signal. Controlled events and multiple tiles should exercise
ready/ordered scheduling, out-of-order completion, multi-source identity, backpressure, cancellation
and failure cleanup. Use a lightweight service when testing the HTTP boundary.

Check bounded live resources across repeated iterations after warmup, including pending work and
retained tensors. A single low memory measurement cannot establish cleanup. Measure event ordering
for functional reward/optimization overlap; use a separate timing experiment to quantify hidden
reward latency under a specified workload.

Reuse [the worker-device regression](../../tests/rewards/test_async_reward_worker_device.py) for
CUDA thread/stream ownership. Keep a clean process so an earlier test cannot pre-initialize GPU 0
and invalidate its oracle. TP peer reward agreement and collective settling have separate tests;
one device-context assertion is not proof of those distributed contracts. Controlled scheduling
tests should not rely on a loaded machine completing work within a two-second wall-clock window.

## Results and policy migration

Results distinguish `pass`, `fail`, `blocked`, and `not_selected`. Missing required hardware is
blocked evidence, not a pass; an unrelated unselected capacity experiment is not a failed gate.
Retain immutable tested commits and test/config/environment identities. Initially allow evidence
reuse only for demonstrably irrelevant changes, such as prose updates with unchanged code and
fixtures; defer sophisticated caching.

P01b updates policy, skills, risk routes, validators and their tests coherently, providing a manual
selected-plan path before P04 automation. Preserve previous campaign reports as historical evidence
or optional experiments. Tests must validate proof obligations and topology requirements rather
than freeze incidental values such as 32 training GPUs or one reward-server deployment.

Calibrate the planner on historical changes, including a BaseTrainer logging edit, distributed
geometry arithmetic, an objective-only edit, a role/state change, and a validator change. Record
both missed necessary tests and unnecessary work. The release's support claims determine broader
qualification; they do not make every PR run every supported combination.

## P01b implementation preparation

Inspected against main `5acbc75e` on 2026-11-04, after P01a and the container preflight fix.
This preparation does not activate the proposed policy or implement the P04 planner.

### Current failure modes

Small synthetic changes passed to the current `classify_changes` / `attach_evidence` functions
produce the following results. These are routing probes, not claims that historical PRs contained
these exact changes.

| Probe | Current result | Required design response |
|---|---|---|
| A literal `logger.info` message in `trainers/abc.py` | R4 and a complete GPU campaign | Permit a narrow, proven observational edit without downgrading lifecycle changes |
| Rank arithmetic in `utils/dist.py`, without a collective keyword | R1, scope only | Recognize geometry ownership even when the changed lines contain no communication call |
| Shard arithmetic in `models/tensor_parallel.py` | R1, model-adapter facet | Recognize the shared layout contract and its affected adapters |
| Result-status validation in `scripts/validate_gpu_campaign.py` | R1, scope only | Classify validation-policy changes as harness work with CPU evidence |

`agent_scope.py` also enumerates every untracked file and decodes its entire contents before
classification. P01b must bound scope collection before reading data. Prefer an explicit tested
snapshot and staged additions; any optional untracked inspection must report its limits and avoid
traversing checkpoint/output trees. An omitted or unclassified change must remain visible.

### One coherent migration

| Owner | Files and change boundary |
|---|---|
| Canonical policy | `AGENTS.md`, constraint 30, `knowledge/topics/gpu_validation.md`: require all affected proof obligations, separate optional campaigns from per-diff gates |
| Skill routing | Develop/review/new-model/new-algorithm/new-reward/new-accelerator skills and their selected references: remove unconditional campaign instructions together |
| Scope and evidence rules | `scripts/agent_scope.py`, `.agents/harness/risk_rules.yaml`, `test_routes.yaml`, `evals/cases.yaml`: align owner coverage, semantic impact and evidence selection |
| Manual plan validation | Add a small selected-plan/result validator; keep `validate_gpu_campaign.py` and the old manifest usable for historical or explicitly requested campaigns |
| Regression and CI | Harness tests, `tests/scripts/test_validate_gpu_campaign.py`, and the P01a CPU entry point: test the new gate and the retained historical reader |

The old campaign tests currently assert 45 jobs, a 32-rank cluster and particular reward-service
deployments. Preserve such assertions only as tests of that historical fixture. Replace the test
that requires the canonical policy to advertise "45 mandatory jobs" with tests of the new policy.
Changing prose alone would leave an executable contradiction.

The manual contract should contain an immutable base/tested revision, changed owners and risk
facets, required obligation IDs, selected cases with reasons, fixture/config identities, Q-level,
backend, mesh and placement, oracle, and actual result/environment identities. Validate complete
coverage of declared obligations, actual backend/topology agreement, duplicate or unknown IDs,
stale evidence, and `pass` / `fail` / `blocked` / `not_selected` semantics. A required obligation
cannot pass through a missing result, a skipped case or an unavailable resource. Human review
still checks that the declared obligations cover the diff; schema validation cannot infer that.

Start with explicit case selection and a small set of representative fixtures. Cost optimization,
whole-catalog discovery, and automatic set cover belong to P04. Q0 can include Torch/Gloo tests;
it does not mean every Q0 test belongs in P01a's dependency-light quality environment.

### Acceptance before opening the implementation PR

1. Reproduce the four routing probes above, then add cases for objective-only edits, real shared
   lifecycle changes, parameter-role isolation, and mixed code/documentation changes. A logging
   exemption must reject changed argument side effects or surrounding control flow.
2. Test bounded collection with binary checkpoints, large files, renames, deletions, staged new
   files and linked worktrees. Use temporary fixtures; do not scan a user's training artifacts.
3. Validate manual plans for one-GPU behavior, a two-rank local group, one rank per machine,
   and the current four-rank FSDP2+TP path. Reject invalid smaller meshes instead of weakening the
   production contract. Future CP/EP layouts remain unavailable until implemented.
4. Exercise failing plan/result fixtures: missing obligations, missing required evidence,
   wrong commit/config/backend/topology, duplicate results and false async-overlap claims.
   Deterministic event scheduling must prove overlap ordering; constant rewards or arbitrary
   sleeps are not numerical or performance evidence.
5. Run the policy/harness/legacy-reader CPU suites and the quality gates on the final diff. Inspect
   all remaining references to the old mandatory campaign. Publish the selected evidence and
   unresolved limits in the PR; leave merging to the maintainer.

P01c can prepare consistency rules against the stable P01a entry point, but changes to that shared
entry point must be coordinated with P01b. P02 can inventory the existing TP/SP, real Gloo geometry,
fresh-process reward-device, overlap and checkpoint tests now; its executable catalog should use
P01b's accepted obligation/result contract. Keep these preparations out of the P01b implementation
unless they are necessary to validate that contract.

After any dependency PR merges, fetch main, inspect both the textual diff and unchanged consumers
of altered contracts, and rerun the affected checks before requesting merge. CI success and a
clean Git merge do not prove that a new external-service probe behaves correctly; exercise its
success and rejection responses before introducing it as a release gate.
