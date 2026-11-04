# Proposed validation design

This document specifies the outcome of [P01-P04](../milestones/v1.1.md), not the current executable
merge policy. Its purpose is to provide strong evidence for a diff at a proportionate cost as
models, backends, TP/SP/CP, and eventually EP increase the configuration space.

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
| Current TP PR's FSDP2 training path | DP-shard=2, TP=2, 4 ranks |
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

## Results and policy migration

Results distinguish `pass`, `fail`, `blocked`, and `not_selected`. Missing required hardware is
blocked evidence, not a pass; an unrelated unselected capacity experiment is not a failed gate.
Retain immutable tested commits and test/config/environment identities. Initially allow evidence
reuse only for demonstrably irrelevant changes, such as prose updates with unchanged code and
fixtures; defer sophisticated caching.

P01 updates policy, skills, risk routes, validators and their tests coherently, providing a manual
selected-plan path before P04 automation. Preserve previous campaign reports as historical evidence
or optional experiments. Tests must validate proof obligations and topology requirements rather
than freeze incidental values such as 32 training GPUs or one reward-server deployment.

Calibrate the planner on historical changes, including a BaseTrainer logging edit, distributed
geometry arithmetic, an objective-only edit, a role/state change, and a validator change. Record
both missed necessary tests and unnecessary work. The release's support claims determine broader
qualification; they do not make every PR run every supported combination.
