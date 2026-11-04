# Flow-Factory roadmap

Flow-Factory's next releases focus on composable distributed training, developer automation and a
proportionate agent validation harness. The sequence is: establish a release baseline, add fast
hooks and CI, finish tensor/sequence parallelism, clarify runtime ownership, add context
parallelism, and then extend to expert parallelism for a real MoE model.

This directory records planned work, dependencies, acceptance criteria, and contribution entry
points. It does not declare a feature supported or replace the current development rules in
[AGENTS.md](../AGENTS.md). Runtime behavior and released support remain documented in
[guidance](../guidance/workflow.md) and the [changelog](../CHANGELOG.md). The proposed validation
policy becomes operational through P01b, not by merging this roadmap.

## Start here

| Need | Read |
|---|---|
| Find a release or work item | The release table below and its linked milestone |
| Claim work, hand it off, or update its status | [Contributing to the roadmap](CONTRIBUTING.md) |
| Understand ownership and upstream reuse | [Architecture direction](design/architecture.md) |
| Plan pre-commit hooks, CI and consistency checks | [Developer automation](design/automation.md) |
| Understand the proposed risk and GPU evidence model | [Validation design](design/validation.md) |
| Implement an extension using today's API | [Development guide](../AGENTS.md) and [model guide](../guidance/new_model.md) |

P00-P16 are stable planning IDs, not GitHub PR numbers. P01 has three implementation children,
P01a-P01c, keeping the existing top-level IDs stable. Each item lives in exactly one milestone
file; its status, dependencies, and coordination link are maintained there. A work item normally
maps to one coherent PR, with justified splits recorded before implementation. This roadmap
documentation PR does not complete any of those implementation items.

## Release sequence

These are proposed release checkpoints, without calendar commitments. Tag the reviewed integration
commit only after all required work and evidence for that checkpoint are complete.

| Proposed tag | Required work | Release promise |
|---|---|---|
| [v1.0.0](milestones/v1.0.md) | P00 | Explicit baseline for existing functionality and compatibility |
| [v1.1.0](milestones/v1.1.md) | P01-P04, including P01a-P01c | Fast hooks/CI, repository consistency, supported TP/SP and diff-driven validation |
| [v1.2.0](milestones/v1.2.md) | P05-P08 | Clear topology, preparation, parameter-state, and tile-stream ownership |
| [v2.0.0-alpha.1](milestones/v2.0.md#p09) | P09, on the v1.2 baseline | First complete CP training path |
| [v2.0.0-alpha.2](milestones/v2.0.md#p10) | P10 | TP + SP + CP composition on the first model |
| [v2.0.0-beta.1](milestones/v2.0.md#p12) | P11-P12 | Algorithm/state integration and a second model structure; feature scope frozen |
| [v2.0.0-rc.1](milestones/v2.0.md#p13) | P13 | Migration, dependency compatibility, examples, and release evidence complete |
| [v2.0.0](milestones/v2.0.md#p14) | P14 | Stable declared TP/SP/CP capabilities and the new harness |
| [v2.1.0](milestones/v2.x.md#p15) | P15 | One real MoE model with a correct NCCL EP implementation |
| [v2.2.0](milestones/v2.x.md#p16) | P16 | Optional DeepEP transport with a validated NCCL fallback |

[P01a](milestones/v1.1.md#p01a) starts early with hooks and basic CPU CI; P01b and P01c then migrate
policy and add consistency checks. P01b/P02 provide a usable manually selected gate before P04
automates selection. P03 and P04 can progress independently after those prerequisites. The CP
feasibility probe starts during P06; it need not wait for all runtime cleanup. Production CP
releases build on the completed v1.2 baseline.

## Starting evidence

Last checked: **2026-11-04**. These are historical anchors, not a live release dashboard.

- The visible release/tag and package version were `v0.1.0`; the tag points to
  [`5ab10a9f`](https://github.com/X-GenGroup/Flow-Factory/commit/5ab10a9f4d36b9271865903aca1fa3c28afe11cb).
  `v1.0.0` above is a proposed new baseline, not a claim that it already exists.
- The inspected main commit was
  [`2a115133`](https://github.com/X-GenGroup/Flow-Factory/commit/2a11513333deb8c4c2c2658ba7f5ad351172138f).
  Existing online/offline contracts, component variants, named snapshots, and reward/optimization
  overlap are starting mechanisms to refine, not new features to recreate.
- [TP/SP PR #270](https://github.com/X-GenGroup/Flow-Factory/pull/270) was open at
  [`11007d41`](https://github.com/X-GenGroup/Flow-Factory/commit/11007d41ea1117b0672dcafe4e5509b950fcfd08).
  Its [implementation notes at that revision](https://github.com/X-GenGroup/Flow-Factory/blob/11007d41ea1117b0672dcafe4e5509b950fcfd08/.agents/knowledge/topics/tensor_parallel.md)
  are evidence for P03. Recheck the current PR before carrying a finding forward; SP defaults are
  already adapter-specific there.

## Scope of v2.0

Keep DDP, DeepSpeed ZeRO-2, and FSDP2 as the primary existing backend configurations. Advanced
TP/SP/CP support initially targets FSDP2 and explicitly declared model layouts. The release must
cover online feedback, offline training, and multi-role execution through representative semantic
paths, with documented restrictions and reproducible evidence.

The harness is part of this release's core functionality. GPU selection follows affected
contracts and valid topologies; a fixed 45-job, 32-rank campaign and a fixed reward deployment are
not the proposed default gate. EP, additional CP transports/layouts, arbitrary in-flight pipeline
recovery, and concurrent rollout/training follow separate acceptance boundaries.
