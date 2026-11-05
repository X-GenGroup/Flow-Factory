# Flow-Factory roadmap

Flow-Factory's next releases focus on composable distributed training, developer automation and a
proportionate agent validation harness. TP/SP and P01a-P01c quality automation have landed on main.
P02-P08 from native GitHub stack #284 are merged. Fresh
[merged-baseline qualification](../guidance/integration-qualification.md) covers the integrated
source and focused cross-node paths. P09 and P10 are delivered for review in native stack #291;
P11 algorithm/state integration reached a local numerical review checkpoint. On 2026-11-05 the
user authorized resuming at P11a and expanding v0.5 to the existing model/algorithm registry,
including CP evaluation, reward/optimization overlap and training-state continuation. See the
[revised delivery sequence](milestones/v0.5.md#delivery-sequence). Expert parallelism remains
future work; P15 records the requested DiT-MoE reference models. Version 1.0
is the final stabilization boundary for that declared scope; development checkpoints advance from the existing v0.1.0
through v0.2.0 and later 0.x versions.

This directory records planned work, dependencies, acceptance criteria, and contribution entry
points. It does not declare a feature supported or replace the current development rules in
[AGENTS.md](../AGENTS.md). Runtime behavior and released support remain documented in
[guidance](../guidance/workflow.md) and the [changelog](../CHANGELOG.md). The proposed validation
policy is operational through the merged P01b implementation; future designs still need their
listed implementation and acceptance evidence.

## Start here

| Need | Read |
|---|---|
| Find a release or work item | The release table below and its linked milestone |
| Claim work, hand it off, or update its status | [Contributing to the roadmap](CONTRIBUTING.md) |
| Understand ownership and upstream reuse | [Architecture direction](design/architecture.md) |
| Plan pre-commit hooks, CI and consistency checks | [Developer automation](design/automation.md) |
| Understand the proposed risk and GPU evidence model | [Validation design](design/validation.md) |
| Implement an extension using today's API | [Development guide](../AGENTS.md) and [model guide](../guidance/new_model.md) |

P00-P17 are stable planning IDs, not GitHub PR numbers. P01 has three implementation children,
P01a-P01c, keeping the existing top-level IDs stable. Each item lives in exactly one milestone
file; its status, dependencies, and coordination link are maintained there. A work item normally
maps to one coherent PR, with justified splits recorded before implementation. This roadmap
directory records P00's source checkpoint, completed P01a-P01c automation, and the merged
qualification/runtime changes in P02-P08. Their PRs and current status live in the milestone files.

## Release sequence

The v0.2 source checkpoint records already-merged code and its existing evidence. Later rows are
planned release checkpoints without calendar commitments. Tag each future integration commit only
after its work, version metadata and required evidence are complete. The earlier v1.x/v2.x plan
was never published as tags and is superseded by this sequence; preserve the actual v0.1.0 tag.

| Tag | Required work | Release promise |
|---|---|---|
| [v0.2.0](milestones/v0.2.md) | P00; merged #270 and #274 | Source checkpoint at `55dcd694` and a published CUDA image: initial FSDP2 TP/SP and async reward device fix, with recorded limits |
| [v0.3.0](milestones/v0.3.md) | P01-P04, including P01a-P01c | Fast hooks/CI, repository consistency, qualified TP/SP support and diff-driven validation |
| [v0.4.0](milestones/v0.4.md) | P05-P08 | Clear topology, preparation, parameter-state, and tile-stream ownership |
| [v0.5.0-alpha.1](milestones/v0.5.md#p09) | P09, on the v0.4 baseline | First complete CP training path |
| [v0.5.0-alpha.2](milestones/v0.5.md#p10) | P10 | TP + SP + CP composition on the first model |
| [v0.5.0-beta.1](milestones/v0.5.md#p12) | P11-P12 | Algorithm/state, eval/overlap/resume and registry coverage disposition complete; scope frozen |
| [v0.5.0-rc.1](milestones/v0.5.md#p13) | P13 | Migration, dependency compatibility, examples, and release evidence complete |
| [v0.5.0](milestones/v0.5.md#p14) | P14 | Qualified TP/CP/TP+CP model-algorithm coverage, eval, overlap and state continuation |
| [v0.6.0](milestones/v0.6-v0.7.md#p15) | P15 | One real MoE model with a correct NCCL EP implementation |
| [v0.7.0](milestones/v0.6-v0.7.md#p16) | P16 | Optional DeepEP transport with a validated NCCL fallback |
| [v1.0.0-rc.1](milestones/v1.0.md#p17), then [v1.0.0](milestones/v1.0.md#release-records) | P17, after P00-P16 | Qualified complete scope, documented compatibility guarantees, maintainable ownership and final integration evidence |

[P01a](milestones/v0.3.md#p01a) supplies the merged hooks and basic CPU CI; P01b supplies the merged
manual selection policy and P01c supplies merged consistency checks. P01b/P02 provide a usable manually selected gate before P04
automates selection. P03 qualifies the merged TP/SP implementation; do not reopen #270 or rebuild
its plans. P03 and P04 can progress independently after those prerequisites. The CP
feasibility probe starts during P06; it need not wait for all runtime cleanup. Production CP
releases build on the completed v0.4 baseline.

P02 #281, P03 #282, P04 #283, P05 #285, P06 #286, P07 #287 and P08 #288 are merged.
The proposed `v0.3.0` target is the P04 integration commit `17e8be7c` (metadata `0.3.0`);
the proposed `v0.4.0` target is P08 `fbb038e3` (metadata `0.4.0`). No tag is implied by the
completed work items. Verify each release target's exact-source evidence before publishing;
the new cumulative report covers `fbb038e3`, not the earlier P04 target.

## Starting evidence

Last checked: **2026-11-05**, after P02-P08 merged and the final source passed cumulative
qualification plus focused two-node validation. Detailed limitations, evidence provenance
and source-tag metadata are recorded in [the v0.2 baseline](milestones/v0.2.md).

- The previous GitHub release is `v0.1.0`; its tag points to
  [`5ab10a9f`](https://github.com/X-GenGroup/Flow-Factory/commit/5ab10a9f4d36b9271865903aca1fa3c28afe11cb).
  The package metadata at the v0.2 source checkpoint still says `0.1.0`; a source tag does not
  publish or relabel a package artifact.
- [v0.2.0](https://github.com/X-GenGroup/Flow-Factory/releases/tag/v0.2.0) now includes the tested
  CUDA image, source build bundle and verification records. [P01a](milestones/v0.3.md#p01a) merged
  separately in [#276](https://github.com/X-GenGroup/Flow-Factory/pull/276). P01b manual selection
  merged in [#279](https://github.com/X-GenGroup/Flow-Factory/pull/279); P01c merged in
  [#280](https://github.com/X-GenGroup/Flow-Factory/pull/280) as `27a2ab15`, with passing
  [main checks](https://github.com/X-GenGroup/Flow-Factory/actions/runs/41830080508).
- [Async reward device PR #274](https://github.com/X-GenGroup/Flow-Factory/pull/274) merged as
  [`a3610cdc`](https://github.com/X-GenGroup/Flow-Factory/commit/a3610cdc64c7217ef0cdafbcb8d3c1a799e8d08a).
  Workers select their CUDA device before entering the reward stream, preventing stray GPU-0
  contexts. Preserve its focused two-GPU regression in the new test catalog.
- [TP/SP PR #270](https://github.com/X-GenGroup/Flow-Factory/pull/270) then merged as
  [`55dcd694`](https://github.com/X-GenGroup/Flow-Factory/commit/55dcd69433deb11159d5c03c22689518ce6f315e),
  the inspected main and v0.2 source target. It includes replica-scoped data/RNG, reward agreement,
  training-aware LoRA/variant plans, collective cleanup and checkpoint tests. These are mechanisms
  to preserve and consolidate, not future features to recreate.

## Next agent handoffs

The item metadata in the milestone files remains authoritative; these are suggested entry points,
not assignments or a second status board.

| Entry point | First action | Dependency boundary |
|---|---|---|
| Release preparation | Inspect the pinned P04/P08 targets and their exact-source evidence | Publishing tags remains a separate delivery action |
| P09/P10 review | Review [#290](https://github.com/X-GenGroup/Flow-Factory/pull/290) and [#292](https://github.com/X-GenGroup/Flow-Factory/pull/292), stacked after integration #289 | Exact-source evidence and checks are attached; the PRs are not merged |
| P11 review checkpoint | Review the [implementation, numerical findings and remaining delivery work](milestones/v0.5.md#p11-review-checkpoint) | Resume with P11a; deliver coherent child PRs with exact-source evidence and check every pushed head |

## Scope of v1.0

Keep DDP, DeepSpeed ZeRO-2, and FSDP2 as the primary existing backend configurations. Advanced
TP/SP/CP support initially targets FSDP2 and explicitly declared model layouts. The release must
cover online feedback, offline training, and multi-role execution through representative semantic
paths, with documented restrictions and reproducible evidence.

The harness is part of this release's core functionality. NCCL EP for one real MoE model and an
optional DeepEP transport enter through v0.6/v0.7; v1.0 qualifies their declared scope alongside
TP/SP/CP, algorithms and state recovery. This does not promise every EP x TP x CP combination.
GPU selection follows affected contracts and valid topologies; a fixed 45-job, 32-rank campaign
and a fixed reward deployment are
not the proposed default gate. CP transports/layouts needed for the accepted v0.5 matrix belong to P12; unrestricted EP composition,
arbitrary in-flight pipeline recovery and concurrent rollout/training remain separate features.
