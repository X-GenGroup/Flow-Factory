# Flow-Factory roadmap

Latest reviewed baseline: main `e3f8e06d` (2026-11-12), including ART/shared replay #377, reference MSE #379 and async media logging #380.
P01–P14 are merged and [v0.5.0 is published](https://github.com/X-GenGroup/Flow-Factory/releases/tag/v0.5.0).
The historical v0.5.0 19-model / 14-trainer TP/SP/Ulysses-CP inventory, SP completion, shared parallelism owners,
P12j sharded loading and P13a evidence continuity remain the implemented foundation. Historical
[algorithm](handoffs/v0.5-algorithm-closure.md), [model](handoffs/v0.5-model-closure.md) and
[SP](handoffs/v0.5-sp-completion.md) handoffs retain original evidence identities, not current tasks.

P15a user documentation merged in #376. [P18 DiffusionART](milestones/v0.5x-v0.6.md#p18)
merged in [#377](https://github.com/X-GenGroup/Flow-Factory/pull/377), bringing the current registry
to 15 trainers; its supported compositions and evidence are recorded in the
[ART guide](../guidance/training/art.md), not inferred from the historical matrix. The next
development work is the P19a/P19b stack for [differentiable rewards and ReFL](milestones/v0.5x-v0.6.md#p19).
After review and merge, [P20a](milestones/v0.5x-v0.6.md#p20a) is the next implementation entry.
ART experiment follow-through and v0.5.x publication remain separate from that dependency.
[P19–P21](milestones/v0.5x-v0.6.md#p19) deliver ReFL,
differentiable rewards, component execution policies and DRTune for v0.6. Backend expansion
waits: P15b–P15f target NCCL EP in v0.7; P16 targets optional DeepEP in v0.8. P17 remains the
v1.0 stabilization boundary. See the [derivative/placement design](design/differentiable-rewards.md).

Use independent worktrees and native Stack PRs when work is dependent; the user reviews and merges.
Reuse valid GPU receipts at their original sources. Changed contracts select targeted local jobs
within eight GPUs, not a Cartesian matrix, cross-node campaign or fixed reward deployment.
Publication of v0.5.0 and the source-only v0.5.1 tag does not publish the later #375 or #377
source as an image or a new release. See [installation identities](../guidance/installation.md).

This directory records planned work, dependencies, acceptance criteria, and contribution entry
points. It does not declare a feature supported or replace the current development rules in
[AGENTS.md](../AGENTS.md). Runtime behavior and released support remain documented in
[guidance](../guidance/training/workflow.md) and the [changelog](../CHANGELOG.md). The proposed validation
policy is operational through the merged P01b implementation; future designs still need their
listed implementation and acceptance evidence.

## Start here

| Need | Read |
|---|---|
| Find a release or work item | The release table below and its linked milestone |
| Claim work, hand it off, or update its status | [Contributing to the roadmap](CONTRIBUTING.md) |
| Understand ownership and upstream reuse | [Architecture direction](design/architecture.md) |
| Plan differentiable algorithms and auxiliary component placement | [Differentiable feedback design](design/differentiable-rewards.md) |
| Plan EP model admission, ownership and upstream reuse | [EP design checkpoint](design/expert-parallelism.md) |
| Plan pre-commit hooks, CI and consistency checks | [Developer automation](design/automation.md) |
| Understand the proposed risk and GPU evidence model | [Validation design](design/validation.md) |
| Implement an extension using today's API | [Development guide](../AGENTS.md) and [model guide](../docs/development/new-model.md) |

P00-P21 are stable planning IDs, not GitHub PR numbers. P01 has three implementation children,
P01a-P01c, keeping the existing top-level IDs stable. Each item lives in exactly one milestone
file; its status, dependencies, and coordination link are maintained there. A work item normally
maps to one coherent PR, with justified splits recorded before implementation. This roadmap
directory records P00's source checkpoint, completed P01a-P01c automation, and the merged
qualification/runtime changes through the merged CP stack. Their PRs and current status live in
the milestone files.

## Release sequence

The table preserves historical acceptance boundaries as well as future releases. Stable tags
through v0.5.1 exist; the alpha/beta/RC rows were planning checkpoints, not published tags.
Future rows carry no calendar commitments. Tag each future integration commit only
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
| `v0.5.1` | Torch 2.10/2.11 compatibility after #374 | Published source-only patch tag; no Release or image |
| [v0.5.x](milestones/v0.5x-v0.6.md#p18) | P18 | DiffusionART with bounded sampler/trajectory semantics |
| [v0.6.0](milestones/v0.5x-v0.6.md#p19) | P19-P21 | ReFL/DRTune, differentiable rewards and component distribution |
| [v0.7.0](milestones/v0.6-v0.7.md#p15) | P15 | One real MoE model with a correct NCCL EP implementation |
| [v0.8.0](milestones/v0.6-v0.7.md#p16) | P16 | Optional DeepEP transport with a validated NCCL fallback |
| [v1.0.0-rc.1](milestones/v1.0.md#p17), then [v1.0.0](milestones/v1.0.md#release-records) | P17, after P00-P16 and P18-P21 | Qualified complete scope, documented compatibility guarantees, maintainable ownership and final integration evidence |

P01a supplied hooks/basic CPU CI, P01b the manual evidence policy, P01c consistency checks and
P04 automatic selection. P03 qualified initial TP/SP; the v0.4 ownership work then supported
CP integration. Those dependencies are complete; preserve their runtime contracts and evidence
rather than reopening the original work items.

P02 #281, P03 #282, P04 #283, P05 #285, P06 #286, P07 #287 and P08 #288 are merged.
Tags `v0.3.0` and `v0.4.0` already point to `17e8be7c` and `fbb038e3`, respectively.
Their package metadata is `0.3.0` and `0.4.0`. Do not move or recreate those tags.
GitHub Release pages exist for v0.1.0, v0.2.0 and v0.5.0. A tag and a Release page are separate
artifacts; v0.5.1 is source-only. Historical release-page backfill is optional.

## Starting evidence

Last checked: **2026-11-11**, main `29593921`. See the
[v0.5 release records](milestones/v0.5.md#release-records) for immutable tag targets and the
post-release Torch qualifications. Alpha/beta/RC rows describe acceptance boundaries, not
published historical tags; do not manufacture those tags retroactively.
Detailed limitations, evidence provenance and source-tag metadata are recorded in [the v0.2 baseline](milestones/v0.2.md).

- The original `v0.1.0` release tag points to
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
| Differentiable reward | [P19a](milestones/v0.5x-v0.6.md#p19a): typed input-gradient contract and one frozen scorer | Next development entry; P18 implementation is merged |
| ReFL | [P19b](milestones/v0.5x-v0.6.md#p19b): replicated derivative reference | P19a before component sharding |
| ART experiment and publication | [P18 follow-through](milestones/v0.5x-v0.6.md#p18): record final experiment outcomes and review release identity | Preserve original evidence; does not block P19a |
| Component distribution | [P20](milestones/v0.5x-v0.6.md#p20) | Reuse runtime/loading owners; replicated reference first |
| DRTune | [P21](milestones/v0.5x-v0.6.md#p21) | Exact selected-step derivative program before optional VJP optimization |
| Deferred EP model admission | [P15b](milestones/v0.6-v0.7.md#p15b): pin DiT-MoE semantics and upstream reference | Resolve conditioning, auxiliary objective and expert ownership before EP runtime |
| NCCL EP integration | [P15c–P15f](milestones/v0.6-v0.7.md#p15c) | EP=1 reference, dispatch, FSDP/state, then online/offline integration |
| Optional DeepEP | [P16a](milestones/v0.6-v0.7.md#p16a): qualify a hardware/toolchain/API tuple | NCCL reference stays supported; no mid-collective fallback |

## Scope of v1.0

Keep DDP, DeepSpeed ZeRO-2, and FSDP2 as the primary existing backend configurations. Advanced
TP/SP/CP support initially targets FSDP2 and explicitly declared model layouts. The release must
cover online feedback, offline training, and multi-role execution through representative semantic
paths, with documented restrictions and reproducible evidence.

The harness is part of this release's core functionality. NCCL EP for one real MoE model and an
optional DeepEP transport enter through v0.7/v0.8; v1.0 qualifies their declared scope alongside
TP/SP/CP, algorithms and state recovery. This does not promise every EP x TP x CP combination.
GPU selection follows affected contracts and valid topologies; a fixed 45-job, 32-rank campaign
and a fixed reward deployment are
not the proposed default gate. CP transports/layouts needed for the accepted v0.5 matrix belong to P12; unrestricted EP composition,
arbitrary in-flight pipeline recovery and concurrent rollout/training remain separate features.
