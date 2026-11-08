# Algorithm coverage

All **14 registered trainers** explicitly declare the shared CP training contract. They compute
objectives from the adapter's complete prediction; the common backward boundary applies the CP
normalization. TP/SP remain model-owned. The combination still has to satisfy the
[model layout](models.md), [backend configuration](configuration.md), tuning mode and algorithm's
sampling rules.

This admission is broader than the set of measured recipes. A model numerical test establishes
its tensor operations; an algorithm lifecycle test also exercises acquisition, feedback, roles,
updates and continuation. Neither alone proves every algorithm × model combination.

## Choose the training path

The registry keys below are the values of `train.trainer_type`.

| Trainer | Training input and feedback | What parallel execution must preserve |
|---|---|---|
| `grpo` | Generated samples and runtime rewards | Stored stochastic transitions, policy/reference predictions and complete reward groups |
| `grpo-guard` | Generated samples and runtime rewards | GRPO semantics plus the guard's additional derivative path |
| `dppo` | Generated samples and runtime rewards | Coupled transition and policy-ratio semantics |
| `nft` | Generated samples and runtime rewards | Stored old-policy state, shared replay noise and sample-level matching |
| `awm` | Generated samples and runtime rewards | Weighted matching, old/current queries and optional reference/EMA penalties |
| `dgpo` | Generated samples and runtime rewards | Complete groups in each global optimizer microbatch |
| `dpo` | Generated samples and runtime rewards | Source-aware best/worst pairs; both arms and reference share noise and time |
| `crd` | Generated samples and runtime rewards | Acquisition-wide first pass, detached centers and old/sampling snapshots |
| `diffusion-opd` | Generated samples, no training rewards | Source-routed teacher snapshots, cached detached targets and student-only updates |
| `dmd2` | Generated samples, no training rewards | Separate fake and generator roles; fake updates precede the generator |
| `tdm` | Generated trajectories, no training rewards | Stored trajectory boundaries, conditional noise and fake-before-generator updates |
| `tdm-r1` | Generated trajectories and runtime rewards | Fake, surrogate and generator phases with complete reward-group identity |
| `sft` | Finite dataset, no training rewards | Target encoding, finite data partitioning and complete accumulation windows |
| `offline-dpo` | Finite preference dataset, no training rewards | Chosen/rejected arms, a frozen reference and shared forward-process noise/time |

Offline training does not generate a training rollout. Reward-free distillation can use evaluation
rewards without adding them to the training objective. See the
[algorithm guide](../training/algorithms.md) for losses, tuning modes and recipe settings.

DMD2, TDM and TDM-R1 keep their roles in one prepared model/optimizer bundle and update them in
separate phases. Frozen references, EMA and teacher snapshots retain their own ownership.
Adding TP or CP does not turn these into independent data replicas or change role cadence.

## Sampling and feedback count logical replicas

A TP2 × CP2 group processes one logical sample, even though four ranks participate. Reward-group
statistics, source identities, preference pairs and sampling counts operate over logical DP.
Do not multiply group size or reward count by TP or CP.

Packed models can depend on which samples share a forward call. Bagel and Flux3 preserve rollout
pack composition during replay. Shuffling individual records out of those packs would change
model computation even with the same weights and noise.

[Reward/optimization overlap](../training/workflow/reward_optimization_overlap.md) has its own
admission table:

| Trainers | Admitted overlap sampler layouts |
|---|---|
| `grpo`, `grpo-guard`, `dppo`, `nft`, `awm`, `crd` | `rank_local`, `global_batch`, `global_tile`, `subgroup_tile` |
| `dgpo` | `global_batch` |
| `dpo` | `rank_local` |
| `tdm-r1` | `rank_local`, `global_batch` |

These paths require asynchronous CPU rewards, tile-local statistics (`global_std: false`), one
inner epoch, no sample shuffling, and the algorithm's complete group/accumulation windows.
Both ordered and ready scheduling are available. Async groupwise rewards require `rank_local`;
cross-rank layouts use pointwise rewards. SFT, offline DPO and reward-free distillation have no
training reward stage to overlap.

## Read recipe evidence at its stated scope

The detailed references distinguish capability declarations from selected native recipes. For
example, Qwen AWM/CRD recipes use TP1 × CP2 × DP2; they are not TP+CP lifecycle evidence. A Flux
CP-only DPO recipe likewise does not establish its TP composition. Model-level TP+CP controls
remain useful evidence for the unchanged model owner, with that narrower scope.

| Recipe family | Detailed evidence and limits |
|---|---|
| SFT, GRPO, DMD2, NFT, DGPO, DiffusionOPD and TDM | [Shared CP reference](../../docs/reference/parallelism/context-parallel.md) and [training algorithms](../training/algorithms.md) |
| Flux GRPO-Guard / DPPO | [Flux contract](../../docs/reference/parallelism/models/flux.md) |
| AWM / CRD | [AWM](../../docs/reference/parallelism/algorithms/awm.md) · [CRD](../../docs/reference/parallelism/algorithms/crd.md) |
| Online / offline preference learning | [Online DPO](../../docs/reference/parallelism/algorithms/online-dpo.md) · [Offline DPO](../../docs/reference/parallelism/algorithms/offline-dpo.md) |
| Reward-driven trajectory training | [TDM-R1](../../docs/reference/parallelism/algorithms/tdm-r1.md) |

Exact continuation uses the recorded runtime identity and a drained acquisition or finite-data
epoch boundary. It does not promise recovery of arbitrary in-flight reward work or continuation
under a changed TP/CP topology. The [configuration guide](configuration.md) explains the boundary.
