# EP design checkpoint

This is planning for [P15/P16](../milestones/v0.6-v0.7.md), not an implemented EP API. Reviewed
2026-11-11. Keep existing typed runtime, algorithm, loading and state owners; add the smallest
model/expert boundaries required by a real admitted model. Do not introduce another trainer or
preparation stack, or infer EP support from a model's name.

## First model: DiT-MoE

The user-selected XL/2-8E2A and G/2-16E2A checkpoints are listed as 256x256 rectified-flow models.
Inspect the pinned [model code](https://github.com/feizc/DiT-MoE/blob/bd34bbb2945280f0ccc2b1221f7dc8e3d6cce65d/models.py),
[weight list](https://github.com/feizc/DiT-MoE/blob/bd34bbb2945280f0ccc2b1221f7dc8e3d6cce65d/README.md)
and [RF wrapper](https://github.com/feizc/DiT-MoE/blob/bd34bbb2945280f0ccc2b1221f7dc8e3d6cce65d/diffusion/rectified_flow.py).
Weight revisions, provenance and usage terms still need admission review; a source pin alone
neither pins the weights nor establishes framework support.

The inspected router performs per-token softmax/top-k over experts, not expert-choice selection
over the whole token set. It leaves selected probabilities unnormalized by default. Routed
experts use gated SiLU MLPs; an additional shared MLP contributes to every token. This supports
an ordinary token-dispatch EP design. Start with XL/8E as the primary checkpoint and use G/16E to
check generalization of expert counts and dimensions, without changing the selected mathematics.

Three integration boundaries precede distributed experts:

- Class labels, null conditioning, label dropout, VAE scaling, RF time/direction and output
  channels must map to the adapter's condition/trajectory contract. This is class-conditioned
  generation, not a text-conditioned model with a placeholder prompt.
- Upstream training and inference use different expert execution paths. Establish the admitted
  generation/replay behavior at EP=1, including precision, before distributing either path.
  `pretraining_tp` performs local matrix slicing; it is not distributed TP support.
- `AddAuxiliaryLoss` injects a router gradient through custom backward. Do not blindly copy it
  into every algorithm query. Define who consumes the auxiliary objective, its coefficient,
  statistical domain, accumulation normalization and role ownership. Frozen/evaluation queries
  and checkpoint recomputation must not silently add or duplicate training objectives.

Per-token routing does not require a global token-ranking operation, but CP still changes the
domain over which router statistics/auxiliary objectives are computed. Establish equivalence
before admitting EP+CP. Under fixed weights/inputs, check routing choices as well as output
error: wrong expert identities are not ordinary floating-point accumulation variation. Preserve
token/top-k-slot identity through combine, without demanding cross-transport bitwise equality.

## Upstream references and reuse

| Reference | Reuse or study | Boundary retained here |
|---|---|---|
| [Transformers EP](https://github.com/huggingface/transformers/blob/07cc8e8c1b19160cfba6893ceb3099dc8f09d90c/src/transformers/distributed/expert_parallel.py) | Autograd all-to-all, uneven/empty routing, original top-k-slot restoration, TP token deduplication | Adapter semantics and existing Flow-Factory training/feedback counting |
| [Transformers layout documentation](https://github.com/huggingface/transformers/blob/07cc8e8c1b19160cfba6893ceb3099dc8f09d90c/docs/source/en/expert_parallelism.md) | Separate dense/expert FSDP groups and gradient normalization | Our mesh, logical samples and checkpoint ownership; its Trainer dispatch counting is not complete |
| [Megatron-Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/0.19.2/user-guide/features/moe.html) | Training dispatcher boundaries, router losses, expert state and communication scheduling | No replacement of the current trainer/backend architecture |
| [DeepEP](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/README.md) | Optional optimized dispatch/combine and explicit async completion | Mathematical routing, optimizer state and NCCL reference/fallback |

These are investigation pins, not dependency upgrades. Prefer supported upstream primitives;
when adapting code, retain attribution/terms and keep the adaptation small, testable and owned.
Do not copy a framework wholesale or use a private API without recording its compatibility scope.

## Ownership and topology decisions

| Owner | Responsibility |
|---|---|
| Model adapter | Router/shared-expert semantics, parameter identity, conditions, native outputs and expert computation |
| Shared preparation/topology | Expert placement, token ownership, dense versus expert groups and explicit capability checks |
| Dispatch/combine transport | Send routed pairs, return contributions, preserve metadata/autograd and complete buffer use |
| Existing optimization/state runtime | Scale gradients once, own expert/dense optimizers, references/EMA/variants, save/load and next-update continuation |
| Existing execution/feedback runtime | Logical samples, reward groups, role cadence and tile-stream overlap |

EP is not automatically a new multiplier of `DP * TP * CP`. Dense and expert views can regroup
existing ranks. Before coding, derive which ranks hold each token and expert, where gradient sums
become means, and which group saves or restores each state. Replicated TP inputs must not dispatch
the same token twice; SP/CP shards must not be discarded or re-reduced as replicas. Keep shared
experts and router parameters distinct from distributed routed experts.

First qualify FSDP2 + NCCL EP with whole local experts. DDP/ZeRO-2 EP and arbitrary combinations
with TP/SP/CP are not implied. Admit compositions separately with a concrete group diagram,
normalization proof and representative test, keeping existing dense-model paths unchanged.

## DeepEP compatibility and fallback

The inspected DeepEP V2.5 source uses `EPBuffer` and removes V1/NVSHMEM APIs. It requires Hopper
or newer, CUDA Toolkit >=13.1, NCCL >=2.32.3 and a C++20 toolchain. Existing cu126/cu128 Torch
qualification is not qualification of that build/runtime tuple. P16a must choose and verify a
concrete revision, driver/toolkit/wheel/NCCL combination and intranode interconnect. An older
pinned DeepEP line is a separate compatibility decision, not interchangeable with current main.

NCCL remains supported. With an automatic transport policy, all ranks agree on availability and
select the fallback before any dispatch. An explicitly required unsupported transport should
fail with a concrete diagnosis. Never switch transport after a collective partially executes.
Shared buffers, autograd contexts and completion events must survive until all consumers finish,
including reward overlap and checkpoint recomputation. Measure benefits on stated workloads;
no universal throughput claim or cross-node test is a default local acceptance gate.
